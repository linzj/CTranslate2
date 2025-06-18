#include "graph_recorder.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "backend_dml.h"
#include "dml/operator_cache.h"
#include "dml/operator_utils.h"
#include "dml_utils.h"

namespace ctranslate2 {
namespace dml {

namespace {  // anonymous

constexpr bool kDumpSubGraphs = false;
constexpr bool kAlwaysEvaluateSubgraphs = false;

// Represents a portion of a larger computation graph. A SubGraph consists of a
// set of operator nodes, their inputs, and their outputs. SubGraphs can be
// split into smaller pieces to handle complex dependencies or resource
// constraints.
class SubGraph {
 public:
  std::vector<OperatorNode*> op_nodes;
  std::vector<BindingNode*> inputs;
  std::vector<BindingNode*> outputs;

  SubGraph() = default;

  // Constructs a SubGraph from all operator nodes and the initial graph's
  // inputs and outputs.
  SubGraph(const std::vector<std::unique_ptr<OperatorNode>>& all_op_nodes,
           const std::vector<BindingNode*>& graph_inputs,
           const std::vector<BindingNode*>& graph_outputs) {
    op_nodes.reserve(all_op_nodes.size());
    for (const auto& node : all_op_nodes) {
      op_nodes.push_back(node.get());
    }
    inputs = graph_inputs;
    outputs = graph_outputs;
  }

  // Splits the current SubGraph into two smaller SubGraphs at a specified
  // operator index. This is crucial for breaking down a large graph into
  // manageable parts that can be compiled and executed independently.
  std::pair<SubGraph, SubGraph> split(size_t split_op_index) const {
    if (split_op_index == 0 || split_op_index >= op_nodes.size()) {
      throw std::invalid_argument("Invalid split index for SubGraph.");
    }

    SubGraph first_half;
    SubGraph second_half;

    // 1. Split operator nodes into two halves based on the split index.
    for (size_t i = 0; i < op_nodes.size(); ++i) {
      if (i < split_op_index) {
        first_half.op_nodes.push_back(op_nodes[i]);
      } else {
        second_half.op_nodes.push_back(op_nodes[i]);
      }
    }

    // 2. Collect all unique outputs produced within each half. This helps in
    // determining the dependencies between the two new subgraphs.
    std::unordered_set<BindingNode*> first_half_produced_outputs;
    for (const auto* op_node : first_half.op_nodes) {
      for (auto* output_node : op_node->outputs) {
        if (output_node->resource) {
          first_half_produced_outputs.insert(output_node);
        }
      }
    }

    std::unordered_set<BindingNode*> second_half_produced_outputs;
    for (const auto* op_node : second_half.op_nodes) {
      for (auto* output_node : op_node->outputs) {
        if (output_node->resource) {
          second_half_produced_outputs.insert(output_node);
        }
      }
    }

    // 3. Determine the inputs for each half. An input to a half is a resource
    // that is consumed by an operator in that half but not produced within the
    // same half.
    std::unordered_set<BindingNode*> first_half_inputs_set;
    for (const auto* op_node : first_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource &&
            first_half_produced_outputs.find(input_node) ==
                first_half_produced_outputs.end()) {
          first_half_inputs_set.insert(input_node);
        }
      }
    }
    first_half.inputs.assign(first_half_inputs_set.begin(),
                             first_half_inputs_set.end());

    std::unordered_set<BindingNode*> second_half_inputs_set;
    for (const auto* op_node : second_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource &&
            second_half_produced_outputs.find(input_node) ==
                second_half_produced_outputs.end()) {
          second_half_inputs_set.insert(input_node);
        }
      }
    }
    second_half.inputs.assign(second_half_inputs_set.begin(),
                              second_half_inputs_set.end());

    // 4. Determine the outputs for each half. An output of the first half is a
    // resource that is either an output of the original graph or is consumed by
    // the second half. An output of the second half is a resource that is also
    // an output of the original graph.
    // 4. Determine the outputs for each half. An output of a subgraph is a
    // resource that is produced within that subgraph but not consumed by any
    // operator within that same subgraph.
    std::unordered_set<BindingNode*> consumed_in_first_half;
    for (const auto* op_node : first_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource) {
          consumed_in_first_half.insert(input_node);
        }
      }
    }

    std::unordered_set<BindingNode*> first_half_outputs_set;
    for (auto* produced_node : first_half_produced_outputs) {
      if (consumed_in_first_half.find(produced_node) ==
          consumed_in_first_half.end()) {
        first_half_outputs_set.insert(produced_node);
      }
    }
    first_half.outputs.assign(first_half_outputs_set.begin(),
                              first_half_outputs_set.end());

    std::unordered_set<BindingNode*> consumed_in_second_half;
    for (const auto* op_node : second_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource) {
          consumed_in_second_half.insert(input_node);
        }
      }
    }

    std::unordered_set<BindingNode*> second_half_outputs_set;
    for (auto* produced_node : second_half_produced_outputs) {
      if (consumed_in_second_half.find(produced_node) ==
          consumed_in_second_half.end()) {
        second_half_outputs_set.insert(produced_node);
      }
    }
    second_half.outputs.assign(second_half_outputs_set.begin(),
                               second_half_outputs_set.end());

    return {std::move(first_half), std::move(second_half)};
  }

  std::string GetCacheKey() const {
    std::string key = "Graph:";
    for (const auto* op_node : op_nodes) {
      key += op_node->op->key();
    }

    auto binding_to_string = [](const BindingNode* node) {
      if (!node->resource) {
        return std::string("0:0");
      }
      return std::to_string(node->offset) + ":" +
             std::to_string(node->size_in_bytes);
    };

    key += "Inputs:";
    for (const auto* input_node : inputs) {
      key += binding_to_string(input_node) + ";";
    }

    key += "Outputs:";
    for (const auto* output_node : outputs) {
      key += binding_to_string(output_node) + ";";
    }

    return key;
  }

  void Dump(std::ostream& os) const {
    std::unordered_map<const BindingNode*, size_t> graph_input_to_index;
    for (size_t i = 0; i < inputs.size(); ++i) {
      graph_input_to_index[inputs[i]] = i;
    }

    std::unordered_map<const BindingNode*, size_t> graph_output_to_index;
    for (size_t i = 0; i < outputs.size(); ++i) {
      graph_output_to_index[outputs[i]] = i;
    }

    std::unordered_map<const BindingNode*, std::pair<size_t, size_t>>
        intermediate_producers;
    for (size_t i = 0; i < op_nodes.size(); ++i) {
      const auto* op_node = op_nodes[i];
      for (size_t j = 0; j < op_node->outputs.size(); ++j) {
        if (op_node->outputs[j] && op_node->outputs[j]->resource) {
          intermediate_producers[op_node->outputs[j]] = {i, j};
        }
      }
    }

    os << "Nodes (" << op_nodes.size() << "):\n";
    for (size_t i = 0; i < op_nodes.size(); ++i) {
      const auto* op_node = op_nodes[i];
      os << "  Node[" << i << "]: "
         << OperatorUtils::DML_OPERATOR_TYPE_toString(op_node->op->GetType())
         << "\n";

      os << "    Inputs (" << op_node->inputs.size() << "):\n";
      for (size_t j = 0; j < op_node->inputs.size(); ++j) {
        const auto* input_binding = op_node->inputs[j];
        os << "      Input[" << j << "]: " << input_binding;
        if (input_binding) {
          if (input_binding->resource) {
            os << ", resource: " << input_binding->resource
               << ", offset: " << input_binding->offset
               << ", size: " << input_binding->size_in_bytes;
          }
          os << ")";
          auto it_graph_input = graph_input_to_index.find(input_binding);
          if (it_graph_input != graph_input_to_index.end()) {
            os << " -> Graph Input Index: " << it_graph_input->second;
          } else {
            auto it = intermediate_producers.find(input_binding);
            if (it != intermediate_producers.end()) {
              os << " -> Produced by Node[" << it->second.first << "], Output["
                 << it->second.second << "]";
            } else if (input_binding->resource) {
              os << " -> ERROR: Intermediate producer not found!";
            }
          }
        }
        os << "\n";
      }

      os << "    Outputs (" << op_node->outputs.size() << "):\n";
      for (size_t j = 0; j < op_node->outputs.size(); ++j) {
        const auto* output_binding = op_node->outputs[j];
        os << "      Output[" << j << "]: " << output_binding;
        if (output_binding) {
          if (output_binding->resource) {
            os << ", resource: " << output_binding->resource
               << ", offset: " << output_binding->offset
               << ", size: " << output_binding->size_in_bytes;
          }
          os << ")";
          auto it_graph_output = graph_output_to_index.find(output_binding);
          if (it_graph_output != graph_output_to_index.end()) {
            os << " -> Graph Output Index: " << it_graph_output->second;
          }
        }
        os << "\n";
      }
    }
  }
};

// Checks if two binding nodes have overlapping memory regions. This is
// essential for detecting potential data hazards, where one operation might
// overwrite data needed by another.
bool bindings_overlap(const BindingNode* a, const BindingNode* b) {
  if (!a || !b || !a->resource || !b->resource) {
    return false;
  }
  if (a->resource != b->resource) {
    return false;
  }
  const UINT64 a_end = a->offset + a->size_in_bytes;
  const UINT64 b_end = b->offset + b->size_in_bytes;
  return a->offset < b_end && b->offset < a_end;
}

// Analyzes a subgraph to find a suitable point to split it. A split is
// necessary if certain data dependency rules are violated, such as an
// operation's output overlapping with a graph input.
std::optional<size_t> find_split_point(
    const SubGraph& subgraph,
    const std::vector<BindingNode*>& original_graph_inputs) {
  std::unordered_set<const BindingNode*> committed_outputs;
  for (size_t op_idx = 0; op_idx < subgraph.op_nodes.size(); ++op_idx) {
    const auto* op_node = subgraph.op_nodes[op_idx];

    // Rule 2: An operator's input must not overlap with the output of a
    // previous operator, unless it's the exact same resource. This prevents
    // read-after-write hazards.
    for (const auto* input_node : op_node->inputs) {
      if (!input_node || !input_node->resource)
        continue;
      for (const auto* prev_output : committed_outputs) {
        if (input_node != prev_output &&
            bindings_overlap(input_node, prev_output)) {
          SPDLOG_DEBUG(
              "Splitting graph: Operator input overlaps with a "
              "previous output with different binding. Op: '{}', "
              "Resource: {}",
              OperatorUtils::DML_OPERATOR_TYPE_toString(op_node->op->GetType()),
              (void*)input_node->resource);
          return op_idx;
        }
      }
    }

    for (const auto* output_node : op_node->outputs) {
      if (!output_node || !output_node->resource)
        continue;

      // Rule 1: An operator's output must not overlap with any of the
      // subgraph's inputs. This prevents write-after-read hazards within the
      // subgraph.
      for (const auto* subgraph_input : subgraph.inputs) {
        if (subgraph_input->resource &&
            bindings_overlap(output_node, subgraph_input)) {
          SPDLOG_DEBUG(
              "Splitting graph: Operator output overlaps with a "
              "subgraph input. Op: '{}', Resource: {}",
              OperatorUtils::DML_OPERATOR_TYPE_toString(op_node->op->GetType()),
              (void*)output_node->resource);
          return op_idx;
        }
      }

      // Rule 3: An operator's output must not overlap with any of the original
      // graph's inputs. This is a broader check to ensure integrity across the
      // entire computation.
      for (const auto* original_graph_input : original_graph_inputs) {
        if (original_graph_input->resource &&
            bindings_overlap(output_node, original_graph_input)) {
          SPDLOG_DEBUG(
              "Splitting graph: Operator output overlaps with a graph input. "
              "Op: '{}', Resource: {}",
              OperatorUtils::DML_OPERATOR_TYPE_toString(op_node->op->GetType()),
              (void*)output_node->resource);
          return op_idx;
        }
      }
    }

    // The outputs of the current operator are added to the set of committed
    // outputs for checking against subsequent operators.
    for (auto* output_node : op_node->outputs) {
      if (output_node && output_node->resource) {
        committed_outputs.insert(output_node);
      }
    }
  }
  return std::nullopt;
}
// A debug/testing function to execute a subgraph without creating a
// fused DML graph. This executes operators one by one.
void EvaluateSubGraphWithoutFusedGraph(const SubGraph& subgraph) {
  // Now execute the graph operator by operator.
  for (const auto& op_node : subgraph.op_nodes) {
    // Prepare DmlBindingArrayBundle for inputs
    std::vector<utils::DmlBufferBindingBundle> inputs_bundles;
    inputs_bundles.reserve(op_node->inputs.size());
    for (const BindingNode* binding_node : op_node->inputs) {
      inputs_bundles.emplace_back(binding_node->resource, binding_node->offset,
                                  binding_node->size_in_bytes);
    }
    utils::DmlBindingArrayBundle current_inputs(std::move(inputs_bundles));

    // Prepare DmlBindingArrayBundle for outputs
    std::vector<utils::DmlBufferBindingBundle> outputs_bundles;
    outputs_bundles.reserve(op_node->outputs.size());
    for (const BindingNode* binding_node : op_node->outputs) {
      outputs_bundles.emplace_back(binding_node->resource, binding_node->offset,
                                   binding_node->size_in_bytes);
    }
    utils::DmlBindingArrayBundle current_outputs(std::move(outputs_bundles));

    op_node->op->Execute(current_inputs, current_outputs);
  }
}

}  // namespace

// Initializes the GraphRecorder, including a special empty binding node used
// for operators that have no input or output.
GraphRecorder::GraphRecorder() : m_empty_binding_node({nullptr, 0, 0}) {}

GraphRecorder::~GraphRecorder() = default;

// Begins a graph recording session. All subsequent calls to Execute will be
// recorded.
void GraphRecorder::Begin() {
  m_has_begun = true;  // Mark that Begin has been called
}

// Records the execution of a single operator. It creates nodes for the
// operator, its inputs, and its outputs, adding them to the graph
// representation.
void GraphRecorder::Execute(Operator* op,
                            const utils::DmlBindingArrayBundle& inputs,
                            const utils::DmlBindingArrayBundle& outputs) {
  if (!m_has_begun) {
    throw std::invalid_argument("GraphRecorder::Execute called before Begin.");
  }
  // Create a new operator node to represent this execution.
  std::unique_ptr<OperatorNode> op_node = std::make_unique<OperatorNode>();
  op_node->op = op;

  // Process input bindings, creating or retrieving existing binding nodes.
  auto input_descs = inputs.get_descs();
  for (size_t i = 0; i < input_descs.size(); ++i) {
    if (input_descs[i].Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(input_descs[i].Desc);
      bool node_created;  // Track if a new node was created
      BindingNode* binding_node =
          GetOrCreateBindingNode(buffer_binding->Buffer, buffer_binding->Offset,
                                 buffer_binding->SizeInBytes, node_created);
      op_node->inputs.push_back(binding_node);

      // If an input binding node is newly created, it's considered an input to
      // the entire graph.
      if (node_created) {
        // Only mark as graph input if this is the first operator being recorded
        // and the binding node was not previously encountered as an output
        m_graph_inputs.push_back(binding_node);
      }
    } else {
      if (input_descs[i].Type != DML_BINDING_TYPE_NONE) {
        throw std::invalid_argument(
            "Unsupported input binding type in Execute.");
      }
      op_node->inputs.push_back(&m_empty_binding_node);
    }
  }

  // Process output bindings similarly.
  auto output_descs = outputs.get_descs();
  for (size_t i = 0; i < output_descs.size(); ++i) {
    if (output_descs[i].Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(output_descs[i].Desc);
      bool node_created;
      BindingNode* binding_node =
          GetOrCreateBindingNode(buffer_binding->Buffer, buffer_binding->Offset,
                                 buffer_binding->SizeInBytes, node_created);
      op_node->outputs.push_back(binding_node);
    } else {
      if (output_descs[i].Type != DML_BINDING_TYPE_NONE) {
        throw std::invalid_argument(
            "Unsupported output binding type in Execute.");
      }
      op_node->outputs.push_back(&m_empty_binding_node);
    }
  }

  m_operator_nodes.push_back(std::move(op_node));
}

// A debug/testing function to execute the recorded graph without creating a
// fused DML graph. This executes operators one by one.
void GraphRecorder::EvaluateGraphWithoutFusedGraph() {
  // Mark the last operator's outputs as graph outputs.
  OperatorNode* last_op_node = m_operator_nodes.back().get();
  for (BindingNode* output_node : last_op_node->outputs) {
    m_graph_outputs.push_back(output_node);
  }

  SubGraph whole_graph(m_operator_nodes, m_graph_inputs, m_graph_outputs);
  EvaluateSubGraphWithoutFusedGraph(whole_graph);
  Reset();
}

// Finalizes the graph recording. This involves identifying graph outputs,
// splitting the graph into executable subgraphs, and then executing each
// subgraph.
void GraphRecorder::End() {
  m_has_begun = false;

  // Option to bypass graph fusion for debugging.
  constexpr bool kEvaluateWithoutFusedGraph = false;
  if (kEvaluateWithoutFusedGraph) {
    EvaluateGraphWithoutFusedGraph();
    return;
  }

  if (m_operator_nodes.empty()) {
    return;
  }

  // Identify the final outputs of the graph. An output is a resource produced
  // by an operator that is not consumed by any other operator in the graph.
  for (const auto& op_node : m_operator_nodes) {
    for (auto& output_node : op_node->outputs) {
      bool is_consumed = false;
      for (const auto& consumer_node : m_operator_nodes) {
        for (const auto& input_node : consumer_node->inputs) {
          if (input_node == output_node) {
            is_consumed = true;
            break;
          }
        }
        if (is_consumed)
          break;
      }
      if (!is_consumed) {
        m_graph_outputs.push_back(output_node);
      }
    }
  }

  // Begin with a single subgraph containing the entire recorded graph.
  std::list<SubGraph> processing_list;
  processing_list.emplace_back(m_operator_nodes, m_graph_inputs,
                               m_graph_outputs);
  std::vector<SubGraph> final_subgraphs;

  // Iteratively process subgraphs, splitting them as necessary until all
  // subgraphs are simple enough to be compiled and executed.
  while (!processing_list.empty()) {
    SubGraph current_subgraph = std::move(processing_list.front());
    processing_list.pop_front();

    if (current_subgraph.op_nodes.empty()) {
      continue;
    }

    // Check if the current subgraph needs to be split.
    std::optional<size_t> split_idx =
        find_split_point(current_subgraph, m_graph_inputs);

    if (split_idx.has_value()) {
      size_t split_at = split_idx.value();
      if (split_at > 0) {
        // If a valid split point is found, split the subgraph and add the
        // two new subgraphs back to the processing list.
        auto [first, second] = current_subgraph.split(split_at);
        if (!second.op_nodes.empty()) {
          processing_list.push_front(std::move(second));
        }
        if (!first.op_nodes.empty()) {
          processing_list.push_front(std::move(first));
        }
      } else {  // split_at == 0
        // Handle the case where the split is at the very beginning of the
        // subgraph.
        if (current_subgraph.op_nodes.size() > 1) {
          auto [first, second] = current_subgraph.split(1);
          final_subgraphs.push_back(std::move(first));
          if (!second.op_nodes.empty()) {
            processing_list.push_front(std::move(second));
          }
        } else {
          final_subgraphs.push_back(std::move(current_subgraph));
        }
      }
    } else {
      // If no split is needed, the subgraph is considered final.
      final_subgraphs.push_back(std::move(current_subgraph));
    }
  }

  // Execute the finalized subgraphs.
  if (kDumpSubGraphs) {
    for (const auto& subgraph : final_subgraphs) {
      subgraph.Dump(std::cout);
    }
  }

  for (const auto& subgraph : final_subgraphs) {
    if (subgraph.op_nodes.empty()) {
      continue;
    }
    if (kAlwaysEvaluateSubgraphs) {
      EvaluateSubGraphWithoutFusedGraph(subgraph);
      continue;
    }

    if (subgraph.op_nodes.size() == 1) {
      EvaluateSubGraphWithoutFusedGraph(subgraph);
      continue;
    }

    // Generate a cache key for the subgraph based on its operators.
    std::string key_accumulator = subgraph.GetCacheKey();

    auto& cache = DMLOperatorCache::instance();
    Operator* graph_op = cache.GetOperator(key_accumulator);
    Microsoft::WRL::ComPtr<Operator> new_graph_op_comptr;

    if (!graph_op) {
      // If the compiled graph is not in the cache, build it.
      std::vector<BindingNode*> sorted_outputs = subgraph.outputs;
      std::sort(sorted_outputs.begin(), sorted_outputs.end(),
                [](const BindingNode* a, const BindingNode* b) {
                  if (a->resource != b->resource)
                    return a->resource < b->resource;
                  if (a->offset != b->offset)
                    return a->offset < b->offset;
                  return a->size_in_bytes < b->size_in_bytes;
                });
      auto it = std::adjacent_find(
          sorted_outputs.begin(), sorted_outputs.end(),
          [](const BindingNode* a, const BindingNode* b) {
            return a->resource == b->resource && a->offset == b->offset &&
                   a->size_in_bytes == b->size_in_bytes;
          });

      if (it != sorted_outputs.end()) {
        throw std::invalid_argument(
            "Multiple subgraph outputs bind to the same resource.");
      }

      // Build the compiled graph from the subgraph definition.
      // NOTE: The GraphBuilder::Build API expects a vector of unique_ptr, but
      // SubGraph provides raw pointers. This implementation assumes the Build
      // API can be overloaded or changed to accept raw pointers, as the
      // recorder's main list owns the operators.
      auto compiled_graph = GraphBuilder::Build(
          subgraph.op_nodes, subgraph.inputs, subgraph.outputs,
          DML_EXECUTION_FLAG_NONE, key_accumulator);

      new_graph_op_comptr = Microsoft::WRL::Make<Operator>(
          get_device(), utils::DmlOperatorDescBundle(), nullptr,
          std::move(compiled_graph), key_accumulator);

      graph_op = new_graph_op_comptr.Get();
      cache.AddOperator(std::move(key_accumulator),
                        std::move(new_graph_op_comptr));
    }

    // Prepare input and output bindings for execution.
    std::vector<utils::DmlBufferBindingBundle> input_binding_bundles;
    input_binding_bundles.reserve(subgraph.inputs.size());
    for (const auto& binding_node : subgraph.inputs) {
      input_binding_bundles.emplace_back(binding_node->resource,
                                         binding_node->offset,
                                         binding_node->size_in_bytes);
    }

    std::vector<utils::DmlBufferBindingBundle> output_binding_bundles;
    output_binding_bundles.reserve(subgraph.outputs.size());
    for (const auto& binding_node : subgraph.outputs) {
      output_binding_bundles.emplace_back(binding_node->resource,
                                          binding_node->offset,
                                          binding_node->size_in_bytes);
    }

    utils::DmlBindingArrayBundle inputs(std::move(input_binding_bundles));
    utils::DmlBindingArrayBundle outputs(std::move(output_binding_bundles));

    // Execute the compiled graph.
    graph_op->Execute(inputs, outputs);
  }

  Reset();
}

// Retrieves an existing binding node or creates a new one if it doesn't exist.
// This ensures that each unique resource (buffer, offset, size) is represented
// by a single node.
BindingNode* GraphRecorder::GetOrCreateBindingNode(ID3D12Resource* resource,
                                                   UINT64 offset,
                                                   UINT64 size,
                                                   bool& created) {
  std::tuple<ID3D12Resource*, UINT64, UINT64> key =
      std::make_tuple(resource, offset, size);

  // For inputs, we first check if a node for this resource already exists.
  auto it = m_binding_lookup.find(key);
  if (it != m_binding_lookup.end()) {
    created = false;    // Node already exists
    return it->second;  // Return existing node
  }

  // If no node exists, create a new one.
  std::unique_ptr<BindingNode> new_node = std::make_unique<BindingNode>();
  new_node->resource = resource;
  new_node->offset = offset;
  new_node->size_in_bytes = size;

  BindingNode* raw_ptr = new_node.get();
  m_all_binding_nodes.push_back(std::move(new_node));  // Store ownership
  m_binding_lookup[key] = raw_ptr;  // Add to lookup map for future access
  created = true;                   // Node was created

  return raw_ptr;
}

// Resets the state of the GraphRecorder, clearing all recorded nodes and graph
// structures. This prepares the recorder for a new recording session.
void GraphRecorder::Reset() {
  // Clear all existing states
  m_all_binding_nodes.clear();
  m_binding_lookup.clear();
  m_operator_nodes.clear();
  m_graph_inputs.clear();
  m_graph_outputs.clear();
}

}  // namespace dml
}  // namespace ctranslate2