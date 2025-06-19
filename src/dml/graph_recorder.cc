#include "graph_recorder.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "backend_dml.h"
#include "dml/bucketized_buffer_allocator.h"
#include "dml/operator_cache.h"
#include "dml/operator_utils.h"
#include "dml_utils.h"

namespace ctranslate2 {
namespace dml {

namespace {  // anonymous

// Checks if two binding nodes have overlapping memory regions. This is
// essential for detecting potential data hazards, where one operation might
// overwrite data needed by another.
static bool bindings_overlap(const BindingNode* a, const BindingNode* b) {
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

// When true, dumps the initial and final subgraphs to the console for
// debugging.
constexpr bool kDumpSubGraphs = false;
// When true, dumps a subgraph to stderr if it's retrieved from the cache.
constexpr bool kDumpSubGraphAfterCacheHit = false;
// When true, forces all subgraphs to be executed operator-by-operator instead
// of being fused.
constexpr bool kAlwaysEvaluateSubgraphs = false;
// When true, compares the output of fused graph execution with unfused
// execution to verify correctness.
constexpr bool kCompareWithUnfused = false;
// When true, forces recompilation of subgraphs even if they are found in the
// cache.
constexpr bool kAlwaysRecompileSubgraphs = false;

constexpr bool kDumpSubGraphBeforeBuild = false;

// Represents a portion of a larger computation graph. A SubGraph consists of a
// set of operator nodes, their inputs, and their outputs. SubGraphs can be
// split into smaller pieces to handle complex dependencies or resource
// constraints.
class SubGraph {
 public:
  // Operators in the subgraph, in execution order.
  std::vector<OperatorNode*> op_nodes;
  // External inputs required by the subgraph.
  std::vector<BindingNode*> inputs;
  // Outputs produced by the subgraph that are live after its execution.
  std::vector<OutputEdge> outputs;

  SubGraph() = default;

  // Constructs a SubGraph from all operator nodes and the initial graph's
  // inputs. The outputs are calculated using liveness analysis.
  SubGraph(std::vector<OperatorNode*>&& all_op_nodes,
           std::vector<BindingNode*>&& graph_inputs) {
    op_nodes = std::move(all_op_nodes);
    inputs = std::move(graph_inputs);
    CalculateAndSetOutputs();
  }

  // Calculates and sets the outputs of the subgraph. An output is defined as
  // the last write to any given resource within the subgraph. This is
  // determined by iterating through all operators and recording the last time
  // each resource is written to. The collected outputs are then sorted by their
  // operator and output index.
  void CalculateAndSetOutputs() {
    outputs.clear();
    std::unordered_map<ID3D12Resource*, OutputEdge> last_kill_of_resource;

    for (size_t i = 0; i < op_nodes.size(); ++i) {
      const auto* op_node = op_nodes[i];
      for (size_t j = 0; j < op_node->outputs.size(); ++j) {
        BindingNode* output_node = op_node->outputs[j];
        if (output_node && output_node->resource) {
          last_kill_of_resource[output_node->resource] =
              OutputEdge{static_cast<UINT32>(i), static_cast<UINT32>(j)};
        }
      }
    }

    for (const auto& pair : last_kill_of_resource) {
      outputs.push_back(pair.second);
    }

    std::sort(outputs.begin(), outputs.end(),
              [](const OutputEdge& a, const OutputEdge& b) {
                if (a.node_index != b.node_index) {
                  return a.node_index < b.node_index;
                }
                return a.output_index < b.output_index;
              });
  }

  // Splits the current SubGraph into two smaller SubGraphs at a specified
  // operator index. This is crucial for breaking down a large graph into
  // manageable parts that can be compiled and executed independently.
  std::pair<SubGraph, SubGraph> Split(size_t split_op_index) const {
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
    // same half. The inputs are ordered by their first use.
    std::unordered_set<BindingNode*> first_half_inputs_seen;
    for (const auto* op_node : first_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource &&
            first_half_produced_outputs.find(input_node) ==
                first_half_produced_outputs.end()) {
          if (first_half_inputs_seen.find(input_node) ==
              first_half_inputs_seen.end()) {
            first_half.inputs.push_back(input_node);
            first_half_inputs_seen.insert(input_node);
          }
        }
      }
    }

    std::unordered_set<BindingNode*> second_half_inputs_seen;
    for (const auto* op_node : second_half.op_nodes) {
      for (auto* input_node : op_node->inputs) {
        if (input_node->resource &&
            second_half_produced_outputs.find(input_node) ==
                second_half_produced_outputs.end()) {
          if (second_half_inputs_seen.find(input_node) ==
              second_half_inputs_seen.end()) {
            second_half.inputs.push_back(input_node);
            second_half_inputs_seen.insert(input_node);
          }
        }
      }
    }

    // 4. Determine the outputs for each half using liveness analysis.
    first_half.CalculateAndSetOutputs();
    second_half.CalculateAndSetOutputs();

    return {std::move(first_half), std::move(second_half)};
  }

  // Generates a unique cache key for the subgraph based on its operators and
  // input/output bindings. This key is used to cache and retrieve compiled DML
  // graphs.
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
    for (const auto& output_node : outputs) {
      key.append(reinterpret_cast<const char*>(&output_node),
                 sizeof(OutputEdge));
    }

    return key;
  }

  // Checks if the subgraph is suitable for direct evaluation, meaning it should
  // not be compiled into a fused graph but rather executed operator by
  // operator. This is typically true for subgraphs containing operators that
  // are lightweight, stateful, or not well-supported by the graph compiler.
  bool IsSuitableForEvaluation() const {
    if (op_nodes.size() == 1) {
      return true;
    }

    for (const auto& output_edge : outputs) {
      for (auto* input_node : inputs) {
        auto* output_node =
            op_nodes[output_edge.node_index]->outputs[output_edge.output_index];
        if (bindings_overlap(output_node, input_node)) {
          return true;
        }
      }
    }

    return false;
  }

  // Finds pairs of output and input bindings that overlap in memory. This is
  // used to detect cases where an operation's output writes to the same memory
  // region that is read by one of the subgraph's inputs, which can lead to
  // data corruption if not handled properly.
  bool FindOutputsOverlapInputs(
      std::vector<std::pair<BindingNode*, BindingNode*>>& overlapping_pairs) {
    for (const auto& output_edge : outputs) {
      for (auto* input_node : inputs) {
        auto* output_node =
            op_nodes[output_edge.node_index]->outputs[output_edge.output_index];
        if (bindings_overlap(output_node, input_node)) {
          overlapping_pairs.emplace_back(output_node, input_node);
        }
      }
    }
    return !overlapping_pairs.empty();
  }

  // Dumps a detailed, human-readable representation of the subgraph to the
  // provided output stream. This is useful for debugging the graph structure.
  void Dump(std::ostream& os) const {
    struct Value {
      enum class Type { kGraphInput, kIntermediate };
      Type type;
      size_t node_index;  // for kGraphInput, it's input index; for
                          // kIntermediate, it's op index
      size_t
          output_index;  // for kIntermediate, it's the output index of the op
    };

    std::unordered_map<ID3D12Resource*, Value> value_map;

    // Initialize value map with graph inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] && inputs[i]->resource) {
        value_map[inputs[i]->resource] = {Value::Type::kGraphInput, i, 0};
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
          if (input_binding->resource) {
            auto it = value_map.find(input_binding->resource);
            if (it != value_map.end()) {
              const auto& value = it->second;
              if (value.type == Value::Type::kGraphInput) {
                os << " -> Graph Input Index: " << value.node_index;
              } else {  // kIntermediate
                os << " -> Produced by Node[" << value.node_index
                   << "], Output[" << value.output_index << "]";
              }
            } else {
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
          OutputEdge maybe_output_edge{static_cast<UINT>(i),
                                       static_cast<UINT>(j)};

          auto it_graph_output =
              std::find(outputs.begin(), outputs.end(), maybe_output_edge);
          if (it_graph_output != outputs.end()) {
            os << " -> Graph Output Index: "
               << static_cast<UINT>(
                      std::distance(outputs.begin(), it_graph_output));
          }
        }
        os << "\n";
      }

      // Update value map with outputs of the current operator
      for (size_t j = 0; j < op_node->outputs.size(); ++j) {
        if (op_node->outputs[j] && op_node->outputs[j]->resource) {
          value_map[op_node->outputs[j]->resource] = {
              Value::Type::kIntermediate, i, j};
        }
      }
    }
  }

  // To execute a subgraph without creating a
  // fused DML graph. This executes operators one by one.
  void EvaluateSubGraphWithoutFusedGraph() const {
    // Now execute the graph operator by operator.
    for (const auto& op_node : op_nodes) {
      // Prepare DmlBindingArrayBundle for inputs
      std::vector<utils::DmlBufferBindingBundle> inputs_bundles;
      inputs_bundles.reserve(op_node->inputs.size());
      for (const BindingNode* binding_node : op_node->inputs) {
        inputs_bundles.emplace_back(binding_node->resource,
                                    binding_node->offset,
                                    binding_node->size_in_bytes);
      }
      utils::DmlBindingArrayBundle current_inputs(std::move(inputs_bundles));

      // Prepare DmlBindingArrayBundle for outputs
      std::vector<utils::DmlBufferBindingBundle> outputs_bundles;
      outputs_bundles.reserve(op_node->outputs.size());
      for (const BindingNode* binding_node : op_node->outputs) {
        outputs_bundles.emplace_back(binding_node->resource,
                                     binding_node->offset,
                                     binding_node->size_in_bytes);
      }
      utils::DmlBindingArrayBundle current_outputs(std::move(outputs_bundles));

      op_node->op->Execute(current_inputs, current_outputs);
    }
  }
};

// Anonymous namespace for helper functions used in correctness verification.
namespace {
// Downloads the content of all output buffers for a given subgraph.
// This is a helper function for verification, allowing comparison of graph
// execution results.
std::vector<std::vector<std::byte>> DownloadSubgraphOutputs(
    const SubGraph& subgraph,
    const char* context) {
  auto device = get_device();
  std::vector<std::vector<std::byte>> results;
  results.reserve(subgraph.outputs.size());

  for (const auto& output_edge : subgraph.outputs) {
    BindingNode* binding_node = subgraph.op_nodes[output_edge.node_index]
                                    ->outputs[output_edge.output_index];
    if (!binding_node->resource) {
      results.emplace_back();
      continue;
    }
    Microsoft::WRL::ComPtr<ID3D12Resource> resource(binding_node->resource);
    std::vector<std::byte> full_resource_data = device->Download(resource);

    std::vector<std::byte> buffer(binding_node->size_in_bytes);
    if (binding_node->offset + binding_node->size_in_bytes <=
        full_resource_data.size()) {
      memcpy(buffer.data(), full_resource_data.data() + binding_node->offset,
             binding_node->size_in_bytes);
    } else {
      SPDLOG_ERROR(
          "Invalid resource size during download for {} graph output. "
          "Resource size: {}, requested offset: {}, requested size: {}",
          context, full_resource_data.size(), binding_node->offset,
          binding_node->size_in_bytes);
      throw std::runtime_error("Invalid resource size during download for " +
                               std::string(context) + " graph output.");
    }
    results.push_back(std::move(buffer));
  }
  return results;
}

// Compares the output of a fused graph execution with a sequential, unfused
// execution. This function is intended for debugging and verifying the
// correctness of the graph fusion process. It works by:
// 1. Downloading the results from the output buffers, which are assumed to
//    have been just populated by a fused graph execution.
// 2. Re-executing the subgraph operator by operator using
//    EvaluateSubGraphWithoutFusedGraph. This overwrites the output buffers.
// 3. Downloading the new results from the output buffers.
// 4. Comparing the two sets of results and logging an error if they don't
//    match.
void CompareAndVerify(const SubGraph& subgraph) {
  // 1. Download results from fused graph execution (which just ran).
  const auto fused_results = DownloadSubgraphOutputs(subgraph, "fused");

  // 2. Execute unfused graph. This will modify output buffers.
  subgraph.EvaluateSubGraphWithoutFusedGraph();

  // 3. Download results from unfused graph execution.
  const auto unfused_results = DownloadSubgraphOutputs(subgraph, "unfused");

  // 4. Compare results.
  if (fused_results.size() != unfused_results.size()) {
    // This should be impossible if we get this far.
    throw std::logic_error(
        "Fused and unfused graph have different number of outputs.");
  }

  static size_t failed_times = 0U;
  static size_t success_times = 0U;
  for (size_t i = 0; i < fused_results.size(); ++i) {
    if (fused_results[i].size() != unfused_results[i].size()) {
      throw std::logic_error(
          "Fused and unfused graph have different output sizes for output " +
          std::to_string(i));
    }

    if (memcmp(fused_results[i].data(), unfused_results[i].data(),
               fused_results[i].size()) != 0) {
      std::ostringstream oss;
      subgraph.Dump(oss);
      // Mismatch found. Print details.
      SPDLOG_ERROR(
          "Mismatch found between fused and unfused graph execution "
          "for subgraph output {}, failed_times: {}, success time: {}, for "
          "subgraph:\n{}",
          i, ++failed_times, success_times, oss.str());
      // TODO: print more details about the mismatch.
      // For now, we only log the error. An exception was previously thrown
      // but has been commented out.
      // throw std::runtime_error(
      //     "Fused and unfused graph execution results do not match for output
      //     " + std::to_string(i));
    } else {
      success_times++;
    }
  }
  SPDLOG_DEBUG("Fused and unfused graph outputs match.");
}
}  // namespace

}  // namespace

// Initializes the GraphRecorder, including a special empty binding node used
// for operators that have no input or output.
GraphRecorder::GraphRecorder() : m_empty_binding_node({nullptr, 0, 0}) {
  auto allocate_function = [](uint64_t size, D3D12_RESOURCE_FLAGS) {
    auto device = get_device();
    return device->CreatePreferredDeviceMemoryBufferWithoutPooling(size);
  };
  m_allocator.reset(new BucketizedBufferAllocator(allocate_function));
}

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
  std::vector<BindingNode*> input_candidates;
  std::vector<BindingNode*> output_pushed_nodes;
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
        input_candidates.push_back(binding_node);
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
      output_pushed_nodes.push_back(binding_node);
    } else {
      if (output_descs[i].Type != DML_BINDING_TYPE_NONE) {
        throw std::invalid_argument(
            "Unsupported output binding type in Execute.");
      }
      op_node->outputs.push_back(&m_empty_binding_node);
    }
  }

  m_graph_inputs.insert(m_graph_inputs.end(), input_candidates.begin(),
                        input_candidates.end());

  m_operator_nodes.push_back(std::move(op_node));
  m_current_operator_nodes.push_back(m_operator_nodes.back().get());
}

// Finalizes the graph recording. This involves identifying graph outputs,
// splitting the graph into executable subgraphs, and then executing each
// subgraph.
void GraphRecorder::End() {
  m_has_begun = false;

  Flush();
  if (m_operator_nodes.empty()) {
    return;
  }

  std::vector<SubGraph> final_subgraphs = std::move(m_subgraphs);
  if (kDumpSubGraphs) {
    for (const auto& subgraph : final_subgraphs) {
      subgraph.Dump(std::cout);
    }
  }

  for (SubGraph& subgraph : final_subgraphs) {
    std::vector<std::pair<BindingNode*, BindingNode*>> overlapping_pairs;
    std::vector<Microsoft::WRL::ComPtr<IResourceWrapper>> overridden_inputs;
    if (subgraph.FindOutputsOverlapInputs(overlapping_pairs)) {
      auto device = get_device();
      for (const auto& pair : overlapping_pairs) {
        SPDLOG_TRACE("Overlapping output: {} with input: {}",
                     static_cast<void*>(pair.first->resource),
                     static_cast<void*>(pair.second->resource));
      }

      // Create a set of unique input nodes that overlap with outputs.
      std::unordered_set<BindingNode*> inputs_to_override;
      for (const auto& pair : overlapping_pairs) {
        inputs_to_override.insert(pair.second);
      }

      std::unordered_map<BindingNode*, BindingNode*> overridden_nodes_map;

      for (BindingNode* old_node : inputs_to_override) {
        if (!old_node || !old_node->resource) {
          continue;
        }

        // Create a new resource and copy the data from the old one.
        auto new_resource_wrapper = m_allocator->Alloc(
            old_node->size_in_bytes, D3D12_RESOURCE_FLAG_NONE);
        overridden_inputs.push_back(new_resource_wrapper);
        ID3D12Resource* new_resource = new_resource_wrapper->GetD3D12Resource();

        device->CopyResourceSubRegion(new_resource,       /*dst*/
                                      old_node->resource, /*src*/
                                      0,                  /*dstOffset*/
                                      old_node->offset,   /*srcOffset*/
                                      old_node->size_in_bytes);

        // Create a new binding node for the new resource.
        bool created;
        BindingNode* new_node = GetOrCreateBindingNode(
            new_resource, 0, old_node->size_in_bytes, created);
        overridden_nodes_map[old_node] = new_node;
      }

      for (auto& input_node : subgraph.inputs) {
        auto it = overridden_nodes_map.find(input_node);
        if (it != overridden_nodes_map.end()) {
          input_node = it->second;
        }
      }

      // Replace the old binding nodes with the new ones throughout the
      // subgraph.
      for (auto* op_node : subgraph.op_nodes) {
        // First, update the inputs of the current operator.
        for (auto& input_node : op_node->inputs) {
          auto it = overridden_nodes_map.find(input_node);
          if (it != overridden_nodes_map.end()) {
            input_node = it->second;
          }
        }

        // Then, check the outputs. If an output redefines an overridden input,
        // subsequent nodes should use the new value, so we remove it from the
        // map.
        for (auto* output_node : op_node->outputs) {
          std::unordered_map<BindingNode*, BindingNode*>::iterator it;
          for (it = overridden_nodes_map.begin();
               it != overridden_nodes_map.end();) {
            if (it->first->resource == output_node->resource) {
              // If the output node is an overridden input, remove it from the
              // map.
              it = overridden_nodes_map.erase(it);
            } else {
              ++it;
            }
          }
        }
        if (overridden_nodes_map.empty()) {
          // If all overridden nodes have been killed.
          break;
        }
      }
    }

    // Execute the finalized subgraphs.
    if (kDumpSubGraphs) {
      subgraph.Dump(std::cout);
    }

    if (subgraph.op_nodes.empty()) {
      continue;
    }
    if (kAlwaysEvaluateSubgraphs) {
      subgraph.EvaluateSubGraphWithoutFusedGraph();
      continue;
    }

    if (subgraph.IsSuitableForEvaluation()) {
      subgraph.EvaluateSubGraphWithoutFusedGraph();
      continue;
    }

    // Generate a cache key for the subgraph based on its operators.
    std::string key_accumulator = subgraph.GetCacheKey();

    auto& cache = DMLOperatorCache::instance();
    Operator* graph_op = cache.GetOperator(key_accumulator);
    Microsoft::WRL::ComPtr<Operator> new_graph_op_comptr;

    if (kAlwaysRecompileSubgraphs || !graph_op) {
      // If the compiled graph is not in the cache, build it.
      std::vector<BindingNode*> sorted_outputs;
      for (const auto output_edge : subgraph.outputs) {
        sorted_outputs.push_back(subgraph.op_nodes[output_edge.node_index]
                                     ->outputs[output_edge.output_index]);
      }
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

      if (kDumpSubGraphBeforeBuild) {
        std::cerr << "Subgraph before build:\n";
        subgraph.Dump(std::cerr);
      }

      // Build the compiled graph from the subgraph definition.
      // NOTE: The GraphBuilder::Build API expects a vector of unique_ptr, but
      // SubGraph provides raw pointers. This implementation assumes the Build
      // API can be overloaded or changed to accept raw pointers, as the
      // recorder's main list owns the operators.
      auto compiled_graph = GraphBuilder::Build(
          subgraph.op_nodes, subgraph.inputs, subgraph.outputs,
          DML_EXECUTION_FLAG_NONE, key_accumulator);

      auto device = get_device();
      new_graph_op_comptr = Microsoft::WRL::Make<Operator>(
          device.Get(), utils::DmlOperatorDescBundle(), nullptr,
          std::move(compiled_graph), key_accumulator);

      graph_op = new_graph_op_comptr.Get();
      device->KeepAliveUntilNextCommandListDispatch(new_graph_op_comptr);
      cache.AddOperator(std::move(key_accumulator),
                        std::move(new_graph_op_comptr));
      if (kDumpSubGraphAfterCacheHit) {
        std::cerr << "Caching subgraph:\n";
        subgraph.Dump(std::cerr);
      }
    } else {
      if (kDumpSubGraphAfterCacheHit) {
        std::cerr << "Get graph op from cache: " << key_accumulator.size()
                  << std::endl;
        std::cerr << "Subgraph after cache hit:\n";
        subgraph.Dump(std::cerr);
      }
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
    for (const auto& output_edge : subgraph.outputs) {
      BindingNode* binding_node = subgraph.op_nodes[output_edge.node_index]
                                      ->outputs[output_edge.output_index];
      output_binding_bundles.emplace_back(binding_node->resource,
                                          binding_node->offset,
                                          binding_node->size_in_bytes);
    }

    utils::DmlBindingArrayBundle inputs(std::move(input_binding_bundles));
    utils::DmlBindingArrayBundle outputs(std::move(output_binding_bundles));

    // Execute the compiled graph.
    graph_op->Execute(inputs, outputs);

    if (kCompareWithUnfused) {
      CompareAndVerify(subgraph);
    }
  }

  Reset();
}

// Flushes the currently recorded operators into a new subgraph. This function
// is called to finalize a segment of the graph, typically when a data
// dependency forces a split. It moves the current operator nodes and graph
// inputs into a new SubGraph object and clears the current resource bindings.
void GraphRecorder::Flush() {
  if (m_operator_nodes.empty()) {
    return;
  }
  m_subgraphs.emplace_back(std::move(m_current_operator_nodes),
                           std::move(m_graph_inputs));
  m_current_resource_bindings.clear();
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

  auto found = m_current_resource_bindings.find(resource);
  if (found != m_current_resource_bindings.end()) {
    if (found->second->offset != offset &&
        found->second->size_in_bytes != size) {
      throw std::runtime_error(
          "Resource already exists with different offset or size.");
    }
  }
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
  m_all_binding_nodes.push_back(std::move(new_node));
  // Add to lookup map for future access
  m_binding_lookup[key] = raw_ptr;
  // Track current resource
  m_current_resource_bindings[resource] = raw_ptr;
  // Node was created
  created = true;

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