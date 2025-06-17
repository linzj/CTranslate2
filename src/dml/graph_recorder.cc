#include "graph_recorder.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "backend_dml.h"
#include "dml/operator_cache.h"
#include "dml/operator_utils.h"
#include "dml_utils.h"

namespace ctranslate2 {
namespace dml {

GraphRecorder::GraphRecorder()
    : m_empty_binding_node({nullptr, 0, 0, false, false}) {}

GraphRecorder::~GraphRecorder() = default;

void GraphRecorder::Begin() {
  m_has_begun = true;  // Mark that Begin has been called
}

void GraphRecorder::Execute(Operator* op,
                            const utils::DmlBindingArrayBundle& inputs,
                            const utils::DmlBindingArrayBundle& outputs) {
  if (!m_has_begun) {
    throw std::invalid_argument("GraphRecorder::Execute called before Begin.");
  }
  // Create a new operator node
  std::unique_ptr<OperatorNode> op_node = std::make_unique<OperatorNode>();
  op_node->op = op;

  // Process inputs
  auto input_descs = inputs.get_descs();
  for (size_t i = 0; i < input_descs.size(); ++i) {
    if (input_descs[i].Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(input_descs[i].Desc);
      bool node_created;  // Track if a new node was created
      BindingNode* binding_node = GetOrCreateBindingNode(
          buffer_binding->Buffer, buffer_binding->Offset,
          buffer_binding->SizeInBytes, node_created, true);
      op_node->inputs.push_back(binding_node);

      // If an input binding node is newly created, it's a graph input
      if (node_created) {
        // Only mark as graph input if this is the first operator being recorded
        // and the binding node was not previously encountered as an output
        binding_node->is_graph_input = true;
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

  // Process outputs
  auto output_descs = outputs.get_descs();
  for (size_t i = 0; i < output_descs.size(); ++i) {
    if (output_descs[i].Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(output_descs[i].Desc);
      bool node_created;
      BindingNode* binding_node = GetOrCreateBindingNode(
          buffer_binding->Buffer, buffer_binding->Offset,
          buffer_binding->SizeInBytes, node_created, false);
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

void GraphRecorder::EvaluateGraphWithoutFusedGraph() {
  // Mark the last operator's outputs as graph outputs.
  OperatorNode* last_op_node = m_operator_nodes.back().get();
  for (BindingNode* output_node : last_op_node->outputs) {
    output_node->is_graph_output = true;
    m_graph_outputs.push_back(output_node);
  }

  // Now execute the graph
  for (const auto& op_node : m_operator_nodes) {
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
  Reset();
}

void GraphRecorder::End() {
  m_has_begun = false;

  constexpr bool kEvaluateWithoutFusedGraph = true;
  if (kEvaluateWithoutFusedGraph) {
    EvaluateGraphWithoutFusedGraph();
    return;
  }

  if (m_operator_nodes.empty()) {
    std::cout << "Graph is empty, nothing to execute." << std::endl;
    return;
  }
  // Mark graph outputs
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
        output_node->is_graph_output = true;
        m_graph_outputs.push_back(output_node);
      }
    }
  }

  std::string key_accumulator = "Graph:";
  for (const auto& op_node : m_operator_nodes) {
    key_accumulator += op_node->op->key();
  }

  auto& cache = DMLOperatorCache::instance();
  Operator* graph_op = cache.GetOperator(key_accumulator);
  bool cache_hit = graph_op != nullptr;
  (void)cache_hit;  // Suppress unused variable warning
  Microsoft::WRL::ComPtr<Operator> new_graph_op_comptr;

  if (!graph_op) {
#if 0
    // Check if any resource is both a graph input and a graph output.
    std::unordered_set<ID3D12Resource*> graph_input_resources;
    for (const auto& input_node : m_graph_inputs) {
      graph_input_resources.insert(input_node->resource);
    }
    for (const auto& output_node : m_graph_outputs) {
      if (graph_input_resources.count(output_node->resource)) {
        // Find the operator that produces this output.
        const Operator* producer_op = nullptr;
        for (const auto& op_node : m_operator_nodes) {
          for (const auto& out_node : op_node->outputs) {
            if (out_node->resource == output_node->resource) {
              producer_op = op_node->op;
              break;
            }
          }
          if (producer_op) {
            break;
          }
        }

        // Find all operators that consume this input.
        std::string consumer_op_keys;
        for (const auto& op_node : m_operator_nodes) {
          for (const auto& in_node : op_node->inputs) {
            if (in_node->resource == output_node->resource) {
              if (!consumer_op_keys.empty()) {
                consumer_op_keys += ", ";
              }
              consumer_op_keys += std::string("'") +
                                  OperatorUtils::DML_OPERATOR_TYPE_toString(
                                      op_node->op->GetType()) +
                                  "'";
            }
          }
        }

        SPDLOG_ERROR(
            "Graph input and output collision detected on resource {}. The "
            "resource is used as an input by operator(s) {} and is also an "
            "output of operator '{}'.",
            (void*)output_node->resource, consumer_op_keys,
            producer_op ? OperatorUtils::DML_OPERATOR_TYPE_toString(
                              producer_op->GetType())
                        : "N/A");
        throw std::invalid_argument(
            "A resource cannot be both a graph input and a graph output.");
      }
    }
#endif

    // Lookup for the graph outputs the share the same resource.
    // If found throw invalid_argument.
    std::sort(m_graph_outputs.begin(), m_graph_outputs.end(),
              [](const BindingNode* a, const BindingNode* b) {
                if (a->resource != b->resource) {
                  return a->resource < b->resource;
                }
                if (a->offset != b->offset) {
                  return a->offset < b->offset;
                }
                return a->size_in_bytes < b->size_in_bytes;
              });
    auto it = std::adjacent_find(
        m_graph_outputs.begin(), m_graph_outputs.end(),
        [](const BindingNode* a, const BindingNode* b) {
          return a->resource == b->resource && a->offset == b->offset &&
                 a->size_in_bytes == b->size_in_bytes;
        });

    if (it != m_graph_outputs.end()) {
      throw std::invalid_argument(
          "Multiple graph outputs bind to the same resource.");
    }

    auto compiled_graph =
        GraphBuilder::Build(m_operator_nodes, m_graph_inputs, m_graph_outputs,
                            DML_EXECUTION_FLAG_NONE, key_accumulator);

    new_graph_op_comptr = Microsoft::WRL::Make<Operator>(
        get_device(), utils::DmlOperatorDescBundle(), nullptr,
        std::move(compiled_graph), key_accumulator);

    graph_op = new_graph_op_comptr.Get();
    cache.AddOperator(std::move(key_accumulator),
                      std::move(new_graph_op_comptr));
  }

  // Execute the graph
  std::vector<utils::DmlBufferBindingBundle> input_binding_bundles;
  input_binding_bundles.reserve(m_graph_inputs.size());
  for (const auto& binding_node : m_graph_inputs) {
    input_binding_bundles.emplace_back(binding_node->resource,
                                       binding_node->offset,
                                       binding_node->size_in_bytes);
  }

  std::vector<utils::DmlBufferBindingBundle> output_binding_bundles;
  output_binding_bundles.reserve(m_graph_outputs.size());
  for (const auto& binding_node : m_graph_outputs) {
    output_binding_bundles.emplace_back(binding_node->resource,
                                        binding_node->offset,
                                        binding_node->size_in_bytes);
  }

  utils::DmlBindingArrayBundle inputs(std::move(input_binding_bundles));
  utils::DmlBindingArrayBundle outputs(std::move(output_binding_bundles));

  graph_op->Execute(inputs, outputs);

// After execution, you might want to log or return something,
// e.g., the graph inputs and outputs.
#if 0
  std::cout << "Graph execution finished." << std::endl;
  std::cout << "Graph Inputs: " << m_graph_inputs.size() << std::endl;
  std::cout << "Graph Outputs: " << m_graph_outputs.size() << std::endl;
#endif
  Reset();
}

BindingNode* GraphRecorder::GetOrCreateBindingNode(ID3D12Resource* resource,
                                                   UINT64 offset,
                                                   UINT64 size,
                                                   bool& created,
                                                   bool is_input) {
  std::tuple<ID3D12Resource*, UINT64, UINT64> key =
      std::make_tuple(resource, offset, size);

  if (is_input) {
    auto it = m_binding_lookup.find(key);
    if (it != m_binding_lookup.end()) {
      created = false;    // Node already exists
      return it->second;  // Return existing node
    }
  }

  // Create a new binding node
  std::unique_ptr<BindingNode> new_node = std::make_unique<BindingNode>();
  new_node->resource = resource;
  new_node->offset = offset;
  new_node->size_in_bytes = size;

  BindingNode* raw_ptr = new_node.get();
  m_all_binding_nodes.push_back(std::move(new_node));  // Store ownership
  m_binding_lookup[key] = raw_ptr;  // Store raw pointer in lookup map
  created = true;                   // Node was created

  return raw_ptr;
}

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