#include "graph_recorder.h"
#include <iostream>
#include <stdexcept>

namespace ctranslate2 {
namespace dml {

GraphRecorder::GraphRecorder()
    : m_empty_binding_node({nullptr, 0, 0, false, false}) {}

GraphRecorder::~GraphRecorder() = default;

void GraphRecorder::Begin() {
  // Clear all existing states
  m_all_binding_nodes.clear();
  m_binding_lookup.clear();
  m_operator_nodes.clear();
  m_graph_inputs.clear();
  m_graph_outputs.clear();
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
      BindingNode* binding_node =
          GetOrCreateBindingNode(buffer_binding->Buffer, buffer_binding->Offset,
                                 buffer_binding->SizeInBytes, node_created);
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
      BindingNode* binding_node =
          GetOrCreateBindingNode(buffer_binding->Buffer, buffer_binding->Offset,
                                 buffer_binding->SizeInBytes, node_created);
      op_node->outputs.push_back(binding_node);

      // If an output node was previously marked as a graph input, it is no
      // longer an input. This happens when an intermediate tensor is produced
      // by one op and consumed by another.
      auto it =
          std::find(m_graph_inputs.begin(), m_graph_inputs.end(), binding_node);
      if (it != m_graph_inputs.end()) {
        m_graph_inputs.erase(it);
        binding_node->is_graph_input = false;
      }
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

void GraphRecorder::End() {
  m_has_begun = false;
  if (m_operator_nodes.empty()) {
    std::cout << "Graph is empty, nothing to execute." << std::endl;
    return;
  }

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

// After execution, you might want to log or return something,
// e.g., the graph inputs and outputs.
#if 1
  std::cout << "Graph execution finished." << std::endl;
  std::cout << "Graph Inputs: " << m_graph_inputs.size() << std::endl;
  std::cout << "Graph Outputs: " << m_graph_outputs.size() << std::endl;
#endif
}

BindingNode* GraphRecorder::GetOrCreateBindingNode(ID3D12Resource* resource,
                                                   UINT64 offset,
                                                   UINT64 size,
                                                   bool& created) {
  std::tuple<ID3D12Resource*, UINT64, UINT64> key =
      std::make_tuple(resource, offset, size);

  auto it = m_binding_lookup.find(key);
  if (it != m_binding_lookup.end()) {
    created = false;    // Node already exists
    return it->second;  // Return existing node
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

}  // namespace dml
}  // namespace ctranslate2