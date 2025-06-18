#include "graph_builder.h"

#include "backend_dml.h"
#include "common.h"
#include "graph_recorder.h"
#include "operator.h"
#include "operator_utils.h"

#include <iostream>
#include <map>
#include <vector>

namespace ctranslate2 {
namespace dml {

namespace {

constexpr const bool kDumpGraphForDebug = false;
void check_for_graph_binding_overlaps(
    std::ostream& os,
    const std::vector<BindingNode*>& graph_inputs,
    const std::vector<BindingNode*>& graph_outputs) {
  bool overlap_found = false;

  for (const auto* input_binding : graph_inputs) {
    if (!input_binding || !input_binding->resource)
      continue;

    for (const auto* output_binding : graph_outputs) {
      if (!output_binding || !output_binding->resource)
        continue;

      if (input_binding->resource == output_binding->resource) {
        UINT64 input_end = input_binding->offset + input_binding->size_in_bytes;
        UINT64 output_end =
            output_binding->offset + output_binding->size_in_bytes;

        if (input_binding->offset < output_end &&
            input_end > output_binding->offset) {
          overlap_found = true;
          os << "  OVERLAP DETECTED between graph input and output:\n";
          os << "    Input Binding:  Resource: " << input_binding->resource
             << ", Offset: " << input_binding->offset
             << ", Size: " << input_binding->size_in_bytes << "\n";
          os << "    Output Binding: Resource: " << output_binding->resource
             << ", Offset: " << output_binding->offset
             << ", Size: " << output_binding->size_in_bytes << "\n";
        }
      }
    }
  }

  if (overlap_found) {
    throw std::runtime_error(
        "Graph input and output bindings overlap! Please check the graph "
        "inputs and outputs for conflicts.");
  }
}

void dump_graph_for_debug(
    std::ostream& os,
    const DML_GRAPH_DESC& graph_desc,
    const std::vector<DML_GRAPH_NODE_DESC>& graph_nodes,
    const std::vector<OperatorNode*>& operator_nodes,
    const std::map<const BindingNode*, std::pair<uint32_t, uint32_t>>&
        intermediate_producers,
    const std::map<const BindingNode*, uint32_t>& graph_input_to_index,
    const std::map<const BindingNode*, uint32_t>& graph_output_to_index,
    const std::vector<BindingNode*>& graph_inputs,
    const std::vector<BindingNode*>& graph_outputs) {
  os << "Dumping DML Graph State for Debugging:\n";
  os << "=======================================\n";

  os << "Graph Description:\n";
  os << "  InputCount: " << graph_desc.InputCount << "\n";
  os << "  OutputCount: " << graph_desc.OutputCount << "\n";
  os << "  NodeCount: " << graph_desc.NodeCount << "\n";
  os << "  InputEdgeCount: " << graph_desc.InputEdgeCount << "\n";
  os << "  OutputEdgeCount: " << graph_desc.OutputEdgeCount << "\n";
  os << "  IntermediateEdgeCount: " << graph_desc.IntermediateEdgeCount << "\n";
  os << "\n";

  os << "Nodes (" << operator_nodes.size() << "):\n";
  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    os << "  Node[" << i << "]: "
       << OperatorUtils::DML_OPERATOR_TYPE_toString(op_node->op->GetType())
       << "\n";

    os << "    Inputs (" << op_node->inputs.size() << "):\n";
    for (size_t j = 0; j < op_node->inputs.size(); ++j) {
      const auto& input_binding = op_node->inputs[j];
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
      const auto& output_binding = op_node->outputs[j];
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
  os << "\n";

  os << "Edges:\n";
  os << "  Input Edges (" << graph_desc.InputEdgeCount << "):\n";
  if (graph_desc.InputEdges) {
    for (size_t i = 0; i < graph_desc.InputEdgeCount; ++i) {
      const auto* edge = static_cast<const DML_INPUT_GRAPH_EDGE_DESC*>(
          graph_desc.InputEdges[i].Desc);
      os << "    - GraphInputIndex: " << edge->GraphInputIndex
         << " -> ToNodeIndex: " << edge->ToNodeIndex
         << ", ToNodeInputIndex: " << edge->ToNodeInputIndex << "\n";
    }
  }

  os << "  Intermediate Edges (" << graph_desc.IntermediateEdgeCount << "):\n";
  if (graph_desc.IntermediateEdges) {
    for (size_t i = 0; i < graph_desc.IntermediateEdgeCount; ++i) {
      const auto* edge = static_cast<const DML_INTERMEDIATE_GRAPH_EDGE_DESC*>(
          graph_desc.IntermediateEdges[i].Desc);
      os << "    - FromNodeIndex: " << edge->FromNodeIndex
         << ", FromNodeOutputIndex: " << edge->FromNodeOutputIndex
         << " -> ToNodeIndex: " << edge->ToNodeIndex
         << ", ToNodeInputIndex: " << edge->ToNodeInputIndex << "\n";
    }
  }

  os << "  Output Edges (" << graph_desc.OutputEdgeCount << "):\n";
  if (graph_desc.OutputEdges) {
    for (size_t i = 0; i < graph_desc.OutputEdgeCount; ++i) {
      const auto* edge = static_cast<const DML_OUTPUT_GRAPH_EDGE_DESC*>(
          graph_desc.OutputEdges[i].Desc);
      os << "    - FromNodeIndex: " << edge->FromNodeIndex
         << ", FromNodeOutputIndex: " << edge->FromNodeOutputIndex
         << " -> GraphOutputIndex: " << edge->GraphOutputIndex << "\n";
    }
  }

  os << "=======================================\n";
}

}  // namespace

// Builds a DirectML graph from a collection of operator nodes and compiles it.
Microsoft::WRL::ComPtr<IDMLCompiledOperator> GraphBuilder::Build(
    const std::vector<OperatorNode*>& operator_nodes,
    const std::vector<BindingNode*>& graph_inputs,
    const std::vector<BindingNode*>& graph_outputs,
    DML_EXECUTION_FLAGS flags,
    const std::string& key) {
  // Get the DML device and query for the IDMLDevice1 interface, which is
  // required for graph compilation.
  auto device = dml::get_device()->DML();
  Microsoft::WRL::ComPtr<IDMLDevice1> device1;
  THROW_IF_FAILED(device->QueryInterface(IID_PPV_ARGS(&device1)));

  // Initialize the graph description with the number of inputs, outputs, and
  // nodes.
  DML_GRAPH_DESC graph_desc = {};
  graph_desc.InputCount = graph_inputs.size();
  graph_desc.OutputCount = graph_outputs.size();
  graph_desc.NodeCount = operator_nodes.size();

  // Prepare vectors to hold the descriptions for DML graph nodes.
  // DML_OPERATOR_GRAPH_NODE_DESC contains the actual DML operator.
  // DML_GRAPH_NODE_DESC is a generic wrapper for different node types.
  std::vector<DML_OPERATOR_GRAPH_NODE_DESC> node_descs;
  std::vector<DML_GRAPH_NODE_DESC> graph_nodes;
  node_descs.reserve(operator_nodes.size());
  graph_nodes.reserve(operator_nodes.size());

  // Map each operator to its index in the graph_nodes vector for later
  // reference.
  std::map<const Operator*, uint32_t> operator_to_node_index;

  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    node_descs.push_back({op_node->op->GetDMLOperator(), nullptr});
    graph_nodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, &node_descs.back()});
    operator_to_node_index[op_node->op] = i;
  }
  graph_desc.Nodes = graph_nodes.data();

  // Prepare data structures for input edges.
  // `graph_input_to_index` maps each graph input binding node to its index in
  // the graph's input list.
  std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> input_edges;
  std::map<const BindingNode*, uint32_t> graph_input_to_index;

  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    graph_input_to_index[graph_inputs[i]] = i;
  }

  // Prepare data structures for output edges.
  // `graph_output_to_index` maps each graph output binding node to its index in
  // the graph's output list.
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> output_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> output_edges;
  std::map<const BindingNode*, uint32_t> graph_output_to_index;
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    graph_output_to_index[graph_outputs[i]] = i;
  }

  // Prepare data structures for intermediate edges, which connect nodes within
  // the graph.
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> intermediate_edges;

  // Identify which node and output index produces each intermediate tensor.
  // This map will be used to connect operator inputs to the outputs of other
  // operators.
  std::map<const BindingNode*, std::pair<uint32_t, uint32_t>>
      intermediate_producers;
  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    for (size_t j = 0; j < op_node->outputs.size(); ++j) {
      auto output = op_node->outputs[j];
      if (output && output->resource) {
        if (intermediate_producers.find(output) !=
            intermediate_producers.end()) {
          throw std::invalid_argument(
              "Intermediate output already has a producer.");
        }
        intermediate_producers[op_node->outputs[j]] = {(uint32_t)i,
                                                       (uint32_t)j};
      }
    }
  }

  // To avoid dangling pointers when vectors are reallocated, we first count
  // the number of edges to reserve the required capacity.
  size_t input_edge_count = 0;
  size_t intermediate_edge_count = 0;
  size_t output_edge_count = 0;
  for (const auto& op_node : operator_nodes) {
    for (const auto& input_binding : op_node->inputs) {
      if (input_binding && input_binding->resource) {
        if (graph_input_to_index.count(input_binding)) {
          input_edge_count++;
        } else {
          intermediate_edge_count++;
        }
      }
    }
    for (const auto& output_binding : op_node->outputs) {
      if (output_binding && graph_output_to_index.count(output_binding)) {
        output_edge_count++;
      }
    }
  }
  input_edge_descs.reserve(input_edge_count);
  input_edges.reserve(input_edge_count);
  intermediate_edge_descs.reserve(intermediate_edge_count);
  intermediate_edges.reserve(intermediate_edge_count);
  output_edge_descs.reserve(output_edge_count);
  output_edges.reserve(output_edge_count);

  // Iterate through all operator nodes to define the connections (edges)
  // between them.
  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    uint32_t dml_input_idx = 0;
    for (size_t j = 0; j < op_node->inputs.size(); ++j) {
      BindingNode* input_binding = op_node->inputs[j];
      if (input_binding && input_binding->resource) {
        auto it_graph_input = graph_input_to_index.find(input_binding);
        if (it_graph_input != graph_input_to_index.end()) {
          // This input is a main graph input. Create an input edge.
          input_edge_descs.push_back(
              {it_graph_input->second, (UINT)i, dml_input_idx, nullptr});
          input_edges.push_back(
              {DML_GRAPH_EDGE_TYPE_INPUT, &input_edge_descs.back()});
        } else {
          // This input is produced by another node in the graph. Create an
          // intermediate edge.
          auto it = intermediate_producers.find(input_binding);
          if (it == intermediate_producers.end())
            THROW_RUNTIME_ERROR("Intermediate input has no producer.");
          const auto& producer = it->second;
          intermediate_edge_descs.push_back({producer.first, producer.second,
                                             (UINT)i, dml_input_idx, nullptr});
          intermediate_edges.push_back({DML_GRAPH_EDGE_TYPE_INTERMEDIATE,
                                        &intermediate_edge_descs.back()});
        }
      }
      dml_input_idx++;
    }

    // Check for outputs that are also main graph outputs.
    for (size_t j = 0; j < op_node->outputs.size(); ++j) {
      BindingNode* output_binding = op_node->outputs[j];
      auto it_graph_output = graph_output_to_index.find(output_binding);
      if (output_binding && it_graph_output != graph_output_to_index.end()) {
        // This output is a main graph output. Create an output edge.
        output_edge_descs.push_back(
            {(UINT)i, (UINT)j, it_graph_output->second, nullptr});
        output_edges.push_back(
            {DML_GRAPH_EDGE_TYPE_OUTPUT, &output_edge_descs.back()});
      }
    }
  }

  // Finalize the graph description with all the edge information.
  graph_desc.InputEdgeCount = input_edges.size();
  graph_desc.InputEdges = input_edges.data();
  graph_desc.OutputEdgeCount = output_edges.size();
  graph_desc.OutputEdges = output_edges.data();
  graph_desc.IntermediateEdgeCount = intermediate_edges.size();
  graph_desc.IntermediateEdges = intermediate_edges.data();

  if (kDumpGraphForDebug) {
    dump_graph_for_debug(std::cerr, graph_desc, graph_nodes, operator_nodes,
                         intermediate_producers, graph_input_to_index,
                         graph_output_to_index, graph_inputs, graph_outputs);
  }

  check_for_graph_binding_overlaps(std::cerr, graph_inputs, graph_outputs);

  // Compile the graph description into a runnable, optimized operator.
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_graph;
  THROW_IF_FAILED(
      device1->CompileGraph(&graph_desc, flags, IID_PPV_ARGS(&compiled_graph)));

  // Return the compiled graph object.
  return compiled_graph;
}

}  // namespace dml
}  // namespace ctranslate2
