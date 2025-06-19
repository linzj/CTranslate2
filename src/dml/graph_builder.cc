#include "graph_builder.h"

#include "backend_dml.h"
#include "common.h"
#include "graph_recorder.h"
#include "operator.h"
#include "operator_utils.h"

#include <iostream>
#include <unordered_map>
#include <vector>

namespace ctranslate2 {
namespace dml {

namespace {

constexpr const bool kDumpGraphForDebug = true;

enum class ValueType {
  GRAPH_INPUT,
  INTERMEDIATE_OUTPUT,
};

struct Value {
  ValueType type;
  union {
    uint32_t graph_input_index;
    struct {
      uint32_t node_index;
      uint32_t node_output_index;
    } producer;
  };
};

void check_for_graph_binding_overlaps(
    std::ostream& os,
    const std::vector<OperatorNode*>& operator_nodes,
    const std::vector<BindingNode*>& graph_inputs,
    const std::vector<OutputEdge>& graph_outputs) {
  bool overlap_found = false;

  for (const auto* input_binding : graph_inputs) {
    if (!input_binding || !input_binding->resource)
      continue;

    for (const auto output_edge : graph_outputs) {
      const auto* output_binding = operator_nodes[output_edge.node_index]
                                       ->outputs[output_edge.output_index];
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
    const std::unordered_map<const ID3D12Resource*, Value>& value_producers,
    const std::vector<BindingNode*>& graph_inputs,
    const std::vector<OutputEdge>& graph_outputs) {
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
        if (input_binding->resource) {
          auto it = value_producers.find(input_binding->resource);
          if (it != value_producers.end()) {
            const auto& producer_info = it->second;
            if (producer_info.type == ValueType::GRAPH_INPUT) {
              os << " -> Graph Input Index: "
                 << producer_info.graph_input_index;
            } else {  // INTERMEDIATE_OUTPUT
              os << " -> Produced by Node[" << producer_info.producer.node_index
                 << "], Output[" << producer_info.producer.node_output_index
                 << "]";
            }
          } else {
            os << " -> ERROR: Producer not found!";
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
        if (output_binding->resource) {
          OutputEdge maybe_output_edge = {static_cast<UINT>(i), (UINT)j};
          auto it_graph_output = std::find(
              graph_outputs.begin(), graph_outputs.end(), maybe_output_edge);
          if (it_graph_output != graph_outputs.end()) {
            os << " -> Graph Output Index: "
               << static_cast<UINT>(
                      std::distance(graph_outputs.begin(), it_graph_output));
          }
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
    const std::vector<OutputEdge>& graph_outputs,
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

  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    node_descs.push_back({op_node->op->GetDMLOperator(), nullptr});
    graph_nodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR, &node_descs.back()});
  }
  graph_desc.Nodes = graph_nodes.data();

  // Map to track the origin of each tensor value in the graph.
  std::unordered_map<const ID3D12Resource*, Value> value_producers;

  // Initialize with graph inputs.
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    const auto& input_binding = graph_inputs[i];
    if (input_binding && input_binding->resource) {
      Value val;
      val.type = ValueType::GRAPH_INPUT;
      val.graph_input_index = static_cast<uint32_t>(i);
      value_producers[input_binding->resource] = val;
    }
  }

  std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> input_edges;
  std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> output_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> output_edges;
  std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edge_descs;
  std::vector<DML_GRAPH_EDGE_DESC> intermediate_edges;

  // Iterate through all operator nodes to define the connections (edges).
  for (size_t i = 0; i < operator_nodes.size(); ++i) {
    const auto& op_node = operator_nodes[i];
    uint32_t dml_input_idx = 0;
    for (const auto& input_binding : op_node->inputs) {
      if (input_binding && input_binding->resource) {
        auto it = value_producers.find(input_binding->resource);
        if (it == value_producers.end())
          THROW_RUNTIME_ERROR("Intermediate input has no producer.");

        const auto& producer_info = it->second;
        if (producer_info.type == ValueType::GRAPH_INPUT) {
          input_edge_descs.push_back({producer_info.graph_input_index, (UINT)i,
                                      dml_input_idx, nullptr});
        } else {  // INTERMEDIATE_OUTPUT
          intermediate_edge_descs.push_back(
              {producer_info.producer.node_index,
               producer_info.producer.node_output_index, (UINT)i, dml_input_idx,
               nullptr});
        }
      }
      dml_input_idx++;
    }

    // Check for outputs that are also main graph outputs and update producers.
    for (size_t j = 0; j < op_node->outputs.size(); ++j) {
      BindingNode* output_binding = op_node->outputs[j];
      if (output_binding && output_binding->resource) {
        OutputEdge maybe_output_edge = {static_cast<UINT>(i), (UINT)j};

        auto it_graph_output = std::find(
            graph_outputs.begin(), graph_outputs.end(), maybe_output_edge);
        if (it_graph_output != graph_outputs.end()) {
          output_edge_descs.push_back(
              {(UINT)i, (UINT)j,
               static_cast<UINT>(
                   std::distance(graph_outputs.begin(), it_graph_output)),
               nullptr});
        }
        // This output is now a value producer for subsequent nodes.
        Value val;
        val.type = ValueType::INTERMEDIATE_OUTPUT;
        val.producer.node_index = (uint32_t)i;
        val.producer.node_output_index = (uint32_t)j;
        value_producers[output_binding->resource] = val;
      }
    }
  }
  for (const auto& input_edge_desc : input_edge_descs) {
    input_edges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, &input_edge_desc});
  }
  for (const auto& output_edge_desc : output_edge_descs) {
    output_edges.push_back({DML_GRAPH_EDGE_TYPE_OUTPUT, &output_edge_desc});
  }
  for (const auto& intermediate_edge_desc : intermediate_edge_descs) {
    intermediate_edges.push_back(
        {DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediate_edge_desc});
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
                         value_producers, graph_inputs, graph_outputs);
  }

  check_for_graph_binding_overlaps(std::cerr, operator_nodes, graph_inputs,
                                   graph_outputs);

  // Compile the graph description into a runnable, optimized operator.
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_graph;
  THROW_IF_FAILED(
      device1->CompileGraph(&graph_desc, flags, IID_PPV_ARGS(&compiled_graph)));

  // Return the compiled graph object.
  return compiled_graph;
}

}  // namespace dml
}  // namespace ctranslate2
