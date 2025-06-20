#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "graph_builder.h"

namespace ctranslate2 {
namespace dml {
namespace {
class SubGraph;
}

class Device;  // Forward declaration for Device
class BucketizedBufferAllocator;

// Represents a unique DML resource binding within the graph. Each binding is
// identified by the combination of a D3D12 resource, an offset, and a size.
// This struct tracks the binding's properties and its role within the larger
// graph.
struct BindingNode {
  Microsoft::WRL::ComPtr<IResourceWrapper> resource;
  UINT64 offset;
  UINT64 size_in_bytes;
};

// Represents a single operator and its connections within the computation
// graph. It holds a pointer to the operator itself and lists of its input and
// output bindings.
struct OperatorNode {
  Operator* op;                       // The DML operator to be executed.
  std::vector<BindingNode*> inputs;   // Pointers to the input BindingNodes.
  std::vector<BindingNode*> outputs;  // Pointers to the output BindingNodes.
};

// Represents a connection from an operator's output to a subsequent part of
// the graph. It identifies a specific output from a specific operator node.
struct OutputEdge {
  UINT32 node_index;    // Index of the operator node producing this output.
  UINT32 output_index;  // Index of the output binding within the operator.
  bool operator==(const OutputEdge& other) const {
    return node_index == other.node_index && output_index == other.output_index;
  }
};

// The GraphRecorder class is responsible for capturing a sequence of DML
// operator executions and transforming them into a series of executable DML
// graphs. It handles resource dependencies, splits the graph when necessary,
// and manages the lifecycle of graph compilation and execution.
class GraphRecorder {
 public:
  explicit GraphRecorder();
  ~GraphRecorder();

  // Starts a recording session. This must be called before any operators are
  // executed. It resets the internal state of the recorder to prepare for a new
  // graph.
  void Begin();

  // Records the execution of a single operator. This method captures the
  // operator and its input/output bindings, creating a node in the graph
  // representation.
  void Execute(Operator* op,
               const utils::DmlBindingArrayBundle& inputs,
               const utils::DmlBindingArrayBundle& outputs);

  // Finalizes the recording session. This method analyzes the recorded
  // operators, determines the final graph outputs, splits the graph into
  // manageable subgraphs based on data dependencies, and then compiles and
  // executes each subgraph.
  void End();

  // Returns true if a recording session is currently active (i.e., Begin() has
  // been called but End() has not).
  bool has_begun() const { return m_has_begun; }

 private:
  // Resets all internal states of the recorder, clearing all nodes and graph
  // data.
  void Reset();

  // Retrieves an existing BindingNode for a given resource or creates a new
  // one. This ensures that each unique resource binding is represented by a
  // single node.
  BindingNode* GetOrCreateBindingNode(IResourceWrapper* resource,
                                      UINT64 offset,
                                      UINT64 size,
                                      bool& created);

  // Flushes the current set of recorded operators into a new subgraph. This is
  // typically called when a data dependency requires a break in the graph.
  void Flush();

  std::unique_ptr<BucketizedBufferAllocator> m_allocator;

  // Storage for all unique BindingNodes in the graph. `unique_ptr` ensures
  // proper ownership.
  std::vector<std::unique_ptr<BindingNode>> m_all_binding_nodes;

  // A lookup map for quick access to existing BindingNodes, keyed by a tuple of
  // (resource, offset, size). This avoids redundant node creation.
  std::map<std::tuple<IResourceWrapper*, UINT64, UINT64>, BindingNode*>
      m_binding_lookup;

  // An ordered list of all recorded operator nodes, representing the intended
  // execution sequence.
  std::vector<std::unique_ptr<OperatorNode>> m_operator_nodes;

  // The current sequence of operator nodes being recorded. This list is flushed
  // into a subgraph when `Flush()` is called.
  std::vector<OperatorNode*> m_current_operator_nodes;
  // A list of all binding nodes that are inputs to the entire recorded graph.
  std::vector<BindingNode*> m_graph_inputs;

  // A list of all binding nodes that are outputs of the entire recorded graph.
  // This is populated during the `End()` call.
  std::vector<BindingNode*> m_graph_outputs;
  // A list of subgraphs created from the recorded operators. Each subgraph
  // contains a set of operator nodes and their inputs.
  std::vector<SubGraph> m_subgraphs;

  // Tracks the current binding for each D3D12 resource to detect redundant
  // bindings or potential conflicts within the current recording scope.
  std::unordered_map<IResourceWrapper*, BindingNode*>
      m_current_resource_bindings;

  // A placeholder binding node used for operators that have no inputs or
  // outputs.
  BindingNode m_empty_binding_node;

  // A flag indicating whether a recording session is active.
  bool m_has_begun = false;
};

}  // namespace dml
}  // namespace ctranslate2