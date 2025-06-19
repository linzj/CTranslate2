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

class Device;  // Forward declaration for Device
class BucketizedBufferAllocator;

// Represents a unique DML resource binding within the graph. Each binding is
// identified by the combination of a D3D12 resource, an offset, and a size.
// This struct tracks the binding's properties and its role within the larger
// graph.
struct BindingNode {
  ID3D12Resource* resource;
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

  // A debugging utility to execute the recorded operators sequentially without
  // building a fused DML graph. This is useful for isolating issues and
  // verifying operator behavior.
  void EvaluateGraphWithoutFusedGraph();

  // Retrieves an existing BindingNode for a given resource or creates a new
  // one. This ensures that each unique resource binding is represented by a
  // single node.
  BindingNode* GetOrCreateBindingNode(ID3D12Resource* resource,
                                      UINT64 offset,
                                      UINT64 size,
                                      bool& created);

  std::unique_ptr<BucketizedBufferAllocator> m_allocator;

  // Storage for all unique BindingNodes in the graph. `unique_ptr` ensures
  // proper ownership.
  std::vector<std::unique_ptr<BindingNode>> m_all_binding_nodes;

  // A lookup map for quick access to existing BindingNodes, keyed by a tuple of
  // (resource, offset, size). This avoids redundant node creation.
  std::map<std::tuple<ID3D12Resource*, UINT64, UINT64>, BindingNode*>
      m_binding_lookup;

  // An ordered list of all recorded operator nodes, representing the intended
  // execution sequence.
  std::vector<std::unique_ptr<OperatorNode>> m_operator_nodes;

  // A list of all binding nodes that are inputs to the entire recorded graph.
  std::vector<BindingNode*> m_graph_inputs;

  // A list of all binding nodes that are outputs of the entire recorded graph.
  // This is populated during the `End()` call.
  std::vector<BindingNode*> m_graph_outputs;

  // A placeholder binding node used for operators that have no inputs or
  // outputs.
  BindingNode m_empty_binding_node;

  // A flag indicating whether a recording session is active.
  bool m_has_begun = false;
};

}  // namespace dml
}  // namespace ctranslate2