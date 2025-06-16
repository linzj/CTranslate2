#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "dml/dml_utils.h"
#include "dml/operator.h"

namespace ctranslate2 {
namespace dml {

class Device;  // Forward declaration for Device

// Represents a unique DML resource binding within the graph.
// Identified by resource pointer, offset, and size.
struct BindingNode {
  ID3D12Resource* resource;
  UINT64 offset;
  UINT64 size_in_bytes;

  // True if this binding is an input to the overall recorded graph.
  bool is_graph_input = false;
  // True if this binding is an output of the overall recorded graph (currently
  // not used to mark intermediate nodes). It will be updated by End method to
  // indicate graph outputs
  bool is_graph_output = false;
};

// Represents an operator within the recorded graph.
struct OperatorNode {
  Operator* op;                       // The DML operator
  std::vector<BindingNode*> inputs;   // Pointers to input BindingNodes
  std::vector<BindingNode*> outputs;  // Pointers to output BindingNodes
};

class GraphRecorder {
 public:
  explicit GraphRecorder();
  ~GraphRecorder();

  // Resets and initializes the graph recorder's states.
  void Begin();

  // Records a new operator node and its binding relationships into the graph.
  // Finds or creates binding nodes based on resource, offset, and size.
  void Execute(Operator* op,
               const utils::DmlBindingArrayBundle& inputs,
               const utils::DmlBindingArrayBundle& outputs);

  // Executes the recorded graph in the order of operator nodes.
  void End();

  bool has_begun() const { return m_has_begun; }

 private:
  // Storage for all unique BindingNodes in the graph. Using unique_ptr for
  // ownership.
  std::vector<std::unique_ptr<BindingNode>> m_all_binding_nodes;

  // Lookup map to quickly find an existing BindingNode. Key is (resource,
  // offset, size).
  std::map<std::tuple<ID3D12Resource*, UINT64, UINT64>, BindingNode*>
      m_binding_lookup;

  // Ordered list of operator nodes, representing the execution sequence.
  std::vector<std::unique_ptr<OperatorNode>> m_operator_nodes;

  // Inputs to the entire recorded graph.
  std::vector<BindingNode*> m_graph_inputs;

  // Outputs of the entire recorded graph (updated during End call).
  std::vector<BindingNode*> m_graph_outputs;

  BindingNode m_empty_binding_node;
  bool m_has_begun = false;  // Indicates if Begin() has been called

  // Helper to get or create a BindingNode.
  BindingNode* GetOrCreateBindingNode(ID3D12Resource* resource,
                                      UINT64 offset,
                                      UINT64 size,
                                      bool& created);
};

}  // namespace dml
}  // namespace ctranslate2