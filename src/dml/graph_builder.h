#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dxmodule.h"

struct IDMLCompiledOperator;

namespace ctranslate2 {
namespace dml {

struct OperatorNode;
struct BindingNode;
struct OutputEdge;

class GraphBuilder {
 public:
  // Builds a DML_GRAPH_DESC and compiles it into an IDMLCompiledOperator.
  static Microsoft::WRL::ComPtr<IDMLCompiledOperator> Build(
      const std::vector<OperatorNode*>& operator_nodes,
      const std::vector<BindingNode*>& graph_inputs,
      const std::vector<OutputEdge>& graph_outputs,
      DML_EXECUTION_FLAGS flags,
      const std::string& key);
};

}  // namespace dml
}  // namespace ctranslate2