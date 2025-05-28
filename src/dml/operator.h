#pragma once
#include <vector>
#include "dxmodule.h"

namespace ctranslate2 {
namespace dml {
using Microsoft::WRL::ComPtr;
class Device;

class Operator {
 public:
  explicit Operator(Device* device,
                    ComPtr<IDMLCompiledOperator>&& compiled_operator);
  ~Operator();

  void Execute(std::vector<DML_BINDING_DESC> inputBindings,
               std::vector<DML_BINDING_DESC> outputBindings);

  void Execute(const std::vector<ID3D12Resource*>& input_resources,
               const std::vector<ID3D12Resource*>& output_resources);

 private:
  ComPtr<IDMLCompiledOperator> m_compiledOperator;
  ComPtr<ID3D12Resource> m_persistentResource;
  std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
};
}  // namespace dml
}  // namespace ctranslate2