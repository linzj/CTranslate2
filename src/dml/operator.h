#pragma once
#include <vector>
#include "dml_utils.h"
#include "dxmodule.h"

#include <wrl/implements.h>

namespace ctranslate2 {
namespace dml {
using Microsoft::WRL::ComPtr;
class Device;

class Operator
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
          IUnknown> {
 public:
  explicit Operator(Device* device,
                    utils::DmlOperatorDescBundle&& op_desc,
                    ComPtr<IDMLCompiledOperator>&& compiled_operator);
  ~Operator();

  void Execute(std::vector<DML_BINDING_DESC> inputBindings,
               std::vector<DML_BINDING_DESC> outputBindings);

  void Execute(const std::vector<ID3D12Resource*>& input_resources,
               const std::vector<ID3D12Resource*>& output_resources);

  DML_OPERATOR_TYPE GetType() const { return m_op_desc.get_desc().Type; }

 private:
  ComPtr<IDMLCompiledOperator> m_compiledOperator;
  ComPtr<ID3D12Resource> m_persistentResource;
  std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
  utils::DmlOperatorDescBundle m_op_desc;
};
}  // namespace dml
}  // namespace ctranslate2