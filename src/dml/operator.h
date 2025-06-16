#pragma once
#include <vector>
#include "dml_utils.h"

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
                    ComPtr<IDMLCompiledOperator>&& compiled_operator,
                  const std::string& key);
  ~Operator();

  void Execute(const utils::DmlBindingArrayBundle& inputs,
               const utils::DmlBindingArrayBundle& outputs);

  void Execute(const std::vector<ID3D12Resource*>& input_resources,
               const std::vector<ID3D12Resource*>& output_resources);

  DML_OPERATOR_TYPE GetType() const { return m_op_desc.get_desc().Type; }

  const std::string& key() const { return m_key; }
 private:
  ComPtr<IDMLCompiledOperator> m_compiledOperator;
  ComPtr<ID3D12Resource> m_persistentResource;
  std::optional<DML_BUFFER_BINDING> m_persistentResourceBinding;
  utils::DmlOperatorDescBundle m_op_desc;
  std::string m_key;
};
}  // namespace dml
}  // namespace ctranslate2