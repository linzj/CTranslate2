#include "operator.h"
#include "backend_dml.h"
#include "dxdevice.h"

namespace ctranslate2 {
namespace dml {

using Microsoft::WRL::ComPtr;

Operator::Operator(Device* device,
                   ComPtr<IDMLCompiledOperator>&& compiled_operator)
    : m_compiledOperator(std::move(compiled_operator)) {
  UINT64 persistentResourceSize =
      m_compiledOperator->GetBindingProperties().PersistentResourceSize;
  if (persistentResourceSize > 0) {
    m_persistentResource =
        device->CreatePreferredDeviceMemoryBuffer(persistentResourceSize);

    m_persistentResourceBinding = DML_BUFFER_BINDING{m_persistentResource.Get(),
                                                     0, persistentResourceSize};
  }

  DML_BINDING_DESC initializationInputBindings{};

  DML_BINDING_DESC persistentResourceBindingDesc = {};
  if (m_persistentResourceBinding) {
    persistentResourceBindingDesc.Type = DML_BINDING_TYPE_BUFFER;
    persistentResourceBindingDesc.Desc = &*m_persistentResourceBinding;
  }
  device->InitializeOperator(m_compiledOperator.Get(),
                             persistentResourceBindingDesc,
                             initializationInputBindings);
}

Operator::~Operator() = default;

void Operator::Execute(std::vector<DML_BINDING_DESC> inputBindings,
                       std::vector<DML_BINDING_DESC> outputBindings) {
  auto device = dml::get_device();
  DML_BINDING_DESC persistentBindingDesc{};
  if (m_persistentResourceBinding) {
    persistentBindingDesc.Type = DML_BINDING_TYPE_BUFFER;
    persistentBindingDesc.Desc = &*m_persistentResourceBinding;
  }
  device->ExecuteOperator(m_compiledOperator.Get(), persistentBindingDesc,
                          std::move(inputBindings), std::move(outputBindings));
}

void Operator::Execute(const std::vector<ID3D12Resource*>& input_resources,
                       const std::vector<ID3D12Resource*>& output_resources) {
  auto device = dml::get_device();
  device->ExecuteOperator(m_compiledOperator.Get(), input_resources,
                          output_resources, m_persistentResource.Get());
}
}  // namespace dml
}  // namespace ctranslate2