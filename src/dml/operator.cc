#include "operator.h"

#include "backend_dml.h"
#include "dml_utils.h"
#include "dxdevice.h"
#include "graph_recorder.h"
#include "operator_utils.h"

#include <spdlog/spdlog.h>
namespace ctranslate2 {
namespace dml {

namespace {
constexpr const bool kEnableTracing = false;

void TraceExecute(DML_OPERATOR_TYPE op_type,
                  const std::vector<DML_BINDING_DESC>& inputBindings,
                  const std::vector<DML_BINDING_DESC>& outputBindings) {
  if (!kEnableTracing)
    return;
  SPDLOG_ERROR("Executing DML Operator: {}",
               OperatorUtils::DML_OPERATOR_TYPE_toString(op_type));

  for (size_t i = 0; i < inputBindings.size(); ++i) {
    const auto& binding = inputBindings[i];
    if (binding.Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(binding.Desc);
      if (buffer_binding) {
        SPDLOG_ERROR("  Input[{}]: Buffer={{Resource={}, Offset={}, Size={}}}",
                     i, static_cast<void*>(buffer_binding->Buffer),
                     buffer_binding->Offset, buffer_binding->SizeInBytes);
      } else {
        SPDLOG_ERROR("  Input[{}]: Buffer=<null>", i);
      }
    } else if (binding.Type == DML_BINDING_TYPE_NONE) {
      SPDLOG_ERROR("  Input[{}]: None", i);
    } else {
      SPDLOG_ERROR("  Input[{}]: Unknown binding type", i);
    }
  }

  for (size_t i = 0; i < outputBindings.size(); ++i) {
    const auto& binding = outputBindings[i];
    if (binding.Type == DML_BINDING_TYPE_BUFFER) {
      auto buffer_binding =
          static_cast<const DML_BUFFER_BINDING*>(binding.Desc);
      if (buffer_binding) {
        SPDLOG_ERROR("  Output[{}]: Buffer={{Resource={}, Offset={}, Size={}}}",
                     i, static_cast<void*>(buffer_binding->Buffer),
                     buffer_binding->Offset, buffer_binding->SizeInBytes);
      } else {
        SPDLOG_ERROR("  Output[{}]: Buffer=<null>", i);
      }
    } else if (binding.Type == DML_BINDING_TYPE_NONE) {
      SPDLOG_ERROR("  Output[{}]: None", i);
    } else {
      SPDLOG_ERROR("  Output[{}]: Unknown binding type", i);
    }
  }
}

void TraceExecute(DML_OPERATOR_TYPE op_type,
                  const std::vector<ID3D12Resource*>& input_resources,
                  const std::vector<ID3D12Resource*>& output_resources) {
  if (!kEnableTracing)
    return;
  SPDLOG_ERROR("Executing DML Operator: {}",
               OperatorUtils::DML_OPERATOR_TYPE_toString(op_type));

  for (size_t i = 0; i < input_resources.size(); ++i) {
    SPDLOG_ERROR("  Input[{}]: Resource={}", i,
                 static_cast<void*>(input_resources[i]));
  }
  for (size_t i = 0; i < output_resources.size(); ++i) {
    SPDLOG_ERROR("  Output[{}]: Resource={}", i,
                 static_cast<void*>(output_resources[i]));
  }
}
}  // namespace

using Microsoft::WRL::ComPtr;

Operator::Operator(Device* device,
                   utils::DmlOperatorDescBundle&& op_desc,
                   ComPtr<IDMLOperator>&& dml_operator,
                   ComPtr<IDMLCompiledOperator>&& compiled_operator,
                   const std::string& key)
    : m_operator(std::move(dml_operator)),
      m_compiledOperator(std::move(compiled_operator)),
      m_op_desc(std::move(op_desc)),
      m_key(key) {
  UINT64 persistentResourceSize =
      m_compiledOperator->GetBindingProperties().PersistentResourceSize;
  if (persistentResourceSize > 0) {
    m_persistentResource =
        device->CreatePreferredDeviceMemoryBufferWithoutPooling(
            persistentResourceSize);

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

void Operator::Execute(const utils::DmlBindingArrayBundle& inputs,
                       const utils::DmlBindingArrayBundle& outputs) {
  auto device = dml::get_device();
  GraphRecorder* graph_recorder = device->GetGraphRecorder();
  if (graph_recorder && graph_recorder->has_begun()) {
    graph_recorder->Execute(this, inputs, outputs);
    return;
  }
  auto inputBindings = inputs.get_descs();
  auto outputBindings = outputs.get_descs();

  TraceExecute(GetType(), inputBindings, outputBindings);
  DML_BINDING_DESC persistentBindingDesc{};
  if (m_persistentResourceBinding) {
    persistentBindingDesc.Type = DML_BINDING_TYPE_BUFFER;
    persistentBindingDesc.Desc = &*m_persistentResourceBinding;
  }
  device->ExecuteOperator(m_compiledOperator.Get(), persistentBindingDesc,
                          std::move(inputBindings), std::move(outputBindings));
}

}  // namespace dml
}  // namespace ctranslate2