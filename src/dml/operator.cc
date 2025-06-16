#include "operator.h"

#include "backend_dml.h"
#include "dml_utils.h"
#include "dxdevice.h"
#include "graph_recorder.h"

#include <spdlog/spdlog.h>
namespace ctranslate2 {
namespace dml {

namespace {
constexpr const bool kEnableTracing = false;
const char* DML_OPERATOR_TYPE_toString(DML_OPERATOR_TYPE type) {
  switch (type) {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
      return "DML_OPERATOR_ELEMENT_WISE_IDENTITY";
    case DML_OPERATOR_GEMM:
      return "DML_OPERATOR_GEMM";
    case DML_OPERATOR_REDUCE:
      return "DML_OPERATOR_REDUCE";
    case DML_OPERATOR_ELEMENT_WISE_ADD:
      return "DML_OPERATOR_ELEMENT_WISE_ADD";
    case DML_OPERATOR_FILL_VALUE_CONSTANT:
      return "DML_OPERATOR_FILL_VALUE_CONSTANT";
    case DML_OPERATOR_CAST:
      return "DML_OPERATOR_CAST";
    case DML_OPERATOR_ACTIVATION_RELU:
      return "DML_OPERATOR_ACTIVATION_RELU";
    case DML_OPERATOR_ACTIVATION_SIGMOID:
      return "DML_OPERATOR_ACTIVATION_SIGMOID";
    case DML_OPERATOR_ACTIVATION_TANH:
      return "DML_OPERATOR_ACTIVATION_TANH";
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT:
      return "DML_OPERATOR_ELEMENT_WISE_SUBTRACT";
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY:
      return "DML_OPERATOR_ELEMENT_WISE_MULTIPLY";
    case DML_OPERATOR_ELEMENT_WISE_MAX:
      return "DML_OPERATOR_ELEMENT_WISE_MAX";
    case DML_OPERATOR_ELEMENT_WISE_MIN:
      return "DML_OPERATOR_ELEMENT_WISE_MIN";
    case DML_OPERATOR_ELEMENT_WISE_EXP:
      return "DML_OPERATOR_ELEMENT_WISE_EXP";
    case DML_OPERATOR_ELEMENT_WISE_LOG:
      return "DML_OPERATOR_ELEMENT_WISE_LOG";
    case DML_OPERATOR_ELEMENT_WISE_SIN:
      return "DML_OPERATOR_ELEMENT_WISE_SIN";
    case DML_OPERATOR_ELEMENT_WISE_COS:
      return "DML_OPERATOR_ELEMENT_WISE_COS";
    case DML_OPERATOR_ARGMAX:
      return "DML_OPERATOR_ARGMAX";
    case DML_OPERATOR_ELEMENT_WISE_ADD1:
      return "DML_OPERATOR_ELEMENT_WISE_ADD1";
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE:
      return "DML_OPERATOR_ELEMENT_WISE_DIVIDE";
    case DML_OPERATOR_ELEMENT_WISE_NEGATE:
      return "DML_OPERATOR_ELEMENT_WISE_NEGATE";
    case DML_OPERATOR_ELEMENT_WISE_ABS:
      return "DML_OPERATOR_ELEMENT_WISE_ABS";
    case DML_OPERATOR_ELEMENT_WISE_ROUND:
      return "DML_OPERATOR_ELEMENT_WISE_ROUND";
    case DML_OPERATOR_CONVOLUTION:
      return "DML_OPERATOR_CONVOLUTION";
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2:
      return "DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2";
    case DML_OPERATOR_JOIN:
      return "DML_OPERATOR_JOIN";
    case DML_OPERATOR_SPLIT:
      return "DML_OPERATOR_SPLIT";
    case DML_OPERATOR_SLICE:
      return "DML_OPERATOR_SLICE";
    case DML_OPERATOR_SLICE1:
      return "DML_OPERATOR_SLICE1";
    case DML_OPERATOR_TILE:
      return "DML_OPERATOR_TILE";
    case DML_OPERATOR_GATHER:
      return "DML_OPERATOR_GATHER";
    case DML_OPERATOR_GATHER_ELEMENTS:
      return "DML_OPERATOR_GATHER_ELEMENTS";
    case DML_OPERATOR_SCATTER_ELEMENTS:
      return "DML_OPERATOR_SCATTER_ELEMENTS";
    case DML_OPERATOR_TOP_K1:
      return "DML_OPERATOR_TOP_K1";
    case DML_OPERATOR_ACTIVATION_GELU:
      return "DML_OPERATOR_ACTIVATION_GELU";
    case DML_OPERATOR_ACTIVATION_SWISH:
      return "DML_OPERATOR_ACTIVATION_SWISH";
    case DML_OPERATOR_ACTIVATION_LINEAR:
      return "DML_OPERATOR_ACTIVATION_LINEAR";
    case DML_OPERATOR_ACTIVATION_IDENTITY:
      return "DML_OPERATOR_ACTIVATION_IDENTITY";
    case DML_OPERATOR_ACTIVATION_SOFTMAX:
      return "DML_OPERATOR_ACTIVATION_SOFTMAX";
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX:
      return "DML_OPERATOR_ACTIVATION_LOG_SOFTMAX";
    case DML_OPERATOR_ACTIVATION_SOFTMAX1:
      return "DML_OPERATOR_ACTIVATION_SOFTMAX1";
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1:
      return "DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1";
    case DML_OPERATOR_RANDOM_GENERATOR:
      return "DML_OPERATOR_RANDOM_GENERATOR";
    case DML_OPERATOR_FILL_VALUE_SEQUENCE:
      return "DML_OPERATOR_FILL_VALUE_SEQUENCE";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL:
      return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN:
      return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN";
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS:
      return "DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS";
    case DML_OPERATOR_ELEMENT_WISE_IF:
      return "DML_OPERATOR_ELEMENT_WISE_IF";
    case DML_OPERATOR_CUMULATIVE_SUMMATION:
      return "DML_OPERATOR_CUMULATIVE_SUMMATION";
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR:
      return "DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR";
    case DML_OPERATOR_ELEMENT_WISE_RECIP:
      return "DML_OPERATOR_ELEMENT_WISE_RECIP";
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR:
      return "DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR";
    case DML_OPERATOR_CONVOLUTION_INTEGER:
      return "DML_OPERATOR_CONVOLUTION_INTEGER";
    case DML_OPERATOR_MATRIX_MULTIPLY_INTEGER:
      return "DML_OPERATOR_MATRIX_MULTIPLY_INTEGER";
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR:
      return "DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR";
    default:
      return "Unknown DML Operator Type";
  }
}

void TraceExecute(DML_OPERATOR_TYPE op_type,
                  const std::vector<DML_BINDING_DESC>& inputBindings,
                  const std::vector<DML_BINDING_DESC>& outputBindings) {
  if (!kEnableTracing)
    return;
  SPDLOG_ERROR("Executing DML Operator: {}",
               DML_OPERATOR_TYPE_toString(op_type));

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
               DML_OPERATOR_TYPE_toString(op_type));

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
                   ComPtr<IDMLCompiledOperator>&& compiled_operator)
    : m_compiledOperator(std::move(compiled_operator)),
      m_op_desc(std::move(op_desc)) {
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

void Operator::Execute(const std::vector<ID3D12Resource*>& input_resources,
                       const std::vector<ID3D12Resource*>& output_resources) {
  dml::utils::DmlBindingArrayBundle inputs{input_resources};
  dml::utils::DmlBindingArrayBundle outputs{output_resources};
  Execute(inputs, outputs);
}
}  // namespace dml
}  // namespace ctranslate2