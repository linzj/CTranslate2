#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/bias_add.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Helper function to convert ctranslate2 DataType to DML_TENSOR_DATA_TYPE
template <typename T>
DML_TENSOR_DATA_TYPE get_dml_data_type() {
  if (std::is_same_v<T, float>) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (std::is_same_v<T, float16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  } else {
    throw std::runtime_error("Unsupported data type for DirectML BiasAdd");
  }
}

template <Device D, typename T>
void BiasAdd::compute(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output) const {
  auto* dml_device = dml::get_dml_device();
  auto* device = dml::get_device();
  std::unique_ptr<DML_OPERATOR_DESC> fused_activation;

  const DML_TENSOR_DATA_TYPE dml_data_type = get_dml_data_type<T>();

  // Get dimensions
  const dim_t depth = bias.size();
  const dim_t batch_size = value.size() / depth;

  // Create value tensor descriptor [batch_size, depth]
  std::vector<UINT> value_sizes = {static_cast<UINT>(batch_size),
                                   static_cast<UINT>(depth)};

  DML_BUFFER_TENSOR_DESC value_desc = {};
  value_desc.DataType = dml_data_type;
  value_desc.Flags = DML_TENSOR_FLAG_NONE;
  value_desc.DimensionCount = static_cast<UINT>(value_sizes.size());
  value_desc.Sizes = value_sizes.data();
  value_desc.Strides = nullptr;  // Use default strides
  value_desc.TotalTensorSizeInBytes = value.size() * sizeof(T);
  value_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC value_tensor = {};
  value_tensor.Type = DML_TENSOR_TYPE_BUFFER;
  value_tensor.Desc = &value_desc;

  // Create bias tensor descriptor [1, depth] (will be broadcasted)
  std::vector<UINT> bias_sizes = {1, static_cast<UINT>(depth)};

  DML_BUFFER_TENSOR_DESC bias_desc = {};
  bias_desc.DataType = dml_data_type;
  bias_desc.Flags = DML_TENSOR_FLAG_NONE;
  bias_desc.DimensionCount = static_cast<UINT>(bias_sizes.size());
  bias_desc.Sizes = bias_sizes.data();
  bias_desc.Strides = nullptr;  // Use default strides
  bias_desc.TotalTensorSizeInBytes = bias.size() * sizeof(T);
  bias_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC bias_tensor = {};
  bias_tensor.Type = DML_TENSOR_TYPE_BUFFER;
  bias_tensor.Desc = &bias_desc;

  // Create output tensor descriptor [batch_size, depth]
  std::vector<UINT> output_sizes = {static_cast<UINT>(batch_size),
                                    static_cast<UINT>(depth)};

  DML_BUFFER_TENSOR_DESC output_desc = {};
  output_desc.DataType = dml_data_type;
  output_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_desc.DimensionCount = static_cast<UINT>(output_sizes.size());
  output_desc.Sizes = output_sizes.data();
  output_desc.Strides = nullptr;  // Use default strides
  output_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);
  output_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC output_tensor = {};
  output_tensor.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor.Desc = &output_desc;

  dml::Operator* compiled_op;

  if (!_activation_type) {
    // Simple addition without activation
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
    add_desc.ATensor = &value_tensor;
    add_desc.BTensor = &bias_tensor;
    add_desc.OutputTensor = &output_tensor;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
    op_desc.Desc = &add_desc;

    compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
  } else {
    // Use ADD1 operator with fused activation for supported types
    // Create activation operator description
    fused_activation = std::make_unique<DML_OPERATOR_DESC>();
    std::unique_ptr<void, void (*)(void*)> activation_desc_storage(
        nullptr, [](void* p) {});

    switch (*_activation_type) {
      case ActivationType::ReLU: {
        auto relu_desc = std::make_unique<DML_ACTIVATION_RELU_OPERATOR_DESC>();
        relu_desc->InputTensor = &output_tensor;
        relu_desc->OutputTensor = &output_tensor;
        fused_activation->Type = DML_OPERATOR_ACTIVATION_RELU;
        fused_activation->Desc = relu_desc.get();
        activation_desc_storage.reset(relu_desc.release());
        activation_desc_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_RELU_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Sigmoid: {
        auto sigmoid_desc =
            std::make_unique<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
        sigmoid_desc->InputTensor = &output_tensor;
        sigmoid_desc->OutputTensor = &output_tensor;
        fused_activation->Type = DML_OPERATOR_ACTIVATION_SIGMOID;
        fused_activation->Desc = sigmoid_desc.get();
        activation_desc_storage.reset(sigmoid_desc.release());
        activation_desc_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_SIGMOID_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Tanh: {
        auto tanh_desc = std::make_unique<DML_ACTIVATION_TANH_OPERATOR_DESC>();
        tanh_desc->InputTensor = &output_tensor;
        tanh_desc->OutputTensor = &output_tensor;
        fused_activation->Type = DML_OPERATOR_ACTIVATION_TANH;
        fused_activation->Desc = tanh_desc.get();
        activation_desc_storage.reset(tanh_desc.release());
        activation_desc_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_TANH_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::GELU: {
        auto gelu_desc = std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
        gelu_desc->InputTensor = &output_tensor;
        gelu_desc->OutputTensor = &output_tensor;
        fused_activation->Type = DML_OPERATOR_ACTIVATION_GELU;
        fused_activation->Desc = gelu_desc.get();
        activation_desc_storage.reset(gelu_desc.release());
        activation_desc_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_GELU_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Swish: {
        auto swish_desc =
            std::make_unique<DML_ACTIVATION_SWISH_OPERATOR_DESC>();
        swish_desc->InputTensor = &output_tensor;
        swish_desc->OutputTensor = &output_tensor;
        swish_desc->SigmoidInputScale = 1.0f;
        fused_activation->Type = DML_OPERATOR_ACTIVATION_SWISH;
        fused_activation->Desc = swish_desc.get();
        activation_desc_storage.reset(swish_desc.release());
        activation_desc_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_SWISH_OPERATOR_DESC*>(p);
        };
        break;
      }
      default:
        // For unsupported fused activations, fall back to separate operations
        fused_activation.reset();
        break;
    }

    if (fused_activation) {
      // Use ADD1 operator with fused activation
      DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add_desc = {};
      add_desc.ATensor = &value_tensor;
      add_desc.BTensor = &bias_tensor;
      add_desc.OutputTensor = &output_tensor;
      add_desc.FusedActivation = fused_activation.get();

      DML_OPERATOR_DESC op_desc = {};
      op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD1;
      op_desc.Desc = &add_desc;

      compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    } else {
      // Fall back to separate add and activation operations
      DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
      add_desc.ATensor = &value_tensor;
      add_desc.BTensor = &bias_tensor;
      add_desc.OutputTensor = &output_tensor;

      DML_OPERATOR_DESC add_op_desc = {};
      add_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
      add_op_desc.Desc = &add_desc;

      auto add_compiled_op = dml::GetOrCreateCompiledOperatorApi(&add_op_desc);

      // Bind inputs for add
      DML_BUFFER_BINDING value_binding = {};
      value_binding.Buffer =
          static_cast<ID3D12Resource*>(const_cast<void*>(value.buffer()));
      value_binding.Offset = 0;
      value_binding.SizeInBytes = value.size() * sizeof(T);

      DML_BUFFER_BINDING bias_binding = {};
      bias_binding.Buffer =
          static_cast<ID3D12Resource*>(const_cast<void*>(bias.buffer()));
      bias_binding.Offset = 0;
      bias_binding.SizeInBytes = bias.size() * sizeof(T);

      std::vector<DML_BINDING_DESC> add_input_bindings = {
          {DML_BINDING_TYPE_BUFFER, &value_binding},
          {DML_BINDING_TYPE_BUFFER, &bias_binding}};

      // Bind output for add
      DML_BUFFER_BINDING output_binding = {};
      output_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
      output_binding.Offset = 0;
      output_binding.SizeInBytes = output.size() * sizeof(T);

      DML_BINDING_DESC add_output_binding_desc = {};
      add_output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
      add_output_binding_desc.Desc = &output_binding;

      // Execute add
      add_compiled_op->Execute(add_input_bindings, {add_output_binding_desc});

      // Now create and execute activation operator
      DML_OPERATOR_DESC activation_op_desc = {};
      switch (*_activation_type) {
        case ActivationType::GELUTanh: {
          // Use tanh-based GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) *
          // (x + 0.044715 * x^3))) For now, fall back to basic GELU
          DML_ACTIVATION_GELU_OPERATOR_DESC gelu_desc = {};
          gelu_desc.InputTensor = &output_tensor;
          gelu_desc.OutputTensor = &output_tensor;
          activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
          activation_op_desc.Desc = &gelu_desc;
          break;
        }
        case ActivationType::GELUSigmoid: {
          // For now, fall back to basic GELU
          DML_ACTIVATION_GELU_OPERATOR_DESC gelu_desc = {};
          gelu_desc.InputTensor = &output_tensor;
          gelu_desc.OutputTensor = &output_tensor;
          activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
          activation_op_desc.Desc = &gelu_desc;
          break;
        }
        default:
          throw std::runtime_error(
              "Unsupported activation type for DirectML BiasAdd");
      }

      compiled_op = dml::GetOrCreateCompiledOperatorApi(&activation_op_desc);
    }
  }

  std::vector<DML_BINDING_DESC> input_bindings;
  if (!_activation_type || fused_activation) {
    // Bind inputs for add operation (or fused add+activation)
    DML_BUFFER_BINDING value_binding = {};
    value_binding.Buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(value.buffer()));
    value_binding.Offset = 0;
    value_binding.SizeInBytes = value.size() * sizeof(T);

    DML_BUFFER_BINDING bias_binding = {};
    bias_binding.Buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias.buffer()));
    bias_binding.Offset = 0;
    bias_binding.SizeInBytes = bias.size() * sizeof(T);

    input_bindings.emplace_back(DML_BINDING_DESC{
        .Type = DML_BINDING_TYPE_BUFFER, .Desc = &value_binding});
    input_bindings.emplace_back(DML_BINDING_DESC{
        .Type = DML_BINDING_TYPE_BUFFER, .Desc = &bias_binding});
  } else {
    // Bind input for activation operation (output from previous add is now
    // input)
    DML_BUFFER_BINDING input_binding = {};
    input_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
    input_binding.Offset = 0;
    input_binding.SizeInBytes = output.size() * sizeof(T);

    DML_BINDING_DESC input_binding_desc = {};
    input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
    input_binding_desc.Desc = &input_binding;
    input_bindings.push_back(input_binding_desc);
  }

  // Bind output
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  compiled_op->Execute(input_bindings, {output_binding_desc});
}

#define DECLARE_IMPL(T)                                                       \
  template void BiasAdd::compute<Device::DirectML, T>(                        \
      const StorageView& value, const StorageView& bias, StorageView& output) \
      const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif