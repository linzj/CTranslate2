#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/dequantize.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <>
void Dequantize::dequantize<Device::DirectML, int8_t, float>(
    const StorageView& input,
    const StorageView& scale,
    StorageView& output) const {
  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);

  // Step 1: Cast int8 input to float
  StorageView float_input(input.shape(), DataType::FLOAT32, Device::DirectML);

  // Create cast operation
  std::vector<UINT> input_sizes(input.rank());
  for (dim_t i = 0; i < input.rank(); ++i) {
    input_sizes[i] = static_cast<UINT>(input.shape()[i]);
  }

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_INT8,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(input.rank()),
      .Sizes = input_sizes.data(),
      .Strides = nullptr,
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(input.size() * sizeof(int8_t)),
      .GuaranteedBaseOffsetAlignment = 0};

  DML_BUFFER_TENSOR_DESC float_output_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_FLOAT32,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(input.rank()),
      .Sizes = input_sizes.data(),
      .Strides = nullptr,
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(input.size() * sizeof(float)),
      .GuaranteedBaseOffsetAlignment = 0};

  DML_TENSOR_DESC input_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                .Desc = &input_buffer_desc};

  DML_TENSOR_DESC cast_output_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                      .Desc = &float_output_desc};

  DML_CAST_OPERATOR_DESC cast_desc = {.InputTensor = &input_desc,
                                      .OutputTensor = &cast_output_desc};

  DML_OPERATOR_DESC cast_op_desc = {.Type = DML_OPERATOR_CAST,
                                    .Desc = &cast_desc};

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Bind cast inputs and outputs
  DML_BUFFER_BINDING cast_input_binding = {
      .Buffer = static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
      .Offset = 0,
      .SizeInBytes = static_cast<UINT64>(input.size() * sizeof(int8_t))};

  DML_BUFFER_BINDING cast_output_binding = {
      .Buffer = static_cast<ID3D12Resource*>(float_input.buffer()),
      .Offset = 0,
      .SizeInBytes = static_cast<UINT64>(input.size() * sizeof(float))};

  DML_BINDING_DESC cast_input_bind = {.Type = DML_BINDING_TYPE_BUFFER,
                                      .Desc = &cast_input_binding};

  DML_BINDING_DESC cast_output_bind = {.Type = DML_BINDING_TYPE_BUFFER,
                                       .Desc = &cast_output_binding};

  cast_compiled_op->Execute({cast_input_bind}, {cast_output_bind});

  // Step 2: Element-wise division (float_input / scale)

  // Handle scale broadcasting - scale is typically per-channel (depth
  // dimension)
  std::vector<UINT> scale_sizes;
  std::vector<UINT> scale_strides;

  if (scale.size() == depth) {
    // Scale has same number of elements as the last dimension
    // Broadcast scale to match input shape
    scale_sizes = input_sizes;
    scale_strides.resize(input.rank());

    // Set strides for broadcasting - only the last dimension has stride 1,
    // others 0
    for (dim_t i = 0; i < input.rank() - 1; ++i) {
      scale_strides[i] = 0;  // Broadcast these dimensions
    }
    scale_strides[input.rank() - 1] = 1;  // Normal stride for last dimension
  } else {
    // Scale has same shape as input
    scale_sizes = input_sizes;
    scale_strides.clear();  // Use default strides
  }

  DML_BUFFER_TENSOR_DESC scale_buffer_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_FLOAT32,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(scale_sizes.size()),
      .Sizes = scale_sizes.data(),
      .Strides = scale_strides.empty() ? nullptr : scale_strides.data(),
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(scale.size() * sizeof(float)),
      .GuaranteedBaseOffsetAlignment = 0};

  DML_BUFFER_TENSOR_DESC div_output_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_FLOAT32,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(output.rank()),
      .Sizes = input_sizes.data(),
      .Strides = nullptr,
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(output.size() * sizeof(float)),
      .GuaranteedBaseOffsetAlignment = 0};

  DML_TENSOR_DESC float_input_tensor_desc = {
      .Type = DML_TENSOR_TYPE_BUFFER,
      .Desc = &float_output_desc  // Reuse the desc from cast output
  };

  DML_TENSOR_DESC scale_tensor_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                       .Desc = &scale_buffer_desc};

  DML_TENSOR_DESC div_output_tensor_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                            .Desc = &div_output_desc};

  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC divide_desc = {
      .ATensor = &float_input_tensor_desc,
      .BTensor = &scale_tensor_desc,
      .OutputTensor = &div_output_tensor_desc};

  DML_OPERATOR_DESC divide_op_desc = {.Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE,
                                      .Desc = &divide_desc};

  auto divide_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&divide_op_desc);

  // Bind division inputs and outputs
  DML_BUFFER_BINDING div_input1_binding = {
      .Buffer = static_cast<ID3D12Resource*>(float_input.buffer()),
      .Offset = 0,
      .SizeInBytes = static_cast<UINT64>(input.size() * sizeof(float))};

  DML_BUFFER_BINDING div_input2_binding = {
      .Buffer = static_cast<ID3D12Resource*>(const_cast<void*>(scale.buffer())),
      .Offset = 0,
      .SizeInBytes = static_cast<UINT64>(scale.size() * sizeof(float))};

  DML_BUFFER_BINDING div_output_binding = {
      .Buffer = static_cast<ID3D12Resource*>(output.buffer()),
      .Offset = 0,
      .SizeInBytes = static_cast<UINT64>(output.size() * sizeof(float))};

  std::vector<DML_BINDING_DESC> div_input_bindings = {
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &div_input1_binding},
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &div_input2_binding}};

  DML_BINDING_DESC div_output_bind = {.Type = DML_BINDING_TYPE_BUFFER,
                                      .Desc = &div_output_binding};

  divide_compiled_op->Execute(div_input_bindings, {div_output_bind});
}

template <>
void Dequantize::dequantize_gemm_output<Device::DirectML, float>(
    const StorageView& c,
    const StorageView& a_scale,
    const StorageView& b_scale,
    const bool transpose_a,
    const bool transpose_b,
    const StorageView* bias,
    StorageView& y) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();
  auto* command_list = device->GetCommandList();

  const dim_t batch_size = a_scale.size();
  const dim_t depth = c.dim(-1);

  // Get input buffers
  auto* c_buffer = static_cast<ID3D12Resource*>(const_cast<void*>(c.buffer()));
  auto* a_scale_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(a_scale.buffer()));
  auto* b_scale_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(b_scale.buffer()));
  auto* y_buffer = static_cast<ID3D12Resource*>(y.buffer());

  // Create tensor descriptors
  UINT c_dims[] = {static_cast<UINT>(batch_size), static_cast<UINT>(depth)};
  UINT scale_dims_a[] = {static_cast<UINT>(transpose_a ? depth : batch_size),
                         1};
  UINT scale_dims_b[] = {static_cast<UINT>(transpose_b ? depth : batch_size),
                         1};
  UINT y_dims[] = {static_cast<UINT>(batch_size), static_cast<UINT>(depth)};

  // Input tensor (int32)
  DML_BUFFER_TENSOR_DESC c_tensor_desc = {};
  c_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_INT32;
  c_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  c_tensor_desc.DimensionCount = 2;
  c_tensor_desc.Sizes = c_dims;
  c_tensor_desc.TotalTensorSizeInBytes = c.size() * sizeof(int32_t);
  c_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC c_desc = {};
  c_desc.Type = DML_TENSOR_TYPE_BUFFER;
  c_desc.Desc = &c_tensor_desc;

  // A scale tensor (float)
  DML_BUFFER_TENSOR_DESC a_scale_tensor_desc = {};
  a_scale_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  a_scale_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  a_scale_tensor_desc.DimensionCount = 2;
  a_scale_tensor_desc.Sizes = scale_dims_a;
  a_scale_tensor_desc.TotalTensorSizeInBytes = a_scale.size() * sizeof(float);
  a_scale_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC a_scale_desc = {};
  a_scale_desc.Type = DML_TENSOR_TYPE_BUFFER;
  a_scale_desc.Desc = &a_scale_tensor_desc;

  // B scale tensor (float)
  DML_BUFFER_TENSOR_DESC b_scale_tensor_desc = {};
  b_scale_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  b_scale_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  b_scale_tensor_desc.DimensionCount = 2;
  b_scale_tensor_desc.Sizes = scale_dims_b;
  b_scale_tensor_desc.TotalTensorSizeInBytes = b_scale.size() * sizeof(float);
  b_scale_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC b_scale_desc = {};
  b_scale_desc.Type = DML_TENSOR_TYPE_BUFFER;
  b_scale_desc.Desc = &b_scale_tensor_desc;

  // Output tensor (float)
  DML_BUFFER_TENSOR_DESC y_tensor_desc = {};
  y_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  y_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  y_tensor_desc.DimensionCount = 2;
  y_tensor_desc.Sizes = y_dims;
  y_tensor_desc.TotalTensorSizeInBytes = y.size() * sizeof(float);
  y_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC y_desc = {};
  y_desc.Type = DML_TENSOR_TYPE_BUFFER;
  y_desc.Desc = &y_tensor_desc;

  // Create intermediate tensors for computation
  DML_BUFFER_TENSOR_DESC intermediate_tensor_desc = y_tensor_desc;
  DML_TENSOR_DESC intermediate_desc = {};
  intermediate_desc.Type = DML_TENSOR_TYPE_BUFFER;
  intermediate_desc.Desc = &intermediate_tensor_desc;

  // Step 1: Cast int32 input to float
  DML_CAST_OPERATOR_DESC cast_desc = {};
  cast_desc.InputTensor = &c_desc;
  cast_desc.OutputTensor = &intermediate_desc;

  DML_OPERATOR_DESC cast_op_desc = {};
  cast_op_desc.Type = DML_OPERATOR_CAST;
  cast_op_desc.Desc = &cast_desc;

  auto cast_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Create temporary buffer for cast output
  auto cast_output =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 2: Multiply scales (a_scale * b_scale)
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC scale_mult_desc = {};
  scale_mult_desc.ATensor = &a_scale_desc;
  scale_mult_desc.BTensor = &b_scale_desc;
  scale_mult_desc.OutputTensor = &intermediate_desc;

  DML_OPERATOR_DESC scale_mult_op_desc = {};
  scale_mult_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
  scale_mult_op_desc.Desc = &scale_mult_desc;

  auto scale_mult_op = dml::GetOrCreateCompiledOperatorApi(&scale_mult_op_desc);

  // Create temporary buffer for scale multiplication
  auto combined_scale =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 3: Divide cast output by combined scale
  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC divide_desc = {};
  divide_desc.ATensor = &intermediate_desc;
  divide_desc.BTensor = &intermediate_desc;
  divide_desc.OutputTensor = &intermediate_desc;

  DML_OPERATOR_DESC divide_op_desc = {};
  divide_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE;
  divide_op_desc.Desc = &divide_desc;

  auto divide_op = dml::GetOrCreateCompiledOperatorApi(&divide_op_desc);

  // Create temporary buffer for division result
  auto divide_output =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 4: Add bias if provided
  dml::Operator* bias_add_op = nullptr;
  Microsoft::WRL::ComPtr<ID3D12Resource> bias_output;

  if (bias) {
    auto* bias_buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer()));

    // Bias tensor
    UINT bias_dims[] = {1, static_cast<UINT>(depth)};
    DML_BUFFER_TENSOR_DESC bias_tensor_desc = {};
    bias_tensor_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    bias_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
    bias_tensor_desc.DimensionCount = 2;
    bias_tensor_desc.Sizes = bias_dims;
    bias_tensor_desc.TotalTensorSizeInBytes = bias->size() * sizeof(float);
    bias_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

    DML_TENSOR_DESC bias_desc = {};
    bias_desc.Type = DML_TENSOR_TYPE_BUFFER;
    bias_desc.Desc = &bias_tensor_desc;

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC bias_add_desc = {};
    bias_add_desc.ATensor = &intermediate_desc;
    bias_add_desc.BTensor = &bias_desc;
    bias_add_desc.OutputTensor = &intermediate_desc;

    DML_OPERATOR_DESC bias_add_op_desc = {};
    bias_add_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
    bias_add_op_desc.Desc = &bias_add_desc;

    bias_add_op = dml::GetOrCreateCompiledOperatorApi(&bias_add_op_desc);
    bias_output =
        device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));
  }

  // Step 5: Apply activation if specified
  dml::Operator* activation_op;

  if (_activation_type) {
    DML_OPERATOR_DESC activation_op_desc = {};

    switch (*_activation_type) {
      case ActivationType::ReLU: {
        DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
        relu_desc.InputTensor = &intermediate_desc;
        relu_desc.OutputTensor = &y_desc;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_RELU;
        activation_op_desc.Desc = &relu_desc;
        break;
      }
      case ActivationType::GELU: {
        DML_ACTIVATION_GELU_OPERATOR_DESC gelu_desc = {};
        gelu_desc.InputTensor = &intermediate_desc;
        gelu_desc.OutputTensor = &y_desc;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
        activation_op_desc.Desc = &gelu_desc;
        break;
      }
      case ActivationType::Sigmoid: {
        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid_desc = {};
        sigmoid_desc.InputTensor = &intermediate_desc;
        sigmoid_desc.OutputTensor = &y_desc;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
        activation_op_desc.Desc = &sigmoid_desc;
        break;
      }
      case ActivationType::Tanh: {
        DML_ACTIVATION_TANH_OPERATOR_DESC tanh_desc = {};
        tanh_desc.InputTensor = &intermediate_desc;
        tanh_desc.OutputTensor = &y_desc;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_TANH;
        activation_op_desc.Desc = &tanh_desc;
        break;
      }
      case ActivationType::Swish: {
        DML_ACTIVATION_SWISH_OPERATOR_DESC swish_desc = {};
        swish_desc.InputTensor = &intermediate_desc;
        swish_desc.OutputTensor = &y_desc;
        swish_desc.SigmoidInputScale = 1.0f;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SWISH;
        activation_op_desc.Desc = &swish_desc;
        break;
      }
      default:
        // For unsupported activations, use identity
        DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity_desc = {};
        identity_desc.InputTensor = &intermediate_desc;
        identity_desc.OutputTensor = &y_desc;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_IDENTITY;
        activation_op_desc.Desc = &identity_desc;
        break;
    }

    activation_op = dml::GetOrCreateCompiledOperatorApi(&activation_op_desc);
  }

  // Execute the operators in sequence

  // 1. Cast int32 to float
  {
    DML_BUFFER_BINDING input_binding = {c_buffer, 0,
                                        c.size() * sizeof(int32_t)};
    DML_BUFFER_BINDING output_binding = {cast_output.Get(), 0,
                                         y.size() * sizeof(float)};

    DML_BINDING_DESC input_bind = {DML_BINDING_TYPE_BUFFER, &input_binding};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    cast_op->Execute({input_bind}, {output_bind});
  }

  // 2. Multiply scales
  {
    DML_BUFFER_BINDING a_scale_binding = {a_scale_buffer, 0,
                                          a_scale.size() * sizeof(float)};
    DML_BUFFER_BINDING b_scale_binding = {b_scale_buffer, 0,
                                          b_scale.size() * sizeof(float)};
    DML_BUFFER_BINDING scale_output_binding = {combined_scale.Get(), 0,
                                               y.size() * sizeof(float)};

    std::vector<DML_BINDING_DESC> inputs = {
        {DML_BINDING_TYPE_BUFFER, &a_scale_binding},
        {DML_BINDING_TYPE_BUFFER, &b_scale_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER,
                                    &scale_output_binding};

    scale_mult_op->Execute(inputs, {output_bind});
  }

  // 3. Divide cast output by combined scale
  {
    DML_BUFFER_BINDING cast_binding = {cast_output.Get(), 0,
                                       y.size() * sizeof(float)};
    DML_BUFFER_BINDING scale_binding = {combined_scale.Get(), 0,
                                        y.size() * sizeof(float)};
    DML_BUFFER_BINDING divide_output_binding = {divide_output.Get(), 0,
                                                y.size() * sizeof(float)};

    std::vector<DML_BINDING_DESC> inputs = {
        {DML_BINDING_TYPE_BUFFER, &cast_binding},
        {DML_BINDING_TYPE_BUFFER, &scale_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER,
                                    &divide_output_binding};

    divide_op->Execute(inputs, {output_bind});
  }

  // 4. Add bias if provided
  Microsoft::WRL::ComPtr<ID3D12Resource> current_output = divide_output;
  if (bias) {
    auto* bias_buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer()));

    DML_BUFFER_BINDING divide_binding = {divide_output.Get(), 0,
                                         y.size() * sizeof(float)};
    DML_BUFFER_BINDING bias_binding = {bias_buffer, 0,
                                       bias->size() * sizeof(float)};
    DML_BUFFER_BINDING bias_output_binding = {bias_output.Get(), 0,
                                              y.size() * sizeof(float)};

    std::vector<DML_BINDING_DESC> inputs = {
        {DML_BINDING_TYPE_BUFFER, &divide_binding},
        {DML_BINDING_TYPE_BUFFER, &bias_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER,
                                    &bias_output_binding};

    bias_add_op->Execute(inputs, {output_bind});
    current_output = bias_output;
  }

  // 5. Apply activation or copy to final output
  if (activation_op) {
    DML_BUFFER_BINDING input_binding = {current_output.Get(), 0,
                                        y.size() * sizeof(float)};
    DML_BUFFER_BINDING output_binding = {y_buffer, 0, y.size() * sizeof(float)};

    DML_BINDING_DESC input_bind = {DML_BINDING_TYPE_BUFFER, &input_binding};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    activation_op->Execute({input_bind}, {output_bind});
  } else {
    // Copy current output to final output
    command_list->CopyBufferRegion(y_buffer, 0, current_output.Get(), 0,
                                   y.size() * sizeof(float));
    device->ExecuteCommandList();
  }
}

}  // namespace ops
}  // namespace ctranslate2
#endif
