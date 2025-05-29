#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/quantize.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename InT, typename OutT>
void Quantize::quantize(const StorageView& input,
                        StorageView& output,
                        StorageView& scale) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");
  static_assert(std::is_same_v<InT, float>, "Input type must be float");
  static_assert(std::is_same_v<OutT, int8_t>, "Output type must be int8_t");

  if (_shift_to_uint8) {
    throw std::invalid_argument(
        "Shift to uint8_t is not supported on DirectML");
  }

  // Check that all StorageViews are on DirectML device
  if (input.device() != Device::DirectML) {
    throw std::invalid_argument("Input StorageView must be on DirectML device");
  }
  if (output.device() != Device::DirectML) {
    throw std::invalid_argument(
        "Output StorageView must be on DirectML device");
  }
  if (scale.device() != Device::DirectML) {
    throw std::invalid_argument("Scale StorageView must be on DirectML device");
  }

  // Get DirectML device
  auto* device = dml::get_device();

  const dim_t batch_size = scale.size();
  const dim_t depth = input.dim(-1);

  const size_t input_size_bytes = input.size() * sizeof(float);
  const size_t output_size_bytes = output.size() * sizeof(int8_t);
  const size_t scale_size_bytes = scale.size() * sizeof(float);

  // Create tensor descriptors
  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  input_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  input_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_buffer_desc.DimensionCount = 2;
  UINT input_sizes[] = {static_cast<UINT>(batch_size),
                        static_cast<UINT>(depth)};
  UINT input_strides[] = {static_cast<UINT>(depth), 1};
  input_buffer_desc.Sizes = input_sizes;
  input_buffer_desc.Strides = input_strides;
  input_buffer_desc.TotalTensorSizeInBytes = input_size_bytes;
  input_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_desc = {};
  input_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_desc.Desc = &input_buffer_desc;

  DML_BUFFER_TENSOR_DESC scale_buffer_desc = {};
  scale_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  scale_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  scale_buffer_desc.DimensionCount = 2;
  UINT scale_sizes[] = {static_cast<UINT>(batch_size), 1};
  UINT scale_strides[] = {1, 1};
  scale_buffer_desc.Sizes = scale_sizes;
  scale_buffer_desc.Strides = scale_strides;
  scale_buffer_desc.TotalTensorSizeInBytes = scale_size_bytes;
  scale_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC scale_desc = {};
  scale_desc.Type = DML_TENSOR_TYPE_BUFFER;
  scale_desc.Desc = &scale_buffer_desc;

  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};
  output_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_INT8;
  output_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_buffer_desc.DimensionCount = 2;
  output_buffer_desc.Sizes = input_sizes;  // Same shape as input
  output_buffer_desc.Strides = input_strides;
  output_buffer_desc.TotalTensorSizeInBytes = output_size_bytes;
  output_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC output_desc = {};
  output_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_desc.Desc = &output_buffer_desc;

  // Get D3D12 resources from StorageView buffers
  ID3D12Resource* input_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  ID3D12Resource* output_resource =
      reinterpret_cast<ID3D12Resource*>(output.buffer());
  ID3D12Resource* scale_resource =
      reinterpret_cast<ID3D12Resource*>(scale.buffer());

  // Create intermediate buffers
  auto abs_buffer = device->CreatePreferredDeviceMemoryBuffer(input_size_bytes);
  auto max_buffer = device->CreatePreferredDeviceMemoryBuffer(scale_size_bytes);
  auto scale_factor_buffer =
      device->CreatePreferredDeviceMemoryBuffer(scale_size_bytes);
  auto scaled_buffer =
      device->CreatePreferredDeviceMemoryBuffer(input_size_bytes);
  auto rounded_buffer =
      device->CreatePreferredDeviceMemoryBuffer(input_size_bytes);

  // Create constant buffers
  const float constant_127 = 127.0f;
  const float constant_epsilon = 1e-8f;  // To avoid division by zero
  auto const_127_buffer = device->Upload(
      sizeof(float),
      std::string_view(reinterpret_cast<const char*>(&constant_127),
                       sizeof(float)));
  auto const_epsilon_buffer = device->Upload(
      sizeof(float),
      std::string_view(reinterpret_cast<const char*>(&constant_epsilon),
                       sizeof(float)));

  // Constant tensor descriptors
  DML_BUFFER_TENSOR_DESC const_buffer_desc = {};
  const_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
  const_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  const_buffer_desc.DimensionCount = 1;
  UINT const_sizes[] = {1};
  UINT const_strides[] = {1};
  const_buffer_desc.Sizes = const_sizes;
  const_buffer_desc.Strides = const_strides;
  const_buffer_desc.TotalTensorSizeInBytes = sizeof(float);
  const_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC const_desc = {};
  const_desc.Type = DML_TENSOR_TYPE_BUFFER;
  const_desc.Desc = &const_buffer_desc;

  // Step 1: Compute absolute values
  DML_ELEMENT_WISE_ABS_OPERATOR_DESC abs_desc = {};
  abs_desc.InputTensor = &input_desc;
  abs_desc.OutputTensor = &input_desc;

  DML_OPERATOR_DESC abs_op_desc = {};
  abs_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ABS;
  abs_op_desc.Desc = &abs_desc;

  auto abs_compiled_op = dml::GetOrCreateCompiledOperatorApi(&abs_op_desc);

  // Step 2: Reduce to find max along depth dimension
  UINT axes[] = {1};
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_MAX;
  reduce_desc.InputTensor = &input_desc;
  reduce_desc.OutputTensor = &scale_desc;
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = axes;

  DML_OPERATOR_DESC reduce_op_desc = {};
  reduce_op_desc.Type = DML_OPERATOR_REDUCE;
  reduce_op_desc.Desc = &reduce_desc;

  auto reduce_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&reduce_op_desc);

  // Step 3: Avoid division by zero - max(max_val, epsilon)
  DML_ELEMENT_WISE_MAX_OPERATOR_DESC max_epsilon_desc = {};
  max_epsilon_desc.ATensor = &scale_desc;
  max_epsilon_desc.BTensor = &const_desc;
  max_epsilon_desc.OutputTensor = &scale_desc;

  DML_OPERATOR_DESC max_epsilon_op_desc = {};
  max_epsilon_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MAX;
  max_epsilon_op_desc.Desc = &max_epsilon_desc;

  auto max_epsilon_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&max_epsilon_op_desc);

  // Step 4: Compute scale factors: 127.0 / max_val
  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC divide_desc = {};
  divide_desc.ATensor = &const_desc;
  divide_desc.BTensor = &scale_desc;
  divide_desc.OutputTensor = &scale_desc;

  DML_OPERATOR_DESC divide_op_desc = {};
  divide_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE;
  divide_op_desc.Desc = &divide_desc;

  auto divide_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&divide_op_desc);

  // Step 5: Broadcast and multiply input by scale factors
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc = {};
  multiply_desc.ATensor = &input_desc;
  multiply_desc.BTensor = &scale_desc;  // This will be broadcasted
  multiply_desc.OutputTensor = &input_desc;

  DML_OPERATOR_DESC multiply_op_desc = {};
  multiply_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
  multiply_op_desc.Desc = &multiply_desc;

  auto multiply_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&multiply_op_desc);

  // Step 6: Optional rounding
  dml::Operator* round_compiled_op;
  if (_round_before_cast) {
    DML_ELEMENT_WISE_ROUND_OPERATOR_DESC round_desc = {};
    round_desc.InputTensor = &input_desc;
    round_desc.OutputTensor = &input_desc;
    round_desc.RoundingMode = DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN;

    DML_OPERATOR_DESC round_op_desc = {};
    round_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ROUND;
    round_op_desc.Desc = &round_desc;

    round_compiled_op = dml::GetOrCreateCompiledOperatorApi(&round_op_desc);
  }

  // Step 7: Cast to int8
  DML_CAST_OPERATOR_DESC cast_desc = {};
  cast_desc.InputTensor = &input_desc;
  cast_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC cast_op_desc = {};
  cast_op_desc.Type = DML_OPERATOR_CAST;
  cast_op_desc.Desc = &cast_desc;

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Execute operations

  // 1. Absolute value
  {
    Microsoft::WRL::ComPtr<IDMLOperatorInitializer> initializer;

    DML_BUFFER_BINDING input_binding = {input_resource, 0, input_size_bytes};
    DML_BUFFER_BINDING output_binding = {abs_buffer.Get(), 0, input_size_bytes};
    std::vector<DML_BINDING_DESC> input_bind = {
        {DML_BINDING_TYPE_BUFFER, &input_binding}};
    std::vector<DML_BINDING_DESC> output_bind = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};

    abs_compiled_op->Execute(input_bind, output_bind);
  }

  // 2. Reduce max
  {
    DML_BUFFER_BINDING input_binding = {abs_buffer.Get(), 0, input_size_bytes};
    DML_BUFFER_BINDING output_binding = {max_buffer.Get(), 0, scale_size_bytes};
    DML_BINDING_DESC input_bind = {DML_BINDING_TYPE_BUFFER, &input_binding};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    reduce_compiled_op->Execute({input_bind}, {output_bind});
  }

  // 3. Avoid division by zero
  {
    DML_BUFFER_BINDING input1_binding = {max_buffer.Get(), 0, scale_size_bytes};
    DML_BUFFER_BINDING input2_binding = {const_epsilon_buffer.Get(), 0,
                                         sizeof(float)};
    DML_BUFFER_BINDING output_binding = {scale_factor_buffer.Get(), 0,
                                         scale_size_bytes};
    std::vector<DML_BINDING_DESC> input_binds = {
        {DML_BINDING_TYPE_BUFFER, &input1_binding},
        {DML_BINDING_TYPE_BUFFER, &input2_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    max_epsilon_compiled_op->Execute(input_binds, {output_bind});
  }

  // 4. Compute scale factors (127 / max_val)
  {
    DML_BUFFER_BINDING input1_binding = {const_127_buffer.Get(), 0,
                                         sizeof(float)};
    DML_BUFFER_BINDING input2_binding = {scale_factor_buffer.Get(), 0,
                                         scale_size_bytes};
    DML_BUFFER_BINDING output_binding = {scale_resource, 0, scale_size_bytes};
    std::vector<DML_BINDING_DESC> input_binds = {
        {DML_BINDING_TYPE_BUFFER, &input1_binding},
        {DML_BINDING_TYPE_BUFFER, &input2_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    divide_compiled_op->Execute(input_binds, {output_bind});
  }

  // 5. Multiply input by scale factors
  {
    DML_BUFFER_BINDING input1_binding = {input_resource, 0, input_size_bytes};
    DML_BUFFER_BINDING input2_binding = {scale_resource, 0, scale_size_bytes};
    DML_BUFFER_BINDING output_binding = {scaled_buffer.Get(), 0,
                                         input_size_bytes};
    std::vector<DML_BINDING_DESC> input_binds = {
        {DML_BINDING_TYPE_BUFFER, &input1_binding},
        {DML_BINDING_TYPE_BUFFER, &input2_binding}};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    multiply_compiled_op->Execute(input_binds, {output_bind});
  }

  // 6. Optional rounding
  ID3D12Resource* final_float_buffer = scaled_buffer.Get();
  if (_round_before_cast) {
    DML_BUFFER_BINDING input_binding = {scaled_buffer.Get(), 0,
                                        input_size_bytes};
    DML_BUFFER_BINDING output_binding = {rounded_buffer.Get(), 0,
                                         input_size_bytes};
    DML_BINDING_DESC input_bind = {DML_BINDING_TYPE_BUFFER, &input_binding};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    round_compiled_op->Execute({input_bind}, {output_bind});

    final_float_buffer = rounded_buffer.Get();
  }

  // 7. Cast to int8
  {
    DML_BUFFER_BINDING input_binding = {final_float_buffer, 0,
                                        input_size_bytes};
    DML_BUFFER_BINDING output_binding = {output_resource, 0, output_size_bytes};
    DML_BINDING_DESC input_bind = {DML_BINDING_TYPE_BUFFER, &input_binding};
    DML_BINDING_DESC output_bind = {DML_BINDING_TYPE_BUFFER, &output_binding};

    cast_compiled_op->Execute({input_bind}, {output_bind});
  }
}

// Explicit template instantiation
template void Quantize::quantize<Device::DirectML, float, int8_t>(
    const StorageView&,
    StorageView&,
    StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2
#endif