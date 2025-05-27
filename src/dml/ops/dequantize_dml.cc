#include "ctranslate2/ops/dequantize.h"
#include "dml/backend_dml.h"
#include "dml/common.h"
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
          static_cast<UINT64>(input.size() * sizeof(int8_t))};

  DML_BUFFER_TENSOR_DESC float_output_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_FLOAT32,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(input.rank()),
      .Sizes = input_sizes.data(),
      .Strides = nullptr,
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(input.size() * sizeof(float))};

  DML_TENSOR_DESC input_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                .Desc = &input_buffer_desc};

  DML_TENSOR_DESC cast_output_desc = {.Type = DML_TENSOR_TYPE_BUFFER,
                                      .Desc = &float_output_desc};

  DML_CAST_OPERATOR_DESC cast_desc = {.InputTensor = &input_desc,
                                      .OutputTensor = &cast_output_desc};

  DML_OPERATOR_DESC cast_op_desc = {.Type = DML_OPERATOR_CAST,
                                    .Desc = &cast_desc};

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Create binding table for cast
  Microsoft::WRL::ComPtr<IDMLBindingTable> cast_binding_table;
  DML_BINDING_TABLE_DESC cast_binding_desc = {
      .Dispatchable = cast_compiled_op.Get(),
      .CPUDescriptorHandle = {},
      .GPUDescriptorHandle = {},
      .SizeInDescriptors = 0};

  THROW_IF_FAILED(dml_device->CreateBindingTable(
      &cast_binding_desc, IID_PPV_ARGS(&cast_binding_table)));

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

  cast_binding_table->BindInputs(1, &cast_input_bind);
  cast_binding_table->BindOutputs(1, &cast_output_bind);

  device->RecordDispatch(cast_compiled_op.Get(), cast_binding_table.Get());

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
    for (size_t i = 0; i < input.rank() - 1; ++i) {
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
          static_cast<UINT64>(scale.size() * sizeof(float))};

  DML_BUFFER_TENSOR_DESC div_output_desc = {
      .DataType = DML_TENSOR_DATA_TYPE_FLOAT32,
      .Flags = DML_TENSOR_FLAG_NONE,
      .DimensionCount = static_cast<UINT>(output.rank()),
      .Sizes = input_sizes.data(),
      .Strides = nullptr,
      .TotalTensorSizeInBytes =
          static_cast<UINT64>(output.size() * sizeof(float))};

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

  // Create binding table for division
  Microsoft::WRL::ComPtr<IDMLBindingTable> div_binding_table;
  DML_BINDING_TABLE_DESC div_binding_desc = {
      .Dispatchable = divide_compiled_op.Get(),
      .CPUDescriptorHandle = {},
      .GPUDescriptorHandle = {},
      .SizeInDescriptors = 0};

  THROW_IF_FAILED(dml_device->CreateBindingTable(
      &div_binding_desc, IID_PPV_ARGS(&div_binding_table)));

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

  DML_BINDING_DESC div_input_bindings[2] = {
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &div_input1_binding},
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &div_input2_binding}};

  DML_BINDING_DESC div_output_bind = {.Type = DML_BINDING_TYPE_BUFFER,
                                      .Desc = &div_output_binding};

  div_binding_table->BindInputs(2, div_input_bindings);
  div_binding_table->BindOutputs(1, &div_output_bind);

  device->RecordDispatch(divide_compiled_op.Get(), div_binding_table.Get());

  // Execute all recorded operations
  device->ExecuteCommandList();
}

}  // namespace ops
}  // namespace ctranslate2