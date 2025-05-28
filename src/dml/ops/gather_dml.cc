#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/gather.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Gather::compute(const StorageView& data,
                     const StorageView& input,
                     const dim_t axis,
                     const dim_t batch_dims,
                     StorageView& output) const {
  if (axis != batch_dims) {
    throw std::invalid_argument(
        "Gather only supports indexing the first non batch dimension");
  }

  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  // Convert data type to DML tensor data type
  DML_TENSOR_DATA_TYPE dml_data_type;
  if (std::is_same_v<T, float>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if (std::is_same_v<T, float16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if (std::is_same_v<T, int32_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT32;
  } else if (std::is_same_v<T, int16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT16;
  } else if (std::is_same_v<T, int8_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT8;
  } else if (std::is_same_v<T, uint32_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_UINT32;
  } else if (std::is_same_v<T, uint16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_UINT16;
  } else if (std::is_same_v<T, uint8_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_UINT8;
  } else {
    throw std::invalid_argument(
        "Unsupported data type for DML gather operation");
  }

  // Get tensor shapes
  const auto& data_shape = data.shape();
  const auto& indices_shape = input.shape();
  const auto& output_shape = output.shape();

  // Convert shapes to DML format (UINT)
  std::vector<UINT> data_sizes(data_shape.begin(), data_shape.end());
  std::vector<UINT> indices_sizes(indices_shape.begin(), indices_shape.end());
  std::vector<UINT> output_sizes(output_shape.begin(), output_shape.end());

  // Create buffer tensor descriptors
  DML_BUFFER_TENSOR_DESC data_buffer_desc = {};
  data_buffer_desc.DataType = dml_data_type;
  data_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  data_buffer_desc.DimensionCount = static_cast<UINT>(data_sizes.size());
  data_buffer_desc.Sizes = data_sizes.data();
  data_buffer_desc.Strides = nullptr;  // Packed layout
  data_buffer_desc.TotalTensorSizeInBytes = data.size() * sizeof(T);
  data_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_BUFFER_TENSOR_DESC indices_buffer_desc = {};
  indices_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_INT32;
  indices_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  indices_buffer_desc.DimensionCount = static_cast<UINT>(indices_sizes.size());
  indices_buffer_desc.Sizes = indices_sizes.data();
  indices_buffer_desc.Strides = nullptr;  // Packed layout
  indices_buffer_desc.TotalTensorSizeInBytes = input.size() * sizeof(int32_t);
  indices_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};
  output_buffer_desc.DataType = dml_data_type;
  output_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_buffer_desc.DimensionCount = static_cast<UINT>(output_sizes.size());
  output_buffer_desc.Sizes = output_sizes.data();
  output_buffer_desc.Strides = nullptr;  // Packed layout
  output_buffer_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);
  output_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  // Create tensor descriptors
  DML_TENSOR_DESC data_tensor_desc = {};
  data_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  data_tensor_desc.Desc = &data_buffer_desc;

  DML_TENSOR_DESC indices_tensor_desc = {};
  indices_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  indices_tensor_desc.Desc = &indices_buffer_desc;

  DML_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor_desc.Desc = &output_buffer_desc;

  // Create gather operator descriptor
  DML_GATHER_OPERATOR_DESC gather_desc = {};
  gather_desc.InputTensor = &data_tensor_desc;
  gather_desc.IndicesTensor = &indices_tensor_desc;
  gather_desc.OutputTensor = &output_tensor_desc;
  gather_desc.Axis = static_cast<UINT>(axis);
  gather_desc.IndexDimensions = static_cast<UINT>(indices_shape.size());

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_GATHER;
  op_desc.Desc = &gather_desc;

  // Get or create compiled operator from cache
  auto compiled_operator = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Convert StorageView buffers to ID3D12Resource*
  // As noted in the requirements, buffer() returns ID3D12Resource* for DML
  // device
  auto data_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(data.buffer()));
  auto indices_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto output_resource = reinterpret_cast<ID3D12Resource*>(output.buffer());

  // Bind input tensors
  DML_BUFFER_BINDING data_binding = {};
  data_binding.Buffer = data_resource;
  data_binding.Offset = 0;
  data_binding.SizeInBytes = data.size() * sizeof(T);

  DML_BUFFER_BINDING indices_binding = {};
  indices_binding.Buffer = indices_resource;
  indices_binding.Offset = 0;
  indices_binding.SizeInBytes = input.size() * sizeof(int32_t);

  std::vector<DML_BINDING_DESC> input_bindings = {
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &data_binding},
      {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &indices_binding}};

  // Bind output tensor
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = output_resource;
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  compiled_operator->Execute(input_bindings, {output_binding_desc});
}

#define DECLARE_IMPL(T)                                                    \
  template void Gather::compute<Device::DirectML, T>(                      \
      const StorageView& data, const StorageView& input, const dim_t axis, \
      const dim_t batch_dims, StorageView& output) const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML