#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/tile.h"

#include "dml/backend_dml.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Tile::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t inner_size,
                   StorageView& output) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Convert ctranslate2 data type to DML data type
  DML_TENSOR_DATA_TYPE dml_data_type;
  if constexpr (std::is_same_v<T, float>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, ctranslate2::float16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT32;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT16;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_INT8;
  } else if constexpr (std::is_same_v<T, ctranslate2::bfloat16_t>) {
    throw std::invalid_argument(
        "DirectML does not support bfloat16 for Tile operation");
  } else {
    static_assert(sizeof(T) == 0,
                  "Unsupported data type for DirectML Tile operation");
  }

  // Set up input tensor descriptor
  // Input shape: [outer_size, inner_size]
  UINT input_sizes[2] = {static_cast<UINT>(outer_size),
                         static_cast<UINT>(inner_size)};

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  input_buffer_desc.DataType = dml_data_type;
  input_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_buffer_desc.DimensionCount = 2;
  input_buffer_desc.Sizes = input_sizes;
  input_buffer_desc.Strides = nullptr;  // Use default strides
  input_buffer_desc.TotalTensorSizeInBytes = input.size() * sizeof(T);
  input_buffer_desc.GuaranteedBaseOffsetAlignment =
      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_tensor_desc.Desc = &input_buffer_desc;

  // Set up output tensor descriptor
  // Output shape: [outer_size, inner_size * _num_tiles]
  UINT output_sizes[2] = {static_cast<UINT>(outer_size),
                          static_cast<UINT>(inner_size * _num_tiles)};

  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};
  output_buffer_desc.DataType = dml_data_type;
  output_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_buffer_desc.DimensionCount = 2;
  output_buffer_desc.Sizes = output_sizes;
  output_buffer_desc.Strides = nullptr;
  output_buffer_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);
  output_buffer_desc.GuaranteedBaseOffsetAlignment =
      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor_desc.Desc = &output_buffer_desc;

  // Set up repeats array: repeat 1 time along outer dim, _num_tiles times along
  // inner dim
  UINT repeats[2] = {1, static_cast<UINT>(_num_tiles)};

  // Create tile operator descriptor
  DML_TILE_OPERATOR_DESC tile_desc = {};
  tile_desc.InputTensor = &input_tensor_desc;
  tile_desc.OutputTensor = &output_tensor_desc;
  tile_desc.RepeatsCount = 2;
  tile_desc.Repeats = repeats;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_TILE;
  op_desc.Desc = &tile_desc;

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Get binding properties
  auto binding_props = compiled_op->GetBindingProperties();

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();
  binding_table_desc
      .CPUDescriptorHandle = {};  // Device will handle descriptor allocation
  binding_table_desc.GPUDescriptorHandle = {};
  binding_table_desc.SizeInDescriptors = binding_props.RequiredDescriptorCount;

  HRESULT hr = dml_device->CreateBindingTable(&binding_table_desc,
                                              IID_PPV_ARGS(&binding_table));
  if (FAILED(hr)) {
    throw std::runtime_error(
        "Failed to create DirectML binding table for Tile operation");
  }

  // Create temporary resource if needed
  Microsoft::WRL::ComPtr<ID3D12Resource> temp_resource;
  if (binding_props.TemporaryResourceSize > 0) {
    temp_resource = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.TemporaryResourceSize);
  }

  // Bind input buffer
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  input_binding.Offset = 0;
  input_binding.SizeInBytes = input.size() * sizeof(T);

  DML_BINDING_DESC input_binding_desc = {};
  input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  input_binding_desc.Desc = &input_binding;

  binding_table->BindInputs(1, &input_binding_desc);

  // Bind output buffer
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  binding_table->BindOutputs(1, &output_binding_desc);

  // Bind temporary resource if needed
  if (temp_resource) {
    DML_BUFFER_BINDING temp_binding = {};
    temp_binding.Buffer = temp_resource.Get();
    temp_binding.Offset = 0;
    temp_binding.SizeInBytes = binding_props.TemporaryResourceSize;

    DML_BINDING_DESC temp_binding_desc = {};
    temp_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
    temp_binding_desc.Desc = &temp_binding;

    binding_table->BindTemporaryResource(&temp_binding_desc);
  }

  // Keep resources alive until dispatch completes
  if (temp_resource) {
    device->KeepAliveUntilNextCommandListDispatch(std::move(temp_resource));
  }
  device->KeepAliveUntilNextCommandListDispatch(compiled_op);
  device->KeepAliveUntilNextCommandListDispatch(binding_table);

  // Record dispatch operation
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());

  // Execute the command list
  device->ExecuteCommandList();
}

#define DECLARE_IMPL(T)                                 \
  template void Tile::compute<Device::DirectML, T>(     \
      const StorageView& input, const dim_t outer_size, \
      const dim_t inner_size, StorageView& output) const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML