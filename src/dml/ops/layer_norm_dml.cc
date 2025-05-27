#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/layer_norm.h"

#include "dml/backend_dml.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void LayerNorm::compute(const StorageView* beta,
                        const StorageView* gamma,
                        const StorageView& input,
                        const dim_t axis,
                        const dim_t outer_size,
                        const dim_t axis_size,
                        const dim_t inner_size,
                        StorageView& output) const {
  if (axis != input.rank() - 1 || !beta || !gamma)
    throw std::invalid_argument(
        "Generalized LayerNorm is currently not implemented on DirectML");

  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  // Convert CT2 data types to DML data types
  DML_TENSOR_DATA_TYPE dml_data_type;
  if constexpr (std::is_same_v<T, float>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else {
    throw std::invalid_argument("Unsupported data type for DirectML LayerNorm");
  }

  // Get input dimensions
  const auto& input_shape = input.shape();
  std::vector<UINT> input_sizes(input_shape.begin(), input_shape.end());

  // Create input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  input_buffer_desc.DataType = dml_data_type;
  input_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_buffer_desc.DimensionCount = static_cast<UINT>(input_sizes.size());
  input_buffer_desc.Sizes = input_sizes.data();
  input_buffer_desc.Strides = nullptr;  // Use default strides
  input_buffer_desc.TotalTensorSizeInBytes = input.size() * sizeof(T);
  input_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_tensor_desc.Desc = &input_buffer_desc;

  // Create scale (gamma) tensor descriptor - broadcast to match input shape
  std::vector<UINT> scale_sizes(input_sizes.size(), 1);
  scale_sizes.back() = axis_size;  // Scale tensor matches the last dimension

  DML_BUFFER_TENSOR_DESC scale_buffer_desc = {};
  scale_buffer_desc.DataType = dml_data_type;
  scale_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  scale_buffer_desc.DimensionCount = static_cast<UINT>(scale_sizes.size());
  scale_buffer_desc.Sizes = scale_sizes.data();
  scale_buffer_desc.Strides = nullptr;
  scale_buffer_desc.TotalTensorSizeInBytes = gamma->size() * sizeof(T);
  scale_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC scale_tensor_desc = {};
  scale_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  scale_tensor_desc.Desc = &scale_buffer_desc;

  // Create bias (beta) tensor descriptor - same shape as scale
  std::vector<UINT> bias_sizes = scale_sizes;

  DML_BUFFER_TENSOR_DESC bias_buffer_desc = {};
  bias_buffer_desc.DataType = dml_data_type;
  bias_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  bias_buffer_desc.DimensionCount = static_cast<UINT>(bias_sizes.size());
  bias_buffer_desc.Sizes = bias_sizes.data();
  bias_buffer_desc.Strides = nullptr;
  bias_buffer_desc.TotalTensorSizeInBytes = beta->size() * sizeof(T);
  bias_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC bias_tensor_desc = {};
  bias_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  bias_tensor_desc.Desc = &bias_buffer_desc;

  // Create output tensor descriptor (same shape as input)
  DML_BUFFER_TENSOR_DESC output_buffer_desc = input_buffer_desc;
  output_buffer_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);

  DML_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor_desc.Desc = &output_buffer_desc;

  // Create Mean Variance Normalization operator
  UINT normalization_axis = static_cast<UINT>(axis);

  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC mvn_desc = {};
  mvn_desc.InputTensor = &input_tensor_desc;
  mvn_desc.ScaleTensor = &scale_tensor_desc;
  mvn_desc.BiasTensor = &bias_tensor_desc;
  mvn_desc.OutputTensor = &output_tensor_desc;
  mvn_desc.AxisCount = 1;
  mvn_desc.Axes = &normalization_axis;
  mvn_desc.UseMean = TRUE;
  mvn_desc.UseVariance = TRUE;
  mvn_desc.Epsilon = _epsilon;
  mvn_desc.FusedActivation = nullptr;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2;
  op_desc.Desc = &mvn_desc;

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Get binding properties
  auto binding_props = compiled_op->GetBindingProperties();

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();
  binding_table_desc.CPUDescriptorHandle = {};  // Managed by device
  binding_table_desc.GPUDescriptorHandle = {};  // Managed by device
  binding_table_desc.SizeInDescriptors = binding_props.RequiredDescriptorCount;

  if (FAILED(dml_device->CreateBindingTable(&binding_table_desc,
                                            IID_PPV_ARGS(&binding_table)))) {
    throw std::runtime_error(
        "Failed to create DML binding table for LayerNorm");
  }

  // Create temporary resource if needed
  Microsoft::WRL::ComPtr<ID3D12Resource> temp_resource;
  if (binding_props.TemporaryResourceSize > 0) {
    temp_resource = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.TemporaryResourceSize);
    device->KeepAliveUntilNextCommandListDispatch(temp_resource);
  }

  // Get D3D12 resources from StorageView buffers
  // StorageView::buffer() returns ID3D12Resource* for DirectML backend
  auto input_resource =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto scale_resource =
      static_cast<ID3D12Resource*>(const_cast<void*>(gamma->buffer()));
  auto bias_resource =
      static_cast<ID3D12Resource*>(const_cast<void*>(beta->buffer()));
  auto output_resource = static_cast<ID3D12Resource*>(output.buffer());

  // Bind input tensors
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer = input_resource;
  input_binding.Offset = 0;
  input_binding.SizeInBytes = input.size() * sizeof(T);

  DML_BUFFER_BINDING scale_binding = {};
  scale_binding.Buffer = scale_resource;
  scale_binding.Offset = 0;
  scale_binding.SizeInBytes = gamma->size() * sizeof(T);

  DML_BUFFER_BINDING bias_binding = {};
  bias_binding.Buffer = bias_resource;
  bias_binding.Offset = 0;
  bias_binding.SizeInBytes = beta->size() * sizeof(T);

  DML_BINDING_DESC input_bindings[] = {
      {DML_BINDING_TYPE_BUFFER, &input_binding},
      {DML_BINDING_TYPE_BUFFER, &scale_binding},
      {DML_BINDING_TYPE_BUFFER, &bias_binding}};

  binding_table->BindInputs(3, input_bindings);

  // Bind output tensor
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = output_resource;
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_bindings[] = {
      {DML_BINDING_TYPE_BUFFER, &output_binding}};

  binding_table->BindOutputs(1, output_bindings);

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

  // Record dispatch command
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());

  // Execute command list to perform the computation on GPU
  device->ExecuteCommandList();
}

#define DECLARE_IMPL(T)                                                   \
  template void LayerNorm::compute<Device::DirectML, T>(                  \
      const StorageView* beta, const StorageView* gamma,                  \
      const StorageView& input, const dim_t axis, const dim_t outer_size, \
      const dim_t axis_size, const dim_t inner_size, StorageView& output) \
      const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML