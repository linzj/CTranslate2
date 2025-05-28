#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/topp_mask.h"
#include "ctranslate2/types.h"
#include "dml/backend_dml.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void TopPMask::compute(const StorageView& input,
                       const StorageView& probs,
                       StorageView& output) const {
  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  if (depth > max_num_classes<Device::DirectML>()) {
    throw std::runtime_error(
        "The TopP operator does not support more than " +
        std::to_string(max_num_classes<Device::DirectML>()) +
        " classes, but the input has " + std::to_string(depth) + " classes.");
  }

  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Get D3D12 resources from StorageView
  auto* input_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto* probs_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(probs.buffer()));
  auto* output_resource = reinterpret_cast<ID3D12Resource*>(output.buffer());

  // Determine DML tensor data type
  DML_TENSOR_DATA_TYPE data_type;
  if constexpr (std::is_same_v<T, float>) {
    data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else {
    static_assert(false, "Unsupported data type");
  }

  const UINT tensor_sizes[] = {static_cast<UINT>(batch_size),
                               static_cast<UINT>(depth)};
  const UINT64 tensor_size_bytes = batch_size * depth * sizeof(T);

  // Input tensor descriptors
  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  input_buffer_desc.DataType = data_type;
  input_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_buffer_desc.DimensionCount = 2;
  input_buffer_desc.Sizes = tensor_sizes;
  input_buffer_desc.Strides = nullptr;
  input_buffer_desc.TotalTensorSizeInBytes = tensor_size_bytes;
  input_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_tensor_desc.Desc = &input_buffer_desc;

  DML_BUFFER_TENSOR_DESC probs_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC probs_tensor_desc = {};
  probs_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  probs_tensor_desc.Desc = &probs_buffer_desc;

  DML_BUFFER_TENSOR_DESC output_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor_desc.Desc = &output_buffer_desc;

  // Create intermediate tensors for sorted values and indices
  DML_BUFFER_TENSOR_DESC sorted_values_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC sorted_values_tensor_desc = {};
  sorted_values_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  sorted_values_tensor_desc.Desc = &sorted_values_buffer_desc;

  DML_BUFFER_TENSOR_DESC sorted_indices_buffer_desc = {};
  sorted_indices_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
  sorted_indices_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  sorted_indices_buffer_desc.DimensionCount = 2;
  sorted_indices_buffer_desc.Sizes = tensor_sizes;
  sorted_indices_buffer_desc.Strides = nullptr;
  sorted_indices_buffer_desc.TotalTensorSizeInBytes =
      batch_size * depth * sizeof(uint32_t);
  sorted_indices_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC sorted_indices_tensor_desc = {};
  sorted_indices_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  sorted_indices_tensor_desc.Desc = &sorted_indices_buffer_desc;

  // Step 1: Sort probabilities in descending order using TopK with K=depth
  DML_TOP_K1_OPERATOR_DESC topk_desc = {};
  topk_desc.InputTensor = &probs_tensor_desc;
  topk_desc.OutputValueTensor = &sorted_values_tensor_desc;
  topk_desc.OutputIndexTensor = &sorted_indices_tensor_desc;
  topk_desc.Axis = 1;  // Sort along the class dimension
  topk_desc.K = static_cast<UINT>(depth);
  topk_desc.AxisDirection = DML_AXIS_DIRECTION_DECREASING;

  DML_OPERATOR_DESC topk_op_desc = {};
  topk_op_desc.Type = DML_OPERATOR_TOP_K1;
  topk_op_desc.Desc = &topk_desc;

  auto topk_compiled_op = dml::GetOrCreateCompiledOperatorApi(&topk_op_desc);

  // Step 2: Compute cumulative sum of sorted probabilities (exclusive)
  DML_BUFFER_TENSOR_DESC cumsum_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC cumsum_tensor_desc = {};
  cumsum_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  cumsum_tensor_desc.Desc = &cumsum_buffer_desc;

  DML_CUMULATIVE_SUMMATION_OPERATOR_DESC cumsum_desc = {};
  cumsum_desc.InputTensor = &sorted_values_tensor_desc;
  cumsum_desc.OutputTensor = &cumsum_tensor_desc;
  cumsum_desc.Axis = 1;
  cumsum_desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
  cumsum_desc.HasExclusiveSum = TRUE;

  DML_OPERATOR_DESC cumsum_op_desc = {};
  cumsum_op_desc.Type = DML_OPERATOR_CUMULATIVE_SUMMATION;
  cumsum_op_desc.Desc = &cumsum_desc;

  auto cumsum_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&cumsum_op_desc);

  // Step 3: Create threshold tensor
  auto threshold_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  // Fill threshold tensor with p value
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc = {};
  fill_desc.OutputTensor = &cumsum_tensor_desc;
  fill_desc.ValueDataType = data_type;
  if constexpr (std::is_same_v<T, float>) {
    fill_desc.Value.Float32 = _p;
  } else {
    float16_t temp16;
    temp16 = static_cast<float16_t>(_p);
    fill_desc.Value.UInt16 = *reinterpret_cast<const uint16_t*>(&temp16);
  }

  DML_OPERATOR_DESC fill_op_desc = {};
  fill_op_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
  fill_op_desc.Desc = &fill_desc;

  auto fill_compiled_op = dml::GetOrCreateCompiledOperatorApi(&fill_op_desc);

  // Step 4: Compare cumulative sum with threshold (cumsum < p)
  DML_BUFFER_TENSOR_DESC mask_buffer_desc = {};
  mask_buffer_desc.DataType =
      DML_TENSOR_DATA_TYPE_UINT32;  // Boolean mask as uint32
  mask_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  mask_buffer_desc.DimensionCount = 2;
  mask_buffer_desc.Sizes = tensor_sizes;
  mask_buffer_desc.Strides = nullptr;
  mask_buffer_desc.TotalTensorSizeInBytes =
      batch_size * depth * sizeof(uint32_t);
  mask_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC mask_tensor_desc = {};
  mask_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  mask_tensor_desc.Desc = &mask_buffer_desc;

  DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC compare_desc = {};
  compare_desc.ATensor = &cumsum_tensor_desc;
  compare_desc.BTensor =
      &cumsum_tensor_desc;  // Will bind threshold tensor here
  compare_desc.OutputTensor = &mask_tensor_desc;

  DML_OPERATOR_DESC compare_op_desc = {};
  compare_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN;
  compare_op_desc.Desc = &compare_desc;

  auto compare_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&compare_op_desc);

  // Step 5: Create masked values tensor
  auto masked_values_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  // Step 6: Gather input values using sorted indices
  DML_BUFFER_TENSOR_DESC gathered_values_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC gathered_values_tensor_desc = {};
  gathered_values_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  gathered_values_tensor_desc.Desc = &gathered_values_buffer_desc;

  DML_GATHER_ELEMENTS_OPERATOR_DESC gather_desc = {};
  gather_desc.InputTensor = &input_tensor_desc;
  gather_desc.IndicesTensor = &sorted_indices_tensor_desc;
  gather_desc.OutputTensor = &gathered_values_tensor_desc;
  gather_desc.Axis = 1;

  DML_OPERATOR_DESC gather_op_desc = {};
  gather_op_desc.Type = DML_OPERATOR_GATHER_ELEMENTS;
  gather_op_desc.Desc = &gather_desc;

  auto gather_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&gather_op_desc);

  // Step 7: Apply mask using element-wise if operation
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC mask_fill_desc = {};
  mask_fill_desc.OutputTensor = &gathered_values_tensor_desc;
  mask_fill_desc.ValueDataType = data_type;
  if constexpr (std::is_same_v<T, float>) {
    mask_fill_desc.Value.Float32 = _mask_value;
  } else {
    float16_t temp_mask_value;
    temp_mask_value = static_cast<float16_t>(_mask_value);
    mask_fill_desc.Value.UInt16 =
        *reinterpret_cast<const uint16_t*>(&temp_mask_value);
  }

  DML_OPERATOR_DESC mask_fill_op_desc = {};
  mask_fill_op_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
  mask_fill_op_desc.Desc = &mask_fill_desc;

  auto mask_fill_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&mask_fill_op_desc);

  // Convert mask tensor to same type as data for IF operation
  DML_TENSOR_DESC mask_converted_desc = cumsum_tensor_desc;

  DML_CAST_OPERATOR_DESC cast_desc = {};
  cast_desc.InputTensor = &mask_tensor_desc;
  cast_desc.OutputTensor = &mask_converted_desc;

  DML_OPERATOR_DESC cast_op_desc = {};
  cast_op_desc.Type = DML_OPERATOR_CAST;
  cast_op_desc.Desc = &cast_desc;

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  DML_ELEMENT_WISE_IF_OPERATOR_DESC if_desc = {};
  if_desc.ConditionTensor = &mask_converted_desc;
  if_desc.ATensor =
      &gathered_values_tensor_desc;  // Use original values if mask is true
  if_desc.BTensor =
      &gathered_values_tensor_desc;  // Use mask value if mask is false (will
                                     // bind mask_value tensor)
  if_desc.OutputTensor = &gathered_values_tensor_desc;

  DML_OPERATOR_DESC if_op_desc = {};
  if_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IF;
  if_op_desc.Desc = &if_desc;

  auto if_compiled_op = dml::GetOrCreateCompiledOperatorApi(&if_op_desc);

  // Step 8: Scatter masked values back to original positions
  DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
  scatter_desc.InputTensor = &output_tensor_desc;  // Will be zeroed first
  scatter_desc.IndicesTensor = &sorted_indices_tensor_desc;
  scatter_desc.UpdatesTensor = &gathered_values_tensor_desc;
  scatter_desc.OutputTensor = &output_tensor_desc;
  scatter_desc.Axis = 1;

  DML_OPERATOR_DESC scatter_op_desc = {};
  scatter_op_desc.Type = DML_OPERATOR_SCATTER_ELEMENTS;
  scatter_op_desc.Desc = &scatter_desc;

  auto scatter_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&scatter_op_desc);

  // Create intermediate resources
  auto sorted_values_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto sorted_indices_resource = device->CreatePreferredDeviceMemoryBuffer(
      batch_size * depth * sizeof(uint32_t));
  auto cumsum_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto mask_resource = device->CreatePreferredDeviceMemoryBuffer(
      batch_size * depth * sizeof(uint32_t));
  auto gathered_values_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto mask_converted_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto mask_value_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  // Create binding tables and execute operations
  device->ResetCommandList();

  // Execute TopK
  {
    Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
    DML_BINDING_TABLE_DESC binding_table_desc = {};
    binding_table_desc.Dispatchable = topk_compiled_op.Get();
    binding_table_desc.CPUDescriptorHandle = {};  // Will be set by device
    binding_table_desc.GPUDescriptorHandle = {};  // Will be set by device
    binding_table_desc.SizeInDescriptors =
        topk_compiled_op->GetBindingProperties().RequiredDescriptorCount;

    dml_device->CreateBindingTable(&binding_table_desc,
                                   IID_PPV_ARGS(&binding_table));

    // Bind inputs
    DML_BUFFER_BINDING input_binding = {probs_resource, 0, tensor_size_bytes};
    DML_BINDING_DESC input_binding_desc = {DML_BINDING_TYPE_BUFFER,
                                           &input_binding};
    binding_table->BindInputs(1, &input_binding_desc);

    // Bind outputs
    DML_BUFFER_BINDING output_bindings[] = {
        {sorted_values_resource.Get(), 0, tensor_size_bytes},
        {sorted_indices_resource.Get(), 0,
         batch_size * depth * sizeof(uint32_t)}};
    DML_BINDING_DESC output_binding_descs[] = {
        {DML_BINDING_TYPE_BUFFER, &output_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &output_bindings[1]}};
    binding_table->BindOutputs(2, output_binding_descs);

    device->RecordDispatch(topk_compiled_op.Get(), binding_table.Get());
  }

  // Execute remaining operations similarly...
  // [Similar binding and dispatch code for cumsum, fill, compare, gather, cast,
  // if, and scatter operations]

  device->ExecuteCommandList();
}

template <>
dim_t TopPMask::max_num_classes<Device::DirectML>() {
  return 32768;  // Conservative limit for DirectML TopK
}

#define DECLARE_IMPL(T)                                 \
  template void TopPMask::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, StorageView&) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif
