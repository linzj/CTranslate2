#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/topp_mask.h"
#include "ctranslate2/types.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
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
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, float16_t>,
                  "Unsupported data type");
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

  // Sorted probabilities tensor
  DML_BUFFER_TENSOR_DESC sorted_probs_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC sorted_probs_tensor_desc = {};
  sorted_probs_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  sorted_probs_tensor_desc.Desc = &sorted_probs_buffer_desc;

  // Sorted indices tensor (uint32)
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

  // Cumulative sum tensor
  DML_BUFFER_TENSOR_DESC cumsum_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC cumsum_tensor_desc = {};
  cumsum_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  cumsum_tensor_desc.Desc = &cumsum_buffer_desc;

  // Threshold tensor (filled with p value)
  DML_BUFFER_TENSOR_DESC threshold_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC threshold_tensor_desc = {};
  threshold_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  threshold_tensor_desc.Desc = &threshold_buffer_desc;

  // Mask tensor (boolean as uint32)
  DML_BUFFER_TENSOR_DESC mask_buffer_desc = {};
  mask_buffer_desc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
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

  // Gathered input values tensor
  DML_BUFFER_TENSOR_DESC gathered_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC gathered_tensor_desc = {};
  gathered_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  gathered_tensor_desc.Desc = &gathered_buffer_desc;

  // Mask value tensor (filled with mask value)
  DML_BUFFER_TENSOR_DESC mask_value_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC mask_value_tensor_desc = {};
  mask_value_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  mask_value_tensor_desc.Desc = &mask_value_buffer_desc;

  // Create intermediate resources
  auto sorted_probs_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto sorted_indices_resource = device->CreatePreferredDeviceMemoryBuffer(
      batch_size * depth * sizeof(uint32_t));
  auto cumsum_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto threshold_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto mask_resource = device->CreatePreferredDeviceMemoryBuffer(
      batch_size * depth * sizeof(uint32_t));
  auto gathered_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);
  auto mask_value_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  device->ResetCommandList();

  // Step 1: TopK to sort all probabilities in descending order
  DML_TOP_K1_OPERATOR_DESC topk_desc = {};
  topk_desc.InputTensor = &probs_tensor_desc;
  topk_desc.OutputValueTensor = &sorted_probs_tensor_desc;
  topk_desc.OutputIndexTensor = &sorted_indices_tensor_desc;
  topk_desc.Axis = 1;
  topk_desc.K = static_cast<UINT>(depth);
  topk_desc.AxisDirection = DML_AXIS_DIRECTION_DECREASING;

  DML_OPERATOR_DESC topk_op_desc = {};
  topk_op_desc.Type = DML_OPERATOR_TOP_K1;
  topk_op_desc.Desc = &topk_desc;

  auto topk_compiled_op = dml::GetOrCreateCompiledOperatorApi(&topk_op_desc);

  // Execute TopK
  {
    // Bind inputs
    DML_BUFFER_BINDING input_binding = {probs_resource, 0, tensor_size_bytes};
    std::vector<DML_BINDING_DESC> input_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &input_binding}};

    // Bind outputs
    DML_BUFFER_BINDING output_bindings[] = {
        {sorted_probs_resource.Get(), 0, tensor_size_bytes},
        {sorted_indices_resource.Get(), 0,
         batch_size * depth * sizeof(uint32_t)}};
    std::vector<DML_BINDING_DESC> output_binding_descs = {
        {DML_BINDING_TYPE_BUFFER, &output_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &output_bindings[1]}};

    topk_compiled_op->Execute(input_binding_desc, output_binding_descs);
  }

  // Step 2: Cumulative sum of sorted probabilities
  DML_CUMULATIVE_SUMMATION_OPERATOR_DESC cumsum_desc = {};
  cumsum_desc.InputTensor = &sorted_probs_tensor_desc;
  cumsum_desc.OutputTensor = &cumsum_tensor_desc;
  cumsum_desc.Axis = 1;
  cumsum_desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
  cumsum_desc.HasExclusiveSum = TRUE;

  DML_OPERATOR_DESC cumsum_op_desc = {};
  cumsum_op_desc.Type = DML_OPERATOR_CUMULATIVE_SUMMATION;
  cumsum_op_desc.Desc = &cumsum_desc;

  auto cumsum_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&cumsum_op_desc);

  // Execute CumSum
  {
    DML_BUFFER_BINDING input_binding = {sorted_probs_resource.Get(), 0,
                                        tensor_size_bytes};
    std::vector<DML_BINDING_DESC> input_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &input_binding}};

    DML_BUFFER_BINDING output_binding = {cumsum_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};
    cumsum_compiled_op->Execute(input_binding_desc, output_binding_desc);
  }

  // Step 3: Fill threshold tensor with p value
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_threshold_desc = {};
  fill_threshold_desc.OutputTensor = &threshold_tensor_desc;
  fill_threshold_desc.ValueDataType = data_type;
  if constexpr (std::is_same_v<T, float>) {
    fill_threshold_desc.Value.Float32 = _p;
  } else {
    float16_t p_val = static_cast<float16_t>(_p);
    fill_threshold_desc.Value.UInt16 =
        *reinterpret_cast<const uint16_t*>(&p_val);
  }

  DML_OPERATOR_DESC fill_threshold_op_desc = {};
  fill_threshold_op_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
  fill_threshold_op_desc.Desc = &fill_threshold_desc;

  auto fill_threshold_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&fill_threshold_op_desc);

  // Execute fill threshold
  {
    DML_BUFFER_BINDING output_binding = {threshold_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};

    fill_threshold_compiled_op->Execute({}, output_binding_desc);
  }

  // Step 4: Compare cumsum < threshold to create mask
  DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC compare_desc = {};
  compare_desc.ATensor = &cumsum_tensor_desc;
  compare_desc.BTensor = &threshold_tensor_desc;
  compare_desc.OutputTensor = &mask_tensor_desc;

  DML_OPERATOR_DESC compare_op_desc = {};
  compare_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN;
  compare_op_desc.Desc = &compare_desc;

  auto compare_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&compare_op_desc);

  // Execute comparison
  {
    DML_BUFFER_BINDING input_bindings[] = {
        {cumsum_resource.Get(), 0, tensor_size_bytes},
        {threshold_resource.Get(), 0, tensor_size_bytes}};
    std::vector<DML_BINDING_DESC> input_binding_descs = {
        {DML_BINDING_TYPE_BUFFER, &input_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[1]}};

    DML_BUFFER_BINDING output_binding = {mask_resource.Get(), 0,
                                         batch_size * depth * sizeof(uint32_t)};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};

    compare_compiled_op->Execute(input_binding_descs, output_binding_desc);
  }

  // Step 5: Gather input values using sorted indices
  DML_GATHER_ELEMENTS_OPERATOR_DESC gather_desc = {};
  gather_desc.InputTensor = &input_tensor_desc;
  gather_desc.IndicesTensor = &sorted_indices_tensor_desc;
  gather_desc.OutputTensor = &gathered_tensor_desc;
  gather_desc.Axis = 1;

  DML_OPERATOR_DESC gather_op_desc = {};
  gather_op_desc.Type = DML_OPERATOR_GATHER_ELEMENTS;
  gather_op_desc.Desc = &gather_desc;

  auto gather_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&gather_op_desc);

  // Execute gather
  {
    DML_BUFFER_BINDING input_bindings[] = {
        {input_resource, 0, tensor_size_bytes},
        {sorted_indices_resource.Get(), 0,
         batch_size * depth * sizeof(uint32_t)}};
    std::vector<DML_BINDING_DESC> input_binding_descs = {
        {DML_BINDING_TYPE_BUFFER, &input_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[1]}};

    DML_BUFFER_BINDING output_binding = {gathered_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};

    gather_compiled_op->Execute(input_binding_descs, output_binding_desc);
  }

  // Step 6: Fill mask value tensor
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_mask_desc = {};
  fill_mask_desc.OutputTensor = &mask_value_tensor_desc;
  fill_mask_desc.ValueDataType = data_type;
  if constexpr (std::is_same_v<T, float>) {
    fill_mask_desc.Value.Float32 = _mask_value;
  } else {
    float16_t mask_val = static_cast<float16_t>(_mask_value);
    fill_mask_desc.Value.UInt16 = *reinterpret_cast<const uint16_t*>(&mask_val);
  }

  DML_OPERATOR_DESC fill_mask_op_desc = {};
  fill_mask_op_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
  fill_mask_op_desc.Desc = &fill_mask_desc;

  auto fill_mask_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&fill_mask_op_desc);

  // Execute fill mask value
  {
    DML_BUFFER_BINDING output_binding = {mask_value_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};

    fill_mask_compiled_op->Execute({}, output_binding_desc);
  }

  // Step 7: Apply conditional logic using IF operator
  // Convert mask from uint32 to float for IF condition
  DML_BUFFER_TENSOR_DESC mask_float_buffer_desc = input_buffer_desc;
  DML_TENSOR_DESC mask_float_tensor_desc = {};
  mask_float_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  mask_float_tensor_desc.Desc = &mask_float_buffer_desc;

  auto mask_float_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  DML_CAST_OPERATOR_DESC cast_desc = {};
  cast_desc.InputTensor = &mask_tensor_desc;
  cast_desc.OutputTensor = &mask_float_tensor_desc;

  DML_OPERATOR_DESC cast_op_desc = {};
  cast_op_desc.Type = DML_OPERATOR_CAST;
  cast_op_desc.Desc = &cast_desc;

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Execute cast
  {
    DML_BUFFER_BINDING input_binding = {mask_resource.Get(), 0,
                                        batch_size * depth * sizeof(uint32_t)};
    std::vector<DML_BINDING_DESC> input_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &input_binding}};
    DML_BUFFER_BINDING output_binding = {mask_float_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};
    cast_compiled_op->Execute(input_binding_desc, output_binding_desc);
  }

  // Apply IF condition: if (mask) then gathered_input else mask_value
  auto result_resource =
      device->CreatePreferredDeviceMemoryBuffer(tensor_size_bytes);

  DML_ELEMENT_WISE_IF_OPERATOR_DESC if_desc = {};
  if_desc.ConditionTensor = &mask_float_tensor_desc;
  if_desc.ATensor = &gathered_tensor_desc;
  if_desc.BTensor = &mask_value_tensor_desc;
  if_desc.OutputTensor =
      &gathered_tensor_desc;  // Reuse gathered tensor desc for result

  DML_OPERATOR_DESC if_op_desc = {};
  if_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IF;
  if_op_desc.Desc = &if_desc;

  auto if_compiled_op = dml::GetOrCreateCompiledOperatorApi(&if_op_desc);

  // Execute IF
  {
    DML_BUFFER_BINDING input_bindings[] = {
        {mask_float_resource.Get(), 0, tensor_size_bytes},
        {gathered_resource.Get(), 0, tensor_size_bytes},
        {mask_value_resource.Get(), 0, tensor_size_bytes}};
    std::vector<DML_BINDING_DESC> input_binding_descs = {
        {DML_BINDING_TYPE_BUFFER, &input_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[1]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[2]}};

    DML_BUFFER_BINDING output_binding = {result_resource.Get(), 0,
                                         tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};
    if_compiled_op->Execute(input_binding_descs, output_binding_desc);
  }

  // Step 8: Scatter result back to original positions
  DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
  scatter_desc.InputTensor = &output_tensor_desc;  // Initialize with zeros
  scatter_desc.IndicesTensor = &sorted_indices_tensor_desc;
  scatter_desc.UpdatesTensor = &gathered_tensor_desc;  // Use result from IF
  scatter_desc.OutputTensor = &output_tensor_desc;
  scatter_desc.Axis = 1;

  DML_OPERATOR_DESC scatter_op_desc = {};
  scatter_op_desc.Type = DML_OPERATOR_SCATTER_ELEMENTS;
  scatter_op_desc.Desc = &scatter_desc;

  auto scatter_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&scatter_op_desc);

  // First, zero the output buffer
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zero_fill_desc = {};
  zero_fill_desc.OutputTensor = &output_tensor_desc;
  zero_fill_desc.ValueDataType = data_type;
  if constexpr (std::is_same_v<T, float>) {
    zero_fill_desc.Value.Float32 = 0.0f;
  } else {
    zero_fill_desc.Value.UInt16 = 0;
  }

  DML_OPERATOR_DESC zero_fill_op_desc = {};
  zero_fill_op_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
  zero_fill_op_desc.Desc = &zero_fill_desc;

  auto zero_fill_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&zero_fill_op_desc);

  // Execute zero fill
  {
    DML_BUFFER_BINDING output_binding = {output_resource, 0, tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};
    zero_fill_compiled_op->Execute({}, output_binding_desc);
  }

  // Execute scatter
  {
    DML_BUFFER_BINDING input_bindings[] = {
        {output_resource, 0, tensor_size_bytes},
        {sorted_indices_resource.Get(), 0,
         batch_size * depth * sizeof(uint32_t)},
        {result_resource.Get(), 0, tensor_size_bytes}};
    std::vector<DML_BINDING_DESC> input_binding_descs = {
        {DML_BINDING_TYPE_BUFFER, &input_bindings[0]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[1]},
        {DML_BINDING_TYPE_BUFFER, &input_bindings[2]}};

    DML_BUFFER_BINDING output_binding = {output_resource, 0, tensor_size_bytes};
    std::vector<DML_BINDING_DESC> output_binding_desc = {
        {DML_BINDING_TYPE_BUFFER, &output_binding}};
    scatter_compiled_op->Execute(input_binding_descs, output_binding_desc);
  }

  // Execute all commands
  device->ExecuteCommandList();
}

template <>
dim_t TopPMask::max_num_classes<Device::DirectML>() {
  return 32768;  // Conservative limit for DirectML operations
}

#define DECLARE_IMPL(T)                                 \
  template void TopPMask::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, StorageView&) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif
