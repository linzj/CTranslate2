#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/topp_mask.h"
#include "ctranslate2/types.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
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

  auto* ct2_dml_device =
      dml::get_device();  // Renamed from 'device' to avoid conflict
  auto* dml_device = dml::get_dml_device();

  auto* input_d3d_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto* probs_d3d_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(probs.buffer()));
  auto* output_d3d_resource =
      reinterpret_cast<ID3D12Resource*>(output.buffer());

  DataType main_ct2_dtype = input.dtype();  // T is float or float16
  DataType indices_ct2_dtype =
      DataType::INT32;  // DML needs UINT32, util handles mapping
  DataType mask_ct2_dtype =
      DataType::INT32;  // DML uses UINT32 for bools, util handles

  std::vector<UINT> common_dims = {static_cast<UINT>(batch_size),
                                   static_cast<UINT>(depth)};

  // Tensor Descriptors using DmlTensorDescBundle
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, input.size() * sizeof(T));
  const DML_TENSOR_DESC& dml_input_tensor_desc =
      input_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle probs_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, probs.size() * sizeof(T));
  const DML_TENSOR_DESC& dml_probs_tensor_desc =
      probs_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle output_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, output.size() * sizeof(T));
  const DML_TENSOR_DESC& dml_output_tensor_desc =
      output_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle sorted_probs_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_sorted_probs_tensor_desc =
      sorted_probs_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle sorted_indices_desc_bundle(
      indices_ct2_dtype, common_dims, nullptr,
      batch_size * depth * sizeof(int32_t));
  const DML_TENSOR_DESC& dml_sorted_indices_tensor_desc =
      sorted_indices_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle cumsum_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_cumsum_tensor_desc =
      cumsum_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle threshold_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_threshold_tensor_desc =
      threshold_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle mask_desc_bundle(
      mask_ct2_dtype, common_dims, nullptr,
      batch_size * depth * sizeof(int32_t));
  const DML_TENSOR_DESC& dml_mask_tensor_desc =
      mask_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle gathered_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_gathered_tensor_desc =
      gathered_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle mask_value_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_mask_value_tensor_desc =
      mask_value_desc_bundle.get_tensor_desc();

  dml::utils::DmlTensorDescBundle mask_float_desc_bundle(
      main_ct2_dtype, common_dims, nullptr,
      batch_size * depth * sizeof(T));  // For cast output
  const DML_TENSOR_DESC& dml_mask_float_tensor_desc =
      mask_float_desc_bundle.get_tensor_desc();

  // Create intermediate resources using sizes from bundles
  auto sorted_probs_resource =
      ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
          sorted_probs_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto sorted_indices_resource =
      ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
          sorted_indices_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto cumsum_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      cumsum_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto threshold_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      threshold_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto mask_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      mask_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto gathered_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      gathered_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto mask_value_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      mask_value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto mask_float_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      mask_float_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  auto result_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  ct2_dml_device->ResetCommandList();

  // Step 1: TopK
  DML_TOP_K1_OPERATOR_DESC topk_desc_payload = {};
  topk_desc_payload.InputTensor = &dml_probs_tensor_desc;
  topk_desc_payload.OutputValueTensor = &dml_sorted_probs_tensor_desc;
  topk_desc_payload.OutputIndexTensor = &dml_sorted_indices_tensor_desc;
  topk_desc_payload.Axis = 1;
  topk_desc_payload.K = static_cast<UINT>(depth);
  topk_desc_payload.AxisDirection = DML_AXIS_DIRECTION_DECREASING;
  DML_OPERATOR_DESC topk_op_desc = {DML_OPERATOR_TOP_K1, &topk_desc_payload};
  auto topk_compiled_op = dml::GetOrCreateCompiledOperatorApi(&topk_op_desc);
  {
    DML_BUFFER_BINDING tk_input_s = dml::utils::create_buffer_binding(
        probs_d3d_resource, 0,
        probs_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC tk_input_d = dml::utils::create_binding_desc(&tk_input_s);
    DML_BUFFER_BINDING tk_outputs_s[] = {
        dml::utils::create_buffer_binding(
            sorted_probs_resource.Get(), 0,
            sorted_probs_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            sorted_indices_resource.Get(), 0,
            sorted_indices_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes)};
    std::vector<DML_BINDING_DESC> tk_outputs_d = {
        dml::utils::create_binding_desc(&tk_outputs_s[0]),
        dml::utils::create_binding_desc(&tk_outputs_s[1])};
    topk_compiled_op->Execute({tk_input_d}, tk_outputs_d);
  }

  // Step 2: Cumulative sum
  DML_CUMULATIVE_SUMMATION_OPERATOR_DESC cumsum_desc_payload = {};
  cumsum_desc_payload.InputTensor = &dml_sorted_probs_tensor_desc;
  cumsum_desc_payload.OutputTensor = &dml_cumsum_tensor_desc;
  cumsum_desc_payload.Axis = 1;
  cumsum_desc_payload.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
  cumsum_desc_payload.HasExclusiveSum = TRUE;
  DML_OPERATOR_DESC cumsum_op_desc = {DML_OPERATOR_CUMULATIVE_SUMMATION,
                                      &cumsum_desc_payload};
  auto cumsum_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&cumsum_op_desc);
  {
    DML_BUFFER_BINDING cs_input_s = dml::utils::create_buffer_binding(
        sorted_probs_resource.Get(), 0,
        sorted_probs_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC cs_input_d = dml::utils::create_binding_desc(&cs_input_s);
    DML_BUFFER_BINDING cs_output_s = dml::utils::create_buffer_binding(
        cumsum_resource.Get(), 0,
        cumsum_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC cs_output_d =
        dml::utils::create_binding_desc(&cs_output_s);
    cumsum_compiled_op->Execute({cs_input_d}, {cs_output_d});
  }

  // Step 3: Fill threshold tensor
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_threshold_desc_payload = {};
  fill_threshold_desc_payload.OutputTensor = &dml_threshold_tensor_desc;
  fill_threshold_desc_payload.ValueDataType =
      dml::utils::get_dml_data_type(main_ct2_dtype);
  if constexpr (std::is_same_v<T, float>) {
    fill_threshold_desc_payload.Value.Float32 = _p;
  } else {
    float16_t p_val = static_cast<float16_t>(_p);
    fill_threshold_desc_payload.Value.UInt16 =
        *reinterpret_cast<const uint16_t*>(&p_val);
  }
  DML_OPERATOR_DESC fill_threshold_op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                              &fill_threshold_desc_payload};
  auto fill_threshold_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&fill_threshold_op_desc);
  {
    DML_BUFFER_BINDING ft_output_s = dml::utils::create_buffer_binding(
        threshold_resource.Get(), 0,
        threshold_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC ft_output_d =
        dml::utils::create_binding_desc(&ft_output_s);
    fill_threshold_compiled_op->Execute({}, {ft_output_d});
  }

  // Step 4: Compare cumsum < threshold
  DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC compare_desc_payload = {};
  compare_desc_payload.ATensor = &dml_cumsum_tensor_desc;
  compare_desc_payload.BTensor = &dml_threshold_tensor_desc;
  compare_desc_payload.OutputTensor = &dml_mask_tensor_desc;
  DML_OPERATOR_DESC compare_op_desc = {
      DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN, &compare_desc_payload};
  auto compare_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&compare_op_desc);
  {
    DML_BUFFER_BINDING cmp_inputs_s[] = {
        dml::utils::create_buffer_binding(
            cumsum_resource.Get(), 0,
            cumsum_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            threshold_resource.Get(), 0,
            threshold_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes)};
    std::vector<DML_BINDING_DESC> cmp_inputs_d = {
        dml::utils::create_binding_desc(&cmp_inputs_s[0]),
        dml::utils::create_binding_desc(&cmp_inputs_s[1])};
    DML_BUFFER_BINDING cmp_output_s = dml::utils::create_buffer_binding(
        mask_resource.Get(), 0,
        mask_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC cmp_output_d =
        dml::utils::create_binding_desc(&cmp_output_s);
    compare_compiled_op->Execute(cmp_inputs_d, {cmp_output_d});
  }

  // Step 5: Gather input values
  DML_GATHER_ELEMENTS_OPERATOR_DESC gather_desc_payload = {};
  gather_desc_payload.InputTensor = &dml_input_tensor_desc;
  gather_desc_payload.IndicesTensor = &dml_sorted_indices_tensor_desc;
  gather_desc_payload.OutputTensor = &dml_gathered_tensor_desc;
  gather_desc_payload.Axis = 1;
  DML_OPERATOR_DESC gather_op_desc = {DML_OPERATOR_GATHER_ELEMENTS,
                                      &gather_desc_payload};
  auto gather_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&gather_op_desc);
  {
    DML_BUFFER_BINDING gth_inputs_s[] = {
        dml::utils::create_buffer_binding(
            input_d3d_resource, 0,
            input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            sorted_indices_resource.Get(), 0,
            sorted_indices_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes)};
    std::vector<DML_BINDING_DESC> gth_inputs_d = {
        dml::utils::create_binding_desc(&gth_inputs_s[0]),
        dml::utils::create_binding_desc(&gth_inputs_s[1])};
    DML_BUFFER_BINDING gth_output_s = dml::utils::create_buffer_binding(
        gathered_resource.Get(), 0,
        gathered_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC gth_output_d =
        dml::utils::create_binding_desc(&gth_output_s);
    gather_compiled_op->Execute(gth_inputs_d, {gth_output_d});
  }

  // Step 6: Fill mask value tensor
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_mask_desc_payload = {};
  fill_mask_desc_payload.OutputTensor = &dml_mask_value_tensor_desc;
  fill_mask_desc_payload.ValueDataType =
      dml::utils::get_dml_data_type(main_ct2_dtype);
  if constexpr (std::is_same_v<T, float>) {
    fill_mask_desc_payload.Value.Float32 = _mask_value;
  } else {
    float16_t mask_val = static_cast<float16_t>(_mask_value);
    fill_mask_desc_payload.Value.UInt16 =
        *reinterpret_cast<const uint16_t*>(&mask_val);
  }
  DML_OPERATOR_DESC fill_mask_op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                         &fill_mask_desc_payload};
  auto fill_mask_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&fill_mask_op_desc);
  {
    DML_BUFFER_BINDING fm_output_s = dml::utils::create_buffer_binding(
        mask_value_resource.Get(), 0,
        mask_value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC fm_output_d =
        dml::utils::create_binding_desc(&fm_output_s);
    fill_mask_compiled_op->Execute({}, {fm_output_d});
  }

  // Step 7a: Cast mask from UINT32 to Float (main_ct2_dtype) for IF
  DML_CAST_OPERATOR_DESC cast_desc_payload = {};
  cast_desc_payload.InputTensor = &dml_mask_tensor_desc;
  cast_desc_payload.OutputTensor = &dml_mask_float_tensor_desc;
  DML_OPERATOR_DESC cast_op_desc = {DML_OPERATOR_CAST, &cast_desc_payload};
  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);
  {
    DML_BUFFER_BINDING cst_input_s = dml::utils::create_buffer_binding(
        mask_resource.Get(), 0,
        mask_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC cst_input_d =
        dml::utils::create_binding_desc(&cst_input_s);
    DML_BUFFER_BINDING cst_output_s = dml::utils::create_buffer_binding(
        mask_float_resource.Get(), 0,
        mask_float_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC cst_output_d =
        dml::utils::create_binding_desc(&cst_output_s);
    cast_compiled_op->Execute({cst_input_d}, {cst_output_d});
  }

  // Step 7b: Apply IF condition
  // Reusing dml_gathered_tensor_desc for IF's OutputTensor is potentially
  // problematic if it implies an in-place update that DML_SCATTER later needs.
  // It's safer to have IF write to `result_resource`.
  // DmlTensorDescBundle for `result_resource` (same spec as gathered)
  dml::utils::DmlTensorDescBundle result_if_desc_bundle(
      main_ct2_dtype, common_dims, nullptr, batch_size * depth * sizeof(T));
  const DML_TENSOR_DESC& dml_result_if_tensor_desc =
      result_if_desc_bundle.get_tensor_desc();

  DML_ELEMENT_WISE_IF_OPERATOR_DESC if_desc_payload = {};
  if_desc_payload.ConditionTensor = &dml_mask_float_tensor_desc;
  if_desc_payload.ATensor = &dml_gathered_tensor_desc;    // Values if true
  if_desc_payload.BTensor = &dml_mask_value_tensor_desc;  // Values if false
  if_desc_payload.OutputTensor =
      &dml_result_if_tensor_desc;  // Output to result_resource
  DML_OPERATOR_DESC if_op_desc = {DML_OPERATOR_ELEMENT_WISE_IF,
                                  &if_desc_payload};
  auto if_compiled_op = dml::GetOrCreateCompiledOperatorApi(&if_op_desc);
  {
    DML_BUFFER_BINDING if_inputs_s[] = {
        dml::utils::create_buffer_binding(
            mask_float_resource.Get(), 0,
            mask_float_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            gathered_resource.Get(), 0,
            gathered_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            mask_value_resource.Get(), 0,
            mask_value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes)};
    std::vector<DML_BINDING_DESC> if_inputs_d = {
        dml::utils::create_binding_desc(&if_inputs_s[0]),
        dml::utils::create_binding_desc(&if_inputs_s[1]),
        dml::utils::create_binding_desc(&if_inputs_s[2])};
    DML_BUFFER_BINDING if_output_s = dml::utils::create_buffer_binding(
        result_resource.Get(), 0,
        result_if_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC if_output_d =
        dml::utils::create_binding_desc(&if_output_s);
    if_compiled_op->Execute(if_inputs_d, {if_output_d});
  }

  // Step 8: Scatter result back
  // First, zero the output buffer that scatter will use as its base
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zero_fill_desc_payload = {};
  zero_fill_desc_payload.OutputTensor = &dml_output_tensor_desc;
  zero_fill_desc_payload.ValueDataType =
      dml::utils::get_dml_data_type(main_ct2_dtype);
  if constexpr (std::is_same_v<T, float>) {
    zero_fill_desc_payload.Value.Float32 = 0.0f;
  } else {
    zero_fill_desc_payload.Value.UInt16 = 0;  // float16 zero
  }
  DML_OPERATOR_DESC zero_fill_op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                         &zero_fill_desc_payload};
  auto zero_fill_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&zero_fill_op_desc);
  {
    DML_BUFFER_BINDING zf_output_s = dml::utils::create_buffer_binding(
        output_d3d_resource, 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC zf_output_d =
        dml::utils::create_binding_desc(&zf_output_s);
    zero_fill_compiled_op->Execute({}, {zf_output_d});
  }

  // Now scatter the results from IF op
  DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc_payload = {};
  scatter_desc_payload.InputTensor =
      &dml_output_tensor_desc;  // Use zeroed output as base
  scatter_desc_payload.IndicesTensor = &dml_sorted_indices_tensor_desc;
  scatter_desc_payload.UpdatesTensor =
      &dml_result_if_tensor_desc;  // Updates come from IF output
  scatter_desc_payload.OutputTensor =
      &dml_output_tensor_desc;  // Write to final output
  scatter_desc_payload.Axis = 1;
  DML_OPERATOR_DESC scatter_op_desc = {DML_OPERATOR_SCATTER_ELEMENTS,
                                       &scatter_desc_payload};
  auto scatter_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&scatter_op_desc);
  {
    DML_BUFFER_BINDING sc_inputs_s[] = {
        dml::utils::create_buffer_binding(
            output_d3d_resource, 0,
            output_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes),  // Input for scatter (already zeroed)
        dml::utils::create_buffer_binding(
            sorted_indices_resource.Get(), 0,
            sorted_indices_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes),
        dml::utils::create_buffer_binding(
            result_resource.Get(), 0,
            result_if_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes)  // Updates from IF
    };
    std::vector<DML_BINDING_DESC> sc_inputs_d = {
        dml::utils::create_binding_desc(&sc_inputs_s[0]),
        dml::utils::create_binding_desc(&sc_inputs_s[1]),
        dml::utils::create_binding_desc(&sc_inputs_s[2])};
    DML_BUFFER_BINDING sc_output_s = dml::utils::create_buffer_binding(
        output_d3d_resource, 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC sc_output_d =
        dml::utils::create_binding_desc(&sc_output_s);
    scatter_compiled_op->Execute(sc_inputs_d, {sc_output_d});
  }

  ct2_dml_device->ExecuteCommandList();
}

template <>
dim_t TopPMask::max_num_classes<Device::DirectML>() {
  return 32768;
}

#define DECLARE_IMPL(T)                                 \
  template void TopPMask::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, StorageView&) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif
