#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/topp_mask.h"
#include "ctranslate2/types.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
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

  auto* ct2_dml_device = dml::get_device();
  ct2_dml_device->ResetCommandList();

  const auto device = Device::DirectML;
  const auto float_type = DataType::FLOAT32;
  const auto uint_type = DataType::INT32;  // For UINT32 size compatibility

  Shape common_shape({batch_size, depth});

  // Intermediate resources
  StorageView sorted_probs(common_shape, float_type, device);
  StorageView sorted_indices(common_shape, uint_type, device);
  StorageView cumsum(common_shape, float_type, device);
  StorageView threshold(common_shape, float_type, device);
  StorageView mask(common_shape, uint_type, device);
  StorageView gathered_input(common_shape, float_type, device);
  StorageView mask_value(common_shape, float_type, device);
  StorageView masked_gathered(common_shape, float_type, device);

  // Step 1: TopK to sort probabilities
  {
    dml::utils::DmlTensorDescBundle probs_desc(probs);
    dml::utils::DmlTensorDescBundle sorted_probs_desc(sorted_probs);
    dml::utils::DmlTensorDescBundle sorted_indices_desc(sorted_indices);

    DML_TOP_K1_OPERATOR_DESC topk_desc = {};
    topk_desc.InputTensor = &probs_desc.get_tensor_desc();
    topk_desc.OutputValueTensor = &sorted_probs_desc.get_tensor_desc();
    topk_desc.OutputIndexTensor = &sorted_indices_desc.get_tensor_desc();
    topk_desc.Axis = 1;
    topk_desc.K = static_cast<UINT>(depth);
    topk_desc.AxisDirection = DML_AXIS_DIRECTION_DECREASING;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_TOP_K1, &topk_desc};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(probs))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(sorted_probs)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(sorted_indices))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Step 2: Cumulative sum
  {
    dml::utils::DmlTensorDescBundle sorted_probs_desc(sorted_probs);
    dml::utils::DmlTensorDescBundle cumsum_desc(cumsum);
    DML_CUMULATIVE_SUMMATION_OPERATOR_DESC cumsum_op_desc = {};
    cumsum_op_desc.InputTensor = &sorted_probs_desc.get_tensor_desc();
    cumsum_op_desc.OutputTensor = &cumsum_desc.get_tensor_desc();
    cumsum_op_desc.Axis = 1;
    cumsum_op_desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
    cumsum_op_desc.HasExclusiveSum = TRUE;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CUMULATIVE_SUMMATION,
                                 &cumsum_op_desc};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(sorted_probs))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(cumsum))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Step 3: Compare cumsum < p
  {
    threshold.fill(static_cast<T>(_p));
    dml::utils::DmlTensorDescBundle cumsum_desc(cumsum);
    dml::utils::DmlTensorDescBundle threshold_desc(threshold);
    dml::utils::DmlTensorDescBundle mask_desc(mask);

    DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC op_payload = {};
    op_payload.ATensor = &cumsum_desc.get_tensor_desc();
    op_payload.BTensor = &threshold_desc.get_tensor_desc();
    op_payload.OutputTensor = &mask_desc.get_tensor_desc();

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN,
                                 &op_payload};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(cumsum)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(threshold))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(mask))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Step 4: Gather original input values in sorted order
  {
    dml::utils::DmlTensorDescBundle input_desc(input);
    dml::utils::DmlTensorDescBundle sorted_indices_desc(sorted_indices);
    dml::utils::DmlTensorDescBundle gathered_desc(gathered_input);
    DML_GATHER_ELEMENTS_OPERATOR_DESC op_payload = {};
    op_payload.InputTensor = &input_desc.get_tensor_desc();
    op_payload.IndicesTensor = &sorted_indices_desc.get_tensor_desc();
    op_payload.OutputTensor = &gathered_desc.get_tensor_desc();
    op_payload.Axis = 1;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GATHER_ELEMENTS, &op_payload};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(input)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(sorted_indices))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(gathered_input))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Step 5: Select between gathered input and mask value based on the top-p
  // mask
  {
    mask_value.fill(static_cast<T>(_mask_value));
    dml::utils::DmlTensorDescBundle mask_desc(mask);
    dml::utils::DmlTensorDescBundle gathered_desc(gathered_input);
    dml::utils::DmlTensorDescBundle mask_val_desc(mask_value);
    dml::utils::DmlTensorDescBundle masked_gathered_desc(masked_gathered);
    DML_ELEMENT_WISE_IF_OPERATOR_DESC op_payload = {};
    op_payload.ConditionTensor = &mask_desc.get_tensor_desc();
    op_payload.ATensor = &gathered_desc.get_tensor_desc();
    op_payload.BTensor = &mask_val_desc.get_tensor_desc();
    op_payload.OutputTensor = &masked_gathered_desc.get_tensor_desc();
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IF, &op_payload};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(mask)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(gathered_input)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(mask_value))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(masked_gathered))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Step 6: Scatter results back to original positions
  {
    output.zero();
    dml::utils::DmlTensorDescBundle output_desc(output);
    dml::utils::DmlTensorDescBundle sorted_indices_desc(sorted_indices);
    dml::utils::DmlTensorDescBundle masked_gathered_desc(masked_gathered);

    DML_SCATTER_ELEMENTS_OPERATOR_DESC op_payload = {};
    op_payload.InputTensor = &output_desc.get_tensor_desc();
    op_payload.IndicesTensor = &sorted_indices_desc.get_tensor_desc();
    op_payload.UpdatesTensor = &masked_gathered_desc.get_tensor_desc();
    op_payload.OutputTensor = &output_desc.get_tensor_desc();
    op_payload.Axis = 1;

    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SCATTER_ELEMENTS, &op_payload};
    auto* op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(output)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(sorted_indices)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(masked_gathered))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
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
