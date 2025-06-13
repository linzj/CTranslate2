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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& probs_desc = op_desc.AddInput(probs);
    auto& sorted_probs_desc = op_desc.AddOutput(sorted_probs);
    auto& sorted_indices_desc = op_desc.AddOutput(sorted_indices);
    auto& topk_desc = op_desc.GetOperatorDesc<DML_TOP_K1_OPERATOR_DESC>();
    topk_desc.InputTensor = &probs_desc.get_tensor_desc();
    topk_desc.OutputValueTensor = &sorted_probs_desc.get_tensor_desc();
    topk_desc.OutputIndexTensor = &sorted_indices_desc.get_tensor_desc();
    topk_desc.Axis = 1;
    topk_desc.K = static_cast<UINT>(depth);
    topk_desc.AxisDirection = DML_AXIS_DIRECTION_DECREASING;
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& sorted_probs_desc = op_desc.AddInput(sorted_probs);
    auto& cumsum_desc = op_desc.AddOutput(cumsum);
    auto& cumsum_op_desc =
        op_desc.GetOperatorDesc<DML_CUMULATIVE_SUMMATION_OPERATOR_DESC>();
    cumsum_op_desc.InputTensor = &sorted_probs_desc.get_tensor_desc();
    cumsum_op_desc.OutputTensor = &cumsum_desc.get_tensor_desc();
    cumsum_op_desc.Axis = 1;
    cumsum_op_desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
    cumsum_op_desc.HasExclusiveSum = TRUE;
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& cumsum_desc = op_desc.AddInput(cumsum);
    auto& threshold_desc = op_desc.AddInput(threshold);
    auto& mask_desc = op_desc.AddOutput(mask);
    auto& op_payload = op_desc.GetOperatorDesc<
        DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>();
    op_payload.ATensor = &cumsum_desc.get_tensor_desc();
    op_payload.BTensor = &threshold_desc.get_tensor_desc();
    op_payload.OutputTensor = &mask_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& input_desc = op_desc.AddInput(input);
    auto& sorted_indices_desc = op_desc.AddInput(sorted_indices);
    auto& gathered_desc = op_desc.AddOutput(gathered_input);
    auto& op_payload =
        op_desc.GetOperatorDesc<DML_GATHER_ELEMENTS_OPERATOR_DESC>();
    op_payload.InputTensor = &input_desc.get_tensor_desc();
    op_payload.IndicesTensor = &sorted_indices_desc.get_tensor_desc();
    op_payload.OutputTensor = &gathered_desc.get_tensor_desc();
    op_payload.Axis = 1;
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& mask_desc = op_desc.AddInput(mask);
    auto& gathered_desc = op_desc.AddInput(gathered_input);
    auto& mask_val_desc = op_desc.AddInput(mask_value);
    auto& masked_gathered_desc = op_desc.AddOutput(masked_gathered);
    auto& op_payload =
        op_desc.GetOperatorDesc<DML_ELEMENT_WISE_IF_OPERATOR_DESC>();
    op_payload.ConditionTensor = &mask_desc.get_tensor_desc();
    op_payload.ATensor = &gathered_desc.get_tensor_desc();
    op_payload.BTensor = &mask_val_desc.get_tensor_desc();
    op_payload.OutputTensor = &masked_gathered_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& output_desc = op_desc.AddOutput(output);
    auto& sorted_indices_desc = op_desc.AddInput(sorted_indices);
    auto& masked_gathered_desc = op_desc.AddInput(masked_gathered);
    auto& op_payload =
        op_desc.GetOperatorDesc<DML_SCATTER_ELEMENTS_OPERATOR_DESC>();
    op_payload.InputTensor = &output_desc.get_tensor_desc();
    op_payload.IndicesTensor = &sorted_indices_desc.get_tensor_desc();
    op_payload.UpdatesTensor = &masked_gathered_desc.get_tensor_desc();
    op_payload.OutputTensor = &output_desc.get_tensor_desc();
    op_payload.Axis = 1;
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

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
