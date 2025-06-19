#include "ctranslate2/ops/multinomial.h"

#ifdef CT2_WITH_DIRECTML

#include <limits>  // For std::numeric_limits
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"

// Ensure THROW_INVALID_ARGUMENT is available (should be via dml_utils.h ->
// utils.h)
#ifndef THROW_INVALID_ARGUMENT
#define THROW_INVALID_ARGUMENT(msg) throw std::invalid_argument(msg)
#endif

namespace ctranslate2 {
namespace ops {
namespace dml_internal {

// Helper function containing the core DML logic for multinomial.
void multinomial_impl(const StorageView& probs_input,
                      StorageView& output_indices,
                      const Multinomial& op_params) {
  const dim_t depth = probs_input.dim(-1);  // class_size
  const dim_t batch_size = probs_input.size() / depth;
  auto device = dml::get_device();

  output_indices.resize({batch_size});

  // --- Handle input cast to FLOAT32 if necessary ---
  StorageView
      probs_f32_sv(  // Temporary StorageView for type/shape metadata if casting
          DataType::FLOAT32,
          device->GetCommandListType() == D3D12_COMMAND_LIST_TYPE_COPY
              ? Device::CPU
              : Device::DirectML);
  ID3D12Resource* current_probs_resource_ptr;
  const StorageView* probs_sv_for_ops_ptr;

  // Hold all operator bundles to manage tensor descriptor lifetimes.
  std::vector<dml::utils::DmlOperatorDescBundle> op_desc_bundles;

  if (probs_input.dtype() != DataType::FLOAT32) {
    probs_f32_sv.resize(probs_input.shape());

    op_desc_bundles.emplace_back();
    auto& cast_op_bundle = op_desc_bundles.back();
    const auto& input_desc = cast_op_bundle.AddInput(probs_input);
    const auto& output_desc = cast_op_bundle.AddOutput(probs_f32_sv);

    auto& cast_desc = cast_op_bundle.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
    cast_desc.InputTensor = &input_desc.get_tensor_desc();
    cast_desc.OutputTensor = &output_desc.get_tensor_desc();

    dml::Operator* cast_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(cast_op_bundle));

    dml::utils::DmlBindingArrayBundle cast_inputs{
        {dml::utils::ResourceFromStorageView(probs_input), 0,
         input_desc.get_buffer_desc().TotalTensorSizeInBytes}};
    dml::utils::DmlBindingArrayBundle cast_outputs{
        {dml::utils::ResourceFromStorageView(probs_f32_sv), 0,
         output_desc.get_buffer_desc().TotalTensorSizeInBytes}};

    cast_op->Execute(cast_inputs, cast_outputs);

    current_probs_resource_ptr =
        dml::utils::ResourceFromStorageView(probs_f32_sv);
    probs_sv_for_ops_ptr = &probs_f32_sv;

  } else {
    // No cast needed, use original probs_input directly
    current_probs_resource_ptr =
        dml::utils::ResourceFromStorageView(probs_input);
    probs_sv_for_ops_ptr = &probs_input;
  }

  // --- Prepare DML Tensor Descriptors for FLOAT32 tensors ---
  // (either original or casted to FLOAT32)

  op_desc_bundles.emplace_back();
  auto& random_op_bundle = op_desc_bundles.back();

  Shape random_shape_ct2 = {batch_size, 1};
  std::vector<UINT> random_dml_dims =
      dml::utils::to_dml_dims(random_shape_ct2, batch_size, true);
  const auto& random_desc_bundle = random_op_bundle.AddInputBroadcastFromSeach(
      *probs_sv_for_ops_ptr, random_dml_dims);

  StorageView random_numbers_sv(random_shape_ct2, DataType::FLOAT32,
                                Device::DirectML);

  op_desc_bundles.emplace_back();
  auto& philox_op_bundle = op_desc_bundles.back();

  Shape philox_state_dims_shape_ct2 = {4};
  const auto& philox_state_tensor_desc_bundle = philox_op_bundle.AddInput(
      DML_TENSOR_DATA_TYPE_UINT32,
      dml::utils::to_dml_dims(philox_state_dims_shape_ct2, 4, true), nullptr);

  StorageView state_in_sv(philox_state_dims_shape_ct2,
                          DataType::INT32,  // Corresponds to UINT32 for size
                          Device::DirectML);
  {  // Zero Init Philox State
    op_desc_bundles.emplace_back();
    auto& fill_zero_op_bundle = op_desc_bundles.back();

    auto& fill_zero_desc =
        fill_zero_op_bundle
            .GetOperatorDesc<DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>();
    fill_zero_desc.OutputTensor =
        &philox_state_tensor_desc_bundle
             .get_tensor_desc();  // Describes state_in_res
    fill_zero_desc.ValueDataType = DML_TENSOR_DATA_TYPE_UINT32;
    fill_zero_desc.Value.UInt32 = 0;

    dml::Operator* fill_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(fill_zero_op_bundle));
    dml::utils::DmlBufferBindingBundle fill_out_b_storage(
        dml::utils::ResourceFromStorageView(state_in_sv));
    // Fill state_in_sv
    dml::utils::DmlBindingArrayBundle fill_outputs{
        {dml::utils::ResourceFromStorageView(state_in_sv), 0, 0}};
    fill_op->Execute({}, fill_outputs);
  }
  StorageView state_out_sv(philox_state_dims_shape_ct2,
                           DataType::INT32,  // Corresponds to UINT32 for size
                           Device::DirectML);

  op_desc_bundles.emplace_back();
  auto& cumsum_op_bundle = op_desc_bundles.back();
  Shape cumsum_shape_ct2 = {batch_size, depth};
  const auto& cumsum_desc_bundle = cumsum_op_bundle.AddOutput(
      DataType::FLOAT32,
      dml::utils::to_dml_dims(cumsum_shape_ct2, batch_size * depth, true),
      nullptr);
  StorageView cumsum_sv(cumsum_shape_ct2, DataType::FLOAT32, Device::DirectML);

  // DML logical ops output UINT8.
  op_desc_bundles.emplace_back();
  auto& compare_op_bundle = op_desc_bundles.back();
  Shape compare_shape_ct2 = {batch_size, depth};
  const auto& compare_tensor_desc_bundle = compare_op_bundle.AddOutput(
      DML_TENSOR_DATA_TYPE_UINT8,
      dml::utils::to_dml_dims(compare_shape_ct2, batch_size * depth, true),
      nullptr);
  StorageView compare_sv(compare_shape_ct2,
                         DataType::INT8,  // Corresponds to UINT8 for size
                         Device::DirectML);

  op_desc_bundles.emplace_back();
  auto& iota_op_bundle = op_desc_bundles.back();
  Shape iota_shape_ct2 = {1, depth};
  std::vector<UINT> iota_strides_vec = {0, 1};  // Broadcast batch dim
  const auto& iota_desc_bundle = iota_op_bundle.AddOutput(
      DataType::FLOAT32, dml::utils::to_dml_dims(iota_shape_ct2, depth, true),
      &iota_strides_vec);
  StorageView iota_sv(iota_shape_ct2, DataType::FLOAT32, Device::DirectML);

  op_desc_bundles.emplace_back();
  auto& max_val_op_bundle = op_desc_bundles.back();
  Shape scalar_shape_ct2 = {1};
  std::vector<UINT> scalar_strides_vec = {0};  // Broadcast
  const auto& max_val_desc_bundle = max_val_op_bundle.AddOutput(
      DataType::FLOAT32, dml::utils::to_dml_dims(scalar_shape_ct2, 1, true),
      &scalar_strides_vec);
  StorageView max_val_sv({1}, DataType::FLOAT32, Device::DirectML);

  op_desc_bundles.emplace_back();
  auto& argmin_op_bundle = op_desc_bundles.back();
  Shape argmin_input_shape_ct2 = {batch_size, depth};
  const auto& argmin_input_desc_bundle = argmin_op_bundle.AddOutput(
      DataType::FLOAT32,
      dml::utils::to_dml_dims(argmin_input_shape_ct2, batch_size * depth, true),
      nullptr);
  StorageView argmin_input_sv(argmin_input_shape_ct2, DataType::FLOAT32,
                              Device::DirectML);

  // --- Define and Execute DML Operators ---

  // Op 1: Random Generator
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    auto& desc =
        op_bundle.GetOperatorDesc<DML_RANDOM_GENERATOR_OPERATOR_DESC>();
    desc.InputStateTensor = &philox_state_tensor_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &random_desc_bundle.get_tensor_desc();
    desc.OutputStateTensor = &philox_state_tensor_desc_bundle.get_tensor_desc();
    desc.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;

    op_bundle.AddInput(philox_state_tensor_desc_bundle);
    op_bundle.AddOutput(random_desc_bundle);
    op_bundle.AddOutput(philox_state_tensor_desc_bundle);

    dml::Operator* op =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle inputs{
        {dml::utils::ResourceFromStorageView(state_in_sv), 0, 0}};
    dml::utils::DmlBindingArrayBundle outputs{
        {dml::utils::ResourceFromStorageView(random_numbers_sv), 0, 0},
        {dml::utils::ResourceFromStorageView(state_out_sv), 0, 0}};
    op->Execute(inputs, outputs);
  }

  // Op 2: Cumulative Sum
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    const auto& input_desc = op_bundle.AddInput(*probs_sv_for_ops_ptr);
    op_bundle.AddOutput(cumsum_desc_bundle);

    auto& desc =
        op_bundle.GetOperatorDesc<DML_CUMULATIVE_SUMMATION_OPERATOR_DESC>();
    desc.InputTensor = &input_desc.get_tensor_desc();
    desc.OutputTensor = &cumsum_desc_bundle.get_tensor_desc();
    desc.Axis = input_desc.get_buffer_desc().DimensionCount - 1;
    desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
    desc.HasExclusiveSum = FALSE;

    dml::Operator* op =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle input_binding{
        {current_probs_resource_ptr, 0, 0}};
    dml::utils::DmlBindingArrayBundle output_binding{
        {dml::utils::ResourceFromStorageView(cumsum_sv), 0, 0}};
    op->Execute(input_binding, output_binding);
  }

  // Op 3: Compare (CumulativeProbs >= RandomSample), output should be UINT8
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    op_bundle.AddInput(cumsum_desc_bundle);
    op_bundle.AddInput(random_desc_bundle);
    op_bundle.AddOutput(compare_tensor_desc_bundle);

    auto& desc = op_bundle.GetOperatorDesc<
        DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC>();
    desc.ATensor = &cumsum_desc_bundle.get_tensor_desc();
    desc.BTensor = &random_desc_bundle.get_tensor_desc();
    desc.OutputTensor =
        &compare_tensor_desc_bundle
             .get_tensor_desc();  // compare_tensor_desc_bundle is
                                  // DML_TENSOR_DATA_TYPE_UINT8

    dml::Operator* op =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle inputs{
        {dml::utils::ResourceFromStorageView(cumsum_sv), 0, 0},
        {dml::utils::ResourceFromStorageView(random_numbers_sv), 0, 0}};
    dml::utils::DmlBindingArrayBundle outputs{
        {dml::utils::ResourceFromStorageView(compare_sv), 0, 0}};
    op->Execute(inputs, outputs);
  }

  // Op 4a: Fill Iota Tensor
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    op_bundle.AddOutput(iota_desc_bundle);

    auto& desc =
        op_bundle.GetOperatorDesc<DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC>();
    desc.OutputTensor = &iota_desc_bundle.get_tensor_desc();
    desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc.ValueStart.Float32 = 0.0f;
    desc.ValueDelta.Float32 = 1.0f;

    dml::Operator* op_seq =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle iota_binding{
        {dml::utils::ResourceFromStorageView(iota_sv), 0, 0}};
    op_seq->Execute({}, iota_binding);
  }

  // Op 4b: Fill Max Value Scalar Tensor
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    op_bundle.AddOutput(max_val_desc_bundle);

    auto& desc_fill_max =
        op_bundle.GetOperatorDesc<DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>();
    desc_fill_max.OutputTensor = &max_val_desc_bundle.get_tensor_desc();
    desc_fill_max.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc_fill_max.Value.Float32 = std::numeric_limits<float>::max();

    dml::Operator* op_fill_max =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle max_val_binding{
        {dml::utils::ResourceFromStorageView(max_val_sv), 0, 0}};
    op_fill_max->Execute({}, max_val_binding);
  }

  // Op 5: Conditional Select
  {  // IF(ConditionUINT8, ATensor, BTensor) -> OutputTensor
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    op_bundle.AddInput(compare_tensor_desc_bundle);
    op_bundle.AddInput(iota_desc_bundle);
    op_bundle.AddInput(max_val_desc_bundle);
    op_bundle.AddOutput(argmin_input_desc_bundle);

    auto& desc = op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_IF_OPERATOR_DESC>();
    desc.ConditionTensor =
        &compare_tensor_desc_bundle.get_tensor_desc();  // This is now UINT8
    desc.ATensor = &iota_desc_bundle.get_tensor_desc();
    desc.BTensor = &max_val_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &argmin_input_desc_bundle.get_tensor_desc();

    dml::Operator* op =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle if_inputs{
        {dml::utils::ResourceFromStorageView(compare_sv), 0, 0},
        {dml::utils::ResourceFromStorageView(iota_sv), 0, 0},
        {dml::utils::ResourceFromStorageView(max_val_sv), 0, 0}};
    dml::utils::DmlBindingArrayBundle if_outputs{
        {dml::utils::ResourceFromStorageView(argmin_input_sv), 0, 0}};
    op->Execute(if_inputs, if_outputs);
  }

  // Op 6: ArgMin
  op_desc_bundles.emplace_back();
  auto& argmin_out_op_bundle = op_desc_bundles.back();

  Shape dml_argmin_out_shape_ct2 = {batch_size, 1};
  const auto& dml_argmin_out_tensor_desc_bundle =
      argmin_out_op_bundle.AddOutput(
          DML_TENSOR_DATA_TYPE_UINT32,
          dml::utils::to_dml_dims(dml_argmin_out_shape_ct2, batch_size, true),
          nullptr);
  StorageView dml_argmin_out_sv(
      dml_argmin_out_shape_ct2,
      DataType::INT32,  // Corresponds to UINT32 for size
      Device::DirectML);
  {
    op_desc_bundles.emplace_back();
    auto& op_bundle = op_desc_bundles.back();

    op_bundle.AddInput(argmin_input_desc_bundle);
    op_bundle.AddOutput(dml_argmin_out_tensor_desc_bundle);

    auto& desc = op_bundle.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
    desc.InputTensor = &argmin_input_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &dml_argmin_out_tensor_desc_bundle.get_tensor_desc();
    desc.Function = DML_REDUCE_FUNCTION_ARGMIN;
    UINT axis_to_reduce =
        argmin_input_desc_bundle.get_buffer_desc().DimensionCount - 1;
    desc.Axes = &axis_to_reduce;
    desc.AxisCount = 1;

    dml::Operator* op =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle argmin_input_binding{
        {dml::utils::ResourceFromStorageView(argmin_input_sv), 0, 0}};
    dml::utils::DmlBindingArrayBundle argmin_output_binding{
        {dml::utils::ResourceFromStorageView(dml_argmin_out_sv), 0, 0}};
    op->Execute(argmin_input_binding, argmin_output_binding);
  }

  // Op 7: Cast/Copy to final output_indices buffer
  Shape original_output_shape = output_indices.shape();
  output_indices.reshape(dml_argmin_out_shape_ct2);

  // DML ARGMIN outputs UINT32. CTranslate2 output_indices is INT32. Cast is
  // needed.
  op_desc_bundles.emplace_back();
  auto& final_cast_op_bundle = op_desc_bundles.back();

  const auto& final_output_desc_bundle =
      final_cast_op_bundle.AddOutput(output_indices);
  final_cast_op_bundle.AddInput(dml_argmin_out_tensor_desc_bundle);

  auto& cast_final_desc =
      final_cast_op_bundle.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
  cast_final_desc.InputTensor = &dml_argmin_out_tensor_desc_bundle
                                     .get_tensor_desc();  // UINT32 from ARGMIN
  cast_final_desc.OutputTensor =
      &final_output_desc_bundle.get_tensor_desc();  // Target type (e.g. INT32)

  dml::Operator* cast_final_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(final_cast_op_bundle));

  dml::utils::DmlBindingArrayBundle cast_final_input_binding{
      {dml::utils::ResourceFromStorageView(dml_argmin_out_sv), 0, 0}};
  dml::utils::DmlBindingArrayBundle cast_final_output_binding{
      {dml::utils::ResourceFromStorageView(output_indices), 0, 0}};
  cast_final_op->Execute(cast_final_input_binding, cast_final_output_binding);

  output_indices.reshape(original_output_shape);
}

}  // namespace dml_internal

template <Device D, typename T>
void Multinomial::compute(const StorageView& probs,
                          StorageView& output_indices) const {
  static_assert(D == Device::DirectML, "This specialization is for DirectML.");
  if (_sample_size != 1) {
    THROW_INVALID_ARGUMENT(
        "DirectML Multinomial currently only supports sample_size = 1");
  }

  PROFILE("MultinomialDML");

  dml_internal::multinomial_impl(probs, output_indices, *this);
}

// Explicit instantiations
#define DECLARE_MULTINOMIAL_DML_IMPL(T)                    \
  template void Multinomial::compute<Device::DirectML, T>( \
      const StorageView& probs, StorageView& output_indices) const;

DECLARE_MULTINOMIAL_DML_IMPL(float)
DECLARE_MULTINOMIAL_DML_IMPL(float16_t)
// DECLARE_MULTINOMIAL_DML_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
