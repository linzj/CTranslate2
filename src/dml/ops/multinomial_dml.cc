#include "ctranslate2/ops/multinomial.h"

#ifdef CT2_WITH_DIRECTML

#include <limits>  // For std::numeric_limits
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

// Ensure THROW_INVALID_ARGUMENT is available (should be via dml_utils.h ->
// utils.h)
#ifndef THROW_INVALID_ARGUMENT
#define THROW_INVALID_ARGUMENT(msg) throw std::invalid_argument(msg)
#endif

namespace ctranslate2 {
namespace ops {
namespace dml_internal {

// Helper function containing the core DML logic for multinomial.
void multinomial_impl(dml::Device* device,  // ctranslate2::dml::Device
                      const StorageView& probs_input,
                      StorageView& output_indices,
                      const Multinomial& op_params) {
  const dim_t depth = probs_input.dim(-1);  // class_size
  const dim_t batch_size = probs_input.size() / depth;

  output_indices.resize({batch_size});

  // --- Handle input cast to FLOAT32 if necessary ---
  StorageView
      probs_f32_sv(  // Temporary StorageView for type/shape metadata if casting
          DataType::FLOAT32,
          device->GetCommandListType() == D3D12_COMMAND_LIST_TYPE_COPY
              ? Device::CPU
              : Device::DirectML);
  ID3D12Resource* current_probs_resource_ptr;
  const dml::utils::DmlTensorDescBundle*
      probs_desc_for_ops_ptr;  // Will point to original or casted

  std::unique_ptr<dml::utils::DmlTensorDescBundle> input_desc_orig_bundle_uptr;
  std::unique_ptr<dml::utils::DmlTensorDescBundle> output_desc_f32_bundle_uptr;

  if (probs_input.dtype() != DataType::FLOAT32) {
    probs_f32_sv.resize(probs_input.shape());

    input_desc_orig_bundle_uptr =
        std::make_unique<dml::utils::DmlTensorDescBundle>(probs_input);
    output_desc_f32_bundle_uptr =
        std::make_unique<dml::utils::DmlTensorDescBundle>(
            probs_f32_sv);  // This will use FLOAT32 type

    DML_CAST_OPERATOR_DESC cast_desc = {};
    cast_desc.InputTensor = &input_desc_orig_bundle_uptr->get_tensor_desc();
    cast_desc.OutputTensor = &output_desc_f32_bundle_uptr->get_tensor_desc();
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_CAST, &cast_desc};

    dml::Operator* cast_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBufferBindingBundle cast_input_b(
        dml::utils::ResourceFromStorageView(probs_input), 0,
        input_desc_orig_bundle_uptr->get_buffer_desc().TotalTensorSizeInBytes);
    dml::utils::DmlBufferBindingBundle cast_output_b(
        dml::utils::ResourceFromStorageView(probs_f32_sv), 0,
        output_desc_f32_bundle_uptr->get_buffer_desc().TotalTensorSizeInBytes);

    cast_op->Execute({cast_input_b.get_desc()}, {cast_output_b.get_desc()});
    current_probs_resource_ptr =
        dml::utils::ResourceFromStorageView(probs_f32_sv);
    probs_desc_for_ops_ptr = output_desc_f32_bundle_uptr.get();
  } else {
    // No cast needed, use original probs_input directly
    input_desc_orig_bundle_uptr =
        std::make_unique<dml::utils::DmlTensorDescBundle>(
            probs_input);  // Still need a bundle for original
    current_probs_resource_ptr =
        dml::utils::ResourceFromStorageView(probs_input);
    probs_desc_for_ops_ptr = input_desc_orig_bundle_uptr.get();
  }

  // --- Prepare DML Tensor Descriptors for FLOAT32 tensors ---
  // probs_desc_for_ops_ptr now points to the DmlTensorDescBundle for probs
  // (either original or casted to FLOAT32)

  Shape random_shape_ct2 = {batch_size, 1};
  std::vector<UINT> random_dml_dims =
      dml::utils::to_dml_dims(random_shape_ct2, batch_size, true);
  // For broadcasting random numbers {B,1} with probs {B,D}, strides for random
  // should be {0,1} if B>1, or {0,0} if B=1, or simply let DML handle with
  // nullptr strides and DimCount matching probs. A safer approach for
  // broadcasting in DML is to have the same DimensionCount, with dims of
  // size 1. Example: if probs is [B,D], random should be [B,1] for element-wise
  // ops. DmlTensorDescBundle should handle this.
  std::vector<UINT> random_strides_vec;
  const std::vector<UINT>* random_strides_ptr = nullptr;
  if (random_dml_dims.size() ==
          probs_desc_for_ops_ptr->get_sizes_vec().size() &&
      random_dml_dims.size() > 0) {  // e.g. both 2D
    random_strides_vec.resize(random_dml_dims.size());
    UINT current_s = 1;
    for (int i = random_dml_dims.size() - 1; i >= 0; --i) {
      random_strides_vec[i] =
          (random_dml_dims[i] == 1 && random_dml_dims.size() > 1)
              ? 0
              : current_s;  // Stride 0 for broadcast dim
      if (random_dml_dims[i] > 0)
        current_s *= random_dml_dims[i];
      else
        current_s = 0;  // Basic stride update logic
    }
    random_strides_ptr = &random_strides_vec;
  }  // else, DmlTensorDescBundle with nullptr strides will compute contiguous
     // for {B,1}

  dml::utils::DmlTensorDescBundle random_desc_bundle(
      DataType::FLOAT32, random_dml_dims, random_strides_ptr);
  StorageView random_numbers_sv(random_shape_ct2, DataType::FLOAT32,
                                Device::DirectML);

  Shape philox_state_dims_shape_ct2 = {4};
  std::vector<UINT> philox_state_dml_dims =
      dml::utils::to_dml_dims(philox_state_dims_shape_ct2, 4, true);
  dml::utils::DmlTensorDescBundle philox_state_tensor_desc_bundle(
      DML_TENSOR_DATA_TYPE_UINT32, philox_state_dml_dims, nullptr);
  StorageView state_in_sv(philox_state_dims_shape_ct2,
                          DataType::INT32,  // Corresponds to UINT32 for size
                          Device::DirectML);
  {  // Zero Init Philox State
    DML_SCALAR_UNION zero_scalar;
    zero_scalar.UInt32 = 0;
    dml::utils::DmlTensorDescBundle temp_fill_bundle(
        DML_TENSOR_DATA_TYPE_UINT32,
        philox_state_tensor_desc_bundle.get_sizes_vec(), nullptr);
    dml::utils::CreateDmlConstantTensor(device, zero_scalar, temp_fill_bundle);
    // The above only *creates* a constant tensor. To fill state_in_res:
    DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_zero_desc = {};
    fill_zero_desc.OutputTensor =
        &philox_state_tensor_desc_bundle
             .get_tensor_desc();  // Describes state_in_res
    fill_zero_desc.ValueDataType = DML_TENSOR_DATA_TYPE_UINT32;
    fill_zero_desc.Value.UInt32 = 0;
    DML_OPERATOR_DESC op_desc_wrapper_fill = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                              &fill_zero_desc};
    dml::Operator* fill_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper_fill);
    dml::utils::DmlBufferBindingBundle fill_out_b_storage(
        dml::utils::ResourceFromStorageView(state_in_sv));
    // Fill state_in_sv
    fill_op->Execute({}, {fill_out_b_storage.get_desc()});
  }
  StorageView state_out_sv(philox_state_dims_shape_ct2,
                           DataType::INT32,  // Corresponds to UINT32 for size
                           Device::DirectML);

  Shape cumsum_shape_ct2 = {batch_size, depth};
  std::vector<UINT> cumsum_dml_dims =
      dml::utils::to_dml_dims(cumsum_shape_ct2, batch_size * depth, true);
  dml::utils::DmlTensorDescBundle cumsum_desc_bundle(DataType::FLOAT32,
                                                     cumsum_dml_dims, nullptr);
  StorageView cumsum_sv(cumsum_shape_ct2, DataType::FLOAT32, Device::DirectML);

  // DML logical ops output UINT8.
  Shape compare_shape_ct2 = {batch_size, depth};
  std::vector<UINT> compare_dml_dims =
      dml::utils::to_dml_dims(compare_shape_ct2, batch_size * depth, true);
  dml::utils::DmlTensorDescBundle compare_tensor_desc_bundle(
      DML_TENSOR_DATA_TYPE_UINT8, compare_dml_dims, nullptr);
  StorageView compare_sv(compare_shape_ct2,
                         DataType::INT8,  // Corresponds to UINT8 for size
                         Device::DirectML);

  Shape iota_shape_ct2 = {1, depth};
  std::vector<UINT> iota_dml_dims =
      dml::utils::to_dml_dims(iota_shape_ct2, depth, true);
  std::vector<UINT> iota_strides_vec = {0, 1};  // Broadcast batch dim
  dml::utils::DmlTensorDescBundle iota_desc_bundle(
      DataType::FLOAT32, iota_dml_dims, &iota_strides_vec);
  StorageView iota_sv(iota_shape_ct2, DataType::FLOAT32, Device::DirectML);

  Shape scalar_shape_ct2 = {1};
  std::vector<UINT> scalar_dml_dims =
      dml::utils::to_dml_dims(scalar_shape_ct2, 1, true);
  std::vector<UINT> scalar_strides_vec = {0};  // Broadcast
  dml::utils::DmlTensorDescBundle max_val_desc_bundle(
      DataType::FLOAT32, scalar_dml_dims, &scalar_strides_vec);
  StorageView max_val_sv({1}, DataType::FLOAT32, Device::DirectML);

  Shape argmin_input_shape_ct2 = {batch_size, depth};
  std::vector<UINT> argmin_input_dml_dims =
      dml::utils::to_dml_dims(argmin_input_shape_ct2, batch_size * depth, true);
  dml::utils::DmlTensorDescBundle argmin_input_desc_bundle(
      DataType::FLOAT32, argmin_input_dml_dims, nullptr);
  StorageView argmin_input_sv(argmin_input_shape_ct2, DataType::FLOAT32,
                              Device::DirectML);

  // --- Define and Execute DML Operators ---

  // Op 1: Random Generator
  {
    DML_RANDOM_GENERATOR_OPERATOR_DESC desc = {};
    desc.InputStateTensor = &philox_state_tensor_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &random_desc_bundle.get_tensor_desc();
    desc.OutputStateTensor = &philox_state_tensor_desc_bundle.get_tensor_desc();
    desc.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_RANDOM_GENERATOR, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(state_in_sv))});
    dml::utils::DmlBindingArrayBundle outputs({
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(random_numbers_sv)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(state_out_sv)),
    });
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Op 2: Cumulative Sum
  {
    DML_CUMULATIVE_SUMMATION_OPERATOR_DESC desc = {};
    desc.InputTensor = &probs_desc_for_ops_ptr->get_tensor_desc();
    desc.OutputTensor = &cumsum_desc_bundle.get_tensor_desc();
    desc.Axis = probs_desc_for_ops_ptr->get_buffer_desc().DimensionCount - 1;
    desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
    desc.HasExclusiveSum = FALSE;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_CUMULATIVE_SUMMATION,
                                         &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBufferBindingBundle input_binding(
        current_probs_resource_ptr);
    dml::utils::DmlBufferBindingBundle output_binding(
        dml::utils::ResourceFromStorageView(cumsum_sv));
    op->Execute({input_binding.get_desc()}, {output_binding.get_desc()});
  }

  // Op 3: Compare (CumulativeProbs >= RandomSample), output should be UINT8
  {
    DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC desc = {};
    desc.ATensor = &cumsum_desc_bundle.get_tensor_desc();
    desc.BTensor = &random_desc_bundle.get_tensor_desc();
    desc.OutputTensor =
        &compare_tensor_desc_bundle
             .get_tensor_desc();  // compare_tensor_desc_bundle is
                                  // DML_TENSOR_DATA_TYPE_UINT8
    DML_OPERATOR_DESC op_desc_wrapper = {
        DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBindingArrayBundle inputs({
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(cumsum_sv)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(random_numbers_sv)),
    });
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(compare_sv))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  // Op 4a: Fill Iota Tensor
  {
    DML_SCALAR_UNION start_val_iota;
    start_val_iota.Float32 = 0.0f;
    DML_SCALAR_UNION delta_val_iota;
    delta_val_iota.Float32 = 1.0f;
    // Use CreateDmlConstantTensor variant or direct FILL_VALUE_SEQUENCE
    DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC desc = {};
    desc.OutputTensor = &iota_desc_bundle.get_tensor_desc();
    desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc.ValueStart.Float32 = 0.0f;
    desc.ValueDelta.Float32 = 1.0f;
    DML_OPERATOR_DESC op_desc_wrapper_seq = {DML_OPERATOR_FILL_VALUE_SEQUENCE,
                                             &desc};
    dml::Operator* op_seq =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper_seq);
    dml::utils::DmlBufferBindingBundle iota_binding(
        dml::utils::ResourceFromStorageView(iota_sv));
    op_seq->Execute({}, {iota_binding.get_desc()});
  }

  // Op 4b: Fill Max Value Scalar Tensor
  {
    DML_SCALAR_UNION max_float_scalar;
    max_float_scalar.Float32 = std::numeric_limits<float>::max();
    dml::utils::DmlTensorDescBundle temp_max_val_bundle(
        DML_TENSOR_DATA_TYPE_FLOAT32, max_val_desc_bundle.get_sizes_vec(),
        nullptr);
    dml::utils::CreateDmlConstantTensor(device, max_float_scalar,
                                        temp_max_val_bundle);
    // Bind max_val_res which was populated by CreateDmlConstantTensor which
    // created its *own* resource. To use pre-allocated max_val_res:
    DML_FILL_VALUE_CONSTANT_OPERATOR_DESC desc_fill_max = {};
    desc_fill_max.OutputTensor = &max_val_desc_bundle.get_tensor_desc();
    desc_fill_max.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc_fill_max.Value.Float32 = std::numeric_limits<float>::max();
    DML_OPERATOR_DESC op_desc_wrapper_fill_max = {
        DML_OPERATOR_FILL_VALUE_CONSTANT, &desc_fill_max};
    dml::Operator* op_fill_max =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper_fill_max);
    dml::utils::DmlBufferBindingBundle max_val_binding(
        dml::utils::ResourceFromStorageView(max_val_sv));
    op_fill_max->Execute({}, {max_val_binding.get_desc()});
  }

  // Op 5: Conditional Select
  {  // IF(ConditionUINT8, ATensor, BTensor) -> OutputTensor
    DML_ELEMENT_WISE_IF_OPERATOR_DESC desc = {};
    desc.ConditionTensor =
        &compare_tensor_desc_bundle.get_tensor_desc();  // This is now UINT8
    desc.ATensor = &iota_desc_bundle.get_tensor_desc();
    desc.BTensor = &max_val_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &argmin_input_desc_bundle.get_tensor_desc();
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_ELEMENT_WISE_IF, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBindingArrayBundle if_inputs({
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(compare_sv)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(iota_sv)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(max_val_sv)),
    });
    dml::utils::DmlBindingArrayBundle if_outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(argmin_input_sv))});
    op->Execute(if_inputs.get_descs(), if_outputs.get_descs());
  }

  // Op 6: ArgMin
  Shape dml_argmin_out_shape_ct2 = {batch_size, 1};
  std::vector<UINT> dml_argmin_out_dml_dims =
      dml::utils::to_dml_dims(dml_argmin_out_shape_ct2, batch_size, true);
  dml::utils::DmlTensorDescBundle dml_argmin_out_tensor_desc_bundle(
      DML_TENSOR_DATA_TYPE_UINT32, dml_argmin_out_dml_dims, nullptr);
  StorageView dml_argmin_out_sv(
      dml_argmin_out_shape_ct2,
      DataType::INT32,  // Corresponds to UINT32 for size
      Device::DirectML);
  {
    DML_REDUCE_OPERATOR_DESC desc = {};
    desc.InputTensor = &argmin_input_desc_bundle.get_tensor_desc();
    desc.OutputTensor = &dml_argmin_out_tensor_desc_bundle.get_tensor_desc();
    desc.Function = DML_REDUCE_FUNCTION_ARGMIN;
    UINT axis_to_reduce =
        argmin_input_desc_bundle.get_buffer_desc().DimensionCount - 1;
    desc.Axes = &axis_to_reduce;
    desc.AxisCount = 1;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_REDUCE, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

    dml::utils::DmlBufferBindingBundle argmin_input_binding(
        dml::utils::ResourceFromStorageView(argmin_input_sv));
    dml::utils::DmlBufferBindingBundle argmin_output_binding(
        dml::utils::ResourceFromStorageView(dml_argmin_out_sv));
    op->Execute({argmin_input_binding.get_desc()},
                {argmin_output_binding.get_desc()});
  }

  // Op 7: Cast/Copy to final output_indices buffer
  Shape original_output_shape = output_indices.shape();
  output_indices.reshape(dml_argmin_out_shape_ct2);
  dml::utils::DmlTensorDescBundle final_output_desc_bundle(
      output_indices);  // Has original output_indices.dtype() (e.g. INT32)

  // DML ARGMIN outputs UINT32. CTranslate2 output_indices is INT32. Cast is
  // needed.
  DML_CAST_OPERATOR_DESC cast_final_desc = {};
  cast_final_desc.InputTensor = &dml_argmin_out_tensor_desc_bundle
                                     .get_tensor_desc();  // UINT32 from ARGMIN
  cast_final_desc.OutputTensor =
      &final_output_desc_bundle.get_tensor_desc();  // Target type (e.g. INT32)
  DML_OPERATOR_DESC op_desc_wrapper_cast_final = {DML_OPERATOR_CAST,
                                                  &cast_final_desc};
  dml::Operator* cast_final_op =
      dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper_cast_final);

  dml::utils::DmlBufferBindingBundle cast_final_input_binding(
      dml::utils::ResourceFromStorageView(dml_argmin_out_sv));
  dml::utils::DmlBufferBindingBundle cast_final_output_binding(
      dml::utils::ResourceFromStorageView(output_indices));
  cast_final_op->Execute({cast_final_input_binding.get_desc()},
                         {cast_final_output_binding.get_desc()});

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

  dml::Device* device = dml::get_device();

  dml_internal::multinomial_impl(device, probs, output_indices, *this);
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
