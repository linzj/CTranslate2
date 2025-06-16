#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/quantize.h"

#include "dml/constant_pool.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename InT, typename OutT>
void Quantize::quantize(const StorageView& input,
                        StorageView& output,
                        StorageView& scale) const {
  static_assert(D == Device::DirectML,
                "Device must be DirectML for this specialization.");
  static_assert(std::is_same_v<InT, float>,
                "Input type must be float for this DML quantization.");
  static_assert(std::is_same_v<OutT, int8_t>,
                "Output type must be int8_t for this DML quantization.");

  if (_shift_to_uint8) {
    THROW_INVALID_ARGUMENT(
        "DML Quantize (float->int8_t) currently only supports symmetric "
        "quantization (where _shift_to_uint8 is false). "
        "Asymmetric quantization with uint8-style shift is not yet supported "
        "for DML.");
  }

  // --- 1. Prepare and Resize Output Scale Tensor ---
  // Scale will be computed per row along the first dimension (batch_size).
  // For an input of shape [dim0, dim1, ..., dimN-1], scale will be [dim0, 1,
  // ..., 1]. Example: Input [B, D] -> Scale [B, 1]. Input [B, S, D] -> Scale
  // [B, S, 1] (if per row of D) However, CPU version for [B,D] gives scale [B].
  // For [B,S,D] it depends on how batch_size & depth are passed. Assuming for
  // [B,D] input, scale is [B]. For DML ops, [B,1] is more convenient. Let's
  // make the output 'scale' tensor have shape [input.dim(0), 1, ..., 1] with
  // rank matching input, where all dimensions except the first are 1. The
  // tiling logic will adapt. Or, more simply, for an input of rank R, scale is
  // rank R, with input.dim(0) and 1s elsewhere. For a 2D input [B, Depth],
  // scale shape will be [B, 1].
  Shape scale_shape_ct;
  std::vector<UINT> reduce_axes;
  if (input.rank() == 0) {
    THROW_INVALID_ARGUMENT("Input tensor cannot be scalar.");
  }
  // Scale is computed based on reduction of all axes except the first one(s)
  // that define a "batch" or "instance". For typical INT8 quantization, scale
  // is per-row of the last matrix. If input is [Batch, SeqLen, Depth], scale is
  // often [Batch, SeqLen]. If input is [Batch, Depth], scale is [Batch]. The
  // CPU kernel `cpu::quantize_s8(..., batch_size, depth, ...)` suggests scale
  // is [batch_size]. Let's assume scale is per input.dim(0).

  Shape computed_scale_shape_ct;  // This will be [input.dim(0), 1, ..., 1]
                                  // matching input rank
  for (dim_t i = 0; i < input.rank(); ++i) {
    if (i == 0)
      computed_scale_shape_ct.push_back(input.dim(i));
    else
      computed_scale_shape_ct.push_back(1);
  }
  // Axes to reduce: all except the first one.
  for (dim_t i = 1; i < input.rank(); ++i) {
    reduce_axes.push_back(static_cast<UINT>(i));
  }
  // Special case: input [D], reduce axis 0, scale shape [1]
  if (input.rank() == 1) {
    reduce_axes = {0};
    computed_scale_shape_ct = {1};
  }
  // Resize the output 'scale' parameter. This is where computed scales will be
  // stored.
  scale.resize(computed_scale_shape_ct);

  // --- 2. Compute Scales ---
  // Storage for intermediate results
  StorageView abs_input_storage(input.shape(), input.dtype(), Device::DirectML);
  StorageView abs_max_storage(computed_scale_shape_ct, input.dtype(),
                              Device::DirectML);  // Will store AMAX

  // 2a. Absolute value of input: abs_input = abs(input)
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& input_desc = op_desc.AddInput(input);
    auto& abs_input_desc = op_desc.AddOutput(abs_input_storage);
    auto& abs_op_def =
        op_desc.GetOperatorDesc<DML_ELEMENT_WISE_ABS_OPERATOR_DESC>();
    abs_op_def.InputTensor = &input_desc.get_tensor_desc();
    abs_op_def.OutputTensor = &abs_input_desc.get_tensor_desc();
    dml::Operator* compiled_abs_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE, L"Quantize_Abs");
    compiled_abs_op->Execute(
        {dml::utils::ResourceFromStorageView(input)},
        {dml::utils::ResourceFromStorageView(abs_input_storage)});
  }

  // 2b. Reduce to get abs_max_value: abs_max = reduce_max(abs_input) along
  // feature dimensions
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& abs_input_desc = op_desc.AddInput(abs_input_storage);
    auto& abs_max_desc = op_desc.AddOutput(abs_max_storage);
    auto& reduce_op_def = op_desc.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
    reduce_op_def.InputTensor = &abs_input_desc.get_tensor_desc();
    reduce_op_def.OutputTensor =
        &abs_max_desc.get_tensor_desc();  // Output to abs_max_storage
    reduce_op_def.Function = DML_REDUCE_FUNCTION_MAX;
    reduce_op_def.AxisCount = static_cast<UINT>(reduce_axes.size());
    reduce_op_def.Axes = reduce_axes.data();
    dml::Operator* compiled_reduce_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE, L"Quantize_ReduceMax");
    compiled_reduce_op->Execute(
        {dml::utils::ResourceFromStorageView(abs_input_storage)},
        {dml::utils::ResourceFromStorageView(abs_max_storage)});
  }

  // 2c. Calculate final scale: scale_val = abs_max / 127.0f. Handle abs_max =
  // 0. If abs_max is 0, scale should be 1.0 to make 0/1.0 = 0. Otherwise,
  // abs_max/127.0. Create constant tensors for 127.0f, 0.0f, and 1.0f.
  // These constants are scalars but need DML descriptors that are broadcastable
  // to the shape of computed_scale_shape_ct.
  // The StorageView will hold a single scalar value.
  // The DmlTensorDescBundle will describe it as having the target shape but
  // with zero strides.

  const StorageView& const_127_storage =
      dml::ConstantPool::get_constant<float>({1}, {127.0f}, Device::DirectML);
  const StorageView& const_0_storage =
      dml::ConstantPool::get_constant<float>({1}, {0.0f}, Device::DirectML);
  const StorageView& const_1_storage =
      dml::ConstantPool::get_constant<float>({1}, {1.0f}, Device::DirectML);

  // Prepare DML tensor descriptors for constants with broadcasting
  DML_TENSOR_DATA_TYPE dml_scalar_dtype =
      dml::utils::get_dml_data_type(input.dtype());
  // Convert computed_scale_shape_ct to DML dimensions to define the shape for
  // broadcasting. The storage_size argument to to_dml_dims is primarily for
  // handling empty input shapes; for non-empty computed_scale_shape_ct, its
  // exact value (0 or 1) has less impact here as we are mostly interested in
  // getting the DML representation of computed_scale_shape_ct. 'true' for
  // ensure_at_least_1d_for_dml is generally safe as DML tensors are rarely rank
  // 0.
  std::vector<UINT> target_dml_dims =
      dml::utils::to_dml_dims(computed_scale_shape_ct, 1, true);
  if (target_dml_dims.empty() && !computed_scale_shape_ct.empty()) {
    // This case should ideally not be hit if to_dml_dims works as expected for
    // non-empty shapes or if computed_scale_shape_ct is always at least rank 1
    // from earlier logic. As a fallback if target_dml_dims became unexpectedly
    // empty for a non-empty source shape:
    for (dim_t d : computed_scale_shape_ct)
      target_dml_dims.push_back(static_cast<UINT>(d));
  }
  if (target_dml_dims.empty() &&
      computed_scale_shape_ct
          .empty()) {  // if input was scalar and computed_scale_shape_ct is {1}
                       // which to_dml_dims makes {1}
    // if input.rank() was 0, computed_scale_shape_ct would be {1}.
    // to_dml_dims({1},1,true) -> {1} this block can be simplified as
    // to_dml_dims covers it.
  }

  std::vector<UINT> broadcast_strides(target_dml_dims.size(),
                                      0);  // Strides of 0 for broadcasting
  UINT64 single_scalar_size_bytes =
      dml::utils::get_dml_element_size_in_bytes(dml_scalar_dtype);

  // Intermediate storage for abs_max / 127.0f
  StorageView scale_if_amax_not_zero_storage(computed_scale_shape_ct,
                                             input.dtype(), Device::DirectML);
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& abs_max_desc = op_desc.AddInput(abs_max_storage);
    auto& const_127_desc =
        op_desc.AddInput(dml_scalar_dtype, target_dml_dims, &broadcast_strides,
                         single_scalar_size_bytes);
    auto& scale_if_amax_not_zero_desc =
        op_desc.AddOutput(scale_if_amax_not_zero_storage);
    auto& div_op_def =
        op_desc.GetOperatorDesc<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>();
    div_op_def.ATensor = &abs_max_desc.get_tensor_desc();
    div_op_def.BTensor = &const_127_desc.get_tensor_desc();
    div_op_def.OutputTensor = &scale_if_amax_not_zero_desc.get_tensor_desc();
    dml::Operator* compiled_div_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE, L"Quantize_DivideBy127");
    compiled_div_op->Execute(
        {dml::utils::ResourceFromStorageView(abs_max_storage),
         dml::utils::ResourceFromStorageView(const_127_storage)},
        {dml::utils::ResourceFromStorageView(scale_if_amax_not_zero_storage)});
  }

  // Condition for IF operator: is_amax_zero = (abs_max == 0)
  StorageView condition_storage(
      computed_scale_shape_ct,
      DataType::INT8,  // Store as INT8, DML EQUALS outputs UINT8
      Device::DirectML);
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& abs_max_desc = op_desc.AddInput(abs_max_storage);
    auto& const_0_desc =
        op_desc.AddInput(dml_scalar_dtype, target_dml_dims, &broadcast_strides,
                         single_scalar_size_bytes);
    auto& condition_desc = op_desc.AddOutput(condition_storage);
    condition_desc.set_data_type(DML_TENSOR_DATA_TYPE_UINT8);
    auto& equals_op_def =
        op_desc
            .GetOperatorDesc<DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>();
    equals_op_def.ATensor = &abs_max_desc.get_tensor_desc();
    equals_op_def.BTensor = &const_0_desc.get_tensor_desc();
    equals_op_def.OutputTensor = &condition_desc.get_tensor_desc();
    dml::Operator* compiled_equals_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE, L"Quantize_IsAmaxZero");
    compiled_equals_op->Execute(
        {dml::utils::ResourceFromStorageView(abs_max_storage),
         dml::utils::ResourceFromStorageView(const_0_storage)},
        {dml::utils::ResourceFromStorageView(condition_storage)});
  }

  // IF Operator: scale = is_amax_zero ? 1.0f : (abs_max / 127.0f)
  // Output of IF goes directly into the 'scale' StorageView.
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& condition_desc = op_desc.AddInput(condition_storage);
    condition_desc.set_data_type(DML_TENSOR_DATA_TYPE_UINT8);
    auto& const_1_desc =
        op_desc.AddInput(dml_scalar_dtype, target_dml_dims, &broadcast_strides,
                         single_scalar_size_bytes);
    auto& scale_if_amax_not_zero_desc =
        op_desc.AddInput(scale_if_amax_not_zero_storage);
    auto& scale_desc = op_desc.AddOutput(scale);
    auto& if_op_def =
        op_desc.GetOperatorDesc<DML_ELEMENT_WISE_IF_OPERATOR_DESC>();
    if_op_def.ConditionTensor = &condition_desc.get_tensor_desc();
    if_op_def.ATensor = &const_1_desc.get_tensor_desc();
    if_op_def.BTensor = &scale_if_amax_not_zero_desc.get_tensor_desc();
    if_op_def.OutputTensor = &scale_desc.get_tensor_desc();
    dml::Operator* compiled_if_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE, L"Quantize_SelectScale");
    compiled_if_op->Execute(
        {dml::utils::ResourceFromStorageView(condition_storage),
         dml::utils::ResourceFromStorageView(const_1_storage),
         dml::utils::ResourceFromStorageView(scale_if_amax_not_zero_storage)},
        {dml::utils::ResourceFromStorageView(scale)});
  }

  // --- Quantize Operation using Broadcasted Scale ---
  // The scale tensor will be broadcast to the input tensor's shape using a
  // specialized constructor for the tensor descriptor.
  const Shape& input_shape_ct = input.shape();

  auto dml_input_dims =
      dml::utils::to_dml_dims(input_shape_ct, input.size(), true);
  auto dml_scale_dims =
      dml::utils::to_dml_dims(scale.shape(), scale.size(), true);

  if (output.dtype() != DataType::INT8) {
    THROW_INVALID_ARGUMENT(
        "Output StorageView for int8 quantization must have DataType::INT8.");
  }
  output.resize(
      input.shape());  // Ensure output is allocated with correct shape

  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& input_desc = op_desc.AddInput(input);
    auto& broadcasted_scale_desc = op_desc.AddInput(
        dml::utils::get_dml_data_type(scale.dtype()), dml_input_dims,
        dml_scale_dims, static_cast<int32_t>(dml_input_dims.size()), 0,
        input.rank() > 1 ? 1 : 0, 0, 0);
    auto& output_desc = op_desc.AddOutput(output);
    auto& quantize_op_definition =
        op_desc
            .GetOperatorDesc<DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC>();
    quantize_op_definition.InputTensor = &input_desc.get_tensor_desc();
    quantize_op_definition.ScaleTensor =
        &broadcasted_scale_desc.get_tensor_desc();
    quantize_op_definition.ZeroPointTensor = nullptr;
    quantize_op_definition.OutputTensor = &output_desc.get_tensor_desc();
    dml::Operator* compiled_quantize_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE,
        L"ElementWiseQuantizeLinear_F32_S8_WithComputedBroadcastedScale");
    compiled_quantize_op->Execute(
        {dml::utils::ResourceFromStorageView(input),
         dml::utils::ResourceFromStorageView(
             scale),              /* Use original scale buffer */
         nullptr /*ZeroPoint*/},  // ZeroPoint tensor is explicitly null for
                                  // int8 symmetric quantization in DML
        {dml::utils::ResourceFromStorageView(output)});
  }
  // --- Correct the 'scale' to be its reciprocal for the output parameter ---
  // The 'scale' StorageView (output parameter) currently holds S_calc (abs_max
  // / 127.0f or 1.0f). The request is for its final value to be 1/S_calc. The
  // 'scale' StorageView is used as both input and output for an in-place
  // modification.
  {
    dml::utils::DmlOperatorDescBundle op_desc;
    auto& scale_desc = op_desc.AddInput(scale);
    auto& output_scale_desc = op_desc.AddOutput(scale);
    auto& recip_op_def =
        op_desc.GetOperatorDesc<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>();
    recip_op_def.InputTensor = &scale_desc.get_tensor_desc();
    recip_op_def.OutputTensor = &output_scale_desc.get_tensor_desc();
    recip_op_def.ScaleBias = nullptr;
    dml::Operator* compiled_recip_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(op_desc), DML_EXECUTION_FLAG_NONE,
        L"Quantize_FinalReciprocalScale");
    compiled_recip_op->Execute({dml::utils::ResourceFromStorageView(scale)},
                               {dml::utils::ResourceFromStorageView(scale)});
  }
}

// Explicit template instantiation
template void Quantize::quantize<Device::DirectML, float, int8_t>(
    const StorageView&,
    StorageView&,
    StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2
#endif