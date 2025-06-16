#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/awq/gemv.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {
namespace {

void dequantize_weights_dml(const StorageView& quantized_weights,
                            const StorageView& scales,
                            const StorageView& zero_points,
                            StorageView& dequantized_weights,
                            dim_t group_size) {
  // This is a placeholder. A real implementation would involve a custom
  // shader for unpacking and dequantization.
}

void slice_tensor_k_dimension(const StorageView& input,
                              StorageView& output,
                              dim_t k_start,
                              dim_t k_end) {
  dml::utils::DmlOperatorDescBundle op_bundle;
  auto& input_desc = op_bundle.AddInput(input);
  auto& output_desc = op_bundle.AddOutput(output);

  std::vector<UINT> offsets(input.rank(), 0);
  std::vector<UINT> sizes(input.shape().begin(), input.shape().end());
  std::vector<UINT> strides(input.rank(), 1);

  for (dim_t i = 0; i < input.rank(); ++i) {
    if (i == input.rank() - 1) {  // K dimension (last dimension)
      offsets[i] = static_cast<UINT>(k_start);
      sizes[i] = static_cast<UINT>(k_end - k_start);
    }
  }

  auto& slice_desc = op_bundle.GetOperatorDesc<DML_SLICE_OPERATOR_DESC>();
  slice_desc.InputTensor = &input_desc.get_tensor_desc();
  slice_desc.OutputTensor = &output_desc.get_tensor_desc();
  slice_desc.DimensionCount = static_cast<UINT>(input.rank());
  slice_desc.Offsets = offsets.data();
  slice_desc.Sizes = sizes.data();
  slice_desc.Strides = strides.data();

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
  dml::utils::DmlBindingArrayBundle inputs{
      {dml::utils::ResourceFromStorageView(input), 0, 0}};
  dml::utils::DmlBindingArrayBundle outputs{
      {dml::utils::ResourceFromStorageView(output), 0,
       output.size() * output.item_size()}};
  compiled_op->Execute(inputs, outputs);
}

void perform_partial_gemv(const StorageView& a_slice,
                          const StorageView& b_slice,
                          StorageView& c_slice) {
  StorageView reshaped_a = a_slice;
  if (a_slice.rank() == 3) {
    reshaped_a.reshape({a_slice.dim(0) * a_slice.dim(1), a_slice.dim(2)});
  }

  dml::utils::DmlOperatorDescBundle op_bundle;
  auto& a_desc = op_bundle.AddInput(reshaped_a);
  auto& b_desc = op_bundle.AddInput(b_slice);
  auto& c_desc = op_bundle.AddOutput(c_slice);

  auto& gemm_desc = op_bundle.GetOperatorDesc<DML_GEMM_OPERATOR_DESC>();
  gemm_desc.ATensor = &a_desc.get_tensor_desc();
  gemm_desc.BTensor = &b_desc.get_tensor_desc();
  gemm_desc.OutputTensor = &c_desc.get_tensor_desc();
  gemm_desc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
  gemm_desc.Alpha = 1.0f;

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  dml::utils::DmlBindingArrayBundle inputs{
      {
          dml::utils::ResourceFromStorageView(reshaped_a),
          0,
          0,
      },
      {dml::utils::ResourceFromStorageView(b_slice), 0, 0}};
  dml::utils::DmlBindingArrayBundle outputs{
      {dml::utils::ResourceFromStorageView(c_slice), 0, 0}};
  compiled_op->Execute(inputs, outputs);
}

void reduce_split_k_results(StorageView& c) {
  dim_t split_k_iters = c.dim(0);
  StorageView final_output;
  if (c.rank() == 3) {
    final_output.resize({c.dim(1), c.dim(2)});
  } else {
    final_output.resize({c.dim(1), c.dim(2), c.dim(3)});
  }

  dml::utils::DmlOperatorDescBundle op_bundle;
  auto& input_desc = op_bundle.AddInput(c);
  auto& output_desc = op_bundle.AddOutput(final_output);

  UINT reduce_axes[] = {0};
  auto& reduce_desc = op_bundle.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
  reduce_desc.InputTensor = &input_desc.get_tensor_desc();
  reduce_desc.OutputTensor = &output_desc.get_tensor_desc();
  reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = reduce_axes;
  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  dml::utils::DmlBindingArrayBundle inputs{
      {dml::utils::ResourceFromStorageView(c), 0, 0}};
  dml::utils::DmlBindingArrayBundle outputs{
      {dml::utils::ResourceFromStorageView(final_output), 0, 0}};
  compiled_op->Execute(inputs, outputs);

  c = std::move(final_output);
}

void zero_tensor_dml(StorageView& tensor) {
  dml::utils::DmlOperatorDescBundle op_bundle;
  auto& tensor_desc = op_bundle.AddOutput(tensor);
  auto& fill_desc =
      op_bundle.GetOperatorDesc<DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>();
  fill_desc.OutputTensor = &tensor_desc.get_tensor_desc();
  fill_desc.ValueDataType = tensor_desc.get_data_type();
  memset(&fill_desc.Value, 0, sizeof(fill_desc.Value));

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  dml::utils::DmlBindingArrayBundle outputs{
      {dml::utils::ResourceFromStorageView(tensor), 0,
       tensor.size() * tensor.item_size()}};
  compiled_op->Execute({}, outputs);
}

}  // namespace

template <>
void GemvAwq::compute_gemv<Device::DirectML, float16_t, int>(
    const StorageView& a,
    const StorageView& b,
    const StorageView& scale,
    const StorageView& zero,
    StorageView& c) const {
  dim_t num_in_channels = a.dim(-1);
  dim_t num_in_feats = a.size() / num_in_channels;
  dim_t num_out_channels = b.dim(0);

  if (a.rank() == 2)
    c.resize({num_in_feats, num_out_channels});
  else if (a.rank() == 3)
    c.resize({a.dim(0), a.dim(1), num_out_channels});

  StorageView dequantized_b({num_out_channels, num_in_channels},
                            DataType::FLOAT16, Device::DirectML);
  dequantize_weights_dml(b, scale, zero, dequantized_b,
                         num_in_channels / scale.dim(-1));

  StorageView reshaped_a = a;
  if (a.rank() == 3) {
    reshaped_a.reshape({a.dim(0) * a.dim(1), a.dim(2)});
  }

  perform_partial_gemv(reshaped_a, dequantized_b, c);
}

template <>
void GemvAwq::compute_gemv2<Device::DirectML, float16_t, int>(
    const StorageView& a,
    const StorageView& b,
    const StorageView& scale,
    const StorageView& zero,
    StorageView& c) const {
  dim_t split_k_iters = 8;
  dim_t num_in_channels = a.dim(-1);
  dim_t num_out_channels = b.dim(0);
  dim_t k_per_split = (num_in_channels + split_k_iters - 1) / split_k_iters;

  c.resize({split_k_iters, a.dim(0), a.dim(1), num_out_channels});

  StorageView dequantized_b({num_out_channels, num_in_channels},
                            DataType::FLOAT16, Device::DirectML);
  dequantize_weights_dml(b, scale, zero, dequantized_b,
                         num_in_channels / scale.dim(-1));

  for (dim_t split_idx = 0; split_idx < split_k_iters; ++split_idx) {
    dim_t k_start = split_idx * k_per_split;
    dim_t k_end = std::min(k_start + k_per_split, num_in_channels);
    dim_t actual_k = k_end - k_start;

    StorageView c_slice(c.dtype(), c.device());
    if (a.rank() == 2) {
      c_slice.view(static_cast<float16_t*>(c.data<float16_t>()) +
                       split_idx * a.dim(0) * num_out_channels,
                   {a.dim(0), num_out_channels});
    } else {
      c_slice.view(static_cast<float16_t*>(c.data<float16_t>()) +
                       split_idx * a.dim(0) * a.dim(1) * num_out_channels,
                   {a.dim(0), a.dim(1), num_out_channels});
    }

    if (actual_k == 0) {
      zero_tensor_dml(c_slice);
      continue;
    }

    StorageView a_slice(
        {a.rank() == 2 ? a.dim(0) : a.dim(0) * a.dim(1), actual_k}, a.dtype(),
        a.device());
    slice_tensor_k_dimension(a, a_slice, k_start, k_end);

    StorageView b_slice({num_out_channels, actual_k}, b.dtype(), b.device());
    slice_tensor_k_dimension(dequantized_b, b_slice, k_start, k_end);

    perform_partial_gemv(a_slice, b_slice, c_slice);
  }

  reduce_split_k_results(c);
}

#define DECLARE_IMPL(T)                                           \
  template void GemvAwq::compute_gemv2<Device::DirectML, T, int>( \
      const StorageView&, const StorageView&, const StorageView&, \
      const StorageView&, StorageView&) const;                    \
  template void GemvAwq::compute_gemv<Device::DirectML, T, int>(  \
      const StorageView&, const StorageView&, const StorageView&, \
      const StorageView&, StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2

#endif
