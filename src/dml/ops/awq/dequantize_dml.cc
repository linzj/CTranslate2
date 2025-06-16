#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/awq/dequantize_awq.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename InT, typename OutT>
void DequantizeAwq::dequantize(const StorageView& input,
                               const StorageView& scale,
                               const StorageView& zero,
                               StorageView& output) const {
  dim_t in_c = input.rank() == 2 ? input.dim(0) : input.dim(1);
  dim_t qout_c = input.rank() == 2 ? input.dim(1) : input.dim(2);
  int num_experts = input.rank() == 2 ? 1 : input.dim(0);
  int out_c = qout_c * 8;
  int G = in_c / (input.rank() == 2 ? scale.dim(0) : scale.dim(1));

  if (num_experts == 1) {
    output.resize({in_c, out_c});
  } else {
    output.resize({num_experts, in_c, out_c});
  }

  std::vector<UINT> input_sizes, scale_strides, zero_strides;
  std::vector<UINT> broadcasted_sizes =
      (num_experts == 1)
          ? std::vector<UINT>{static_cast<UINT>(in_c), static_cast<UINT>(out_c)}
          : std::vector<UINT>{static_cast<UINT>(num_experts),
                              static_cast<UINT>(in_c),
                              static_cast<UINT>(out_c)};

  if (num_experts == 1) {
    input_sizes = {static_cast<UINT>(in_c), static_cast<UINT>(qout_c)};
    scale_strides = {static_cast<UINT>(out_c) * static_cast<UINT>(G), 1};
    zero_strides = {static_cast<UINT>(qout_c) * static_cast<UINT>(G), 8};
  } else {
    input_sizes = {static_cast<UINT>(num_experts), static_cast<UINT>(in_c),
                   static_cast<UINT>(qout_c)};
    scale_strides = {static_cast<UINT>((in_c / G) * out_c),
                     static_cast<UINT>(out_c) * static_cast<UINT>(G), 1};
    zero_strides = {static_cast<UINT>((in_c / G) * qout_c),
                    static_cast<UINT>(qout_c) * static_cast<UINT>(G), 8};
  }

  dml::utils::DmlOperatorDescBundle op_desc_bundle;
  auto& input_desc =
      op_desc_bundle.AddInput(dml::utils::get_dml_data_type(input.dtype()),
                              input_sizes, nullptr, input.reserved_memory());
  auto& scale_desc = op_desc_bundle.AddInput(
      dml::utils::get_dml_data_type(scale.dtype()), broadcasted_sizes,
      &scale_strides, scale.reserved_memory());
  auto& zero_desc = op_desc_bundle.AddInput(
      dml::utils::get_dml_data_type(zero.dtype()), broadcasted_sizes,
      &zero_strides, zero.reserved_memory());
  auto& output_desc = op_desc_bundle.AddOutput(output);

  auto& dequantize_desc =
      op_desc_bundle
          .GetOperatorDesc<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>();
  dequantize_desc.InputTensor = &input_desc.get_tensor_desc();
  dequantize_desc.ScaleTensor = &scale_desc.get_tensor_desc();
  dequantize_desc.ZeroPointTensor = &zero_desc.get_tensor_desc();
  dequantize_desc.OutputTensor = &output_desc.get_tensor_desc();

  auto* compiled_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(op_desc_bundle));

  dml::utils::DmlBindingArrayBundle inputs{
      {dml::utils::ResourceFromStorageView(input), 0, 0},
      {dml::utils::ResourceFromStorageView(scale), 0, 0},
      {dml::utils::ResourceFromStorageView(zero), 0, 0}};
  dml::utils::DmlBindingArrayBundle outputs{
      {dml::utils::ResourceFromStorageView(output), 0, 0}};

  compiled_op->Execute(inputs, outputs);
}

#define DECLARE_IMPL(T)                                              \
  template void DequantizeAwq::dequantize<Device::DirectML, int, T>( \
      const StorageView&, const StorageView&, const StorageView&,    \
      StorageView&) const;

DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
