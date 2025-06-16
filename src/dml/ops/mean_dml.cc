#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/mean.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Mean::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t axis_size,
                   const dim_t inner_size,
                   const bool get_sum,
                   StorageView& output) const {
  std::vector<UINT> input_sizes = {static_cast<UINT>(outer_size),
                                   static_cast<UINT>(axis_size),
                                   static_cast<UINT>(inner_size)};

  std::vector<UINT> input_strides = {static_cast<UINT>(axis_size * inner_size),
                                     static_cast<UINT>(inner_size), 1};

  dml::utils::DmlOperatorDescBundle op_desc_bundle;
  auto& input_desc_bundle = op_desc_bundle.AddInput(
      input.dtype(), input_sizes, &input_strides, input.reserved_memory());

  std::vector<UINT> output_sizes = {static_cast<UINT>(outer_size), 1,
                                    static_cast<UINT>(inner_size)};
  std::vector<UINT> output_strides = {static_cast<UINT>(inner_size),
                                      static_cast<UINT>(inner_size), 1};

  auto& output_desc_bundle = op_desc_bundle.AddOutput(
      output.dtype(), output_sizes, &output_strides, output.reserved_memory());

  UINT axis_to_reduce = 1;
  auto& reduce_desc =
      op_desc_bundle.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
  reduce_desc.Function =
      get_sum ? DML_REDUCE_FUNCTION_SUM : DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  reduce_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = &axis_to_reduce;

  auto* compiled_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(op_desc_bundle));

  dml::utils::DmlBindingArrayBundle input_bindings{
      {dml::utils::ResourceFromStorageView(input), 0, 0}};
  dml::utils::DmlBindingArrayBundle output_bindings{
      {dml::utils::ResourceFromStorageView(output), 0, 0}};

  compiled_op->Execute(input_bindings, output_bindings);
}

#define DECLARE_IMPL(T)                                                        \
  template void Mean::compute<Device::DirectML, T>(                            \
      const StorageView& input, const dim_t outer_size, const dim_t axis_size, \
      const dim_t inner_size, const bool get_sum, StorageView& output) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif
