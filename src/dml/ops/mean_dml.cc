#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/mean.h"
#include "dml/backend_dml.h"
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

  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), input_sizes, &input_strides, input.reserved_memory());

  std::vector<UINT> output_sizes = {static_cast<UINT>(outer_size), 1,
                                    static_cast<UINT>(inner_size)};
  std::vector<UINT> output_strides = {static_cast<UINT>(inner_size),
                                      static_cast<UINT>(inner_size), 1};

  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), output_sizes, &output_strides, output.reserved_memory());

  UINT axis_to_reduce = 1;
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function =
      get_sum ? DML_REDUCE_FUNCTION_SUM : DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  reduce_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = &axis_to_reduce;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  dml::utils::DmlBindingArrayBundle input_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(input))});
  dml::utils::DmlBindingArrayBundle output_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(output))});

  compiled_op->Execute(input_bindings.get_descs(), output_bindings.get_descs());
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
