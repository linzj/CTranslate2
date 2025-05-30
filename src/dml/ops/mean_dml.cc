#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/mean.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Local get_dml_data_type removed

template <Device D, typename T>
void Mean::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t axis_size,
                   const dim_t inner_size,
                   const bool get_sum,
                   StorageView& output) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // The input is conceptually reshaped to [outer_size, axis_size, inner_size]
  // We need to reduce along the middle dimension (axis = 1)
  std::vector<UINT> input_sizes = {static_cast<UINT>(outer_size),
                                   static_cast<UINT>(axis_size),
                                   static_cast<UINT>(inner_size)};

  // Calculate strides for row-major layout
  std::vector<UINT> input_strides = {static_cast<UINT>(axis_size * inner_size),
                                     static_cast<UINT>(inner_size), 1};

  // Create input tensor descriptor using DmlTensorDescBundle
  // DataType is derived from input StorageView
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), input_sizes, &input_strides, input.size() * sizeof(T));

  // Output shape after reduction: [outer_size, 1, inner_size]
  std::vector<UINT> output_sizes = {static_cast<UINT>(outer_size), 1,
                                    static_cast<UINT>(inner_size)};
  std::vector<UINT> output_strides = {static_cast<UINT>(inner_size),
                                      static_cast<UINT>(inner_size), 1};

  // Create output tensor descriptor using DmlTensorDescBundle
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), output_sizes, &output_strides, output.size() * sizeof(T));

  // Create reduce operator - reduce along axis 1 (the middle dimension)
  UINT axis_to_reduce = 1;
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function =
      get_sum ? DML_REDUCE_FUNCTION_SUM : DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  reduce_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = &axis_to_reduce;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_REDUCE;
  op_desc.Desc = &reduce_desc;

  // Get compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Bind input
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
          0, input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  // Bind output
  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  // Record and execute the dispatch
  compiled_op->Execute({input_binding_desc_for_op},
                       {output_binding_desc_for_op});
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
