#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/layer_norm.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void LayerNorm::compute(const StorageView* beta,
                        const StorageView* gamma,
                        const StorageView& input,
                        const dim_t axis,
                        const dim_t outer_size,
                        const dim_t axis_size,
                        const dim_t inner_size,
                        StorageView& output) const {
  if (axis != input.rank() - 1 || !beta || !gamma)
    throw std::invalid_argument(
        "Generalized LayerNorm is currently not implemented on DirectML");

  const bool is_inplace = (input.buffer() == output.buffer());
  StorageView* output_ptr = &output;
  StorageView output_tmp;

  // FIXME(linzj): avoid copying the output if it is not inplace.
  if (is_inplace) {
    output_tmp = StorageView(output.shape(), output.dtype(), output.device());
    output_ptr = &output_tmp;
  }

  std::vector<UINT> dml_input_dims =
      dml::utils::to_dml_dims(input.shape(), input.size());
  std::vector<UINT> scale_bias_dml_dims(dml_input_dims.size(), 1);
  if (!scale_bias_dml_dims.empty()) {
    scale_bias_dml_dims.back() = static_cast<UINT>(axis_size);
  } else if (input.is_scalar() && axis_size == 1) {
    scale_bias_dml_dims = {1};
  }

  dml::utils::DmlOperatorDescBundle op_desc_bundle;
  dml::utils::DmlTensorDescBundle& input_desc_bundle =
      op_desc_bundle.AddInput(input);
  dml::utils::DmlTensorDescBundle& scale_desc_bundle =
      op_desc_bundle.AddInput(gamma->dtype(), scale_bias_dml_dims, nullptr,
                              gamma->size() * gamma->item_size());
  dml::utils::DmlTensorDescBundle& bias_desc_bundle =
      op_desc_bundle.AddInput(beta->dtype(), scale_bias_dml_dims, nullptr,
                              beta->size() * beta->item_size());
  dml::utils::DmlTensorDescBundle& output_desc_bundle =
      op_desc_bundle.AddOutput(*output_ptr);

  UINT normalization_axis = static_cast<UINT>(axis);
  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC& mvn_desc =
      op_desc_bundle
          .GetOperatorDesc<DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC>();
  mvn_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  mvn_desc.ScaleTensor = &scale_desc_bundle.get_tensor_desc();
  mvn_desc.BiasTensor = &bias_desc_bundle.get_tensor_desc();
  mvn_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  mvn_desc.AxisCount = 1;
  mvn_desc.Axes = &normalization_axis;
  mvn_desc.UseMean = TRUE;
  mvn_desc.UseVariance = TRUE;
  mvn_desc.Epsilon = _epsilon;

  auto* compiled_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(op_desc_bundle));

  dml::utils::DmlBindingArrayBundle input_bindings{
      {dml::utils::ResourceFromStorageView(input), 0, 0},
      {dml::utils::ResourceFromStorageView(*gamma), 0, 0},
      {dml::utils::ResourceFromStorageView(*beta), 0, 0}};
  dml::utils::DmlBindingArrayBundle output_bindings{
      {dml::utils::ResourceFromStorageView(*output_ptr), 0, 0}};

  compiled_op->Execute(input_bindings, output_bindings);

  if (is_inplace) {
    output.copy_from(*output_ptr);
  }
}

#define DECLARE_IMPL(T)                                                   \
  template void LayerNorm::compute<Device::DirectML, T>(                  \
      const StorageView* beta, const StorageView* gamma,                  \
      const StorageView& input, const dim_t axis, const dim_t outer_size, \
      const dim_t axis_size, const dim_t inner_size, StorageView& output) \
      const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML