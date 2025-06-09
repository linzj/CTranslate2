#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"

#include "ctranslate2/storage_view.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale)
    throw std::runtime_error(
        "Quantization is not supported in this Conv1D implementation");

  const DML_TENSOR_DATA_TYPE dml_data_type =
      dml::utils::get_dml_data_type(input.dtype());

  // 1. Prepare Tensor Descriptors
  std::vector<UINT> input_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(input);
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), input_dml_dims_vec, nullptr, input.reserved_memory());

  std::vector<UINT> weight_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(weight, true);
  dml::utils::DmlTensorDescBundle weight_desc_bundle(
      weight.dtype(), weight_dml_dims_vec, nullptr, weight.reserved_memory());

  std::vector<UINT> output_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(output);
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), output_dml_dims_vec, nullptr, output.reserved_memory());

  std::unique_ptr<dml::utils::DmlTensorDescBundle> bias_desc_bundle_ptr;
  if (bias) {
    std::vector<UINT> bias_dml_dims_vec =
        dml::utils::get_dml_tensor_shape_4d(*bias, true);
    bias_desc_bundle_ptr = std::make_unique<dml::utils::DmlTensorDescBundle>(
        bias->dtype(), bias_dml_dims_vec, nullptr, bias->reserved_memory());
  }

  // 2. Create Convolution Operator Descriptor
  DML_CONVOLUTION_OPERATOR_DESC conv_op_desc{};
  conv_op_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  conv_op_desc.FilterTensor = &weight_desc_bundle.get_tensor_desc();
  conv_op_desc.BiasTensor =
      bias ? &bias_desc_bundle_ptr->get_tensor_desc() : nullptr;
  conv_op_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  conv_op_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
  conv_op_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  conv_op_desc.DimensionCount = 2;
  UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
  conv_op_desc.Strides = dml_strides;
  UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
  conv_op_desc.Dilations = dml_dilations;
  UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.StartPadding = dml_start_padding;
  UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.EndPadding = dml_end_padding;
  UINT dml_output_padding[] = {0, 0};
  conv_op_desc.OutputPadding = dml_output_padding;
  conv_op_desc.GroupCount = static_cast<UINT>(_groups);

  DML_OPERATOR_DESC dml_op_desc_wrapper{};
  dml_op_desc_wrapper.Type = DML_OPERATOR_CONVOLUTION;
  dml_op_desc_wrapper.Desc = &conv_op_desc;

  // 3. Get or Create Compiled Operator from Cache
  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
  if (dml_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
    execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  }

  dml::Operator* dml_convolution_operator = dml::GetOrCreateCompiledOperatorApi(
      &dml_op_desc_wrapper, execution_flags);

  // 4. Prepare Bindings and Execute
  std::vector<dml::utils::DmlBufferBindingBundle> input_bundles;
  input_bundles.emplace_back(dml::utils::ResourceFromStorageView(input));
  input_bundles.emplace_back(dml::utils::ResourceFromStorageView(weight));
  if (bias) {
    input_bundles.emplace_back(dml::utils::ResourceFromStorageView(*bias));
  } else {
    input_bundles
        .emplace_back();  // Add a "none" binding for the optional bias.
  }

  dml::utils::DmlBindingArrayBundle input_bindings(std::move(input_bundles));
  dml::utils::DmlBindingArrayBundle output_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(output))});

  dml_convolution_operator->Execute(input_bindings.get_descs(),
                                    output_bindings.get_descs());
}

#define DECLARE_CONV1D_DML_IMPL(T)                                             \
  template void Conv1D::compute<Device::DirectML, T>(                          \
      const StorageView& input, const StorageView& weight,                     \
      const StorageView* bias, StorageView& output, const StorageView* qscale) \
      const;

DECLARE_CONV1D_DML_IMPL(float)
DECLARE_CONV1D_DML_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
