#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/layer_norm.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
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

  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  // Data type conversion and tensor descriptors using DmlTensorDescBundle
  // DML_TENSOR_DATA_TYPE dml_data_type =
  // dml::utils::get_dml_data_type(input.dtype()); // DmlTensorDescBundle
  // handles this

  dml::utils::DmlTensorDescBundle input_desc_bundle(input);
  dml::utils::DmlTensorDescBundle output_desc_bundle(output);

  // For gamma (scale) and beta (bias), shapes need to be DML
  // broadcast-compatible. DML_MEAN_VARIANCE_NORMALIZATION2 requires ScaleTensor
  // and BiasTensor to have the same DimensionCount as InputTensor. For
  // dimensions being normalized, their size must match InputTensor. For other
  // dimensions, their size must be 1. The condition `axis != input.rank() - 1`
  // means normalization is not only on the last dim, which this code path says
  // is not implemented ("Generalized LayerNorm is currently not implemented").
  // The current check `axis != input.rank() - 1 || !beta || !gamma` means we
  // only handle normalization on the last axis. In this case, `axis_size` is
  // `input.dim(-1)`. Beta and Gamma StorageView shapes are typically
  // [axis_size]. For DML, they need to be broadcastable, e.g. [1, 1, ...,
  // axis_size].
  std::vector<UINT> dml_input_dims =
      dml::utils::to_dml_dims(input.shape(), input.size());
  std::vector<UINT> scale_bias_dml_dims(dml_input_dims.size(), 1);
  if (!scale_bias_dml_dims.empty()) {
    scale_bias_dml_dims.back() = static_cast<UINT>(axis_size);
  } else if (input.is_scalar() &&
             axis_size == 1) {  // Handle scalar input normalized as if it's [1]
                                // and axis_size is 1.
    scale_bias_dml_dims = {1};
  }

  dml::utils::DmlTensorDescBundle scale_desc_bundle(
      gamma->dtype(), scale_bias_dml_dims, nullptr,
      gamma->size() * gamma->item_size());
  dml::utils::DmlTensorDescBundle bias_desc_bundle(
      beta->dtype(), scale_bias_dml_dims, nullptr,
      beta->size() * beta->item_size());

  // Create Mean Variance Normalization operator
  UINT normalization_axis = static_cast<UINT>(axis);

  DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC mvn_desc = {};
  mvn_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  mvn_desc.ScaleTensor = &scale_desc_bundle.get_tensor_desc();
  mvn_desc.BiasTensor = &bias_desc_bundle.get_tensor_desc();
  mvn_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  mvn_desc.AxisCount = 1;
  mvn_desc.Axes = &normalization_axis;
  mvn_desc.UseMean = TRUE;
  mvn_desc.UseVariance = TRUE;
  mvn_desc.Epsilon = _epsilon;
  mvn_desc.FusedActivation = nullptr;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2;
  op_desc.Desc = &mvn_desc;

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Get D3D12 resources from StorageView buffers
  // StorageView::buffer() returns ID3D12Resource* for DirectML backend
  // input and output must share the same buffer.
  ID3D12Resource* input_resource = dml::utils::ResourceFromStorageView(input);
  device->KeepAliveUntilNextCommandListDispatch(input_resource);
  StorageView input_storage;
  if (input.buffer() == output.buffer()) {
    input_storage = std::move(input);
    StorageView new_output(input_storage.shape(), input_storage.dtype(),
                           input_storage.device());
    output = std::move(new_output);
  }
  auto scale_resource = dml::utils::ResourceFromStorageView(*gamma);
  auto bias_resource = dml::utils::ResourceFromStorageView(*beta);
  auto output_resource = dml::utils::ResourceFromStorageView(output);

  // Bind input tensors
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          input_resource, 0,
          input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BUFFER_BINDING scale_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          scale_resource, 0,
          scale_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BUFFER_BINDING bias_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          bias_resource, 0,
          bias_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  std::vector<DML_BINDING_DESC> input_bindings_for_op = {
      dml::utils::create_binding_desc(&input_buffer_binding_storage),
      dml::utils::create_binding_desc(&scale_buffer_binding_storage),
      dml::utils::create_binding_desc(&bias_buffer_binding_storage)};

  // Bind output tensor
  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          output_resource, 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  std::vector<DML_BINDING_DESC> output_bindings_for_op = {
      dml::utils::create_binding_desc(&output_buffer_binding_storage)};

  compiled_op->Execute(input_bindings_for_op, output_bindings_for_op);
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