#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"  // Class declaration for Conv1D

#include "dml/backend_dml.h"
// #include "dml/dxdevice.h" // dml::utils::create_buffer_binding is in
// dml_utils.h which includes backend_dml.h
#include "ctranslate2/storage_view.h"
#include "dml/dml_utils.h"  // Added for centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Anonymous namespace with local helpers removed.
// Helpers like get_dml_tensor_shape_4d, get_dml_data_type,
// create_dml_buffer_tensor_desc, wrap_dml_buffer_tensor_desc, to_dml_dims will
// be replaced by dml::utils versions.

// Private helper method to encapsulate DML-specific logic for Conv1D

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale)
    throw std::runtime_error(
        "Quantization is not supported in this Conv1D implementation");
  dml::Device* dml_device_handle =
      dml::get_device();  // ctranslate2::dml::Device wrapper

  DML_TENSOR_DATA_TYPE dml_data_type =
      dml::utils::get_dml_data_type(input.dtype());

  // 1. Prepare Tensor Descriptors using DmlTensorDescBundle
  // These bundles manage the lifetime of internal DML_BUFFER_TENSOR_DESC and
  // their dimension/stride vectors.

  // For input, weight, output, we use their specific 4D shapes for convolution.
  // The DmlTensorDescBundle constructor taking (DataType, sizes, strides,
  // total_size_bytes) will be useful here. We'll need to prepare the dml_dims
  // first.

  std::vector<UINT> input_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(input);
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), input_dml_dims_vec, nullptr,
      input.size() * input.item_size());

  std::vector<UINT> weight_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(weight, true);
  dml::utils::DmlTensorDescBundle weight_desc_bundle(
      weight.dtype(), weight_dml_dims_vec, nullptr,
      weight.size() * weight.item_size());

  std::vector<UINT> output_dml_dims_vec =
      dml::utils::get_dml_tensor_shape_4d(output);
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), output_dml_dims_vec, nullptr,
      output.size() * output.item_size());

  // Optional Bias Tensor
  std::unique_ptr<dml::utils::DmlTensorDescBundle> bias_desc_bundle_ptr;
  const DML_TENSOR_DESC* bias_tensor_desc_for_op_ptr = nullptr;

  if (bias) {
    std::vector<UINT> bias_dml_dims_vec =
        dml::utils::get_dml_tensor_shape_4d(*bias, true);
    bias_desc_bundle_ptr = std::make_unique<dml::utils::DmlTensorDescBundle>(
        bias->dtype(), bias_dml_dims_vec, nullptr,
        bias->size() * bias->item_size());
    bias_tensor_desc_for_op_ptr = &bias_desc_bundle_ptr->get_tensor_desc();
  }

  // 2. Create Convolution Operator Descriptor
  DML_CONVOLUTION_OPERATOR_DESC conv_op_desc{};
  conv_op_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  conv_op_desc.FilterTensor = &weight_desc_bundle.get_tensor_desc();
  conv_op_desc.BiasTensor = bias_tensor_desc_for_op_ptr;  // Null if no bias
  conv_op_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();

  conv_op_desc.Mode =
      DML_CONVOLUTION_MODE_CROSS_CORRELATION;  // Standard convolution behavior
  conv_op_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  conv_op_desc.DimensionCount =
      2;  // Number of spatial dimensions (Height, Width). For 1D Conv, H=1.

  // Strides, Dilations, Paddings are for {H, W} dimensions.
  // For 1D convolution, H-dimension parameters are 1 (stride/dilation) or 0
  // (padding).
  UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
  conv_op_desc.Strides = dml_strides;

  UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
  conv_op_desc.Dilations = dml_dilations;

  // _padding is assumed to be symmetric padding for the width/length dimension.
  UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.StartPadding = dml_start_padding;

  UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.EndPadding = dml_end_padding;

  // OutputPadding is mainly for deconvolution/transposed convolution. Usually
  // {0,0} for standard convolution.
  UINT dml_output_padding[] = {0, 0};
  conv_op_desc.OutputPadding = dml_output_padding;

  conv_op_desc.GroupCount = static_cast<UINT>(_groups);
  conv_op_desc.FusedActivation =
      nullptr;  // No fused activation in this basic implementation

  DML_OPERATOR_DESC dml_op_desc_wrapper{};
  dml_op_desc_wrapper.Type = DML_OPERATOR_CONVOLUTION;
  dml_op_desc_wrapper.Desc = &conv_op_desc;

  // 3. Get or Create Compiled Operator from Cache
  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
  if (dml_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
    execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  }

  // Use the global API which internally gets the device.
  dml::Operator* dml_convolution_operator = dml::GetOrCreateCompiledOperatorApi(
      &dml_op_desc_wrapper, execution_flags);

  // 4. Prepare Bindings for Execution
  // Input resources (order: Input, Filter, Bias (if present))
  // StorageView::buffer() returns const void*. We need non-const void* for
  // static_cast to ID3D12Resource*. Then, this ID3D12Resource* is used in
  // DML_BUFFER_BINDING.
  std::vector<DML_BINDING_DESC> input_bindings_for_op;

  // Storage for DML_BUFFER_BINDING structs, must outlive DML_BINDING_DESC if
  // using pointers.
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
          0, input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  input_bindings_for_op.push_back(
      dml::utils::create_binding_desc(&input_buffer_binding_storage));

  DML_BUFFER_BINDING weight_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(weight.buffer())),
          0, weight_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  input_bindings_for_op.push_back(
      dml::utils::create_binding_desc(&weight_buffer_binding_storage));

  DML_BUFFER_BINDING bias_buffer_binding_storage;  // Define outside if
  if (bias && bias_desc_bundle_ptr) {
    bias_buffer_binding_storage = dml::utils::create_buffer_binding(
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer())), 0,
        bias_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);
    input_bindings_for_op.push_back(
        dml::utils::create_binding_desc(&bias_buffer_binding_storage));
  } else {
    input_bindings_for_op.push_back({DML_BINDING_TYPE_NONE, nullptr});
  }

  // Output resources
  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);
  std::vector<DML_BINDING_DESC> output_bindings_for_op = {
      output_binding_desc_for_op};

  // 5. Execute the Operator
  // This records the dispatch into the command list associated with
  // dml_device_handle. The command list needs to be executed later by the
  // framework.
  dml_convolution_operator->Execute(input_bindings_for_op,
                                    output_bindings_for_op);
}

// Template specializations for Conv1D::compute for Device::DirectML
// The primary template compute<Device D, typename T> should correctly dispatch.
// Explicit specializations might be redundant if the primary template correctly
// uses dml_data_type. However, if these specializations are intended to add
// specific checks or logic before calling the primary, they can stay. For this
// refactoring, assuming the primary template is now correct for both float and
// float16 using dml::utils. So, the content of these specializations can be
// simplified or removed if the primary template fully handles it. Let's keep
// them for now, but ensure they call the primary correctly.

template <>
void Conv1D::compute<Device::DirectML, float>(const StorageView& input,
                                              const StorageView& weight,
                                              const StorageView* bias,
                                              StorageView& output,
                                              const StorageView* qscale) const {
  // Call the primary template directly, all logic is now there.
  this->Conv1D::compute<Device::DirectML, float>(input, weight, bias, output,
                                                 qscale);
}

template <>
void Conv1D::compute<Device::DirectML, float16_t>(
    const StorageView& input,
    const StorageView& weight,
    const StorageView* bias,
    StorageView& output,
    const StorageView* qscale) const {
  // Call the primary template directly
  this->Conv1D::compute<Device::DirectML, float16_t>(input, weight, bias,
                                                     output, qscale);
}

// Note: If bfloat16_t is to be supported, a similar specialization would be
// needed. DML has limited or no native bfloat16 support; it might require
// conversion to/from float32 or rely on newer DML versions if support is added.
// For now, only float32 and float16.

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
