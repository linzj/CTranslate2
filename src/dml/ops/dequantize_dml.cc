#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/dequantize.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <>
void Dequantize::dequantize<Device::DirectML, int8_t, float>(
    const StorageView& input,
    const StorageView& scale,
    StorageView& output) const {
  const dim_t depth = input.dim(-1);

  // Step 1: Compute reciprocal of scale (1.0f / scale).
  // This is because DML's DEQUANTIZE op multiplies by scale, while
  // ctranslate2's definition divides. We compute 1/scale and then use it in the
  // DEQUANTIZE op. The RECIP op does not support broadcasting, so input and
  // output tensors must have the same dimensions.
  StorageView reciprocal_scale(scale.shape(), scale.dtype(), scale.device());

  dml::utils::DmlOperatorDescBundle recip_op_bundle;
  auto& scale_desc = recip_op_bundle.AddInput(scale);
  auto& recip_output_desc = recip_op_bundle.AddOutput(reciprocal_scale);
  auto& recip_desc =
      recip_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>();
  recip_desc.InputTensor = &scale_desc.get_tensor_desc();
  recip_desc.OutputTensor = &recip_output_desc.get_tensor_desc();

  auto recip_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(recip_op_bundle));

  dml::utils::DmlBindingArrayBundle inputs_for_recip_op{
      {dml::utils::ResourceFromStorageView(scale), 0,
       scale.size() * sizeof(float)}};

  dml::utils::DmlBindingArrayBundle outputs_for_recip_op{
      {dml::utils::ResourceFromStorageView(reciprocal_scale), 0,
       reciprocal_scale.size() * sizeof(float)}};

  recip_compiled_op->Execute(inputs_for_recip_op, outputs_for_recip_op);

  // Step 2: Dequantize using DML_ELEMENT_WISE_DEQUANTIZE_LINEAR.
  // This operation implicitly handles the int8 -> float32 cast and multiplies
  // by the (reciprocal) scale. The (reciprocal) scale tensor might need to be
  // broadcast to match the input tensor's shape.

  dml::utils::DmlOperatorDescBundle dequantize_op_bundle;
  auto& input_desc_bundle = dequantize_op_bundle.AddInput(input);

  const auto& dml_input_shape_vec = input_desc_bundle.get_sizes_vec();
  std::vector<UINT> pyhsical_scale_shape = dml::utils::to_dml_dims(
      reciprocal_scale.shape(), reciprocal_scale.size());
  pyhsical_scale_shape.push_back(1);
  auto& broadcast_reciprocal_scale_desc_bundle = dequantize_op_bundle.AddInput(
      dml::utils::get_dml_data_type(reciprocal_scale.dtype()),
      dml_input_shape_vec, pyhsical_scale_shape,
      static_cast<int32_t>(dml_input_shape_vec.size()), 0, 0, 0, 0);

  auto& output_desc_bundle = dequantize_op_bundle.AddOutput(output);

  auto& dequantize_desc =
      dequantize_op_bundle
          .GetOperatorDesc<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>();
  dequantize_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  dequantize_desc.ScaleTensor =
      &broadcast_reciprocal_scale_desc_bundle.get_tensor_desc();
  dequantize_desc.ZeroPointTensor = nullptr;
  dequantize_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();

  auto dequantize_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(dequantize_op_bundle));

  dml::utils::DmlBindingArrayBundle dequant_inputs{
      {dml::utils::ResourceFromStorageView(input), 0,
       input.size() * sizeof(int8_t)},
      {dml::utils::ResourceFromStorageView(reciprocal_scale), 0,
       reciprocal_scale.size() * sizeof(float)},
      {static_cast<IResourceWrapper*>(nullptr), 0, 0}
      // explicit nullptr for ZeroPointTensor
  };

  dml::utils::DmlBindingArrayBundle dequant_outputs{
      {dml::utils::ResourceFromStorageView(output), 0,
       output.size() * sizeof(float)}};

  dequantize_compiled_op->Execute(dequant_inputs, dequant_outputs);
}  // namespace ops

template <>
void Dequantize::dequantize_gemm_output<Device::DirectML, float>(
    const StorageView& c,
    const StorageView& a_scale,
    const StorageView& b_scale,
    const bool transpose_a,
    const bool transpose_b,
    const StorageView* bias,
    StorageView& y) const {
  const std::vector<UINT> y_dml_sizes = [&y]() {
    dml::utils::DmlOperatorDescBundle temp_op_bundle;
    auto& y_desc_bundle_ref = temp_op_bundle.AddOutput(y);
    return y_desc_bundle_ref.get_sizes_vec();
  }();

  StorageView combined_scale(y.shape(), y.dtype(), y.device());
  dml::utils::DmlOperatorDescBundle scale_mult_op_bundle;
  auto& a_scale_desc_bundle =
      scale_mult_op_bundle.AddInputBroadcastFromSeach(a_scale, y_dml_sizes);
  auto& b_scale_desc_bundle =
      scale_mult_op_bundle.AddInputBroadcastFromSeach(b_scale, y_dml_sizes);
  auto& intermediate_fp_desc_bundle =
      scale_mult_op_bundle.AddOutput(combined_scale);
  auto& scale_mult_desc =
      scale_mult_op_bundle
          .GetOperatorDesc<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>();
  scale_mult_desc.ATensor = &a_scale_desc_bundle.get_tensor_desc();
  scale_mult_desc.BTensor = &b_scale_desc_bundle.get_tensor_desc();
  scale_mult_desc.OutputTensor = &intermediate_fp_desc_bundle.get_tensor_desc();
  auto scale_mult_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(scale_mult_op_bundle));

  StorageView dequantize_output_buffer;
  dml::utils::DmlOperatorDescBundle dequantize_op_bundle;
  auto& c_desc = dequantize_op_bundle.AddInput(c);
  auto& scale_desc = dequantize_op_bundle.AddInput(combined_scale);
  const DML_TENSOR_DESC* dequantize_output_desc_ptr;
  if (bias || _activation_type) {
    dequantize_output_buffer = StorageView(y.shape(), y.dtype(), y.device());
    dequantize_output_desc_ptr =
        &dequantize_op_bundle.AddOutput(dequantize_output_buffer)
             .get_tensor_desc();
  } else {
    dequantize_output_desc_ptr =
        &dequantize_op_bundle.AddOutput(y).get_tensor_desc();
  }
  auto& dequantize_desc =
      dequantize_op_bundle
          .GetOperatorDesc<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>();
  dequantize_desc.InputTensor = &c_desc.get_tensor_desc();
  dequantize_desc.ScaleTensor = &scale_desc.get_tensor_desc();
  dequantize_desc.OutputTensor = dequantize_output_desc_ptr;
  auto dequantize_op =
      dml::GetOrCreateCompiledOperatorApi(std::move(dequantize_op_bundle));

  dml::Operator* bias_add_op = nullptr;
  StorageView bias_add_output_buffer;
  const DML_TENSOR_DESC* bias_add_output_desc_ptr = nullptr;
  const DML_TENSOR_DESC* current_op_input_desc = dequantize_output_desc_ptr;

  if (bias) {
    dml::utils::DmlOperatorDescBundle bias_add_op_bundle;
    auto& bias_input_a_desc = bias_add_op_bundle.AddInput(
        dequantize_output_buffer);  // A is the output of previous op
    auto& bias_input_b_desc =
        bias_add_op_bundle.AddInputBroadcastFromSeach(*bias, y_dml_sizes);

    if (_activation_type) {
      bias_add_output_buffer = StorageView(y.shape(), y.dtype(), y.device());
    } else {
      bias_add_output_buffer.view(y.buffer(), y.shape());
    }
    bias_add_output_desc_ptr =
        &bias_add_op_bundle.AddOutput(bias_add_output_buffer).get_tensor_desc();

    auto& bias_add_desc =
        bias_add_op_bundle
            .GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    bias_add_desc.ATensor = &bias_input_a_desc.get_tensor_desc();
    bias_add_desc.BTensor = &bias_input_b_desc.get_tensor_desc();
    bias_add_desc.OutputTensor = bias_add_output_desc_ptr;
    bias_add_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(bias_add_op_bundle));
    current_op_input_desc = bias_add_output_desc_ptr;
  }

  dml::Operator* activation_op = nullptr;
  dml::utils::DmlOperatorDescBundle activation_op_bundle;
  if (_activation_type) {
    StorageView* current_op_input_buffer =
        bias ? &bias_add_output_buffer : &dequantize_output_buffer;
    auto& activation_input_desc =
        activation_op_bundle.AddInput(*current_op_input_buffer);
    auto& activation_output_desc = activation_op_bundle.AddOutput(y);

#define GET_ACTIVATION_DESC(type)                                             \
  auto& desc = activation_op_bundle                                           \
                   .GetOperatorDesc<DML_ACTIVATION_##type##_OPERATOR_DESC>(); \
  desc.InputTensor = &activation_input_desc.get_tensor_desc();                \
  desc.OutputTensor = &activation_output_desc.get_tensor_desc()

    switch (*_activation_type) {
      case ActivationType::ReLU: {
        GET_ACTIVATION_DESC(RELU);
        break;
      }
      case ActivationType::GELU: {
        GET_ACTIVATION_DESC(GELU);
        break;
      }
      case ActivationType::Sigmoid: {
        GET_ACTIVATION_DESC(SIGMOID);
        break;
      }
      case ActivationType::Tanh: {
        GET_ACTIVATION_DESC(TANH);
        break;
      }
      case ActivationType::Swish: {
        auto& desc = activation_op_bundle
                         .GetOperatorDesc<DML_ACTIVATION_SWISH_OPERATOR_DESC>();
        desc.InputTensor = &activation_input_desc.get_tensor_desc();
        desc.OutputTensor = &activation_output_desc.get_tensor_desc();
        desc.SigmoidInputScale = 1.0f;
        break;
      }
      default: {
        GET_ACTIVATION_DESC(IDENTITY);
        break;
      }
    }
#undef GET_ACTIVATION_DESC
    activation_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(activation_op_bundle));
  }

  // --- Execution ---
  dml::utils::DmlBindingArrayBundle scale_mult_inputs{
      {dml::utils::ResourceFromStorageView(a_scale), 0, 0},
      {dml::utils::ResourceFromStorageView(b_scale), 0, 0}};
  dml::utils::DmlBindingArrayBundle scale_mult_outputs{
      {dml::utils::ResourceFromStorageView(combined_scale), 0, 0}};
  scale_mult_op->Execute(scale_mult_inputs, scale_mult_outputs);

  IResourceWrapper* dequantize_output_resource =
      (bias || _activation_type)
          ? dml::utils::ResourceFromStorageView(dequantize_output_buffer)
          : dml::utils::ResourceFromStorageView(y);
  dml::utils::DmlBindingArrayBundle dequantize_inputs_combined{
      {dml::utils::ResourceFromStorageView(c), 0, 0},
      {dml::utils::ResourceFromStorageView(combined_scale), 0, 0},
      {static_cast<IResourceWrapper*>(nullptr), 0, 0}};  // ZeroPointTensor
  dml::utils::DmlBindingArrayBundle dequantize_outputs_combined{
      {dequantize_output_resource, 0, 0}};
  dequantize_op->Execute(dequantize_inputs_combined,
                         dequantize_outputs_combined);

  IResourceWrapper* current_buffer_resource = dequantize_output_resource;

  if (bias_add_op) {
    dml::utils::DmlBindingArrayBundle bias_add_inputs{
        {dequantize_output_resource, 0, 0},
        {dml::utils::ResourceFromStorageView(*bias), 0, 0}};
    dml::utils::DmlBindingArrayBundle bias_add_outputs{
        {dml::utils::ResourceFromStorageView(bias_add_output_buffer), 0, 0}};
    bias_add_op->Execute(bias_add_inputs, bias_add_outputs);
    current_buffer_resource =
        dml::utils::ResourceFromStorageView(bias_add_output_buffer);
  }

  if (activation_op) {
    dml::utils::DmlBindingArrayBundle activation_inputs{
        {current_buffer_resource, 0, 0}};
    dml::utils::DmlBindingArrayBundle activation_outputs{
        {dml::utils::ResourceFromStorageView(y), 0, 0}};
    activation_op->Execute(activation_inputs, activation_outputs);
  } else if (current_buffer_resource !=
             dml::utils::ResourceFromStorageView(y)) {
    throw std::invalid_argument(
        "Dequantize GEMM output: current buffer resource does not match ");
  }
}

}  // namespace ops
}  // namespace ctranslate2
#endif
