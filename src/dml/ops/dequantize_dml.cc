#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/dequantize.h"
#include "dml/backend_dml.h"
#include "dml/common.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

#include <variant>

namespace ctranslate2 {
namespace ops {

template <>
void Dequantize::dequantize<Device::DirectML, int8_t, float>(
    const StorageView& input,
    const StorageView& scale,
    StorageView& output) const {
  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);

  // Step 1: Compute reciprocal of scale (1.0f / scale).
  // This is because DML's DEQUANTIZE op multiplies by scale, while
  // ctranslate2's definition divides. We compute 1/scale and then use it in the
  // DEQUANTIZE op. The RECIP op does not support broadcasting, so input and
  // output tensors must have the same dimensions.
  StorageView reciprocal_scale(scale.shape(), scale.dtype(), scale.device());

  dml::utils::DmlTensorDescBundle scale_desc_bundle(scale);
  dml::utils::DmlTensorDescBundle recip_output_desc_bundle(reciprocal_scale);

  DML_ELEMENT_WISE_RECIP_OPERATOR_DESC recip_desc = {
      .InputTensor = &scale_desc_bundle.get_tensor_desc(),
      .OutputTensor = &recip_output_desc_bundle.get_tensor_desc(),
      .ScaleBias = nullptr};

  DML_OPERATOR_DESC recip_op_desc = {.Type = DML_OPERATOR_ELEMENT_WISE_RECIP,
                                     .Desc = &recip_desc};
  auto recip_compiled_op = dml::GetOrCreateCompiledOperatorApi(&recip_op_desc);

  DML_BUFFER_BINDING scale_binding = dml::utils::create_buffer_binding(
      dml::utils::ResourceFromStorageView(scale), 0,
      scale.size() * sizeof(float));
  std::vector<DML_BINDING_DESC> recip_input_bindings = {
      dml::utils::create_binding_desc(&scale_binding)};

  DML_BUFFER_BINDING recip_output_binding = dml::utils::create_buffer_binding(
      dml::utils::ResourceFromStorageView(reciprocal_scale), 0,
      reciprocal_scale.size() * sizeof(float));
  std::vector<DML_BINDING_DESC> recip_output_bindings = {
      dml::utils::create_binding_desc(&recip_output_binding)};

  recip_compiled_op->Execute(recip_input_bindings, recip_output_bindings);

  // Step 2: Dequantize using DML_ELEMENT_WISE_DEQUANTIZE_LINEAR.
  // This operation implicitly handles the int8 -> float32 cast and multiplies
  // by the (reciprocal) scale. The (reciprocal) scale tensor might need to be
  // broadcast to match the input tensor's shape.

  dml::utils::DmlTensorDescBundle input_desc_bundle(input);

  // The reciprocal scale tensor is broadcast to the input tensor's shape.
  const auto& dml_input_shape_vec = input_desc_bundle.get_sizes_vec();
  std::vector<UINT> pyhsical_scale_shape = dml::utils::to_dml_dims(
      reciprocal_scale.shape(), reciprocal_scale.size());
  pyhsical_scale_shape.push_back(1);
  dml::utils::DmlTensorDescBundle broadcast_reciprocal_scale_desc_bundle(
      dml::utils::get_dml_data_type(reciprocal_scale.dtype()),
      dml_input_shape_vec,   // Target dimensions for broadcasting
      pyhsical_scale_shape,  // Physical (non-broadcast) dimensions
      static_cast<int32_t>(dml_input_shape_vec.size()),  // coerceAxis: >= rank
                                                         // disables flattening
      0,  // placement: no padding with '1's
      0,  // leftAlignedDimensionCount: 0 for right-aligned broadcast
      0,  // minDimensionCount
      0   // guaranteedBaseOffsetAlignment
  );

  dml::utils::DmlTensorDescBundle output_desc_bundle(output);

  DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequantize_desc = {
      .InputTensor = &input_desc_bundle.get_tensor_desc(),
      .ScaleTensor = &broadcast_reciprocal_scale_desc_bundle.get_tensor_desc(),
      .ZeroPointTensor = nullptr,
      .OutputTensor = &output_desc_bundle.get_tensor_desc()};

  DML_OPERATOR_DESC dequantize_op_desc = {
      .Type = DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR,
      .Desc = &dequantize_desc};
  auto dequantize_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&dequantize_op_desc);

  dml::utils::DmlBufferBindingBundle dequant_input_binding(
      dml::utils::ResourceFromStorageView(input), 0,
      input.size() * sizeof(int8_t));
  dml::utils::DmlBufferBindingBundle dequant_scale_binding(
      dml::utils::ResourceFromStorageView(reciprocal_scale), 0,
      reciprocal_scale.size() * sizeof(float));

  dml::utils::DmlBindingArrayBundle dequant_input_bindings(
      {dequant_input_binding, dequant_scale_binding, nullptr});

  dml::utils::DmlBufferBindingBundle dequant_output_binding(
      dml::utils::ResourceFromStorageView(output), 0,
      output.size() * sizeof(float));

  dequantize_compiled_op->Execute(dequant_input_bindings.get_descs(),
                                  {dequant_output_binding.get_desc()});
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
  auto* device = dml::get_device();
  auto* command_list = device->GetCommandList();
  auto* dml_device = dml::get_dml_device();

  dml::utils::DmlTensorDescBundle c_desc_bundle(c);
  dml::utils::DmlTensorDescBundle y_desc_bundle(y);
  const auto& y_dml_sizes = y_desc_bundle.get_sizes_vec();

  auto create_broadcast_desc = [&](const StorageView& tensor) {
    return dml::utils::DmlTensorDescBundle::broadcastFromSeach(tensor,
                                                               y_dml_sizes);
  };

  auto a_scale_desc_bundle = create_broadcast_desc(a_scale);
  auto b_scale_desc_bundle = create_broadcast_desc(b_scale);

  dml::utils::DmlTensorDescBundle intermediate_fp_desc_bundle(
      y.dtype(), y_dml_sizes, nullptr, y.size() * y.item_size());

  StorageView combined_scale(y.shape(), y.dtype(), y.device());
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC scale_mult_desc = {
      .ATensor = &a_scale_desc_bundle.get_tensor_desc(),
      .BTensor = &b_scale_desc_bundle.get_tensor_desc(),
      .OutputTensor = &intermediate_fp_desc_bundle.get_tensor_desc()};
  DML_OPERATOR_DESC scale_mult_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                          &scale_mult_desc};
  auto scale_mult_op = dml::GetOrCreateCompiledOperatorApi(&scale_mult_op_desc);

  StorageView reciprocal_scale(y.shape(), y.dtype(), y.device());
  DML_ELEMENT_WISE_RECIP_OPERATOR_DESC recip_desc = {
      .InputTensor = &intermediate_fp_desc_bundle.get_tensor_desc(),
      .OutputTensor = &intermediate_fp_desc_bundle.get_tensor_desc(),
      .ScaleBias = nullptr};
  DML_OPERATOR_DESC recip_op_desc = {DML_OPERATOR_ELEMENT_WISE_RECIP,
                                     &recip_desc};
  auto recip_op = dml::GetOrCreateCompiledOperatorApi(&recip_op_desc);

  StorageView dequantize_output_buffer;
  dml::utils::DmlTensorDescBundle dequantize_output_desc_bundle;
  const DML_TENSOR_DESC* dequantize_output_desc_ptr;

  if (bias || _activation_type) {
    dequantize_output_buffer = StorageView(y.shape(), y.dtype(), y.device());
    dequantize_output_desc_bundle =
        dml::utils::DmlTensorDescBundle(dequantize_output_buffer);
    dequantize_output_desc_ptr =
        &dequantize_output_desc_bundle.get_tensor_desc();
  } else {
    dequantize_output_desc_ptr = &y_desc_bundle.get_tensor_desc();
  }

  DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequantize_desc = {
      .InputTensor = &c_desc_bundle.get_tensor_desc(),
      .ScaleTensor = &intermediate_fp_desc_bundle.get_tensor_desc(),
      .ZeroPointTensor = nullptr,
      .OutputTensor = dequantize_output_desc_ptr};
  DML_OPERATOR_DESC dequantize_op_desc = {
      DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &dequantize_desc};
  auto dequantize_op = dml::GetOrCreateCompiledOperatorApi(&dequantize_op_desc);

  dml::Operator* bias_add_op = nullptr;
  StorageView bias_add_output_buffer;
  dml::utils::DmlTensorDescBundle bias_add_output_desc_bundle;
  const DML_TENSOR_DESC* bias_add_output_desc_ptr = nullptr;

  const DML_TENSOR_DESC* current_op_input_desc = dequantize_output_desc_ptr;

  if (bias) {
    auto bias_desc_bundle = create_broadcast_desc(*bias);
    bias_add_output_buffer = StorageView(y.shape(), y.dtype(), y.device());
    bias_add_output_desc_bundle =
        dml::utils::DmlTensorDescBundle(bias_add_output_buffer);
    bias_add_output_desc_ptr = &bias_add_output_desc_bundle.get_tensor_desc();

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC bias_add_desc = {
        .ATensor = current_op_input_desc,
        .BTensor = &bias_desc_bundle.get_tensor_desc(),
        .OutputTensor = bias_add_output_desc_ptr};
    DML_OPERATOR_DESC bias_add_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                          &bias_add_desc};
    bias_add_op = dml::GetOrCreateCompiledOperatorApi(&bias_add_op_desc);
    current_op_input_desc = bias_add_output_desc_ptr;
  }

  dml::Operator* activation_op = nullptr;
  DML_OPERATOR_DESC activation_op_desc{};
  std::variant<
      DML_ACTIVATION_RELU_OPERATOR_DESC, DML_ACTIVATION_GELU_OPERATOR_DESC,
      DML_ACTIVATION_SIGMOID_OPERATOR_DESC, DML_ACTIVATION_TANH_OPERATOR_DESC,
      DML_ACTIVATION_SWISH_OPERATOR_DESC, DML_ACTIVATION_IDENTITY_OPERATOR_DESC>
      activation_desc_storage;

  if (_activation_type) {
    switch (*_activation_type) {
      case ActivationType::ReLU:
        activation_desc_storage = DML_ACTIVATION_RELU_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc()};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_RELU,
                              &std::get<DML_ACTIVATION_RELU_OPERATOR_DESC>(
                                  activation_desc_storage)};
        break;
      case ActivationType::GELU:
        activation_desc_storage = DML_ACTIVATION_GELU_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc()};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_GELU,
                              &std::get<DML_ACTIVATION_GELU_OPERATOR_DESC>(
                                  activation_desc_storage)};
        break;
      case ActivationType::Sigmoid:
        activation_desc_storage = DML_ACTIVATION_SIGMOID_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc()};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_SIGMOID,
                              &std::get<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>(
                                  activation_desc_storage)};
        break;
      case ActivationType::Tanh:
        activation_desc_storage = DML_ACTIVATION_TANH_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc()};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_TANH,
                              &std::get<DML_ACTIVATION_TANH_OPERATOR_DESC>(
                                  activation_desc_storage)};
        break;
      case ActivationType::Swish:
        activation_desc_storage = DML_ACTIVATION_SWISH_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc(), 1.0f};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_SWISH,
                              &std::get<DML_ACTIVATION_SWISH_OPERATOR_DESC>(
                                  activation_desc_storage)};
        break;
      default:
        activation_desc_storage = DML_ACTIVATION_IDENTITY_OPERATOR_DESC{
            current_op_input_desc, &y_desc_bundle.get_tensor_desc()};
        activation_op_desc = {DML_OPERATOR_ACTIVATION_IDENTITY,
                              &std::get<DML_ACTIVATION_IDENTITY_OPERATOR_DESC>(
                                  activation_desc_storage)};
    }
    activation_op = dml::GetOrCreateCompiledOperatorApi(&activation_op_desc);
  }

  // --- Execution ---
  dml::utils::DmlBufferBindingBundle a_scale_binding(
      dml::utils::ResourceFromStorageView(a_scale));
  dml::utils::DmlBufferBindingBundle b_scale_binding(
      dml::utils::ResourceFromStorageView(b_scale));
  dml::utils::DmlBufferBindingBundle combined_scale_binding(
      dml::utils::ResourceFromStorageView(combined_scale));
  scale_mult_op->Execute(
      {a_scale_binding.get_desc(), b_scale_binding.get_desc()},
      {combined_scale_binding.get_desc()});

  dml::utils::DmlBufferBindingBundle reciprocal_scale_binding(
      dml::utils::ResourceFromStorageView(reciprocal_scale));
  recip_op->Execute({combined_scale_binding.get_desc()},
                    {reciprocal_scale_binding.get_desc()});

  ID3D12Resource* dequantize_output_resource =
      (bias || _activation_type)
          ? dml::utils::ResourceFromStorageView(dequantize_output_buffer)
          : dml::utils::ResourceFromStorageView(y);
  dml::utils::DmlBufferBindingBundle c_binding(
      dml::utils::ResourceFromStorageView(c));
  dml::utils::DmlBufferBindingBundle dequant_out_binding(
      dequantize_output_resource);
  dequantize_op->Execute(dml::utils::DmlBindingArrayBundle(
                             {c_binding, reciprocal_scale_binding, nullptr})
                             .get_descs(),
                         {dequant_out_binding.get_desc()});

  ID3D12Resource* current_buffer_resource = dequantize_output_resource;

  if (bias_add_op) {
    dml::utils::DmlBufferBindingBundle bias_binding(
        dml::utils::ResourceFromStorageView(*bias));
    dml::utils::DmlBufferBindingBundle bias_add_out_binding(
        dml::utils::ResourceFromStorageView(bias_add_output_buffer));
    bias_add_op->Execute(
        {dequant_out_binding.get_desc(), bias_binding.get_desc()},
        {bias_add_out_binding.get_desc()});
    current_buffer_resource =
        dml::utils::ResourceFromStorageView(bias_add_output_buffer);
  }

  if (activation_op) {
    dml::utils::DmlBufferBindingBundle activation_input_binding(
        current_buffer_resource);
    dml::utils::DmlBufferBindingBundle y_binding(
        dml::utils::ResourceFromStorageView(y));
    activation_op->Execute({activation_input_binding.get_desc()},
                           {y_binding.get_desc()});
  } else if (current_buffer_resource !=
             dml::utils::ResourceFromStorageView(y)) {
    // Final copy if needed
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(
            dml::utils::ResourceFromStorageView(y),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST),
        CD3DX12_RESOURCE_BARRIER::Transition(
            current_buffer_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE)};
    command_list->ResourceBarrier(_countof(barriers), barriers);
    command_list->CopyResource(dml::utils::ResourceFromStorageView(y),
                               current_buffer_resource);
    std::swap(barriers[0].Transition.StateBefore,
              barriers[0].Transition.StateAfter);
    std::swap(barriers[1].Transition.StateBefore,
              barriers[1].Transition.StateAfter);
    command_list->ResourceBarrier(_countof(barriers), barriers);
  }
}

}  // namespace ops
}  // namespace ctranslate2
#endif
