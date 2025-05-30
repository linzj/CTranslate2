#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/quantize.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename InT, typename OutT>
void Quantize::quantize(const StorageView& input,
                        StorageView& output,
                        StorageView& scale) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");
  static_assert(std::is_same_v<InT, float>, "Input type must be float");
  static_assert(std::is_same_v<OutT, int8_t>, "Output type must be int8_t");

  if (_shift_to_uint8) {
    throw std::invalid_argument(
        "Shift to uint8_t is not supported on DirectML");
  }

  // Device checks already performed by the framework or Op a higher level

  auto* device = dml::get_device();

  const dim_t batch_size = scale.size();  // Assuming scale is [batch_size]
  const dim_t depth = input.dim(-1);      // Assuming input is [batch_size, ...,
                                          // depth] or effectively [N, depth]

  // Shapes for DML. Assuming input is effectively 2D [batch_size, depth] for
  // this op. Scale is [batch_size, 1] for broadcasting. Output is [batch_size,
  // depth]
  std::vector<UINT> dml_input_sizes = {static_cast<UINT>(batch_size),
                                       static_cast<UINT>(depth)};
  std::vector<UINT> dml_input_strides = {static_cast<UINT>(depth), 1};

  std::vector<UINT> dml_scale_sizes = {static_cast<UINT>(batch_size), 1};
  // Strides for [batch_size, 1] for broadcasting with [batch_size, depth]
  // typically {1,0} or {1,1} if DML handles last dim broadcast. If scale is
  // truly just [batch_size] StorageView, its DmlTensorDescBundle with shape
  // [batch_size,1] and default strides would be {1,1} effectively
  // DML_MEAN_VARIANCE_NORMALIZATION expects scale to align with normalized
  // axes. Here we make it explicit [batch, 1].
  std::vector<UINT> dml_scale_strides = {
      1, 0};  // Broadcast along the depth dimension for scale.

  // Tensor Descriptors using DmlTensorDescBundle
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_input_sizes, &dml_input_strides,
      input.size() * sizeof(float));
  // For scale, its original StorageView 'scale' is likely [batch_size]. We
  // describe it to DML as [batch_size, 1] for broadcasting. The
  // TotalTensorSizeInBytes should still reflect the actual 'scale' StorageView
  // size.
  dml::utils::DmlTensorDescBundle scale_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_scale_sizes, &dml_scale_strides,
      scale.size() * sizeof(float));
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      DML_TENSOR_DATA_TYPE_INT8, dml_input_sizes, &dml_input_strides,
      output.size() * sizeof(int8_t));

  // Constant tensor descriptors
  std::vector<UINT> const_dims = {1};
  dml::utils::DmlTensorDescBundle const_127_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, const_dims, nullptr, sizeof(float));
  dml::utils::DmlTensorDescBundle const_epsilon_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, const_dims, nullptr, sizeof(float));

  // Get D3D12 resources from StorageView buffers
  ID3D12Resource* input_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  ID3D12Resource* output_resource =
      reinterpret_cast<ID3D12Resource*>(output.buffer());
  ID3D12Resource* scale_resource_target = reinterpret_cast<ID3D12Resource*>(
      scale.buffer());  // This is where final scale is written

  // Create intermediate buffers
  // These DmlTensorDescBundles will describe the layout of these intermediate
  // buffers. Their shapes will typically match dml_input_sizes or
  // dml_scale_sizes.
  dml::utils::DmlTensorDescBundle abs_buffer_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_input_sizes, &dml_input_strides);
  auto abs_buffer = device->CreatePreferredDeviceMemoryBuffer(
      abs_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  dml::utils::DmlTensorDescBundle max_buffer_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_scale_sizes,
      &dml_scale_strides);  // max is [batch_size, 1] like scale
  auto max_buffer = device->CreatePreferredDeviceMemoryBuffer(
      max_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // scale_factor_buffer will hold result of max_val with epsilon
  dml::utils::DmlTensorDescBundle scale_factor_buffer_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_scale_sizes, &dml_scale_strides);
  auto scale_factor_buffer = device->CreatePreferredDeviceMemoryBuffer(
      scale_factor_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  dml::utils::DmlTensorDescBundle scaled_buffer_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_input_sizes, &dml_input_strides);
  auto scaled_buffer = device->CreatePreferredDeviceMemoryBuffer(
      scaled_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  dml::utils::DmlTensorDescBundle rounded_buffer_desc_bundle(
      DML_TENSOR_DATA_TYPE_FLOAT32, dml_input_sizes, &dml_input_strides);
  auto rounded_buffer = device->CreatePreferredDeviceMemoryBuffer(
      rounded_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // Create constant buffers on GPU
  const float constant_127_val = 127.0f;
  const float constant_epsilon_val = 1e-8f;

  DML_SCALAR_UNION scalar_127;
  scalar_127.Float32 = constant_127_val;
  dml::utils::DmlTensorDescBundle
      bundle_127;  // Dummy, will be populated by CreateDmlConstantTensor
  auto const_127_buffer_res = dml::utils::CreateDmlConstantTensor(
      device, {1}, DML_TENSOR_DATA_TYPE_FLOAT32, scalar_127, bundle_127);

  DML_SCALAR_UNION scalar_eps;
  scalar_eps.Float32 = constant_epsilon_val;
  dml::utils::DmlTensorDescBundle bundle_eps;  // Dummy
  auto const_epsilon_buffer_res = dml::utils::CreateDmlConstantTensor(
      device, {1}, DML_TENSOR_DATA_TYPE_FLOAT32, scalar_eps, bundle_eps);

  // DML_BUFFER_BINDING storage for Execute calls
  DML_BUFFER_BINDING exec_bindings[3];

  // Step 1: Compute absolute values: output to abs_buffer
  DML_ELEMENT_WISE_ABS_OPERATOR_DESC abs_op_payload = {};
  abs_op_payload.InputTensor = &input_desc_bundle.get_tensor_desc();
  abs_op_payload.OutputTensor =
      &abs_buffer_desc_bundle
           .get_tensor_desc();  // Output to intermediate abs_buffer
  DML_OPERATOR_DESC abs_op_wrapper = {DML_OPERATOR_ELEMENT_WISE_ABS,
                                      &abs_op_payload};
  auto abs_compiled_op = dml::GetOrCreateCompiledOperatorApi(&abs_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        input_resource, 0,
        input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        abs_buffer.Get(), 0,
        abs_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    abs_compiled_op->Execute(
        {dml::utils::create_binding_desc(&exec_bindings[0])},
        {dml::utils::create_binding_desc(&exec_bindings[1])});
  }

  // Step 2: Reduce to find max along depth dimension (axis 1 of [batch,
  // depth]): output to max_buffer
  UINT reduce_axes[] = {1};  // Reduce along the 'depth' dimension
  DML_REDUCE_OPERATOR_DESC reduce_op_payload = {};
  reduce_op_payload.Function = DML_REDUCE_FUNCTION_MAX;
  reduce_op_payload.InputTensor =
      &abs_buffer_desc_bundle.get_tensor_desc();  // Input from abs_buffer
  reduce_op_payload.OutputTensor =
      &max_buffer_desc_bundle
           .get_tensor_desc();  // Output to intermediate max_buffer
  reduce_op_payload.AxisCount = 1;
  reduce_op_payload.Axes = reduce_axes;
  DML_OPERATOR_DESC reduce_op_wrapper = {DML_OPERATOR_REDUCE,
                                         &reduce_op_payload};
  auto reduce_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&reduce_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        abs_buffer.Get(), 0,
        abs_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        max_buffer.Get(), 0,
        max_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    reduce_compiled_op->Execute(
        {dml::utils::create_binding_desc(&exec_bindings[0])},
        {dml::utils::create_binding_desc(&exec_bindings[1])});
  }

  // Step 3: Avoid division by zero - max(max_val, epsilon): output to
  // scale_factor_buffer
  DML_ELEMENT_WISE_MAX_OPERATOR_DESC max_eps_op_payload = {};
  max_eps_op_payload.ATensor =
      &max_buffer_desc_bundle.get_tensor_desc();  // max_val from reduce
  max_eps_op_payload.BTensor =
      &bundle_eps.get_tensor_desc();  // constant_epsilon
  max_eps_op_payload.OutputTensor =
      &scale_factor_buffer_desc_bundle
           .get_tensor_desc();  // Output to scale_factor_buffer
  DML_OPERATOR_DESC max_eps_op_wrapper = {DML_OPERATOR_ELEMENT_WISE_MAX,
                                          &max_eps_op_payload};
  auto max_eps_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&max_eps_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        max_buffer.Get(), 0,
        max_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        const_epsilon_buffer_res.Get(), 0,
        bundle_eps.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[2] = dml::utils::create_buffer_binding(
        scale_factor_buffer.Get(), 0,
        scale_factor_buffer_desc_bundle.get_buffer_desc()
            .TotalTensorSizeInBytes);
    std::vector<DML_BINDING_DESC> inputs = {
        dml::utils::create_binding_desc(&exec_bindings[0]),
        dml::utils::create_binding_desc(&exec_bindings[1])};
    max_eps_compiled_op->Execute(
        inputs, {dml::utils::create_binding_desc(&exec_bindings[2])});
  }

  // Step 4: Compute scale factors: 127.0 / max_val_with_epsilon: output to
  // scale_resource_target (final scale StorageView)
  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC div_op_payload = {};
  div_op_payload.ATensor = &bundle_127.get_tensor_desc();  // constant_127
  div_op_payload.BTensor =
      &scale_factor_buffer_desc_bundle
           .get_tensor_desc();  // max_val from previous step
  div_op_payload.OutputTensor =
      &scale_desc_bundle.get_tensor_desc();  // Output to final scale buffer
  DML_OPERATOR_DESC div_op_wrapper = {DML_OPERATOR_ELEMENT_WISE_DIVIDE,
                                      &div_op_payload};
  auto div_compiled_op = dml::GetOrCreateCompiledOperatorApi(&div_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        const_127_buffer_res.Get(), 0,
        bundle_127.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        scale_factor_buffer.Get(), 0,
        scale_factor_buffer_desc_bundle.get_buffer_desc()
            .TotalTensorSizeInBytes);
    exec_bindings[2] = dml::utils::create_buffer_binding(
        scale_resource_target, 0,
        scale_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    std::vector<DML_BINDING_DESC> inputs = {
        dml::utils::create_binding_desc(&exec_bindings[0]),
        dml::utils::create_binding_desc(&exec_bindings[1])};
    div_compiled_op->Execute(
        inputs, {dml::utils::create_binding_desc(&exec_bindings[2])});
  }

  // Step 5: Broadcast and multiply input by scale factors: output to
  // scaled_buffer
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mult_op_payload = {};
  mult_op_payload.ATensor =
      &input_desc_bundle.get_tensor_desc();  // original input
  mult_op_payload.BTensor =
      &scale_desc_bundle
           .get_tensor_desc();  // final scale factors (broadcasted)
  mult_op_payload.OutputTensor =
      &scaled_buffer_desc_bundle.get_tensor_desc();  // Output to scaled_buffer
  DML_OPERATOR_DESC mult_op_wrapper = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                       &mult_op_payload};
  auto mult_compiled_op = dml::GetOrCreateCompiledOperatorApi(&mult_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        input_resource, 0,
        input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        scale_resource_target, 0,
        scale_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    exec_bindings[2] = dml::utils::create_buffer_binding(
        scaled_buffer.Get(), 0,
        scaled_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    std::vector<DML_BINDING_DESC> inputs = {
        dml::utils::create_binding_desc(&exec_bindings[0]),
        dml::utils::create_binding_desc(&exec_bindings[1])};
    mult_compiled_op->Execute(
        inputs, {dml::utils::create_binding_desc(&exec_bindings[2])});
  }

  // Step 6: Optional rounding
  ID3D12Resource* next_input_for_cast = scaled_buffer.Get();
  const dml::utils::DmlTensorDescBundle* next_input_desc_bundle_for_cast =
      &scaled_buffer_desc_bundle;

  if (_round_before_cast) {
    DML_ELEMENT_WISE_ROUND_OPERATOR_DESC round_op_payload = {};
    round_op_payload.InputTensor =
        &scaled_buffer_desc_bundle
             .get_tensor_desc();  // Input from scaled_buffer
    round_op_payload.OutputTensor =
        &rounded_buffer_desc_bundle
             .get_tensor_desc();  // Output to rounded_buffer
    round_op_payload.RoundingMode = DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN;
    DML_OPERATOR_DESC round_op_wrapper = {DML_OPERATOR_ELEMENT_WISE_ROUND,
                                          &round_op_payload};
    auto round_compiled_op =
        dml::GetOrCreateCompiledOperatorApi(&round_op_wrapper);
    {
      exec_bindings[0] = dml::utils::create_buffer_binding(
          scaled_buffer.Get(), 0,
          scaled_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
      exec_bindings[1] = dml::utils::create_buffer_binding(
          rounded_buffer.Get(), 0,
          rounded_buffer_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
      round_compiled_op->Execute(
          {dml::utils::create_binding_desc(&exec_bindings[0])},
          {dml::utils::create_binding_desc(&exec_bindings[1])});
    }
    next_input_for_cast = rounded_buffer.Get();
    next_input_desc_bundle_for_cast = &rounded_buffer_desc_bundle;
  }

  // Step 7: Cast to int8
  DML_CAST_OPERATOR_DESC cast_op_payload = {};
  cast_op_payload.InputTensor =
      &next_input_desc_bundle_for_cast->get_tensor_desc();
  cast_op_payload.OutputTensor =
      &output_desc_bundle.get_tensor_desc();  // Final output
  DML_OPERATOR_DESC cast_op_wrapper = {DML_OPERATOR_CAST, &cast_op_payload};
  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_wrapper);
  {
    exec_bindings[0] = dml::utils::create_buffer_binding(
        next_input_for_cast, 0,
        next_input_desc_bundle_for_cast->get_buffer_desc()
            .TotalTensorSizeInBytes);
    exec_bindings[1] = dml::utils::create_buffer_binding(
        output_resource, 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    cast_compiled_op->Execute(
        {dml::utils::create_binding_desc(&exec_bindings[0])},
        {dml::utils::create_binding_desc(&exec_bindings[1])});
  }
}

// Explicit template instantiation
template void Quantize::quantize<Device::DirectML, float, int8_t>(
    const StorageView&,
    StorageView&,
    StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2
#endif