#ifdef CT2_WITH_DIRECTML  // Guard for the whole file

#include "ctranslate2/ops/gumbel_max.h"

#include <limits>

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

#define GUMBEL_MAX_CHECK(ans, msg)                                            \
  {                                                                           \
    bool r = ans;                                                             \
    if (r != true)                                                            \
      throw std::runtime_error(std::string("GUMBEL MAX failed with error ") + \
                               msg);                                          \
  }

namespace ctranslate2 {
namespace ops {

// Anonymous namespace with local helpers (get_dml_data_type, get_dml_dims,
// dml_data_type_size_bytes, make_buffer_tensor_desc,
// create_dml_buffer_tensor_desc, CreateDmlConstantTensor) removed. These are
// now replaced by functions/classes in dml::utils.

template <Device D, typename T>
void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
  static_assert(D == Device::DirectML,
                "Device must be DirectML for this implementation.");

  // Runtime checks for device, shape, and type consistency
  GUMBEL_MAX_CHECK(
      x.device() == Device::DirectML && y.device() == Device::DirectML,
      "Input and output tensors must be on the DirectML device.");
  GUMBEL_MAX_CHECK(x.shape() == y.shape(),
                   "Input and output tensor shapes must match.");
  GUMBEL_MAX_CHECK(x.dtype() == y.dtype(),
                   "Input and output tensor data types must match.");
  GUMBEL_MAX_CHECK(x.size() > 0, "Input tensor cannot be empty.");

  dml::Device* device =
      dml::get_device();  // Get the ctranslate2 DML device wrapper

  y.resize(x.shape());  // Ensure output is allocated

  // Determine data types
  DML_TENSOR_DATA_TYPE dml_input_type =
      dml::utils::get_dml_data_type(x.dtype());
  DML_TENSOR_DATA_TYPE dml_compute_type = DML_TENSOR_DATA_TYPE_FLOAT32;

  // Get DML-compatible dimensions
  // Pass x.size() to to_dml_dims to handle scalar case correctly.
  std::vector<UINT> dml_dims_vec = dml::utils::to_dml_dims(x.shape(), x.size());

  // --- Prepare Tensor Descriptors and GPU Resources ---
  // Use DmlTensorDescBundle to manage descriptors and their underlying
  // std::vector for dims.

  dml::utils::DmlOperatorDescBundle tensor_provider_bundle;
  const auto& x_desc_bundle = tensor_provider_bundle.AddInput(x);
  const auto& y_desc_bundle = tensor_provider_bundle.AddOutput(y);

  // Intermediate resources for computation in FLOAT32
  // uniform_fp32_resource will hold uniform random numbers (0,1]
  const auto& uniform_fp32_desc_bundle = tensor_provider_bundle.AddInput(
      dml_compute_type, dml_dims_vec, nullptr, 0);
  auto uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // log_uniform_fp32_resource will hold log(uniform_fp32_resource)
  const auto& log_uniform_fp32_desc_bundle = tensor_provider_bundle.AddInput(
      dml_compute_type, dml_dims_vec, nullptr, 0);
  auto log_uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      log_uniform_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // gumbel_noise_fp32_resource will hold -log(uniform_fp32_resource)
  const auto& gumbel_noise_fp32_desc_bundle = tensor_provider_bundle.AddInput(
      dml_compute_type, dml_dims_vec, nullptr, 0);
  auto gumbel_noise_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      gumbel_noise_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  bool cast_input_to_fp32 = (dml_input_type != dml_compute_type);
  Microsoft::WRL::ComPtr<IResourceWrapper>
      x_fp32_intermediate_resource;  // Holds x cast to FP32
  const dml::utils::DmlTensorDescBundle* x_fp32_desc_bundle_ptr;

  const DML_TENSOR_DESC* current_x_tensor_desc_for_op_ptr;
  ID3D12Resource* current_x_resource_for_op_ptr;

  if (cast_input_to_fp32) {
    x_fp32_desc_bundle_ptr = &tensor_provider_bundle.AddInput(
        dml_compute_type, dml_dims_vec, nullptr, 0);
    x_fp32_intermediate_resource = device->CreatePreferredDeviceMemoryBuffer(
        x_fp32_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);

    current_x_tensor_desc_for_op_ptr =
        &x_fp32_desc_bundle_ptr->get_tensor_desc();
    current_x_resource_for_op_ptr =
        x_fp32_intermediate_resource->GetD3D12Resource();
  } else {
    current_x_tensor_desc_for_op_ptr = &x_desc_bundle.get_tensor_desc();
    current_x_resource_for_op_ptr = dml::utils::ResourceFromStorageView(x);
  }

  Microsoft::WRL::ComPtr<IResourceWrapper> sum_fp32_intermediate_resource;
  const dml::utils::DmlTensorDescBundle* sum_fp32_desc_bundle_ptr;

  ID3D12Resource* target_sum_resource_for_op_ptr;
  const DML_TENSOR_DESC* target_sum_tensor_desc_for_op_ptr;

  if (cast_input_to_fp32) {  // If x was not FP32, y is not FP32. Sum result
                             // needs intermediate FP32 buffer before casting
                             // back.
    sum_fp32_desc_bundle_ptr = &tensor_provider_bundle.AddInput(
        dml_compute_type, dml_dims_vec, nullptr, 0);
    sum_fp32_intermediate_resource = device->CreatePreferredDeviceMemoryBuffer(
        sum_fp32_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);
    target_sum_resource_for_op_ptr =
        sum_fp32_intermediate_resource->GetD3D12Resource();
    target_sum_tensor_desc_for_op_ptr =
        &sum_fp32_desc_bundle_ptr->get_tensor_desc();
  } else {  // If x is FP32, y is also FP32. Sum directly into y's resource.
    target_sum_resource_for_op_ptr = dml::utils::ResourceFromStorageView(y);
    target_sum_tensor_desc_for_op_ptr = &y_desc_bundle.get_tensor_desc();
  }

  // --- Operator Execution Sequence ---

  if (cast_input_to_fp32) {
    dml::utils::DmlOperatorDescBundle cast_to_fp32_op;
    auto& cast_to_fp32_desc =
        cast_to_fp32_op.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
    cast_to_fp32_desc.InputTensor = &x_desc_bundle.get_tensor_desc();
    cast_to_fp32_desc.OutputTensor =
        current_x_tensor_desc_for_op_ptr;  // This is
                                           // x_fp32_desc_bundle_ptr->get_tensor_desc()

    dml::GetOrCreateCompiledOperatorApi(std::move(cast_to_fp32_op))
        ->Execute({{dml::utils::ResourceFromStorageView(x), 0,
                    x_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes}},
                  {{current_x_resource_for_op_ptr, 0,
                    x_fp32_desc_bundle_ptr->get_buffer_desc()
                        .TotalTensorSizeInBytes}});
  }

  // Step 1.1: Generate Uniform UINT32 Random Numbers
  const auto& uniform_uint32_desc_bundle = tensor_provider_bundle.AddInput(
      DML_TENSOR_DATA_TYPE_UINT32, dml_dims_vec, nullptr, 0);
  auto uniform_uint32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_uint32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  std::vector<UINT> state_dims_cpu = {
      6};  // PHILOX_4X32_10 state size is 6 (4 for counter, 2 for key)
  const auto& state_desc_bundle = tensor_provider_bundle.AddInput(
      DML_TENSOR_DATA_TYPE_UINT32, state_dims_cpu, nullptr, 0);
  // Input state resource (assumed zero-initialized by D3D runtime for the first
  // call)
  auto input_random_generator_state_resource =
      device->CreatePreferredDeviceMemoryBuffer(
          state_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // Output state resource
  auto output_random_generator_state_resource =
      device->CreatePreferredDeviceMemoryBuffer(
          state_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  // For a sequence of calls, these two resources would be swapped.
  // TODO: Consider explicit zero-initialization for

  dml::utils::DmlOperatorDescBundle random_op_bundle;
  auto& random_op_desc_payload =
      random_op_bundle.GetOperatorDesc<DML_RANDOM_GENERATOR_OPERATOR_DESC>();
  random_op_desc_payload.InputStateTensor =
      &state_desc_bundle.get_tensor_desc();  // Must be non-null
  random_op_desc_payload.OutputTensor =
      &uniform_uint32_desc_bundle.get_tensor_desc();
  random_op_desc_payload.OutputStateTensor =
      &state_desc_bundle.get_tensor_desc();
  random_op_desc_payload.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;

  // Bind to two separate resources for input and output state
  dml::utils::DmlBindingArrayBundle random_op_inputs{
      {input_random_generator_state_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle random_op_outputs{
      {uniform_uint32_resource->GetD3D12Resource(), 0, 0},
      {output_random_generator_state_resource->GetD3D12Resource(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(random_op_bundle))
      ->Execute(random_op_inputs, random_op_outputs);

  // Step 1.2: Cast UINT32 random to FLOAT32 -> uniform_fp32_resource
  dml::utils::DmlOperatorDescBundle cast_uint_op_bundle;
  auto& cast_uint_to_fp32_desc =
      cast_uint_op_bundle.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
  cast_uint_to_fp32_desc.InputTensor =
      &uniform_uint32_desc_bundle.get_tensor_desc();
  cast_uint_to_fp32_desc.OutputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();

  dml::utils::DmlBindingArrayBundle cast_uint_inputs{
      {uniform_uint32_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle cast_uint_outputs{
      {uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(cast_uint_op_bundle))
      ->Execute(cast_uint_inputs, cast_uint_outputs);

  // Step 1.3: Scale FLOAT32 random numbers to (0, 1]
  DML_SCALAR_UNION scale_val_scalar;
  scale_val_scalar.Float32 =
      1.0f / static_cast<float>(std::numeric_limits<uint32_t>::max());

  // Apply fused linear transformation: Result = scale_factor * Input +
  // (epsilon_const * scale_factor) 'eps_val_scalar' (providing epsilon_const =
  // 1e-9f) is defined here. 'scale_val_scalar' (providing scale_factor = 1.0f /
  // uint32_max) is defined above (lines 223-225).
  DML_SCALAR_UNION eps_val_scalar;
  eps_val_scalar.Float32 = 1e-9f;

  dml::utils::DmlOperatorDescBundle fused_linear_op_bundle;
  auto& fused_linear_op_payload =
      fused_linear_op_bundle
          .GetOperatorDesc<DML_ACTIVATION_LINEAR_OPERATOR_DESC>();
  fused_linear_op_payload.InputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();
  fused_linear_op_payload.OutputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();  // In-place
  fused_linear_op_payload.Alpha =
      scale_val_scalar.Float32;  // This is 'scale_factor'
  fused_linear_op_payload.Beta =
      eps_val_scalar.Float32 *
      scale_val_scalar.Float32;  // This is 'epsilon_const * scale_factor'

  dml::utils::DmlBindingArrayBundle fused_linear_inputs{
      {uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle fused_linear_outputs{
      {uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(fused_linear_op_bundle))
      ->Execute(fused_linear_inputs, fused_linear_outputs);

  // Step 2: Compute Logarithm
  dml::utils::DmlOperatorDescBundle log_op_bundle;
  auto& log_desc =
      log_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_LOG_OPERATOR_DESC>();
  log_desc.InputTensor = &uniform_fp32_desc_bundle.get_tensor_desc();
  log_desc.OutputTensor = &log_uniform_fp32_desc_bundle.get_tensor_desc();

  dml::utils::DmlBindingArrayBundle log_op_inputs{
      {uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle log_op_outputs{
      {log_uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(log_op_bundle))
      ->Execute(log_op_inputs, log_op_outputs);

  // Step 3: Negate
#if DML_TARGET_VERSION >= 0x5000 && !defined(CT2_DML_USE_MULTIPLY_FOR_NEGATE)
  dml::utils::DmlOperatorDescBundle negate_op_bundle;
  auto& negate_desc_payload =
      negate_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC>();
  negate_desc_payload.InputTensor =
      &log_uniform_fp32_desc_bundle.get_tensor_desc();
  negate_desc_payload.OutputTensor =
      &gumbel_noise_fp32_desc_bundle.get_tensor_desc();

  dml::utils::DmlBindingArrayBundle negate_op_inputs{
      {log_uniform_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle negate_op_outputs{
      {gumbel_noise_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(negate_op_bundle))
      ->Execute(negate_op_inputs, negate_op_outputs);
#else  // Fallback: Multiply by -1
  DML_SCALAR_UNION m1_val_scalar;
  m1_val_scalar.Float32 = -1.0f;
  // Use standalone DML_OPERATOR_ACTIVATION_LINEAR for negate fallback
  dml::utils::DmlOperatorDescBundle negate_op_bundle;
  auto& negate_op_payload =
      negate_op_bundle.GetOperatorDesc<DML_ACTIVATION_LINEAR_OPERATOR_DESC>();
  negate_op_payload.InputTensor =
      &log_uniform_fp32_desc_bundle.get_tensor_desc();
  negate_op_payload.OutputTensor =
      &gumbel_noise_fp32_desc_bundle
           .get_tensor_desc();                      // Not in-place for negate
  negate_op_payload.Alpha = m1_val_scalar.Float32;  // -1.0f
  negate_op_payload.Beta = 0.0f;

  dml::utils::DmlBindingArrayBundle neg_fallback_inputs{
      {log_uniform_fp32_resource.Get(), 0, 0}};
  dml::utils::DmlBindingArrayBundle neg_fallback_outputs{
      {gumbel_noise_fp32_resource.Get(), 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(negate_op_bundle))
      ->Execute(neg_fallback_inputs, neg_fallback_outputs);
#endif

  // Step 4: Add Gumbel noise to input x
  dml::utils::DmlOperatorDescBundle add_noise_op_bundle;
  auto& add_noise_desc =
      add_noise_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
  add_noise_desc.ATensor = current_x_tensor_desc_for_op_ptr;
  add_noise_desc.BTensor = &gumbel_noise_fp32_desc_bundle.get_tensor_desc();
  add_noise_desc.OutputTensor = target_sum_tensor_desc_for_op_ptr;

  dml::utils::DmlBindingArrayBundle add_noise_inputs = {
      {current_x_resource_for_op_ptr, 0, 0},
      {gumbel_noise_fp32_resource->GetD3D12Resource(), 0, 0}};
  dml::utils::DmlBindingArrayBundle add_noise_outputs = {
      {target_sum_resource_for_op_ptr, 0, 0}};
  dml::GetOrCreateCompiledOperatorApi(std::move(add_noise_op_bundle))
      ->Execute(add_noise_inputs, add_noise_outputs);

  // Optional Step: Cast sum back to y's original data type
  if (cast_input_to_fp32) {
    dml::utils::DmlOperatorDescBundle cast_back_op;
    auto& cast_back_desc =
        cast_back_op.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
    cast_back_desc.InputTensor =
        target_sum_tensor_desc_for_op_ptr;  // Sum in FP32
    cast_back_desc.OutputTensor =
        &y_desc_bundle.get_tensor_desc();  // Target y tensor (original type)

    dml::utils::DmlBindingArrayBundle cast_back_inputs{
        {target_sum_resource_for_op_ptr, 0, 0}};
    dml::utils::DmlBindingArrayBundle cast_back_outputs{
        {dml::utils::ResourceFromStorageView(y), 0, 0}};
    dml::GetOrCreateCompiledOperatorApi(std::move(cast_back_op))
        ->Execute(cast_back_inputs, cast_back_outputs);
  }
}

// Template instantiations for supported data types
#define DECLARE_IMPL_DML_GUMBEL(T)                                \
  template void GumbelMax::add_gumbel_noise<Device::DirectML, T>( \
      const StorageView& x, StorageView& y) const;

DECLARE_IMPL_DML_GUMBEL(float)
DECLARE_IMPL_DML_GUMBEL(float16_t)

#ifdef CT2_WITH_BFLOAT16
DECLARE_IMPL_DML_GUMBEL(bfloat16_t)
#endif

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
