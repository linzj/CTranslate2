#ifdef CT2_WITH_DIRECTML  // Guard for the whole file

#include "ctranslate2/ops/gumbel_max.h"

#include "dml/backend_dml.h"
// #include "dml/dxdevice.h" // dml::utils included via dml_utils.h, which
// includes backend_dml.h
#include <limits>           // For std::numeric_limits
#include "dml/dml_utils.h"  // Added for centralized DML utilities
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

  dml::utils::DmlTensorDescBundle x_desc_bundle(x);
  dml::utils::DmlTensorDescBundle y_desc_bundle(y);

  // Intermediate resources for computation in FLOAT32
  // uniform_fp32_resource will hold uniform random numbers (0,1]
  dml::utils::DmlTensorDescBundle uniform_fp32_desc_bundle(
      dml_compute_type, dml_dims_vec, nullptr);
  auto uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // log_uniform_fp32_resource will hold log(uniform_fp32_resource)
  dml::utils::DmlTensorDescBundle log_uniform_fp32_desc_bundle(
      dml_compute_type, dml_dims_vec, nullptr);
  auto log_uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      log_uniform_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  // gumbel_noise_fp32_resource will hold -log(uniform_fp32_resource)
  dml::utils::DmlTensorDescBundle gumbel_noise_fp32_desc_bundle(
      dml_compute_type, dml_dims_vec, nullptr);
  auto gumbel_noise_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      gumbel_noise_fp32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  bool cast_input_to_fp32 = (dml_input_type != dml_compute_type);
  Microsoft::WRL::ComPtr<ID3D12Resource>
      x_fp32_intermediate_resource;  // Holds x cast to FP32
  std::unique_ptr<dml::utils::DmlTensorDescBundle> x_fp32_desc_bundle_ptr;

  const DML_TENSOR_DESC* current_x_tensor_desc_for_op_ptr;
  ID3D12Resource* current_x_resource_for_op_ptr;

  if (cast_input_to_fp32) {
    x_fp32_desc_bundle_ptr = std::make_unique<dml::utils::DmlTensorDescBundle>(
        dml_compute_type, dml_dims_vec, nullptr);
    x_fp32_intermediate_resource = device->CreatePreferredDeviceMemoryBuffer(
        x_fp32_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);

    current_x_tensor_desc_for_op_ptr =
        &x_fp32_desc_bundle_ptr->get_tensor_desc();
    current_x_resource_for_op_ptr = x_fp32_intermediate_resource.Get();
  } else {
    current_x_tensor_desc_for_op_ptr = &x_desc_bundle.get_tensor_desc();
    current_x_resource_for_op_ptr =
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(x.buffer()));
  }

  Microsoft::WRL::ComPtr<ID3D12Resource> sum_fp32_intermediate_resource;
  std::unique_ptr<dml::utils::DmlTensorDescBundle> sum_fp32_desc_bundle_ptr;

  ID3D12Resource* target_sum_resource_for_op_ptr;
  const DML_TENSOR_DESC* target_sum_tensor_desc_for_op_ptr;

  if (cast_input_to_fp32) {  // If x was not FP32, y is not FP32. Sum result
                             // needs intermediate FP32 buffer before casting
                             // back.
    sum_fp32_desc_bundle_ptr =
        std::make_unique<dml::utils::DmlTensorDescBundle>(
            dml_compute_type, dml_dims_vec, nullptr);
    sum_fp32_intermediate_resource = device->CreatePreferredDeviceMemoryBuffer(
        sum_fp32_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);
    target_sum_resource_for_op_ptr = sum_fp32_intermediate_resource.Get();
    target_sum_tensor_desc_for_op_ptr =
        &sum_fp32_desc_bundle_ptr->get_tensor_desc();
  } else {  // If x is FP32, y is also FP32. Sum directly into y's resource.
    target_sum_resource_for_op_ptr =
        reinterpret_cast<ID3D12Resource*>(y.buffer());
    target_sum_tensor_desc_for_op_ptr = &y_desc_bundle.get_tensor_desc();
  }

  // --- Operator Execution Sequence ---

  if (cast_input_to_fp32) {
    DML_CAST_OPERATOR_DESC cast_to_fp32_desc{};
    cast_to_fp32_desc.InputTensor = &x_desc_bundle.get_tensor_desc();
    cast_to_fp32_desc.OutputTensor =
        current_x_tensor_desc_for_op_ptr;  // This is
                                           // x_fp32_desc_bundle_ptr->get_tensor_desc()
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CAST, &cast_to_fp32_desc};

    dml::utils::DmlBufferBindingBundle input_cast_binding(
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(x.buffer())), 0,
        x_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    dml::utils::DmlBufferBindingBundle output_cast_binding(
        current_x_resource_for_op_ptr, 0,
        x_fp32_desc_bundle_ptr->get_buffer_desc().TotalTensorSizeInBytes);

    dml::GetOrCreateCompiledOperatorApi(&op_desc)->Execute(
        {input_cast_binding.get_desc()}, {output_cast_binding.get_desc()});
  }

  // Step 1.1: Generate Uniform UINT32 Random Numbers
  dml::utils::DmlTensorDescBundle uniform_uint32_desc_bundle(
      DML_TENSOR_DATA_TYPE_UINT32, dml_dims_vec, nullptr);
  auto uniform_uint32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_uint32_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  std::vector<UINT> state_dims_cpu = {
      6};  // PHILOX_4X32_10 state size is 6 (4 for counter, 2 for key)
  dml::utils::DmlTensorDescBundle state_desc_bundle(DML_TENSOR_DATA_TYPE_UINT32,
                                                    state_dims_cpu, nullptr);
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
  // input_random_generator_state_resource if relying on default heap
  // initialization isn't guaranteed or desired.

  DML_RANDOM_GENERATOR_OPERATOR_DESC random_op_desc_payload{};
  random_op_desc_payload.InputStateTensor =
      &state_desc_bundle.get_tensor_desc();  // Must be non-null
  random_op_desc_payload.OutputTensor =
      &uniform_uint32_desc_bundle.get_tensor_desc();
  random_op_desc_payload.OutputStateTensor =
      &state_desc_bundle.get_tensor_desc();
  random_op_desc_payload.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;
  DML_OPERATOR_DESC random_op_desc = {DML_OPERATOR_RANDOM_GENERATOR,
                                      &random_op_desc_payload};

  // Bind to two separate resources for input and output state
  dml::utils::DmlBufferBindingBundle random_op_input_state_binding(
      input_random_generator_state_resource.Get());
  dml::utils::DmlBufferBindingBundle random_op_output_val_binding(
      uniform_uint32_resource.Get());
  dml::utils::DmlBufferBindingBundle random_op_output_state_binding(
      output_random_generator_state_resource
          .Get());  // State will be written here

  std::vector<DML_BINDING_DESC> random_op_inputs = {
      random_op_input_state_binding.get_desc()};
  std::vector<DML_BINDING_DESC> random_op_outputs = {
      random_op_output_val_binding.get_desc(),
      random_op_output_state_binding.get_desc()};
  dml::GetOrCreateCompiledOperatorApi(&random_op_desc)
      ->Execute(random_op_inputs, random_op_outputs);

  // Step 1.2: Cast UINT32 random to FLOAT32 -> uniform_fp32_resource
  DML_CAST_OPERATOR_DESC cast_uint_to_fp32_desc{};
  cast_uint_to_fp32_desc.InputTensor =
      &uniform_uint32_desc_bundle.get_tensor_desc();
  cast_uint_to_fp32_desc.OutputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();
  DML_OPERATOR_DESC cast_uint_op_desc = {DML_OPERATOR_CAST,
                                         &cast_uint_to_fp32_desc};

  dml::utils::DmlBufferBindingBundle cast_uint_input_binding(
      uniform_uint32_resource.Get());
  dml::utils::DmlBufferBindingBundle cast_uint_output_binding(
      uniform_fp32_resource.Get());
  dml::GetOrCreateCompiledOperatorApi(&cast_uint_op_desc)
      ->Execute({cast_uint_input_binding.get_desc()},
                {cast_uint_output_binding.get_desc()});

  // Step 1.3: Scale FLOAT32 random numbers to (0, 1]
  DML_SCALAR_UNION scale_val_scalar;
  scale_val_scalar.Float32 =
      1.0f / static_cast<float>(std::numeric_limits<uint32_t>::max());

  // Use standalone DML_OPERATOR_ACTIVATION_LINEAR for scaling
  DML_ACTIVATION_LINEAR_OPERATOR_DESC scale_op_payload{};
  scale_op_payload.InputTensor = &uniform_fp32_desc_bundle.get_tensor_desc();
  scale_op_payload.OutputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();  // In-place
  scale_op_payload.Alpha = scale_val_scalar.Float32;
  scale_op_payload.Beta = 0.0f;
  DML_OPERATOR_DESC scale_op_desc = {DML_OPERATOR_ACTIVATION_LINEAR,
                                     &scale_op_payload};

  dml::utils::DmlBufferBindingBundle scale_input_binding(
      uniform_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle scale_output_binding(
      uniform_fp32_resource.Get());  // In-place

  dml::GetOrCreateCompiledOperatorApi(&scale_op_desc)
      ->Execute({scale_input_binding.get_desc()},
                {scale_output_binding.get_desc()});

  // Add epsilon
  DML_SCALAR_UNION eps_val_scalar;
  eps_val_scalar.Float32 = 1e-9f;
  // Use standalone DML_OPERATOR_ACTIVATION_LINEAR for adding epsilon
  DML_ACTIVATION_LINEAR_OPERATOR_DESC add_eps_op_payload = {};
  add_eps_op_payload.InputTensor = &uniform_fp32_desc_bundle.get_tensor_desc();
  add_eps_op_payload.OutputTensor =
      &uniform_fp32_desc_bundle.get_tensor_desc();  // In-place
  add_eps_op_payload.Alpha = 1.0f;
  add_eps_op_payload.Beta = eps_val_scalar.Float32;
  DML_OPERATOR_DESC add_eps_op_desc = {DML_OPERATOR_ACTIVATION_LINEAR,
                                       &add_eps_op_payload};

  dml::utils::DmlBufferBindingBundle add_eps_input_binding(
      uniform_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle add_eps_output_binding(
      uniform_fp32_resource.Get());  // In-place

  dml::GetOrCreateCompiledOperatorApi(&add_eps_op_desc)
      ->Execute({add_eps_input_binding.get_desc()},
                {add_eps_output_binding.get_desc()});

  // Step 2: Compute Logarithm
  DML_ELEMENT_WISE_LOG_OPERATOR_DESC log_desc{};
  log_desc.InputTensor = &uniform_fp32_desc_bundle.get_tensor_desc();
  log_desc.OutputTensor = &log_uniform_fp32_desc_bundle.get_tensor_desc();
  DML_OPERATOR_DESC log_op_desc = {DML_OPERATOR_ELEMENT_WISE_LOG, &log_desc};

  dml::utils::DmlBufferBindingBundle log_op_input_binding(
      uniform_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle log_op_output_binding(
      log_uniform_fp32_resource.Get());
  dml::GetOrCreateCompiledOperatorApi(&log_op_desc)
      ->Execute({log_op_input_binding.get_desc()},
                {log_op_output_binding.get_desc()});

  // Step 3: Negate
#if DML_TARGET_VERSION >= 0x5000 && !defined(CT2_DML_USE_MULTIPLY_FOR_NEGATE)
  DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC negate_desc_payload{};
  negate_desc_payload.InputTensor =
      &log_uniform_fp32_desc_bundle.get_tensor_desc();
  negate_desc_payload.OutputTensor =
      &gumbel_noise_fp32_desc_bundle.get_tensor_desc();
  DML_OPERATOR_DESC negate_op_desc = {DML_OPERATOR_ELEMENT_WISE_NEGATE,
                                      &negate_desc_payload};

  dml::utils::DmlBufferBindingBundle negate_op_input_binding(
      log_uniform_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle negate_op_output_binding(
      gumbel_noise_fp32_resource.Get());
  dml::GetOrCreateCompiledOperatorApi(&negate_op_desc)
      ->Execute({negate_op_input_binding.get_desc()},
                {negate_op_output_binding.get_desc()});
#else  // Fallback: Multiply by -1
  DML_SCALAR_UNION m1_val_scalar;
  m1_val_scalar.Float32 = -1.0f;
  // Use standalone DML_OPERATOR_ACTIVATION_LINEAR for negate fallback
  DML_ACTIVATION_LINEAR_OPERATOR_DESC negate_op_payload = {};
  negate_op_payload.InputTensor =
      &log_uniform_fp32_desc_bundle.get_tensor_desc();
  negate_op_payload.OutputTensor =
      &gumbel_noise_fp32_desc_bundle
           .get_tensor_desc();                      // Not in-place for negate
  negate_op_payload.Alpha = m1_val_scalar.Float32;  // -1.0f
  negate_op_payload.Beta = 0.0f;
  DML_OPERATOR_DESC negate_op_desc = {DML_OPERATOR_ACTIVATION_LINEAR,
                                      &negate_op_payload};

  dml::utils::DmlBufferBindingBundle neg_fallback_input_binding(
      log_uniform_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle neg_fallback_output_binding(
      gumbel_noise_fp32_resource.Get());

  dml::GetOrCreateCompiledOperatorApi(&negate_op_desc)
      ->Execute({neg_fallback_input_binding.get_desc()},
                {neg_fallback_output_binding.get_desc()});
#endif

  // Step 4: Add Gumbel noise to input x
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_noise_desc{};
  add_noise_desc.ATensor = current_x_tensor_desc_for_op_ptr;
  add_noise_desc.BTensor = &gumbel_noise_fp32_desc_bundle.get_tensor_desc();
  add_noise_desc.OutputTensor = target_sum_tensor_desc_for_op_ptr;
  DML_OPERATOR_DESC add_noise_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                         &add_noise_desc};

  dml::utils::DmlBufferBindingBundle add_noise_A_binding(
      current_x_resource_for_op_ptr);
  dml::utils::DmlBufferBindingBundle add_noise_B_binding(
      gumbel_noise_fp32_resource.Get());
  dml::utils::DmlBufferBindingBundle add_noise_Out_binding(
      target_sum_resource_for_op_ptr);

  std::vector<DML_BINDING_DESC> add_noise_inputs = {
      add_noise_A_binding.get_desc(), add_noise_B_binding.get_desc()};
  std::vector<DML_BINDING_DESC> add_noise_outputs = {
      add_noise_Out_binding.get_desc()};
  dml::GetOrCreateCompiledOperatorApi(&add_noise_op_desc)
      ->Execute(add_noise_inputs, add_noise_outputs);

  // Optional Step: Cast sum back to y's original data type
  if (cast_input_to_fp32) {
    DML_CAST_OPERATOR_DESC cast_back_desc{};
    cast_back_desc.InputTensor =
        target_sum_tensor_desc_for_op_ptr;  // Sum in FP32
    cast_back_desc.OutputTensor =
        &y_desc_bundle.get_tensor_desc();  // Target y tensor (original type)
    DML_OPERATOR_DESC op_desc_cast_back = {DML_OPERATOR_CAST, &cast_back_desc};

    dml::utils::DmlBufferBindingBundle cast_back_input_binding(
        target_sum_resource_for_op_ptr);
    dml::utils::DmlBufferBindingBundle cast_back_output_binding(
        reinterpret_cast<ID3D12Resource*>(y.buffer()));
    dml::GetOrCreateCompiledOperatorApi(&op_desc_cast_back)
        ->Execute({cast_back_input_binding.get_desc()},
                  {cast_back_output_binding.get_desc()});
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
