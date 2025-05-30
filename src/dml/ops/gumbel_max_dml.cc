#ifdef CT2_WITH_DIRECTML  // Guard for the whole file

#include "ctranslate2/ops/gumbel_max.h"

#include "dml/backend_dml.h"
#include "dml/dxdevice.h"
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

namespace {  // Anonymous namespace for helpers

// Helper to get DML tensor data type from ctranslate2::DataType
DML_TENSOR_DATA_TYPE get_dml_data_type(DataType ct2_type) {
  switch (ct2_type) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
#ifdef CT2_WITH_BFLOAT16
    // DirectML does not have a native BFLOAT16 type.
    // It's typically handled by casting to FLOAT32 for computation.
    // For storage/tensor description, FLOAT16 is a common proxy if ops support
    // it, or UINT16 if just storing bits. The main code casts to FLOAT32 for
    // compute.
    case DataType::BFLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;  // Or DML_TENSOR_DATA_TYPE_UINT16
                                            // then cast. Using FLOAT16 implies
                                            // DML ops might try to use it. The
                                            // existing add_gumbel_noise casts
                                            // to FLOAT32 anyway.
#endif
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DataType::INT16:
      return DML_TENSOR_DATA_TYPE_INT16;
    case DataType::INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
    default:
      throw std::invalid_argument(
          "Unsupported ctranslate2::DataType for DirectML: " +
          dtype_name(ct2_type));
  }
}

// Helper to get DML dimensions from ctranslate2::Shape
std::vector<UINT> get_dml_dims(const Shape& shape) {
  std::vector<UINT> dml_dims;
  dml_dims.reserve(shape.size());
  for (dim_t d : shape) {
    GUMBEL_MAX_CHECK(d >= 0 && d <= std::numeric_limits<UINT>::max(),
                     "Dimension size out of UINT range");
    dml_dims.push_back(static_cast<UINT>(d));
  }
  // DML often expects at least 1D. If shape is empty (scalar),
  // the calling code (add_gumbel_noise) adjusts it to {1}.
  return dml_dims;
}

// Helper to get byte size of a DML_TENSOR_DATA_TYPE
static size_t dml_data_type_size_bytes(DML_TENSOR_DATA_TYPE type) {
  switch (type) {
    case DML_TENSOR_DATA_TYPE_FLOAT64:
      return 8;
    case DML_TENSOR_DATA_TYPE_UINT64:
      return 8;
    case DML_TENSOR_DATA_TYPE_INT64:
      return 8;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_UINT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_INT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return 2;
    case DML_TENSOR_DATA_TYPE_UINT16:
      return 2;
    case DML_TENSOR_DATA_TYPE_INT16:
      return 2;
    case DML_TENSOR_DATA_TYPE_UINT8:
      return 1;
    case DML_TENSOR_DATA_TYPE_INT8:
      return 1;
    default:
      throw std::invalid_argument(
          "Unsupported DML_TENSOR_DATA_TYPE for size calculation");
  }
  return 0;  // Should not reach
}

// Helper to create DML_BUFFER_TENSOR_DESC
DML_BUFFER_TENSOR_DESC make_buffer_tensor_desc(
    DML_TENSOR_DATA_TYPE data_type,
    const std::vector<UINT>& dims,  // DML compatible dimensions
    const std::vector<UINT>* strides_opt = nullptr) {
  DML_BUFFER_TENSOR_DESC desc{};
  desc.DataType = data_type;
  desc.Flags = DML_TENSOR_FLAG_NONE;

  GUMBEL_MAX_CHECK(
      dims.size() <=
          DML_TENSOR_DIMENSION_COUNT_MAX1,  // Use
                                            // DML_TENSOR_DIMENSION_COUNT_MAX1
                                            // for newer DML
      "Tensor dimension count exceeds DML maximum.");
  desc.DimensionCount = static_cast<UINT>(dims.size());
  desc.Sizes = dims.data();

  if (strides_opt && !strides_opt->empty()) {
    GUMBEL_MAX_CHECK(strides_opt->size() == dims.size(),
                     "Strides vector size must match dimension count.");
    desc.Strides = strides_opt->data();
  } else {
    desc.Strides = nullptr;  // DML will assume packed tensor
  }

  uint64_t num_elements = 1;
  if (desc.DimensionCount == 0) {  // DML scalar (0 dimensions)
    num_elements = 1;
  } else {
    bool has_zero_dim = false;
    for (UINT i = 0; i < desc.DimensionCount; ++i) {
      if (desc.Sizes[i] == 0) {
        has_zero_dim = true;
        break;
      }
      num_elements *= desc.Sizes[i];
    }
    if (has_zero_dim)
      num_elements = 0;
  }

  desc.TotalTensorSizeInBytes =
      num_elements * dml_data_type_size_bytes(data_type);
  desc.GuaranteedBaseOffsetAlignment = 0;  // Use DML default alignment

  return desc;
}

// Helper to initialize DML_BUFFER_TENSOR_DESC and DML_TENSOR_DESC from
// StorageView attributes
void create_dml_buffer_tensor_desc(
    DML_BUFFER_TENSOR_DESC& out_buffer_desc,
    DML_TENSOR_DESC& out_tensor_desc,
    const StorageView&
        storage_view,  // Input StorageView (for consistency checks)
    const std::vector<UINT>&
        dml_dims,  // DML-compatible dimensions for this tensor
    DML_TENSOR_DATA_TYPE dml_tensor_type) {  // DML data type for this tensor

  out_buffer_desc = make_buffer_tensor_desc(dml_tensor_type, dml_dims);

  // Consistency check
  dim_t expected_elements = 1;
  if (out_buffer_desc.DimensionCount == 0) {
    expected_elements = 1;  // Scalar
  } else {
    bool has_zero_dim = false;
    for (UINT i = 0; i < out_buffer_desc.DimensionCount; ++i) {
      if (out_buffer_desc.Sizes[i] == 0) {
        has_zero_dim = true;
        break;
      }
      expected_elements *= out_buffer_desc.Sizes[i];
    }
    if (has_zero_dim)
      expected_elements = 0;
  }

  GUMBEL_MAX_CHECK(storage_view.size() == expected_elements,
                   "Mismatch between StorageView element count (" +
                       std::to_string(storage_view.size()) +
                       ") and DML dimension product (" +
                       std::to_string(expected_elements) + ").");

  GUMBEL_MAX_CHECK(
      out_buffer_desc.TotalTensorSizeInBytes ==
          storage_view.size() * dml_data_type_size_bytes(dml_tensor_type),
      "Calculated DML TotalTensorSizeInBytes does not match StorageView "
      "size in bytes.");

  out_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  out_tensor_desc.Desc = &out_buffer_desc;
}

// Helper to create a constant tensor on the GPU using
// DML_OPERATOR_FILL_VALUE_CONSTANT (Moved to anonymous namespace and uses local
// helpers)
Microsoft::WRL::ComPtr<ID3D12Resource> CreateDmlConstantTensor(
    dml::Device* ct2_dml_device,           // ctranslate2 dml::Device wrapper
    const std::vector<UINT>& dims,         // Dimensions of the constant tensor
    DML_TENSOR_DATA_TYPE dml_tensor_type,  // Data type of the tensor
    DML_SCALAR_UNION scalar_value,         // Scalar value to fill
    DML_BUFFER_TENSOR_DESC& out_buffer_desc,  // Output: DML_BUFFER_TENSOR_DESC
                                              // for the created tensor
    DML_TENSOR_DESC&
        out_tensor_desc) {  // Output: DML_TENSOR_DESC for the created tensor

  out_buffer_desc =
      make_buffer_tensor_desc(dml_tensor_type, dims);  // Uses local helper
  out_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &out_buffer_desc};

  auto constant_resource = ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
      out_buffer_desc.TotalTensorSizeInBytes);
  ct2_dml_device->KeepAliveUntilNextCommandListDispatch(constant_resource);

  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc{};
  fill_desc.OutputTensor = &out_tensor_desc;
  fill_desc.ValueDataType = dml_tensor_type;
  fill_desc.Value = scalar_value;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT, &fill_desc};

  dml::Operator* fill_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  fill_op->Execute({}, {dml::create_binding_desc(dml::create_buffer_binding(
                           constant_resource.Get()))});
  return constant_resource;
}

}  // anonymous namespace

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

  // Ensure the output tensor y is allocated with the correct shape
  y.resize(x.shape());

  // Determine data types for input and computation
  DataType input_storage_dtype = x.dtype();  // ctranslate2::DataType
  DML_TENSOR_DATA_TYPE dml_input_type = get_dml_data_type(input_storage_dtype);
  DML_TENSOR_DATA_TYPE dml_compute_type =
      DML_TENSOR_DATA_TYPE_FLOAT32;  // Gumbel noise math is best in FP32

  // Get DML-compatible dimensions from the input shape
  std::vector<UINT> dml_dims = get_dml_dims(x.shape());
  if (dml_dims.empty() &&
      x.size() == 1) {  // Handle scalar as a 1D tensor of size 1
    dml_dims = {1};
  } else if (dml_dims.empty() &&
             x.size() != 1) {  // Should not happen if x.size() > 0
    throw std::invalid_argument(
        "Cannot process empty shape for non-scalar tensor in DML GumbelMax.");
  }

  // --- Prepare Tensor Descriptors and GPU Resources ---

  // Input tensor x
  DML_BUFFER_TENSOR_DESC x_buffer_desc{};
  DML_TENSOR_DESC x_tensor_desc{};
  // The dml::utils::create_dml_buffer_tensor_desc helper initializes these
  // based on the StorageView 'x'. It uses dml_input_type for the
  // DML_TENSOR_DATA_TYPE field.
  create_dml_buffer_tensor_desc(x_buffer_desc, x_tensor_desc, x, dml_dims,
                                dml_input_type);

  // Output tensor y
  DML_BUFFER_TENSOR_DESC y_buffer_desc{};
  DML_TENSOR_DESC y_tensor_desc{};
  create_dml_buffer_tensor_desc(y_buffer_desc, y_tensor_desc, y, dml_dims,
                                dml_input_type);

  // Intermediate resources for computation in FLOAT32
  // Uniform random numbers (after scaling and epsilon addition)
  DML_BUFFER_TENSOR_DESC uniform_fp32_buffer_desc =
      make_buffer_tensor_desc(dml_compute_type, dml_dims);
  DML_TENSOR_DESC uniform_fp32_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                              &uniform_fp32_buffer_desc};
  auto uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_fp32_buffer_desc.TotalTensorSizeInBytes);
  device->KeepAliveUntilNextCommandListDispatch(uniform_fp32_resource);

  // Logarithm of uniform random numbers
  DML_BUFFER_TENSOR_DESC log_uniform_fp32_buffer_desc =
      uniform_fp32_buffer_desc;  // Same shape and type
  DML_TENSOR_DESC log_uniform_fp32_tensor_desc = {
      DML_TENSOR_TYPE_BUFFER, &log_uniform_fp32_buffer_desc};
  auto log_uniform_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      log_uniform_fp32_buffer_desc.TotalTensorSizeInBytes);
  device->KeepAliveUntilNextCommandListDispatch(log_uniform_fp32_resource);

  // Gumbel noise (-log(uniform))
  DML_BUFFER_TENSOR_DESC gumbel_noise_fp32_buffer_desc =
      uniform_fp32_buffer_desc;  // Same shape and type
  DML_TENSOR_DESC gumbel_noise_fp32_tensor_desc = {
      DML_TENSOR_TYPE_BUFFER, &gumbel_noise_fp32_buffer_desc};
  auto gumbel_noise_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
      gumbel_noise_fp32_buffer_desc.TotalTensorSizeInBytes);
  device->KeepAliveUntilNextCommandListDispatch(gumbel_noise_fp32_resource);

  // Resources for casting input x to FP32, if necessary
  bool cast_input_to_fp32 = (dml_input_type != dml_compute_type);
  Microsoft::WRL::ComPtr<ID3D12Resource>
      x_fp32_resource;  // Holds x cast to FP32
  DML_BUFFER_TENSOR_DESC
  x_fp32_buffer_desc_storage;  // Descriptor for x_fp32_resource
  DML_TENSOR_DESC
  x_fp32_tensor_desc_storage;  // DML_TENSOR_DESC for x_fp32_resource

  if (cast_input_to_fp32) {
    x_fp32_buffer_desc_storage =
        make_buffer_tensor_desc(dml_compute_type, dml_dims);
    x_fp32_tensor_desc_storage = {DML_TENSOR_TYPE_BUFFER,
                                  &x_fp32_buffer_desc_storage};
    x_fp32_resource = device->CreatePreferredDeviceMemoryBuffer(
        x_fp32_buffer_desc_storage.TotalTensorSizeInBytes);
    device->KeepAliveUntilNextCommandListDispatch(x_fp32_resource);
  }

  // Pointers to current x resource and descriptor (either original x or its
  // FP32 cast)
  const DML_TENSOR_DESC* current_x_tensor_desc_ptr =
      cast_input_to_fp32 ? &x_fp32_tensor_desc_storage : &x_tensor_desc;
  ID3D12Resource* current_x_resource_ptr =
      cast_input_to_fp32
          ? x_fp32_resource.Get()
          : static_cast<ID3D12Resource*>(const_cast<void*>(x.buffer()));

  // Resource for the sum (x + gumbel_noise) in FP32
  Microsoft::WRL::ComPtr<ID3D12Resource>
      sum_fp32_intermediate_resource;  // Used if final output y is not FP32
  DML_BUFFER_TENSOR_DESC sum_fp32_buffer_desc_storage;
  DML_TENSOR_DESC sum_fp32_tensor_desc_storage;
  ID3D12Resource*
      target_sum_resource_ptr;  // Points to y.buffer() (if y is FP32) or
                                // sum_fp32_intermediate_resource

  if (cast_input_to_fp32) {  // If x's type was not FP32, y's type is also not
                             // FP32. Sum is intermediate.
    sum_fp32_buffer_desc_storage =
        make_buffer_tensor_desc(dml_compute_type, dml_dims);
    sum_fp32_tensor_desc_storage = {DML_TENSOR_TYPE_BUFFER,
                                    &sum_fp32_buffer_desc_storage};
    sum_fp32_intermediate_resource = device->CreatePreferredDeviceMemoryBuffer(
        sum_fp32_buffer_desc_storage.TotalTensorSizeInBytes);
    device->KeepAliveUntilNextCommandListDispatch(
        sum_fp32_intermediate_resource);
    target_sum_resource_ptr = sum_fp32_intermediate_resource.Get();
  } else {  // If x is FP32, y is also FP32. Sum directly into y's resource.
    sum_fp32_buffer_desc_storage = y_buffer_desc;  // Use y's descriptor
    sum_fp32_tensor_desc_storage = y_tensor_desc;
    target_sum_resource_ptr = static_cast<ID3D12Resource*>(y.buffer());
  }
  const DML_TENSOR_DESC* target_sum_tensor_desc_ptr =
      &sum_fp32_tensor_desc_storage;

  // --- Operator Execution Sequence ---

  // Optional Step: Cast input x to FP32 if its original type is not FP32
  if (cast_input_to_fp32) {
    DML_CAST_OPERATOR_DESC cast_desc{};
    cast_desc.InputTensor = &x_tensor_desc;  // Original x tensor
    cast_desc.OutputTensor =
        &x_fp32_tensor_desc_storage;  // Target x_fp32 tensor
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CAST, &cast_desc};
    dml::GetOrCreateCompiledOperatorApi(&op_desc)->Execute(
        {dml::create_binding_desc(dml::create_buffer_binding(
            static_cast<ID3D12Resource*>(const_cast<void*>(x.buffer()))))},
        {dml::create_binding_desc(
            dml::create_buffer_binding(x_fp32_resource.Get()))});
  }

  // Step 1.1: Generate Uniform UINT32 Random Numbers -> uniform_uint32_resource
  DML_BUFFER_TENSOR_DESC uniform_uint32_buffer_desc =
      make_buffer_tensor_desc(DML_TENSOR_DATA_TYPE_UINT32, dml_dims);
  DML_TENSOR_DESC uniform_uint32_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                                &uniform_uint32_buffer_desc};
  auto uniform_uint32_resource = device->CreatePreferredDeviceMemoryBuffer(
      uniform_uint32_buffer_desc.TotalTensorSizeInBytes);
  device->KeepAliveUntilNextCommandListDispatch(uniform_uint32_resource);

  // State for DML_OPERATOR_RANDOM_GENERATOR.
  // Simplified: zero-initialized state, updated in-place. For true
  // non-determinism or better statistical properties over sequences of calls,
  // this state should be managed more carefully (e.g., seeded from CPU,
  // persisted across calls, or use a thread-local/instance-local generator
  // state).
  std::vector<UINT> state_dims_vec = {
      4};  // Example size for Philox state {counter_low, counter_high, key0,
           // key1}
  DML_BUFFER_TENSOR_DESC state_buffer_desc =
      make_buffer_tensor_desc(DML_TENSOR_DATA_TYPE_UINT32, state_dims_vec);
  DML_TENSOR_DESC state_tensor_desc = {DML_TENSOR_TYPE_BUFFER,
                                       &state_buffer_desc};
  auto random_generator_state_resource =
      device->CreatePreferredDeviceMemoryBuffer(
          state_buffer_desc
              .TotalTensorSizeInBytes);  // Zero-initialized by default heap
  device->KeepAliveUntilNextCommandListDispatch(
      random_generator_state_resource);

  DML_RANDOM_GENERATOR_OPERATOR_DESC
  random_op_internal_desc{};  // Renamed to avoid conflict
  random_op_internal_desc.InputStateTensor = &state_tensor_desc;
  random_op_internal_desc.OutputTensor =
      &uniform_uint32_tensor_desc;  // Output UINT32 random numbers
  random_op_internal_desc.OutputStateTensor =
      &state_tensor_desc;  // Update state in-place
  random_op_internal_desc.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;

  DML_OPERATOR_DESC random_op_desc = {DML_OPERATOR_RANDOM_GENERATOR,
                                      &random_op_internal_desc};
  dml::GetOrCreateCompiledOperatorApi(&random_op_desc)
      ->Execute({dml::create_binding_desc(dml::create_buffer_binding(
                    random_generator_state_resource.Get()))},  // Input State
                {dml::create_binding_desc(dml::create_buffer_binding(
                     uniform_uint32_resource
                         .Get())),  // Output Value (random numbers)
                 dml::create_binding_desc(dml::create_buffer_binding(
                     random_generator_state_resource.Get()))}
                // Output State (updated in-place)
      );

  // Step 1.2: Cast UINT32 random numbers to FLOAT32 -> uniform_fp32_resource
  DML_CAST_OPERATOR_DESC cast_uint_to_fp32_desc{};
  cast_uint_to_fp32_desc.InputTensor = &uniform_uint32_tensor_desc;
  cast_uint_to_fp32_desc.OutputTensor =
      &uniform_fp32_tensor_desc;  // Output to the FP32 uniform resource
  DML_OPERATOR_DESC cast_uint_op_desc = {DML_OPERATOR_CAST,
                                         &cast_uint_to_fp32_desc};
  dml::GetOrCreateCompiledOperatorApi(&cast_uint_op_desc)
      ->Execute({dml::create_binding_desc(
                    dml::create_buffer_binding(uniform_uint32_resource.Get()))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(uniform_fp32_resource.Get()))});

  // Step 1.3: Scale FLOAT32 random numbers to (0, 1]
  // uniform_fp32_resource = (uniform_fp32_resource / UINT32_MAX_AS_FLOAT) +
  // epsilon First, multiply by (1.0f / UINT32_MAX)
  DML_SCALAR_UNION scale_val_scalar;
  scale_val_scalar.Float32 =
      1.0f / static_cast<float>(std::numeric_limits<uint32_t>::max());
  DML_BUFFER_TENSOR_DESC scale_const_buffer_desc;
  DML_TENSOR_DESC scale_const_tensor_desc;
  auto scale_const_res =
      CreateDmlConstantTensor(device, {1}, dml_compute_type, scale_val_scalar,
                              scale_const_buffer_desc, scale_const_tensor_desc);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc{};
  multiply_desc.ATensor =
      &uniform_fp32_tensor_desc;  // Current content of uniform_fp32_resource
  multiply_desc.BTensor = &scale_const_tensor_desc;  // Constant scale factor
  multiply_desc.OutputTensor =
      &uniform_fp32_tensor_desc;  // Store result back in-place
  DML_OPERATOR_DESC mult_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                    &multiply_desc};
  dml::GetOrCreateCompiledOperatorApi(&mult_op_desc)
      ->Execute({dml::create_binding_desc(
                     dml::create_buffer_binding(uniform_fp32_resource.Get())),
                 dml::create_binding_desc(
                     dml::create_buffer_binding(scale_const_res.Get()))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(uniform_fp32_resource.Get()))});

  // Then, add a small epsilon to avoid log(0)
  DML_SCALAR_UNION eps_val_scalar;
  eps_val_scalar.Float32 = 1e-9f;  // Small positive value
  DML_BUFFER_TENSOR_DESC eps_const_buffer_desc;
  DML_TENSOR_DESC eps_const_tensor_desc;
  auto eps_const_res =
      CreateDmlConstantTensor(device, {1}, dml_compute_type, eps_val_scalar,
                              eps_const_buffer_desc, eps_const_tensor_desc);

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_eps_desc{};
  add_eps_desc.ATensor =
      &uniform_fp32_tensor_desc;  // Current scaled uniform numbers
  add_eps_desc.BTensor = &eps_const_tensor_desc;  // Constant epsilon
  add_eps_desc.OutputTensor =
      &uniform_fp32_tensor_desc;  // Store result back in-place
  DML_OPERATOR_DESC add_eps_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                       &add_eps_desc};
  dml::GetOrCreateCompiledOperatorApi(&add_eps_op_desc)
      ->Execute({dml::create_binding_desc(
                     dml::create_buffer_binding(uniform_fp32_resource.Get())),
                 dml::create_binding_desc(
                     dml::create_buffer_binding(eps_const_res.Get()))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(uniform_fp32_resource.Get()))});
  // uniform_fp32_resource now contains values in approx. (epsilon, 1.0 +
  // epsilon], suitable for log.

  // Step 2: Compute Logarithm: log(uniform_fp32_resource) ->
  // log_uniform_fp32_resource
  DML_ELEMENT_WISE_LOG_OPERATOR_DESC log_desc{};
  log_desc.InputTensor = &uniform_fp32_tensor_desc;
  log_desc.OutputTensor = &log_uniform_fp32_tensor_desc;
  DML_OPERATOR_DESC log_op_desc = {DML_OPERATOR_ELEMENT_WISE_LOG, &log_desc};
  dml::GetOrCreateCompiledOperatorApi(&log_op_desc)
      ->Execute({dml::create_binding_desc(
                    dml::create_buffer_binding(uniform_fp32_resource.Get()))},
                {dml::create_binding_desc(dml::create_buffer_binding(
                    log_uniform_fp32_resource.Get()))});

// Step 3: Negate: -log_uniform_fp32_resource -> gumbel_noise_fp32_resource
#if DML_TARGET_VERSION >= 0x5000 && \
    !defined(CT2_DML_USE_MULTIPLY_FOR_NEGATE)  // DML_ELEMENT_WISE_NEGATE is
                                               // available
  DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC negate_op_internal_desc{};  // Renamed
  negate_op_internal_desc.InputTensor = &log_uniform_fp32_tensor_desc;
  negate_op_internal_desc.OutputTensor = &gumbel_noise_fp32_tensor_desc;
  DML_OPERATOR_DESC negate_op_desc = {DML_OPERATOR_ELEMENT_WISE_NEGATE,
                                      &negate_op_internal_desc};
  dml::GetOrCreateCompiledOperatorApi(&negate_op_desc)
      ->Execute({dml::create_binding_desc(dml::create_buffer_binding(
                    log_uniform_fp32_resource.Get()))},
                {dml::create_binding_desc(dml::create_buffer_binding(
                    gumbel_noise_fp32_resource.Get()))});
#else  // Fallback: Multiply by -1 for older DML versions or if preferred
  DML_SCALAR_UNION m1_val_scalar;
  m1_val_scalar.Float32 = -1.0f;
  DML_BUFFER_TENSOR_DESC m1_const_buffer_desc;
  DML_TENSOR_DESC m1_const_tensor_desc;
  auto m1_const_res =
      CreateDmlConstantTensor(device, {1}, dml_compute_type, m1_val_scalar,
                              m1_const_buffer_desc, m1_const_tensor_desc);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_m1_desc{};
  multiply_m1_desc.ATensor = &log_uniform_fp32_tensor_desc;
  multiply_m1_desc.BTensor = &m1_const_tensor_desc;
  multiply_m1_desc.OutputTensor = &gumbel_noise_fp32_tensor_desc;
  DML_OPERATOR_DESC negate_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                      &multiply_m1_desc};
  dml::GetOrCreateCompiledOperatorApi(&negate_op_desc)
      ->Execute({dml::create_binding_desc(dml::create_buffer_binding(
                     log_uniform_fp32_resource.Get())),
                 dml::create_binding_desc(
                     dml::create_buffer_binding(m1_const_res.Get()))},
                {dml::create_binding_desc(dml::create_buffer_binding(
                    gumbel_noise_fp32_resource.Get()))});
#endif
  // gumbel_noise_fp32_resource now holds the Gumbel noise.

  // Step 4: Add Gumbel noise to input x: current_x_resource_ptr +
  // gumbel_noise_fp32_resource -> target_sum_resource_ptr
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_noise_desc{};
  add_noise_desc.ATensor = current_x_tensor_desc_ptr;  // x (or x cast to FP32)
  add_noise_desc.BTensor =
      &gumbel_noise_fp32_tensor_desc;  // Gumbel noise (FP32)
  add_noise_desc.OutputTensor =
      target_sum_tensor_desc_ptr;  // Output to sum resource (FP32)
  DML_OPERATOR_DESC add_noise_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                         &add_noise_desc};
  dml::GetOrCreateCompiledOperatorApi(&add_noise_op_desc)
      ->Execute({dml::create_binding_desc(
                     dml::create_buffer_binding(current_x_resource_ptr)),
                 dml::create_binding_desc(dml::create_buffer_binding(
                     gumbel_noise_fp32_resource.Get()))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(target_sum_resource_ptr))});

  // Optional Step: Cast sum back to y's original data type if computation was
  // in FP32 and y is not FP32
  if (cast_input_to_fp32) {  // This implies y's original type was not FP32
    DML_CAST_OPERATOR_DESC cast_back_desc{};
    cast_back_desc.InputTensor =
        target_sum_tensor_desc_ptr;  // Sum in FP32 (from
                                     // sum_fp32_intermediate_resource)
    cast_back_desc.OutputTensor =
        &y_tensor_desc;  // Target y tensor (original data type)
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_CAST, &cast_back_desc};
    dml::GetOrCreateCompiledOperatorApi(&op_desc)->Execute(
        {dml::create_binding_desc(
            dml::create_buffer_binding(target_sum_resource_ptr))},
        {dml::create_binding_desc(dml::create_buffer_binding(
            static_cast<ID3D12Resource*>(y.buffer())))});
  }
  // If not cast_input_to_fp32, target_sum_resource_ptr was y.buffer(), so y is
  // already populated. All GPU commands are recorded. Actual execution on GPU
  // happens when the command list is submitted by the device.
}

// Template instantiations for supported data types
#define DECLARE_IMPL_DML_GUMBEL(T)                                \
  template void GumbelMax::add_gumbel_noise<Device::DirectML, T>( \
      const StorageView& x, StorageView& y) const;

DECLARE_IMPL_DML_GUMBEL(float)
DECLARE_IMPL_DML_GUMBEL(float16_t)

#ifdef CT2_WITH_BFLOAT16
// For bfloat16_t, ensure dml::utils::get_dml_data_type promotes it to
// DML_TENSOR_DATA_TYPE_FLOAT32 or DML_TENSOR_DATA_TYPE_FLOAT16, so the casting
// logic (cast_input_to_fp32) handles it. If StorageView for BFLOAT16 on DML
// actually holds data as raw UINT16 representing bfloat16 bits, then
// dml_input_type should reflect this (e.g., DML_TENSOR_DATA_TYPE_UINT16 or a
// hypothetical BF16 DML type). The current logic assumes that if dml_input_type
// is not DML_TENSOR_DATA_TYPE_FLOAT32, it's a type that DML_OPERATOR_CAST can
// convert to/from FLOAT32.
DECLARE_IMPL_DML_GUMBEL(bfloat16_t)
#endif

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
