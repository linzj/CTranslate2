#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rms_norm.h"

#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

namespace {

template <typename T>
DML_TENSOR_DATA_TYPE get_dml_data_type() {
  if constexpr (std::is_same_v<T, float>) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  }
  return DML_TENSOR_DATA_TYPE_FLOAT32;
}

uint32_t get_type_size(DML_TENSOR_DATA_TYPE data_type) {
  return (data_type == DML_TENSOR_DATA_TYPE_FLOAT16) ? 2 : 4;
}

Microsoft::WRL::ComPtr<ID3D12Resource> create_epsilon_constant_buffer(
    dml::Device* device,
    float epsilon_value,
    dim_t batch_size,
    DML_TENSOR_DATA_TYPE data_type) {
  const uint32_t type_size = get_type_size(data_type);
  const uint64_t buffer_size = batch_size * type_size;

  auto buffer = device->CreatePreferredDeviceMemoryBuffer(buffer_size);

  // Upload epsilon values
  std::vector<float> epsilon_data(batch_size, epsilon_value);

  if (data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
    std::vector<uint16_t> epsilon_data_fp16(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      // Convert float to float16 (simplified)
      epsilon_data_fp16[i] =
          static_cast<uint16_t>(epsilon_value * 65504.0f / 65504.0f);
    }
    auto upload_buffer = device->Upload(
        buffer_size, std::string_view(reinterpret_cast<const char*>(
                                          epsilon_data_fp16.data()),
                                      buffer_size));
    device->GetCommandList()->CopyResource(buffer.Get(), upload_buffer.Get());
  } else {
    auto upload_buffer = device->Upload(
        buffer_size,
        std::string_view(reinterpret_cast<const char*>(epsilon_data.data()),
                         buffer_size));
    device->GetCommandList()->CopyResource(buffer.Get(), upload_buffer.Get());
  }

  return buffer;
}

}  // namespace

template <Device D, typename T>
void RMSNorm::compute(const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  // Get input/output buffers
  auto input_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto gamma_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(gamma.buffer()));
  auto output_buffer = static_cast<ID3D12Resource*>(output.buffer());

  const DML_TENSOR_DATA_TYPE data_type = get_dml_data_type<T>();
  const uint32_t type_size = get_type_size(data_type);

  device->ResetCommandList();

  // Input tensor: [batch_size, depth]
  const uint32_t input_sizes[] = {static_cast<uint32_t>(batch_size),
                                  static_cast<uint32_t>(depth)};
  const uint32_t input_strides[] = {static_cast<uint32_t>(depth), 1};

  DML_BUFFER_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.DataType = data_type;
  input_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_tensor_desc.DimensionCount = 2;
  input_tensor_desc.Sizes = input_sizes;
  input_tensor_desc.Strides = input_strides;
  input_tensor_desc.TotalTensorSizeInBytes = input.size() * type_size;
  input_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_desc = {DML_TENSOR_TYPE_BUFFER, &input_tensor_desc};

  // Gamma tensor: [1, depth] for broadcasting
  const uint32_t gamma_sizes[] = {1, static_cast<uint32_t>(depth)};
  const uint32_t gamma_strides[] = {static_cast<uint32_t>(depth), 1};

  DML_BUFFER_TENSOR_DESC gamma_tensor_desc = {};
  gamma_tensor_desc.DataType = data_type;
  gamma_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  gamma_tensor_desc.DimensionCount = 2;
  gamma_tensor_desc.Sizes = gamma_sizes;
  gamma_tensor_desc.Strides = gamma_strides;
  gamma_tensor_desc.TotalTensorSizeInBytes = gamma.size() * type_size;
  gamma_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC gamma_desc = {DML_TENSOR_TYPE_BUFFER, &gamma_tensor_desc};

  // Output tensor: [batch_size, depth]
  DML_BUFFER_TENSOR_DESC output_tensor_desc = input_tensor_desc;
  DML_TENSOR_DESC output_desc = {DML_TENSOR_TYPE_BUFFER, &output_tensor_desc};

  // Step 1: Compute input_squared = input * input
  auto input_squared_buffer =
      device->CreatePreferredDeviceMemoryBuffer(input.size() * type_size);
  DML_BUFFER_TENSOR_DESC input_squared_desc = input_tensor_desc;
  DML_TENSOR_DESC input_squared_tensor = {DML_TENSOR_TYPE_BUFFER,
                                          &input_squared_desc};

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc = {};
  multiply_desc.ATensor = &input_desc;
  multiply_desc.BTensor = &input_desc;
  multiply_desc.OutputTensor = &input_squared_tensor;

  DML_OPERATOR_DESC multiply_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                        &multiply_desc};
  auto multiply_op = dml::GetOrCreateCompiledOperatorApi(&multiply_op_desc);

  // Step 2: Compute mean_squared = reduce_mean(input_squared, axis=1)
  const uint32_t mean_sizes[] = {static_cast<uint32_t>(batch_size), 1};
  const uint32_t mean_strides[] = {1, 1};

  DML_BUFFER_TENSOR_DESC mean_squared_desc = {};
  mean_squared_desc.DataType = data_type;
  mean_squared_desc.Flags = DML_TENSOR_FLAG_NONE;
  mean_squared_desc.DimensionCount = 2;
  mean_squared_desc.Sizes = mean_sizes;
  mean_squared_desc.Strides = mean_strides;
  mean_squared_desc.TotalTensorSizeInBytes = batch_size * type_size;
  mean_squared_desc.GuaranteedBaseOffsetAlignment = 0;

  auto mean_squared_buffer =
      device->CreatePreferredDeviceMemoryBuffer(batch_size * type_size);
  DML_TENSOR_DESC mean_squared_tensor = {DML_TENSOR_TYPE_BUFFER,
                                         &mean_squared_desc};

  const uint32_t reduce_axes[] = {1};
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_squared_tensor;
  reduce_desc.OutputTensor = &mean_squared_tensor;
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = reduce_axes;

  DML_OPERATOR_DESC reduce_op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};
  auto reduce_op = dml::GetOrCreateCompiledOperatorApi(&reduce_op_desc);

  // Step 3: Create epsilon constant and add to mean_squared
  auto epsilon_buffer =
      create_epsilon_constant_buffer(device, _epsilon, batch_size, data_type);
  DML_BUFFER_TENSOR_DESC epsilon_desc = mean_squared_desc;
  DML_TENSOR_DESC epsilon_tensor = {DML_TENSOR_TYPE_BUFFER, &epsilon_desc};

  auto variance_buffer =
      device->CreatePreferredDeviceMemoryBuffer(batch_size * type_size);
  DML_BUFFER_TENSOR_DESC variance_desc = mean_squared_desc;
  DML_TENSOR_DESC variance_tensor = {DML_TENSOR_TYPE_BUFFER, &variance_desc};

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &mean_squared_tensor;
  add_desc.BTensor = &epsilon_tensor;
  add_desc.OutputTensor = &variance_tensor;

  DML_OPERATOR_DESC add_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc};
  auto add_op = dml::GetOrCreateCompiledOperatorApi(&add_op_desc);

  // Step 4: Compute inv_rms = rsqrt(variance) = 1/sqrt(variance)
  auto inv_rms_buffer =
      device->CreatePreferredDeviceMemoryBuffer(batch_size * type_size);
  DML_BUFFER_TENSOR_DESC inv_rms_desc = mean_squared_desc;
  DML_TENSOR_DESC inv_rms_tensor = {DML_TENSOR_TYPE_BUFFER, &inv_rms_desc};

  DML_ELEMENT_WISE_SQRT_OPERATOR_DESC sqrt_desc = {};
  sqrt_desc.InputTensor = &variance_tensor;
  sqrt_desc.OutputTensor = &inv_rms_tensor;
  sqrt_desc.ScaleBias = nullptr;

  DML_OPERATOR_DESC sqrt_op_desc = {DML_OPERATOR_ELEMENT_WISE_SQRT, &sqrt_desc};
  auto sqrt_op = dml::GetOrCreateCompiledOperatorApi(&sqrt_op_desc);

  // Reciprocal to get 1/sqrt
  DML_ELEMENT_WISE_RECIP_OPERATOR_DESC recip_desc = {};
  recip_desc.InputTensor = &inv_rms_tensor;
  recip_desc.OutputTensor = &inv_rms_tensor;  // In-place
  recip_desc.ScaleBias = nullptr;

  DML_OPERATOR_DESC recip_op_desc = {DML_OPERATOR_ELEMENT_WISE_RECIP,
                                     &recip_desc};
  auto recip_op = dml::GetOrCreateCompiledOperatorApi(&recip_op_desc);

  // Step 5: Normalize input = input * inv_rms (with broadcasting)
  // inv_rms needs to broadcast from [batch_size, 1] to [batch_size, depth]
  const uint32_t inv_rms_broadcast_sizes[] = {static_cast<uint32_t>(batch_size),
                                              static_cast<uint32_t>(depth)};
  const uint32_t inv_rms_broadcast_strides[] = {
      1, 0};  // Stride 0 for broadcasting

  DML_BUFFER_TENSOR_DESC inv_rms_broadcast_desc = {};
  inv_rms_broadcast_desc.DataType = data_type;
  inv_rms_broadcast_desc.Flags = DML_TENSOR_FLAG_NONE;
  inv_rms_broadcast_desc.DimensionCount = 2;
  inv_rms_broadcast_desc.Sizes = inv_rms_broadcast_sizes;
  inv_rms_broadcast_desc.Strides = inv_rms_broadcast_strides;
  inv_rms_broadcast_desc.TotalTensorSizeInBytes = batch_size * type_size;
  inv_rms_broadcast_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC inv_rms_broadcast_tensor = {DML_TENSOR_TYPE_BUFFER,
                                              &inv_rms_broadcast_desc};

  auto normalized_buffer =
      device->CreatePreferredDeviceMemoryBuffer(input.size() * type_size);
  DML_BUFFER_TENSOR_DESC normalized_desc = input_tensor_desc;
  DML_TENSOR_DESC normalized_tensor = {DML_TENSOR_TYPE_BUFFER,
                                       &normalized_desc};

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC norm_multiply_desc = {};
  norm_multiply_desc.ATensor = &input_desc;
  norm_multiply_desc.BTensor = &inv_rms_broadcast_tensor;
  norm_multiply_desc.OutputTensor = &normalized_tensor;

  DML_OPERATOR_DESC norm_multiply_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                             &norm_multiply_desc};
  auto norm_multiply_op =
      dml::GetOrCreateCompiledOperatorApi(&norm_multiply_op_desc);

  // Step 6: Final computation - handle residual connection
  dml::Operator* final_op;
  Microsoft::WRL::ComPtr<ID3D12Resource> gamma_modified_buffer;
  DML_TENSOR_DESC gamma_final_desc = gamma_desc;
  DML_BUFFER_TENSOR_DESC gamma_modified_desc;

  if (_use_residual) {
    // Compute (1 + gamma) first
    gamma_modified_buffer =
        device->CreatePreferredDeviceMemoryBuffer(gamma.size() * type_size);

    // Create constant "1" tensor
    std::vector<float> ones_data(depth, 1.0f);
    Microsoft::WRL::ComPtr<ID3D12Resource> ones_upload;

    if (data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
      std::vector<uint16_t> ones_data_fp16(depth, 0x3C00);  // 1.0 in FP16
      ones_upload = device->Upload(
          depth * type_size,
          std::string_view(reinterpret_cast<const char*>(ones_data_fp16.data()),
                           depth * type_size));
    } else {
      ones_upload = device->Upload(
          depth * type_size,
          std::string_view(reinterpret_cast<const char*>(ones_data.data()),
                           depth * type_size));
    }

    auto ones_buffer =
        device->CreatePreferredDeviceMemoryBuffer(gamma.size() * type_size);
    device->GetCommandList()->CopyResource(ones_buffer.Get(),
                                           ones_upload.Get());

    DML_BUFFER_TENSOR_DESC ones_desc = gamma_tensor_desc;
    DML_TENSOR_DESC ones_tensor = {DML_TENSOR_TYPE_BUFFER, &ones_desc};

    gamma_modified_desc = gamma_tensor_desc;
    gamma_modified_desc.TotalTensorSizeInBytes = gamma.size() * type_size;
    DML_TENSOR_DESC gamma_modified_tensor = {DML_TENSOR_TYPE_BUFFER,
                                             &gamma_modified_desc};

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC gamma_add_desc = {};
    gamma_add_desc.ATensor = &ones_tensor;
    gamma_add_desc.BTensor = &gamma_desc;
    gamma_add_desc.OutputTensor = &gamma_modified_tensor;

    DML_OPERATOR_DESC gamma_add_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                           &gamma_add_desc};
    auto gamma_add_op = dml::GetOrCreateCompiledOperatorApi(&gamma_add_op_desc);

    // Execute gamma addition first
    DML_BUFFER_BINDING ones_binding = {
        ones_buffer.Get(), 0, static_cast<UINT64>(gamma.size() * type_size)};
    DML_BUFFER_BINDING gamma_binding = {
        gamma_buffer, 0, static_cast<UINT64>(gamma.size() * type_size)};
    DML_BUFFER_BINDING gamma_modified_binding = {
        gamma_modified_buffer.Get(), 0,
        static_cast<UINT64>(gamma.size() * type_size)};

    std::vector<DML_BINDING_DESC> input_bindings = {
        {DML_BINDING_TYPE_BUFFER, &ones_binding},
        {DML_BINDING_TYPE_BUFFER, &gamma_binding}};
    DML_BINDING_DESC output_binding = {DML_BINDING_TYPE_BUFFER,
                                       &gamma_modified_binding};

    gamma_add_op->Execute(input_bindings, {output_binding});

    gamma_final_desc.Desc = &gamma_modified_desc;
  }

  // Final multiply: normalized * gamma (or normalized * (1+gamma) if residual)
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC final_multiply_desc = {};
  final_multiply_desc.ATensor = &normalized_tensor;
  final_multiply_desc.BTensor = &gamma_final_desc;
  final_multiply_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC final_multiply_op_desc = {
      DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &final_multiply_desc};
  final_op = dml::GetOrCreateCompiledOperatorApi(&final_multiply_op_desc);

  // Execute all operations in sequence
  struct OperationStep {
    dml::Operator* op;
    std::vector<DML_BUFFER_BINDING> input_bindings;
    std::vector<DML_BUFFER_BINDING> output_bindings;
  };

  std::vector<OperationStep> operations;

  // Operation 1: input * input
  operations.push_back(
      {multiply_op,
       {{input_buffer, 0, static_cast<UINT64>(input.size() * type_size)},
        {input_buffer, 0, static_cast<UINT64>(input.size() * type_size)}},
       {{input_squared_buffer.Get(), 0,
         static_cast<UINT64>(input.size() * type_size)}}});

  // Operation 2: reduce mean
  operations.push_back({reduce_op,
                        {{input_squared_buffer.Get(), 0,
                          static_cast<UINT64>(input.size() * type_size)}},
                        {{mean_squared_buffer.Get(), 0,
                          static_cast<UINT64>(batch_size * type_size)}}});

  // Operation 3: add epsilon
  operations.push_back(
      {add_op,
       {{mean_squared_buffer.Get(), 0,
         static_cast<UINT64>(batch_size * type_size)},
        {epsilon_buffer.Get(), 0, static_cast<UINT64>(batch_size * type_size)}},
       {{variance_buffer.Get(), 0,
         static_cast<UINT64>(batch_size * type_size)}}});

  // Operation 4: sqrt
  operations.push_back({sqrt_op,
                        {{variance_buffer.Get(), 0,
                          static_cast<UINT64>(batch_size * type_size)}},
                        {{inv_rms_buffer.Get(), 0,
                          static_cast<UINT64>(batch_size * type_size)}}});

  // Operation 5: reciprocal
  operations.push_back(
      {recip_op,
       {{inv_rms_buffer.Get(), 0, static_cast<UINT64>(batch_size * type_size)}},
       {{inv_rms_buffer.Get(), 0,
         static_cast<UINT64>(batch_size * type_size)}}});

  // Operation 6: normalize
  operations.push_back(
      {norm_multiply_op,
       {{input_buffer, 0, static_cast<UINT64>(input.size() * type_size)},
        {inv_rms_buffer.Get(), 0, static_cast<UINT64>(batch_size * type_size)}},
       {{normalized_buffer.Get(), 0,
         static_cast<UINT64>(input.size() * type_size)}}});

  // Operation 7: final multiply
  auto final_gamma_buffer =
      _use_residual ? gamma_modified_buffer.Get() : gamma_buffer;
  operations.push_back(
      {final_op,
       {{normalized_buffer.Get(), 0,
         static_cast<UINT64>(input.size() * type_size)},
        {final_gamma_buffer, 0, static_cast<UINT64>(gamma.size() * type_size)}},
       {{output_buffer, 0, static_cast<UINT64>(output.size() * type_size)}}});

  // Execute all operations
  for (auto& operation : operations) {
    std::vector<DML_BINDING_DESC> input_binding_descs;
    for (auto& binding : operation.input_bindings) {
      input_binding_descs.push_back({DML_BINDING_TYPE_BUFFER, &binding});
    }

    std::vector<DML_BINDING_DESC> output_binding_descs;
    for (auto& binding : operation.output_bindings) {
      output_binding_descs.push_back({DML_BINDING_TYPE_BUFFER, &binding});
    }

    operation.op->Execute(input_binding_descs, output_binding_descs);
  }

  device->ExecuteCommandList();
}

#define DECLARE_IMPL(T)                                \
  template void RMSNorm::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, StorageView&) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
