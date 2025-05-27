#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/softmax.h"

#include "dml/backend_dml.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void SoftMax::compute(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output) const {
  // Get DML device and context
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  // Convert CT2 data type to DML data type
  DML_TENSOR_DATA_TYPE dml_data_type;
  if constexpr (std::is_same_v<T, float>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    dml_data_type = DML_TENSOR_DATA_TYPE_FLOAT16;
  } else {
    throw std::invalid_argument("Unsupported data type for DirectML SoftMax");
  }

  // Setup tensor dimensions - reshape to 2D for softmax along last dimension
  std::vector<UINT> dims = {static_cast<UINT>(batch_size),
                            static_cast<UINT>(depth)};

  // Input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.DataType = dml_data_type;
  input_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_tensor_desc.DimensionCount = 2;
  input_tensor_desc.Sizes = dims.data();
  input_tensor_desc.Strides = nullptr;  // Defaults to compact strides
  input_tensor_desc.TotalTensorSizeInBytes = input.size() * sizeof(T);
  input_tensor_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_desc = {};
  input_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_desc.Desc = &input_tensor_desc;

  // Output tensor descriptor (same as input)
  DML_BUFFER_TENSOR_DESC output_tensor_desc = input_tensor_desc;
  output_tensor_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);

  DML_TENSOR_DESC output_desc = {};
  output_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_desc.Desc = &output_tensor_desc;

  // Create the appropriate operator descriptor
  DML_OPERATOR_DESC op_desc = {};

// Use the newer softmax operators that support axis specification
#if DML_TARGET_VERSION >= 0x5100
  if (_log) {
    // Log softmax operator with axis support
    UINT axis = 1;  // Last dimension in our 2D tensor

    DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC log_softmax_desc = {};
    log_softmax_desc.InputTensor = &input_desc;
    log_softmax_desc.OutputTensor = &output_desc;
    log_softmax_desc.AxisCount = 1;
    log_softmax_desc.Axes = &axis;

    op_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
    op_desc.Desc = &log_softmax_desc;
  } else {
    // Regular softmax operator with axis support
    UINT axis = 1;  // Last dimension in our 2D tensor

    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax_desc = {};
    softmax_desc.InputTensor = &input_desc;
    softmax_desc.OutputTensor = &output_desc;
    softmax_desc.AxisCount = 1;
    softmax_desc.Axes = &axis;

    op_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
    op_desc.Desc = &softmax_desc;
  }
#else
  // Fallback to older operators (no axis support - operates on entire tensor)
  if (_log) {
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC log_softmax_desc = {};
    log_softmax_desc.InputTensor = &input_desc;
    log_softmax_desc.OutputTensor = &output_desc;

    op_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
    op_desc.Desc = &log_softmax_desc;
  } else {
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax_desc = {};
    softmax_desc.InputTensor = &input_desc;
    softmax_desc.OutputTensor = &output_desc;

    op_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
    op_desc.Desc = &softmax_desc;
  }
#endif

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Get binding properties
  auto binding_props = compiled_op->GetBindingProperties();

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();
  binding_table_desc.CPUDescriptorHandle = {};  // Will be set by device
  binding_table_desc.GPUDescriptorHandle = {};  // Will be set by device
  binding_table_desc.SizeInDescriptors = binding_props.RequiredDescriptorCount;

  HRESULT hr = dml_device->CreateBindingTable(&binding_table_desc,
                                              IID_PPV_ARGS(&binding_table));
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create DML binding table");
  }

  // Setup input binding
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  input_binding.Offset = 0;
  input_binding.SizeInBytes = input.size() * sizeof(T);

  DML_BINDING_DESC input_binding_desc = {};
  input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  input_binding_desc.Desc = &input_binding;

  binding_table->BindInputs(1, &input_binding_desc);

  // Setup output binding
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  binding_table->BindOutputs(1, &output_binding_desc);

  // Handle temporary resource if needed
  if (binding_props.TemporaryResourceSize > 0) {
    auto temp_buffer = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.TemporaryResourceSize);

    DML_BUFFER_BINDING temp_binding = {};
    temp_binding.Buffer = temp_buffer.Get();
    temp_binding.Offset = 0;
    temp_binding.SizeInBytes = binding_props.TemporaryResourceSize;

    DML_BINDING_DESC temp_binding_desc = {};
    temp_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
    temp_binding_desc.Desc = &temp_binding;

    binding_table->BindTemporaryResource(&temp_binding_desc);

    // Keep the buffer alive
    device->KeepAliveUntilNextCommandListDispatch(temp_buffer);
  }

  // Record and dispatch the operation
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
  device->ExecuteCommandList();

  // Handle lengths parameter if provided - mask output for out-of-sequence
  // positions
  if (lengths) {
    // Use primitives to zero out positions beyond sequence length
    auto output_data = output.data<T>();
    auto length_data = lengths->data<int32_t>();

    for (dim_t batch = 0; batch < batch_size; ++batch) {
      dim_t seq_len = static_cast<dim_t>(length_data[batch]);
      if (seq_len < depth) {
        T* batch_output = output_data + batch * depth;
        // Zero out positions beyond sequence length
        for (dim_t i = seq_len; i < depth; ++i) {
          batch_output[i] = T(0);
        }
      }
    }
  }
}

#define DECLARE_IMPL(T)                                     \
  template void SoftMax::compute<Device::DirectML, T>(      \
      const StorageView& input, const StorageView* lengths, \
      StorageView& output) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML