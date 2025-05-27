#if defined(CT2_WITH_DIRECTML)
#include "ctranslate2/ops/mean.h"
#include "dml/backend_dml.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <typename T>
static DML_TENSOR_DATA_TYPE get_dml_data_type() {
  if constexpr (std::is_same_v<T, float>) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;  // DirectML doesn't have native
                                          // bfloat16, use float16
  } else {
    static_assert(sizeof(T) == 0, "Unsupported data type");
  }
}

template <Device D, typename T>
void Mean::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t axis_size,
                   const dim_t inner_size,
                   const bool get_sum,
                   StorageView& output) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // The input is conceptually reshaped to [outer_size, axis_size, inner_size]
  // We need to reduce along the middle dimension (axis = 1)
  std::vector<UINT> input_sizes = {static_cast<UINT>(outer_size),
                                   static_cast<UINT>(axis_size),
                                   static_cast<UINT>(inner_size)};

  // Calculate strides for row-major layout
  std::vector<UINT> input_strides = {static_cast<UINT>(axis_size * inner_size),
                                     static_cast<UINT>(inner_size), 1};

  // Create input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  input_buffer_desc.DataType = get_dml_data_type<T>();
  input_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_buffer_desc.DimensionCount = 3;
  input_buffer_desc.Sizes = input_sizes.data();
  input_buffer_desc.Strides = input_strides.data();
  input_buffer_desc.TotalTensorSizeInBytes = input.size() * sizeof(T);
  input_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_tensor_desc.Desc = &input_buffer_desc;

  // Output shape after reduction: [outer_size, 1, inner_size] or [outer_size,
  // inner_size]
  std::vector<UINT> output_sizes = {static_cast<UINT>(outer_size), 1,
                                    static_cast<UINT>(inner_size)};

  std::vector<UINT> output_strides = {static_cast<UINT>(inner_size),
                                      static_cast<UINT>(inner_size), 1};

  // Create output tensor descriptor
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};
  output_buffer_desc.DataType = get_dml_data_type<T>();
  output_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_buffer_desc.DimensionCount = 3;
  output_buffer_desc.Sizes = output_sizes.data();
  output_buffer_desc.Strides = output_strides.data();
  output_buffer_desc.TotalTensorSizeInBytes = output.size() * sizeof(T);
  output_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_tensor_desc.Desc = &output_buffer_desc;

  // Create reduce operator - reduce along axis 1 (the middle dimension)
  UINT axis_to_reduce = 1;
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function =
      get_sum ? DML_REDUCE_FUNCTION_SUM : DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_tensor_desc;
  reduce_desc.OutputTensor = &output_tensor_desc;
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = &axis_to_reduce;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_REDUCE;
  op_desc.Desc = &reduce_desc;

  // Get compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Get binding properties
  auto binding_props = compiled_op->GetBindingProperties();

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  HRESULT hr =
      dml_device->CreateBindingTable(nullptr, IID_PPV_ARGS(&binding_table));
  if (FAILED(hr)) {
    throw std::runtime_error("Failed to create binding table");
  }

  // Bind input
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  input_binding.Offset = 0;
  input_binding.SizeInBytes = input.size() * sizeof(T);

  DML_BINDING_DESC input_binding_desc = {};
  input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  input_binding_desc.Desc = &input_binding;

  binding_table->BindInputs(1, &input_binding_desc);

  // Bind output
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = static_cast<ID3D12Resource*>(output.buffer());
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output.size() * sizeof(T);

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  binding_table->BindOutputs(1, &output_binding_desc);

  // Create temporary resource if needed
  if (binding_props.TemporaryResourceSize > 0) {
    auto temp_resource = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.TemporaryResourceSize);

    DML_BUFFER_BINDING temp_binding = {};
    temp_binding.Buffer = temp_resource.Get();
    temp_binding.Offset = 0;
    temp_binding.SizeInBytes = binding_props.TemporaryResourceSize;

    DML_BINDING_DESC temp_binding_desc = {};
    temp_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
    temp_binding_desc.Desc = &temp_binding;

    binding_table->BindTemporaryResource(&temp_binding_desc);

    // Keep the resource alive until command list execution
    device->KeepAliveUntilNextCommandListDispatch(std::move(temp_resource));
  }

  // Record and execute the dispatch
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
  device->ExecuteCommandList();
}

#define DECLARE_IMPL(T)                                                        \
  template void Mean::compute<Device::DirectML, T>(                            \
      const StorageView& input, const dim_t outer_size, const dim_t axis_size, \
      const dim_t inner_size, const bool get_sum, StorageView& output) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif
