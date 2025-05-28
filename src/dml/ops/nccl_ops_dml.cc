#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/nccl_ops.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

#ifdef CT2_WITH_DIRECTML
DML_TENSOR_DATA_TYPE getDMLDataTypeFromDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DataType::INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
    case DataType::INT16:
      return DML_TENSOR_DATA_TYPE_INT16;
    default:
      throw std::invalid_argument("The current datatype " +
                                  std::to_string(static_cast<int>(type)) +
                                  " is not supported for DirectML operations");
  }
}

DML_REDUCE_FUNCTION redop_to_dml_reduce_function(ReduceAll::RED_OP op) {
  switch (op) {
    case ReduceAll::RED_OP::SUM:
      return DML_REDUCE_FUNCTION_SUM;
    case ReduceAll::RED_OP::PROD:
      return DML_REDUCE_FUNCTION_MULTIPLY;
    case ReduceAll::RED_OP::MAX:
      return DML_REDUCE_FUNCTION_MAX;
    case ReduceAll::RED_OP::MIN:
      return DML_REDUCE_FUNCTION_MIN;
    case ReduceAll::RED_OP::AVG:
      return DML_REDUCE_FUNCTION_AVERAGE;
    default:
      throw std::runtime_error("the current reduce operation " +
                               std::to_string(static_cast<int>(op)) +
                               " is not supported");
  }
}

template <typename T>
void perform_dml_reduce_operation(const StorageView& input,
                                  StorageView& output,
                                  ReduceAll::RED_OP reduce_op) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Get input and output buffers (cast from void* to ID3D12Resource*)
  auto input_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto output_buffer = static_cast<ID3D12Resource*>(output.buffer());

  // Get data type and reduction function
  DML_TENSOR_DATA_TYPE dml_data_type =
      getDMLDataTypeFromDataType(input.dtype());
  DML_REDUCE_FUNCTION reduce_function = redop_to_dml_reduce_function(reduce_op);

  // Prepare tensor dimensions
  const auto& shape = input.shape();
  std::vector<UINT> sizes;
  for (dim_t dim : shape) {
    sizes.push_back(static_cast<UINT>(dim));
  }

  // If input is empty, set to 1D with size 1
  if (sizes.empty()) {
    sizes.push_back(1);
  }

  // Create input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_tensor_desc = {};
  input_tensor_desc.DataType = dml_data_type;
  input_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  input_tensor_desc.DimensionCount = static_cast<UINT>(sizes.size());
  input_tensor_desc.Sizes = sizes.data();
  input_tensor_desc.Strides = nullptr;  // Use default strides
  input_tensor_desc.TotalTensorSizeInBytes = input.size() * input.item_size();
  input_tensor_desc.GuaranteedBaseOffsetAlignment =
      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC input_desc = {};
  input_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_desc.Desc = &input_tensor_desc;

  // Create output tensor descriptor (typically a scalar or same as input)
  const auto& output_shape = output.shape();
  std::vector<UINT> output_sizes;
  for (dim_t dim : output_shape) {
    output_sizes.push_back(static_cast<UINT>(dim));
  }

  if (output_sizes.empty()) {
    output_sizes.push_back(1);
  }

  DML_BUFFER_TENSOR_DESC output_tensor_desc = {};
  output_tensor_desc.DataType = dml_data_type;
  output_tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  output_tensor_desc.DimensionCount = static_cast<UINT>(output_sizes.size());
  output_tensor_desc.Sizes = output_sizes.data();
  output_tensor_desc.Strides = nullptr;
  output_tensor_desc.TotalTensorSizeInBytes =
      output.size() * output.item_size();
  output_tensor_desc.GuaranteedBaseOffsetAlignment =
      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC output_desc = {};
  output_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_desc.Desc = &output_tensor_desc;

  // Create axes for reduction
  std::vector<UINT> axes;
  for (UINT i = 0; i < sizes.size(); ++i) {
    axes.push_back(i);
  }

  // Create reduce operator descriptor
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = reduce_function;
  reduce_desc.InputTensor = &input_desc;
  reduce_desc.OutputTensor = &output_desc;
  reduce_desc.AxisCount = static_cast<UINT>(axes.size());
  reduce_desc.Axes = axes.data();

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_REDUCE;
  op_desc.Desc = &reduce_desc;

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Bind inputs
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer = input_buffer;
  input_binding.Offset = 0;
  input_binding.SizeInBytes = input_tensor_desc.TotalTensorSizeInBytes;

  DML_BINDING_DESC input_binding_desc = {};
  input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  input_binding_desc.Desc = &input_binding;

  // Bind outputs
  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = output_buffer;
  output_binding.Offset = 0;
  output_binding.SizeInBytes = output_tensor_desc.TotalTensorSizeInBytes;

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  // Record the dispatch operation
  compiled_op->Execute({input_binding_desc}, {output_binding_desc});
}
#endif

template <Device D, typename T>
void ReduceAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  // Note: DirectML doesn't have native multi-device collective operations like
  // NCCL This implementation performs a local reduction operation For true
  // distributed tensor parallel operations, additional coordination between
  // devices/processes would be needed outside of DirectML
  perform_dml_reduce_operation<T>(input, output, _reduce_op);
#endif
  (void)input;
  (void)output;
}

template <Device D, typename T>
void GatherAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  // DirectML doesn't have a direct equivalent to AllGather collective operation
  // AllGather requires coordination between multiple devices/processes
  // This would need to be implemented using a different communication mechanism
  // or higher-level orchestration outside of DirectML

  // For now, we can implement a simple copy operation as a placeholder
  // In a real distributed scenario, this would need external coordination
  auto* device = dml::get_device();

  auto input_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto output_buffer = static_cast<ID3D12Resource*>(output.buffer());

  // Simple copy operation using DirectML identity operator
  DML_TENSOR_DATA_TYPE dml_data_type =
      getDMLDataTypeFromDataType(input.dtype());

  const auto& shape = input.shape();
  std::vector<UINT> sizes;
  for (dim_t dim : shape) {
    sizes.push_back(static_cast<UINT>(dim));
  }

  if (sizes.empty()) {
    sizes.push_back(1);
  }

  // Create tensor descriptors for identity operation
  DML_BUFFER_TENSOR_DESC tensor_desc = {};
  tensor_desc.DataType = dml_data_type;
  tensor_desc.Flags = DML_TENSOR_FLAG_NONE;
  tensor_desc.DimensionCount = static_cast<UINT>(sizes.size());
  tensor_desc.Sizes = sizes.data();
  tensor_desc.Strides = nullptr;
  tensor_desc.TotalTensorSizeInBytes = input.size() * input.item_size();
  tensor_desc.GuaranteedBaseOffsetAlignment =
      DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC input_desc = {};
  input_desc.Type = DML_TENSOR_TYPE_BUFFER;
  input_desc.Desc = &tensor_desc;

  DML_TENSOR_DESC output_desc = {};
  output_desc.Type = DML_TENSOR_TYPE_BUFFER;
  output_desc.Desc = &tensor_desc;

  // Create identity operator
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
  identity_desc.InputTensor = &input_desc;
  identity_desc.OutputTensor = &output_desc;
  identity_desc.ScaleBias = nullptr;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
  op_desc.Desc = &identity_desc;

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
  auto binding_props = compiled_op->GetBindingProperties();

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  dml::get_dml_device()->CreateBindingTable(nullptr,
                                            IID_PPV_ARGS(&binding_table));

  // Bind input and output
  DML_BUFFER_BINDING input_binding = {};
  input_binding.Buffer = input_buffer;
  input_binding.Offset = 0;
  input_binding.SizeInBytes = tensor_desc.TotalTensorSizeInBytes;

  DML_BINDING_DESC input_binding_desc = {};
  input_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  input_binding_desc.Desc = &input_binding;

  DML_BUFFER_BINDING output_binding = {};
  output_binding.Buffer = output_buffer;
  output_binding.Offset = 0;
  output_binding.SizeInBytes = tensor_desc.TotalTensorSizeInBytes;

  DML_BINDING_DESC output_binding_desc = {};
  output_binding_desc.Type = DML_BINDING_TYPE_BUFFER;
  output_binding_desc.Desc = &output_binding;

  binding_table->BindInputs(1, &input_binding_desc);
  binding_table->BindOutputs(1, &output_binding_desc);

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
    device->KeepAliveUntilNextCommandListDispatch(std::move(temp_buffer));
  }

  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
  device->ExecuteCommandList();

  // Note: This is just a copy operation. True AllGather would require
  // external coordination between multiple devices/processes
#endif
  (void)input;
  (void)output;
}

#define DECLARE_IMPL(T)                                                      \
  template void GatherAll::compute<Device::DirectML, T>(const StorageView&,  \
                                                        StorageView&) const; \
  template void ReduceAll::compute<Device::DirectML, T>(const StorageView&,  \
                                                        StorageView&) const;
DECLARE_ALL_TYPES(DECLARE_IMPL)
}  // namespace ops
}  // namespace ctranslate2
#endif
