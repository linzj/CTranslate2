#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/split.h"

#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

namespace {
// Helper to convert ctranslate2 DataType to DML_TENSOR_DATA_TYPE
DML_TENSOR_DATA_TYPE get_dml_data_type(DataType dtype) {
  switch (dtype) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DataType::INT16:
      return DML_TENSOR_DATA_TYPE_INT16;
    case DataType::INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
    default:
      throw std::runtime_error("Unsupported data type for DirectML");
  }
}

// Helper to create DML buffer tensor descriptor
DML_BUFFER_TENSOR_DESC create_buffer_tensor_desc(const StorageView& storage) {
  std::vector<UINT> sizes(storage.rank());
  for (dim_t i = 0; i < storage.rank(); ++i) {
    sizes[i] = static_cast<UINT>(storage.dim(i));
  }

  DML_BUFFER_TENSOR_DESC desc = {};
  desc.DataType = get_dml_data_type(storage.dtype());
  desc.Flags = DML_TENSOR_FLAG_NONE;
  desc.DimensionCount = static_cast<UINT>(storage.rank());
  desc.Sizes = sizes.data();
  desc.Strides = nullptr;  // Use default strides
  desc.TotalTensorSizeInBytes = storage.size() * storage.item_size();
  desc.GuaranteedBaseOffsetAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;
  return desc;
}

// Helper to create tensor descriptor
DML_TENSOR_DESC create_tensor_desc(const DML_BUFFER_TENSOR_DESC& buffer_desc) {
  DML_TENSOR_DESC desc = {};
  desc.Type = DML_TENSOR_TYPE_BUFFER;
  desc.Desc = &buffer_desc;
  return desc;
}

// Helper to create buffer binding
DML_BUFFER_BINDING create_buffer_binding(const StorageView& storage) {
  DML_BUFFER_BINDING binding = {};
  binding.Buffer =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(storage.buffer()));
  binding.Offset = 0;
  binding.SizeInBytes = storage.size() * storage.item_size();
  return binding;
}

// Helper to create binding descriptor
DML_BINDING_DESC create_binding_desc(const DML_BUFFER_BINDING& buffer_binding) {
  DML_BINDING_DESC desc = {};
  desc.Type = DML_BINDING_TYPE_BUFFER;
  desc.Desc = &buffer_binding;
  return desc;
}
}  // namespace

template <Device D, typename T>
void Concat::compute(const std::vector<const StorageView*>& inputs,
                     StorageView& output) const {
  if (inputs.empty())
    return;

  const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;

  // Special case: single input, just copy
  if (inputs.size() == 1) {
    primitives<D>::copy(inputs[0]->data<T>(), output.data<T>(),
                        inputs[0]->size());
    return;
  }

  auto* device = dml::get_device();

  // Create input tensor descriptors
  std::vector<DML_BUFFER_TENSOR_DESC> input_buffer_descs;
  std::vector<DML_TENSOR_DESC> input_tensor_descs;
  input_buffer_descs.reserve(inputs.size());
  input_tensor_descs.reserve(inputs.size());

  for (const auto* input : inputs) {
    input_buffer_descs.push_back(create_buffer_tensor_desc(*input));
    input_tensor_descs.push_back(create_tensor_desc(input_buffer_descs.back()));
  }

  // Create output tensor descriptor
  DML_BUFFER_TENSOR_DESC output_buffer_desc = create_buffer_tensor_desc(output);
  DML_TENSOR_DESC output_tensor_desc = create_tensor_desc(output_buffer_desc);

  // Create JOIN operator descriptor
  DML_JOIN_OPERATOR_DESC join_desc = {};
  join_desc.InputCount = static_cast<UINT>(inputs.size());
  join_desc.InputTensors = input_tensor_descs.data();
  join_desc.OutputTensor = &output_tensor_desc;
  join_desc.Axis = static_cast<UINT>(axis);

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_JOIN;
  op_desc.Desc = &join_desc;

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create input bindings
  std::vector<DML_BUFFER_BINDING> input_buffer_bindings;
  std::vector<DML_BINDING_DESC> input_binding_descs;
  input_buffer_bindings.reserve(inputs.size());
  input_binding_descs.reserve(inputs.size());

  for (const auto* input : inputs) {
    input_buffer_bindings.push_back(create_buffer_binding(*input));
    input_binding_descs.push_back(
        create_binding_desc(input_buffer_bindings.back()));
  }

  // Create output binding
  DML_BUFFER_BINDING output_buffer_binding = create_buffer_binding(output);
  DML_BINDING_DESC output_binding_desc =
      create_binding_desc(output_buffer_binding);

  compiled_op->Execute(input_binding_descs, {output_binding_desc});
}

template <Device D, typename T>
void Split::compute(const StorageView& input,
                    std::vector<StorageView*>& outputs) const {
  if (outputs.empty())
    return;

  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_buffer_desc = create_buffer_tensor_desc(input);
  DML_TENSOR_DESC input_tensor_desc = create_tensor_desc(input_buffer_desc);

  // Create output tensor descriptors
  std::vector<DML_BUFFER_TENSOR_DESC> output_buffer_descs;
  std::vector<DML_TENSOR_DESC> output_tensor_descs;
  output_buffer_descs.reserve(outputs.size());
  output_tensor_descs.reserve(outputs.size());

  for (const auto* output : outputs) {
    output_buffer_descs.push_back(create_buffer_tensor_desc(*output));
    output_tensor_descs.push_back(
        create_tensor_desc(output_buffer_descs.back()));
  }

  // Create SPLIT operator descriptor
  DML_SPLIT_OPERATOR_DESC split_desc = {};
  split_desc.InputTensor = &input_tensor_desc;
  split_desc.OutputCount = static_cast<UINT>(outputs.size());
  split_desc.OutputTensors = output_tensor_descs.data();
  split_desc.Axis = static_cast<UINT>(axis);

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_SPLIT;
  op_desc.Desc = &split_desc;

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create input binding
  DML_BUFFER_BINDING input_buffer_binding = create_buffer_binding(input);
  DML_BINDING_DESC input_binding_desc =
      create_binding_desc(input_buffer_binding);

  // Create output bindings
  std::vector<DML_BUFFER_BINDING> output_buffer_bindings;
  std::vector<DML_BINDING_DESC> output_binding_descs;
  output_buffer_bindings.reserve(outputs.size());
  output_binding_descs.reserve(outputs.size());

  for (const auto* output : outputs) {
    output_buffer_bindings.push_back(create_buffer_binding(*output));
    output_binding_descs.push_back(
        create_binding_desc(output_buffer_bindings.back()));
  }

  compiled_op->Execute({input_binding_desc}, output_binding_descs);
}

template <Device D, typename T>
void Slide::compute(const StorageView& input,
                    StorageView& output,
                    const dim_t& index) const {
  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create input tensor descriptor
  DML_BUFFER_TENSOR_DESC input_buffer_desc = create_buffer_tensor_desc(input);
  DML_TENSOR_DESC input_tensor_desc = create_tensor_desc(input_buffer_desc);

  // Create output tensor descriptor
  DML_BUFFER_TENSOR_DESC output_buffer_desc = create_buffer_tensor_desc(output);
  DML_TENSOR_DESC output_tensor_desc = create_tensor_desc(output_buffer_desc);

  // Calculate slice parameters
  std::vector<UINT> offsets(input.rank(), 0);
  std::vector<UINT> sizes(input.rank());
  std::vector<UINT> strides(input.rank(), 1);

  for (dim_t i = 0; i < input.rank(); ++i) {
    if (i == static_cast<dim_t>(axis)) {
      offsets[i] = static_cast<UINT>(index);
      sizes[i] = static_cast<UINT>(output.dim(i));
    } else {
      sizes[i] = static_cast<UINT>(input.dim(i));
    }
  }

  // Create SLICE operator descriptor
  DML_SLICE_OPERATOR_DESC slice_desc = {};
  slice_desc.InputTensor = &input_tensor_desc;
  slice_desc.OutputTensor = &output_tensor_desc;
  slice_desc.DimensionCount = static_cast<UINT>(input.rank());
  slice_desc.Offsets = offsets.data();
  slice_desc.Sizes = sizes.data();
  slice_desc.Strides = strides.data();

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_SLICE;
  op_desc.Desc = &slice_desc;

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create input binding
  DML_BUFFER_BINDING input_buffer_binding = create_buffer_binding(input);
  DML_BINDING_DESC input_binding_desc =
      create_binding_desc(input_buffer_binding);

  // Create output binding
  DML_BUFFER_BINDING output_buffer_binding = create_buffer_binding(output);
  DML_BINDING_DESC output_binding_desc =
      create_binding_desc(output_buffer_binding);

  compiled_op->Execute({input_binding_desc}, {output_binding_desc});
}

// Explicit template instantiations for DirectML
#define DECLARE_IMPL(T)                                                    \
  template void Concat::compute<Device::DirectML, T>(                      \
      const std::vector<const StorageView*>& inputs, StorageView& output)  \
      const;                                                               \
  template void Split::compute<Device::DirectML, T>(                       \
      const StorageView& input, std::vector<StorageView*>& outputs) const; \
  template void Slide::compute<Device::DirectML, T>(                       \
      const StorageView& input, StorageView& output, const dim_t& index)   \
      const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML