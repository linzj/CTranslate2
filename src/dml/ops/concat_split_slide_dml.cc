#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/split.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added for centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

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

  // Create input tensor descriptors using DmlTensorDescBundle
  std::vector<dml::utils::DmlTensorDescBundle> input_desc_bundles;
  std::vector<DML_TENSOR_DESC>
      input_tensor_descs_for_op;  // For DML_JOIN_OPERATOR_DESC
  input_desc_bundles.reserve(inputs.size());
  input_tensor_descs_for_op.reserve(inputs.size());

  for (const auto* input_sv : inputs) {
    input_desc_bundles.emplace_back(*input_sv);
    input_tensor_descs_for_op.push_back(
        input_desc_bundles.back().get_tensor_desc());
  }

  // Create output tensor descriptor
  dml::utils::DmlTensorDescBundle output_desc_bundle(output);
  const DML_TENSOR_DESC& output_tensor_desc_ref =
      output_desc_bundle.get_tensor_desc();

  // Create JOIN operator descriptor
  DML_JOIN_OPERATOR_DESC join_desc = {};
  join_desc.InputCount = static_cast<UINT>(inputs.size());
  join_desc.InputTensors = input_tensor_descs_for_op.data();
  join_desc.OutputTensor = &output_tensor_desc_ref;
  join_desc.Axis = static_cast<UINT>(axis);

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_JOIN;
  op_desc.Desc = &join_desc;

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create input bindings
  std::vector<DML_BUFFER_BINDING>
      input_buffer_bindings_storage;  // To keep DML_BUFFER_BINDING alive
  std::vector<DML_BINDING_DESC> input_binding_descs_for_op;
  input_buffer_bindings_storage.reserve(inputs.size());
  input_binding_descs_for_op.reserve(inputs.size());

  for (const auto* input_sv : inputs) {
    input_buffer_bindings_storage.push_back(dml::utils::create_buffer_binding(
        dml::utils::ResourceFromStorageView(*input_sv), 0,
        input_sv->size() * input_sv->item_size()));
    input_binding_descs_for_op.push_back(
        dml::utils::create_binding_desc(&input_buffer_bindings_storage.back()));
  }

  // Create output binding
  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(output), 0,
          output.size() * output.item_size());
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  compiled_op->Execute(input_binding_descs_for_op,
                       {output_binding_desc_for_op});
}

template <Device D, typename T>
void Split::compute(const StorageView& input,
                    std::vector<StorageView*>& outputs) const {
  if (outputs.empty())
    return;

  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create input tensor descriptor
  dml::utils::DmlTensorDescBundle input_desc_bundle(input);
  const DML_TENSOR_DESC& input_tensor_desc_ref =
      input_desc_bundle.get_tensor_desc();

  // Create output tensor descriptors
  std::vector<dml::utils::DmlTensorDescBundle> output_desc_bundles;
  std::vector<DML_TENSOR_DESC> output_tensor_descs_for_op;
  output_desc_bundles.reserve(outputs.size());
  output_tensor_descs_for_op.reserve(outputs.size());

  for (const auto* output_sv : outputs) {
    output_desc_bundles.emplace_back(*output_sv);
    output_tensor_descs_for_op.push_back(
        output_desc_bundles.back().get_tensor_desc());
  }

  // Create SPLIT operator descriptor
  DML_SPLIT_OPERATOR_DESC split_desc = {};
  split_desc.InputTensor = &input_tensor_desc_ref;
  split_desc.OutputCount = static_cast<UINT>(outputs.size());
  split_desc.OutputTensors = output_tensor_descs_for_op.data();
  split_desc.Axis = static_cast<UINT>(axis);

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_SPLIT;
  op_desc.Desc = &split_desc;

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create input binding
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(input), 0,
          input.size() * input.item_size());
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  // Create output bindings
  std::vector<DML_BUFFER_BINDING> output_buffer_bindings_storage;  // Keep alive
  std::vector<DML_BINDING_DESC> output_binding_descs_for_op;
  output_buffer_bindings_storage.reserve(outputs.size());
  output_binding_descs_for_op.reserve(outputs.size());

  for (const auto* output_sv : outputs) {
    output_buffer_bindings_storage.push_back(dml::utils::create_buffer_binding(
        dml::utils::ResourceFromStorageView(*output_sv), 0,
        output_sv->size() * output_sv->item_size()));
    output_binding_descs_for_op.push_back(dml::utils::create_binding_desc(
        &output_buffer_bindings_storage.back()));
  }

  compiled_op->Execute({input_binding_desc_for_op},
                       output_binding_descs_for_op);
}

template <Device D, typename T>
void Slide::compute(const StorageView& input,
                    StorageView& output,
                    const dim_t& index) const {
  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create input tensor descriptor
  dml::utils::DmlTensorDescBundle input_desc_bundle(input);
  const DML_TENSOR_DESC& input_tensor_desc_ref =
      input_desc_bundle.get_tensor_desc();

  // Create output tensor descriptor
  dml::utils::DmlTensorDescBundle output_desc_bundle(output);
  const DML_TENSOR_DESC& output_tensor_desc_ref =
      output_desc_bundle.get_tensor_desc();

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
  slice_desc.InputTensor = &input_tensor_desc_ref;
  slice_desc.OutputTensor = &output_tensor_desc_ref;
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
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(input), 0,
          input.size() * input.item_size());
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  // Create output binding
  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(output), 0,
          output.size() * output.item_size());
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  compiled_op->Execute({input_binding_desc_for_op},
                       {output_binding_desc_for_op});
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