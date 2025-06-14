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

  // Create JOIN operator descriptor using DmlOperatorDescBundle
  dml::utils::DmlOperatorDescBundle op_bundle;
  std::vector<DML_TENSOR_DESC> input_descs_for_op;
  input_descs_for_op.reserve(inputs.size());

  for (const auto* input_sv : inputs) {
    input_descs_for_op.push_back(
        op_bundle.AddInput(*input_sv).get_tensor_desc());
  }

  const auto& output_desc_bundle = op_bundle.AddOutput(output);

  auto& join_desc = op_bundle.GetOperatorDesc<DML_JOIN_OPERATOR_DESC>();
  join_desc.InputCount = static_cast<UINT>(inputs.size());
  join_desc.InputTensors = input_descs_for_op.data();
  join_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  join_desc.Axis = static_cast<UINT>(axis);

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  // Create input bindings using DmlBindingArrayBundle
  std::vector<dml::utils::DmlBufferBindingBundle> input_binding_bundles;
  input_binding_bundles.reserve(inputs.size());
  for (const auto* input_sv : inputs) {
    input_binding_bundles.emplace_back(
        dml::utils::ResourceFromStorageView(*input_sv), 0,
        input_sv->size() * input_sv->item_size());
  }
  dml::utils::DmlBindingArrayBundle input_bindings(
      std::move(input_binding_bundles));

  // Create output binding using DmlBufferBindingBundle
  dml::utils::DmlBufferBindingBundle output_binding(
      dml::utils::ResourceFromStorageView(output), 0,
      output.size() * output.item_size());

  compiled_op->Execute(input_bindings.get_descs(), {output_binding.get_desc()});
}

template <Device D, typename T>
void Split::compute(const StorageView& input,
                    std::vector<StorageView*>& outputs) const {
  if (outputs.empty())
    return;

  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create SPLIT operator descriptor using DmlOperatorDescBundle
  dml::utils::DmlOperatorDescBundle op_bundle;
  const auto& input_desc_bundle = op_bundle.AddInput(input);

  std::vector<DML_TENSOR_DESC> output_descs_for_op;
  output_descs_for_op.reserve(outputs.size());
  for (const auto* output_sv : outputs) {
    output_descs_for_op.push_back(
        op_bundle.AddOutput(*output_sv).get_tensor_desc());
  }

  auto& split_desc = op_bundle.GetOperatorDesc<DML_SPLIT_OPERATOR_DESC>();
  split_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  split_desc.OutputCount = static_cast<UINT>(outputs.size());
  split_desc.OutputTensors = output_descs_for_op.data();
  split_desc.Axis = static_cast<UINT>(axis);

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  // Create input binding using DmlBufferBindingBundle
  dml::utils::DmlBufferBindingBundle input_binding(
      dml::utils::ResourceFromStorageView(input), 0,
      input.size() * input.item_size());

  // Create output bindings using DmlBindingArrayBundle
  std::vector<dml::utils::DmlBufferBindingBundle> output_binding_bundles;
  output_binding_bundles.reserve(outputs.size());
  for (const auto* output_sv : outputs) {
    output_binding_bundles.emplace_back(
        dml::utils::ResourceFromStorageView(*output_sv), 0,
        output_sv->size() * output_sv->item_size());
  }
  dml::utils::DmlBindingArrayBundle output_bindings(
      std::move(output_binding_bundles));

  compiled_op->Execute({input_binding.get_desc()}, output_bindings.get_descs());
}

template <Device D, typename T>
void Slide::compute(const StorageView& input,
                    StorageView& output,
                    const dim_t& index) const {
  const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;

  auto* device = dml::get_device();

  // Create SLICE operator descriptor using DmlOperatorDescBundle
  dml::utils::DmlOperatorDescBundle op_bundle;
  const auto& input_desc_bundle = op_bundle.AddInput(input);
  const auto& output_desc_bundle = op_bundle.AddOutput(output);

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

  auto& slice_desc = op_bundle.GetOperatorDesc<DML_SLICE_OPERATOR_DESC>();
  slice_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  slice_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  slice_desc.DimensionCount = static_cast<UINT>(input.rank());
  slice_desc.Offsets = offsets.data();
  slice_desc.Sizes = sizes.data();
  slice_desc.Strides = strides.data();

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

  // Create input and output bindings
  dml::utils::DmlBindingArrayBundle input_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(input))});
  dml::utils::DmlBindingArrayBundle output_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(output), 0,
          output.size() * output.item_size())});

  compiled_op->Execute(input_bindings.get_descs(), output_bindings.get_descs());
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