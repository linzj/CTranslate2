#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/nccl_ops.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

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
void perform_dml_reduce_operation(const StorageView& input_sv,
                                  StorageView& output_sv,
                                  ReduceAll::RED_OP reduce_op_enum) {
  DML_REDUCE_FUNCTION dml_reduce_function =
      redop_to_dml_reduce_function(reduce_op_enum);

  dml::utils::DmlOperatorDescBundle op_desc;
  auto& input_desc_bundle = op_desc.AddInput(input_sv);
  auto& output_desc_bundle = op_desc.AddOutput(output_sv);

  std::vector<UINT> axes_to_reduce_vec;
  const auto& actual_input_dims = input_desc_bundle.get_sizes_vec();
  if (!actual_input_dims.empty()) {
    axes_to_reduce_vec.reserve(actual_input_dims.size());
    for (UINT i = 0; i < actual_input_dims.size(); ++i) {
      axes_to_reduce_vec.push_back(i);
    }
  }
  if (actual_input_dims.size() == 1 && axes_to_reduce_vec.empty()) {
    axes_to_reduce_vec.push_back(0);
  }

  auto& reduce_op_payload = op_desc.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
  reduce_op_payload.Function = dml_reduce_function;
  reduce_op_payload.InputTensor = &input_desc_bundle.get_tensor_desc();
  reduce_op_payload.OutputTensor = &output_desc_bundle.get_tensor_desc();
  reduce_op_payload.AxisCount = static_cast<UINT>(axes_to_reduce_vec.size());
  reduce_op_payload.Axes =
      axes_to_reduce_vec.empty() ? nullptr : axes_to_reduce_vec.data();

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

  dml::utils::DmlBindingArrayBundle inputs({dml::utils::DmlBufferBindingBundle(
      dml::utils::ResourceFromStorageView(input_sv))});
  dml::utils::DmlBindingArrayBundle outputs({dml::utils::DmlBufferBindingBundle(
      dml::utils::ResourceFromStorageView(output_sv))});
  compiled_op->Execute(inputs.get_descs(), outputs.get_descs());
}

template <Device D, typename T>
void ReduceAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  perform_dml_reduce_operation<T>(input, output, _reduce_op);
#else
  (void)input;
  (void)output;
#endif
}

template <Device D, typename T>
void GatherAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  dml::utils::DmlOperatorDescBundle op_desc;
  auto& input_desc_bundle = op_desc.AddInput(input);
  auto& output_desc_bundle = op_desc.AddOutput(output);

  auto& identity_desc =
      op_desc.GetOperatorDesc<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
  identity_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  identity_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();

  auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

  dml::utils::DmlBindingArrayBundle inputs({dml::utils::DmlBufferBindingBundle(
      dml::utils::ResourceFromStorageView(input))});
  dml::utils::DmlBindingArrayBundle outputs({dml::utils::DmlBufferBindingBundle(
      dml::utils::ResourceFromStorageView(output))});

  auto* device = dml::get_device();
  auto binding_props = compiled_op->GetBindingProperties();
  StorageView temp_resource, persistent_resource;

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  dml::get_dml_device()->CreateBindingTable(nullptr,
                                            IID_PPV_ARGS(&binding_table));

  binding_table->BindInputs(inputs.size(), inputs.get_descs().data());
  binding_table->BindOutputs(outputs.size(), outputs.get_descs().data());

  if (binding_props.TemporaryResourceSize > 0) {
    temp_resource.resize(binding_props.TemporaryResourceSize);
    DML_BUFFER_BINDING temp_binding = {
        dml::utils::ResourceFromStorageView(temp_resource), 0,
        binding_props.TemporaryResourceSize};
    binding_table->BindTemporaryResource(
        &dml::utils::create_binding_desc(&temp_binding));
  }

  if (binding_props.PersistentResourceSize > 0) {
    persistent_resource.resize(binding_props.PersistentResourceSize);
    DML_BUFFER_BINDING persistent_binding = {
        dml::utils::ResourceFromStorageView(persistent_resource), 0,
        binding_props.PersistentResourceSize};
    binding_table->BindPersistentResource(
        &dml::utils::create_binding_desc(&persistent_binding));
  }

  device->RecordDispatch(compiled_op.Get(), binding_table.Get());

#else
  (void)input;
  (void)output;
#endif
}

#define DECLARE_IMPL(T)                                                      \
  template void GatherAll::compute<Device::DirectML, T>(const StorageView&,  \
                                                        StorageView&) const; \
  template void ReduceAll::compute<Device::DirectML, T>(const StorageView&,  \
                                                        StorageView&) const;
DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2
#endif  // CT2_WITH_DIRECTML
