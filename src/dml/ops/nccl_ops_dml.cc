#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/nccl_ops.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

// #ifdef CT2_WITH_DIRECTML // This ifdef is redundant as the whole file is
// guarded. Local getDMLDataTypeFromDataType removed. Will use
// dml::utils::get_dml_data_type.

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
void perform_dml_reduce_operation(
    const StorageView& input_sv,         // Renamed for clarity
    StorageView& output_sv,              // Renamed for clarity
    ReduceAll::RED_OP reduce_op_enum) {  // Renamed for clarity
  auto* dml_device_wrapper = dml::get_device();

  auto input_buffer =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input_sv.buffer()));
  auto output_buffer = reinterpret_cast<ID3D12Resource*>(output_sv.buffer());

  DML_REDUCE_FUNCTION dml_reduce_function =
      redop_to_dml_reduce_function(reduce_op_enum);

  dml::utils::DmlTensorDescBundle input_desc_bundle(input_sv);
  dml::utils::DmlTensorDescBundle output_desc_bundle(output_sv);
  // Output shape for ReduceAll is typically scalar or matches input for
  // element-wise application context (not here) Here, it's a reduction, so
  // output will be scalar-like. DmlTensorDescBundle for output_sv (likely
  // scalar or [1]) is correct.

  // Axes for reduction - reduce all axes
  std::vector<UINT> axes_to_reduce_vec;
  const auto& actual_input_dims = input_desc_bundle.get_sizes_vec();
  if (!actual_input_dims.empty()) {  // Only add axes if dims exist
    axes_to_reduce_vec.reserve(actual_input_dims.size());
    for (UINT i = 0; i < actual_input_dims.size(); ++i) {
      axes_to_reduce_vec.push_back(i);
    }
  }
  // Handle 0-rank (scalar) input: to_dml_dims makes it rank 1, size 1. Reducing
  // axis 0 is fine.
  if (actual_input_dims.size() == 1 && axes_to_reduce_vec.empty()) {
    axes_to_reduce_vec.push_back(0);
  }

  DML_REDUCE_OPERATOR_DESC reduce_op_payload = {};
  reduce_op_payload.Function = dml_reduce_function;
  reduce_op_payload.InputTensor = &input_desc_bundle.get_tensor_desc();
  reduce_op_payload.OutputTensor = &output_desc_bundle.get_tensor_desc();
  reduce_op_payload.AxisCount = static_cast<UINT>(axes_to_reduce_vec.size());
  reduce_op_payload.Axes =
      axes_to_reduce_vec.empty() ? nullptr : axes_to_reduce_vec.data();

  DML_OPERATOR_DESC dml_op_wrapper = {};
  dml_op_wrapper.Type = DML_OPERATOR_REDUCE;
  dml_op_wrapper.Desc = &reduce_op_payload;

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&dml_op_wrapper);

  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          input_buffer, 0,
          input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          output_buffer, 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  compiled_op->Execute({input_binding_desc_for_op},
                       {output_binding_desc_for_op});
}
// #endif // Redundant CT2_WITH_DIRECTML removed from here

template <Device D, typename T>
void ReduceAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  perform_dml_reduce_operation<T>(input, output, _reduce_op);
#else
  (void)input;  // Supress unused parameter warning if CT2_WITH_TENSOR_PARALLEL
                // is not defined
  (void)output;
#endif
}

template <Device D, typename T>
void GatherAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
  auto* device = dml::get_device();  // ctranslate2::dml::Device

  auto input_buffer =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto output_buffer = reinterpret_cast<ID3D12Resource*>(output.buffer());

  // Simple copy operation using DirectML identity operator
  dml::utils::DmlTensorDescBundle input_desc_bundle(input);
  // For GatherAll, output is typically larger, concatenation of inputs from all
  // devices. Here, as a placeholder for single-device/local op, it's just a
  // copy. So, output StorageView should be sized correctly by the caller for a
  // copy.
  dml::utils::DmlTensorDescBundle output_desc_bundle(output);

  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
  identity_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  identity_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  identity_desc.ScaleBias = nullptr;

  DML_OPERATOR_DESC op_desc_wrapper = {};
  op_desc_wrapper.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
  op_desc_wrapper.Desc = &identity_desc;

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);

  // Binding Table method (more robust for ops that might need temp/persistent
  // resources)
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  dml::get_dml_device()->CreateBindingTable(nullptr,
                                            IID_PPV_ARGS(&binding_table));

  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          input_buffer, 0,
          input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          output_buffer, 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  binding_table->BindInputs(1, &input_binding_desc_for_op);
  binding_table->BindOutputs(1, &output_binding_desc_for_op);

  auto binding_props = compiled_op->GetBindingProperties();
  Microsoft::WRL::ComPtr<ID3D12Resource> temp_resource;
  if (binding_props.TemporaryResourceSize > 0) {
    temp_resource = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.TemporaryResourceSize);
    DML_BUFFER_BINDING temp_binding_storage = dml::utils::create_buffer_binding(
        temp_resource.Get(), 0, binding_props.TemporaryResourceSize);
    DML_BINDING_DESC temp_binding_desc_for_op =
        dml::utils::create_binding_desc(&temp_binding_storage);
    binding_table->BindTemporaryResource(&temp_binding_desc_for_op);
    // KeepAliveUntilNextCommandListDispatch might be handled by device wrapper
    // if it owns temp_resource
  }
  Microsoft::WRL::ComPtr<ID3D12Resource> persistent_resource;
  if (binding_props.PersistentResourceSize > 0) {
    // This should typically be managed (created once, or provided if op needs
    // initialized state)
    persistent_resource = device->CreatePreferredDeviceMemoryBuffer(
        binding_props.PersistentResourceSize);
    DML_BUFFER_BINDING persistent_binding_storage =
        dml::utils::create_buffer_binding(persistent_resource.Get(), 0,
                                          binding_props.PersistentResourceSize);
    DML_BINDING_DESC persistent_binding_desc_for_op =
        dml::utils::create_binding_desc(&persistent_binding_storage);
    binding_table->BindPersistentResource(&persistent_binding_desc_for_op);
  }

  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
  // ExecuteCommandList might be managed by the Device wrapper or called
  // explicitly by the framework after multiple op recordings. For simplicity
  // here, assuming RecordDispatch also submits or queues for later submission.
  // If using the more complex dml::Operator wrapper's Execute, it often handles
  // this. Since we used GetOrCreateCompiledOperatorApi, RecordDispatch is the
  // next step for direct DML API style. The simple compiled_op->Execute(inputs,
  // outputs) is a higher-level abstraction not directly used with manual
  // binding table.

#else
  (void)input;  // Supress unused
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
