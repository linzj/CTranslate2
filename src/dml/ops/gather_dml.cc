#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/gather.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Gather::compute(
    const StorageView& data,  // The tensor to gather from
    const StorageView&
        indices,  // The indices to gather (renamed from 'input' for clarity)
    const dim_t axis,
    const dim_t batch_dims,
    StorageView& output) const {
  if (axis != batch_dims) {
    throw std::invalid_argument(
        "Gather only supports indexing the first non batch dimension (axis == "
        "batch_dims)");
  }

  auto* dml_device_wrapper = dml::get_device();  // ctranslate2::dml::Device

  // Create tensor descriptors using DmlTensorDescBundle
  // For 'indices', DML typically expects INT32 or UINT32.
  // Assuming 'indices' StorageView is already INT32 as per typical Gather op
  // usage. DmlTensorDescBundle will use dml::utils::get_dml_data_type.
  dml::utils::DmlTensorDescBundle data_desc_bundle(data);
  dml::utils::DmlTensorDescBundle indices_desc_bundle(indices);
  dml::utils::DmlTensorDescBundle output_desc_bundle(output);

  // Get DML tensor shapes for operator descriptor configuration if needed
  // explicitly const auto& data_dml_shape = data_desc_bundle.get_sizes_vec();
  const auto& indices_dml_shape = indices_desc_bundle.get_sizes_vec();
  // const auto& output_dml_shape = output_desc_bundle.get_sizes_vec();

  // Create gather operator descriptor
  DML_GATHER_OPERATOR_DESC gather_desc = {};
  gather_desc.InputTensor = &data_desc_bundle.get_tensor_desc();
  gather_desc.IndicesTensor = &indices_desc_bundle.get_tensor_desc();
  gather_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  gather_desc.Axis = static_cast<UINT>(axis);

  // IndexDimensions: "The number of an N-D input indices tensor that make up
  // the M-D logical indices tensor..." "This value must be between 1 and
  // DimensionCount of the indices tensor, inclusive." Typically, for gather
  // along one axis, if `indices` is purely an index tensor (not sharing batch
  // dims with `data`), IndexDimensions is the rank of `indices`. If `indices`
  // has batch dims that align with `data`'s batch dims up to `axis`, then
  // IndexDimensions should be rank(indices) - batch_dims. Given the constraint
  // axis == batch_dims, and `indices` are applied at that axis: E.g. data: [B,
  // S, H], indices: [B, I], axis=1 (S dim), batch_dims=1 (B dim) Here,
  // `indices` [B,I] implies the `I` part indexes along S. The effective index
  // rank is 1 (I). So IndexDimensions would be rank(indices) - batch_dims (if
  // batch_dims is for shared prefix with data) But if indices are (I, J, K) to
  // pick along axis, then indices_shape.size() is correct. The original
  // `indices_shape.size()` is often used if `indices` is purely
  // index-specifying. CTranslate2's Gather logic might need more detailed
  // mapping here. Sticking with the original `indices_shape.size()` as a
  // baseline assuming `indices` does not have additional batch dimensions
  // beyond what DML's Gather inherently handles by tensor alignment for other
  // axes. If `indices` has rank R, and all R dimensions are used for indexing
  // at `axis` (e.g. an R-dimensional block of indices), then IndexDimensions
  // would be R.
  gather_desc.IndexDimensions =
      static_cast<UINT>(indices_desc_bundle.get_sizes_vec().size());

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_GATHER;
  op_desc.Desc = &gather_desc;

  // Get or create compiled operator from cache
  auto compiled_operator = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create and bind resources using dml::utils
  // Keep DML_BUFFER_BINDING structs alive for create_binding_desc
  DML_BUFFER_BINDING data_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(data.buffer())),
          0, data_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BUFFER_BINDING indices_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(
              const_cast<void*>(indices.buffer())),
          0, indices_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  std::vector<DML_BINDING_DESC> input_bindings_for_op = {
      dml::utils::create_binding_desc(&data_buffer_binding_storage),
      dml::utils::create_binding_desc(&indices_buffer_binding_storage)};

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  std::vector<DML_BINDING_DESC> output_bindings_for_op = {
      dml::utils::create_binding_desc(&output_buffer_binding_storage)};

  compiled_operator->Execute(input_bindings_for_op, output_bindings_for_op);
}

#define DECLARE_IMPL(T)                                                    \
  template void Gather::compute<Device::DirectML, T>(                      \
      const StorageView& data, const StorageView& input, const dim_t axis, \
      const dim_t batch_dims, StorageView& output) const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML