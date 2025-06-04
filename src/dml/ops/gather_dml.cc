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
  // If indices rank is less than data rank, reshape a copy of indices by
  // prepending 1s. StorageView's copy constructor/assignment typically results
  // in an owning copy if the source has data. The reshape method then only
  // modifies the metadata (_shape) of this copy.
  // Create a copy to potentially modify its shape.

  const dim_t current_indices_rank = indices.rank();
  const dim_t target_data_rank = data.rank();

  Shape new_indices_shape = indices.shape();
  if (current_indices_rank < target_data_rank) {
    const size_t num_dims_to_prepend =
        static_cast<size_t>(target_data_rank - current_indices_rank);
    Shape new_shape_tmp(num_dims_to_prepend, 1);

    // Prepend 1s to match data's rank
    for (dim_t i = 0; i < current_indices_rank; ++i) {
      new_shape_tmp.push_back(indices.shape()[i]);
    }
    new_indices_shape = std::move(new_shape_tmp);
  }
  ID3D12Resource* indices_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(indices.buffer()));
  indices_buffer->AddRef();
  if (indices.dtype() != DataType::INT32) {
    throw std::invalid_argument(
        "Gather indices must be of type INT32, but got " +
        dtype_name(indices.dtype()));
  }
  StorageView indices_for_dml =
      StorageView(new_indices_shape,
                  static_cast<int32_t*>(static_cast<void*>(indices_buffer)),
                  Device::DirectML);

  dml::utils::DmlTensorDescBundle data_desc_bundle(data);
  dml::utils::DmlTensorDescBundle indices_desc_bundle(indices_for_dml);
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

  gather_desc.IndexDimensions = static_cast<UINT>(current_indices_rank);

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
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(
              indices_for_dml.buffer())),  // Use buffer from the (potentially
                                           // reshaped) copy
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