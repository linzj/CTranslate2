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
void Gather::compute(const StorageView& data,
                     const StorageView& indices,
                     const dim_t axis,
                     const dim_t batch_dims,
                     StorageView& output) const {
  if (axis != batch_dims) {
    throw std::invalid_argument(
        "Gather only supports indexing the first non batch dimension (axis == "
        "batch_dims)");
  }

  if (indices.dtype() != DataType::INT32) {
    throw std::invalid_argument(
        "Gather indices must be of type INT32, but got " +
        dtype_name(indices.dtype()));
  }

  const dim_t original_data_rank = data.rank();
  const dim_t original_indices_rank =
      indices.rank();  // Used for DML's IndexDimensions
  const dim_t target_dml_rank =
      output.rank();  // All DML tensors need this rank
  auto* dml_device_wrapper = dml::get_device();  // ctranslate2::dml::Device

  // Use copies for potential reshaping. StorageView copy is shallow.
  // ScopedReshape will modify these copies' metadata and restore on
  // destruction.
  StorageView data_for_dml = data;
  StorageView indices_for_dml = indices;

  dim_t dml_axis = axis;  // Axis to be used for DML, may be adjusted

  std::unique_ptr<dml::utils::ScopedReshape> scoped_data_reshape;
  if (original_data_rank < target_dml_rank) {
    Shape new_data_shape;
    dim_t dims_to_prepend = target_dml_rank - original_data_rank;
    for (dim_t i = 0; i < dims_to_prepend; ++i) {
      new_data_shape.push_back(1);
    }
    for (dim_t i = 0; i < original_data_rank; ++i) {
      new_data_shape.push_back(data.shape()[i]);
    }
    // data_for_dml is non-const, ScopedReshape is fine.
    scoped_data_reshape.reset(
        new dml::utils::ScopedReshape(data_for_dml, new_data_shape));
    dml_axis += dims_to_prepend;  // Adjust axis due to prepended dimensions
  }

  std::unique_ptr<dml::utils::ScopedReshape> scoped_indices_reshape;
  if (original_indices_rank < target_dml_rank) {
    Shape new_indices_shape;
    dim_t dims_to_prepend = target_dml_rank - original_indices_rank;
    for (dim_t i = 0; i < dims_to_prepend; ++i) {
      new_indices_shape.push_back(1);
    }
    for (dim_t i = 0; i < original_indices_rank; ++i) {
      new_indices_shape.push_back(indices.shape()[i]);
    }
    // indices_for_dml is non-const, ScopedReshape is fine.
    scoped_indices_reshape.reset(
        new dml::utils::ScopedReshape(indices_for_dml, new_indices_shape));
  }
  // Note: The original ScopedReshape for indices compared indices.rank with
  // data.rank. The new logic correctly compares with target_dml_rank
  // (output.rank).

  // Create tensor descriptors using the (potentially reshaped) views
  dml::utils::DmlTensorDescBundle data_desc_bundle(data_for_dml);
  dml::utils::DmlTensorDescBundle indices_desc_bundle(indices_for_dml);
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output);  // Output is already at target_dml_rank

  // Create gather operator descriptor
  DML_GATHER_OPERATOR_DESC gather_desc = {};
  gather_desc.InputTensor = &data_desc_bundle.get_tensor_desc();
  gather_desc.IndicesTensor = &indices_desc_bundle.get_tensor_desc();
  gather_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
  gather_desc.Axis = static_cast<UINT>(dml_axis);  // Use the adjusted axis
  gather_desc.IndexDimensions = static_cast<UINT>(original_indices_rank);

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_GATHER;
  op_desc.Desc = &gather_desc;

  // Get or create compiled operator from cache
  auto compiled_operator = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create and bind resources using dml::utils
  DML_BUFFER_BINDING data_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(data_for_dml), 0,
          data_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BUFFER_BINDING indices_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(indices_for_dml), 0,
          indices_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  std::vector<DML_BINDING_DESC> input_bindings_for_op = {
      dml::utils::create_binding_desc(&data_buffer_binding_storage),
      dml::utils::create_binding_desc(&indices_buffer_binding_storage)};

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          dml::utils::ResourceFromStorageView(output), 0,
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