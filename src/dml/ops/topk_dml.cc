#include "ctranslate2/ops/topk.h"

#ifdef CT2_WITH_DIRECTML

#include "dml/dml_utils.h"
#include "dml/dxdevice.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename DataType, typename IndexType>
void TopK::compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only.");
  static_assert(std::is_same<IndexType, int32_t>::value,
                "DirectML TopK implementation expects IndexType = int32_t (for "
                "ctranslate2 StorageView).");

  if (_k <= 0) {
    THROW_INVALID_ARGUMENT("TopK: k must be greater than 0, but got " +
                           std::to_string(_k));
  }

  const Shape& input_shape = x.shape();
  if (input_shape.empty()) {
    THROW_INVALID_ARGUMENT(
        "Input tensor for TopK cannot be a scalar or 0-rank.");
  }

  const dim_t rank = input_shape.size();
  const UINT dml_axis = static_cast<UINT>(rank - 1);
  const dim_t depth = input_shape[dml_axis];

  if (static_cast<dim_t>(_k) > depth) {
    THROW_INVALID_ARGUMENT(
        "TopK: k (" + std::to_string(_k) +
        ") cannot be greater than the size of the axis dimension (" +
        std::to_string(depth) + ").");
  }

  if (x.size() == 0) {
    return;
  }

  dml::utils::DmlTensorDescBundle input_desc_bundle(x);
  dml::utils::DmlTensorDescBundle values_desc_bundle(values);
  dml::utils::DmlTensorDescBundle indices_desc_bundle(indices);

  // Need to reset the dml data type for indices if x is sint32.
  if (indices.dtype() == ctranslate2::DataType::INT32) {
    indices_desc_bundle.set_data_type(DML_TENSOR_DATA_TYPE_UINT32);
  }

  const DML_TENSOR_DESC& dml_input_tensor_desc =
      input_desc_bundle.get_tensor_desc();
  const DML_TENSOR_DESC& dml_values_tensor_desc =
      values_desc_bundle.get_tensor_desc();
  const DML_TENSOR_DESC& dml_indices_tensor_desc =
      indices_desc_bundle.get_tensor_desc();

  DML_TOP_K1_OPERATOR_DESC topk_op_desc_payload = {};
  topk_op_desc_payload.InputTensor = &dml_input_tensor_desc;
  topk_op_desc_payload.OutputValueTensor = &dml_values_tensor_desc;
  topk_op_desc_payload.OutputIndexTensor = &dml_indices_tensor_desc;
  topk_op_desc_payload.Axis = dml_axis;
  topk_op_desc_payload.K = static_cast<UINT>(_k);
  topk_op_desc_payload.AxisDirection = DML_AXIS_DIRECTION_DECREASING;

  DML_OPERATOR_DESC dml_op_desc = {};
  dml_op_desc.Type = DML_OPERATOR_TOP_K1;
  dml_op_desc.Desc = &topk_op_desc_payload;

  // Ensure using GetOrCreateCompiledOperatorApi from dml namespace if that's
  // intended
  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      &dml_op_desc, DML_EXECUTION_FLAG_NONE);

  // Bindings using dml::utils::create_buffer_binding and
  // dml::utils::create_binding_desc
  DML_BUFFER_BINDING input_buffer_binding_s =
      dml::utils::create_buffer_binding(  // Fully qualified
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(x.buffer())), 0,
          input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc = dml::utils::create_binding_desc(
      &input_buffer_binding_s);  // Fully qualified

  DML_BUFFER_BINDING values_buffer_binding_s =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(
              values.buffer()),  // Fully qualified
          0, values_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC values_binding_desc = dml::utils::create_binding_desc(
      &values_buffer_binding_s);  // Fully qualified

  DML_BUFFER_BINDING indices_buffer_binding_s =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(
              indices.buffer()),  // Fully qualified
          0, indices_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC indices_binding_desc = dml::utils::create_binding_desc(
      &indices_buffer_binding_s);  // Fully qualified

  compiled_op->Execute({input_binding_desc},
                       {values_binding_desc, indices_binding_desc});
}

#define DECLARE_IMPL_DML_TOPK(DataType, IndexType)                    \
  template void TopK::compute<Device::DirectML, DataType, IndexType>( \
      const StorageView& x, StorageView& values, StorageView& indices) const;

DECLARE_IMPL_DML_TOPK(float, int32_t)
DECLARE_IMPL_DML_TOPK(float16_t, int32_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
