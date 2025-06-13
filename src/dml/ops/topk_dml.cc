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

  dml::utils::DmlOperatorDescBundle op_desc_bundle;
  auto& input_desc_bundle = op_desc_bundle.AddInput(x);
  auto& values_desc_bundle = op_desc_bundle.AddOutput(values);
  auto& indices_desc_bundle = op_desc_bundle.AddOutput(indices);

  // Need to reset the dml data type for indices if x is sint32.
  if (indices.dtype() == ctranslate2::DataType::INT32) {
    indices_desc_bundle.set_data_type(DML_TENSOR_DATA_TYPE_UINT32);
  }

  auto& topk_op_desc =
      op_desc_bundle.GetOperatorDesc<DML_TOP_K1_OPERATOR_DESC>();
  topk_op_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
  topk_op_desc.OutputValueTensor = &values_desc_bundle.get_tensor_desc();
  topk_op_desc.OutputIndexTensor = &indices_desc_bundle.get_tensor_desc();
  topk_op_desc.Axis = dml_axis;
  topk_op_desc.K = static_cast<UINT>(_k);
  topk_op_desc.AxisDirection = DML_AXIS_DIRECTION_DECREASING;

  // Ensure using GetOrCreateCompiledOperatorApi from dml namespace if that's
  // intended
  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      std::move(op_desc_bundle), DML_EXECUTION_FLAG_NONE);

  // Bindings
  dml::utils::DmlBindingArrayBundle input_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(x))});

  dml::utils::DmlBindingArrayBundle output_bindings(
      {dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(values)),
       dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(indices))});

  compiled_op->Execute(input_bindings.get_descs(), output_bindings.get_descs());
}

#define DECLARE_IMPL_DML_TOPK(DataType, IndexType)                    \
  template void TopK::compute<Device::DirectML, DataType, IndexType>( \
      const StorageView& x, StorageView& values, StorageView& indices) const;

DECLARE_IMPL_DML_TOPK(float, int32_t)
DECLARE_IMPL_DML_TOPK(float16_t, int32_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
