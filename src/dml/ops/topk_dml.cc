#include "ctranslate2/ops/topk.h"

#ifdef CT2_WITH_DIRECTML

#include "dml/dxdevice.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {
namespace {
// Helper to convert ctranslate2::DataType to DML_TENSOR_DATA_TYPE
inline DML_TENSOR_DATA_TYPE ToDmlDataType(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    // DML TOP_K* output indices are UINT32 or UINT64.
    // ctranslate2 uses int32_t for indices. We'll request UINT32 from DML
    // and the StorageView for indices should be INT32.
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_UINT32;
    // Add other type mappings as needed for other operators
    // e.g., DML_TENSOR_DATA_TYPE_UINT8 for boolean outputs from comparisons
    default:
      THROW_INVALID_ARGUMENT(
          "Unsupported ctranslate2::DataType for DirectML: " +
          dtype_name(type));
  }
}

// Helper to create DML_TENSOR_DESC from StorageView
// - dml_dims_buffer: Output parameter, filled with DML-compatible dimension
// sizes. Must outlive DML_BUFFER_TENSOR_DESC.
// - buffer_desc: Output parameter, filled with the DML buffer tensor
// description.
// - dml_strides_buffer: Optional output parameter for strides. If provided,
// filled and DML_BUFFER_TENSOR_DESC points to it. Must outlive
// DML_BUFFER_TENSOR_DESC.
inline DML_TENSOR_DESC make_tensor_desc(
    const StorageView& storage,
    std::vector<UINT>& dml_dims_buffer,
    DML_BUFFER_TENSOR_DESC* buffer_desc,
    std::vector<UINT>* dml_strides_buffer = nullptr) {
  dml_dims_buffer.clear();
  const auto& shape = storage.shape();
  const dim_t rank = storage.rank();

  if (rank == 0) {  // Scalar
    if (storage.size() != 1) {
      THROW_INVALID_ARGUMENT("Scalar StorageView must have size 1, but got " +
                             std::to_string(storage.size()));
    }
    dml_dims_buffer.push_back(1);  // Represent scalar as 1D tensor of size 1
  } else {
    for (dim_t d : shape) {
      // DML dimensions must be positive. A dimension of 0 means an empty
      // tensor.
      if (d < 0) {
        THROW_INVALID_ARGUMENT(
            "Tensor dimension must be non-negative, but got " +
            std::to_string(d));
      }
      dml_dims_buffer.push_back(static_cast<UINT>(d));
    }
  }
  // DML requires a minimum of 1 dimension for buffer tensors if not using
  // DML_TENSOR_DIMENSION_COUNT_MAX (which implies broadcasting). If
  // dml_dims_buffer is empty (e.g. truly 0D tensor, not scalar), make it {1} if
  // size is 1, or {0} if size is 0.
  if (dml_dims_buffer.empty()) {
    if (storage.size() == 1)
      dml_dims_buffer.push_back(1);  // Scalar represented as {1}
    else if (storage.size() == 0)
      dml_dims_buffer.push_back(0);  // Empty tensor {0}
    else
      THROW_INVALID_ARGUMENT(
          "Cannot determine DML dimensions for a 0-rank tensor with size " +
          std::to_string(storage.size()));
  }

  buffer_desc->DataType = ToDmlDataType(storage.dtype());
  buffer_desc->Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc->DimensionCount = static_cast<UINT>(dml_dims_buffer.size());
  buffer_desc->Sizes = dml_dims_buffer.data();

  if (dml_strides_buffer) {
    dml_strides_buffer->clear();
    if (buffer_desc->DimensionCount > 0) {
      dml_strides_buffer->resize(buffer_desc->DimensionCount);
      // Assuming ctranslate2::StorageView stores data contiguously by default,
      // or that its internal layout implies standard row-major/column-major
      // strides. DirectML strides are in elements, not bytes. Standard
      // contiguous strides (C-layout / row-major like):
      (*dml_strides_buffer)[buffer_desc->DimensionCount - 1] = 1;
      for (int i = static_cast<int>(buffer_desc->DimensionCount) - 2; i >= 0;
           --i) {
        // If a dimension is 0 or 1, the stride calculation needs care,
        // but standard formula T_i = T_{i+1} * S_{i+1} holds.
        // If S_{i+1} is 0, then T_i can be considered 0 or based on higher
        // dims. DML usually expects product of trailing dimensions for stride
        // if that dim is not 0. If S_{i+1} is 0, the tensor is empty. Strides
        // for empty tensors might not matter or be {0...0}. If S_{i+1} is 1,
        // T_i = T_{i+1}.
        UINT next_dim_size = dml_dims_buffer[i + 1];
        (*dml_strides_buffer)[i] =
            (*dml_strides_buffer)[i + 1] *
            (next_dim_size == 0
                 ? 1
                 : next_dim_size);  // Treat dim 0 as 1 for stride calc to avoid
                                    // 0 stride unless it's the only element.
      }
      // Correct strides for dimensions of size 0 or 1.
      // If a dimension size is 0, the total number of elements is 0. Strides
      // can be all 0s. If a dimension size is 1, its stride doesn't change the
      // offset calculation for that dimension. The above calculation is
      // standard for contiguous.
      if (storage.size() == 0) {  // If tensor is empty, set all strides to 0
        std::fill(dml_strides_buffer->begin(), dml_strides_buffer->end(), 0);
      }

      buffer_desc->Strides = dml_strides_buffer->data();
    } else {
      buffer_desc->Strides = nullptr;
    }
  } else {
    buffer_desc->Strides = nullptr;  // Null strides implies contiguous tensor.
  }

  // Calculate TotalTensorSizeInBytes: minimum size in bytes for the described
  // tensor.
  const UINT element_size_in_bytes = storage.item_size();
  if (storage.size() == 0) {
    buffer_desc->TotalTensorSizeInBytes = 0;
  } else if (buffer_desc->Strides) {
    UINT64 max_element_offset = 0;
    for (UINT i = 0; i < buffer_desc->DimensionCount; ++i) {
      if (dml_dims_buffer[i] >
          0) {  // Only contribute if dimension is not empty
        max_element_offset += (static_cast<UINT64>(dml_dims_buffer[i]) - 1) *
                              (*dml_strides_buffer)[i];
      }
    }
    buffer_desc->TotalTensorSizeInBytes =
        (max_element_offset + 1) * element_size_in_bytes;
  } else {  // Contiguous
    buffer_desc->TotalTensorSizeInBytes =
        static_cast<UINT64>(storage.size()) * element_size_in_bytes;
  }

  // GuaranteedBaseOffsetAlignment is usually 0 unless specific alignment is
  // required by an op or hardware. DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT (16
  // bytes) is a general minimum.
  buffer_desc->GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC tensor_desc;
  tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc.Desc = buffer_desc;
  return tensor_desc;
}
}  // namespace

template <Device D, typename DataType, typename IndexType>
void TopK::compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only.");
  static_assert(std::is_same<IndexType, int32_t>::value,
                "DirectML TopK implementation expects IndexType = int32_t (for "
                "ctranslate2 StorageView).");

  // Ensure k is valid
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
  const UINT dml_axis =
      static_cast<UINT>(rank - 1);  // TopK along the last dimension
  const dim_t depth = input_shape[dml_axis];

  if (static_cast<dim_t>(_k) > depth) {
    THROW_INVALID_ARGUMENT(
        "TopK: k (" + std::to_string(_k) +
        ") cannot be greater than the size of the axis dimension (" +
        std::to_string(depth) + ").");
  }

  // If input is empty, outputs are also empty. DML should handle this if tensor
  // descs are correct.
  if (x.size() == 0) {
    // Ensure output shapes are correctly set to empty by the caller
    // (Op::compute_output_storage) No DML execution needed for empty inputs.
    return;
  }

  // These vectors store the dimension values for DML tensor descriptions.
  // They must remain in scope until the DML operator is created.
  std::vector<UINT> dml_input_dims, dml_values_dims, dml_indices_dims;
  // These structs store the DML buffer tensor descriptions.
  DML_BUFFER_TENSOR_DESC input_buffer_desc_storage, values_buffer_desc_storage,
      indices_buffer_desc_storage;

  DML_TENSOR_DESC input_tensor_desc =
      make_tensor_desc(x, dml_input_dims, &input_buffer_desc_storage);
  DML_TENSOR_DESC values_tensor_desc =
      make_tensor_desc(values, dml_values_dims, &values_buffer_desc_storage);
  // For indices, make_tensor_desc uses ToDmlDataType(indices.dtype()),
  // which maps DataType::INT32 to DML_TENSOR_DATA_TYPE_UINT32. This is correct
  // for DML's TopK.
  DML_TENSOR_DESC indices_tensor_desc =
      make_tensor_desc(indices, dml_indices_dims, &indices_buffer_desc_storage);

  // Create DML_TOP_K1_OPERATOR_DESC
  DML_TOP_K1_OPERATOR_DESC topk_op_desc_payload = {};
  topk_op_desc_payload.InputTensor = &input_tensor_desc;
  topk_op_desc_payload.OutputValueTensor = &values_tensor_desc;
  topk_op_desc_payload.OutputIndexTensor = &indices_tensor_desc;
  topk_op_desc_payload.Axis = dml_axis;
  topk_op_desc_payload.K = static_cast<UINT>(_k);
  topk_op_desc_payload.AxisDirection =
      DML_AXIS_DIRECTION_DECREASING;  // Get largest values (standard TopK)

  DML_OPERATOR_DESC dml_op_desc = {};
  dml_op_desc.Type =
      DML_OPERATOR_TOP_K1;  // Use TOP_K1 for AxisDirection support
  dml_op_desc.Desc = &topk_op_desc_payload;

  // Get or create the compiled DML operator from cache
  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      &dml_op_desc, DML_EXECUTION_FLAG_NONE);

  // Prepare bindings for execution
  // Input binding
  DML_BUFFER_BINDING input_buffer_binding = dml::create_buffer_binding(
      static_cast<ID3D12Resource*>(const_cast<void*>(x.buffer())), 0,
      x.reserved_memory());
  DML_BINDING_DESC input_binding_desc =
      dml::create_binding_desc(input_buffer_binding);

  // Output Values binding
  DML_BUFFER_BINDING values_buffer_binding =
      dml::create_buffer_binding(static_cast<ID3D12Resource*>(values.buffer()),
                                 0, values.reserved_memory());
  DML_BINDING_DESC values_binding_desc =
      dml::create_binding_desc(values_buffer_binding);

  // Output Indices binding
  DML_BUFFER_BINDING indices_buffer_binding =
      dml::create_buffer_binding(static_cast<ID3D12Resource*>(indices.buffer()),
                                 0, indices.reserved_memory());
  DML_BINDING_DESC indices_binding_desc =
      dml::create_binding_desc(indices_buffer_binding);

  // Execute the operator
  // The dml::Operator::Execute method is expected to handle command list
  // recording, binding table setup (including temporary/persistent resources if
  // the DML op needs them), and execution submission.
  compiled_op->Execute(
      {input_binding_desc},                        // Input bindings
      {values_binding_desc, indices_binding_desc}  // Output bindings
  );
}

// Explicitly instantiate templates for supported types on DirectML.
// DirectML TOP_K supports FLOAT32, FLOAT16.
// BFLOAT16 is not listed as a DML_TENSOR_DATA_TYPE in the provided DirectML.h
// snippet. If bfloat16_t is needed, it would require casting to/from float32 or
// a newer DML version.

#define DECLARE_IMPL_DML_TOPK(DataType, IndexType)                    \
  template void TopK::compute<Device::DirectML, DataType, IndexType>( \
      const StorageView& x, StorageView& values, StorageView& indices) const;

DECLARE_IMPL_DML_TOPK(float, int32_t)
DECLARE_IMPL_DML_TOPK(float16_t, int32_t)
// To enable bfloat16_t, ensure DML supports it or add manual casting
// operations. DECLARE_IMPL_DML_TOPK(bfloat16_t, int32_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
