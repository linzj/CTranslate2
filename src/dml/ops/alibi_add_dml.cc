#include "ctranslate2/ops/alibi_add.h"

#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/utils.h"
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Helper to convert DataType to DML_TENSOR_DATA_TYPE
static DML_TENSOR_DATA_TYPE to_dml_data_type(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    default:
      THROW_INVALID_ARGUMENT(
          "Unsupported data type for DirectML in AlibiAdd: " +
          dtype_name(type));
  }
}

// Helper structure for tensor descriptions
struct DmlTensorDescBundle {
  DML_BUFFER_TENSOR_DESC buffer_desc{};
  DML_TENSOR_DESC tensor_desc{};
  std::vector<UINT> sizes_vec;
  std::vector<UINT> strides_vec;

  // Constructor for standard tensors
  DmlTensorDescBundle(const StorageView& storage) {
    const auto& shape = storage.shape();
    const auto dtype = storage.dtype();
    const size_t item_size = storage.item_size();

    sizes_vec.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      sizes_vec[i] = static_cast<UINT>(shape[i]);
    }

    buffer_desc.DataType = to_dml_data_type(dtype);
    buffer_desc.DimensionCount = static_cast<UINT>(sizes_vec.size());
    buffer_desc.Sizes = sizes_vec.data();
    buffer_desc.Strides = nullptr;  // Contiguous by default
    buffer_desc.TotalTensorSizeInBytes = storage.size() * storage.item_size();
    buffer_desc.GuaranteedBaseOffsetAlignment = 0;

    tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    tensor_desc.Desc = &buffer_desc;
  }

  // Constructor for broadcasted views
  DmlTensorDescBundle(DataType data_type,
                      const std::vector<UINT>& sizes,
                      const std::vector<UINT>& strides,
                      UINT64 total_size_bytes) {
    sizes_vec = sizes;
    strides_vec = strides;

    buffer_desc.DataType = to_dml_data_type(data_type);
    buffer_desc.DimensionCount = static_cast<UINT>(sizes_vec.size());
    buffer_desc.Sizes = sizes_vec.data();
    buffer_desc.Strides = strides_vec.data();
    buffer_desc.TotalTensorSizeInBytes = total_size_bytes;
    buffer_desc.GuaranteedBaseOffsetAlignment = 0;

    tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    tensor_desc.Desc = &buffer_desc;
  }
};

template <Device D, typename T>
void AlibiAdd::compute(const StorageView& input,
                       const StorageView& alibi,
                       const dim_t alibi_offset,
                       StorageView& output) const {
  if (input.dtype() == DataType::BFLOAT16) {
    THROW_INVALID_ARGUMENT("DirectML AlibiAdd doesn't support bfloat16");
  }

  // Get device and tensor properties
  dml::Device* device = dml::get_device();
  const auto& input_shape = input.shape();
  const dim_t batch_size = input_shape[0];
  const dim_t num_heads = input_shape[1];
  const dim_t query_length = input_shape[2];
  const dim_t key_length = input_shape[3];
  const dim_t cached_key_length = alibi.dim(1);
  const size_t item_size = input.item_size();
  const UINT64 alibi_byte_offset =
      static_cast<UINT64>(alibi_offset) * item_size;

  // Resize output to match input
  output.resize_as(input);

  // Create tensor descriptors
  DmlTensorDescBundle input_desc(input);
  DmlTensorDescBundle output_desc(output);

  // Create broadcasted alibi view
  std::vector<UINT> alibi_sizes = {1,  // Batch (broadcasted)
                                   static_cast<UINT>(num_heads),
                                   1,  // Query length (broadcasted)
                                   static_cast<UINT>(key_length)};

  std::vector<UINT> alibi_strides = {
      0,  // Stride for batch dimension (broadcast)
      static_cast<UINT>(cached_key_length),  // Stride between heads
      0,  // Stride for query dimension (broadcast)
      1   // Stride within head
  };

  DmlTensorDescBundle alibi_view_desc(alibi.dtype(), alibi_sizes, alibi_strides,
                                      alibi.size() * alibi.item_size());

  // Create element-wise add operator
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &input_desc.tensor_desc;
  add_desc.BTensor = &alibi_view_desc.tensor_desc;
  add_desc.OutputTensor = &output_desc.tensor_desc;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc};

  // Get or create compiled operator
  dml::Operator* add_operator = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Execute the operator
  ID3D12Resource* input_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  ID3D12Resource* alibi_resource =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(alibi.buffer()));
  ID3D12Resource* output_resource =
      reinterpret_cast<ID3D12Resource*>(output.buffer());
  DML_BUFFER_BINDING input_buffer_binding = {
      input_resource, 0, static_cast<UINT64>(input.size() * item_size)};
  DML_BUFFER_BINDING alibi_buffer_binding = {
      alibi_resource, alibi_byte_offset,
      static_cast<UINT64>(alibi.size() * item_size)};
  std::vector<DML_BINDING_DESC> input_bindings = {
      {DML_BINDING_TYPE_BUFFER, &input_buffer_binding},
      {DML_BINDING_TYPE_BUFFER, &alibi_buffer_binding}};
  DML_BUFFER_BINDING output_buffer_binding = {
      output_resource, 0, static_cast<UINT64>(output.size() * item_size)};
  DML_BINDING_DESC output_binding_desc = {DML_BINDING_TYPE_BUFFER,
                                          &output_buffer_binding};

  add_operator->Execute(input_bindings, {output_binding_desc});
}

#define DECLARE_IMPL(T)                                 \
  template void AlibiAdd::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, dim_t, StorageView&) const;

DECLARE_IMPL(float)
#ifdef CT2_HAS_FLOAT16
DECLARE_IMPL(float16_t)
#endif

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
