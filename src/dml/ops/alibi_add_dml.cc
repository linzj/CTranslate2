#include "ctranslate2/ops/alibi_add.h"

#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/utils.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added for centralized DML utilities
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Local to_dml_data_type and DmlTensorDescBundle are removed.
// They will be replaced by ctranslate2::dml::utils::get_dml_data_type
// and ctranslate2::dml::utils::DmlTensorDescBundle respectively.

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
  dml::utils::DmlTensorDescBundle input_desc(input);
  dml::utils::DmlTensorDescBundle output_desc(output);

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

  dml::utils::DmlTensorDescBundle alibi_view_desc(
      alibi.dtype(), alibi_sizes, &alibi_strides,  // Pass pointer to strides
      alibi.size() * alibi.item_size());

  // Create element-wise add operator
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &input_desc.get_tensor_desc();
  add_desc.BTensor = &alibi_view_desc.get_tensor_desc();
  add_desc.OutputTensor = &output_desc.get_tensor_desc();

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
