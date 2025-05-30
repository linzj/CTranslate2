#pragma once

#include "ctranslate2/storage_view.h"
#include "dml/dxdevice.h"

// Forward declaration for a helper function that might be used internally or
// defined later
namespace ctranslate2 {
std::string dtype_name(
    DataType type);  // Declaration from ctranslate2/types.h or similar
}

namespace ctranslate2 {
namespace dml {
namespace utils {

// --- Error Handling ---
// Basic THROW_INVALID_ARGUMENT (can be replaced by ctranslate2's existing macro
// if available)
#ifndef THROW_INVALID_ARGUMENT
#define THROW_INVALID_ARGUMENT(msg) throw std::invalid_argument(msg)
#endif

// --- Data Type & Size Utilities ---
DML_TENSOR_DATA_TYPE get_dml_data_type(DataType ct2_type);
size_t get_dml_element_size_in_bytes(DML_TENSOR_DATA_TYPE dml_type);
// DataType from_dml_data_type(DML_TENSOR_DATA_TYPE dml_type); // TODO:
// Implement if needed

// --- Shape & Dimension Utilities ---
// Converts ctranslate2::Shape to std::vector<UINT> for DML.
// If ensure_at_least_1d_for_dml is true and the input shape is scalar/empty,
// it will produce {1} if storage_size is 1, or {0} if storage_size is 0.
std::vector<UINT> to_dml_dims(const Shape& shape,
                              size_t storage_size,
                              bool ensure_at_least_1d_for_dml = true);

// Special case for 4D shapes often required by DML ops (e.g., NCHW for Conv).
// Maps (Batch, Channels, Width) to (Batch, Channels, 1, Width) for
// input/output. Weight: (OutChannels, InChannels/Groups, KernelW) ->
// (OutChannels, InChannels/Groups, 1, KernelW). Bias: (OutChannels) -> (1,
// OutChannels, 1, 1) for broadcasting.
std::vector<UINT> get_dml_tensor_shape_4d(const StorageView& tensor,
                                          bool is_filter_or_bias = false);

// --- DML Tensor Descriptor Utilities ---

// Helper to create a DML_BUFFER_TENSOR_DESC and corresponding DML_TENSOR_DESC.
// `dml_dims_buffer` and `dml_strides_buffer` (if not null) are populated and
// used by `buffer_desc`. These buffers must outlive `buffer_desc` and
// `tensor_desc`.
DML_TENSOR_DESC make_tensor_desc_from_storage(
    const StorageView& storage,
    std::vector<UINT>&
        dml_dims_buffer,  // Output: populated with DML dimensions
    DML_BUFFER_TENSOR_DESC&
        buffer_desc,  // Output: populated DML buffer tensor desc
    std::vector<UINT>* dml_strides_buffer =
        nullptr  // Optional Output: populated with DML strides
);

// Manages DML_BUFFER_TENSOR_DESC and DML_TENSOR_DESC, including internal
// storage for sizes/strides vectors. This is useful for situations where tensor
// metadata needs to be managed throughout DML operations.
class DmlTensorDescBundle {
 public:
  DmlTensorDescBundle() = default;

  // Constructor from StorageView. Calculates contiguous strides by default.
  // `strides_override` can be used for non-contiguous or broadcasted views. If
  // null, contiguous strides computed.
  DmlTensorDescBundle(const StorageView& storage,
                      const std::vector<UINT>* strides_override = nullptr);

  // Constructor from explicit DML data type, sizes, strides, and total byte
  // size. If `strides` is null, tensor is assumed contiguous or strides
  // computed internally from `sizes`. If `total_tensor_size_in_bytes` is 0,
  // it's calculated.
  DmlTensorDescBundle(DML_TENSOR_DATA_TYPE dml_data_type,
                      const std::vector<UINT>& sizes,
                      const std::vector<UINT>* strides,
                      UINT64 total_tensor_size_in_bytes = 0);

  // Constructor from ctranslate2 DataType, sizes, strides, and total byte size.
  DmlTensorDescBundle(DataType ct2_data_type,
                      const std::vector<UINT>& sizes,
                      const std::vector<UINT>* strides,
                      UINT64 total_tensor_size_in_bytes = 0);

  const DML_BUFFER_TENSOR_DESC& get_buffer_desc() const {
    return buffer_desc_internal;
  }
  const DML_TENSOR_DESC& get_tensor_desc() const {
    return tensor_desc_internal;
  }

  const std::vector<UINT>& get_sizes_vec() const { return internal_sizes_vec; }
  const std::vector<UINT>& get_strides_vec() const {
    return internal_strides_vec;
  }

 private:
  std::vector<UINT> internal_sizes_vec;
  std::vector<UINT> internal_strides_vec;  // May be empty if Strides = nullptr
  DML_BUFFER_TENSOR_DESC buffer_desc_internal{};
  DML_TENSOR_DESC tensor_desc_internal{};

  void init_from_storage(const StorageView& storage,
                         const std::vector<UINT>* strides_override);
  void init_from_details(DML_TENSOR_DATA_TYPE dml_dtype,
                         const std::vector<UINT>& sizes_in,
                         const std::vector<UINT>* strides_in,
                         UINT64 total_tensor_size_bytes_in);
  void calculate_strides_and_total_size(DML_TENSOR_DATA_TYPE dml_dtype,
                                        const std::vector<UINT>& sizes,
                                        const std::vector<UINT>* strides_input,
                                        UINT64& total_size_in_bytes_ref);
};

// Helper to wrap an existing DML_BUFFER_TENSOR_DESC into a DML_TENSOR_DESC
// The DML_BUFFER_TENSOR_DESC pointed to by buffer_desc_ptr must remain valid.
inline DML_TENSOR_DESC wrap_dml_buffer_tensor_desc(
    const DML_BUFFER_TENSOR_DESC* buffer_desc_ptr) {
  DML_TENSOR_DESC desc{};
  desc.Type = DML_TENSOR_TYPE_BUFFER;
  desc.Desc = buffer_desc_ptr;
  return desc;
}

// --- DML Binding Utilities ---
// Creates a DML_BUFFER_BINDING for a given D3D12 resource.
inline DML_BUFFER_BINDING create_buffer_binding(
    ID3D12Resource* resource,
    UINT64 offset = 0,
    UINT64 size_in_bytes = 0  // if 0, DML will deduce from tensor desc usually
) {
  return {resource, offset, size_in_bytes};
}

// Wraps a DML_BUFFER_BINDING in a DML_BINDING_DESC.
// The DML_BUFFER_BINDING pointed to by buffer_binding_ptr must remain valid.
inline DML_BINDING_DESC create_binding_desc(
    const DML_BUFFER_BINDING* buffer_binding_ptr) {
  return {DML_BINDING_TYPE_BUFFER, buffer_binding_ptr};
}

// --- DML Operator & Resource Creation Utilities ---

// Helper to create a constant tensor on the GPU using
// DML_OPERATOR_FILL_VALUE_CONSTANT. Manages the lifetime of the
// DML_BUFFER_TENSOR_DESC and DML_TENSOR_DESC. Returns the GPU resource and
// fills out_bundle with the descriptor bundle.
Microsoft::WRL::ComPtr<ID3D12Resource> CreateDmlConstantTensor(
    dml::Device* ct2_dml_device,           // ctranslate2 dml::Device wrapper
    const std::vector<UINT>& dims,         // Dimensions of the constant tensor
    DML_TENSOR_DATA_TYPE dml_tensor_type,  // Data type of the tensor
    DML_SCALAR_UNION scalar_value,         // Scalar value to fill
    DmlTensorDescBundle& out_bundle        // Output: Bundle containing descs
);

}  // namespace utils
}  // namespace dml
}  // namespace ctranslate2