#pragma once

#include "ctranslate2/storage_view.h"
#include "dml/dxdevice.h"
#include "resource_wrapper.h"

// Forward declaration for a helper function that might be used internally or
// defined later
namespace ctranslate2 {
std::string dtype_name(
    DataType type);  // Declaration from ctranslate2/types.h or similar

template <typename T>
struct type_to_dtype {};
template <>
struct type_to_dtype<float> {
  static constexpr DataType value = DataType::FLOAT32;
};
template <>
struct type_to_dtype<float16_t> {
  static constexpr DataType value = DataType::FLOAT16;
};
template <>
struct type_to_dtype<int32_t> {
  static constexpr DataType value = DataType::INT32;
};
template <>
struct type_to_dtype<int16_t> {
  static constexpr DataType value = DataType::INT16;
};
template <>
struct type_to_dtype<int8_t> {
  static constexpr DataType value = DataType::INT8;
};
template <>
struct type_to_dtype<uint8_t> {
  static constexpr DataType value = DataType::INT8;
};
template <>
struct type_to_dtype<bfloat16_t> {
  static constexpr DataType value = DataType::BFLOAT16;
};

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

  DmlTensorDescBundle(DML_TENSOR_DATA_TYPE dataType,
                      const std::vector<UINT>& dimensions,
                      const std::vector<UINT>& nonBroadcastDimensions,
                      int32_t coerceAxis,
                      int32_t placement,
                      int32_t leftAlignedDimensionCount,
                      uint32_t minDimensionCount,
                      uint32_t guaranteedBaseOffsetAlignment);
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

  DML_TENSOR_DATA_TYPE get_data_type() const {
    return buffer_desc_internal.DataType;
  }

  void set_data_type(DML_TENSOR_DATA_TYPE dml_dtype) {
    buffer_desc_internal.DataType = dml_dtype;
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

// Manages a DML_BUFFER_BINDING and provides a DML_BINDING_DESC for it.
// This helps ensure the DML_BUFFER_BINDING outlives the DML_BINDING_DESC
// that might be stored or used elsewhere temporarily.
class DmlBufferBindingBundle {
 public:
  // Default constructor for cases where a binding might be optional
  DmlBufferBindingBundle() : buffer_binding_{}, type_(DML_BINDING_TYPE_NONE) {}

  // Constructor for a valid buffer binding
  DmlBufferBindingBundle(
      ID3D12Resource* resource,
      UINT64 offset = 0,
      UINT64 size_in_bytes = 0  // if 0, we will deduce from tensor desc
  );

  // Returns a DML_BINDING_DESC.
  // The DML_BUFFER_BINDING is managed by this class instance.
  DML_BINDING_DESC get_desc() const {
    if (type_ == DML_BINDING_TYPE_BUFFER) {
      return {DML_BINDING_TYPE_BUFFER, &buffer_binding_};
    }
    return {DML_BINDING_TYPE_NONE, nullptr};
  }

  // Returns a const pointer to the internal DML_BUFFER_BINDING.
  // Useful if direct access to the DML_BUFFER_BINDING is needed,
  // but get_desc() is preferred for creating DML_BINDING_DESC.
  const DML_BUFFER_BINDING* get_buffer_binding_ptr() const {
    return (type_ == DML_BINDING_TYPE_BUFFER) ? &buffer_binding_ : nullptr;
  }

  // Returns the type of the binding.
  DML_BINDING_TYPE get_type() const { return type_; }

  // Allow move construction and assignment
  DmlBufferBindingBundle(DmlBufferBindingBundle&& other) noexcept
      : buffer_binding_(other.buffer_binding_), type_(other.type_) {
    // Reset the other to a safe state (optional, but good practice)
    other.type_ = DML_BINDING_TYPE_NONE;
    other.buffer_binding_ = {};
  }

  DmlBufferBindingBundle& operator=(DmlBufferBindingBundle&& other) noexcept {
    if (this != &other) {
      buffer_binding_ = other.buffer_binding_;
      type_ = other.type_;
      // Reset the other to a safe state
      other.type_ = DML_BINDING_TYPE_NONE;
      other.buffer_binding_ = {};
    }
    return *this;
  }

  DmlBufferBindingBundle(const DmlBufferBindingBundle& other) noexcept
      : buffer_binding_(other.buffer_binding_), type_(other.type_) {}

  DmlBufferBindingBundle& operator=(const DmlBufferBindingBundle& other) {
    if (this != &other) {
      buffer_binding_ = other.buffer_binding_;
      type_ = other.type_;
    }
    return *this;
  }

 private:
  DML_BUFFER_BINDING buffer_binding_;
  DML_BINDING_TYPE type_;  // To handle optional/empty bindings gracefully
};

// Manages a collection of DML_BUFFER_BINDING objects via DmlBufferBindingBundle
// and provides a way to get a vector of DML_BINDING_DESC.
// This is useful for input or output arrays in DML operator execution.
class DmlBindingArrayBundle {
 public:
  // Default constructor for an empty array of bindings
  DmlBindingArrayBundle() = default;

  // Constructor from a vector of D3D12 resources
  explicit DmlBindingArrayBundle(
      const std::vector<ID3D12Resource*>& resources) {
    buffer_binding_bundles_.reserve(resources.size());
    for (ID3D12Resource* resource : resources) {
      // Assuming default offset (0) and size (0, to be deduced by DML)
      // for each resource. If specific offsets/sizes are needed per resource,
      // a more complex input structure would be required for this constructor.
      buffer_binding_bundles_.emplace_back(resource);
    }
  }

  // Constructor from a vector of DmlBufferBindingBundle (e.g., if already
  // created)
  explicit DmlBindingArrayBundle(
      std::vector<DmlBufferBindingBundle>&& bundles) noexcept
      : buffer_binding_bundles_(std::move(bundles)) {}

  // Returns a vector of DML_BINDING_DESC.
  // The DML_BUFFER_BINDING structures are managed by the DmlBufferBindingBundle
  // instances within this class.
  std::vector<DML_BINDING_DESC> get_descs() const {
    std::vector<DML_BINDING_DESC> descs;
    descs.reserve(buffer_binding_bundles_.size());
    for (const auto& bundle : buffer_binding_bundles_) {
      descs.push_back(bundle.get_desc());
    }
    return descs;
  }

  // Returns the number of bindings in the array.
  size_t size() const { return buffer_binding_bundles_.size(); }

  // Checks if the array of bindings is empty.
  bool empty() const { return buffer_binding_bundles_.empty(); }

  // Allow move construction and assignment
  DmlBindingArrayBundle(DmlBindingArrayBundle&& other) noexcept = default;
  DmlBindingArrayBundle& operator=(DmlBindingArrayBundle&& other) noexcept =
      default;

  // Delete copy constructor and copy assignment operator.
  // Similar to DmlBufferBindingBundle, copying could lead to issues if not
  // handled with deep copies or shared ownership. Moving or creating new
  // array bundles is generally safer for DML binding patterns.
  DmlBindingArrayBundle(const DmlBindingArrayBundle&) = delete;
  DmlBindingArrayBundle& operator=(const DmlBindingArrayBundle&) = delete;

 private:
  std::vector<DmlBufferBindingBundle> buffer_binding_bundles_;
};

// A RAII helper to temporarily reshape a StorageView and restore it on scope
// exit.
class ScopedReshape {
 public:
  // Reshapes `view` to `new_shape` and stores the original shape.
  ScopedReshape(StorageView& view, Shape new_shape)
      : _view(view), _original_shape(view.shape()) {
    _view.reshape(std::move(new_shape));
  }

  // Restores the original shape.
  ~ScopedReshape() { _view.reshape(std::move(_original_shape)); }

  // Disallow copy and move operations to prevent misuse.
  ScopedReshape(const ScopedReshape&) = delete;
  ScopedReshape& operator=(const ScopedReshape&) = delete;
  ScopedReshape(ScopedReshape&&) = delete;
  ScopedReshape& operator=(ScopedReshape&&) = delete;

 private:
  StorageView& _view;
  Shape _original_shape;
};

// --- DML Operator & Resource Creation Utilities ---

// Helper to create a constant tensor on the GPU using
// DML_OPERATOR_FILL_VALUE_CONSTANT. Manages the lifetime of the
// DML_BUFFER_TENSOR_DESC and DML_TENSOR_DESC. Returns the GPU resource and
// fills out_bundle with the descriptor bundle.
Microsoft::WRL::ComPtr<IResourceWrapper> CreateDmlConstantTensor(
    dml::Device* ct2_dml_device,    // ctranslate2 dml::Device wrapper
    DML_SCALAR_UNION scalar_value,  // Scalar value to fill
    const DmlTensorDescBundle& out_bundle);

inline ID3D12Resource* ResourceFromStorageView(
    const StorageView& storage_view) {
  return static_cast<IResourceWrapper*>(
             const_cast<void*>(storage_view.buffer()))
      ->GetD3D12Resource();
}

template <typename T>
inline ID3D12Resource* ResourceFromRawBuffer(const T* buffer) {
  return reinterpret_cast<IResourceWrapper*>(const_cast<T*>(buffer))
      ->GetD3D12Resource();
}

template <typename T>
inline T* ResourceToBuffer(ID3D12Resource* resource) {
  return reinterpret_cast<T*>(resource);
}

template <typename T>
inline IResourceWrapper* ResourceWrapperFromRawBuffer(const T* buffer) {
  return reinterpret_cast<IResourceWrapper*>(const_cast<T*>(buffer));
}
}  // namespace utils
}  // namespace dml
}  // namespace ctranslate2