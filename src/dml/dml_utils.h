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

// Manages DML_BUFFER_TENSOR_DESC and DML_TENSOR_DESC, including internal
// storage for sizes/strides vectors. This is useful for situations where tensor
// metadata needs to be managed throughout DML operations.
class DmlTensorDescBundle {
 public:
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

  // @brief Constructor for creating a DmlTensorDescBundle with advanced layout
  // control, including broadcasting, dimension coercion, and alignment.
  // @param dataType The DML data type of the tensor.
  // @param dimensions The target dimensions of the tensor, which can include
  // broadcasted dimensions.
  // @param nonBroadcastDimensions The physical dimensions of the source tensor.
  // Used to calculate strides for broadcasting.
  // @param coerceAxis If set to a value less than the rank of `dimensions`, the
  // tensor shape is flattened into a 2D tensor.
  //        All dimensions up to `coerceAxis` (exclusive) are collapsed into the
  //        first dimension, and the rest are collapsed into the second. Set to
  //        >= rank to disable.
  // @param placement Controls padding with '1's. Positive values add leading
  // '1's, negative values add trailing '1's.
  // @param leftAlignedDimensionCount Specifies how many dimensions from the
  // start of the `dimensions` array are treated as left-aligned. The remaining
  // are right-aligned. This is crucial for broadcasting non-contiguous tensors.
  // @param minDimensionCount The minimum rank of the output tensor description.
  // @param guaranteedBaseOffsetAlignment Memory alignment hint for DML.
  DmlTensorDescBundle(DML_TENSOR_DATA_TYPE dataType,
                      const std::vector<UINT>& dimensions,
                      const std::vector<UINT>& nonBroadcastDimensions,
                      int32_t coerceAxis,
                      int32_t placement,
                      int32_t leftAlignedDimensionCount,
                      uint32_t minDimensionCount,
                      uint32_t guaranteedBaseOffsetAlignment);

  DmlTensorDescBundle(const DmlTensorDescBundle& other) {
    internal_sizes_vec = other.internal_sizes_vec;
    internal_strides_vec = other.internal_strides_vec;
    buffer_desc_internal = other.buffer_desc_internal;
    tensor_desc_internal = other.tensor_desc_internal;

    buffer_desc_internal.Sizes = internal_sizes_vec.data();
    buffer_desc_internal.Strides =
        internal_strides_vec.empty() ? nullptr : internal_strides_vec.data();
    tensor_desc_internal.Desc = &buffer_desc_internal;
  }

  DmlTensorDescBundle& operator=(const DmlTensorDescBundle& other) {
    if (this != &other) {
      internal_sizes_vec = other.internal_sizes_vec;
      internal_strides_vec = other.internal_strides_vec;
      buffer_desc_internal = other.buffer_desc_internal;
      tensor_desc_internal = other.tensor_desc_internal;

      buffer_desc_internal.Sizes = internal_sizes_vec.data();
      buffer_desc_internal.Strides =
          internal_strides_vec.empty() ? nullptr : internal_strides_vec.data();
      tensor_desc_internal.Desc = &buffer_desc_internal;
    }
    return *this;
  }

  static DmlTensorDescBundle broadcastFromSeach(
      const StorageView& storage,
      const std::vector<UINT>& target_dims);

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
  friend class DmlOperatorDescBundle;
};

// A helper class to construct a `DML_OPERATOR_DESC` by managing the lifetimes
// of its constituent parts.
//
// DML operator descriptions often point to other structures (like tensor
// descriptions), and this class ensures those structures remain valid while the
// operator description is being assembled and used. It is move-only to prevent
// accidental copies which could invalidate internal pointers.
class DmlOperatorDescBundle {
 public:
  DmlOperatorDescBundle() = default;
  ~DmlOperatorDescBundle() = default;

  DmlOperatorDescBundle(const DmlOperatorDescBundle&) = delete;
  DmlOperatorDescBundle& operator=(const DmlOperatorDescBundle&) = delete;

  DmlOperatorDescBundle(DmlOperatorDescBundle&&) = default;
  DmlOperatorDescBundle& operator=(DmlOperatorDescBundle&&) = default;

  // @brief Adds an input tensor description to the operator.
  // @details Arguments are forwarded to the `DmlTensorDescBundle` constructor.
  // @return A reference to the newly created `DmlTensorDescBundle`.
  template <typename... Args>
  DmlTensorDescBundle& AddInput(Args&&... args) {
    std::unique_ptr<DmlTensorDescBundle> input_desc(
        new DmlTensorDescBundle(std::forward<Args>(args)...));
    input_descs_.emplace_back(std::move(input_desc));
    return *input_descs_.back();
  }

  /**
   * @brief Adds an input tensor descriptor with broadcasting from search
   * dimensions.
   *
   * This method creates a new DML tensor descriptor by broadcasting the given
   * storage view to match the specified target dimensions, then adds it to the
   * input descriptors collection. The broadcasting is performed using a
   * search-based algorithm to determine the optimal dimension mapping.
   *
   * @param storage The source storage view containing the tensor data to be
   * broadcasted
   * @param target_dims Vector of target dimensions (UINT) that the tensor
   * should be broadcasted to
   * @return Reference to the newly added DmlTensorDescBundle for method
   * chaining
   *
   * @note The method adds the descriptor to the internal input_descs_
   * collection
   * @note Returns a reference to the last added descriptor bundle
   */
  DmlTensorDescBundle& AddInputBroadcastFromSeach(
      const StorageView& storage,
      const std::vector<UINT>& target_dims) {
    std::unique_ptr<DmlTensorDescBundle> input_desc(new DmlTensorDescBundle(
        (DmlTensorDescBundle::broadcastFromSeach(storage, target_dims))));
    input_descs_.emplace_back(std::move(input_desc));
    return *input_descs_.back();
  }

  // @brief Adds an output tensor description to the operator.
  // @details Arguments are forwarded to the `DmlTensorDescBundle` constructor.
  // @return A reference to the newly created `DmlTensorDescBundle`.
  template <typename... Args>
  DmlTensorDescBundle& AddOutput(Args&&... args) {
    std::unique_ptr<DmlTensorDescBundle> output_desc(
        new DmlTensorDescBundle(std::forward<Args>(args)...));
    output_descs_.emplace_back(std::move(output_desc));
    return *output_descs_.back();
  }

  // @brief Gets and configures the underlying DML operator-specific description
  // struct.
  // @details This function also sets the `DML_OPERATOR_DESC`'s type field based
  // on the template parameter `T`.
  // @tparam T The DML operator-specific description type (e.g.,
  // `DML_GEMM_OPERATOR_DESC`).
  // @return A reference to the resized and prepared description struct.
  template <typename T>
  T& GetOperatorDesc() {
    size_t size = sizeof(T);
    operator_desc_storage_.resize(size);
    T& desc = *reinterpret_cast<T*>(operator_desc_storage_.data());
    DML_OPERATOR_TYPE type = GetOperatorType<T>();
    operator_desc_ = {type, &desc};
    return desc;
  }

  /**
   * @brief Creates and returns a reference to a fused operator descriptor of
   * the specified type.
   *
   * This template function allocates storage for a DirectML operator descriptor
   * of type T, constructs it in-place, and sets up the fused operator
   * descriptor structure with the appropriate operator type and pointer to the
   * descriptor data.
   *
   * @tparam T The type of the DirectML operator descriptor to create
   * @return T& Reference to the constructed operator descriptor
   *
   * @note The returned reference is valid as long as this object exists and no
   * subsequent calls to GetFusedOperatorDesc() are made, as they would
   * invalidate the storage.
   * @note The storage is managed internally and will be resized to accommodate
   * the descriptor.
   */
  template <typename T>
  T& GetFusedOperatorDesc() {
    size_t size = sizeof(T);
    fused_operator_desc_storage_.resize(size);
    T& desc = *reinterpret_cast<T*>(fused_operator_desc_storage_.data());
    DML_OPERATOR_TYPE type = GetOperatorType<T>();
    fused_operator_desc_ = {type, &desc};
    return desc;
  }

  const DML_OPERATOR_DESC& get_desc() const { return operator_desc_; }

  const DML_OPERATOR_DESC& get_fused_desc() const {
    return fused_operator_desc_;
  }

 private:
  template <typename T>
  DML_OPERATOR_TYPE GetOperatorType() {
    DML_OPERATOR_TYPE type;
    if constexpr (std::is_same_v<T, DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
    } else if constexpr (std::is_same_v<T, DML_GEMM_OPERATOR_DESC>) {
      type = DML_OPERATOR_GEMM;
    } else if constexpr (std::is_same_v<T, DML_REDUCE_OPERATOR_DESC>) {
      type = DML_OPERATOR_REDUCE;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_ADD_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_ADD;
    } else if constexpr (std::is_same_v<
                             T, DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>) {
      type = DML_OPERATOR_FILL_VALUE_CONSTANT;
    } else if constexpr (std::is_same_v<T, DML_CAST_OPERATOR_DESC>) {
      type = DML_OPERATOR_CAST;
    } else if constexpr (std::is_same_v<T, DML_ACTIVATION_RELU_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_RELU;
    } else if constexpr (std::is_same_v<T,
                                        DML_ACTIVATION_SIGMOID_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_SIGMOID;
    } else if constexpr (std::is_same_v<T, DML_ACTIVATION_TANH_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_TANH;
    } else if constexpr (std::is_same_v<
                             T, DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_SUBTRACT;
    } else if constexpr (std::is_same_v<
                             T, DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_MAX_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_MAX;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_MIN_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_MIN;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_EXP_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_EXP;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_LOG_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_LOG;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_SIN_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_SIN;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_COS_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_COS;
    } else if constexpr (std::is_same_v<T, DML_ARGMAX_OPERATOR_DESC>) {
      type = DML_OPERATOR_ARGMAX;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_ADD1_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_ADD1;
    } else if constexpr (std::is_same_v<
                             T, DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_DIVIDE;
    } else if constexpr (std::is_same_v<
                             T, DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_NEGATE;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_ABS_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_ABS;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_ROUND_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_ROUND;
    } else if constexpr (std::is_same_v<T, DML_CONVOLUTION_OPERATOR_DESC>) {
      type = DML_OPERATOR_CONVOLUTION;
    } else if constexpr (std::is_same_v<
                             T,
                             DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC>) {
      type = DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION;
    } else if constexpr (std::is_same_v<
                             T, DML_CONVOLUTION_INTEGER_OPERATOR_DESC>) {
      type = DML_OPERATOR_CONVOLUTION_INTEGER;
    } else if constexpr (std::is_same_v<
                             T, DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC>) {
      type = DML_OPERATOR_MATRIX_MULTIPLY_INTEGER;
    } else if constexpr (std::is_same_v<
                             T,
                             DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC>) {
      type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2;
    } else if constexpr (std::is_same_v<T, DML_JOIN_OPERATOR_DESC>) {
      type = DML_OPERATOR_JOIN;
    } else if constexpr (std::is_same_v<T, DML_SPLIT_OPERATOR_DESC>) {
      type = DML_OPERATOR_SPLIT;
    } else if constexpr (std::is_same_v<T, DML_SLICE_OPERATOR_DESC>) {
      type = DML_OPERATOR_SLICE;
    } else if constexpr (std::is_same_v<T, DML_SLICE1_OPERATOR_DESC>) {
      type = DML_OPERATOR_SLICE1;
    } else if constexpr (std::is_same_v<T, DML_TILE_OPERATOR_DESC>) {
      type = DML_OPERATOR_TILE;
    } else if constexpr (std::is_same_v<T, DML_GATHER_OPERATOR_DESC>) {
      type = DML_OPERATOR_GATHER;
    } else if constexpr (std::is_same_v<T, DML_GATHER_ELEMENTS_OPERATOR_DESC>) {
      type = DML_OPERATOR_GATHER_ELEMENTS;
    } else if constexpr (std::is_same_v<T,
                                        DML_SCATTER_ELEMENTS_OPERATOR_DESC>) {
      type = DML_OPERATOR_SCATTER_ELEMENTS;
    } else if constexpr (std::is_same_v<T, DML_TOP_K1_OPERATOR_DESC>) {
      type = DML_OPERATOR_TOP_K1;
    } else if constexpr (std::is_same_v<T, DML_ACTIVATION_GELU_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_GELU;
    } else if constexpr (std::is_same_v<T,
                                        DML_ACTIVATION_SWISH_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_SWISH;
    } else if constexpr (std::is_same_v<T,
                                        DML_ACTIVATION_LINEAR_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_LINEAR;
    } else if constexpr (std::is_same_v<
                             T, DML_ACTIVATION_IDENTITY_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_IDENTITY;
    } else if constexpr (std::is_same_v<T,
                                        DML_ACTIVATION_SOFTMAX_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_SOFTMAX;
    } else if constexpr (std::is_same_v<
                             T, DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
    } else if constexpr (std::is_same_v<
                             T, DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
    } else if constexpr (std::is_same_v<
                             T, DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC>) {
      type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
    } else if constexpr (std::is_same_v<T,
                                        DML_RANDOM_GENERATOR_OPERATOR_DESC>) {
      type = DML_OPERATOR_RANDOM_GENERATOR;
    } else if constexpr (std::is_same_v<
                             T, DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC>) {
      type = DML_OPERATOR_FILL_VALUE_SEQUENCE;
    } else if constexpr (
        std::is_same_v<
            T, DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL;
    } else if constexpr (
        std::is_same_v<T, DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN;
    } else if constexpr (std::is_same_v<
                             T,
                             DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS;
    } else if constexpr (std::is_same_v<T, DML_ELEMENT_WISE_IF_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_IF;
    } else if constexpr (std::is_same_v<
                             T, DML_CUMULATIVE_SUMMATION_OPERATOR_DESC>) {
      type = DML_OPERATOR_CUMULATIVE_SUMMATION;
    } else if constexpr (std::is_same_v<
                             T,
                             DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR;
    } else if constexpr (
        std::is_same_v<T, DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR;
    } else if constexpr (std::is_same_v<
                             T, DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC>) {
      type = DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_RECIP;
    } else if constexpr (std::is_same_v<T,
                                        DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>) {
      type = DML_OPERATOR_ELEMENT_WISE_SQRT;
    } else {
      static_assert(false,
                    "Unsupported operator type for DmlOperatorDescBundle");
    }
    return type;
  }

  // Type-erased storage for the operator-specific description struct (e.g.,
  // DML_GEMM_OPERATOR_DESC).
  std::vector<std::byte> operator_desc_storage_;
  std::vector<std::byte> fused_operator_desc_storage_;
  // Owns the input tensor descriptions.
  std::vector<std::unique_ptr<DmlTensorDescBundle>> input_descs_;
  // Owns the output tensor descriptions.
  std::vector<std::unique_ptr<DmlTensorDescBundle>> output_descs_;
  // The top-level DML operator description. `Desc` will point to data in
  // `operator_desc_storage_`.
  DML_OPERATOR_DESC operator_desc_{};
  /**
   * @brief Descriptor for a fused DML (DirectML) operator.
   *
   * This member holds the configuration and parameters for a DirectML operator
   * that combines multiple operations into a single fused operation for
   * improved performance. The descriptor defines the operator's input/output
   * tensors, attributes, and execution behavior within the DirectML compute
   * graph.
   */
  DML_OPERATOR_DESC fused_operator_desc_{};
};

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

  // Constructor from initializer list that accepts tuples with arguments for
  // DmlBufferBindingBundle. This constructor perfectly forwards arguments to
  // DmlBufferBindingBundle's constructor, avoiding copies.
  DmlBindingArrayBundle(
      std::initializer_list<std::tuple<ID3D12Resource*, UINT64, UINT64>>
          init_list) {
    buffer_binding_bundles_.reserve(init_list.size());
    for (const auto& args_tuple : init_list) {
      std::apply(
          [this](auto&&... args) {
            buffer_binding_bundles_.emplace_back(
                std::forward<decltype(args)>(args)...);
          },
          args_tuple);
    }
  }

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