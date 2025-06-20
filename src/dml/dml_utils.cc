#include "dml_utils.h"

#include "backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "dml/resource_wrapper.h"

namespace ctranslate2 {
namespace dml {
namespace utils {
namespace {
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

inline UINT64 DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE dataType,
                                      UINT dimensionCount,
                                      _In_reads_(dimensionCount)
                                          const UINT* sizes,
                                      _In_reads_opt_(dimensionCount)
                                          const UINT* strides) {
  UINT elementSizeInBits = 0;
  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      elementSizeInBits = 32;
      break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      elementSizeInBits = 16;
      break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
      elementSizeInBits = 8;
      break;

#if DML_TARGET_VERSION >= 0x6300
    case DML_TENSOR_DATA_TYPE_UINT4:
    case DML_TENSOR_DATA_TYPE_INT4:
      elementSizeInBits = 4;
      break;
#endif

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
      elementSizeInBits = 64;
      break;

    default:
      return 0;  // Invalid data type
  }

  UINT64 minimumImpliedSizeInBits = 0;
  if (!strides) {
    minimumImpliedSizeInBits = sizes[0];
    for (UINT i = 1; i < dimensionCount; ++i) {
      minimumImpliedSizeInBits *= sizes[i];
    }
    minimumImpliedSizeInBits *= elementSizeInBits;
  } else {
    UINT indexOfLastElement = 0;
    for (UINT i = 0; i < dimensionCount; ++i) {
      indexOfLastElement += (sizes[i] - 1) * strides[i];
    }

    minimumImpliedSizeInBits =
        (static_cast<UINT64>(indexOfLastElement) + 1) * elementSizeInBits;
  }

  UINT64 minimumImpliedSizeInBytes = (minimumImpliedSizeInBits + 7) / 8;

  // Round up to the nearest 4 bytes.
  minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ull;

  return minimumImpliedSizeInBytes;
}
}  // namespace

// --- Data Type & Size Utilities ---

DML_TENSOR_DATA_TYPE get_dml_data_type(DataType ct2_type) {
  switch (ct2_type) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DataType::INT16:
      return DML_TENSOR_DATA_TYPE_INT16;
    case DataType::INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
      // Removed UINT types as they are not in ctranslate2::DataType
#ifdef CT2_WITH_BFLOAT16
    // DirectML typically handles BFLOAT16 by casting to/from FLOAT32.
    // For tensor description, FLOAT16 is sometimes used as a proxy if ops
    // support it, or UINT16 if just storing bits. Using FLOAT16 here for
    // simplicity, actual computation path should handle casts if needed.
    case DataType::BFLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
#endif
    default:
      THROW_INVALID_ARGUMENT(
          "Unsupported ctranslate2::DataType for DirectML: " +
          dtype_name(ct2_type));
  }
}

size_t get_dml_element_size_in_bytes(DML_TENSOR_DATA_TYPE dml_type) {
  switch (dml_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_INT64:
      return 8;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      return 2;
    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
      return 1;
    default:
      THROW_INVALID_ARGUMENT(
          "Unsupported DML_TENSOR_DATA_TYPE for size calculation: " +
          std::to_string(dml_type));
  }
  return 0;  // Should not reach
}

// --- Shape & Dimension Utilities ---

std::vector<UINT> to_dml_dims(const Shape& shape,
                              size_t storage_size,
                              bool ensure_at_least_1d_for_dml) {
  std::vector<UINT> dml_dims;
  dml_dims.reserve(shape.size());
  for (dim_t d : shape) {
    if (d < 0)
      THROW_INVALID_ARGUMENT("Tensor dimension must be non-negative, but got " +
                             std::to_string(d));
    dml_dims.push_back(static_cast<UINT>(d));
  }

  if (ensure_at_least_1d_for_dml && dml_dims.empty()) {
    if (storage_size == 1) {  // Scalar
      dml_dims.push_back(1);
    } else if (storage_size == 0) {  // Truly empty tensor
      dml_dims.push_back(0);
    } else {
      // This case (empty shape but size > 1 or size unknown & non-zero) is
      // ambiguous for DML. Defaulting to a 1D tensor of that size.
      dml_dims.push_back(static_cast<UINT>(storage_size));
    }
  }
  return dml_dims;
}

// --- DmlTensorDescBundle Implementation ---
DmlTensorDescBundle::DmlTensorDescBundle(
    const StorageView& storage,
    const std::vector<UINT>* strides_override) {
  init_from_storage(storage, strides_override);
}

DmlTensorDescBundle::DmlTensorDescBundle(DML_TENSOR_DATA_TYPE dml_data_type,
                                         const std::vector<UINT>& sizes,
                                         const std::vector<UINT>* strides,
                                         UINT64 total_tensor_size_in_bytes) {
  init_from_details(dml_data_type, sizes, strides, total_tensor_size_in_bytes);
}

DmlTensorDescBundle::DmlTensorDescBundle(DataType ct2_data_type,
                                         const std::vector<UINT>& sizes,
                                         const std::vector<UINT>* strides,
                                         UINT64 total_tensor_size_in_bytes) {
  init_from_details(get_dml_data_type(ct2_data_type), sizes, strides,
                    total_tensor_size_in_bytes);
}

void DmlTensorDescBundle::init_from_storage(
    const StorageView& storage,
    const std::vector<UINT>* strides_override) {
  internal_sizes_vec = to_dml_dims(storage.shape(), storage.size());
  DML_TENSOR_DATA_TYPE dml_dtype = get_dml_data_type(storage.dtype());

  buffer_desc_internal.DataType = dml_dtype;
  buffer_desc_internal.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc_internal.DimensionCount =
      static_cast<UINT>(internal_sizes_vec.size());
  buffer_desc_internal.Sizes = internal_sizes_vec.data();

  UINT64 total_bytes = DMLCalcBufferTensorSize(
      dml_dtype, internal_sizes_vec.size(), internal_sizes_vec.data(),
      strides_override ? strides_override->data() : nullptr);
  if (strides_override) {
    internal_strides_vec = *strides_override;
  }

  buffer_desc_internal.TotalTensorSizeInBytes =
      total_bytes;  // Use the possibly refined total_bytes
  buffer_desc_internal.Strides =
      internal_strides_vec.empty() ? nullptr : internal_strides_vec.data();
  buffer_desc_internal.GuaranteedBaseOffsetAlignment = 0;

  tensor_desc_internal.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc_internal.Desc = &buffer_desc_internal;
}

void DmlTensorDescBundle::init_from_details(DML_TENSOR_DATA_TYPE dml_dtype,
                                            const std::vector<UINT>& sizes_in,
                                            const std::vector<UINT>* strides_in,
                                            UINT64 total_tensor_size_bytes_in) {
  internal_sizes_vec = sizes_in;
  if (internal_sizes_vec
          .empty()) {  // Ensure at least 1D for DML consistency if representing
                       // a scalar or empty from size 0
    size_t num_elements_from_sizes = 0;  // Calculate this correctly below
    if (!internal_sizes_vec.empty()) {
      num_elements_from_sizes = 1;
      for (UINT s : internal_sizes_vec)
        num_elements_from_sizes *= s;
    }

    if (num_elements_from_sizes == 0 && total_tensor_size_bytes_in == 0)
      internal_sizes_vec.push_back(0);
    else
      internal_sizes_vec.push_back(1);
  }

  buffer_desc_internal.DataType = dml_dtype;
  buffer_desc_internal.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc_internal.DimensionCount =
      static_cast<UINT>(internal_sizes_vec.size());
  buffer_desc_internal.Sizes = internal_sizes_vec.data();

  UINT64 final_total_bytes = total_tensor_size_bytes_in;
  final_total_bytes = DMLCalcBufferTensorSize(
      dml_dtype, internal_sizes_vec.size(), internal_sizes_vec.data(),
      strides_in ? strides_in->data() : nullptr);
  if (strides_in) {
    internal_strides_vec = *strides_in;
  }

  buffer_desc_internal.TotalTensorSizeInBytes = final_total_bytes;
  buffer_desc_internal.Strides =
      internal_strides_vec.empty() ? nullptr : internal_strides_vec.data();
  buffer_desc_internal.GuaranteedBaseOffsetAlignment = 0;

  tensor_desc_internal.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc_internal.Desc = &buffer_desc_internal;
}

DmlTensorDescBundle::DmlTensorDescBundle(
    DML_TENSOR_DATA_TYPE dataType,
    const std::vector<UINT>& dimensions,
    const std::vector<UINT>& nonBroadcastDimensions,
    int32_t coerceAxis,
    int32_t placement,
    int32_t leftAlignedDimensionCount,
    uint32_t minDimensionCount,
    uint32_t guaranteedBaseOffsetAlignment) {
  tensor_desc_internal.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc_internal.Desc = &buffer_desc_internal;
  buffer_desc_internal.DataType = dataType;
  if (coerceAxis < 0)
    THROW_INVALID_ARGUMENT("coerceAxis must be non-negative");

  const std::vector<UINT>* sizes_ptr = &dimensions;
  std::vector<UINT> coercedSizes;
  if (dimensions.size() > 1 &&
      coerceAxis < static_cast<int32_t>(dimensions.size())) {
    UINT dimension0 = 1u;
    UINT dimension1 = dimensions[coerceAxis];

    for (int32_t i = 0; i < coerceAxis; ++i) {
      dimension0 *= dimensions[i];
    }

    for (size_t i = static_cast<int64_t>(coerceAxis) + 1,
                ci = dimensions.size();
         i < ci; ++i) {
      dimension1 *= dimensions[i];
    }

    coercedSizes.push_back(dimension0);
    coercedSizes.push_back(dimension1);
    sizes_ptr = &coercedSizes;
  }

  const auto& sizes = *sizes_ptr;
  const int32_t rank = static_cast<int32_t>(sizes.size());
  leftAlignedDimensionCount =
      leftAlignedDimensionCount < 0
          ? std::max(0, leftAlignedDimensionCount + rank)
          : std::min(rank, leftAlignedDimensionCount);

  buffer_desc_internal.DimensionCount =
      std::max(rank, (int32_t)minDimensionCount);
  if (buffer_desc_internal.DimensionCount > DML_TENSOR_DIMENSION_COUNT_MAX1)
    THROW_INVALID_ARGUMENT(
        "DmlTensorDescBundle dimension count cannot exceed " +
        std::to_string(DML_TENSOR_DIMENSION_COUNT_MAX1));

  internal_sizes_vec.assign(DML_TENSOR_DIMENSION_COUNT_MAX1, 1);

  {
    const int32_t totalFillerCount = buffer_desc_internal.DimensionCount - rank;
    const int32_t leadingFillerCount =
        std::clamp(placement, 0, totalFillerCount);
    const int32_t remainingFillerCount = totalFillerCount - leadingFillerCount;
    const int32_t trailingFillerCount =
        std::clamp(-placement, 0, remainingFillerCount);
    const int32_t middleFillerCount =
        remainingFillerCount - trailingFillerCount;
    const int32_t firstRightAlignedDim =
        leadingFillerCount + leftAlignedDimensionCount + middleFillerCount;

    int i = 0, j = 0;
    while (j < leadingFillerCount) {
      internal_sizes_vec[j++] = 1;
    }
    while (i < leftAlignedDimensionCount) {
      internal_sizes_vec[j++] = sizes[i++];
    }
    while (j < firstRightAlignedDim) {
      internal_sizes_vec[j++] = 1;
    }
    while (i < rank) {
      internal_sizes_vec[j++] = sizes[i++];
    }
  }

  internal_sizes_vec.resize(buffer_desc_internal.DimensionCount);

  bool useStrides = false;
  internal_strides_vec.clear();

  if (dimensions != nonBroadcastDimensions) {
    if (!std::equal(dimensions.begin(), dimensions.end(),
                    &internal_sizes_vec[buffer_desc_internal.DimensionCount -
                                        dimensions.size()])) {
      THROW_INVALID_ARGUMENT(
          "Broadcasting is only supported for contiguously right-aligned "
          "dimensions");
    }

    useStrides = true;
    internal_strides_vec.resize(buffer_desc_internal.DimensionCount);

    auto nonBroadcastDimsIter = nonBroadcastDimensions.rbegin();
    uint32_t elementCount = 1;
    for (int descDimIndex = buffer_desc_internal.DimensionCount - 1;
         descDimIndex >= 0; --descDimIndex) {
      if (nonBroadcastDimsIter == nonBroadcastDimensions.rend() ||
          (*nonBroadcastDimsIter == 1)) {
        internal_strides_vec[descDimIndex] = 0;
      } else {
        internal_strides_vec[descDimIndex] = elementCount;
        elementCount *= (*nonBroadcastDimsIter);
      }
      if (nonBroadcastDimsIter != nonBroadcastDimensions.rend()) {
        ++nonBroadcastDimsIter;
      }
    }
  }

  buffer_desc_internal.Sizes = internal_sizes_vec.data();
  buffer_desc_internal.Strides =
      useStrides ? internal_strides_vec.data() : nullptr;
  buffer_desc_internal.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc_internal.GuaranteedBaseOffsetAlignment =
      guaranteedBaseOffsetAlignment;

  buffer_desc_internal.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
      buffer_desc_internal.DataType, buffer_desc_internal.DimensionCount,
      internal_sizes_vec.data(),
      useStrides ? internal_strides_vec.data() : nullptr);

  size_t non_bcast_bytes =
      get_dml_element_size_in_bytes(buffer_desc_internal.DataType);
  if (!nonBroadcastDimensions.empty()) {
    for (auto dim : nonBroadcastDimensions)
      non_bcast_bytes *= dim;
  } else {
    non_bcast_bytes = 0;
  }

  if (buffer_desc_internal.TotalTensorSizeInBytes < non_bcast_bytes) {
    THROW_INVALID_ARGUMENT(
        "TotalTensorSizeInBytes is smaller than what is required by the "
        "physical dimensions of the tensor");
  }
}

DmlTensorDescBundle DmlTensorDescBundle::broadcastFromSeach(
    const StorageView& tensor,
    const std::vector<UINT>& target_dims) {
  auto physical_shape =
      dml::utils::to_dml_dims(tensor.shape(), tensor.size(), false);
  if (physical_shape.empty()) {
    physical_shape.push_back(1);
  }
  if (physical_shape.size() < target_dims.size()) {
    auto it = std::search(target_dims.begin(), target_dims.end(),
                          physical_shape.begin(), physical_shape.end());
    if (it != target_dims.end()) {
      size_t index = std::distance(target_dims.begin(), it);
      std::vector<UINT> new_shape(index, 1);
      new_shape.insert(new_shape.end(), physical_shape.begin(),
                       physical_shape.end());
      new_shape.resize(target_dims.size(), 1);
      physical_shape = new_shape;
    } else {
      while (physical_shape.size() < target_dims.size()) {
        physical_shape.push_back(1);
      }
    }
  }
  return dml::utils::DmlTensorDescBundle(
      dml::utils::get_dml_data_type(tensor.dtype()), target_dims,
      physical_shape, static_cast<int32_t>(target_dims.size()), 0, 0, 0, 0);
}

DmlBufferBindingBundle::DmlBufferBindingBundle(IResourceWrapper* resource,
                                               UINT64 offset,
                                               UINT64 size_in_bytes)
    : resource_wrapper_(resource)

{
  if (resource == nullptr) {
    type_ = DML_BINDING_TYPE_NONE;
  } else {
    type_ = DML_BINDING_TYPE_BUFFER;
    buffer_binding_ = {nullptr, offset, size_in_bytes};

#if 0
    D3D12_RESOURCE_DESC desc = d3d12_resource->GetDesc();
    if (desc.Format != DXGI_FORMAT_UNKNOWN) {
      THROW_INVALID_ARGUMENT(
          "DML buffer binding size cannot be 0 for non-buffer resources.");
    }
    UINT64 size_in_byte = desc.Width - offset;
#else
    UINT64 size_in_byte = resource->GetActualSize() - offset;
#endif

    buffer_binding_.SizeInBytes = size_in_byte;
  }
}

}  // namespace utils
}  // namespace dml
}  // namespace ctranslate2