#include "dml_utils.h"

#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace dml {
namespace utils {

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

std::vector<UINT> get_dml_tensor_shape_4d(const StorageView& tensor,
                                          bool is_filter_or_bias) {
  const auto& sv_shape = tensor.shape();
  std::vector<UINT> dml_dims;

  if (is_filter_or_bias) {
    if (tensor.rank() == 3) {  // Filter/Weight tensor (OutChannels,
                               // InChannelsPerGroup, KernelWidth)
      dml_dims = {static_cast<UINT>(sv_shape[0]),
                  static_cast<UINT>(sv_shape[1]), 1,
                  static_cast<UINT>(sv_shape[2])};
    } else if (tensor.rank() == 1) {  // Bias tensor (OutChannels)
      dml_dims = {1, static_cast<UINT>(sv_shape[0]), 1, 1};
    } else {
      THROW_INVALID_ARGUMENT("Conv1D DML: Unsupported rank for filter (" +
                             std::to_string(tensor.rank()) +
                             "). Expected rank 3 for filter, 1 for bias.");
    }
  } else {                     // Input or Output tensor
    if (tensor.rank() == 3) {  // (Batch, Channels, Width)
      dml_dims = {static_cast<UINT>(sv_shape[0]),
                  static_cast<UINT>(sv_shape[1]), 1,
                  static_cast<UINT>(sv_shape[2])};
    } else {
      THROW_INVALID_ARGUMENT(
          "Conv1D DML: Unsupported rank for input/output tensor (" +
          std::to_string(tensor.rank()) + "). Expected rank 3.");
    }
  }
  return dml_dims;
}

// --- DML Tensor Descriptor Utilities ---

DML_TENSOR_DESC make_tensor_desc_from_storage(
    const StorageView& storage,
    std::vector<UINT>& dml_dims_buffer,
    DML_BUFFER_TENSOR_DESC& buffer_desc,
    std::vector<UINT>* dml_strides_buffer) {
  dml_dims_buffer = to_dml_dims(storage.shape(), storage.size());

  buffer_desc.DataType = get_dml_data_type(storage.dtype());
  buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc.DimensionCount = static_cast<UINT>(dml_dims_buffer.size());
  buffer_desc.Sizes = dml_dims_buffer.data();

  if (dml_strides_buffer) {
    dml_strides_buffer->clear();
    if (buffer_desc.DimensionCount > 0) {
      dml_strides_buffer->resize(buffer_desc.DimensionCount);
      // Assuming standard contiguous strides (row-major) if not overridden by
      // caller implicitly or explicitly. If storage view were non-contiguous
      // and dml_strides_buffer was meant to receive them, this would need
      // StorageView to expose stride data. For this generic helper, we compute
      // contiguous strides here. If the caller intended specific strides, they
      // should pre-populate dml_strides_buffer or use DmlTensorDescBundle.
      (*dml_strides_buffer)[buffer_desc.DimensionCount - 1] = 1;
      for (int i = static_cast<int>(buffer_desc.DimensionCount) - 2; i >= 0;
           --i) {
        UINT next_dim_size = dml_dims_buffer[i + 1];
        // If next_dim_size is 0, product is 0, but DML strides are usually > 0
        // if overall tensor not empty. The `TotalTensorSizeInBytes` calculation
        // will handle the 0-element case correctly. Stride for a dim before a
        // 0-sized dim can be thought of as 'size of the 0-sized slice',
        // effectively.
        (*dml_strides_buffer)[i] = (*dml_strides_buffer)[i + 1] *
                                   (next_dim_size == 0 ? 1 : next_dim_size);
      }

      bool has_zero_dim = false;
      for (UINT dim_size : dml_dims_buffer) {
        if (dim_size == 0) {
          has_zero_dim = true;
          break;
        }
      }
      if (has_zero_dim) {  // If tensor is logically empty due to a zero
                           // dimension
        std::fill(dml_strides_buffer->begin(), dml_strides_buffer->end(), 0);
      }
      buffer_desc.Strides = dml_strides_buffer->data();
    } else {  // Rank 0 tensor (e.g. scalar after to_dml_dims made it {1} or
              // {0})
      buffer_desc.Strides =
          nullptr;  // No strides for 0D/scalar represented as 1D by convention.
    }
  } else {
    // If dml_strides_buffer is null, DML will assume packed/contiguous tensor.
    buffer_desc.Strides = nullptr;
  }

  if (storage.size() == 0) {
    buffer_desc.TotalTensorSizeInBytes = 0;
  } else if (buffer_desc.Strides && buffer_desc.DimensionCount > 0) {
    UINT64 max_element_offset = 0;
    bool is_empty_by_dims = false;
    for (UINT i = 0; i < buffer_desc.DimensionCount; ++i) {
      if (dml_dims_buffer[i] == 0) {
        is_empty_by_dims = true;
        break;
      }
      max_element_offset += (static_cast<UINT64>(dml_dims_buffer[i]) - 1) *
                            (*dml_strides_buffer)[i];
    }
    if (is_empty_by_dims) {
      buffer_desc.TotalTensorSizeInBytes = 0;
    } else {
      buffer_desc.TotalTensorSizeInBytes =
          (max_element_offset + 1) * storage.item_size();
    }
  } else {  // Contiguous or Strides = nullptr
    buffer_desc.TotalTensorSizeInBytes =
        static_cast<UINT64>(storage.size()) * storage.item_size();
  }
  buffer_desc.GuaranteedBaseOffsetAlignment =
      0;  // DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  DML_TENSOR_DESC tensor_desc;
  tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc.Desc = &buffer_desc;
  return tensor_desc;
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

  UINT64 total_bytes =
      static_cast<UINT64>(storage.size()) * storage.item_size();

  calculate_strides_and_total_size(dml_dtype, internal_sizes_vec,
                                   strides_override, total_bytes);

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
  calculate_strides_and_total_size(dml_dtype, internal_sizes_vec, strides_in,
                                   final_total_bytes);

  buffer_desc_internal.TotalTensorSizeInBytes = final_total_bytes;
  buffer_desc_internal.Strides =
      internal_strides_vec.empty() ? nullptr : internal_strides_vec.data();
  buffer_desc_internal.GuaranteedBaseOffsetAlignment = 0;

  tensor_desc_internal.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc_internal.Desc = &buffer_desc_internal;
}

void DmlTensorDescBundle::calculate_strides_and_total_size(
    DML_TENSOR_DATA_TYPE dml_dtype,
    const std::vector<UINT>& current_sizes,  // internal_sizes_vec
    const std::vector<UINT>*
        strides_input,               // User-provided strides (can be nullptr)
    UINT64& total_size_in_bytes_ref  // Input: preferred total size or 0.
                                     // Output: calculated total size.
) {
  internal_strides_vec.clear();
  const size_t rank = current_sizes.size();

  if (strides_input) {
    internal_strides_vec = *strides_input;
    if (internal_strides_vec.size() != rank && rank > 0) {
      THROW_INVALID_ARGUMENT(
          "Provided strides_override rank does not match tensor rank.");
    }
  } else {
#if 0
    if (rank > 0) {
      internal_strides_vec.resize(rank);
      internal_strides_vec[rank - 1] = 1;
      for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        UINT next_dim_size = current_sizes[i + 1];
        internal_strides_vec[i] = internal_strides_vec[i + 1] *
                                  (next_dim_size == 0 ? 1 : next_dim_size);
      }
      // Handle cases where a dimension is 0, leading to 0 elements.
      bool has_zero_dim = false;
      for (UINT s : current_sizes) {
        if (s == 0) {
          has_zero_dim = true;
          break;
        }
      }
      if (has_zero_dim) {
        // if any dim is 0, all strides for that and outer become effectively 0.
        // This requires careful definition. For total size calc, 0 elements = 0
        // bytes. DML itself might treat strides differently for zero-sized
        // dimensions for broadcasting, but for TotalTensorSizeInBytes for a
        // non-broadcasted buffer, 0 elements mean 0 bytes. If the goal is
        // broadcasting a zero-size dim, its stride being 0 is fine. Standard
        // contiguous stride calculation might give non-zero strides for dims
        // outside a 0 dim. Example: shape {2,0,3}. Contiguous strides: {0,3,1}.
        // Here `internal_strides_vec` is okay.
      }
    }
#else
    internal_strides_vec.clear();
#endif
  }

  if (total_size_in_bytes_ref ==
      0) {  // If total size wasn't provided, calculate it.
    UINT64 num_elements = 1;
    if (rank ==
        0) {  // Scalar by rank, assume 1 element unless sizes_vec indicates {0}
      if (!current_sizes.empty() && current_sizes[0] == 0)
        num_elements = 0;
      else
        num_elements = 1;

    } else {
      bool has_zero_dim = false;
      for (UINT s : current_sizes) {
        if (s == 0) {
          has_zero_dim = true;
          break;
        }
        num_elements *= s;
      }
      if (has_zero_dim)
        num_elements = 0;
    }
    total_size_in_bytes_ref =
        num_elements * get_dml_element_size_in_bytes(dml_dtype);

  } else {  // total_size_in_bytes_ref was provided, use it (but verify
            // consistency if strides are also given)
    if (!internal_strides_vec.empty() && rank > 0) {
      UINT64 calculated_min_bytes_from_strides = 0;
      bool has_zero_dim_for_stride_calc = false;
      for (UINT s : current_sizes)
        if (s == 0) {
          has_zero_dim_for_stride_calc = true;
          break;
        }

      if (rank > 0 && !has_zero_dim_for_stride_calc) {  // Only if all dims > 0
        UINT64 max_offset = 0;
        for (size_t i = 0; i < rank; ++i) {
          if (current_sizes[i] > 0) {  // only add to offset if dim is not 0
            max_offset += (static_cast<UINT64>(current_sizes[i]) - 1) *
                          internal_strides_vec[i];
          }
        }
        calculated_min_bytes_from_strides =
            (max_offset + 1) * get_dml_element_size_in_bytes(dml_dtype);
      } else if (has_zero_dim_for_stride_calc) {
        calculated_min_bytes_from_strides =
            0;  // Tensor is empty if any dimension is 0
      }

      // This is a soft check; DML will do the hard validation.
      // User-provided TotalTensorSizeInBytes might be larger due to
      // padding/allocations. It must be AT LEAST
      // calculated_min_bytes_from_strides.
      if (total_size_in_bytes_ref < calculated_min_bytes_from_strides &&
          !has_zero_dim_for_stride_calc) {
        // Allow provided total_size_in_bytes_ref if it implies a different
        // physical layout (e.g. a view into a larger buffer) However, for
        // typical "create tensor" ops, it should match or be larger. For this
        // utility, we primarily focus on logical size matching. Let DML ops
        // validate this.
      }
    }
  }
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateDmlConstantTensor(
    dml::Device* resolved_ct2_dml_device,
    const std::vector<UINT>& resolved_dims,
    DML_TENSOR_DATA_TYPE resolved_dml_tensor_type,
    DML_SCALAR_UNION resolved_scalar_value,
    DmlTensorDescBundle& bundle_for_constant) {
  // Initialize the bundle with the details
  bundle_for_constant =
      DmlTensorDescBundle(resolved_dml_tensor_type, resolved_dims, nullptr, 0);

  Microsoft::WRL::ComPtr<ID3D12Resource> constant_resource =
      resolved_ct2_dml_device->CreatePreferredDeviceMemoryBuffer(
          bundle_for_constant.get_buffer_desc().TotalTensorSizeInBytes);
  resolved_ct2_dml_device->KeepAliveUntilNextCommandListDispatch(
      constant_resource);

  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc{};
  fill_desc.OutputTensor = &bundle_for_constant.get_tensor_desc();
  fill_desc.ValueDataType = resolved_dml_tensor_type;
  fill_desc.Value = resolved_scalar_value;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT, &fill_desc};

  dml::Operator* fill_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  DML_BUFFER_BINDING temp_output_binding = create_buffer_binding(
      constant_resource.Get(), 0,
      bundle_for_constant.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc =
      create_binding_desc(&temp_output_binding);

  fill_op->Execute({}, {output_binding_desc});
  return constant_resource;
}

DmlBufferBindingBundle::DmlBufferBindingBundle(ID3D12Resource* resource,
                                               UINT64 offset,
                                               UINT64 size_in_bytes)
    : buffer_binding_{resource, offset, size_in_bytes},
      type_(DML_BINDING_TYPE_BUFFER) {
  if (resource == nullptr) {
    type_ = DML_BINDING_TYPE_NONE;
  }
  if (size_in_bytes == 0 && resource) {
    D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Format != DXGI_FORMAT_UNKNOWN) {
      THROW_INVALID_ARGUMENT(
          "DML buffer binding size cannot be 0 for non-buffer resources.");
    }
    UINT64 size_in_byte = desc.Width;

    buffer_binding_.SizeInBytes = size_in_byte;
  }
}

}  // namespace utils
}  // namespace dml
}  // namespace ctranslate2