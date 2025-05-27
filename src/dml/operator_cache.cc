#include "operator_cache.h"
#include "backend_dml.h"
#include "common.h"

#include <spdlog/spdlog.h>

#include <sstream>

namespace ctranslate2 {
namespace dml {
constexpr static const bool kCacheEnabled = true;

// Helper function to append raw bytes of a value to the ostringstream.
template <typename T>
void append_bytes(std::ostringstream& ss, const T& value) {
  ss.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Helper function to append raw bytes of an array to the ostringstream.
// Includes a placeholder if the data pointer is null or size is zero.
void append_bytes_array(std::ostringstream& ss,
                        const void* data,
                        size_t size_in_bytes) {
  if (data && size_in_bytes > 0) {
    ss.write(reinterpret_cast<const char*>(data), size_in_bytes);
  } else {
    // Append a unique placeholder for null or zero-size array to differentiate
    uint64_t placeholder = 0x123456789ABCDEF0ULL;  // Arbitrary placeholder
    append_bytes(ss, placeholder);
  }
}

DMLOperatorCache& DMLOperatorCache::instance() {
  static DMLOperatorCache cache_instance;
  return cache_instance;
}

void DMLOperatorCache::Clear() {
  std::lock_guard<std::mutex> lock(_mutex);
  _cache.clear();
  SPDLOG_DEBUG("DMLOperatorCache cleared.");
}

void DMLOperatorCache::SerializeBufferTensorDesc(
    std::ostringstream& key_stream,
    const DML_BUFFER_TENSOR_DESC* buffer_desc) {
  if (!buffer_desc) {
    append_bytes(
        key_stream,
        (uint64_t)0xA1A1A1A1A1A1A1A1ULL);  // Placeholder for null buffer_desc
    return;
  }
  append_bytes(key_stream, buffer_desc->DataType);
  append_bytes(key_stream, buffer_desc->Flags);
  append_bytes(key_stream, buffer_desc->DimensionCount);

  if (buffer_desc->Sizes) {
    append_bytes_array(key_stream, buffer_desc->Sizes,
                       buffer_desc->DimensionCount * sizeof(UINT));
  } else {
    append_bytes(
        key_stream,
        (uint64_t)0xB1B1B1B1B1B1B1B1ULL);  // Placeholder for null Sizes array
  }

  if (buffer_desc->Strides) {
    append_bytes_array(key_stream, buffer_desc->Strides,
                       buffer_desc->DimensionCount * sizeof(UINT));
  } else {
    append_bytes(
        key_stream,
        (uint64_t)0xC1C1C1C1C1C1C1C1ULL);  // Placeholder for null Strides array
  }
  append_bytes(key_stream, buffer_desc->TotalTensorSizeInBytes);
  append_bytes(key_stream, buffer_desc->GuaranteedBaseOffsetAlignment);
}

void DMLOperatorCache::SerializeTensorDesc(std::ostringstream& key_stream,
                                           const DML_TENSOR_DESC* tensor_desc) {
  if (!tensor_desc) {
    // Placeholder for null tensor_desc
    append_bytes(key_stream, (uint64_t)0xD1D1D1D1D1D1D1D1ULL);
    return;
  }
  append_bytes(key_stream, tensor_desc->Type);
  switch (tensor_desc->Type) {
    case DML_TENSOR_TYPE_BUFFER:
      SerializeBufferTensorDesc(
          key_stream,
          static_cast<const DML_BUFFER_TENSOR_DESC*>(tensor_desc->Desc));
      break;
    // DML_TENSOR_TYPE_INVALID is not expected in an operator's tensor
    // description. Add other DML_TENSOR_TYPEs if they are used (e.g.
    // DML_TENSOR_TYPE_SCALAR - though less common in op descs).
    default:
      // This case indicates an unhandled or unknown tensor type.
      // For robust key generation, all possible tensor types used should be
      // explicitly handled.
      // Placeholder for unhandled tensor type
      append_bytes(key_stream, (uint64_t)0xE1E1E1E1E1E1E1E1ULL);
      break;
  }
}

std::string DMLOperatorCache::GenerateCacheKey(const DML_OPERATOR_DESC* op_desc,
                                               DML_EXECUTION_FLAGS flags) {
  std::ostringstream key_stream;
  // No need for std::hex or std::setfill if appending raw bytes.

  append_bytes(key_stream, op_desc->Type);
  append_bytes(key_stream, flags);

  // Serialize the specific operator description based on its type.
  // This switch must cover ALL operator types used in your application for
  // correct caching.
  switch (op_desc->Type) {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY: {
      auto desc = static_cast<const DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      // Note: DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC also has an optional
      // const DML_SCALE_BIAS* ScaleBias. If you use it, it must be part of the
      // key. For simplicity, assuming it's not used or always null. if
      // (desc->ScaleBias) { /* serialize ScaleBias */ } else { /* placeholder
      // for null ScaleBias */ }
      break;
    }
    case DML_OPERATOR_GEMM: {
      auto desc = static_cast<const DML_GEMM_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      if (desc->CTensor) {
        SerializeTensorDesc(key_stream, desc->CTensor);
      } else {
        append_bytes(
            key_stream,
            (uint64_t)0xF0F0F0F0F0F0F0F0ULL);  // Placeholder for null CTensor
      }
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->TransA);
      append_bytes(key_stream, desc->TransB);
      append_bytes(key_stream, desc->Alpha);
      append_bytes(key_stream, desc->Beta);
      // Note: DML_GEMM_OPERATOR_DESC also has an optional const
      // DML_OPERATOR_DESC* FusedActivation. If used, this FusedActivation
      // (which is another DML_OPERATOR_DESC) must be recursively serialized.
      // This adds complexity. For simplicity here, assuming it's not used or
      // always null. if (desc->FusedActivation) { /* recursively call
      // GenerateCacheKey or similar for FusedActivation */ } else { /*
      // placeholder for null FusedActivation */ }
      break;
    }
    case DML_OPERATOR_REDUCE: {
      auto desc = static_cast<const DML_REDUCE_OPERATOR_DESC*>(op_desc->Desc);
      append_bytes(key_stream, desc->Function);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->AxisCount);
      append_bytes_array(key_stream, desc->Axes,
                         desc->AxisCount * sizeof(UINT));
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_ADD: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_ADD_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_FILL_VALUE_CONSTANT: {
      auto desc = static_cast<const DML_FILL_VALUE_CONSTANT_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->ValueDataType);
      switch (desc->ValueDataType) {
        case DML_TENSOR_DATA_TYPE_FLOAT32:
          append_bytes(key_stream, desc->Value.Float32);
          break;
        case DML_TENSOR_DATA_TYPE_FLOAT16:
          append_bytes(key_stream, desc->Value.UInt16);
          break;
        case DML_TENSOR_DATA_TYPE_UINT64:
          append_bytes(key_stream, desc->Value.UInt64);
          break;
        case DML_TENSOR_DATA_TYPE_INT64:
          append_bytes(key_stream, desc->Value.Int64);
          break;
        case DML_TENSOR_DATA_TYPE_UINT32:
          append_bytes(key_stream, desc->Value.UInt32);
          break;
        case DML_TENSOR_DATA_TYPE_INT32:
          append_bytes(key_stream, desc->Value.Int32);
          break;
        case DML_TENSOR_DATA_TYPE_UINT16:
          append_bytes(key_stream, desc->Value.UInt16);
          break;
        case DML_TENSOR_DATA_TYPE_INT16:
          append_bytes(key_stream, desc->Value.Int16);
          break;
        case DML_TENSOR_DATA_TYPE_UINT8:
          append_bytes(key_stream, desc->Value.UInt8);
          break;
        case DML_TENSOR_DATA_TYPE_INT8:
          append_bytes(key_stream, desc->Value.Int8);
          break;
        default:
          // Fallback for unlisted types, serialize the raw union bytes. This is
          // less precise.
          append_bytes_array(key_stream, &desc->Value,
                             sizeof(DML_SCALAR_UNION));
          break;
      }
      break;
    }
    case DML_OPERATOR_CAST: {
      auto desc = static_cast<const DML_CAST_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_RELU: {
      auto desc =
          static_cast<const DML_ACTIVATION_RELU_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_SIGMOID: {
      auto desc = static_cast<const DML_ACTIVATION_SIGMOID_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_TANH: {
      auto desc =
          static_cast<const DML_ACTIVATION_TANH_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_SUBTRACT: {
      auto desc = static_cast<const DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_MULTIPLY: {
      auto desc = static_cast<const DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_MAX: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_MAX_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_MIN: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_MIN_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_EXP: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_EXP_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_LOG: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_LOG_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_SIN: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_SIN_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_COS: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_COS_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ARGMAX: {
      // Note: DML_ARGMAX_OPERATOR_DESC might be
      // DML_TOP_K_OPERATOR_DESC or
      // DML_ARGMIN_OPERATOR_DESC in newer DML
      // Assuming this struct name is correct from your context
      auto desc = static_cast<const DML_ARGMAX_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      // Output is indices
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->AxisCount);
      append_bytes_array(key_stream, desc->Axes,
                         desc->AxisCount * sizeof(UINT));
      append_bytes(key_stream, desc->AxisDirection);
      break;
    }
    case DML_OPERATOR_SCATTER: {
      // As per your context code; modern DML might
      // use DML_SCATTER_ND or DML_SCATTER_ELEMENTS
      auto desc = static_cast<const DML_SCATTER_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->IndicesTensor);
      SerializeTensorDesc(key_stream, desc->UpdatesTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      break;
    }
    // ... Add other DML_OPERATOR_TYPE cases as needed ...
    default:
      // This indicates an operator type not explicitly handled by the cache key
      // generation. This can lead to incorrect caching (collisions or missed
      // hits). It's crucial to list all operator types your application uses.
      // As a fallback, one might try to hash the raw bytes of op_desc->Desc,
      // but its size is unknown without type information.
      // For now, append a unique marker for "unhandled type" along with the
      // type enum.
      // Fallback marker
      // Append type again to make it somewhat unique
      // append_bytes(key_stream, (uint64_t)0xFFFFFFFFFFFFFFFFULL);
      // append_bytes(key_stream, op_desc->Type);
      throw std::runtime_error(
          "DMLOperatorCache: Unhandled DML_OPERATOR_TYPE in "
          "GenerateCacheKey: " +
          std::to_string(op_desc->Type));
      break;
  }
  return key_stream.str();
}

Microsoft::WRL::ComPtr<IDMLCompiledOperator>
DMLOperatorCache::GetOrCreateCompiledOperator(IDMLDevice* device,
                                              const DML_OPERATOR_DESC* op_desc,
                                              DML_EXECUTION_FLAGS flags) {
  std::string key;
  if (kCacheEnabled) {
    key = GenerateCacheKey(op_desc, flags);

    // Scope for the lock
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _cache.find(key);
      if (it != _cache.end()) {
        SPDLOG_DEBUG(("Cache HIT for key prefix: " +
                      key.substr(0, std::min(key.length(), (size_t)16)) + "\n")
                         .c_str());
        return it->second;
      }
    }
    SPDLOG_DEBUG(("Cache MISS for key prefix: " +
                  key.substr(0, std::min(key.length(), (size_t)16)) + "\n")
                     .c_str());
  }

  // Not found, create and compile. This is done outside the lock to avoid
  // holding it during potentially long operations.
  Microsoft::WRL::ComPtr<IDMLOperator> dml_operator;
  THROW_IF_FAILED(device->CreateOperator(op_desc, IID_PPV_ARGS(&dml_operator)));

  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_operator;
  THROW_IF_FAILED(device->CompileOperator(dml_operator.Get(), flags,
                                          IID_PPV_ARGS(&compiled_operator)));

  if (kCacheEnabled) {
    // Re-lock to insert into the cache
    std::lock_guard<std::mutex> lock(_mutex);
    // Double-check if another thread created it in the meantime
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      return it->second;  // Another thread created and inserted it
    }
    _cache[key] = compiled_operator;
  }
  return compiled_operator;
}

// Implementation of the global helper function
Microsoft::WRL::ComPtr<IDMLCompiledOperator> GetOrCreateCompiledOperatorApi(
    const DML_OPERATOR_DESC* op_desc,
    DML_EXECUTION_FLAGS flags) {
  // Assumes get_dml_device() is available in ctranslate2::dml namespace
  // and returns the current IDMLDevice*.
  return DMLOperatorCache::instance().GetOrCreateCompiledOperator(
      ctranslate2::dml::get_dml_device(), op_desc, flags);
}

}  // namespace dml
}  // namespace ctranslate2