#include "operator_cache.h"
#include "backend_dml.h"
#include "common.h"
#include "operator.h"

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

DMLOperatorCache::DMLOperatorCache(Device* device) : _device(device) {}

DMLOperatorCache::~DMLOperatorCache() = default;

DMLOperatorCache& DMLOperatorCache::instance() {
  DMLOperatorCache* instance = dml::get_device()->GetOperatorCache();
  if (!instance) {
    SPDLOG_ERROR("DMLOperatorCache instance is not initialized.");
    throw std::runtime_error("DMLOperatorCache instance is not initialized.");
  }
  return *instance;
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

// Helper function to serialize DML_SCALE_BIAS (defined as a static free
// function in the namespace)
static void SerializeScaleBias(std::ostringstream& key_stream,
                               const DML_SCALE_BIAS* scale_bias) {
  if (!scale_bias) {
    append_bytes(
        key_stream,
        (uint64_t)0xABABABABABABABABULL);  // Placeholder for null ScaleBias
    return;
  }
  append_bytes(key_stream, scale_bias->Scale);
  append_bytes(key_stream, scale_bias->Bias);
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

// Private helper method to serialize the operator-specific description part of
// the key.
void DMLOperatorCache::GenerateCacheKeyForDesc(
    std::ostringstream& key_stream,
    const DML_OPERATOR_DESC* op_desc) {
  // Serialize the specific operator description based on its type.
  // This switch must cover ALL operator types used in your application for
  // correct caching.
  switch (op_desc->Type) {
    case DML_OPERATOR_ELEMENT_WISE_IDENTITY: {
      auto desc = static_cast<const DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
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
      if (desc->FusedActivation) {
        append_bytes(key_stream, desc->FusedActivation->Type);
        GenerateCacheKeyForDesc(key_stream,
                                desc->FusedActivation);  // Recursive call
      } else {
        append_bytes(key_stream,
                     (uint64_t)0xF1F1F1F1F1F1F1F1ULL);  // Placeholder for null
                                                        // FusedActivation
      }
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
        case DML_TENSOR_DATA_TYPE_FLOAT16:  // Stored as UInt16 in
                                            // DML_SCALAR_UNION
          append_bytes(key_stream, desc->Value.UInt16);
          break;
        case DML_TENSOR_DATA_TYPE_FLOAT64:
          append_bytes(key_stream, desc->Value.Float64);
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
        // Note: DML_TENSOR_DATA_TYPE_UINT4 and DML_TENSOR_DATA_TYPE_INT4 are
        // not directly represented in DML_SCALAR_UNION as distinct members. The
        // default fallback handles other cases by serializing raw bytes.
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
      // DML_ELEMENT_WISE_EXP_OPERATOR_DESC from DirectML.h has ScaleBias.
      SerializeScaleBias(key_stream, desc->ScaleBias);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_LOG: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_LOG_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_SIN: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_SIN_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_COS: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_COS_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
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
    // Removed DML_OPERATOR_SCATTER case to avoid duplication with
    // DML_OPERATOR_SCATTER_ELEMENTS as they share the same enum value (94) in
    // the current DML headers. DML_OPERATOR_SCATTER_ELEMENTS is used in the ops
    // files.

    // Newly added operator types
    case DML_OPERATOR_ELEMENT_WISE_ADD1: {
      auto desc = static_cast<const DML_ELEMENT_WISE_ADD1_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      if (desc->FusedActivation) {
        append_bytes(key_stream, desc->FusedActivation->Type);
        GenerateCacheKeyForDesc(key_stream,
                                desc->FusedActivation);  // Recursive call
      } else {
        append_bytes(key_stream,
                     (uint64_t)0xF1F1F1F1F1F1F1F1ULL);  // Placeholder for null
                                                        // FusedActivation
      }
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_DIVIDE: {
      auto desc = static_cast<const DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_NEGATE: {
      auto desc = static_cast<const DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_ABS: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_ABS_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_ROUND: {
      auto desc = static_cast<const DML_ELEMENT_WISE_ROUND_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->RoundingMode);
      break;
    }
    case DML_OPERATOR_CONVOLUTION: {
      auto desc =
          static_cast<const DML_CONVOLUTION_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->FilterTensor);
      SerializeTensorDesc(
          key_stream,
          desc->BiasTensor);  // Handles null via SerializeTensorDesc
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Mode);
      append_bytes(key_stream, desc->Direction);
      append_bytes(key_stream, desc->DimensionCount);
      append_bytes_array(key_stream, desc->Strides,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->Dilations,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->StartPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->EndPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->OutputPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes(key_stream, desc->GroupCount);
      if (desc->FusedActivation) {
        append_bytes(key_stream, desc->FusedActivation->Type);
        GenerateCacheKeyForDesc(key_stream,
                                desc->FusedActivation);  // Recursive call
      } else {
        append_bytes(key_stream,
                     (uint64_t)0xF1F1F1F1F1F1F1F1ULL);  // Placeholder for null
                                                        // FusedActivation
      }
      break;
    }
    case DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION: {
      auto desc =
          static_cast<const DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->InputScaleTensor);
      SerializeTensorDesc(key_stream,
                          desc->InputZeroPointTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->FilterTensor);
      SerializeTensorDesc(key_stream, desc->FilterScaleTensor);
      SerializeTensorDesc(key_stream,
                          desc->FilterZeroPointTensor);   // Handles null
      SerializeTensorDesc(key_stream, desc->BiasTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputScaleTensor);
      SerializeTensorDesc(key_stream,
                          desc->OutputZeroPointTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->DimensionCount);
      append_bytes_array(key_stream, desc->Strides,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->Dilations,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->StartPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->EndPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes(key_stream, desc->GroupCount);
      break;
    }
    case DML_OPERATOR_CONVOLUTION_INTEGER: {
      auto desc = static_cast<const DML_CONVOLUTION_INTEGER_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->InputZeroPointTensor);
      SerializeTensorDesc(key_stream, desc->FilterTensor);
      SerializeTensorDesc(key_stream,
                          desc->FilterZeroPointTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->DimensionCount);
      append_bytes_array(key_stream, desc->Strides,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->Dilations,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->StartPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->EndPadding,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes(key_stream, desc->GroupCount);
      break;
    }
    case DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2: {
      auto desc =
          static_cast<const DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->ScaleTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->BiasTensor);   // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->AxisCount);
      append_bytes_array(key_stream, desc->Axes,
                         desc->AxisCount * sizeof(UINT));
      append_bytes(key_stream, desc->UseMean);
      append_bytes(key_stream, desc->UseVariance);
      append_bytes(key_stream, desc->Epsilon);
      if (desc->FusedActivation) {
        append_bytes(key_stream, desc->FusedActivation->Type);
        GenerateCacheKeyForDesc(key_stream,
                                desc->FusedActivation);  // Recursive call
      } else {
        append_bytes(key_stream,
                     (uint64_t)0xF1F1F1F1F1F1F1F1ULL);  // Placeholder for null
                                                        // FusedActivation
      }
      break;
    }
    case DML_OPERATOR_JOIN: {
      auto desc = static_cast<const DML_JOIN_OPERATOR_DESC*>(op_desc->Desc);
      append_bytes(key_stream, desc->InputCount);
      for (UINT i = 0; i < desc->InputCount; ++i) {
        SerializeTensorDesc(key_stream, desc->InputTensors + i);
      }
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      break;
    }
    case DML_OPERATOR_SPLIT: {
      auto desc = static_cast<const DML_SPLIT_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      append_bytes(key_stream, desc->OutputCount);
      for (UINT i = 0; i < desc->OutputCount; ++i) {
        SerializeTensorDesc(key_stream, desc->OutputTensors + i);
      }
      append_bytes(key_stream, desc->Axis);
      break;
    }
    case DML_OPERATOR_SLICE: {
      auto desc = static_cast<const DML_SLICE_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->DimensionCount);
      append_bytes_array(key_stream, desc->Offsets,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->Sizes,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->Strides,
                         desc->DimensionCount * sizeof(UINT));
      break;
    }
    case DML_OPERATOR_SLICE1: {
      auto desc = static_cast<const DML_SLICE1_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->DimensionCount);
      append_bytes_array(key_stream, desc->InputWindowOffsets,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->InputWindowSizes,
                         desc->DimensionCount * sizeof(UINT));
      append_bytes_array(key_stream, desc->InputWindowStrides,
                         desc->DimensionCount * sizeof(INT));
      break;
    }
    case DML_OPERATOR_TILE: {
      auto desc = static_cast<const DML_TILE_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->RepeatsCount);
      append_bytes_array(key_stream, desc->Repeats,
                         desc->RepeatsCount * sizeof(UINT));
      break;
    }
    case DML_OPERATOR_GATHER: {
      auto desc = static_cast<const DML_GATHER_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->IndicesTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      append_bytes(key_stream, desc->IndexDimensions);
      break;
    }
    case DML_OPERATOR_GATHER_ELEMENTS: {
      auto desc =
          static_cast<const DML_GATHER_ELEMENTS_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->IndicesTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      break;
    }
    case DML_OPERATOR_SCATTER_ELEMENTS: {
      auto desc =
          static_cast<const DML_SCATTER_ELEMENTS_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->IndicesTensor);
      SerializeTensorDesc(key_stream, desc->UpdatesTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      break;
    }
    case DML_OPERATOR_TOP_K1: {
      auto desc = static_cast<const DML_TOP_K1_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputValueTensor);
      SerializeTensorDesc(key_stream, desc->OutputIndexTensor);
      append_bytes(key_stream, desc->Axis);
      append_bytes(key_stream, desc->K);
      append_bytes(key_stream, desc->AxisDirection);
      break;
    }
    case DML_OPERATOR_ACTIVATION_GELU: {
      auto desc =
          static_cast<const DML_ACTIVATION_GELU_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_SWISH: {
      auto desc =
          static_cast<const DML_ACTIVATION_SWISH_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->SigmoidInputScale);
      break;
    }
    case DML_OPERATOR_ACTIVATION_LINEAR: {
      auto desc = static_cast<const DML_ACTIVATION_LINEAR_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Alpha);
      append_bytes(key_stream, desc->Beta);
      break;
    }
    case DML_OPERATOR_ACTIVATION_IDENTITY: {
      // For DML_ACTIVATION_IDENTITY_OPERATOR_DESC
      auto desc = static_cast<const DML_ACTIVATION_IDENTITY_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_SOFTMAX: {  // Older version, no axis
      auto desc = static_cast<const DML_ACTIVATION_SOFTMAX_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX: {  // Older version, no axis
      auto desc = static_cast<const DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ACTIVATION_SOFTMAX1: {  // Newer version with axis
      auto desc = static_cast<const DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->AxisCount);
      append_bytes_array(key_stream, desc->Axes,
                         desc->AxisCount * sizeof(UINT));
      break;
    }
    case DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1: {  // Newer version with axis
      auto desc = static_cast<const DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->AxisCount);
      append_bytes_array(key_stream, desc->Axes,
                         desc->AxisCount * sizeof(UINT));
      break;
    }
    case DML_OPERATOR_RANDOM_GENERATOR: {
      auto desc =
          static_cast<const DML_RANDOM_GENERATOR_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputStateTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeTensorDesc(key_stream, desc->OutputStateTensor);  // Handles null
      append_bytes(key_stream, desc->Type);
      break;
    }
    case DML_OPERATOR_FILL_VALUE_SEQUENCE: {
      auto desc = static_cast<const DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->ValueDataType);
      // Serialize ValueStart and ValueDelta based on ValueDataType
      // This is similar to DML_OPERATOR_FILL_VALUE_CONSTANT's Value
      // serialization
      switch (desc->ValueDataType) {
        case DML_TENSOR_DATA_TYPE_FLOAT32:
          append_bytes(key_stream, desc->ValueStart.Float32);
          append_bytes(key_stream, desc->ValueDelta.Float32);
          break;
        case DML_TENSOR_DATA_TYPE_FLOAT16:  // Stored as UInt16
          append_bytes(key_stream, desc->ValueStart.UInt16);
          append_bytes(key_stream, desc->ValueDelta.UInt16);
          break;
        case DML_TENSOR_DATA_TYPE_FLOAT64:
          append_bytes(key_stream, desc->ValueStart.Float64);
          append_bytes(key_stream, desc->ValueDelta.Float64);
          break;
        case DML_TENSOR_DATA_TYPE_UINT64:
          append_bytes(key_stream, desc->ValueStart.UInt64);
          append_bytes(key_stream, desc->ValueDelta.UInt64);
          break;
        case DML_TENSOR_DATA_TYPE_INT64:
          append_bytes(key_stream, desc->ValueStart.Int64);
          append_bytes(key_stream, desc->ValueDelta.Int64);
          break;
        case DML_TENSOR_DATA_TYPE_UINT32:
          append_bytes(key_stream, desc->ValueStart.UInt32);
          append_bytes(key_stream, desc->ValueDelta.UInt32);
          break;
        case DML_TENSOR_DATA_TYPE_INT32:
          append_bytes(key_stream, desc->ValueStart.Int32);
          append_bytes(key_stream, desc->ValueDelta.Int32);
          break;
        case DML_TENSOR_DATA_TYPE_UINT16:
          append_bytes(key_stream, desc->ValueStart.UInt16);
          append_bytes(key_stream, desc->ValueDelta.UInt16);
          break;
        case DML_TENSOR_DATA_TYPE_INT16:
          append_bytes(key_stream, desc->ValueStart.Int16);
          append_bytes(key_stream, desc->ValueDelta.Int16);
          break;
        case DML_TENSOR_DATA_TYPE_UINT8:
          append_bytes(key_stream, desc->ValueStart.UInt8);
          append_bytes(key_stream, desc->ValueDelta.UInt8);
          break;
        case DML_TENSOR_DATA_TYPE_INT8:
          append_bytes(key_stream, desc->ValueStart.Int8);
          append_bytes(key_stream, desc->ValueDelta.Int8);
          break;
        default:
          // Fallback for unlisted types, serialize the raw union bytes.
          append_bytes_array(key_stream, &desc->ValueStart,
                             sizeof(DML_SCALAR_UNION));
          append_bytes_array(key_stream, &desc->ValueDelta,
                             sizeof(DML_SCALAR_UNION));
          break;
      }
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL: {
      auto desc = static_cast<
          const DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_IF: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_IF_OPERATOR_DESC*>(op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->ConditionTensor);
      SerializeTensorDesc(key_stream, desc->ATensor);
      SerializeTensorDesc(key_stream, desc->BTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_CUMULATIVE_SUMMATION: {
      auto desc = static_cast<const DML_CUMULATIVE_SUMMATION_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      append_bytes(key_stream, desc->Axis);
      append_bytes(key_stream, desc->AxisDirection);
      append_bytes(key_stream, desc->HasExclusiveSum);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->ScaleTensor);
      SerializeTensorDesc(key_stream, desc->ZeroPointTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR: {
      auto desc =
          static_cast<const DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC*>(
              op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->ScaleTensor);
      SerializeTensorDesc(key_stream, desc->ZeroPointTensor);  // Handles null
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      break;
    }
    case DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR: {
      auto desc = static_cast<const DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeTensorDesc(key_stream, desc->OutputScaleTensor);
      SerializeTensorDesc(key_stream,
                          desc->OutputZeroPointTensor);  // Handles null
      break;
    }
    case DML_OPERATOR_ELEMENT_WISE_RECIP: {
      auto desc = static_cast<const DML_ELEMENT_WISE_RECIP_OPERATOR_DESC*>(
          op_desc->Desc);
      SerializeTensorDesc(key_stream, desc->InputTensor);
      SerializeTensorDesc(key_stream, desc->OutputTensor);
      SerializeScaleBias(key_stream, desc->ScaleBias);
      break;
    }

    default:
      // This indicates an operator type not explicitly handled by the cache key
      // generation. This can lead to incorrect caching (collisions or missed
      // hits). It's crucial to list all operator types your application uses.
      throw std::runtime_error(
          "DMLOperatorCache: Unhandled DML_OPERATOR_TYPE in "
          "GenerateCacheKey: " +
          std::to_string(op_desc->Type));
      break;
  }
  // key_stream.str() is not returned here as this is a helper.
  // The main GenerateCacheKey will return the final string.
}

std::string DMLOperatorCache::GenerateCacheKey(const DML_OPERATOR_DESC* op_desc,
                                               DML_EXECUTION_FLAGS flags) {
  std::ostringstream key_stream;
  // No need for std::hex or std::setfill if appending raw bytes.

  append_bytes(key_stream, op_desc->Type);  // Serialize main operator Type
  append_bytes(key_stream, flags);          // Serialize main operator Flags

  GenerateCacheKeyForDesc(key_stream,
                          op_desc);  // Serialize operator-specific description

  return key_stream.str();
}

Operator* DMLOperatorCache::GetOrCreateCompiledOperator(
    const DML_OPERATOR_DESC* op_desc,
    DML_EXECUTION_FLAGS flags,
    PCWSTR name) {
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
        return it->second.get();
      }
    }
    SPDLOG_DEBUG(("Cache MISS for key prefix: " +
                  key.substr(0, std::min(key.length(), (size_t)16)) + "\n")
                     .c_str());
  }

  auto dml_device = _device->DML();
  // Not found, create and compile. This is done outside the lock to avoid
  // holding it during potentially long operations.
  Microsoft::WRL::ComPtr<IDMLOperator> dml_operator;
  THROW_IF_FAILED(
      dml_device->CreateOperator(op_desc, IID_PPV_ARGS(&dml_operator)));

  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_operator;
  THROW_IF_FAILED(dml_device->CompileOperator(
      dml_operator.Get(), flags, IID_PPV_ARGS(&compiled_operator)));
  if (name)
    compiled_operator->SetName(name);
  std::unique_ptr<Operator> operator_obj(
      new Operator(_device, op_desc->Type, std::move(compiled_operator)));

  if (kCacheEnabled) {
    // Re-lock to insert into the cache
    std::lock_guard<std::mutex> lock(_mutex);
    // Double-check if another thread created it in the meantime
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      return it->second.get();  // Another thread created and inserted it
    }
    _cache[key] = std::move(operator_obj);
    return _cache[key].get();
  }
  return operator_obj.release();
}

// Implementation of the global helper function
Operator* GetOrCreateCompiledOperatorApi(const DML_OPERATOR_DESC* op_desc,
                                         DML_EXECUTION_FLAGS flags,
                                         PCWSTR name) {
  // Assumes get_dml_device() is available in ctranslate2::dml namespace
  // and returns the current IDMLDevice*.
  return DMLOperatorCache::instance().GetOrCreateCompiledOperator(
      op_desc, flags | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS, name);
}

}  // namespace dml
}  // namespace ctranslate2