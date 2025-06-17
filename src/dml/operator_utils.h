#pragma once

#include <DirectML.h>
#include <sstream>

namespace ctranslate2 {
namespace dml {

class OperatorUtils {
 public:
  // Helper function to append raw bytes of a value to the ostringstream.
  template <typename T>
  static void append_bytes(std::ostringstream& ss, const T& value) {
    ss.write(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  // Helper function to append raw bytes of an array to the ostringstream.
  // Includes a placeholder if the data pointer is null or size is zero.
  static void append_bytes_array(std::ostringstream& ss,
                                 const void* data,
                                 size_t size_in_bytes);

  // Generates a unique string key based on the operator description and flags.
  static std::string GenerateCacheKey(const DML_OPERATOR_DESC* op_desc,
                                      DML_EXECUTION_FLAGS flags);

  static const char* DML_OPERATOR_TYPE_toString(DML_OPERATOR_TYPE type);

 private:
  // Private helper method to serialize the operator-specific description part
  // of the key.
  static void GenerateCacheKeyForDesc(std::ostringstream& key_stream,
                                      const DML_OPERATOR_DESC* op_desc);

  // Helper to serialize DML_TENSOR_DESC into the key stream.
  static void SerializeTensorDesc(std::ostringstream& key_stream,
                                  const DML_TENSOR_DESC* tensor_desc);

  // Helper to serialize DML_BUFFER_TENSOR_DESC into the key stream.
  static void SerializeBufferTensorDesc(
      std::ostringstream& key_stream,
      const DML_BUFFER_TENSOR_DESC* buffer_desc);

  // Helper function to serialize DML_SCALE_BIAS
  static void SerializeScaleBias(std::ostringstream& key_stream,
                                 const DML_SCALE_BIAS* scale_bias);
};

}  // namespace dml
}  // namespace ctranslate2
