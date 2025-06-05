#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "dxmodule.h"

namespace ctranslate2 {
namespace dml {

class Operator;
class Device;

class DMLOperatorCache {
 public:
  // Returns the singleton instance of the cache.
  static DMLOperatorCache& instance();

  // Retrieves an existing compiled operator from the cache or creates,
  // compiles, and caches a new one if not found. device: The DML device used
  // for operator creation and compilation. op_desc: The description of the
  // operator to get or create. flags: Execution flags for compiling the
  // operator. Returns a ComPtr to the compiled DML operator.
  Operator* GetOrCreateCompiledOperator(Device* device,
                                        const DML_OPERATOR_DESC* op_desc,
                                        DML_EXECUTION_FLAGS flags,
                                        PCWSTR name = nullptr);

  // Clears all cached operators.
  // Useful if, for example, the DML device is recreated.
  void Clear();

 private:
  // Private constructor and destructor for singleton pattern.
  DMLOperatorCache() = default;
  ~DMLOperatorCache() = default;

  // Delete copy constructor and assignment operator.
  DMLOperatorCache(const DMLOperatorCache&) = delete;
  DMLOperatorCache& operator=(const DMLOperatorCache&) = delete;

  // Generates a unique string key based on the operator description and flags.
  std::string GenerateCacheKey(const DML_OPERATOR_DESC* op_desc,
                               DML_EXECUTION_FLAGS flags);

  void GenerateCacheKeyForDesc(std::ostringstream& key_stream,
                               const DML_OPERATOR_DESC* op_desc);

  // Helper to serialize DML_TENSOR_DESC into the key stream.
  void SerializeTensorDesc(std::ostringstream& key_stream,
                           const DML_TENSOR_DESC* tensor_desc);
  // Helper to serialize DML_BUFFER_TENSOR_DESC into the key stream.
  void SerializeBufferTensorDesc(std::ostringstream& key_stream,
                                 const DML_BUFFER_TENSOR_DESC* buffer_desc);

  std::unordered_map<std::string, std::unique_ptr<Operator>> _cache;
  std::mutex _mutex;  // Mutex to protect cache access.
};

// Global helper function to easily get or create a compiled operator using the
// singleton cache. op_desc: The description of the operator. flags: Execution
// flags for compilation (defaults to DML_EXECUTION_FLAG_NONE). Returns a ComPtr
// to the compiled DML operator. This function assumes get_dml_device() is
// available to provide the IDMLDevice.
Operator* GetOrCreateCompiledOperatorApi(
    const DML_OPERATOR_DESC* op_desc,
    DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE,
    PCWSTR name = nullptr);

}  // namespace dml
}  // namespace ctranslate2