#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include "dxmodule.h"

namespace ctranslate2 {
namespace dml {

class Operator;
class Device;
namespace utils {
class DmlOperatorDescBundle;
}

class DMLOperatorCache {
 public:
  explicit DMLOperatorCache(Device*);
  ~DMLOperatorCache();

  // Returns the singleton instance of the cache.
  static DMLOperatorCache& instance();

  // Retrieves an existing compiled operator from the cache or creates,
  // compiles, and caches a new one if not found. device: The DML device used
  // for operator creation and compilation. op_desc: The description of the
  // operator to get or create. flags: Execution flags for compiling the
  // operator. Returns a ComPtr to the compiled DML operator.
  Operator* GetOrCreateCompiledOperator(utils::DmlOperatorDescBundle&& op_desc,
                                        DML_EXECUTION_FLAGS flags,
                                        PCWSTR name = nullptr);

  // Clears all cached operators.
  // Useful if, for example, the DML device is recreated.
  void Clear();

  void AddOperator(std::string&& key, Microsoft::WRL::ComPtr<Operator>&& op);

  Operator* GetOperator(const std::string& key);

 private:
  // Delete copy constructor and assignment operator.
  DMLOperatorCache(const DMLOperatorCache&) = delete;
  DMLOperatorCache& operator=(const DMLOperatorCache&) = delete;

  std::unordered_map<std::string, Microsoft::WRL::ComPtr<Operator>> _cache;
  std::mutex _mutex;      // Mutex to protect cache access.
  class Device* _device;  // The DML device used for operator creation.
};

// Global helper function to easily get or create a compiled operator using the
// singleton cache. op_desc: The description of the operator. flags: Execution
// flags for compilation (defaults to DML_EXECUTION_FLAG_NONE). Returns a ComPtr
// to the compiled DML operator. This function assumes get_dml_device() is
// available to provide the IDMLDevice.
Operator* GetOrCreateCompiledOperatorApi(
    utils::DmlOperatorDescBundle&& op_desc,
    DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE,
    PCWSTR name = nullptr);

}  // namespace dml
}  // namespace ctranslate2