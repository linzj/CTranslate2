#include "operator_cache.h"
#include "backend_dml.h"
#include "common.h"
#include "dml_utils.h"
#include "operator.h"
#include "operator_utils.h"

#include <spdlog/spdlog.h>

#include <sstream>

namespace ctranslate2 {
namespace dml {
constexpr static const bool kCacheEnabled = true;
using Microsoft::WRL::ComPtr;

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
}

Operator* DMLOperatorCache::GetOrCreateCompiledOperator(
    utils::DmlOperatorDescBundle&& op_desc,
    DML_EXECUTION_FLAGS flags,
    PCWSTR name) {
  std::string key;
  if (kCacheEnabled) {
    key = OperatorUtils::GenerateCacheKey(&op_desc.get_desc(), flags);

    // Scope for the lock
    {
      std::lock_guard<std::mutex> lock(_mutex);
      auto it = _cache.find(key);
      if (it != _cache.end()) {
        return it->second.Get();
      }
    }
  }

  auto dml_device = _device->DML();
  // Not found, create and compile. This is done outside the lock to avoid
  // holding it during potentially long operations.
  Microsoft::WRL::ComPtr<IDMLOperator> dml_operator;
  THROW_IF_FAILED(dml_device->CreateOperator(&op_desc.get_desc(),
                                             IID_PPV_ARGS(&dml_operator)));

  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled_operator;
  THROW_IF_FAILED(dml_device->CompileOperator(
      dml_operator.Get(), flags, IID_PPV_ARGS(&compiled_operator)));
  if (name)
    compiled_operator->SetName(name);
  ComPtr<Operator> operator_obj(Microsoft::WRL::Make<Operator>(
      _device, std::move(op_desc), std::move(compiled_operator)));

  if (kCacheEnabled) {
    // Re-lock to insert into the cache
    std::lock_guard<std::mutex> lock(_mutex);
    // Double-check if another thread created it in the meantime
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      return it->second.Get();  // Another thread created and inserted it
    }
    _cache[key] = operator_obj;
  }
  if (!kCacheEnabled) {
    _device->KeepAliveUntilNextCommandListDispatch(operator_obj);
  }
  return operator_obj.Get();
}

// Implementation of the global helper function
Operator* GetOrCreateCompiledOperatorApi(utils::DmlOperatorDescBundle&& op_desc,
                                         DML_EXECUTION_FLAGS flags,
                                         PCWSTR name) {
  // Assumes get_dml_device() is available in ctranslate2::dml namespace
  // and returns the current IDMLDevice*.
  return DMLOperatorCache::instance().GetOrCreateCompiledOperator(
      std::move(op_desc), flags | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
      name);
}

}  // namespace dml
}  // namespace ctranslate2