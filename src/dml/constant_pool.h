#pragma once

#include <memory>
#include <unordered_map>

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
namespace dml {
class ConstantPool {
 public:
  ConstantPool();
  ~ConstantPool();

  // Disable copy and move semantics.
  ConstantPool(const ConstantPool&) = delete;
  ConstantPool& operator=(const ConstantPool&) = delete;
  ConstantPool(ConstantPool&&) = delete;
  ConstantPool& operator=(ConstantPool&&) = delete;
  template <typename T>
  static StorageView& get_constant(
      Shape shape,
      const std::vector<T>& init,
      ctranslate2::Device device = ctranslate2::Device::DirectML);

 private:
  // Get a constant value from the pool, creating it if necessary.
  template <typename T>
  StorageView& _get_constant(
      Shape shape,
      const std::vector<T>& init,
      ctranslate2::Device device = ctranslate2::Device::DirectML);

  std::unordered_map<std::string, std::unique_ptr<StorageView>> _constants;
};
}  // namespace dml
}  // namespace ctranslate2