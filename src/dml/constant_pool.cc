#include "constant_pool.h"
#include "backend_dml.h"
#include "dxdevice.h"

namespace ctranslate2 {
namespace dml {
ConstantPool::ConstantPool() = default;

ConstantPool::~ConstantPool() = default;

template <typename T>
const StorageView& ConstantPool::_get_constant(
    Shape shape,
    const std::vector<T>& init,
    ctranslate2::Device device) const {
  // Create a unique key for the constant based on its shape and initial values.
  std::string key;
  if constexpr (std::is_same_v<T, float>) {
    key = "float";
  } else if constexpr (std::is_same_v<T, int8_t>) {
    key = "int8";
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    key = "uint8";
  } else {
    THROW_INVALID_ARGUMENT("Unsupported type for constant pool");
  }

  // append each element of shape to the key bitwise.
  for (const auto& dim : shape) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&dim);
    key.append(bytes, bytes + sizeof(dim));
  }

  for (const auto& value : init) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    key.append(bytes, bytes + sizeof(value));
  }

  if (device == ctranslate2::Device::DirectML) {
    key += "_dml";
  } else if (device == ctranslate2::Device::CPU) {
    key += "_cpu";
  } else {
    THROW_INVALID_ARGUMENT("Unsupported device for constant pool: " +
                           std::to_string(static_cast<int>(device)));
  }

  // Check if the constant already exists in the pool.
  auto it = _constants.find(key);
  if (it != _constants.end()) {
    return *(it->second);
  }

  // Create a new constant StorageView.
  StorageView* constant = new StorageView(shape, init, device);
  _constants[key] = std::unique_ptr<StorageView>(constant);
  return *constant;
}

template <typename T>
const StorageView& ConstantPool::get_constant(Shape shape,
                                              const std::vector<T>& init,
                                              ctranslate2::Device device) {
  return get_device()->GetConstantPool()->_get_constant(shape, init, device);
}

template const StorageView& ConstantPool::get_constant<float>(
    Shape shape,
    const std::vector<float>& init,
    ctranslate2::Device device);

template const StorageView& ConstantPool::get_constant<int8_t>(
    Shape shape,
    const std::vector<int8_t>& init,
    ctranslate2::Device device);
}  // namespace dml
}  // namespace ctranslate2