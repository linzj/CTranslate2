#ifdef CT2_WITH_DIRECTML

#include "allocator_dml.h"
#include "backend_dml.h"
#include "ctranslate2/devices.h"
#include "resource_wrapper.h"

#include <spdlog/spdlog.h>

#include <d3d12.h>
#include <wrl/client.h>

namespace ctranslate2 {
namespace dml {
DMLAllocator::DMLAllocator() {}

DMLAllocator::~DMLAllocator() = default;

void* DMLAllocator::allocate(size_t size, int device_index) {
  if (size == 0)
    return nullptr;

  return dml::get_device()->CreatePreferredDeviceMemoryBuffer(size).Detach();
}

void DMLAllocator::free(void* ptr, int device_index) {
  if (ptr) {
    static_cast<ID3D12Resource*>(ptr)->Release();
  }
}
}  // namespace dml

template <>
Allocator& get_allocator<Device::DirectML>() {
  static dml::DMLAllocator allocator;
  return allocator;
}

}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML