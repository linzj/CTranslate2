#ifdef CT2_WITH_DIRECTML

#include "allocator_dml.h"
#include "backend_dml.h"
#include "common.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/utils.h"

#include <spdlog/spdlog.h>

#include <d3d12.h>
#include <wrl/client.h>

namespace ctranslate2 {
namespace dml {

void* DMLAllocator::allocate(size_t size, int device_index) {
  if (size == 0)
    return nullptr;

  D3D12_HEAP_PROPERTIES heap_props = {};
  heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_props.CreationNodeMask = 1;
  heap_props.VisibleNodeMask = 1;

  D3D12_RESOURCE_FLAGS res_flags = D3D12_RESOURCE_FLAG_NONE;

  heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
  res_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;  // Required for DML

  D3D12_RESOURCE_DESC desc = {};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Alignment = 0;  // Default alignment
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_UNKNOWN;  // For buffers
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = res_flags;

  ID3D12Resource* dml_buffer = nullptr;
  auto dxdevice = get_device();
  auto d3ddevice = dxdevice->D3D();

  THROW_IF_FAILED(d3ddevice->CreateCommittedResource(
      &heap_props, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_COMMON,  // Initial state, will be transitioned as
                                    // needed
      nullptr, IID_PPV_ARGS(&dml_buffer)));

  // Note: For actual DML operations, these resources will likely need
  // to be placed on a specific DML_BUFFER_BINDING to bind them to operators.
  // The pointer returned here is just the ID3D12Resource*.
  return dml_buffer;
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