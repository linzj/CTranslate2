#ifdef CT2_WITH_DIRECTML

#include <spdlog/spdlog.h>
#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/utils.h"

#include <DirectML.h>
#include <d3d12.h>
#include <wrl/client.h>  // For Microsoft::WRL::ComPtr

// Helper for COM error checking - defined also in backend_dml.cc
#ifndef DML_CHECK
#define DML_CHECK(expr)                                    \
  do {                                                     \
    HRESULT hr = (expr);                                   \
    if (FAILED(hr))                                        \
      THROW_RUNTIME_ERROR(#expr " failed with HRESULT: " + \
                          std::to_string(hr));             \
  } while (0)
#endif

namespace ctranslate2 {
namespace dml {

// Forward declarations from backend_dml.cc to get device resources
ID3D12Device* get_d3d12_device();
IDMLDevice* get_dml_device();
ID3D12CommandQueue* get_command_queue();

class DMLAllocator : public Allocator {
 public:
  DMLAllocator() {
    // No explicit initialization needed here beyond base class for now.
    // DML devices are initialized globally in backend_dml.cc via
    // std::call_once.
  }

  ~DMLAllocator() override {
    // Allocated resources are managed by ComPtr, which automatically releases.
    // No explicit deallocation for individual resources is typically needed
    // here.
  }

  void* allocate(size_t size, int device_index) override {
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
    DML_CHECK(get_d3d12_device()->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON,  // Initial state, will be transitioned as
                                      // needed
        nullptr, IID_PPV_ARGS(&dml_buffer)));

    // Note: For actual DML operations, these resources will likely need
    // to be placed on a specific DML_BUFFER_BINDING to bind them to operators.
    // The pointer returned here is just the ID3D12Resource*.
    return dml_buffer;
  }

  void free(void* ptr, int device_index) override {
    if (ptr) {
      static_cast<ID3D12Resource*>(ptr)->Release();
    }
  }
};
}  // namespace dml

template <>
Allocator& get_allocator<Device::DirectML>() {
  static dml::DMLAllocator allocator;
  return allocator;
}

}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML