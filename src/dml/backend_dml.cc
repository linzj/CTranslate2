#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/devices.h"
#include "ctranslate2/utils.h"
#include "ctranslate2/log.h"

// DirectML specific headers
#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectML.h>
#include <wrl/client.h> // For Microsoft::WRL::ComPtr

// Helper for COM error checking
#define DML_CHECK(expr)                                                       \
  do {                                                                        \
    HRESULT hr = (expr);                                                      \
    if (FAILED(hr))                                                           \
      THROW_RUNTIME_ERROR(#expr " failed with HRESULT: " + std::to_string(hr)); \
  } while (0)

using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
  namespace dml {

    // Global DML resources
    static ComPtr<ID3D12Device> g_d3d12_device;
    static ComPtr<IDMLDevice> g_dml_device;
    static ComPtr<ID3D12CommandQueue> g_command_queue;

    bool has_directml_device() {
      // Try to create a DXGI factory
      ComPtr<IDXGIFactory4> factory;
      if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        return false;
      }

      // Try to find a compatible adapter
      ComPtr<IDXGIAdapter1> adapter;
      for (UINT adapter_idx = 0; DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapter_idx, &adapter); ++adapter_idx) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
          // Don't use the software adapter
          continue;
        }

        // Check if D3D12 device can be created on this adapter
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_d3d12_device)))) {
          // Check if DML device can be created on this D3D12 device
          if (SUCCEEDED(DMLCreateDevice(g_d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&g_dml_device)))) {
            // Found a valid DirectML device
            return true;
          }
        }
      }
      return false; // No DirectML capable device found
    }

    void initialize_directml() {
      if (!has_directml_device()) {
        THROW_RUNTIME_ERROR("DirectML device not found or initialization failed.");
      }

      LOG_INFO("DirectML backend initialized on device: {}", static_cast<void*>(g_d3d12_device.Get()));

      // Create command queue for D3D12 device
      D3D12_COMMAND_QUEUE_DESC queue_desc = {};
      queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
      queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
      DML_CHECK(g_d3d12_device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&g_command_queue)));
    }

    void release_directml() {
      // Release DirectML and D3D12 resources
      if (g_command_queue)
        g_command_queue->Release();
      if (g_dml_device)
        g_dml_device->Release();
      if (g_d3d12_device)
        g_d3d12_device->Release();

      g_command_queue.Reset();
      g_dml_device.Reset();
      g_d3d12_device.Reset();

      LOG_INFO("DirectML backend released.");
    }

:start_line:93
-------
    ID3D12Device* get_d3d12_device() {
      return g_d3d12_device.Get();
    }

    IDMLDevice* get_dml_device() {
      return g_dml_device.Get();
    }

    ID3D12CommandQueue* get_command_queue() {
      return g_command_queue.Get();
    }

  } // namespace dml
} // namespace ctranslate2

#endif // CT2_WITH_DIRECTML