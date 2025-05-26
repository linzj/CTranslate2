#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/utils.h"
#include <spdlog/spdlog.h>
#include <windows.h> // Required for LoadLibraryW and GetProcAddress

// DirectML specific headers
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h> // For Microsoft::WRL::ComPtr

// Helper for COM error checking
#define DML_CHECK(expr)                                                        \
  do {                                                                         \
    HRESULT hr = (expr);                                                       \
    if (FAILED(hr))                                                            \
      THROW_RUNTIME_ERROR(#expr " failed with HRESULT: " +                     \
                          std::to_string(hr));                                 \
  } while (0)

using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
namespace dml {

// Function pointers for dynamic loading
typedef HRESULT(WINAPI *PFN_CREATE_DXGI_FACTORY2)(
    UINT Flags, REFIID riid, _COM_Outptr_ void **ppFactory);
typedef HRESULT(WINAPI *PFN_D3D12_CREATE_DEVICE)(
    _In_opt_ IUnknown *pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel,
    REFIID riid, _COM_Outptr_ void **ppDevice);
typedef HRESULT(WINAPI *PFN_DML_CREATE_DEVICE)(
    _In_ ID3D12Device *d3d12Device, DML_CREATE_DEVICE_FLAGS flags, REFIID riid,
    _COM_Outptr_ IDMLDevice **ppvDevice);

// Global DML resources
static HMODULE g_h_dxgi_dll = nullptr;
static HMODULE g_h_d3d12_dll = nullptr;
static HMODULE g_h_directml_dll = nullptr;

static PFN_CREATE_DXGI_FACTORY2 g_pfn_CreateDXGIFactory2 = nullptr;
static PFN_D3D12_CREATE_DEVICE g_pfn_D3D12CreateDevice = nullptr;
static PFN_DML_CREATE_DEVICE g_pfn_DMLCreateDevice = nullptr;

static ComPtr<ID3D12Device> g_d3d12_device;
static ComPtr<IDMLDevice> g_dml_device;
static ComPtr<ID3D12CommandQueue> g_command_queue;

bool has_directml_device() {
  // Load DLLs and get function pointers
  g_h_dxgi_dll = LoadLibraryW(L"dxgi.dll");
  if (!g_h_dxgi_dll) {
    SPDLOG_WARN("Failed to load dxgi.dll");
    return false;
  }
  g_pfn_CreateDXGIFactory2 =
      reinterpret_cast<PFN_CREATE_DXGI_FACTORY2>(reinterpret_cast<void *>(
          GetProcAddress(g_h_dxgi_dll, "CreateDXGIFactory2")));
  if (!g_pfn_CreateDXGIFactory2) {
    SPDLOG_WARN("Failed to get CreateDXGIFactory2 address");
    return false;
  }

  g_h_d3d12_dll = LoadLibraryW(L"d3d12.dll");
  if (!g_h_d3d12_dll) {
    SPDLOG_WARN("Failed to load d3d12.dll");
    return false;
  }
  g_pfn_D3D12CreateDevice =
      reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(reinterpret_cast<void *>(
          GetProcAddress(g_h_d3d12_dll, "D3D12CreateDevice")));
  if (!g_pfn_D3D12CreateDevice) {
    SPDLOG_WARN("Failed to get D3D12CreateDevice address");
    return false;
  }

  g_h_directml_dll = LoadLibraryW(L"directml.dll");
  if (!g_h_directml_dll) {
    SPDLOG_WARN("Failed to load directml.dll");
    return false;
  }
  g_pfn_DMLCreateDevice =
      reinterpret_cast<PFN_DML_CREATE_DEVICE>(reinterpret_cast<void *>(
          GetProcAddress(g_h_directml_dll, "DMLCreateDevice")));
  if (!g_pfn_DMLCreateDevice) {
    SPDLOG_WARN("Failed to get DMLCreateDevice address");
    return false;
  }

  // Try to create a DXGI factory
  ComPtr<IDXGIFactory4> factory;
  if (FAILED(g_pfn_CreateDXGIFactory2(0, __uuidof(IDXGIFactory4),
                                      (void **)(factory.GetAddressOf())))) {
    return false;
  }

  // Try to find a compatible adapter
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT adapter_idx = 0;
       DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapter_idx, &adapter);
       ++adapter_idx) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      // Don't use the software adapter
      continue;
    }

    // Check if D3D12 device can be created on this adapter
    if (SUCCEEDED(g_pfn_D3D12CreateDevice(
            adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device),
            (void **)(g_d3d12_device.GetAddressOf())))) {
      // Check if DML device can be created on this D3D12 device
      if (SUCCEEDED(g_pfn_DMLCreateDevice(
              g_d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
              __uuidof(IDMLDevice), g_dml_device.GetAddressOf()))) {
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

  SPDLOG_INFO("DirectML backend initialized on device: {}",
              static_cast<void *>(g_d3d12_device.Get()));

  // Create command queue for D3D12 device
  D3D12_COMMAND_QUEUE_DESC queue_desc = {};
  queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  DML_CHECK(g_d3d12_device->CreateCommandQueue(&queue_desc,
                                               IID_PPV_ARGS(&g_command_queue)));
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

  // Free loaded DLLs
  if (g_h_directml_dll) {
    FreeLibrary(g_h_directml_dll);
    g_h_directml_dll = nullptr;
  }
  if (g_h_d3d12_dll) {
    FreeLibrary(g_h_d3d12_dll);
    g_h_d3d12_dll = nullptr;
  }
  if (g_h_dxgi_dll) {
    FreeLibrary(g_h_dxgi_dll);
    g_h_dxgi_dll = nullptr;
  }

  SPDLOG_INFO("DirectML backend released.");
}

ID3D12Device *get_d3d12_device() { return g_d3d12_device.Get(); }

IDMLDevice *get_dml_device() { return g_dml_device.Get(); }

ID3D12CommandQueue *get_command_queue() { return g_command_queue.Get(); }

} // namespace dml
} // namespace ctranslate2

#endif // CT2_WITH_DIRECTML