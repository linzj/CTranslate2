#ifdef CT2_WITH_DIRECTML

// Moved dxmodule.h and dxdevice.h to be before backend_dml.h
#include "common.h"
#include "dxdevice.h"  // Defines ctranslate2::dml::Device
#include "dxmodule.h"  // For D3d12Module and DmlModule


#include <spdlog/spdlog.h>
#include "backend_dml.h"  // Self header
#include "ctranslate2/utils.h"

// DirectX/DirectML headers
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <memory>
#include <string>

using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
namespace dml {

// Global Device object and Modules
static std::unique_ptr<Device> g_device;
static std::shared_ptr<D3d12Module> g_d3d12_module;
static std::shared_ptr<DmlModule> g_dml_module;

// For DXGI functions loaded dynamically
static HMODULE g_h_dxgi_dll = nullptr;
typedef HRESULT(WINAPI* PFN_CREATE_DXGI_FACTORY2)(
    UINT Flags,
    REFIID riid,
    _COM_Outptr_ void** ppFactory);
static PFN_CREATE_DXGI_FACTORY2 g_pfn_CreateDXGIFactory2 = nullptr;

// Helper to convert WCHAR array to std::string
std::string to_string(const WCHAR* wstr) {
  if (!wstr)
    return "";
  int size_needed =
      WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
  if (size_needed == 0)
    return "";
  std::string strTo(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &strTo[0], size_needed, NULL, NULL);
  // Remove null terminator if WideCharToMultiByte includes it
  if (!strTo.empty() && strTo.back() == '\0') {
    strTo.pop_back();
  }
  return strTo;
}

bool has_directml_device() {
  if (g_device) {
    return true;
  }

  try {
    // 1. Instantiate D3D12 and DML modules
    g_d3d12_module =
        std::make_shared<D3d12Module>(false);  // false = disableAgilitySDK
    g_dml_module = std::make_shared<DmlModule>();

    if (!g_d3d12_module->GetHandle() || !g_dml_module->GetHandle()) {
      SPDLOG_WARN("Failed to load D3D12 or DML module.");
      return false;
    }

    // 2. Load DXGI library and GetProcAddress for CreateDXGIFactory2
    if (!g_pfn_CreateDXGIFactory2) {
      g_h_dxgi_dll = LoadLibraryW(L"dxgi.dll");
      if (!g_h_dxgi_dll) {
        SPDLOG_WARN("Failed to load dxgi.dll");
        return false;
      }
      g_pfn_CreateDXGIFactory2 =
          reinterpret_cast<PFN_CREATE_DXGI_FACTORY2>(reinterpret_cast<void*>(
              GetProcAddress(g_h_dxgi_dll, "CreateDXGIFactory2")));
      if (!g_pfn_CreateDXGIFactory2) {
        SPDLOG_WARN("Failed to get CreateDXGIFactory2 address from dxgi.dll");
        FreeLibrary(g_h_dxgi_dll);
        g_h_dxgi_dll = nullptr;
        return false;
      }
    }

    // 3. Create DXGI Factory
    ComPtr<IDXGIFactory4> factory;
    THROW_IF_FAILED(
        g_pfn_CreateDXGIFactory2(0, IID_PPV_ARGS(factory.GetAddressOf())));

    // 4. Enumerate Adapters
    ComPtr<IDXGIAdapter1> selected_adapter;
    for (UINT adapter_idx = 0;
         factory->EnumAdapters1(adapter_idx,
                                selected_adapter.ReleaseAndGetAddressOf()) !=
         DXGI_ERROR_NOT_FOUND;
         ++adapter_idx) {
      DXGI_ADAPTER_DESC1 desc;
      THROW_IF_FAILED(selected_adapter->GetDesc1(&desc));

      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        continue;  // Skip software adapter
      }

      // Basic check: Can a D3D12 device be tentatively created on this adapter?
      // The Device constructor will do the actual full D3D and DML device
      // creation. Here we're just checking adapter viability before
      // constructing the main Device object.
      ComPtr<ID3D12Device> temp_d3d_device;
      HRESULT hr_check_d3d = g_d3d12_module->CreateDevice(
          selected_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
          __uuidof(ID3D12Device),
          nullptr);  // Pass nullptr for ppDevice to just check support

      if (SUCCEEDED(hr_check_d3d)) {
        // Found a suitable hardware adapter, break and use this one
        break;
      }
      selected_adapter.Reset();  // Try next adapter
    }

    if (!selected_adapter) {
      SPDLOG_WARN("No suitable D3D12 capable hardware adapter found.");
      return false;
    }

    // 5. Create the main Device object
    // Sensible defaults for Device constructor parameters.
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    DML_FEATURE_LEVEL dmlFeatureLevel = DML_FEATURE_LEVEL_2_0;

    g_device = std::make_unique<Device>(
        selected_adapter.Get(),  // The chosen hardware adapter
        featureLevel, dmlFeatureLevel,
        false,                           // debugLayersEnabled
        D3D12_COMMAND_LIST_TYPE_DIRECT,  // commandListType
        1,                               // dispatchRepeat
        true,                            // uavBarrierAfterDispatch
        false,                           // aliasingBarrierAfterDispatch
        false,                           // clearShaderCaches
        false,                           // disableGpuTimeout
        false,                           // enableDred
        false,                           // disableBackgroundProcessing
        false,                           // setStablePowerState
        false,  // preferCustomHeaps (false means use default behavior which
                // might use custom if available and preferred by device.h
                // logic)
        false,  // usePresentSeparator
        0,      // maxGpuTimeMeasurements
        g_d3d12_module, g_dml_module);

    // 6. Check if device construction and internal D3D/DML objects are valid
    if (!g_device || !g_device->D3D() || !g_device->DML()) {
      SPDLOG_WARN(
          "Failed to create Device wrapper or internal D3D/DML objects.");
      g_device.reset();  // Ensure it's cleaned up if partially formed
      return false;
    }

    return true;

  } catch (const std::exception& e) {
    SPDLOG_WARN("Exception during DirectML device initialization: {}",
                e.what());
    g_device.reset();
    // Modules and dxgi.dll are cleaned up in release_directml or by shared_ptr
    return false;
  }
  // Should not be reached if all paths return explicitly
  return false;
}

void initialize_directml() {
  if (!g_device && !has_directml_device()) {
    THROW_RUNTIME_ERROR("DirectML device not found or initialization failed.");
  }
  SPDLOG_INFO("DirectML backend initialized using Device wrapper.");
}

void release_directml() {
  if (g_device) {
    g_device.reset();  // Device destructor handles its D3D/DML resources.
  }

  g_dml_module
      .reset();  // Release shared_ptr, actual module unloads if ref count is 0.
  g_d3d12_module.reset();

  if (g_h_dxgi_dll) {
    FreeLibrary(g_h_dxgi_dll);
    g_h_dxgi_dll = nullptr;
    g_pfn_CreateDXGIFactory2 = nullptr;
  }

  SPDLOG_INFO("DirectML backend released.");
}

Device* get_device() {
  if (!g_device) {  // Attempt to initialize if not already
    has_directml_device();
  }
  return g_device.get();
}

IDMLDevice1* get_dml_device() {
  return g_device->DML();
}
}  // namespace dml
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML