#include "dxmodule.h"
#include <mutex>

#ifndef WIN32
#include <dlfcn.h>
#endif
using namespace Microsoft::WRL;

std::string g_moduleDir;
std::mutex g_moduleDirMutex;

#if !defined(_GAMING_XBOX) && defined(WIN32) && 0
extern "C" {
__declspec(dllexport) extern UINT D3D12SDKVersion =
    DIRECT3D_AGILITY_SDK_VERSION;
}
extern "C" {
__declspec(dllexport) extern const char* D3D12SDKPath = u8"./D3D12/";
}
#endif

Module::Module(const char* moduleName) {
  if (!moduleName) {
    return;
  }

#ifdef WIN32
  std::string fullPath;
  {
    std::lock_guard<std::mutex> lock(g_moduleDirMutex);
    fullPath = g_moduleDir;
  }
  if (!fullPath.empty() && fullPath.back() != '\\') {
    fullPath += '\\';
  }
  fullPath += moduleName;

  int len = MultiByteToWideChar(CP_UTF8, 0, fullPath.c_str(), -1, nullptr, 0);
  if (len == 0) {
    return;
  }
  std::wstring wideFullPath(len - 1, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, fullPath.c_str(), -1, &wideFullPath[0], len - 1);

  m_module = LoadLibraryW(wideFullPath.c_str());
#else
  m_module = dlopen(moduleName, RTLD_LAZY);
#endif
}

Module::Module(void* handle) {
#ifdef WIN32
  if (m_module) {
    FreeLibrary(m_module);
  }
  m_module = reinterpret_cast<HMODULE>(handle);
#else
  m_module = handle;
#endif
}

Module::Module(Module&& other) {
  if (m_module) {
    FreeLibrary(m_module);
  }
  m_module = std::move(other.m_module);
  other.m_module = nullptr;
}

Module& Module::operator=(Module&& other) {
  if (m_module) {
    FreeLibrary(m_module);
  }
  m_module = std::move(other.m_module);
  other.m_module = nullptr;
  return *this;
}

Module::~Module() {
  if (m_module) {
#ifndef WIN32
    dlclose(m_module);
#else
    if (m_module) {
      FreeLibrary(m_module);
    }
#endif
    m_module = nullptr;
  }
}

void* Module::GetSymbol(const char* name) {
#ifdef WIN32
  return (void*)GetProcAddress(m_module, name);
#else
  return dlsym(m_module, name);
#endif
}

D3d12Module::D3d12Module(bool disableAgilitySDK, const char* moduleName)
    : Module(moduleName) {
  if (m_module) {
#if !defined(_GAMING_XBOX) && defined(WIN32) && 0
    if (!disableAgilitySDK) {
      InitSymbol(&m_d3d12SDKConfiguration, "D3D12GetInterface");
      if (m_d3d12SDKConfiguration) {
        ComPtr<ID3D12SDKConfiguration1> pD3D12SDKConfiguration;
        THROW_IF_FAILED(
            m_d3d12SDKConfiguration(CLSID_D3D12SDKConfiguration,
                                    IID_PPV_ARGS(&pD3D12SDKConfiguration)));

        ComPtr<ID3D12DeviceFactory> deviceFactory;
        THROW_IF_FAILED(pD3D12SDKConfiguration->CreateDeviceFactory(
            D3D12SDKVersion, D3D12SDKPath, IID_PPV_ARGS(&deviceFactory)));
        THROW_IF_FAILED(deviceFactory->ApplyToGlobalState());
      }
    }
#endif

    InitSymbol(&m_d3d12CreateDevice, "D3D12CreateDevice");
    InitSymbol(&m_d3d12GetDebugInterface, "D3D12GetDebugInterface");
    InitSymbol(&m_d3d12SerializeVersionedRootSignature,
               "D3D12SerializeVersionedRootSignature");
  }
}

DxCoreModule::DxCoreModule(const char* moduleName) : Module(moduleName) {
  if (m_module) {
    InitSymbol(&m_dxCoreCreateAdapterFactory, "DXCoreCreateAdapterFactory");
  }
}

DmlModule::DmlModule(const char* moduleName) : Module(moduleName) {
  if (m_module) {
    InitSymbol(&m_dmlCreateDevice1, "DMLCreateDevice1");
  }
}
