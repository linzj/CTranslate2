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

namespace ctranslate2 {
namespace dml {
Module::Module(const char* moduleName) : m_module(nullptr) {
  if (!moduleName || !*moduleName) {  // Check for nullptr or empty string
    return;
  }

#ifdef WIN32
  // Convert moduleName to std::wstring once for reuse, as WinAPI functions
  // generally use wide strings. moduleName is char*, assumed to be
  std::wstring wideModuleName;
  int len_for_wide_module_name =
      MultiByteToWideChar(CP_ACP, 0, moduleName, -1, nullptr, 0);
  if (len_for_wide_module_name == 0) {
    // Error during conversion measurement or moduleName is effectively
    // empty/invalid.
    return;  // m_module remains nullptr
  }
  // Create std::wstring with space for characters (len includes null
  // terminator).
  wideModuleName.assign(len_for_wide_module_name - 1, L'\0');
  if (MultiByteToWideChar(CP_ACP, 0, moduleName, -1, &wideModuleName[0],
                          len_for_wide_module_name) == 0) {
    // Actual conversion failed.
    return;  // m_module remains nullptr
  }

  // Attempt 1: Relative to current module's directory
  HMODULE hCurrentModule = GetModuleHandleW(nullptr);
  if (hCurrentModule) {
    wchar_t currentModulePathCStr[MAX_PATH] = {0};
    // GetModuleFileNameW returns 0 on failure, or length of string (excluding
    // null) on success.
    if (GetModuleFileNameW(hCurrentModule, currentModulePathCStr, MAX_PATH) >
        0) {
      std::wstring currentModulePath(currentModulePathCStr);
      size_t lastSlashPos = currentModulePath.find_last_of(L"\\/");
      if (lastSlashPos != std::wstring::npos) {
        std::wstring currentModuleDir =
            currentModulePath.substr(0, lastSlashPos);
        std::wstring pathAttempt1 = currentModuleDir + L"\\" + wideModuleName;
        m_module = LoadLibraryW(pathAttempt1.c_str());
        if (m_module) {
          return;
        }
      }
    }
  }

  // Attempt 2: Relative to current working directory
  // No need to check m_module here as it's checked before return in previous
  // block.
  wchar_t currentWorkingDirCStr[MAX_PATH] = {0};
  // GetCurrentDirectoryW returns 0 on failure, or length of string (excluding
  // null) on success.
  if (GetCurrentDirectoryW(MAX_PATH, currentWorkingDirCStr) > 0) {
    std::wstring currentWorkingDir(currentWorkingDirCStr);
    std::wstring pathAttempt2 = currentWorkingDir + L"\\" + wideModuleName;
    m_module = LoadLibraryW(pathAttempt2.c_str());
    if (m_module) {
      return;
    }
  }

  // Attempt 3: Fallback to original logic (using g_moduleDir or just
  // moduleName) This re-implements the logic that was originally in lines
  // 30-48.
  std::string originalLogicFullPathStr;  // This will hold char* path
  {
    std::lock_guard<std::mutex> lock(g_moduleDirMutex);
    originalLogicFullPathStr = g_moduleDir;  // g_moduleDir is std::string
  }

  if (!originalLogicFullPathStr.empty()) {  // If g_moduleDir was set
    // Append separator if needed (defensive against double separators if
    // g_moduleDir already ends with one)
    if (originalLogicFullPathStr.back() != '\\' &&
        originalLogicFullPathStr.back() != '/') {
      originalLogicFullPathStr += '\\';
    }
    originalLogicFullPathStr +=
        moduleName;  // Append the original char* moduleName
  } else {
    // If g_moduleDir was empty, the path to try is just moduleName itself.
    originalLogicFullPathStr = moduleName;
  }

  // Convert this potentially combined path to wide string and try loading
  std::wstring wideOriginalLogicFullPath;
  int len_orig_logic = MultiByteToWideChar(
      CP_ACP, 0, originalLogicFullPathStr.c_str(), -1, nullptr, 0);
  if (len_orig_logic > 0) {
    wideOriginalLogicFullPath.assign(len_orig_logic - 1, L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, originalLogicFullPathStr.c_str(), -1,
                            &wideOriginalLogicFullPath[0],
                            len_orig_logic) > 0) {
      m_module = LoadLibraryW(wideOriginalLogicFullPath.c_str());
      // If m_module is loaded, constructor will naturally proceed. If not,
      // m_module remains nullptr.
    }
  }
  // No more attempts after this. If all failed, m_module is still nullptr.
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

}  // namespace dml
}  // namespace ctranslate2