#pragma once
#include <optional>
#include <string>

#if defined(_WIN32)
constexpr const char* c_directmlModuleName = "directml.dll";
constexpr const char* c_direct3dModuleName = "d3d12.dll";
constexpr const char* c_direct3dCoreModuleName = "d3d12core.dll";
constexpr const char* c_dxcoreModuleName = "dxcore.dll";
constexpr const char* c_dxcompilerModuleName = "dxcompiler.dll";
constexpr const char* c_pixModuleName = "winpixeventruntime.dll";
constexpr const char* c_ortModuleName = "onnxruntime.dll";
constexpr const char* c_ortExtensionsModuleName = "ortextensions.dll";
#else
constexpr const char* c_directmlModuleName = "libdirectml.so";
constexpr const char* c_direct3dModuleName = "libd3d12.so";
constexpr const char* c_direct3dCoreModuleName = c_direct3dModuleName;
constexpr const char* c_dxcoreModuleName = "libdxcore.so";
constexpr const char* c_dxcompilerModuleName = nullptr;
constexpr const char* c_pixModuleName = nullptr;
constexpr const char* c_ortModuleName = nullptr;
constexpr const char* c_ortExtensionsModuleName = nullptr;
#endif

struct ModuleInfo {
  std::wstring path;
  std::wstring version;
};

std::optional<ModuleInfo> GetModuleInfo(std::string_view moduleName);

void PrintDependencies();
