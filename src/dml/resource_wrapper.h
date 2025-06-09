#pragma once

#include <combaseapi.h>

typedef interface ID3D12Resource ID3D12Resource;

interface __declspec(uuid(
    "d430f6f1-5c43-48d1-97e6-f080cc7fa0c5")) IResourceWrapper
    : public IUnknown {
 public:
  virtual ID3D12Resource* GetD3D12Resource() const = 0;
};