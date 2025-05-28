#pragma once

#include "dxmodule.h"

namespace ctranslate2 {
namespace dml {
struct GpuEvent {
  uint64_t fenceValue;
  Microsoft::WRL::ComPtr<ID3D12Fence> fence;

  bool IsSignaled() const { return fence->GetCompletedValue() >= fenceValue; }

  // Blocks until IsSignaled returns true.
  void WaitForSignal(bool cpuSyncSpinningEnabled = false) const {
    if (IsSignaled())
      return;  // early-out

    (void)cpuSyncSpinningEnabled;
    fence->SetEventOnCompletion(fenceValue, nullptr);
  }
};
}  // namespace dml
}  // namespace ctranslate2