#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <functional>
#include <vector>

typedef interface IResourceWrapper IResourceWrapper;

namespace ctranslate2 {
namespace dml {

class Device;
// Implements a memory pooling strategy for D3D12 resources.
class BucketizedBufferAllocator {
 public:
  using AllocFunction = std::function<Microsoft::WRL::ComPtr<ID3D12Resource>(
      uint64_t size,
      D3D12_RESOURCE_FLAGS)>;
  explicit BucketizedBufferAllocator(AllocFunction&&);
  ~BucketizedBufferAllocator();

  Microsoft::WRL::ComPtr<IResourceWrapper> Alloc(
      uint64_t size,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  void FreeResource(IResourceWrapper*);

  void EnableRounding(bool enabled) { m_roundingEnabled = enabled; }

  void DisablePooling(bool disable) { m_disablePooling = disable; }

 private:
  using Bucket = std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>;

  uint32_t GetBucketIndexFromSize(uint64_t size) const;
  uint64_t GetBucketSizeFromIndex(uint32_t index) const;

  AllocFunction m_allocFunction;
  std::vector<Bucket> m_pool;

  static constexpr uint32_t c_minResourceSizeExponent = 4;  // 2^4 = 16

  bool m_roundingEnabled = true;
  bool m_disablePooling = false;
};

}  // namespace dml
}  // namespace ctranslate2