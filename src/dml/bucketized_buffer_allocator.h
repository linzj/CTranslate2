#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>

typedef interface IResourceWrapper IResourceWrapper;

namespace ctranslate2 {
namespace dml {

class Device;
// Implements a memory pooling strategy for D3D12 resources.
class BucketizedBufferAllocator {
 public:
  BucketizedBufferAllocator(Device* device);
  ~BucketizedBufferAllocator();

  Microsoft::WRL::ComPtr<IResourceWrapper> Alloc(
      uint64_t size,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

  void FreeResource(IResourceWrapper*);

  void EnableRounding(bool enabled) { m_roundingEnabled = enabled; }

 private:
  using Bucket = std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>;

  uint32_t GetBucketIndexFromSize(uint64_t size) const;
  uint64_t GetBucketSizeFromIndex(uint32_t index) const;

  Device* m_device;
  std::vector<Bucket> m_pool;

  static constexpr uint32_t c_minResourceSizeExponent = 16;  // 2^16 = 64KB

  bool m_roundingEnabled = true;
};

}  // namespace dml
}  // namespace ctranslate2