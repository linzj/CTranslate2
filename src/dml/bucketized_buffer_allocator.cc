#include "bucketized_buffer_allocator.h"
#include "dxdevice.h"

#include "resource_wrapper.h"

#include <wrl/implements.h>

namespace ctranslate2 {
namespace dml {
namespace {
constexpr bool kForceDisablePooling = false;
class AllocationInfo
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
          IResourceWrapper> {
 public:
  AllocationInfo(BucketizedBufferAllocator* owner,
                 size_t id,
                 ID3D12Resource* resource,
                 size_t requestedSize)
      : m_owner(owner),
        m_bucketId(id),
        m_resource(resource),
        m_requestedSize(requestedSize) {}

  ~AllocationInfo();

  BucketizedBufferAllocator* GetOwner() const { return m_owner; }

  Microsoft::WRL::ComPtr<ID3D12Resource> DetachResource() {
    return std::move(m_resource);
  }

  size_t GetRequestedSize() const { return m_requestedSize; }

  size_t GetBucketId() const { return m_bucketId; }

  ID3D12Resource* GetD3D12Resource() const override { return m_resource.Get(); }

 private:
  // The bucketized buffer allocator must outlive the allocation info
  BucketizedBufferAllocator* m_owner;
  size_t m_bucketId;
  Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;

  // The size requested during Alloc(), which may be smaller than the physical
  // resource size
  size_t m_requestedSize;
};

AllocationInfo::~AllocationInfo() {
  if (m_owner) {
    m_owner->FreeResource(this);
  }
}
}  // namespace

BucketizedBufferAllocator::BucketizedBufferAllocator(Device* device)
    : m_device(device) {}

BucketizedBufferAllocator::~BucketizedBufferAllocator() = default;

Microsoft::WRL::ComPtr<IResourceWrapper> BucketizedBufferAllocator::Alloc(
    uint64_t size,
    D3D12_RESOURCE_FLAGS resourceFlags) {
  constexpr uint64_t tensor_size_mask =
      ~(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT - 1);
  size = (size + DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT - 1) & tensor_size_mask;

  uint32_t bucketIndex = GetBucketIndexFromSize(size);
  uint64_t bucketSize = GetBucketSizeFromIndex(bucketIndex);

  bool isPooled =
      (m_roundingEnabled || (size == bucketSize)) && !kForceDisablePooling;

  if (isPooled) {
    if (bucketIndex >= m_pool.size()) {
      m_pool.resize(bucketIndex + 1);
    }

    Bucket& bucket = m_pool[bucketIndex];

    if (!bucket.empty()) {
      Microsoft::WRL::ComPtr<ID3D12Resource> resource =
          std::move(bucket.back());
      bucket.pop_back();
      Microsoft::WRL::ComPtr<IResourceWrapper> resourceWrapper =
          Microsoft::WRL::Make<AllocationInfo>(this, bucketIndex,
                                               resource.Detach(), size);
      return resourceWrapper;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource> resource =
        m_device->CreatePreferredDeviceMemoryBufferWithoutPooling(
            bucketSize, resourceFlags);

    Microsoft::WRL::ComPtr<IResourceWrapper> resourceWrapper =
        Microsoft::WRL::Make<AllocationInfo>(this, bucketIndex,
                                             resource.Detach(), size);

    return resourceWrapper;
  } else {
    Microsoft::WRL::ComPtr<ID3D12Resource> d3d12_resource =
        m_device->CreatePreferredDeviceMemoryBufferWithoutPooling(
            size, resourceFlags);
    Microsoft::WRL::ComPtr<IResourceWrapper> resourceWrapper =
        Microsoft::WRL::Make<AllocationInfo>(this, bucketIndex,
                                             d3d12_resource.Detach(), size);
    return resourceWrapper;
  }
}

void BucketizedBufferAllocator::FreeResource(
    IResourceWrapper* resourceWrapper) {
  if (!resourceWrapper) {
    return;
  }
  bool isPooled = m_roundingEnabled && !kForceDisablePooling;
  if (!isPooled)
    return;

  AllocationInfo* allocationInfo =
      static_cast<AllocationInfo*>(resourceWrapper);

  uint32_t bucketIndex = allocationInfo->GetBucketId();
  if (bucketIndex >= m_pool.size()) {
    m_pool.resize(bucketIndex + 1);
  }

  Bucket& bucket = m_pool[bucketIndex];
  bucket.push_back(Microsoft::WRL::ComPtr<ID3D12Resource>(
      resourceWrapper->GetD3D12Resource()));
}

uint32_t BucketizedBufferAllocator::GetBucketIndexFromSize(
    uint64_t size) const {
  if (size == 0) {
    return 0;
  }
  uint64_t minBucketSize = 1ull << c_minResourceSizeExponent;
  if (size <= minBucketSize) {
    return 0;
  }

  // std::log2 returns the base-2 logarithm.
  // The smallest power of 2 greater than or equal to a value is
  // 2^(ceil(log2(value))). Example: size=90KB. log2(90KB)=16.49. ceil=17.
  // 2^17=128KB.
  uint32_t power =
      static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(size))));

  // The bucket index is the delta from the minimum resource size exponent.
  // Example: min_exp=16. power=17. 17-16=1. Index=1.
  return power - c_minResourceSizeExponent;
}

uint64_t BucketizedBufferAllocator::GetBucketSizeFromIndex(
    uint32_t index) const {
  return 1ull << (static_cast<uint64_t>(index) + c_minResourceSizeExponent);
}

}  // namespace dml
}  // namespace ctranslate2