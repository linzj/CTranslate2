#include "bucketized_buffer_allocator.h"
#include "dxdevice.h"

#include "resource_wrapper.h"

#include <wrl/implements.h>

namespace ctranslate2 {
namespace dml {
namespace {
constexpr bool kForceDisablePooling = false;
static constexpr uint32_t kResourceSizeExponent = 4;  // 2^4 = 16

uint32_t GetBucketIndexFromSize(uint64_t size) {
  if (size == 0) {
    return 0;
  }
  uint64_t minBucketSize = 1ull << kResourceSizeExponent;
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
  return power - kResourceSizeExponent;
}

uint64_t GetBucketSizeFromIndex(uint32_t index) {
  return 1ull << (static_cast<uint64_t>(index) + kResourceSizeExponent);
}

class AllocationInfo
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
          IResourceWrapper> {
 public:
  AllocationInfo(BucketizedBufferAllocator* owner,
                 size_t id,
                 size_t requestedSize,
                 D3D12_RESOURCE_FLAGS resourceFlags)

      : m_owner(owner),
        m_bucketId(id),
        m_requestedSize(requestedSize),
        m_resourceFlags(resourceFlags) {}

  ~AllocationInfo();

  BucketizedBufferAllocator* GetOwner() const { return m_owner; }

  Microsoft::WRL::ComPtr<ID3D12Resource> DetachResource() {
    return std::move(m_resource);
  }

  UINT32 GetRequestedSize() const override { return m_requestedSize; }

  size_t GetBucketId() const { return m_bucketId; }

  ID3D12Resource* GetD3D12Resource() const override {
    if (!m_resource) {
      m_resource =
          m_owner->AllocPrivate(m_bucketId, m_requestedSize, m_resourceFlags);
    }
    return m_resource.Get();
  }

  Microsoft::WRL::ComPtr<ID3D12Resource> GetD3D12ResourceDirect() const {
    return std::move(m_resource);
  }

  UINT32 GetActualSize() const override {
    return static_cast<UINT32>(GetBucketSizeFromIndex(m_bucketId));
  }

 private:
  // The bucketized buffer allocator must outlive the allocation info
  BucketizedBufferAllocator* m_owner;
  size_t m_bucketId;
  mutable Microsoft::WRL::ComPtr<ID3D12Resource> m_resource;

  // The size requested during Alloc(), which may be smaller than the physical
  // resource size
  size_t m_requestedSize;
  D3D12_RESOURCE_FLAGS m_resourceFlags;
};

AllocationInfo::~AllocationInfo() {
  if (m_owner) {
    m_owner->FreeResource(this);
  }
}
}  // namespace

BucketizedBufferAllocator::BucketizedBufferAllocator(
    AllocFunction&& allocFunction)
    : m_allocFunction(std::move(allocFunction)) {}

BucketizedBufferAllocator::~BucketizedBufferAllocator() = default;

Microsoft::WRL::ComPtr<IResourceWrapper> BucketizedBufferAllocator::Alloc(
    uint64_t size,
    D3D12_RESOURCE_FLAGS resourceFlags) {
  constexpr uint64_t tensor_size_mask =
      ~(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT - 1);
  size = (size + DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT - 1) & tensor_size_mask;

  uint32_t bucketIndex = GetBucketIndexFromSize(size);

  Microsoft::WRL::ComPtr<IResourceWrapper> resourceWrapper =
      Microsoft::WRL::Make<AllocationInfo>(this, bucketIndex, size,
                                           resourceFlags);
  return resourceWrapper;
}

void BucketizedBufferAllocator::FreeResource(
    IResourceWrapper* resourceWrapper) {
  if (!resourceWrapper) {
    return;
  }
  bool isPooled =
      !m_disablePooling && m_roundingEnabled && !kForceDisablePooling;
  if (!isPooled)
    return;

  AllocationInfo* allocationInfo =
      static_cast<AllocationInfo*>(resourceWrapper);

  uint32_t bucketIndex = allocationInfo->GetBucketId();
  if (bucketIndex >= m_pool.size()) {
    m_pool.resize(bucketIndex + 1);
  }

  Bucket& bucket = m_pool[bucketIndex];
  Microsoft::WRL::ComPtr<ID3D12Resource> detached_resource(
      allocationInfo->GetD3D12ResourceDirect());
  if (detached_resource) {
    bucket.push_back(std::move(detached_resource));
  }
}

Microsoft::WRL::ComPtr<ID3D12Resource> BucketizedBufferAllocator::AllocPrivate(
    uint32_t bucketIndex,
    uint64_t requestedSize,
    D3D12_RESOURCE_FLAGS resourceFlags) {
  bool isPooled =
      !m_disablePooling && (m_roundingEnabled) && !kForceDisablePooling;

  uint64_t bucketSize = GetBucketSizeFromIndex(bucketIndex);
  if (isPooled) {
    if (bucketIndex >= m_pool.size()) {
      m_pool.resize(bucketIndex + 1);
    }

    Bucket& bucket = m_pool[bucketIndex];

    if (!bucket.empty()) {
      Microsoft::WRL::ComPtr<ID3D12Resource> resource =
          std::move(bucket.back());
      bucket.pop_back();
      return resource;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource> resource =
        m_allocFunction(bucketSize, resourceFlags);

    return resource;
  } else {
    Microsoft::WRL::ComPtr<ID3D12Resource> d3d12_resource =
        m_allocFunction(requestedSize, resourceFlags);
    return d3d12_resource;
  }
}

}  // namespace dml
}  // namespace ctranslate2