#pragma once
#include <combaseapi.h>
#include <d3d12.h>

class ComError {
 public:
  explicit ComError(HRESULT hr) : hr_(hr) {}
  ~ComError() = default;
  HRESULT hr() const { return hr_; }

 private:
  HRESULT hr_;
};

#define THROW_IF_FAILED(_hr) \
  {                          \
    auto my_hr = (_hr);      \
    if (FAILED(my_hr)) {     \
      throw ComError(my_hr); \
    }                        \
  }

#define THROW_HR(_hr) throw ComError(_hr);

inline UINT8 D3D12GetFormatPlaneCount(_In_ ID3D12Device* pDevice,
                                      DXGI_FORMAT Format) {
  D3D12_FEATURE_DATA_FORMAT_INFO formatInfo = {Format, 0};
  if (FAILED(pDevice->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO,
                                          &formatInfo, sizeof(formatInfo)))) {
    return 0;
  }
  return formatInfo.PlaneCount;
}

inline UINT D3D12CalcSubresource(UINT MipSlice,
                                 UINT ArraySlice,
                                 UINT PlaneSlice,
                                 UINT MipLevels,
                                 UINT ArraySize) {
  return MipSlice + ArraySlice * MipLevels + PlaneSlice * MipLevels * ArraySize;
}

struct CD3DX12_DEFAULT {};
extern const DECLSPEC_SELECTANY CD3DX12_DEFAULT D3D12_DEFAULT;
struct CD3DX12_RESOURCE_BARRIER : public D3D12_RESOURCE_BARRIER {
  CD3DX12_RESOURCE_BARRIER() = default;
  explicit CD3DX12_RESOURCE_BARRIER(const D3D12_RESOURCE_BARRIER& o)
      : D3D12_RESOURCE_BARRIER(o) {}
  static inline CD3DX12_RESOURCE_BARRIER Transition(
      _In_ ID3D12Resource* pResource,
      D3D12_RESOURCE_STATES stateBefore,
      D3D12_RESOURCE_STATES stateAfter,
      UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_BARRIER_FLAGS flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
    CD3DX12_RESOURCE_BARRIER result = {};
    D3D12_RESOURCE_BARRIER& barrier = result;
    result.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    result.Flags = flags;
    barrier.Transition.pResource = pResource;
    barrier.Transition.StateBefore = stateBefore;
    barrier.Transition.StateAfter = stateAfter;
    barrier.Transition.Subresource = subresource;
    return result;
  }
  static inline CD3DX12_RESOURCE_BARRIER Aliasing(
      _In_ ID3D12Resource* pResourceBefore,
      _In_ ID3D12Resource* pResourceAfter) {
    CD3DX12_RESOURCE_BARRIER result = {};
    D3D12_RESOURCE_BARRIER& barrier = result;
    result.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    barrier.Aliasing.pResourceBefore = pResourceBefore;
    barrier.Aliasing.pResourceAfter = pResourceAfter;
    return result;
  }
  static inline CD3DX12_RESOURCE_BARRIER UAV(_In_ ID3D12Resource* pResource) {
    CD3DX12_RESOURCE_BARRIER result = {};
    D3D12_RESOURCE_BARRIER& barrier = result;
    result.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = pResource;
    return result;
  }
};

struct CD3DX12_RESOURCE_DESC : public D3D12_RESOURCE_DESC {
  CD3DX12_RESOURCE_DESC() = default;
  explicit CD3DX12_RESOURCE_DESC(const D3D12_RESOURCE_DESC& o)
      : D3D12_RESOURCE_DESC(o) {}
  CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION dimension,
                        UINT64 alignment,
                        UINT64 width,
                        UINT height,
                        UINT16 depthOrArraySize,
                        UINT16 mipLevels,
                        DXGI_FORMAT format,
                        UINT sampleCount,
                        UINT sampleQuality,
                        D3D12_TEXTURE_LAYOUT layout,
                        D3D12_RESOURCE_FLAGS flags) {
    Dimension = dimension;
    Alignment = alignment;
    Width = width;
    Height = height;
    DepthOrArraySize = depthOrArraySize;
    MipLevels = mipLevels;
    Format = format;
    SampleDesc.Count = sampleCount;
    SampleDesc.Quality = sampleQuality;
    Layout = layout;
    Flags = flags;
  }
  static inline CD3DX12_RESOURCE_DESC Buffer(
      const D3D12_RESOURCE_ALLOCATION_INFO& resAllocInfo,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    return CD3DX12_RESOURCE_DESC(
        D3D12_RESOURCE_DIMENSION_BUFFER, resAllocInfo.Alignment,
        resAllocInfo.SizeInBytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags);
  }
  static inline CD3DX12_RESOURCE_DESC Buffer(
      UINT64 width,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      UINT64 alignment = 0) {
    return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_BUFFER, alignment,
                                 width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, 1, 0,
                                 D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags);
  }
  static inline CD3DX12_RESOURCE_DESC Tex1D(
      DXGI_FORMAT format,
      UINT64 width,
      UINT16 arraySize = 1,
      UINT16 mipLevels = 0,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      UINT64 alignment = 0) {
    return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE1D, alignment,
                                 width, 1, arraySize, mipLevels, format, 1, 0,
                                 layout, flags);
  }
  static inline CD3DX12_RESOURCE_DESC Tex2D(
      DXGI_FORMAT format,
      UINT64 width,
      UINT height,
      UINT16 arraySize = 1,
      UINT16 mipLevels = 0,
      UINT sampleCount = 1,
      UINT sampleQuality = 0,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      UINT64 alignment = 0) {
    return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE2D, alignment,
                                 width, height, arraySize, mipLevels, format,
                                 sampleCount, sampleQuality, layout, flags);
  }
  static inline CD3DX12_RESOURCE_DESC Tex3D(
      DXGI_FORMAT format,
      UINT64 width,
      UINT height,
      UINT16 depth,
      UINT16 mipLevels = 0,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      D3D12_TEXTURE_LAYOUT layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
      UINT64 alignment = 0) {
    return CD3DX12_RESOURCE_DESC(D3D12_RESOURCE_DIMENSION_TEXTURE3D, alignment,
                                 width, height, depth, mipLevels, format, 1, 0,
                                 layout, flags);
  }
  inline UINT16 Depth() const {
    return (Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize
                                                            : 1);
  }
  inline UINT16 ArraySize() const {
    return (Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D ? DepthOrArraySize
                                                            : 1);
  }
  inline UINT8 PlaneCount(_In_ ID3D12Device* pDevice) const {
    return D3D12GetFormatPlaneCount(pDevice, Format);
  }
  inline UINT Subresources(_In_ ID3D12Device* pDevice) const {
    return MipLevels * ArraySize() * PlaneCount(pDevice);
  }
  inline UINT CalcSubresource(UINT MipSlice, UINT ArraySlice, UINT PlaneSlice) {
    return D3D12CalcSubresource(MipSlice, ArraySlice, PlaneSlice, MipLevels,
                                ArraySize());
  }
};

struct CD3DX12_HEAP_PROPERTIES : public D3D12_HEAP_PROPERTIES {
  CD3DX12_HEAP_PROPERTIES() = default;
  explicit CD3DX12_HEAP_PROPERTIES(const D3D12_HEAP_PROPERTIES& o)
      : D3D12_HEAP_PROPERTIES(o) {}
  CD3DX12_HEAP_PROPERTIES(D3D12_CPU_PAGE_PROPERTY cpuPageProperty,
                          D3D12_MEMORY_POOL memoryPoolPreference,
                          UINT creationNodeMask = 1,
                          UINT nodeMask = 1) {
    Type = D3D12_HEAP_TYPE_CUSTOM;
    CPUPageProperty = cpuPageProperty;
    MemoryPoolPreference = memoryPoolPreference;
    CreationNodeMask = creationNodeMask;
    VisibleNodeMask = nodeMask;
  }
  explicit CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE type,
                                   UINT creationNodeMask = 1,
                                   UINT nodeMask = 1) {
    Type = type;
    CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    CreationNodeMask = creationNodeMask;
    VisibleNodeMask = nodeMask;
  }
  bool IsCPUAccessible() const {
    return Type == D3D12_HEAP_TYPE_UPLOAD || Type == D3D12_HEAP_TYPE_READBACK ||
           (Type == D3D12_HEAP_TYPE_CUSTOM &&
            (CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE ||
             CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
  }
};

struct CD3DX12_RANGE : public D3D12_RANGE {
  CD3DX12_RANGE() = default;
  explicit CD3DX12_RANGE(const D3D12_RANGE& o) : D3D12_RANGE(o) {}
  CD3DX12_RANGE(SIZE_T begin, SIZE_T end) {
    Begin = begin;
    End = end;
  }
};

struct CD3DX12_CPU_DESCRIPTOR_HANDLE : public D3D12_CPU_DESCRIPTOR_HANDLE {
  CD3DX12_CPU_DESCRIPTOR_HANDLE() = default;
  explicit CD3DX12_CPU_DESCRIPTOR_HANDLE(
      const D3D12_CPU_DESCRIPTOR_HANDLE& o) noexcept
      : D3D12_CPU_DESCRIPTOR_HANDLE(o) {}
  CD3DX12_CPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
  CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other,
                                INT offsetScaledByIncrementSize) noexcept {
    InitOffsetted(other, offsetScaledByIncrementSize);
  }
  CD3DX12_CPU_DESCRIPTOR_HANDLE(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other,
                                INT offsetInDescriptors,
                                UINT descriptorIncrementSize) noexcept {
    InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
  }
  CD3DX12_CPU_DESCRIPTOR_HANDLE& Offset(INT offsetInDescriptors,
                                        UINT descriptorIncrementSize) noexcept {
    ptr = SIZE_T(INT64(ptr) +
                 INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    return *this;
  }
  CD3DX12_CPU_DESCRIPTOR_HANDLE& Offset(
      INT offsetScaledByIncrementSize) noexcept {
    ptr = SIZE_T(INT64(ptr) + INT64(offsetScaledByIncrementSize));
    return *this;
  }
  bool operator==(
      _In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other) const noexcept {
    return (ptr == other.ptr);
  }
  bool operator!=(
      _In_ const D3D12_CPU_DESCRIPTOR_HANDLE& other) const noexcept {
    return (ptr != other.ptr);
  }
  CD3DX12_CPU_DESCRIPTOR_HANDLE& operator=(
      const D3D12_CPU_DESCRIPTOR_HANDLE& other) noexcept {
    ptr = other.ptr;
    return *this;
  }

  inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& base,
                            INT offsetScaledByIncrementSize) noexcept {
    InitOffsetted(*this, base, offsetScaledByIncrementSize);
  }

  inline void InitOffsetted(_In_ const D3D12_CPU_DESCRIPTOR_HANDLE& base,
                            INT offsetInDescriptors,
                            UINT descriptorIncrementSize) noexcept {
    InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
  }

  static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE& handle,
                                   _In_ const D3D12_CPU_DESCRIPTOR_HANDLE& base,
                                   INT offsetScaledByIncrementSize) noexcept {
    handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
  }

  static inline void InitOffsetted(_Out_ D3D12_CPU_DESCRIPTOR_HANDLE& handle,
                                   _In_ const D3D12_CPU_DESCRIPTOR_HANDLE& base,
                                   INT offsetInDescriptors,
                                   UINT descriptorIncrementSize) noexcept {
    handle.ptr = SIZE_T(INT64(base.ptr) + INT64(offsetInDescriptors) *
                                              INT64(descriptorIncrementSize));
  }
};

struct CD3DX12_GPU_DESCRIPTOR_HANDLE : public D3D12_GPU_DESCRIPTOR_HANDLE
{
    CD3DX12_GPU_DESCRIPTOR_HANDLE() = default;
    explicit CD3DX12_GPU_DESCRIPTOR_HANDLE(const D3D12_GPU_DESCRIPTOR_HANDLE &o) noexcept :
        D3D12_GPU_DESCRIPTOR_HANDLE(o)
    {}
    CD3DX12_GPU_DESCRIPTOR_HANDLE(CD3DX12_DEFAULT) noexcept { ptr = 0; }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(other, offsetScaledByIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &other, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(other, offsetInDescriptors, descriptorIncrementSize);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE& Offset(INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        ptr = UINT64(INT64(ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
        return *this;
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE& Offset(INT offsetScaledByIncrementSize) noexcept
    {
        ptr = UINT64(INT64(ptr) + INT64(offsetScaledByIncrementSize));
        return *this;
    }
    inline bool operator==(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr == other.ptr);
    }
    inline bool operator!=(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE& other) const noexcept
    {
        return (ptr != other.ptr);
    }
    CD3DX12_GPU_DESCRIPTOR_HANDLE &operator=(const D3D12_GPU_DESCRIPTOR_HANDLE &other) noexcept
    {
        ptr = other.ptr;
        return *this;
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetScaledByIncrementSize);
    }

    inline void InitOffsetted(_In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        InitOffsetted(*this, base, offsetInDescriptors, descriptorIncrementSize);
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetScaledByIncrementSize) noexcept
    {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetScaledByIncrementSize));
    }

    static inline void InitOffsetted(_Out_ D3D12_GPU_DESCRIPTOR_HANDLE &handle, _In_ const D3D12_GPU_DESCRIPTOR_HANDLE &base, INT offsetInDescriptors, UINT descriptorIncrementSize) noexcept
    {
        handle.ptr = UINT64(INT64(base.ptr) + INT64(offsetInDescriptors) * INT64(descriptorIncrementSize));
    }
};
