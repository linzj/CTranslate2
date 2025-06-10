#include "dxdevice.h"
#include <spdlog/spdlog.h>
#include "bucketized_buffer_allocator.h"
#include "command_queue.h"
#include "common.h"
#include "descriptor_pool.h"
#include "dml/dml_utils.h"
#include "dml/operator_cache.h"

#include <assert.h>
#include <dxgi1_6.h>

#if defined(max)
#undef max
#endif

using Microsoft::WRL::ComPtr;

static void __stdcall DebugMessageCallback(D3D12_MESSAGE_CATEGORY cat,
                                           D3D12_MESSAGE_SEVERITY sev,
                                           D3D12_MESSAGE_ID id,
                                           LPCSTR message,
                                           void* context) {
  if (context) {
    if (id == D3D12_MESSAGE_ID_META_COMMAND_UNSUPPORTED_PARAMS) {
      // Ignore this message, since DML internally may try to create
      // metacommands that are not supported.
      return;
    }

    if ((D3D12_MESSAGE_SEVERITY_INFO == sev) ||
        (D3D12_MESSAGE_SEVERITY_MESSAGE == sev)) {
      SPDLOG_INFO("%d %d %s", int(cat), int(id), message);
    } else if (D3D12_MESSAGE_SEVERITY_WARNING == sev) {
      SPDLOG_WARN("%d %d %s", int(cat), int(id), message);
    } else {
      SPDLOG_ERROR("%d %d %s", int(cat), int(id), message);
    }
  }
}

namespace ctranslate2 {
namespace dml {

Device::Device(IAdapter* adapter,
               D3D_FEATURE_LEVEL featureLevel,
               DML_FEATURE_LEVEL dmlFeatureLevel,
               bool debugLayersEnabled,
               D3D12_COMMAND_LIST_TYPE commandListType,
               uint32_t dispatchRepeat,
               bool uavBarrierAfterDispatch,
               bool aliasingBarrierAfterDispatch,
               bool clearShaderCaches,
               bool disableGpuTimeout,
               bool enableDred,
               bool disableBackgroundProcessing,
               bool setStablePowerState,
               bool preferCustomHeaps,
               bool usePresentSeparator,
               uint32_t maxGpuTimeMeasurements,
               std::shared_ptr<D3d12Module> d3dModule,
               std::shared_ptr<DmlModule> dmlModule)
    : m_d3dModule(d3dModule),
      m_dmlModule(dmlModule),
      m_dispatchRepeat(dispatchRepeat),
      m_restoreBackgroundProcessing(disableBackgroundProcessing),
      m_restoreStablePowerState(setStablePowerState),
      m_useCustomHeaps(preferCustomHeaps) {
  HRESULT hr;
  DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags =
      debugLayersEnabled ? DML_CREATE_DEVICE_FLAG_DEBUG
                         : DML_CREATE_DEVICE_FLAG_NONE;

#if 0
  if (debugLayersEnabled) {
    ComPtr<ID3D12Debug3> d3dDebug;
    THROW_IF_FAILED(m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&d3dDebug)));
    d3dDebug->EnableDebugLayer();
    d3dDebug->SetEnableGPUBasedValidation(true);
  }
#else
  // debug flag work on media player.
  if (debugLayersEnabled) {
    ComPtr<ID3D12Debug3> d3dDebug;
    THROW_IF_FAILED(m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&d3dDebug)));
    d3dDebug->EnableDebugLayer();
    d3dDebug->SetEnableSynchronizedCommandQueueValidation(true);
  }
#endif

  if (featureLevel == D3D_FEATURE_LEVEL_1_0_CORE) {
    // Attempt to create a D3D_FEATURE_LEVEL_1_0_CORE device first, in case the
    // device supports this feature level and the D3D runtime does not support
    // D3D_FEATURE_LEVEL_1_0_GENERIC
    HRESULT hrUnused = m_d3dModule->CreateDevice(
        adapter, D3D_FEATURE_LEVEL_1_0_CORE, IID_PPV_ARGS(&m_d3d));
  }

  if (!m_d3d) {
    THROW_IF_FAILED(
        m_d3dModule->CreateDevice(adapter, featureLevel, IID_PPV_ARGS(&m_d3d)));
  }

  if (enableDred) {
    // Enables more debug info for TDRs, can be used with Debugger
    // extension see following link for more info:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred
    ComPtr<ID3D12DeviceRemovedExtendedDataSettings> pDredSettings;
    if (SUCCEEDED(
            m_d3dModule->GetDebugInterface(IID_PPV_ARGS(&pDredSettings)))) {
      pDredSettings->SetAutoBreadcrumbsEnablement(
          D3D12_DRED_ENABLEMENT_FORCED_ON);
      pDredSettings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
  }

  if (debugLayersEnabled) {
    hr = m_d3d->QueryInterface(m_infoQueue.GetAddressOf());
    if (SUCCEEDED(hr)) {
      m_infoQueue->RegisterMessageCallback(DebugMessageCallback,
                                           D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                                           nullptr, &m_callbackCookie);
    }
  }

  if (disableBackgroundProcessing) {
    HRESULT hr = m_d3d->SetBackgroundProcessingMode(
        D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr);

    if (FAILED(hr)) {
      SPDLOG_ERROR(
          "Failed to disable background processing. Do you have developer mode "
          "enabled?");
      THROW_HR(hr);
    }
  }

  if (setStablePowerState) {
    HRESULT hr = m_d3d->SetStablePowerState(TRUE);
    if (FAILED(hr)) {
      SPDLOG_ERROR(
          "Failed to disable background processing. Do you have developer mode "
          "enabled?");
      THROW_HR(hr);
    }
  }

  D3D12_FEATURE_DATA_ARCHITECTURE1 archData = {};
  if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1,
                                           &archData, sizeof(archData)))) {
    m_architectureSupport = archData;
  }

  // Custom heaps should only be used on UMA systems with cache-coherent memory.
  m_useCustomHeaps = m_useCustomHeaps && m_architectureSupport->UMA &&
                     m_architectureSupport->CacheCoherentUMA;

  D3D_FEATURE_LEVEL featureLevelsList[] = {
      D3D_FEATURE_LEVEL_1_0_CORE, D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,     D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,     D3D_FEATURE_LEVEL_12_1};

  D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};
  featureLevels.NumFeatureLevels = _countof(featureLevelsList);
  featureLevels.pFeatureLevelsRequested = featureLevelsList;
  THROW_IF_FAILED(m_d3d->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS, &featureLevels, sizeof(featureLevels)));

  m_descriptorPool = std::make_unique<DescriptorPool>(m_d3d.Get(), 256);
  m_allocator = std::make_unique<BucketizedBufferAllocator>(this);
  m_operatorCache = std::make_unique<DMLOperatorCache>(this);
#if 0
  // Custom heaps are optional for MCDM devices, so we also need to check for
  // support.
  if (featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE ||
      featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 options = {};
    if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,
                                             &options, sizeof(options)))) {
      m_useCustomHeaps =
          m_useCustomHeaps && options.ComputeOnlyCustomHeapSupported;
    }
  }
#endif

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Flags = disableGpuTimeout
                        ? D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT
                        : D3D12_COMMAND_QUEUE_FLAG_NONE;
  queueDesc.Type = commandListType;
  ComPtr<ID3D12CommandQueue> queue;
  THROW_IF_FAILED(m_d3d->CreateCommandQueue(
      &queueDesc, IID_GRAPHICS_PPV_ARGS(queue.ReleaseAndGetAddressOf())));
  m_queue = std::make_unique<CommandQueue>(queue.Get(), false);
  m_commandListType = queueDesc.Type;

#if defined(INCLUDE_DXGI)
  // Create dummy swapchain for frame indication
  if (usePresentSeparator) {
    ComPtr<IDXGIFactory2> factory;
    THROW_IF_FAILED(
        CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));
    DXGI_SWAP_CHAIN_DESC1 desc = {0};
    desc.Width = 2;
    desc.Height = 2;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.Stereo = FALSE;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER;
    desc.BufferCount = 3;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    const auto hr = factory->CreateSwapChainForComposition(
        m_queue.Get(), &desc, nullptr, m_dummySwapChain.GetAddressOf());
    if (FAILED(hr)) {
      SRLOG(LOG_WARNING)
          << "Creating dummy swap chain for present seperator failed";
    }
  }
#endif

  THROW_IF_FAILED(m_dmlModule->CreateDevice1(m_d3d.Get(), dmlCreateDeviceFlags,
                                             dmlFeatureLevel,
                                             IID_PPV_ARGS(&m_dml)));

  THROW_IF_FAILED(m_d3d->CreateCommandAllocator(
      m_commandListType,
      IID_GRAPHICS_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(m_d3d->CreateCommandList(
      0, m_commandListType, m_commandAllocator.Get(), nullptr,
      IID_GRAPHICS_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(
      m_dml->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));
  THROW_IF_FAILED(m_dml->CreateOperatorInitializer(
      0, nullptr, IID_PPV_ARGS(&m_initializer)));

  // Each GPU time measurement requires a pair of timestamps
  m_timestampCapacity = maxGpuTimeMeasurements * 2;

  if (m_timestampCapacity > 0) {
    D3D12_QUERY_HEAP_DESC queryHeapDesc;
    queryHeapDesc.Count = m_timestampCapacity;
    queryHeapDesc.NodeMask = 0;
    queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

    THROW_IF_FAILED(
        m_d3d->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_timestampHeap)));
  }

  if (uavBarrierAfterDispatch) {
    m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
  }

  if (aliasingBarrierAfterDispatch) {
    m_postDispatchBarriers.emplace_back(
        CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));
  }

  if (clearShaderCaches) {
    ClearShaderCaches();
  }
}

Device::Device(ID3D12Device* d3ddevice,
               ID3D12CommandQueue* command_queue,
               DML_FEATURE_LEVEL dmlFeatureLevel,
               D3D12_COMMAND_LIST_TYPE commandListType,
               uint32_t dispatchRepeat,
               bool uavBarrierAfterDispatch,
               bool aliasingBarrierAfterDispatch,
               bool clearShaderCaches,
               bool disableGpuTimeout,
               bool preferCustomHeaps,
               bool usePresentSeparator,
               uint32_t maxGpuTimeMeasurements,
               std::shared_ptr<D3d12Module> d3dModule,
               std::shared_ptr<DmlModule> dmlModule)
    : m_d3dModule(d3dModule),
      m_dmlModule(dmlModule),
      m_commandListType(commandListType),
      m_dispatchRepeat(dispatchRepeat),
      m_queue(std::make_unique<CommandQueue>(command_queue, false)),
      m_restoreBackgroundProcessing(false),
      m_restoreStablePowerState(false),
      m_useCustomHeaps(preferCustomHeaps) {
  HRESULT hr;

  THROW_IF_FAILED(d3ddevice->QueryInterface(IID_PPV_ARGS(&m_d3d)));

  D3D12_FEATURE_DATA_ARCHITECTURE1 archData = {};
  if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1,
                                           &archData, sizeof(archData)))) {
    m_architectureSupport = archData;
  }

  // Custom heaps should only be used on UMA systems with cache-coherent memory.
  m_useCustomHeaps = m_useCustomHeaps && m_architectureSupport->UMA &&
                     m_architectureSupport->CacheCoherentUMA;

  D3D_FEATURE_LEVEL featureLevelsList[] = {
      D3D_FEATURE_LEVEL_1_0_CORE, D3D_FEATURE_LEVEL_1_0_CORE,
      D3D_FEATURE_LEVEL_11_0,     D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_12_0,     D3D_FEATURE_LEVEL_12_1};

  D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};
  featureLevels.NumFeatureLevels = _countof(featureLevelsList);
  featureLevels.pFeatureLevelsRequested = featureLevelsList;
  THROW_IF_FAILED(m_d3d->CheckFeatureSupport(
      D3D12_FEATURE_FEATURE_LEVELS, &featureLevels, sizeof(featureLevels)));

#if 0
  // Custom heaps are optional for MCDM devices, so we also need to check for
  // support.
  if (featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE ||
      featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 options = {};
    if (SUCCEEDED(m_d3d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,
                                             &options, sizeof(options)))) {
      m_useCustomHeaps =
          m_useCustomHeaps && options.ComputeOnlyCustomHeapSupported;
    }
  }
#endif

#if defined(INCLUDE_DXGI)
  // Create dummy swapchain for frame indication
  if (usePresentSeparator) {
    ComPtr<IDXGIFactory2> factory;
    THROW_IF_FAILED(
        CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&factory)));
    DXGI_SWAP_CHAIN_DESC1 desc = {0};
    desc.Width = 2;
    desc.Height = 2;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.Stereo = FALSE;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_BACK_BUFFER;
    desc.BufferCount = 3;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    const auto hr = factory->CreateSwapChainForComposition(
        m_queue.Get(), &desc, nullptr, m_dummySwapChain.GetAddressOf());
    if (FAILED(hr)) {
      SRLOG(LOG_WARNING)
          << "Creating dummy swap chain for present seperator failed";
    }
  }
#endif

  THROW_IF_FAILED(
      m_dmlModule->CreateDevice1(m_d3d.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                 dmlFeatureLevel, IID_PPV_ARGS(&m_dml)));

  THROW_IF_FAILED(m_d3d->CreateCommandAllocator(
      m_commandListType,
      IID_GRAPHICS_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(m_d3d->CreateCommandList(
      0, m_commandListType, m_commandAllocator.Get(), nullptr,
      IID_GRAPHICS_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

  THROW_IF_FAILED(
      m_dml->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));

  // Each GPU time measurement requires a pair of timestamps
  m_timestampCapacity = maxGpuTimeMeasurements * 2;

  if (m_timestampCapacity > 0) {
    D3D12_QUERY_HEAP_DESC queryHeapDesc;
    queryHeapDesc.Count = m_timestampCapacity;
    queryHeapDesc.NodeMask = 0;
    queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

    THROW_IF_FAILED(
        m_d3d->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_timestampHeap)));
  }

  if (uavBarrierAfterDispatch) {
    m_postDispatchBarriers.emplace_back(CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
  }

  if (aliasingBarrierAfterDispatch) {
    m_postDispatchBarriers.emplace_back(
        CD3DX12_RESOURCE_BARRIER::Aliasing(nullptr, nullptr));
  }

  if (clearShaderCaches) {
    ClearShaderCaches();
  }

  m_descriptorPool = std::make_unique<DescriptorPool>(m_d3d.Get(), 1024 * 1024);
  m_allocator = std::make_unique<BucketizedBufferAllocator>(this);
  m_operatorCache = std::make_unique<DMLOperatorCache>(this);
}

Device::~Device() {
  if (m_callbackCookie != 0) {
    m_infoQueue->UnregisterMessageCallback(m_callbackCookie);
    m_callbackCookie = 0;
  }

  if (m_d3d) {
    // Restore state for certain features that may have been toggled. Normally
    // this isn't required, since the state changes don't persist beyond the
    // process lifetime, but the real D3D12 device is a singleton that may have
    // other refs in the process (e.g., DxDispatch instance used in a test DLL).

    if (m_restoreBackgroundProcessing) {
      (void)m_d3d->SetBackgroundProcessingMode(
          D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
          D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr);
    }

    if (m_restoreStablePowerState) {
      (void)m_d3d->SetStablePowerState(FALSE);
    }
  }
  // Leak operator cache now, for its destruction causes crash.
  m_operatorCache.release();
}

Microsoft::WRL::ComPtr<IResourceWrapper>
Device::CreatePreferredDeviceMemoryBuffer(uint64_t sizeInBytes,
                                          D3D12_RESOURCE_FLAGS resourceFlags,
                                          uint64_t alignment,
                                          D3D12_HEAP_FLAGS heapFlags) {
  auto resource_wrapper = m_allocator->Alloc(sizeInBytes, resourceFlags);
  KeepAliveUntilNextCommandListDispatch(resource_wrapper);
  return resource_wrapper;
}

Microsoft::WRL::ComPtr<ID3D12Resource>
Device::CreatePreferredDeviceMemoryBufferWithoutPooling(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags) {
  auto return_value =
      m_useCustomHeaps
          ? CreateCustomBuffer(sizeInBytes, resourceFlags, alignment, heapFlags)
          : CreateDefaultBuffer(sizeInBytes, resourceFlags, alignment,
                                heapFlags);
  KeepAliveUntilNextCommandListDispatch(return_value);
  return return_value;
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::CreateCustomBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags) {
  auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
  auto heapProps = CD3DX12_HEAP_PROPERTIES(
      D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0);

  ComPtr<ID3D12Resource> resource;
  THROW_IF_FAILED(m_d3d->CreateCommittedResource(
      &heapProps, heapFlags, &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

  return resource;
}

ComPtr<ID3D12Resource> Device::CreateDefaultBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags) {
  auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
  auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

  ComPtr<ID3D12Resource> resource;
  THROW_IF_FAILED(m_d3d->CreateCommittedResource(
      &heapProps, heapFlags, &resourceDesc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

  return resource;
}

ComPtr<ID3D12Resource> Device::CreateUploadBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags) {
  auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
  auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

  ComPtr<ID3D12Resource> resource;
  THROW_IF_FAILED(m_d3d->CreateCommittedResource(
      &heapProps, heapFlags, &resourceDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr, IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

  return resource;
}

ComPtr<ID3D12Resource> Device::CreateReadbackBuffer(
    uint64_t sizeInBytes,
    D3D12_RESOURCE_FLAGS resourceFlags,
    uint64_t alignment,
    D3D12_HEAP_FLAGS heapFlags) {
  auto resourceDesc =
      CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, resourceFlags, alignment);
  auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

  ComPtr<ID3D12Resource> resource;
  THROW_IF_FAILED(m_d3d->CreateCommittedResource(
      &heapProps, heapFlags, &resourceDesc, D3D12_RESOURCE_STATE_COPY_DEST,
      nullptr, IID_GRAPHICS_PPV_ARGS(resource.ReleaseAndGetAddressOf())));

  return resource;
}

void Device::WaitForGpuWorkToComplete() {
  m_queue->GetCurrentCompletionEvent().WaitForSignal();

  m_descriptorPool->Trim();
}

void Device::RecordDispatch(const char* name,
                            uint32_t threadGroupX,
                            uint32_t threadGroupY,
                            uint32_t threadGroupZ) {
  // PIXBeginEvent(m_commandList.Get(), PIX_COLOR(255, 255, 0), "HLSL: '%s'",
  //               name);
  RecordTimestamp();

  for (uint32_t i = 0; i < m_dispatchRepeat; i++) {
    m_commandList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
    if (!m_postDispatchBarriers.empty()) {
      if (m_postDispatchBarriers.size() >
          std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "ResourceBarrier " + std::to_string(m_postDispatchBarriers.size()) +
            " is too large.");
      }
      m_commandList->ResourceBarrier(
          static_cast<uint32_t>(m_postDispatchBarriers.size()),
          m_postDispatchBarriers.data());
    }
  }

  RecordTimestamp();
  // PIXEndEvent(m_commandList.Get());
}

void Device::KeepAliveUntilNextCommandListDispatch(
    Microsoft::WRL::ComPtr<IGraphicsUnknown>&& object) {
  if (object) {
    m_temporaryResources.emplace(std::move(object));
  }
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::Upload(uint64_t totalSize,
                                                      std::string_view data,
                                                      std::wstring_view name) {
  if (data.size() > totalSize) {
    throw std::invalid_argument(
        "Attempting to upload more data than the size of the buffer");
  }

  ComPtr<ID3D12Resource> buffer;

  if (m_useCustomHeaps) {
    buffer = CreateCustomBuffer(totalSize);
  } else {
    buffer = CreateDefaultBuffer(totalSize);
  }

  Upload(totalSize, data, buffer.Get(), name);
  return buffer;
}

Microsoft::WRL::ComPtr<ID3D12Resource> Device::Upload(uint64_t totalSize,
                                                      std::string_view data,
                                                      ID3D12Resource* buffer,
                                                      std::wstring_view name) {
  if (data.size() > totalSize) {
    throw std::invalid_argument(
        "Attempting to upload more data than the size of the buffer");
  }

  ComPtr<ID3D12Resource> uploadBuffer;
  ComPtr<ID3D12Resource> resourceToMap;

  uploadBuffer = data.empty() ? nullptr : CreateUploadBuffer(totalSize);
  uploadBuffer->SetName(L"Device::Upload");
  resourceToMap = uploadBuffer;

  if (!name.empty()) {
    buffer->SetName(name.data());
  }

  if (resourceToMap) {
    void* mappedBufferData = nullptr;
    THROW_IF_FAILED(resourceToMap->Map(0, nullptr, &mappedBufferData));
    memcpy(mappedBufferData, data.data(), data.size());
    resourceToMap->Unmap(0, nullptr);

    if (resourceToMap == uploadBuffer) {
      D3D12_RESOURCE_BARRIER barriers[] = {CD3DX12_RESOURCE_BARRIER::Transition(
          buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_COPY_DEST)};

      m_commandList->ResourceBarrier(_countof(barriers), barriers);
      m_commandList->CopyBufferRegion(buffer, 0, uploadBuffer.Get(), 0,
                                      totalSize);
      std::swap(barriers[0].Transition.StateBefore,
                barriers[0].Transition.StateAfter);
      m_commandList->ResourceBarrier(_countof(barriers), barriers);

      m_temporaryResources.emplace(std::move(uploadBuffer));
    }
  }

  return buffer;
}

std::vector<std::byte> Device::Download(
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer) {
  D3D12_RESOURCE_DESC resource_desc = buffer->GetDesc();

  std::vector<std::byte> outputBuffer(
      static_cast<size_t>(buffer->GetDesc().Width));

  size_t dataSize = static_cast<size_t>(buffer->GetDesc().Width);
  Download(buffer, outputBuffer.data(), dataSize);
  return outputBuffer;
}

void Device::Download(Microsoft::WRL::ComPtr<ID3D12Resource> buffer,
                      void* data,
                      size_t dataSize) {
  D3D12_RESOURCE_DESC resource_desc = buffer->GetDesc();
  if (resource_desc.Width > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("Buffer width '" +
                                std::to_string(buffer->GetDesc().Width) +
                                "' is too large.");
  }

  if (dataSize > resource_desc.Width) {
    throw std::invalid_argument(
        "Attempting to download more data than the size of the buffer");
  }

  ComPtr<ID3D12Resource> resourceToMap;

  // Can't assume the input buffer was created as a custom heap (e.g., ONNX
  // dispatchable with a deferred resource allocated by the DML EP), so check
  // the heap properties.
  D3D12_HEAP_PROPERTIES heapProps = {};
  D3D12_HEAP_FLAGS heapFlags = {};

  if (SUCCEEDED(buffer->GetHeapProperties(&heapProps, &heapFlags)) &&
      heapProps.MemoryPoolPreference == D3D12_MEMORY_POOL_L0 &&
      heapProps.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE) {
    resourceToMap = buffer;
  } else {
    resourceToMap = CreateReadbackBuffer(buffer->GetDesc().Width);
    resourceToMap->SetName(L"Device::Download");

    D3D12_RESOURCE_BARRIER barriers[] = {CD3DX12_RESOURCE_BARRIER::Transition(
        buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE)};

    m_commandList->ResourceBarrier(_countof(barriers), barriers);
    m_commandList->CopyBufferRegion(resourceToMap.Get(), 0, buffer.Get(), 0,
                                    dataSize);
    std::swap(barriers[0].Transition.StateBefore,
              barriers[0].Transition.StateAfter);
    m_commandList->ResourceBarrier(_countof(barriers), barriers);
    ExecuteCommandListAndWait();
  }

  CD3DX12_RANGE readRange(0, dataSize);
  void* mappedBufferData = nullptr;
  THROW_IF_FAILED(resourceToMap->Map(0, &readRange, &mappedBufferData));
  memcpy(data, mappedBufferData, dataSize);
  resourceToMap->Unmap(0, nullptr);
}

void Device::ExecuteCommandList() {
  THROW_IF_FAILED(m_commandList->Close());

  ID3D12CommandList* commandLists[] = {m_commandList.Get()};
  m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
  THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
  m_needResetAllocator = true;
}

void Device::ExecuteCommandListAndWait() {
  THROW_IF_FAILED(m_commandList->Close());

  ID3D12CommandList* commandLists[] = {m_commandList.Get()};
  m_queue->ExecuteCommandLists(_countof(commandLists), commandLists);
  WaitForGpuWorkToComplete();
  THROW_IF_FAILED(m_d3d->GetDeviceRemovedReason());
  THROW_IF_FAILED(m_commandAllocator->Reset());
  THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

  m_temporaryResources.clear();
}

void Device::ResetCommandList() {
  if (!m_needResetAllocator) {
    return;
  }
  m_needResetAllocator = false;
  WaitForGpuWorkToComplete();
  THROW_IF_FAILED(m_commandList->Close());
  THROW_IF_FAILED(m_commandAllocator->Reset());
  THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}

void Device::RecordTimestamp() {
  if (!GpuTimingEnabled()) {
    return;
  }

  m_commandList->EndQuery(m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                          m_timestampHeadIndex);
  m_timestampHeadIndex = (m_timestampHeadIndex + 1) % m_timestampCapacity;
  if (m_timestampCount < m_timestampCapacity) {
    m_timestampCount++;
  }
}

std::vector<uint64_t> Device::ResolveTimestamps() {
  assert(m_timestampCount <= m_timestampCapacity);

  if (!GpuTimingEnabled()) {
    return {};
  }

  auto timestampReadbackBuffer =
      CreateReadbackBuffer(sizeof(uint64_t) * m_timestampCount);

  m_commandList->ResolveQueryData(
      m_timestampHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, m_timestampCount,
      timestampReadbackBuffer.Get(), 0);
  ExecuteCommandListAndWait();

  void* pData = nullptr;
  D3D12_RANGE readRange = {0, sizeof(uint64_t) * m_timestampCount};
  timestampReadbackBuffer->Map(0, &readRange, &pData);

  std::vector<uint64_t> timestamps;
  const uint64_t* pTimestamps = reinterpret_cast<uint64_t*>(pData);
  timestamps.insert(timestamps.end(), pTimestamps,
                    pTimestamps + m_timestampCount);

  m_timestampHeadIndex = 0;
  m_timestampCount = 0;

  return timestamps;
}

std::vector<double> Device::ResolveTimingSamples() {
  std::vector<uint64_t> timestamps = ResolveTimestamps();
  if (timestamps.empty()) {
    return {};
  }

  uint64_t frequency = m_queue->GetTimestampFrequency();

  std::vector<double> samples(timestamps.size() / 2);

  for (uint32_t i = 0; i < samples.size(); ++i) {
    uint64_t timestampDelta =
        (timestamps[2 * i + 1] - timestamps[2 * i]) * 1000;
    samples[i] = double(timestampDelta) / frequency / m_dispatchRepeat;
  }

  return samples;
}

#ifndef DXCOMPILER_NONE
void Device::EnsureDxcInterfaces() {
#if defined(_GAMING_XBOX) || defined(_AMD64_)
  if (!m_dxcCompiler) {
    // Lazily create DXC compiler and helpers.
    THROW_IF_FAILED(
        DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils)));
    THROW_IF_FAILED(
        m_dxcUtils->CreateDefaultIncludeHandler(&m_dxcIncludeHandler));
    THROW_IF_FAILED(
        DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler)));
    THROW_IF_FAILED(
        DxcCreateInstance(CLSID_DxcValidator, IID_PPV_ARGS(&m_dxcValidator)));
    THROW_IF_FAILED(
        DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&m_dxcLibrary)));
  }
#endif
}

IDxcUtils* Device::GetDxcUtils() {
  EnsureDxcInterfaces();
  return m_dxcUtils.Get();
}

IDxcIncludeHandler* Device::GetDxcIncludeHandler() {
  EnsureDxcInterfaces();
  return m_dxcIncludeHandler.Get();
}

IDxcCompiler3* Device::GetDxcCompiler() {
  EnsureDxcInterfaces();
  return m_dxcCompiler.Get();
}

IDxcValidator* Device::GetDxcValidator() {
  EnsureDxcInterfaces();
  return m_dxcValidator.Get();
}

IDxcLibrary* Device::GetDxcLibrary() {
  EnsureDxcInterfaces();
  return m_dxcLibrary.Get();
}
#endif  // !DXCOMPILER_NONE

void Device::ClearShaderCaches() {
  struct {
    D3D12_SHADER_CACHE_KIND_FLAGS kind;
    const char* name;
  } caches[] = {
      {D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER,
       "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CACHE_FOR_DRIVER"},
      {D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS,
       "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_D3D_CONVERSIONS"},
      {D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED,
       "D3D12_SHADER_CACHE_KIND_FLAG_IMPLICIT_DRIVER_MANAGED"},
      {D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED,
       "D3D12_SHADER_CACHE_KIND_FLAG_APPLICATION_MANAGED"},
  };

  for (auto cache : caches) {
    auto hr = m_d3d->ShaderCacheControl(cache.kind,
                                        D3D12_SHADER_CACHE_CONTROL_FLAG_CLEAR);
    if (FAILED(hr)) {
      SPDLOG_INFO("Clearing %s failed. Do you have developer mode enabled?",
                  cache.name);
    } else {
      SPDLOG_INFO("Clearing %s succeeded.", cache.name);
    }
  }
}

/*static*/ uint32_t Device::GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType) {
  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_INT8:
    case DML_TENSOR_DATA_TYPE_UINT8:
      return 1;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_INT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
      return 2;

    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_INT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
      return 4;

    case DML_TENSOR_DATA_TYPE_FLOAT64:
    case DML_TENSOR_DATA_TYPE_INT64:
    case DML_TENSOR_DATA_TYPE_UINT64:
      return 8;

    default:
      throw std::invalid_argument("Unknown data type");
  }
}

/*static*/ DXGI_FORMAT
Device::GetDxgiFormatFromDmlTensorDataType(DML_TENSOR_DATA_TYPE dataType) {
  switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return DXGI_FORMAT_R16_FLOAT;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return DXGI_FORMAT_R32_FLOAT;
    // case DML_TENSOR_DATA_TYPE_FLOAT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_UINT8:
      return DXGI_FORMAT_R8_UINT;
    case DML_TENSOR_DATA_TYPE_UINT16:
      return DXGI_FORMAT_R16_UINT;
    case DML_TENSOR_DATA_TYPE_UINT32:
      return DXGI_FORMAT_R32_UINT;
    // case DML_TENSOR_DATA_TYPE_UINT64: no DXGI type exists
    case DML_TENSOR_DATA_TYPE_INT8:
      return DXGI_FORMAT_R8_SINT;
    case DML_TENSOR_DATA_TYPE_INT16:
      return DXGI_FORMAT_R16_SINT;
    case DML_TENSOR_DATA_TYPE_INT32:
      return DXGI_FORMAT_R32_SINT;
    // case DML_TENSOR_DATA_TYPE_INT64: no DXGI type exists
    default:
      throw std::invalid_argument("No DXGI_FORMAT exists for given data type.");
  }
}

void Device::DummyPresent() {
#if defined(INCLUDE_DXGI)
  if (m_dummySwapChain) {
    m_dummySwapChain->Present(0, 0);
  }
#endif
}

void Device::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    const DML_BINDING_DESC& inputArrayBinding) {
  // Reset the initializer to reference the input operator.
  IDMLCompiledOperator* ops[] = {op};
  THROW_IF_FAILED(m_initializer->Reset(ARRAYSIZE(ops), ops));

  DML_BINDING_PROPERTIES initBindingProps =
      m_initializer->GetBindingProperties();

  const uint32_t numDescriptors = initBindingProps.RequiredDescriptorCount;
  DescriptorRange descriptorRange = m_descriptorPool->AllocDescriptors(
      numDescriptors, m_queue->GetNextCompletionEvent());

  // Create a binding table for initialization.
  DML_BINDING_TABLE_DESC bindingTableDesc = {};
  bindingTableDesc.Dispatchable = m_initializer.Get();
  bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
  bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
  bindingTableDesc.SizeInDescriptors = numDescriptors;

  ComPtr<IDMLBindingTable> bindingTable;
  THROW_IF_FAILED(m_dml->CreateBindingTable(&bindingTableDesc,
                                            IID_PPV_ARGS(&bindingTable)));

  // Create a temporary resource for initializing the op, if it's required.
  UINT64 temporaryResourceSize = initBindingProps.TemporaryResourceSize;
  if (temporaryResourceSize > 0) {
    ComPtr<IResourceWrapper> resource_wrapper =
        CreatePreferredDeviceMemoryBuffer(
            temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Bind the temporary resource.
    DML_BUFFER_BINDING bufferBinding = {resource_wrapper->GetD3D12Resource(), 0,
                                        temporaryResourceSize};
    DML_BINDING_DESC bindingDesc = {DML_BINDING_TYPE_BUFFER, &bufferBinding};
    bindingTable->BindTemporaryResource(&bindingDesc);
  }

  // Bind inputs, if provided.
  if (inputArrayBinding.Type != DML_BINDING_TYPE_NONE) {
    // An operator with inputs to bind MUST use a BUFFER_ARRAY.
    assert(inputArrayBinding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
    bindingTable->BindInputs(1, &inputArrayBinding);
  }

  // Bind the persistent resource, which is an output of initialization.
  if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) {
    // Persistent resources MUST be bound as buffers.
    assert(persistentResourceBinding.Type == DML_BINDING_TYPE_BUFFER);
    bindingTable->BindOutputs(1, &persistentResourceBinding);
  }

  // Record the initialization work.
  SetDescriptorHeap(descriptorRange.heap);
  m_commandRecorder->RecordDispatch(m_commandList.Get(), m_initializer.Get(),
                                    bindingTable.Get());

  // Barrier if there's an output (i.e. persistent resource), or if any temps
  // are used.
  if ((persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) ||
      (temporaryResourceSize > 0)) {
    auto uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_commandList->ResourceBarrier(1, &uav);
  }
  ExecuteCommandList();
}

void Device::ExecuteOperator(IDMLCompiledOperator* op,
                             const DML_BINDING_DESC& persistentResourceBinding,
                             std::vector<DML_BINDING_DESC> inputBindings,
                             std::vector<DML_BINDING_DESC> outputBindings) {
  DML_BINDING_PROPERTIES execBindingProps = op->GetBindingProperties();

  const uint32_t numDescriptors = execBindingProps.RequiredDescriptorCount;
  DescriptorRange descriptorRange = m_descriptorPool->AllocDescriptors(
      numDescriptors, m_queue->GetNextCompletionEvent());

  // Create a binding table for execution.
  DML_BINDING_TABLE_DESC bindingTableDesc = {};
  bindingTableDesc.Dispatchable = op;
  bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
  bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
  bindingTableDesc.SizeInDescriptors = numDescriptors;

  ComPtr<IDMLBindingTable> bindingTable;
  THROW_IF_FAILED(m_dml->CreateBindingTable(&bindingTableDesc,
                                            IID_PPV_ARGS(&bindingTable)));

  // Create a temporary resource for executing the op, if it's required.
  UINT64 temporaryResourceSize = execBindingProps.TemporaryResourceSize;
  if (temporaryResourceSize > 0) {
    ComPtr<IResourceWrapper> resource_wrapper =
        CreatePreferredDeviceMemoryBuffer(
            temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Bind the temporary resource.
    DML_BUFFER_BINDING bufferBinding = {resource_wrapper->GetD3D12Resource(), 0,
                                        temporaryResourceSize};
    DML_BINDING_DESC bindingDesc = {DML_BINDING_TYPE_BUFFER, &bufferBinding};
    bindingTable->BindTemporaryResource(&bindingDesc);
  }

  if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) {
    bindingTable->BindPersistentResource(&persistentResourceBinding);
  }

  bindingTable->BindInputs(static_cast<uint32_t>(inputBindings.size()),
                           inputBindings.data());
  bindingTable->BindOutputs(static_cast<uint32_t>(outputBindings.size()),
                            outputBindings.data());

  // Record the execution work.
  SetDescriptorHeap(descriptorRange.heap);
  m_commandRecorder->RecordDispatch(m_commandList.Get(), op,
                                    bindingTable.Get());

  // Barrier all outputs.
  ExecuteCommandList();
}

void Device::ExecuteOperator(IDMLCompiledOperator* op,
                             std::vector<DML_BINDING_DESC> inputBindings,
                             std::vector<DML_BINDING_DESC> outputBindings) {
  DML_BINDING_DESC persistentBindingDesc = {};
  ExecuteOperator(op, persistentBindingDesc, std::move(inputBindings),
                  std::move(outputBindings));
}

void Device::ExecuteOperator(
    IDMLCompiledOperator* compiled_op,
    const std::vector<ID3D12Resource*>& input_resources,
    const std::vector<ID3D12Resource*>& output_resources,
    ID3D12Resource* persistent_resource) {
  // Use DmlBufferBindingBundle for the optional persistent resource.
  utils::DmlBufferBindingBundle persistent_binding_bundle;
  if (persistent_resource) {
    persistent_binding_bundle =
        utils::DmlBufferBindingBundle(persistent_resource);
  }
  DML_BINDING_DESC persistent_binding_desc =
      persistent_binding_bundle.get_desc();

  // Use DmlBindingArrayBundle for input and output resources.
  // These bundles will manage the underlying DmlBufferBindingBundle objects
  // and provide a vector of DML_BINDING_DESC.
  utils::DmlBindingArrayBundle input_array_bundle(input_resources);
  utils::DmlBindingArrayBundle output_array_bundle(output_resources);

  ExecuteOperator(compiled_op, persistent_binding_desc,
                  input_array_bundle.get_descs(),
                  output_array_bundle.get_descs());
}

void Device::SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap) {
  if (descriptorHeap != nullptr) {
    if (descriptorHeap != m_currentDescriptorHeap) {
      m_currentDescriptorHeap = descriptorHeap;
    }
    ID3D12DescriptorHeap* descriptorHeaps[] = {descriptorHeap};
    m_commandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                      descriptorHeaps);
  }
}

}  // namespace dml
}  // namespace ctranslate2