#pragma once

#include <memory>
#include "dxmodule.h"

#ifdef INCLUDE_DXGI
#include <dxgi1_6.h>
#endif

#define IGraphicsUnknown IUnknown
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
typedef interface IDXGIAdapter1 IDXGIAdapter1;
typedef interface IResourceWrapper IResourceWrapper;
#define DXCOMPILER_NONE

using IAdapter = IDXGIAdapter1;

#include <DirectML.h>

#include <string_view>
#include <unordered_set>

// Simplified abstraction for submitting work to a device with a single command
// queue. Not thread safe. This "device" includes a single command list that is
// always open for recording work.

namespace ctranslate2 {
namespace dml {

class CommandQueue;
class DescriptorPool;
class BucketizedBufferAllocator;
class DMLOperatorCache;
class ConstantPool;
class GraphRecorder;

template <typename T>
struct ComPtrHasher {
  // The operator() that the unordered_set will call
  std::size_t operator()(const Microsoft::WRL::ComPtr<T>& ptr) const {
    // Get the raw pointer and hash it.
    // std::hash already has a specialization for pointer types.
    return std::hash<T*>()(ptr.Get());
  }
};

class Device {
 public:
  Device(IAdapter* adapter,  // IAdapter is ::IDXGIAdapter1, should be fine
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
         std::shared_ptr<DmlModule> dmlModule);

  Device(ID3D12Device* d3ddevice,
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
         std::shared_ptr<DmlModule> dmlModule);
  ~Device();

  D3d12Module* D3DModule() { return m_d3dModule.get(); }

  ID3D12Device9* D3D() { return m_d3d.Get(); }
  IDMLDevice1* DML() { return m_dml.Get(); }
  // ID3D12CommandQueue* GetCommandQueue();
  ID3D12QueryHeap* GetTimestampHeap() { return m_timestampHeap.Get(); }
  D3D12_COMMAND_LIST_TYPE GetCommandListType() const {
    return m_commandListType;
  }
  ID3D12GraphicsCommandList* GetCommandList() { return m_commandList.Get(); }

#ifndef DXCOMPILER_NONE
  IDxcUtils* GetDxcUtils();
  IDxcIncludeHandler* GetDxcIncludeHandler();
  IDxcCompiler3* GetDxcCompiler();
  IDxcValidator* GetDxcValidator();
  IDxcLibrary* GetDxcLibrary();
#endif

  // Creates either a default buffer or custom buffer based on support for
  // custom heaps and whether or not they are allowed.
  Microsoft::WRL::ComPtr<IResourceWrapper> CreatePreferredDeviceMemoryBuffer(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);
  Microsoft::WRL::ComPtr<ID3D12Resource>
  CreatePreferredDeviceMemoryBufferWithoutPooling(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

  Microsoft::WRL::ComPtr<ID3D12Resource> CreateCustomBuffer(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

  Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags =
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

  Microsoft::WRL::ComPtr<ID3D12Resource> CreateReadbackBuffer(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

  // Waits for all work submitted to this device's queue to complete.
  void WaitForGpuWorkToComplete();

  // Submits all commands recorded into the device's command list for execution.
  void ExecuteCommandList();

  // Submits the device command list for execution and blocks the CPU thread
  // until the commands have finished on the GPU.
  void ExecuteCommandListAndWait();

  void ResetCommandList();

  void CopyResourceSubRegion(ID3D12Resource* dst,
                             ID3D12Resource* src,
                             uint64_t dstOffset,
                             uint64_t srcOffset,
                             uint64_t sizeInBytes);

  // Records the dispatch of an HLSL shader.
  void RecordDispatch(const char* name,
                      uint32_t threadGroupX,
                      uint32_t threadGroupY,
                      uint32_t threadGroupZ);

  // Records a GPU timestamp in the device's command list. The device has a
  // limit on the number of unresolved timestamps; if this capacity is exceeded,
  // the oldest timestamps are dropped.
  void RecordTimestamp();

  // Resolves and returns all timestamp values recorded since the last call to
  // ResolveTimestamps. This is a blocking call that forces the CPU and GPU to
  // sync.
  std::vector<uint64_t> ResolveTimestamps();

  // Calls ResolveTimestamps() and converts timestamp pairs into timing samples.
  std::vector<double> ResolveTimingSamples();

  bool GpuTimingEnabled() const { return m_timestampCapacity > 0; }

  void KeepAliveUntilNextCommandListDispatch(
      Microsoft::WRL::ComPtr<IGraphicsUnknown>&& object);

  Microsoft::WRL::ComPtr<ID3D12Resource> Upload(uint64_t totalSize,
                                                std::string_view data,
                                                std::wstring_view name = {});

  Microsoft::WRL::ComPtr<ID3D12Resource> Upload(uint64_t totalSize,
                                                std::string_view data,
                                                ID3D12Resource* dst,
                                                std::wstring_view name = {});

  std::vector<std::byte> Download(Microsoft::WRL::ComPtr<ID3D12Resource>);
  void Download(Microsoft::WRL::ComPtr<ID3D12Resource>,
                void* data,
                size_t size);

  void ClearShaderCaches();

  static uint32_t GetSizeInBytes(DML_TENSOR_DATA_TYPE dataType);
  static DXGI_FORMAT GetDxgiFormatFromDmlTensorDataType(
      DML_TENSOR_DATA_TYPE dataType);

  void InitializeOperator(IDMLCompiledOperator* op,
                          const DML_BINDING_DESC& persistentResourceBinding,
                          const DML_BINDING_DESC& inputArrayBinding);

  void ExecuteOperator(IDMLCompiledOperator* op,
                       const DML_BINDING_DESC& persistentResourceBinding,
                       std::vector<DML_BINDING_DESC> inputBindings,
                       std::vector<DML_BINDING_DESC> outputBindings);

  void ExecuteOperator(IDMLCompiledOperator* op,
                       std::vector<DML_BINDING_DESC> inputBindings,
                       std::vector<DML_BINDING_DESC> outputBindings);

  void DummyPresent();

  DMLOperatorCache* GetOperatorCache() { return m_operatorCache.get(); }
  ConstantPool* GetConstantPool() { return m_constantPool.get(); }

  void BeginGraphRecording();
  void EndGraphRecording();
  void SplitGraphRecording();
  bool HasGraphRecordingBegun() const;
  GraphRecorder* GetGraphRecorder() { return m_graphRecorder.get(); }

 private:
  void EnsureDxcInterfaces();
  void SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap);
  void ExecuteCommandListInternal();
  Microsoft::WRL::ComPtr<ID3D12Resource> CreateUploadBuffer(
      uint64_t sizeInBytes,
      D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_NONE,
      uint64_t alignment = 0,
      D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE);

 private:
  std::shared_ptr<D3d12Module> m_d3dModule;
  std::shared_ptr<DmlModule> m_dmlModule;
  Microsoft::WRL::ComPtr<ID3D12Device9> m_d3d;
#ifndef _GAMING_XBOX
  Microsoft::WRL::ComPtr<ID3D12InfoQueue1> m_infoQueue;
#endif
  Microsoft::WRL::ComPtr<IDMLDevice1> m_dml;
  Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
  Microsoft::WRL::ComPtr<ID3D12QueryHeap> m_timestampHeap;
  ID3D12DescriptorHeap* m_currentDescriptorHeap = nullptr;
  uint32_t m_timestampCapacity = 0;
  uint32_t m_timestampHeadIndex = 0;
  uint32_t m_timestampCount = 0;
  D3D12_COMMAND_LIST_TYPE m_commandListType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;

  std::unordered_set<Microsoft::WRL::ComPtr<IGraphicsUnknown>,
                     ComPtrHasher<IGraphicsUnknown>>
      m_temporaryResources;
  uint32_t m_dispatchRepeat = 1;
  std::vector<D3D12_RESOURCE_BARRIER> m_postDispatchBarriers;
  std::optional<D3D12_FEATURE_DATA_ARCHITECTURE1> m_architectureSupport;
  std::unique_ptr<CommandQueue> m_queue;
  std::unique_ptr<DescriptorPool> m_descriptorPool;
  std::unique_ptr<BucketizedBufferAllocator> m_allocator;
  std::unique_ptr<BucketizedBufferAllocator> m_uploadAllocator;
  // Must split temporary allocator from the main allocator to avoid
  // overlapping allocations that can cause issues with DML operators.
  std::unique_ptr<BucketizedBufferAllocator> m_temporaryAllocator;
  std::unique_ptr<DMLOperatorCache> m_operatorCache;
  std::unique_ptr<ConstantPool> m_constantPool;
  std::unique_ptr<GraphRecorder> m_graphRecorder;

  DWORD m_callbackCookie = 0;
  int m_recordCommands = 0;
  bool m_restoreBackgroundProcessing = false;
  bool m_restoreStablePowerState = false;
  bool m_useCustomHeaps = false;
  bool m_needResetAllocator = false;

#ifndef DXCOMPILER_NONE
  Microsoft::WRL::ComPtr<IDxcUtils> m_dxcUtils;
  Microsoft::WRL::ComPtr<IDxcIncludeHandler> m_dxcIncludeHandler;
  Microsoft::WRL::ComPtr<IDxcCompiler3> m_dxcCompiler;
  Microsoft::WRL::ComPtr<IDxcValidator> m_dxcValidator;
  Microsoft::WRL::ComPtr<IDxcLibrary> m_dxcLibrary;
#endif

#if defined(INCLUDE_DXGI)
  Microsoft::WRL::ComPtr<IDXGISwapChain1> m_dummySwapChain;
#endif
};

}  // namespace dml
}  // namespace ctranslate2