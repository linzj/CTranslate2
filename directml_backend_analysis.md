# DirectML Backend Support in CTranslate2

## Overview

CTranslate2 is a C++ and Python library for efficient inference with Transformer models that supports multiple backends including CPU, CUDA, and **DirectML**. DirectML is Microsoft's hardware-accelerated DirectX 12 library for machine learning that provides GPU acceleration on Windows systems, particularly useful for AMD GPUs and older NVIDIA GPUs where CUDA support might be limited.

## Architecture and Design

### 1. Build System Integration

The DirectML backend is integrated through CMake with compile-time configuration:

```cmake
option(WITH_DIRECTML "Compile with DirectML backend" OFF)
```

When enabled, the build system:
- Adds the `CT2_WITH_DIRECTML` preprocessor definition
- Includes DirectML-specific source files from `src/dml/`
- Links against Microsoft.AI.DirectML NuGet package (version 1.15.4)
- Sets up DirectX headers and dependencies

### 2. Device Abstraction Layer

#### Device Enumeration and Selection
The DirectML backend integrates into CTranslate2's device abstraction through several key components:

**Device Types**: DirectML is represented as `Device::DirectML` alongside `Device::CPU` and `Device::CUDA`.

**Device Detection**: The `has_directml_device()` function performs sophisticated adapter enumeration:
- Uses DXGI factory to enumerate available adapters
- Prioritizes dedicated graphics adapters over integrated ones
- Validates D3D12 device creation capability
- Implements fallback logic for adapter selection

**Device Dispatch**: A macro-based system (`DEVICE_DISPATCH`) handles runtime dispatch to appropriate backend code based on compilation flags and device availability.

#### Initialization and Management
```cpp
void initialize_directml() {
    if (!g_device && !has_directml_device()) {
        THROW_RUNTIME_ERROR("DirectML device not found or initialization failed.");
    }
}
```

The backend uses a singleton pattern with:
- One-time initialization using `std::once_flag`
- Thread-safe device access through `GuardedPtr<Device>`
- Automatic cleanup and resource management

### 3. Core DirectML Implementation

#### Key Components

**DXDevice Wrapper** (`src/dml/dxdevice.h`):
- Encapsulates D3D12 device, command queue, and DirectML device
- Manages GPU memory allocation and command execution
- Handles synchronization and resource barriers

**Memory Management**:
- Custom allocator (`allocator_dml.cc`) for GPU memory
- Bucketized buffer allocator for efficient memory reuse
- Integration with CTranslate2's storage system

**Graph Recording and Execution**:
- Command list recording for batched operations
- Graph-based optimization for operator fusion
- Asynchronous execution with proper synchronization

#### Utility Infrastructure

**DML Utilities** (`src/dml/dml_utils.h`):
- `DmlTensorDescBundle`: Manages DirectML tensor descriptors with automatic lifetime management
- `DmlOperatorDescBundle`: Handles operator description creation and memory management
- Type conversion utilities between CTranslate2 and DirectML data types
- Broadcasting and dimension manipulation helpers

### 4. Operator Implementation

The DirectML backend provides specialized implementations for numerous operations:

#### Core Operations
- **Matrix Operations**: GEMM, convolution, layer normalization
- **Element-wise Operations**: Add, multiply, activation functions (ReLU, GELU, Swish)
- **Reduction Operations**: Mean, softmax, top-k
- **Data Movement**: Gather, tile, concat/split/slice
- **Quantization**: Quantize/dequantize operations for INT8/INT16 support

#### Advanced Operations
- **Attention Mechanisms**: Flash attention, rotary embeddings, ALiBi
- **Sampling**: Multinomial sampling, Gumbel max
- **Specialized**: AWQ quantization support, NCCL operations

#### Implementation Pattern
Each operation follows a consistent pattern:
```cpp
#ifdef CT2_WITH_DIRECTML
template void OperationName::compute<Device::DirectML, DataType>(
    const StorageView& input,
    StorageView& output) const;
#endif
```

### 5. Memory and Resource Management

#### Storage Integration
DirectML tensors integrate seamlessly with CTranslate2's `StorageView` system:
- Automatic device-to-device memory transfers
- Lazy allocation and copy-on-write semantics
- Support for various data types (FP32, FP16, INT8, etc.)

#### Resource Pooling
- Operator caching to avoid recompilation
- Command list recycling
- Memory pool management for reduced allocation overhead

### 6. Performance Optimizations

#### Graph Optimization
- Operator fusion where supported by DirectML
- Dead code elimination
- Memory layout optimization

#### Execution Efficiency
- Batched operations to amortize dispatch overhead
- Asynchronous execution with proper dependency tracking
- Minimized CPU-GPU synchronization points

### 7. Integration Points

#### Device Synchronization
```cpp
void synchronize_device(Device device, int index) {
#ifdef CT2_WITH_DIRECTML
    if (device == Device::DirectML) {
        dml::get_device()->ExecuteCommandListAndWait();
    }
#endif
}
```

#### Cross-Device Operations
The storage system handles automatic data movement between devices:
- CPU ↔ DirectML transfers
- Efficient memory mapping where possible
- Fallback through CPU for unsupported device pairs

## Key Benefits

### 1. Hardware Compatibility
- Supports AMD GPUs through DirectML
- Works with older NVIDIA GPUs
- Leverages Windows GPU drivers and optimizations

### 2. Performance
- Hardware-accelerated inference on Windows
- Optimized memory usage patterns
- Efficient operator implementations

### 3. Integration
- Seamless fallback from DirectML to CPU
- Consistent API across all backends
- Automatic device selection ("auto" mode)

## Limitations and Considerations

### 1. Platform Dependency
- Windows-only support (DirectX 12 requirement)
- Requires compatible GPU drivers

### 2. Feature Parity
- Some operations may have different performance characteristics
- Subset of operations compared to CUDA backend
- Limited by DirectML's operator support

### 3. Development Complexity
- Additional build dependencies (DirectX SDK, DirectML)
- Platform-specific testing requirements
- Resource management complexity

## Usage

### Build Configuration
```bash
cmake -DWITH_DIRECTML=ON ..
```

### Runtime Usage
```python
import ctranslate2
# Automatically selects DirectML if available
translator = ctranslate2.Translator(model_path, device="auto")
# Or explicitly use DirectML
translator = ctranslate2.Translator(model_path, device="directml")
```

## Conclusion

The DirectML backend in CTranslate2 provides a comprehensive solution for GPU acceleration on Windows systems, particularly valuable for AMD GPU users and environments where CUDA is not available. The implementation demonstrates sophisticated engineering with proper abstraction layers, efficient resource management, and extensive operator coverage while maintaining API compatibility with other backends.

The architecture successfully balances performance, maintainability, and platform integration, making CTranslate2 accessible to a broader range of hardware configurations and deployment scenarios.