#include "ctranslate2/primitives.h"

#ifdef CT2_WITH_DIRECTML

#include <DirectML.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "dml/backend_dml.h"
#include "type_dispatch.h"


using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
namespace dml {

// Helper function to get DML data type from C++ type
template <typename T>
DML_TENSOR_DATA_TYPE get_dml_data_type() {
  if constexpr (std::is_same_v<T, float>) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return DML_TENSOR_DATA_TYPE_INT32;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return DML_TENSOR_DATA_TYPE_INT8;
  } else {
    return DML_TENSOR_DATA_TYPE_FLOAT32;  // fallback
  }
}

// Helper to create a 1D tensor description
template <typename T>
DML_TENSOR_DESC create_tensor_desc(dim_t size,
                                   DML_BUFFER_TENSOR_DESC& buffer_desc) {
  static const UINT dims[] = {1, 1, 1, static_cast<UINT>(size)};
  static const UINT strides[] = {static_cast<UINT>(size),
                                 static_cast<UINT>(size),
                                 static_cast<UINT>(size), 1};

  buffer_desc.DataType = get_dml_data_type<T>();
  buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc.DimensionCount = 4;
  buffer_desc.Sizes = dims;
  buffer_desc.Strides = strides;
  buffer_desc.TotalTensorSizeInBytes = size * sizeof(T);
  buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC desc = {};
  desc.Type = DML_TENSOR_TYPE_BUFFER;
  desc.Desc = &buffer_desc;

  return desc;
}

// Helper to execute a DML operator
void execute_dml_operator(
    IDMLCompiledOperator* compiled_op,
    const std::vector<DML_BINDING_DESC>& input_bindings,
    const std::vector<DML_BINDING_DESC>& output_bindings) {
  auto device = get_d3d12_device();
  auto command_queue = get_command_queue();

  // Create command allocator and list
  ComPtr<ID3D12CommandAllocator> command_allocator;
  ComPtr<ID3D12GraphicsCommandList> command_list;

  device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                 IID_PPV_ARGS(&command_allocator));
  device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                            command_allocator.Get(), nullptr,
                            IID_PPV_ARGS(&command_list));

  // Create binding table
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op;
  binding_table_desc.CPUDescriptorHandle = {};  // Would need descriptor heap
  binding_table_desc.GPUDescriptorHandle = {};  // Would need descriptor heap
  binding_table_desc.SizeInDescriptors =
      static_cast<UINT>(input_bindings.size() + output_bindings.size());

  // This is a simplified version - real implementation would need proper
  // descriptor heap management and resource binding

  command_list->Close();

  ID3D12CommandList* command_lists[] = {command_list.Get()};
  command_queue->ExecuteCommandLists(1, command_lists);

  // Wait for completion
  ComPtr<ID3D12Fence> fence;
  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

  const UINT64 fence_value = 1;
  command_queue->Signal(fence.Get(), fence_value);

  if (fence->GetCompletedValue() < fence_value) {
    HANDLE event_handle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(fence_value, event_handle);
    WaitForSingleObject(event_handle, INFINITE);
    CloseHandle(event_handle);
  }
}

}  // namespace dml

// Template specializations for DirectML
template <>
template <typename T>
T primitives<Device::DirectML>::at(const T* x, dim_t index) {
  // For single element access, copy to CPU temporarily
  T result;
  // This would need proper DML buffer to CPU copy implementation
  return result;
}

template <>
template <typename T>
void primitives<Device::DirectML>::fill(T* x, T a, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  // Create fill operation using DML_ELEMENT_WISE_IDENTITY with a constant
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
  identity_desc.InputTensor = &input_desc;
  identity_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
  op_desc.Desc = &identity_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  // Execute with proper bindings (simplified)
  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::strided_fill(T* x,
                                                T a,
                                                dim_t inc_x,
                                                dim_t size) {
  // For strided operations, we'd need to use DML slicing operations
  // Simplified implementation
  fill(x, a, size);
}

template <>
template <typename T>
void primitives<Device::DirectML>::indexed_fill(T* x,
                                                T a,
                                                const int32_t* indices,
                                                dim_t num_indices) {
  // DML doesn't have direct indexed fill, would need scatter operation
  // Simplified implementation
  fill(x, a, num_indices);
}

template <>
template <typename T>
void primitives<Device::DirectML>::copy(const T* x, T* y, dim_t size) {
  auto device = dml::get_d3d12_device();
  auto command_queue = dml::get_command_queue();

  // Use D3D12 copy operation for simple buffer copy
  ComPtr<ID3D12CommandAllocator> command_allocator;
  ComPtr<ID3D12GraphicsCommandList> command_list;

  device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                 IID_PPV_ARGS(&command_allocator));
  device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                            command_allocator.Get(), nullptr,
                            IID_PPV_ARGS(&command_list));

  // This would need proper resource handling
  // command_list->CopyBufferRegion(dst_resource, 0, src_resource, 0, size *
  // sizeof(T));

  command_list->Close();

  ID3D12CommandList* command_lists[] = {command_list.Get()};
  command_queue->ExecuteCommandLists(1, command_lists);
}

template <>
template <typename U, typename V>
void primitives<Device::DirectML>::convert(const U* x, V* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<U>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<V>(size, output_buffer_desc);

  // Use DML cast operation
  DML_CAST_OPERATOR_DESC cast_desc = {};
  cast_desc.InputTensor = &input_desc;
  cast_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_CAST;
  op_desc.Desc = &cast_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
T primitives<Device::DirectML>::sum(const T* array, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(1, output_buffer_desc);

  // Use DML reduce sum operation
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
  reduce_desc.InputTensor = &input_desc;
  reduce_desc.OutputTensor = &output_desc;

  static const UINT axes[] = {3};  // Reduce along the last dimension
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = axes;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_REDUCE;
  op_desc.Desc = &reduce_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});

  // Return result (would need proper readback)
  return T{};
}

template <>
template <typename T>
dim_t primitives<Device::DirectML>::max_element(const T* array, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<int32_t>(1, output_buffer_desc);

  // Use DML argmax operation
  DML_ARGMAX_OPERATOR_DESC argmax_desc = {};
  argmax_desc.InputTensor = &input_desc;
  argmax_desc.OutputTensor = &output_desc;
  argmax_desc.AxisCount = 1;
  static const UINT axis = 3;
  argmax_desc.Axes = &axis;
  argmax_desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ARGMAX;
  op_desc.Desc = &argmax_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});

  return 0;  // Would need proper readback
}

template <>
template <typename T>
T primitives<Device::DirectML>::max(const T* array, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(1, output_buffer_desc);

  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_MAX;
  reduce_desc.InputTensor = &input_desc;
  reduce_desc.OutputTensor = &output_desc;

  static const UINT axes[] = {3};
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = axes;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_REDUCE;
  op_desc.Desc = &reduce_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});

  return T{};
}

template <>
template <typename T>
void primitives<Device::DirectML>::add(T a, const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc =
      dml::create_tensor_desc<T>(1, a_buffer_desc);  // Scalar
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  // Create element-wise add operation with broadcast scalar
  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &a_desc;
  add_desc.BTensor = &b_desc;
  add_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
  op_desc.Desc = &add_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::add(const T* a,
                                       const T* b,
                                       T* c,
                                       dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(size, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &a_desc;
  add_desc.BTensor = &b_desc;
  add_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
  op_desc.Desc = &add_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::sub(const T* a,
                                       const T* b,
                                       T* c,
                                       dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(size, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC sub_desc = {};
  sub_desc.ATensor = &a_desc;
  sub_desc.BTensor = &b_desc;
  sub_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_SUBTRACT;
  op_desc.Desc = &sub_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::mul(T a, const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc =
      dml::create_tensor_desc<T>(1, a_buffer_desc);  // Scalar
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_desc = {};
  mul_desc.ATensor = &a_desc;
  mul_desc.BTensor = &b_desc;
  mul_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
  op_desc.Desc = &mul_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::mul(const T* a,
                                       const T* b,
                                       T* c,
                                       dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(size, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_desc = {};
  mul_desc.ATensor = &a_desc;
  mul_desc.BTensor = &b_desc;
  mul_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
  op_desc.Desc = &mul_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

// Activation functions
template <>
template <typename T>
void primitives<Device::DirectML>::relu(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
  relu_desc.InputTensor = &input_desc;
  relu_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ACTIVATION_RELU;
  op_desc.Desc = &relu_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::sigmoid(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid_desc = {};
  sigmoid_desc.InputTensor = &input_desc;
  sigmoid_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
  op_desc.Desc = &sigmoid_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::tanh(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ACTIVATION_TANH_OPERATOR_DESC tanh_desc = {};
  tanh_desc.InputTensor = &input_desc;
  tanh_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ACTIVATION_TANH;
  op_desc.Desc = &tanh_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

// Matrix operations
template <>
template <typename In, typename Out>
void primitives<Device::DirectML>::gemm(bool a_is_packed,
                                        bool b_is_packed,
                                        bool transpose_a,
                                        bool transpose_b,
                                        dim_t m,
                                        dim_t n,
                                        dim_t k,
                                        float alpha,
                                        const In* a,
                                        dim_t lda,
                                        const In* b,
                                        dim_t ldb,
                                        float beta,
                                        Out* c,
                                        dim_t ldc,
                                        const Out* a_shift_compensation) {
  auto dml_device = dml::get_dml_device();

  // Create tensor descriptors for matrices
  static const UINT a_dims[] = {1, 1, static_cast<UINT>(transpose_a ? k : m),
                                static_cast<UINT>(transpose_a ? m : k)};
  static const UINT b_dims[] = {1, 1, static_cast<UINT>(transpose_b ? n : k),
                                static_cast<UINT>(transpose_b ? k : n)};
  static const UINT c_dims[] = {1, 1, static_cast<UINT>(m),
                                static_cast<UINT>(n)};

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  a_buffer_desc.DataType = dml::get_dml_data_type<In>();
  a_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  a_buffer_desc.DimensionCount = 4;
  a_buffer_desc.Sizes = a_dims;
  a_buffer_desc.Strides = nullptr;
  a_buffer_desc.TotalTensorSizeInBytes = m * k * sizeof(In);
  a_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC a_desc = {};
  a_desc.Type = DML_TENSOR_TYPE_BUFFER;
  a_desc.Desc = &a_buffer_desc;

  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  b_buffer_desc.DataType = dml::get_dml_data_type<In>();
  b_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  b_buffer_desc.DimensionCount = 4;
  b_buffer_desc.Sizes = b_dims;
  b_buffer_desc.Strides = nullptr;
  b_buffer_desc.TotalTensorSizeInBytes = k * n * sizeof(In);
  b_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC b_desc = {};
  b_desc.Type = DML_TENSOR_TYPE_BUFFER;
  b_desc.Desc = &b_buffer_desc;

  DML_BUFFER_TENSOR_DESC c_buffer_desc = {};
  c_buffer_desc.DataType = dml::get_dml_data_type<Out>();
  c_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  c_buffer_desc.DimensionCount = 4;
  c_buffer_desc.Sizes = c_dims;
  c_buffer_desc.Strides = nullptr;
  c_buffer_desc.TotalTensorSizeInBytes = m * n * sizeof(Out);
  c_buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  DML_TENSOR_DESC c_desc = {};
  c_desc.Type = DML_TENSOR_TYPE_BUFFER;
  c_desc.Desc = &c_buffer_desc;

  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = &a_desc;
  gemm_desc.BTensor = &b_desc;
  gemm_desc.CTensor = (beta != 0.0f) ? &c_desc : nullptr;
  gemm_desc.OutputTensor = &c_desc;
  gemm_desc.TransA =
      transpose_a ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
  gemm_desc.TransB =
      transpose_b ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
  gemm_desc.Alpha = alpha;
  gemm_desc.Beta = beta;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_GEMM;
  op_desc.Desc = &gemm_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

// Stub implementations for remaining methods
template <>
template <typename T>
void primitives<Device::DirectML>::add_batch_broadcast(const T* a,
                                                       const T* b,
                                                       T* c,
                                                       dim_t a_size,
                                                       dim_t b_size) {
  // Implement using DML broadcast operations
  add(a, b, c, std::min(a_size, b_size));
}

template <>
template <typename T>
void primitives<Device::DirectML>::add_depth_broadcast(const T* a,
                                                       const T* b,
                                                       T* c,
                                                       dim_t a_size,
                                                       dim_t b_size) {
  // Implement using DML broadcast operations
  add(a, b, c, std::min(a_size, b_size));
}

template <>
template <typename T>
void primitives<Device::DirectML>::mul_batch_broadcast(const T* a,
                                                       const T* b,
                                                       T* c,
                                                       dim_t a_size,
                                                       dim_t b_size) {
  // Implement using DML broadcast operations
  mul(a, b, c, std::min(a_size, b_size));
}

template <>
template <typename T>
void primitives<Device::DirectML>::max(T a, const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(1, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MAX_OPERATOR_DESC max_desc = {};
  max_desc.ATensor = &a_desc;
  max_desc.BTensor = &b_desc;
  max_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MAX;
  op_desc.Desc = &max_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::max(const T* a,
                                       const T* b,
                                       T* c,
                                       dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(size, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MAX_OPERATOR_DESC max_desc = {};
  max_desc.ATensor = &a_desc;
  max_desc.BTensor = &b_desc;
  max_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MAX;
  op_desc.Desc = &max_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::min(T a, const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(1, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MIN_OPERATOR_DESC min_desc = {};
  min_desc.ATensor = &a_desc;
  min_desc.BTensor = &b_desc;
  min_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MIN;
  op_desc.Desc = &min_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::min(const T* a,
                                       const T* b,
                                       T* c,
                                       dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC a_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC b_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC a_desc = dml::create_tensor_desc<T>(size, a_buffer_desc);
  DML_TENSOR_DESC b_desc = dml::create_tensor_desc<T>(size, b_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_MIN_OPERATOR_DESC min_desc = {};
  min_desc.ATensor = &a_desc;
  min_desc.BTensor = &b_desc;
  min_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MIN;
  op_desc.Desc = &min_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

// Remaining stub implementations for completeness
template <>
template <typename T>
void primitives<Device::DirectML>::gelu(const T* x, T* y, dim_t size) {
  // DML doesn't have native GELU, would need to implement as composite
  // operation
  sigmoid(x, y, size);  // Placeholder
}

template <>
template <typename T>
void primitives<Device::DirectML>::gelu_tanh(const T* x, T* y, dim_t size) {
  tanh(x, y, size);  // Placeholder
}

template <>
template <typename T>
void primitives<Device::DirectML>::gelu_sigmoid(const T* x, T* y, dim_t size) {
  sigmoid(x, y, size);  // Placeholder
}

template <>
template <typename T>
void primitives<Device::DirectML>::swish(const T* x, T* y, dim_t size) {
  // Swish = x * sigmoid(x), would need composite operation
  sigmoid(x, y, size);  // Placeholder
}

template <>
template <typename T>
void primitives<Device::DirectML>::exp(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_EXP_OPERATOR_DESC exp_desc = {};
  exp_desc.InputTensor = &input_desc;
  exp_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_EXP;
  op_desc.Desc = &exp_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::log(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_LOG_OPERATOR_DESC log_desc = {};
  log_desc.InputTensor = &input_desc;
  log_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_LOG;
  op_desc.Desc = &log_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::sin(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_SIN_OPERATOR_DESC sin_desc = {};
  sin_desc.InputTensor = &input_desc;
  sin_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_SIN;
  op_desc.Desc = &sin_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

template <>
template <typename T>
void primitives<Device::DirectML>::cos(const T* x, T* y, dim_t size) {
  auto dml_device = dml::get_dml_device();

  DML_BUFFER_TENSOR_DESC input_buffer_desc = {};
  DML_BUFFER_TENSOR_DESC output_buffer_desc = {};

  DML_TENSOR_DESC input_desc =
      dml::create_tensor_desc<T>(size, input_buffer_desc);
  DML_TENSOR_DESC output_desc =
      dml::create_tensor_desc<T>(size, output_buffer_desc);

  DML_ELEMENT_WISE_COS_OPERATOR_DESC cos_desc = {};
  cos_desc.InputTensor = &input_desc;
  cos_desc.OutputTensor = &output_desc;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_ELEMENT_WISE_COS;
  op_desc.Desc = &cos_desc;

  ComPtr<IDMLOperator> op;
  dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op));

  ComPtr<IDMLCompiledOperator> compiled_op;
  dml_device->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE,
                              IID_PPV_ARGS(&compiled_op));

  dml::execute_dml_operator(compiled_op.Get(), {}, {});
}

// Additional required implementations (stubs for now)
template <>
template <typename T>
T primitives<Device::DirectML>::amax(const T* array, dim_t size) {
  return max(array, size);
}

template <>
template <typename T>
float primitives<Device::DirectML>::logsumexp(const T* x, dim_t size) {
  // Would need composite operation: log(sum(exp(x)))
  return 0.0f;
}

template <>
template <typename T>
void primitives<Device::DirectML>::penalize_previous_tokens(
    T* scores,
    const T* previous_scores,
    const int32_t* previous_ids,
    T penalty,
    dim_t batch_size,
    dim_t length,
    dim_t vocabulary_size) {
  // Complex operation requiring scatter/gather - stub implementation
}

template <>
void primitives<Device::DirectML>::prepare_length_mask(const int32_t* lengths,
                                                       dim_t batch_size,
                                                       dim_t num_heads,
                                                       dim_t num_queries,
                                                       bool mask_future,
                                                       bool multi_query,
                                                       int32_t* mask) {
  // Complex mask preparation - stub implementation
}

template <>
template <typename T>
void primitives<Device::DirectML>::transpose_2d(const T* a,
                                                const dim_t* dims,
                                                T* b) {
  // Use DML slice/reshape operations
}

template <>
template <typename T>
void primitives<Device::DirectML>::transpose_3d(const T* a,
                                                const dim_t* dims,
                                                const dim_t* perm,
                                                T* b) {
  // Use DML slice/reshape operations
}

template <>
template <typename T>
void primitives<Device::DirectML>::transpose_4d(const T* a,
                                                const dim_t* dims,
                                                const dim_t* perm,
                                                T* b) {
  // Use DML slice/reshape operations
}

template <>
void primitives<Device::DirectML>::compute_u8_compensation(
    const int8_t* b,
    bool transpose_b,
    dim_t k,
    dim_t n,
    float alpha,
    int32_t* compensation) {
  // Quantization compensation - stub implementation
}

template <>
template <typename T>
dim_t primitives<Device::DirectML>::gemm_pack_b(const T* b,
                                                const bool transpose_b,
                                                const dim_t k,
                                                const dim_t n,
                                                const float alpha,
                                                T* dest) {
  // Matrix packing - return 0 indicating no packing support
  return 0;
}

template <>
template <typename In, typename Out>
void primitives<Device::DirectML>::gemm_batch_strided(bool transpose_a,
                                                      bool transpose_b,
                                                      dim_t m,
                                                      dim_t n,
                                                      dim_t k,
                                                      float alpha,
                                                      const In* a,
                                                      dim_t lda,
                                                      dim_t stridea,
                                                      const In* b,
                                                      dim_t ldb,
                                                      dim_t strideb,
                                                      float beta,
                                                      Out* c,
                                                      dim_t ldc,
                                                      dim_t stridec,
                                                      dim_t batch_size) {
  // Batched GEMM - would need loop over batch dimension
  gemm(false, false, transpose_a, transpose_b, m, n, k, alpha, a, lda, b, ldb,
       beta, c, ldc);
}

// Cross-device copy operations
template <>
template <typename T>
void cross_device_primitives<Device::CPU, Device::DirectML>::copy(const T* x,
                                                                  T* y,
                                                                  dim_t size) {
  // Copy from CPU to DML (D3D12) buffer
  auto device = dml::get_d3d12_device();
  // Would need proper D3D12 upload heap and copy operation
}

template <>
template <typename T>
void cross_device_primitives<Device::DirectML, Device::CPU>::copy(const T* x,
                                                                  T* y,
                                                                  dim_t size) {
  // Copy from DML (D3D12) buffer to CPU
  auto device = dml::get_d3d12_device();
  // Would need proper D3D12 readback heap and copy operation
}

// Explicit template instantiations
#define DECLARE_IMPL(T)                                                       \
  template T primitives<Device::DirectML>::at(const T* x, dim_t index);       \
  template void primitives<Device::DirectML>::fill(T* x, T a, dim_t size);    \
  template void primitives<Device::DirectML>::strided_fill(                   \
      T* x, T a, dim_t inc_x, dim_t size);                                    \
  template void primitives<Device::DirectML>::indexed_fill(                   \
      T*, T, const int32_t*, dim_t);                                          \
  template void primitives<Device::DirectML>::copy<T>(const T* x, T* y,       \
                                                      dim_t size);            \
  template T primitives<Device::DirectML>::sum(const T* array, dim_t size);   \
  template dim_t primitives<Device::DirectML>::max_element(const T* array,    \
                                                           dim_t size);       \
  template T primitives<Device::DirectML>::max(const T* array, dim_t size);   \
  template T primitives<Device::DirectML>::amax(const T* array, dim_t size);  \
  template void primitives<Device::DirectML>::add(T a, const T* x, T* y,      \
                                                  dim_t size);                \
  template void primitives<Device::DirectML>::add(const T* a, const T* b,     \
                                                  T* c, dim_t size);          \
  template void primitives<Device::DirectML>::add_batch_broadcast(            \
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);              \
  template void primitives<Device::DirectML>::add_depth_broadcast(            \
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);              \
  template void primitives<Device::DirectML>::sub(const T* a, const T* b,     \
                                                  T* c, dim_t size);          \
  template void primitives<Device::DirectML>::min(T a, const T* x, T* y,      \
                                                  dim_t size);                \
  template void primitives<Device::DirectML>::min(const T* a, const T* b,     \
                                                  T* c, dim_t size);          \
  template void primitives<Device::DirectML>::max(T a, const T* x, T* y,      \
                                                  dim_t size);                \
  template void primitives<Device::DirectML>::max(const T* a, const T* b,     \
                                                  T* c, dim_t size);          \
  template void primitives<Device::DirectML>::mul(T a, const T* x, T* y,      \
                                                  dim_t size);                \
  template void primitives<Device::DirectML>::mul(const T* a, const T* b,     \
                                                  T* c, dim_t size);          \
  template void primitives<Device::DirectML>::mul_batch_broadcast(            \
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size);              \
  template void primitives<Device::DirectML>::penalize_previous_tokens(       \
      T*, const T*, const int32_t*, T, dim_t, dim_t, dim_t);                  \
  template void primitives<Device::DirectML>::transpose_2d(                   \
      const T* a, const dim_t* dims, T* b);                                   \
  template void primitives<Device::DirectML>::transpose_3d(                   \
      const T* a, const dim_t* dims, const dim_t* perm, T* b);                \
  template void primitives<Device::DirectML>::transpose_4d(                   \
      const T* a, const dim_t* dims, const dim_t* perm, T* b);                \
  template void                                                               \
  cross_device_primitives<Device::CPU, Device::DirectML>::copy<T>(const T*,   \
                                                                  T*, dim_t); \
  template void                                                               \
  cross_device_primitives<Device::DirectML, Device::CPU>::copy<T>(const T*,   \
                                                                  T*, dim_t);

DECLARE_ALL_TYPES(DECLARE_IMPL)

#define DECLARE_FLOAT_IMPL(T)                                                 \
  template void primitives<Device::DirectML>::relu(const T*, T*, dim_t);      \
  template void primitives<Device::DirectML>::gelu(const T*, T*, dim_t);      \
  template void primitives<Device::DirectML>::gelu_tanh(const T*, T*, dim_t); \
  template void primitives<Device::DirectML>::gelu_sigmoid(const T*, T*,      \
                                                           dim_t);            \
  template void primitives<Device::DirectML>::sigmoid(const T*, T*, dim_t);   \
  template void primitives<Device::DirectML>::swish(const T*, T*, dim_t);     \
  template float primitives<Device::DirectML>::logsumexp(const T*, dim_t);    \
  template void primitives<Device::DirectML>::sin(const T*, T*, dim_t);       \
  template void primitives<Device::DirectML>::cos(const T*, T*, dim_t);       \
  template void primitives<Device::DirectML>::tanh(const T*, T*, dim_t);      \
  template void primitives<Device::DirectML>::exp(const T*, T*, dim_t);       \
  template void primitives<Device::DirectML>::log(const T*, T*, dim_t);

DECLARE_FLOAT_IMPL(float)
DECLARE_FLOAT_IMPL(float16_t)
DECLARE_FLOAT_IMPL(bfloat16_t)

// GEMM specializations
template void primitives<Device::DirectML>::gemm<float, float>(bool,
                                                               bool,
                                                               bool,
                                                               bool,
                                                               dim_t,
                                                               dim_t,
                                                               dim_t,
                                                               float,
                                                               const float*,
                                                               dim_t,
                                                               const float*,
                                                               dim_t,
                                                               float,
                                                               float*,
                                                               dim_t,
                                                               const float*);
template void primitives<Device::DirectML>::gemm<float16_t, float16_t>(
    bool,
    bool,
    bool,
    bool,
    dim_t,
    dim_t,
    dim_t,
    float,
    const float16_t*,
    dim_t,
    const float16_t*,
    dim_t,
    float,
    float16_t*,
    dim_t,
    const float16_t*);
template void primitives<Device::DirectML>::gemm<int8_t, int32_t>(
    bool,
    bool,
    bool,
    bool,
    dim_t,
    dim_t,
    dim_t,
    float,
    const int8_t*,
    dim_t,
    const int8_t*,
    dim_t,
    float,
    int32_t*,
    dim_t,
    const int32_t*);

template void primitives<Device::DirectML>::gemm_batch_strided<float, float>(
    bool,
    bool,
    dim_t,
    dim_t,
    dim_t,
    float,
    const float*,
    dim_t,
    dim_t,
    const float*,
    dim_t,
    dim_t,
    float,
    float*,
    dim_t,
    dim_t,
    dim_t);
template void
primitives<Device::DirectML>::gemm_batch_strided<float16_t, float16_t>(
    bool,
    bool,
    dim_t,
    dim_t,
    dim_t,
    float,
    const float16_t*,
    dim_t,
    dim_t,
    const float16_t*,
    dim_t,
    dim_t,
    float,
    float16_t*,
    dim_t,
    dim_t,
    dim_t);

template void primitives<Device::DirectML>::convert<float, float16_t>(
    const float*,
    float16_t*,
    dim_t);
template void primitives<Device::DirectML>::convert<float16_t, float>(
    const float16_t*,
    float*,
    dim_t);
template void primitives<Device::DirectML>::convert<float, bfloat16_t>(
    const float*,
    bfloat16_t*,
    dim_t);
template void primitives<Device::DirectML>::convert<bfloat16_t, float>(
    const bfloat16_t*,
    float*,
    dim_t);
template void primitives<Device::DirectML>::convert<float16_t, bfloat16_t>(
    const float16_t*,
    bfloat16_t*,
    dim_t);
template void primitives<Device::DirectML>::convert<bfloat16_t, float16_t>(
    const bfloat16_t*,
    float16_t*,
    dim_t);

}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML