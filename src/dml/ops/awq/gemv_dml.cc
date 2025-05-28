#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/awq/gemv.h"
#include "dml/backend_dml.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

namespace {
// Helper function to create DML tensor descriptor from StorageView
DML_BUFFER_TENSOR_DESC create_buffer_tensor_desc(
    const StorageView& tensor,
    DML_TENSOR_DATA_TYPE data_type) {
  DML_BUFFER_TENSOR_DESC desc = {};
  desc.DataType = data_type;
  desc.Flags = DML_TENSOR_FLAG_NONE;
  desc.DimensionCount = static_cast<UINT>(tensor.rank());

  // Convert dimensions to UINT array
  static thread_local std::vector<UINT> sizes;
  sizes.clear();
  sizes.reserve(tensor.rank());
  for (dim_t i = 0; i < tensor.rank(); ++i) {
    sizes.push_back(static_cast<UINT>(tensor.dim(i)));
  }
  desc.Sizes = sizes.data();

  desc.Strides = nullptr;  // Use default strides
  desc.TotalTensorSizeInBytes = tensor.size() * tensor.item_size();
  desc.GuaranteedBaseOffsetAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT;

  return desc;
}

// Helper function to get DML data type from StorageView
DML_TENSOR_DATA_TYPE get_dml_data_type(const StorageView& tensor) {
  switch (tensor.dtype()) {
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case DataType::INT8:
      return DML_TENSOR_DATA_TYPE_INT8;
    default:
      throw std::runtime_error("Unsupported data type for DirectML");
  }
}

// Helper function to dequantize weights using DirectML
void dequantize_weights_dml(const StorageView& quantized_weights,
                            const StorageView& scales,
                            const StorageView& zero_points,
                            StorageView& dequantized_weights,
                            dim_t group_size) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Reshape quantized weights from packed int4 to individual values
  // Each int32 contains 8 int4 values
  dim_t oc = quantized_weights.dim(0);
  dim_t ic_packed = quantized_weights.dim(1);
  dim_t ic = ic_packed * 8;  // 8 int4 values per int32

  // Create intermediate tensor for unpacked int4 values
  StorageView unpacked_weights({oc, ic}, DataType::INT8, Device::DirectML);

  // Create tensor descriptors for unpacking operation
  auto packed_desc =
      create_buffer_tensor_desc(quantized_weights, DML_TENSOR_DATA_TYPE_INT32);
  auto unpacked_desc =
      create_buffer_tensor_desc(unpacked_weights, DML_TENSOR_DATA_TYPE_INT8);

  // TODO: Implement int4 unpacking operation using custom DML operator or
  // compute shader For now, we'll use a simplified approach assuming weights
  // are already in correct format

  // Resize dequantized weights tensor
  dequantized_weights.resize({oc, ic});

  // Create tensor descriptors for dequantization
  auto input_desc =
      create_buffer_tensor_desc(unpacked_weights, DML_TENSOR_DATA_TYPE_INT8);
  auto scale_desc =
      create_buffer_tensor_desc(scales, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto zero_desc =
      create_buffer_tensor_desc(zero_points, DML_TENSOR_DATA_TYPE_INT8);
  auto output_desc = create_buffer_tensor_desc(dequantized_weights,
                                               DML_TENSOR_DATA_TYPE_FLOAT16);

  DML_TENSOR_DESC input_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &input_desc};
  DML_TENSOR_DESC scale_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &scale_desc};
  DML_TENSOR_DESC zero_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &zero_desc};
  DML_TENSOR_DESC output_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &output_desc};

  // Create dequantize linear operator
  DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequant_desc = {};
  dequant_desc.InputTensor = &input_tensor_desc;
  dequant_desc.ScaleTensor = &scale_tensor_desc;
  dequant_desc.ZeroPointTensor = &zero_tensor_desc;
  dequant_desc.OutputTensor = &output_tensor_desc;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR,
                               &dequant_desc};

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create binding table
  DML_BINDING_PROPERTIES binding_props = compiled_op->GetBindingProperties();

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();

  dml_device->CreateBindingTable(&binding_table_desc,
                                 IID_PPV_ARGS(&binding_table));

  // Bind inputs and outputs
  DML_BUFFER_BINDING input_binding = {
      reinterpret_cast<ID3D12Resource*>(
          const_cast<void*>(unpacked_weights.buffer())),
      0,
      static_cast<UINT64>(unpacked_weights.size() *
                          unpacked_weights.item_size())};
  DML_BUFFER_BINDING scale_binding = {
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(scales.buffer())), 0,
      static_cast<UINT64>(scales.size() * scales.item_size())};
  DML_BUFFER_BINDING zero_binding = {
      reinterpret_cast<ID3D12Resource*>(
          const_cast<void*>(zero_points.buffer())),
      0, static_cast<UINT64>(zero_points.size() * zero_points.item_size())};
  DML_BUFFER_BINDING output_binding = {
      reinterpret_cast<ID3D12Resource*>(dequantized_weights.buffer()), 0,
      static_cast<UINT64>(dequantized_weights.size() *
                          dequantized_weights.item_size())};

  DML_BINDING_DESC input_bindings[] = {
      {DML_BINDING_TYPE_BUFFER, &input_binding},
      {DML_BINDING_TYPE_BUFFER, &scale_binding},
      {DML_BINDING_TYPE_BUFFER, &zero_binding}};
  DML_BINDING_DESC output_bindings[] = {
      {DML_BINDING_TYPE_BUFFER, &output_binding}};

  binding_table->BindInputs(3, input_bindings);
  binding_table->BindOutputs(1, output_bindings);

  // Record dispatch
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
}

void execute_dml_operator(IDMLCompiledOperator* compiled_op,
                          const std::vector<const StorageView*>& inputs,
                          const std::vector<StorageView*>& outputs) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op;

  dml_device->CreateBindingTable(&binding_table_desc,
                                 IID_PPV_ARGS(&binding_table));

  // Create input bindings
  std::vector<DML_BUFFER_BINDING> input_buffer_bindings;
  std::vector<DML_BINDING_DESC> input_bindings;

  for (const auto* input : inputs) {
    DML_BUFFER_BINDING buffer_binding = {
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input->buffer())),
        0, static_cast<UINT64>(input->size() * input->item_size())};
    input_buffer_bindings.push_back(buffer_binding);

    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER,
                                     &input_buffer_bindings.back()};
    input_bindings.push_back(binding_desc);
  }

  // Create output bindings
  std::vector<DML_BUFFER_BINDING> output_buffer_bindings;
  std::vector<DML_BINDING_DESC> output_bindings;

  for (auto* output : outputs) {
    DML_BUFFER_BINDING buffer_binding = {
        reinterpret_cast<ID3D12Resource*>(output->buffer()), 0,
        static_cast<UINT64>(output->size() * output->item_size())};
    output_buffer_bindings.push_back(buffer_binding);

    DML_BINDING_DESC binding_desc = {DML_BINDING_TYPE_BUFFER,
                                     &output_buffer_bindings.back()};
    output_bindings.push_back(binding_desc);
  }

  // Bind inputs and outputs
  binding_table->BindInputs(static_cast<UINT>(input_bindings.size()),
                            input_bindings.data());
  binding_table->BindOutputs(static_cast<UINT>(output_bindings.size()),
                             output_bindings.data());

  // Record dispatch
  device->RecordDispatch(compiled_op, binding_table.Get());

  // Execute immediately for this operation
  device->ExecuteCommandList();
}

void slice_tensor_k_dimension(const StorageView& input,
                              StorageView& output,
                              dim_t k_start,
                              dim_t k_end) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Create tensor descriptors
  auto input_desc =
      create_buffer_tensor_desc(input, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto output_desc =
      create_buffer_tensor_desc(output, DML_TENSOR_DATA_TYPE_FLOAT16);

  DML_TENSOR_DESC input_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &input_desc};
  DML_TENSOR_DESC output_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &output_desc};

  // Create slice operator
  std::vector<UINT> offsets(input.rank(), 0);
  std::vector<UINT> sizes;
  std::vector<UINT> strides(input.rank(), 1);

  for (dim_t i = 0; i < input.rank(); ++i) {
    if (i == input.rank() - 1) {  // K dimension (last dimension)
      offsets[i] = static_cast<UINT>(k_start);
      sizes.push_back(static_cast<UINT>(k_end - k_start));
    } else {
      sizes.push_back(static_cast<UINT>(input.dim(i)));
    }
  }

  DML_SLICE_OPERATOR_DESC slice_desc = {};
  slice_desc.InputTensor = &input_tensor_desc;
  slice_desc.OutputTensor = &output_tensor_desc;
  slice_desc.DimensionCount = static_cast<UINT>(input.rank());
  slice_desc.Offsets = offsets.data();
  slice_desc.Sizes = sizes.data();
  slice_desc.Strides = strides.data();

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE, &slice_desc};

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create and execute binding
  execute_dml_operator(compiled_op.Get(), {&input}, {&output});
}

void perform_partial_gemv(const StorageView& a_slice,
                          const StorageView& b_slice,
                          StorageView& c_slice) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Reshape for matrix multiplication if needed
  StorageView reshaped_a = a_slice;
  StorageView reshaped_c = c_slice;

  if (a_slice.rank() == 3) {
    reshaped_a.reshape({a_slice.dim(0) * a_slice.dim(1), a_slice.dim(2)});
    reshaped_c.reshape({c_slice.dim(0) * c_slice.dim(1), c_slice.dim(2)});
  }

  // Create tensor descriptors
  auto a_desc =
      create_buffer_tensor_desc(reshaped_a, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto b_desc =
      create_buffer_tensor_desc(b_slice, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto c_desc =
      create_buffer_tensor_desc(reshaped_c, DML_TENSOR_DATA_TYPE_FLOAT16);

  DML_TENSOR_DESC a_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &a_desc};
  DML_TENSOR_DESC b_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &b_desc};
  DML_TENSOR_DESC c_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &c_desc};

  // Create GEMM operator
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = &a_tensor_desc;
  gemm_desc.BTensor = &b_tensor_desc;
  gemm_desc.CTensor = nullptr;
  gemm_desc.OutputTensor = &c_tensor_desc;
  gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
  gemm_desc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
  gemm_desc.Alpha = 1.0f;
  gemm_desc.Beta = 0.0f;
  gemm_desc.FusedActivation = nullptr;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GEMM, &gemm_desc};

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Execute the operation
  execute_dml_operator(compiled_op.Get(), {&reshaped_a, &b_slice},
                       {&reshaped_c});
}

void reduce_split_k_results(StorageView& c) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  dim_t split_k_iters = c.dim(0);

  // Create final output tensor (without split-k dimension)
  StorageView final_output;
  if (c.rank() == 3) {
    final_output.resize({c.dim(1), c.dim(2)});
  } else {
    final_output.resize({c.dim(1), c.dim(2), c.dim(3)});
  }

  // Sum across split-k dimension (dimension 0)
  auto input_desc = create_buffer_tensor_desc(c, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto output_desc =
      create_buffer_tensor_desc(final_output, DML_TENSOR_DATA_TYPE_FLOAT16);

  DML_TENSOR_DESC input_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &input_desc};
  DML_TENSOR_DESC output_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &output_desc};

  // Create reduce sum operator
  UINT reduce_axes[] = {0};  // Reduce along split-k dimension

  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
  reduce_desc.InputTensor = &input_tensor_desc;
  reduce_desc.OutputTensor = &output_tensor_desc;
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = reduce_axes;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Execute the reduction
  execute_dml_operator(compiled_op.Get(), {&c}, {&final_output});

  // Copy final result back to c with correct shape
  c = std::move(final_output);
}

// Helper function to zero out a tensor using DirectML
void zero_tensor_dml(StorageView& tensor) {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Create tensor descriptor
  auto tensor_desc =
      create_buffer_tensor_desc(tensor, DML_TENSOR_DATA_TYPE_FLOAT16);
  DML_TENSOR_DESC output_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &tensor_desc};

  // Create fill value constant operator to fill with zeros
  DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc = {};
  fill_desc.OutputTensor = &output_tensor_desc;
  fill_desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
  fill_desc.Value.Float32 = 0.0f;  // Zero value

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_FILL_VALUE_CONSTANT, &fill_desc};

  // Get or create compiled operator
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create binding table
  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();

  dml_device->CreateBindingTable(&binding_table_desc,
                                 IID_PPV_ARGS(&binding_table));

  // Bind output
  DML_BUFFER_BINDING output_binding = {
      reinterpret_cast<ID3D12Resource*>(tensor.buffer()), 0,
      static_cast<UINT64>(tensor.size() * tensor.item_size())};

  DML_BINDING_DESC output_bind_desc = {DML_BINDING_TYPE_BUFFER,
                                       &output_binding};
  binding_table->BindOutputs(1, &output_bind_desc);

  // Record dispatch
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());
}
}  // namespace

template <>
void GemvAwq::compute_gemv<Device::DirectML, float16_t, int>(
    const StorageView& a,
    const StorageView& b,
    const StorageView& scale,
    const StorageView& zero,
    StorageView& c) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Get dimensions
  dim_t num_in_channels = a.dim(-1);
  dim_t num_in_feats = a.size() / num_in_channels;
  dim_t num_out_channels = b.dim(0);
  dim_t group_size = num_in_channels / scale.dim(-1);

  // Resize output tensor
  if (a.rank() == 2)
    c.resize({num_in_feats, num_out_channels});
  else if (a.rank() == 3)
    c.resize({a.dim(0), a.dim(1), num_out_channels});

  // Dequantize weights
  StorageView dequantized_b({num_out_channels, num_in_channels},
                            DataType::FLOAT16, Device::DirectML);
  dequantize_weights_dml(b, scale, zero, dequantized_b, group_size);

  // Reshape input for matrix multiplication
  StorageView reshaped_a = a;
  if (a.rank() == 3) {
    reshaped_a.reshape({a.dim(0) * a.dim(1), a.dim(2)});
  }

  // Create tensor descriptors for GEMM
  auto a_desc =
      create_buffer_tensor_desc(reshaped_a, DML_TENSOR_DATA_TYPE_FLOAT16);
  auto b_desc =
      create_buffer_tensor_desc(dequantized_b, DML_TENSOR_DATA_TYPE_FLOAT16);

  StorageView reshaped_c;
  if (c.rank() == 3) {
    reshaped_c = c;
    reshaped_c.reshape({c.dim(0) * c.dim(1), c.dim(2)});
  } else {
    reshaped_c = c;
  }
  auto c_desc =
      create_buffer_tensor_desc(reshaped_c, DML_TENSOR_DATA_TYPE_FLOAT16);

  DML_TENSOR_DESC a_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &a_desc};
  DML_TENSOR_DESC b_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &b_desc};
  DML_TENSOR_DESC c_tensor_desc = {DML_TENSOR_TYPE_BUFFER, &c_desc};

  // Create GEMM operator
  DML_GEMM_OPERATOR_DESC gemm_desc = {};
  gemm_desc.ATensor = &a_tensor_desc;
  gemm_desc.BTensor = &b_tensor_desc;
  gemm_desc.CTensor = nullptr;
  gemm_desc.OutputTensor = &c_tensor_desc;
  gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
  gemm_desc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;  // Transpose B for A * B^T
  gemm_desc.Alpha = 1.0f;
  gemm_desc.Beta = 0.0f;
  gemm_desc.FusedActivation = nullptr;

  DML_OPERATOR_DESC op_desc = {DML_OPERATOR_GEMM, &gemm_desc};

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Create binding table
  DML_BINDING_PROPERTIES binding_props = compiled_op->GetBindingProperties();

  Microsoft::WRL::ComPtr<IDMLBindingTable> binding_table;
  DML_BINDING_TABLE_DESC binding_table_desc = {};
  binding_table_desc.Dispatchable = compiled_op.Get();

  dml_device->CreateBindingTable(&binding_table_desc,
                                 IID_PPV_ARGS(&binding_table));

  // Bind inputs and outputs
  DML_BUFFER_BINDING a_binding = {
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(reshaped_a.buffer())),
      0, reshaped_a.size() * sizeof(float16_t)};
  DML_BUFFER_BINDING b_binding = {
      reinterpret_cast<ID3D12Resource*>(
          const_cast<void*>(dequantized_b.buffer())),
      0, dequantized_b.size() * sizeof(float16_t)};
  DML_BUFFER_BINDING c_binding = {
      reinterpret_cast<ID3D12Resource*>(reshaped_c.buffer()), 0,
      reshaped_c.size() * sizeof(float16_t)};

  DML_BINDING_DESC input_bindings[] = {{DML_BINDING_TYPE_BUFFER, &a_binding},
                                       {DML_BINDING_TYPE_BUFFER, &b_binding}};
  DML_BINDING_DESC output_bindings[] = {{DML_BINDING_TYPE_BUFFER, &c_binding}};

  binding_table->BindInputs(2, input_bindings);
  binding_table->BindOutputs(1, output_bindings);

  // Record dispatch
  device->RecordDispatch(compiled_op.Get(), binding_table.Get());

  // Execute command list
  device->ExecuteCommandList();
}

template <>
void GemvAwq::compute_gemv2<Device::DirectML, float16_t, int>(
    const StorageView& a,
    const StorageView& b,
    const StorageView& scale,
    const StorageView& zero,
    StorageView& c) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Get dimensions
  dim_t split_k_iters = 8;
  dim_t num_in_channels = a.dim(-1);
  dim_t num_in_feats = a.size() / num_in_channels;
  dim_t num_out_channels = b.dim(0);
  dim_t group_size = num_in_channels / scale.dim(-1);

  // Calculate split dimensions
  dim_t k_per_split = (num_in_channels + split_k_iters - 1) / split_k_iters;

  // Resize output for split-k results
  if (a.rank() == 2)
    c.resize({split_k_iters, num_in_feats, num_out_channels});
  else if (a.rank() == 3)
    c.resize({split_k_iters, a.dim(0), a.dim(1), num_out_channels});

  // Dequantize the full weight matrix first
  StorageView dequantized_b({num_out_channels, num_in_channels},
                            DataType::FLOAT16, Device::DirectML);
  dequantize_weights_dml(b, scale, zero, dequantized_b, group_size);

  // Process each split-k iteration
  for (dim_t split_idx = 0; split_idx < split_k_iters; ++split_idx) {
    dim_t k_start = split_idx * k_per_split;
    dim_t k_end = std::min(k_start + k_per_split, num_in_channels);
    dim_t actual_k = k_end - k_start;

    if (actual_k == 0) {
      // Zero out this split's output using DirectML
      StorageView split_output;
      if (a.rank() == 2) {
        // Create view into the split_idx slice of c
        dim_t slice_size = num_in_feats * num_out_channels;
        split_output.view(static_cast<float16_t*>(c.data<float16_t>()) +
                              split_idx * slice_size,
                          {num_in_feats, num_out_channels});
      } else {
        // Create view into the split_idx slice of c for 3D case
        dim_t slice_size = a.dim(0) * a.dim(1) * num_out_channels;
        split_output.view(static_cast<float16_t*>(c.data<float16_t>()) +
                              split_idx * slice_size,
                          {a.dim(0), a.dim(1), num_out_channels});
      }

      // Zero out using DirectML fill operation
      zero_tensor_dml(split_output);
      continue;
    }

    // Create sliced input tensor A[:, k_start:k_end]
    StorageView a_slice;
    if (a.rank() == 2) {
      a_slice = StorageView({num_in_feats, actual_k}, DataType::FLOAT16,
                            Device::DirectML);
    } else {
      a_slice = StorageView({a.dim(0), a.dim(1), actual_k}, DataType::FLOAT16,
                            Device::DirectML);
    }

    // Create slice operation for input A
    slice_tensor_k_dimension(a, a_slice, k_start, k_end);

    // Create sliced weight tensor B[:, k_start:k_end]
    StorageView b_slice({num_out_channels, actual_k}, DataType::FLOAT16,
                        Device::DirectML);
    slice_tensor_k_dimension(dequantized_b, b_slice, k_start, k_end);

    // Get output slice for this split
    StorageView c_slice;
    if (a.rank() == 2) {
      c_slice = StorageView({num_in_feats, num_out_channels}, DataType::FLOAT16,
                            Device::DirectML);
      c_slice.view(static_cast<float16_t*>(c.data<float16_t>()) +
                       split_idx * num_in_feats * num_out_channels,
                   {num_in_feats, num_out_channels});
    } else {
      c_slice = StorageView({a.dim(0), a.dim(1), num_out_channels},
                            DataType::FLOAT16, Device::DirectML);
      c_slice.view(static_cast<float16_t*>(c.data<float16_t>()) +
                       split_idx * a.dim(0) * a.dim(1) * num_out_channels,
                   {a.dim(0), a.dim(1), num_out_channels});
    }

    // Perform partial GEMV: c_slice = a_slice * b_slice^T
    perform_partial_gemv(a_slice, b_slice, c_slice);
  }

  // Reduce split-k results to get final output
  reduce_split_k_results(c);
}

// Explicit template instantiations
#define DECLARE_IMPL(T)                                           \
  template void GemvAwq::compute_gemv2<Device::DirectML, T, int>( \
      const StorageView&, const StorageView&, const StorageView&, \
      const StorageView&, StorageView&) const;                    \
  template void GemvAwq::compute_gemv<Device::DirectML, T, int>(  \
      const StorageView&, const StorageView&, const StorageView&, \
      const StorageView&, StorageView&) const;

}  // namespace ops
}  // namespace ctranslate2

#endif
