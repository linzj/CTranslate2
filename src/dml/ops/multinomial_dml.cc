#include "ctranslate2/ops/multinomial.h"

#ifdef CT2_WITH_DIRECTML

#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

// Ensure dml_utils.h would have something like this:
namespace ctranslate2 {
namespace dml {

// Minimal DML_TENSOR_DATA_TYPE mapping
inline DML_TENSOR_DATA_TYPE get_dml_data_type(ctranslate2::DataType dtype) {
  switch (dtype) {
    case ctranslate2::DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case ctranslate2::DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case ctranslate2::DataType::INT32:
      return DML_TENSOR_DATA_TYPE_INT32;
    // Add BFLOAT16 if/when supported and needed
    default:
      THROW_INVALID_ARGUMENT("Unsupported data type for DML: " +
                             dtype_name(dtype));
  }
}

// DmlTensorDesc helper class (simplified version for this context)
class DmlTensorDesc {
 public:
  DML_BUFFER_TENSOR_DESC buffer_desc{};
  DML_TENSOR_DESC tensor_desc{};
  std::vector<UINT> sizes_vec;
  std::vector<UINT> strides_vec;

  DmlTensorDesc() = default;  // Default constructor

  DmlTensorDesc(const StorageView& storage,
                bool for_broadcasting_input = false) {
    Shape shape = storage.shape();
    if (storage.is_scalar()) {  // Represent scalar as 1D tensor of size 1
      shape.assign({1});
    } else if (shape.empty() &&
               storage.size() > 0) {  // Rank 0 but has elements (should not
                                      // happen for typical tensors)
      shape.assign({(dim_t)storage.size()});  // Treat as 1D
    }

    for (dim_t d : shape) {
      sizes_vec.push_back(static_cast<UINT>(d));
    }
    // Ensure there's at least one dimension for DML.
    if (sizes_vec.empty()) {
      sizes_vec.push_back(1);  // Treat as {1} if completely empty.
    }

    if (!sizes_vec.empty()) {
      strides_vec.resize(sizes_vec.size());
      UINT current_stride = 1;
      for (int i = static_cast<int>(sizes_vec.size()) - 1; i >= 0; --i) {
        if (for_broadcasting_input && sizes_vec[i] == 1 &&
            sizes_vec.size() > 1) {
          strides_vec[i] = 0;  // Broadcast this dimension
        } else {
          strides_vec[i] = current_stride;
        }
        // Only advance stride if not broadcasting this dimension or if it's not
        // size 0
        if (!(for_broadcasting_input && sizes_vec[i] == 1 &&
              sizes_vec.size() > 1)) {
          if (sizes_vec[i] > 0)
            current_stride *= sizes_vec[i];
          else
            current_stride = 0;
        }
      }
    }

    buffer_desc.DataType = get_dml_data_type(storage.dtype());
    buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
    buffer_desc.DimensionCount = static_cast<UINT>(sizes_vec.size());
    buffer_desc.Sizes = sizes_vec.data();
    buffer_desc.Strides = strides_vec.empty() ? nullptr : strides_vec.data();

    uint64_t logical_size_bytes = 1;
    if (buffer_desc.DimensionCount == 0) {
      // Truly scalar or empty
      logical_size_bytes = (storage.size() == 0) ? 0 : storage.item_size();
    } else {
      logical_size_bytes = storage.item_size();
      for (UINT s : sizes_vec) {
        // if any dimension is 0, logical size is 0
        if (s == 0) {
          logical_size_bytes = 0;
          break;
        }
        logical_size_bytes *= s;
      }
    }
    buffer_desc.TotalTensorSizeInBytes = logical_size_bytes;
    buffer_desc.GuaranteedBaseOffsetAlignment = 0;

    tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    tensor_desc.Desc = &buffer_desc;
  }

  // Constructor for known shape and type (intermediate tensors)
  DmlTensorDesc(DataType ctranslate2_dtype,
                const Shape& shape,
                bool for_broadcasting_input = false) {
    // This constructor is similar to the StorageView one, but directly from
    // CTranslate2 types.
    Shape current_shape = shape;
    if (shape.empty()) {
      current_shape.assign({1});
    } else if (shape.empty() && shape.size() > 0) {
      current_shape.assign({(dim_t)shape.size()});
    }

    for (dim_t d : current_shape) {
      sizes_vec.push_back(static_cast<UINT>(d));
    }
    if (sizes_vec.empty()) {
      sizes_vec.push_back(1);
    }

    if (!sizes_vec.empty()) {
      strides_vec.resize(sizes_vec.size());
      UINT current_stride = 1;
      for (int i = static_cast<int>(sizes_vec.size()) - 1; i >= 0; --i) {
        if (for_broadcasting_input && sizes_vec[i] == 1 &&
            sizes_vec.size() > 1) {
          strides_vec[i] = 0;
        } else {
          strides_vec[i] = current_stride;
        }
        if (!(for_broadcasting_input && sizes_vec[i] == 1 &&
              sizes_vec.size() > 1)) {
          if (sizes_vec[i] > 0)
            current_stride *= sizes_vec[i];
          else
            current_stride = 0;
        }
      }
    }

    buffer_desc.DataType = get_dml_data_type(ctranslate2_dtype);
    buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
    buffer_desc.DimensionCount = static_cast<UINT>(sizes_vec.size());
    buffer_desc.Sizes = sizes_vec.data();
    buffer_desc.Strides = strides_vec.empty() ? nullptr : strides_vec.data();

    uint64_t logical_size_bytes = 1;
    if (buffer_desc.DimensionCount == 0) {
      TYPE_DISPATCH(ctranslate2_dtype, logical_size_bytes = sizeof(T));
    } else {
      TYPE_DISPATCH(ctranslate2_dtype, logical_size_bytes = sizeof(T));
      for (UINT s : sizes_vec) {
        if (s == 0) {
          logical_size_bytes = 0;
          break;
        }
        logical_size_bytes *= s;
      }
    }
    buffer_desc.TotalTensorSizeInBytes = logical_size_bytes;
    buffer_desc.GuaranteedBaseOffsetAlignment = 0;

    tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    tensor_desc.Desc = &buffer_desc;
  }
};

}  // namespace dml
}  // namespace ctranslate2
// End of dml_utils.h content section

namespace ctranslate2 {
namespace ops {
namespace dml_internal {

// Helper function containing the core DML logic for multinomial.
void multinomial_impl(
    dml::Device* device,  // ctranslate2::dml::Device
    // IDMLDevice1* dml_device, // This is accessible via device->DML()
    const StorageView& probs_input,
    StorageView& output_indices,
    const Multinomial& op_params) {
  const dim_t depth = probs_input.dim(-1);  // class_size
  const dim_t batch_size = probs_input.size() / depth;

  output_indices.resize({batch_size});

  // --- Handle input cast to FLOAT32 if necessary ---
  StorageView probs_f32_sv(
      DataType::FLOAT32,
      device->GetCommandListType() == D3D12_COMMAND_LIST_TYPE_COPY
          ? Device::CPU
          : Device::DirectML);  // Temporary StorageView descriptor
  Microsoft::WRL::ComPtr<ID3D12Resource>
      probs_f32_intermediate_res;  // D3D resource for casted float32 probs
  bool cast_to_f32_performed = false;
  ID3D12Resource* current_probs_resource_ptr;

  if (probs_input.dtype() != DataType::FLOAT32) {
    cast_to_f32_performed = true;
    probs_f32_sv.resize(probs_input.shape());  // Set shape for DmlTensorDesc
    probs_f32_intermediate_res = device->CreatePreferredDeviceMemoryBuffer(
        probs_f32_sv.reserved_memory());
    device->KeepAliveUntilNextCommandListDispatch(probs_f32_intermediate_res);

    dml::DmlTensorDesc input_desc_orig(probs_input);
    dml::DmlTensorDesc output_desc_f32(probs_f32_sv);

    DML_CAST_OPERATOR_DESC cast_desc = {};
    cast_desc.InputTensor = &input_desc_orig.tensor_desc;
    cast_desc.OutputTensor = &output_desc_f32.tensor_desc;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_CAST, &cast_desc};

    dml::Operator* cast_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    cast_op->Execute({dml::create_binding_desc(dml::create_buffer_binding(
                         static_cast<ID3D12Resource*>(
                             const_cast<void*>(probs_input.buffer()))))},
                     {dml::create_binding_desc(dml::create_buffer_binding(
                         probs_f32_intermediate_res.Get()))});
    current_probs_resource_ptr = probs_f32_intermediate_res.Get();
  } else {
    probs_f32_sv = probs_input;  // Share metadata if no cast
    current_probs_resource_ptr =
        static_cast<ID3D12Resource*>(const_cast<void*>(probs_input.buffer()));
  }

  // --- Prepare DML Tensor Descriptors for FLOAT32 tensors ---
  Shape probs_f32_shape = {batch_size, depth};
  dml::DmlTensorDesc probs_desc_f32(DataType::FLOAT32, probs_f32_shape);

  // Intermediate tensor resources (all based on FLOAT32 processing path)
  // 1. Random Numbers (batch_size, 1), FLOAT32
  Shape random_shape = {batch_size, 1};
  StorageView random_numbers_sv(DataType::FLOAT32,
                                probs_f32_sv.device());  // Use same device
  random_numbers_sv.resize(random_shape);
  dml::DmlTensorDesc random_desc(random_numbers_sv,
                                 /*for_broadcasting_input=*/true);
  Microsoft::WRL::ComPtr<ID3D12Resource> random_numbers_res =
      device->CreatePreferredDeviceMemoryBuffer(
          random_numbers_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(random_numbers_res);

  // Philox state for random generator
  Shape philox_state_dims = {4};  // 4 UINT32s for state
  StorageView philox_state_sv(DataType::INT32, probs_f32_sv.device());
  philox_state_sv.resize(philox_state_dims);
  dml::DmlTensorDesc philox_state_tensor_desc(philox_state_sv);

  Microsoft::WRL::ComPtr<ID3D12Resource> state_in_res =
      device->CreatePreferredDeviceMemoryBuffer(
          philox_state_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(state_in_res);
  // Zero-initialize state_in_res for DML_RANDOM_GENERATOR first use
  {
    DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_zero_desc = {};
    fill_zero_desc.OutputTensor = &philox_state_tensor_desc.tensor_desc;
    fill_zero_desc.ValueDataType = DML_TENSOR_DATA_TYPE_UINT32;
    fill_zero_desc.Value.UInt32 = 0;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                         &fill_zero_desc};
    dml::Operator* fill_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    fill_op->Execute({}, {dml::create_binding_desc(
                             dml::create_buffer_binding(state_in_res.Get()))});
  }
  Microsoft::WRL::ComPtr<ID3D12Resource> state_out_res =
      device->CreatePreferredDeviceMemoryBuffer(
          philox_state_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(state_out_res);

  // 2. Cumulative Probabilities (batch_size, class_size), FLOAT32
  Shape cumsum_shape = {batch_size, depth};
  StorageView cumsum_sv(DataType::FLOAT32, probs_f32_sv.device());
  cumsum_sv.resize(cumsum_shape);
  dml::DmlTensorDesc cumsum_desc(cumsum_sv);
  Microsoft::WRL::ComPtr<ID3D12Resource> cumsum_res =
      device->CreatePreferredDeviceMemoryBuffer(cumsum_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(cumsum_res);

  // 3. Comparison Result (batch_size, class_size), UINT8
  Shape compare_shape = {batch_size, depth};
  StorageView compare_sv(DataType::INT8, probs_f32_sv.device());
  compare_sv.resize(compare_shape);
  dml::DmlTensorDesc compare_tensor_desc(compare_sv);
  Microsoft::WRL::ComPtr<ID3D12Resource> compare_res =
      device->CreatePreferredDeviceMemoryBuffer(compare_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(compare_res);

  // 4. Iota Tensor (1, class_size), FLOAT32
  Shape iota_shape = {1, depth};
  StorageView iota_sv(DataType::FLOAT32, probs_f32_sv.device());
  iota_sv.resize(iota_shape);
  dml::DmlTensorDesc iota_desc(iota_sv, /*for_broadcasting_input=*/true);
  Microsoft::WRL::ComPtr<ID3D12Resource> iota_res =
      device->CreatePreferredDeviceMemoryBuffer(iota_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(iota_res);

  // 4b. Scalar Max Value Tensor (1), FLOAT32
  Shape scalar_shape = {1};
  StorageView max_val_sv(DataType::FLOAT32, probs_f32_sv.device());
  max_val_sv.resize(scalar_shape);
  dml::DmlTensorDesc max_val_desc(max_val_sv, /*for_broadcasting_input=*/true);
  Microsoft::WRL::ComPtr<ID3D12Resource> max_val_res =
      device->CreatePreferredDeviceMemoryBuffer(max_val_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(max_val_res);

  // 5. ArgminInput Tensor (batch_size, class_size), FLOAT32
  Shape argmin_input_shape = {batch_size, depth};
  StorageView argmin_input_sv(DataType::FLOAT32, probs_f32_sv.device());
  argmin_input_sv.resize(argmin_input_shape);
  dml::DmlTensorDesc argmin_input_desc(argmin_input_sv);
  Microsoft::WRL::ComPtr<ID3D12Resource> argmin_input_res =
      device->CreatePreferredDeviceMemoryBuffer(
          argmin_input_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(argmin_input_res);

  // --- Define and Execute DML Operators ---
  // Op 1: Random Generator
  {
    DML_RANDOM_GENERATOR_OPERATOR_DESC desc = {};
    desc.InputStateTensor = &philox_state_tensor_desc.tensor_desc;
    desc.OutputTensor = &random_desc.tensor_desc;
    desc.OutputStateTensor = &philox_state_tensor_desc.tensor_desc;
    desc.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_RANDOM_GENERATOR, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute({dml::create_binding_desc(
                    dml::create_buffer_binding(state_in_res.Get()))},
                {dml::create_binding_desc(
                     dml::create_buffer_binding(random_numbers_res.Get())),
                 dml::create_binding_desc(
                     dml::create_buffer_binding(state_out_res.Get()))});
  }

  // Op 2: Cumulative Sum
  {
    DML_CUMULATIVE_SUMMATION_OPERATOR_DESC desc = {};
    desc.InputTensor = &probs_desc_f32.tensor_desc;
    desc.OutputTensor = &cumsum_desc.tensor_desc;
    desc.Axis = probs_desc_f32.buffer_desc.DimensionCount - 1;
    desc.AxisDirection = DML_AXIS_DIRECTION_INCREASING;
    desc.HasExclusiveSum = FALSE;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_CUMULATIVE_SUMMATION,
                                         &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute({dml::create_binding_desc(
                    dml::create_buffer_binding(current_probs_resource_ptr))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(cumsum_res.Get()))});
  }

  // Op 3: Compare (CumulativeProbs >= RandomSample)
  {
    DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC desc = {};
    desc.ATensor = &cumsum_desc.tensor_desc;
    desc.BTensor = &random_desc.tensor_desc;
    desc.OutputTensor = &compare_tensor_desc.tensor_desc;
    DML_OPERATOR_DESC op_desc_wrapper = {
        DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute(
        {dml::create_binding_desc(dml::create_buffer_binding(cumsum_res.Get())),
         dml::create_binding_desc(
             dml::create_buffer_binding(random_numbers_res.Get()))},
        {dml::create_binding_desc(
            dml::create_buffer_binding(compare_res.Get()))});
  }

  // Op 4a: Fill Iota Tensor
  {
    DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC desc = {};
    desc.OutputTensor = &iota_desc.tensor_desc;
    desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc.ValueStart.Float32 = 0.0f;
    desc.ValueDelta.Float32 = 1.0f;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_FILL_VALUE_SEQUENCE,
                                         &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute(
        {},
        {dml::create_binding_desc(dml::create_buffer_binding(iota_res.Get()))});
  }

  // Op 4b: Fill Max Value Scalar Tensor
  {
    DML_FILL_VALUE_CONSTANT_OPERATOR_DESC desc = {};
    desc.OutputTensor = &max_val_desc.tensor_desc;
    desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    desc.Value.Float32 = std::numeric_limits<float>::max();
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_FILL_VALUE_CONSTANT,
                                         &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute({}, {dml::create_binding_desc(
                        dml::create_buffer_binding(max_val_res.Get()))});
  }

  // Op 5: Conditional Select
  {
    DML_ELEMENT_WISE_IF_OPERATOR_DESC desc = {};
    desc.ConditionTensor = &compare_tensor_desc.tensor_desc;
    desc.ATensor = &iota_desc.tensor_desc;
    desc.BTensor = &max_val_desc.tensor_desc;
    desc.OutputTensor = &argmin_input_desc.tensor_desc;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_ELEMENT_WISE_IF, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute(
        {dml::create_binding_desc(
             dml::create_buffer_binding(compare_res.Get())),
         dml::create_binding_desc(dml::create_buffer_binding(iota_res.Get())),
         dml::create_binding_desc(
             dml::create_buffer_binding(max_val_res.Get()))},
        {dml::create_binding_desc(
            dml::create_buffer_binding(argmin_input_res.Get()))});
  }

  // Op 6: ArgMin
  Shape dml_argmin_out_shape = {batch_size,
                                1};  // DML Reduce keeps reduced dims as 1.
  StorageView dml_argmin_out_sv(DataType::INT32, probs_f32_sv.device());
  dml_argmin_out_sv.resize(dml_argmin_out_shape);
  dml::DmlTensorDesc dml_argmin_out_tensor_desc(dml_argmin_out_sv);
  Microsoft::WRL::ComPtr<ID3D12Resource> dml_argmin_out_res =
      device->CreatePreferredDeviceMemoryBuffer(
          dml_argmin_out_sv.reserved_memory());
  device->KeepAliveUntilNextCommandListDispatch(dml_argmin_out_res);
  {
    DML_REDUCE_OPERATOR_DESC desc = {};
    desc.InputTensor = &argmin_input_desc.tensor_desc;
    desc.OutputTensor = &dml_argmin_out_tensor_desc.tensor_desc;
    desc.Function = DML_REDUCE_FUNCTION_ARGMIN;
    UINT axis_to_reduce = argmin_input_desc.buffer_desc.DimensionCount - 1;
    desc.Axes = &axis_to_reduce;
    desc.AxisCount = 1;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_REDUCE, &desc};
    dml::Operator* op = dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    op->Execute({dml::create_binding_desc(
                    dml::create_buffer_binding(argmin_input_res.Get()))},
                {dml::create_binding_desc(
                    dml::create_buffer_binding(dml_argmin_out_res.Get()))});
  }

  // Op 7: Cast/Copy to final output_indices buffer
  Shape original_output_shape =
      output_indices.shape();  // Should be {batch_size}
  output_indices.reshape(
      dml_argmin_out_shape);  // Reshape to {batch_size, 1} for DML op
  dml::DmlTensorDesc final_output_desc(output_indices);

  if (output_indices.dtype() != DataType::INT32) {  // e.g. output is INT32
    DML_CAST_OPERATOR_DESC cast_desc = {};
    cast_desc.InputTensor =
        &dml_argmin_out_tensor_desc.tensor_desc;  // UINT32 input
    cast_desc.OutputTensor =
        &final_output_desc.tensor_desc;  // INT32 (or other) output
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_CAST, &cast_desc};
    dml::Operator* cast_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    cast_op->Execute({dml::create_binding_desc(
                         dml::create_buffer_binding(dml_argmin_out_res.Get()))},
                     {dml::create_binding_desc(dml::create_buffer_binding(
                         static_cast<ID3D12Resource*>(
                             const_cast<void*>(output_indices.buffer()))))});
  } else {  // Output is already UINT32
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
    identity_desc.InputTensor = &dml_argmin_out_tensor_desc.tensor_desc;
    identity_desc.OutputTensor = &final_output_desc.tensor_desc;
    DML_OPERATOR_DESC op_desc_wrapper = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                         &identity_desc};
    dml::Operator* identity_op =
        dml::GetOrCreateCompiledOperatorApi(&op_desc_wrapper);
    identity_op->Execute(
        {dml::create_binding_desc(
            dml::create_buffer_binding(dml_argmin_out_res.Get()))},
        {dml::create_binding_desc(
            dml::create_buffer_binding(static_cast<ID3D12Resource*>(
                const_cast<void*>(output_indices.buffer()))))});
  }

  output_indices.reshape(original_output_shape);  // Restore to {batch_size}
}

}  // namespace dml_internal

template <Device D, typename T>
void Multinomial::compute(const StorageView& probs,
                          StorageView& output_indices) const {
  static_assert(D == Device::DirectML, "This specialization is for DirectML.");
  if (_sample_size != 1) {
    THROW_INVALID_ARGUMENT(
        "DirectML Multinomial currently only supports sample_size = 1");
  }

  PROFILE("MultinomialDML");

  dml::Device* device = dml::get_device();

  dml_internal::multinomial_impl(device, probs, output_indices, *this);
}

// Explicit instantiations
#define DECLARE_MULTINOMIAL_DML_IMPL(T)                    \
  template void Multinomial::compute<Device::DirectML, T>( \
      const StorageView& probs, StorageView& output_indices) const;

DECLARE_MULTINOMIAL_DML_IMPL(float)
DECLARE_MULTINOMIAL_DML_IMPL(float16_t)
// DECLARE_MULTINOMIAL_DML_IMPL(bfloat16_t) // If/when bfloat16_t is a distinct
// type and supported.

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
