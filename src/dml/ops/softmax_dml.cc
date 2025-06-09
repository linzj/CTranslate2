#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/softmax.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void SoftMax::compute(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output) const {
  // Get DML device and context
  auto* device_context =
      dml::get_device();  // Renamed for clarity vs dml_device
  auto* dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  std::vector<UINT> dml_dims_vec = {static_cast<UINT>(batch_size),
                                    static_cast<UINT>(depth)};

  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), dml_dims_vec, nullptr, input.size() * sizeof(T));
  const DML_TENSOR_DESC& dml_input_desc_ref =
      input_desc_bundle.get_tensor_desc();

  StorageView input_storage;
  ID3D12Resource* input_resource = dml::utils::ResourceFromStorageView(input);
  device_context->KeepAliveUntilNextCommandListDispatch(input_resource);
  if (input.buffer() == output.buffer()) {
    input_storage = std::move(output);
    StorageView new_output(input_storage.shape(), input_storage.dtype(),
                           input_storage.device());
    output = std::move(new_output);
  }

  // dml_output_desc_ref will refer to the final output tensor.
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), dml_dims_vec, nullptr, output.size() * sizeof(T));
  const DML_TENSOR_DESC& dml_output_desc_ref =
      output_desc_bundle.get_tensor_desc();

  UINT axis = 1;
  DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC log_softmax_op_def_storage;
  DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax_op_def_storage;
  DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC log_softmax_old_op_def_storage;
  DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax_old_op_def_storage;

  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          input_resource, 0,
          input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);
  std::vector<DML_BINDING_DESC> softmax_input_bindings = {input_binding_desc};

  if (lengths) {
    // --- Path with masking: MASK FIRST, THEN SOFTMAX ---

    // 1. Prepare Condition Tensor entirely on GPU (iota < lengths)
    // This section creates dml_condition_desc_ref and
    // condition_binding_cmp_output. 1.a Create Iota tensor: [0, 1, ...,
    // depth-1] with shape [1, depth], type INT32
    StorageView sequence_indices_gpu(DataType::INT32, output.device());
    sequence_indices_gpu.resize(Shape{1, depth});
    std::vector<UINT> seq_idx_tensor_dims = {1, static_cast<UINT>(depth)};
    dml::utils::DmlTensorDescBundle sequence_indices_desc_bundle(
        DML_TENSOR_DATA_TYPE_INT32, seq_idx_tensor_dims, nullptr,
        sequence_indices_gpu.size() * sizeof(int32_t));
    const DML_TENSOR_DESC& dml_sequence_indices_desc_ref =
        sequence_indices_desc_bundle.get_tensor_desc();

    DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC fill_seq_desc = {};
    fill_seq_desc.OutputTensor = &dml_sequence_indices_desc_ref;
    fill_seq_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
    fill_seq_desc.ValueStart.Int32 = 0;
    fill_seq_desc.ValueDelta.Int32 = 1;
    DML_OPERATOR_DESC fill_seq_op_meta_desc = {DML_OPERATOR_FILL_VALUE_SEQUENCE,
                                               &fill_seq_desc};
    auto compiled_fill_seq_op =
        dml::GetOrCreateCompiledOperatorApi(&fill_seq_op_meta_desc);

    DML_BUFFER_BINDING seq_indices_buffer_binding =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(sequence_indices_gpu), 0,
            sequence_indices_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes);
    DML_BINDING_DESC seq_indices_binding_fill_output =
        dml::utils::create_binding_desc(&seq_indices_buffer_binding);
    compiled_fill_seq_op->Execute({}, {seq_indices_binding_fill_output});

    // batch_size of lengths for LESS_THAN.
    StorageView tiled_sequence_indices_gpu(DataType::INT32, output.device());
    tiled_sequence_indices_gpu.resize(
        Shape{batch_size, depth});  // Target shape [batch_size, depth]

    // dml_dims_vec is {static_cast<UINT>(batch_size), static_cast<UINT>(depth)}
    dml::utils::DmlTensorDescBundle tiled_sequence_indices_desc_bundle(
        DML_TENSOR_DATA_TYPE_INT32,  // Data type of the tiled tensor
        dml_dims_vec,                // Target dimensions [batch_size, depth]
        nullptr,                     // Strides (can be null for contiguous)
        tiled_sequence_indices_gpu.size() * sizeof(int32_t));  // Total size
    const DML_TENSOR_DESC& dml_tiled_sequence_indices_desc_ref =
        tiled_sequence_indices_desc_bundle.get_tensor_desc();

    DML_TILE_OPERATOR_DESC tile_op_desc_iota = {};  // Renamed
    tile_op_desc_iota.InputTensor =
        &dml_sequence_indices_desc_ref;  // Input is [1, depth] (output of
                                         // FILL_SEQUENCE)
    tile_op_desc_iota.OutputTensor =
        &dml_tiled_sequence_indices_desc_ref;  // Output is [batch_size, depth]
    UINT repeats_array_iota[] = {
        // Renamed
        static_cast<UINT>(batch_size),
        1};  // Repeat batch_size times along dim 0, 1 time along dim 1
    tile_op_desc_iota.RepeatsCount = ARRAYSIZE(repeats_array_iota);
    tile_op_desc_iota.Repeats = repeats_array_iota;
    DML_OPERATOR_DESC tile_op_meta_desc_iota = {DML_OPERATOR_TILE,
                                                &tile_op_desc_iota};  // Renamed
    auto compiled_tile_op_iota =                                      // Renamed
        dml::GetOrCreateCompiledOperatorApi(&tile_op_meta_desc_iota);

    DML_BUFFER_BINDING tiled_seq_indices_buffer_binding_for_tile_output =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(tiled_sequence_indices_gpu), 0,
            tiled_sequence_indices_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes);
    DML_BINDING_DESC tiled_seq_indices_binding_tile_output =
        dml::utils::create_binding_desc(
            &tiled_seq_indices_buffer_binding_for_tile_output);

    compiled_tile_op_iota->Execute(
        {seq_indices_binding_fill_output},  // Renamed
        {tiled_seq_indices_binding_tile_output});

    // 1.c Prepare the original `lengths` tensor for tiling by describing it as
    // 2D [batch_size, 1] The physical buffer is still 1D, but DML needs a 2D
    // descriptor for the TILE operator's input.
    std::vector<UINT> lengths_input_tile_dims = {static_cast<UINT>(batch_size),
                                                 1};
    std::vector<UINT> lengths_input_tile_strides = {
        1, 0};  // Stride 1 for batch, 0 for the new dimension of size 1

    // Manually create the DML_BUFFER_TENSOR_DESC for the lengths tensor to make
    // it appear as [batch_size, 1]
    DML_BUFFER_TENSOR_DESC lengths_buffer_tensor_desc_for_tile_input = {};
    lengths_buffer_tensor_desc_for_tile_input.DataType =
        DML_TENSOR_DATA_TYPE_INT32;  // lengths are int32_t
    lengths_buffer_tensor_desc_for_tile_input.Flags = DML_TENSOR_FLAG_NONE;
    lengths_buffer_tensor_desc_for_tile_input.DimensionCount =
        static_cast<UINT>(lengths_input_tile_dims.size());
    lengths_buffer_tensor_desc_for_tile_input.Sizes =
        lengths_input_tile_dims.data();
    lengths_buffer_tensor_desc_for_tile_input.Strides =
        lengths_input_tile_strides.data();
    // TotalTensorSizeInBytes must refer to the actual physical size of the
    // `lengths` StorageView buffer.
    lengths_buffer_tensor_desc_for_tile_input.TotalTensorSizeInBytes =
        lengths->size() * sizeof(int32_t);
    lengths_buffer_tensor_desc_for_tile_input.GuaranteedBaseOffsetAlignment = 0;

    DML_TENSOR_DESC dml_lengths_input_for_tile_desc =
        {};  // This will be InputTensor for TILE
    dml_lengths_input_for_tile_desc.Type = DML_TENSOR_TYPE_BUFFER;
    dml_lengths_input_for_tile_desc.Desc =
        &lengths_buffer_tensor_desc_for_tile_input;

    // Binding for the original lengths buffer, using its actual physical size.
    DML_BUFFER_BINDING original_lengths_buffer_binding =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(*lengths), 0,
            lengths_buffer_tensor_desc_for_tile_input
                .TotalTensorSizeInBytes);  // Size of the physical buffer
    DML_BINDING_DESC dml_binding_for_lengths_tile_input =
        dml::utils::create_binding_desc(&original_lengths_buffer_binding);

    // 1.d Tile the "lengths-as-2D" tensor ([batch_size, 1]) to [batch_size,
    // depth]
    StorageView tiled_lengths_gpu(DataType::INT32, output.device());
    tiled_lengths_gpu.resize(Shape{batch_size, depth});  // Target shape

    // Output tensor descriptor for tiled_lengths_gpu (shape [batch_size,
    // depth]) dml_dims_vec is {batch_size, depth}
    dml::utils::DmlTensorDescBundle tiled_lengths_output_desc_bundle(
        DML_TENSOR_DATA_TYPE_INT32,
        dml_dims_vec,  // shape [batch_size, depth]
        nullptr,       // Contiguous strides for the output
        tiled_lengths_gpu.size() * sizeof(int32_t));
    const DML_TENSOR_DESC& dml_tiled_lengths_desc_actual_output_ref =
        tiled_lengths_output_desc_bundle.get_tensor_desc();

    DML_TILE_OPERATOR_DESC tile_op_desc_lengths = {};
    tile_op_desc_lengths.InputTensor =
        &dml_lengths_input_for_tile_desc;  // Input is 2D: [batch_size, 1]
    tile_op_desc_lengths.OutputTensor =
        &dml_tiled_lengths_desc_actual_output_ref;  // Output is 2D:
                                                    // [batch_size, depth]
    UINT repeats_array_lengths[] = {
        1, static_cast<UINT>(depth)};  // Repeat existing batch dim once, repeat
                                       // new depth dim `depth` times
    tile_op_desc_lengths.RepeatsCount = ARRAYSIZE(repeats_array_lengths);
    tile_op_desc_lengths.Repeats = repeats_array_lengths;
    DML_OPERATOR_DESC tile_op_meta_desc_lengths = {DML_OPERATOR_TILE,
                                                   &tile_op_desc_lengths};
    auto compiled_tile_op_lengths =
        dml::GetOrCreateCompiledOperatorApi(&tile_op_meta_desc_lengths);

    DML_BUFFER_BINDING tiled_lengths_buffer_binding_for_tile_output =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(tiled_lengths_gpu), 0,
            tiled_lengths_output_desc_bundle.get_buffer_desc()
                .TotalTensorSizeInBytes);
    DML_BINDING_DESC tiled_lengths_binding_tile_output =
        dml::utils::create_binding_desc(
            &tiled_lengths_buffer_binding_for_tile_output);

    // Input to this TILE op uses the original_lengths_buffer_binding (referring
    // to the actual 1D lengths StorageView buffer)
    compiled_tile_op_lengths->Execute({dml_binding_for_lengths_tile_input},
                                      {tiled_lengths_binding_tile_output});

    // 1.e Create Condition Tensor using
    // DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN
    StorageView condition_tensor_gpu(DataType::INT8, output.device());
    condition_tensor_gpu.resize(Shape{batch_size, depth});
    dml::utils::DmlTensorDescBundle condition_desc_bundle(
        DML_TENSOR_DATA_TYPE_UINT8, dml_dims_vec, nullptr,
        condition_tensor_gpu.size());
    const DML_TENSOR_DESC& dml_condition_desc_ref =
        condition_desc_bundle.get_tensor_desc();

    DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC less_than_op_desc = {};
    less_than_op_desc.ATensor =
        &dml_tiled_sequence_indices_desc_ref;  // Tiled iota [batch_size, depth]
    less_than_op_desc.BTensor =
        &dml_tiled_lengths_desc_actual_output_ref;  // Tiled lengths
                                                    // [batch_size, depth]
    less_than_op_desc.OutputTensor =
        &dml_condition_desc_ref;  // Output [batch_size, depth]
    DML_OPERATOR_DESC less_than_op_meta_desc = {
        DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN, &less_than_op_desc};
    auto compiled_less_than_op =
        dml::GetOrCreateCompiledOperatorApi(&less_than_op_meta_desc);

    DML_BUFFER_BINDING condition_buffer_binding_for_cmp_output =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(condition_tensor_gpu), 0,
            condition_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC condition_binding_cmp_output =
        dml::utils::create_binding_desc(
            &condition_buffer_binding_for_cmp_output);

    // Inputs to LESS_THAN are tiled_sequence_indices and tiled_lengths
    std::vector<DML_BINDING_DESC> less_than_inputs = {
        tiled_seq_indices_binding_tile_output,
        tiled_lengths_binding_tile_output};
    compiled_less_than_op->Execute(less_than_inputs,
                                   {condition_binding_cmp_output});

    // 2. Prepare Padding Value Tensor (on GPU using DML_FILL_VALUE_CONSTANT)
    // For pre-softmax masking, padding with -infinity correctly excludes
    // elements.
    T padding_scalar_value = -std::numeric_limits<T>::infinity();

    StorageView padding_value_gpu_storage(
        output.dtype(), output.device());  // Renamed for clarity
    padding_value_gpu_storage.resize(
        output.shape());  // Same shape as input/output
    dml::utils::DmlTensorDescBundle padding_value_desc_bundle(
        padding_value_gpu_storage.dtype(), dml_dims_vec, nullptr,
        padding_value_gpu_storage.size() * sizeof(T));
    const DML_TENSOR_DESC& dml_padding_value_desc_ref =
        padding_value_desc_bundle.get_tensor_desc();

    DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_op_padding_desc = {};
    fill_op_padding_desc.OutputTensor = &dml_padding_value_desc_ref;
    // Get DML data type from the DML tensor descriptor itself
    const auto* buffer_desc = reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(
        dml_padding_value_desc_ref.Desc);
    fill_op_padding_desc.ValueDataType = buffer_desc->DataType;
    static_assert(sizeof(T) <= sizeof(fill_op_padding_desc.Value.Bytes),
                  "Size of T exceeds DML_SCALAR_UNION's byte array");
    std::memcpy(fill_op_padding_desc.Value.Bytes, &padding_scalar_value,
                sizeof(T));
    if (sizeof(T) < sizeof(fill_op_padding_desc.Value.Bytes)) {
      std::memset(fill_op_padding_desc.Value.Bytes + sizeof(T), 0,
                  sizeof(fill_op_padding_desc.Value.Bytes) - sizeof(T));
    }

    DML_OPERATOR_DESC fill_padding_op_meta_desc = {};
    fill_padding_op_meta_desc.Type = DML_OPERATOR_FILL_VALUE_CONSTANT;
    fill_padding_op_meta_desc.Desc = &fill_op_padding_desc;
    auto compiled_fill_padding_op =
        dml::GetOrCreateCompiledOperatorApi(&fill_padding_op_meta_desc);

    DML_BUFFER_BINDING padding_value_buffer_binding =  // Renamed
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(padding_value_gpu_storage), 0,
            padding_value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC padding_value_binding_desc =  // Renamed
        dml::utils::create_binding_desc(&padding_value_buffer_binding);
    std::vector<DML_BINDING_DESC> fill_padding_output_bindings = {
        // Renamed
        padding_value_binding_desc};
    compiled_fill_padding_op->Execute({}, fill_padding_output_bindings);

    // 3. ELEMENT_WISE_IF to create `masked_logits`
    StorageView masked_logits_storage(output.dtype(), output.device());
    masked_logits_storage.resize(output.shape());
    dml::utils::DmlTensorDescBundle masked_logits_desc_bundle(
        masked_logits_storage.dtype(), dml_dims_vec, nullptr,
        masked_logits_storage.size() * sizeof(T));
    const DML_TENSOR_DESC& dml_masked_logits_desc_ref =
        masked_logits_desc_bundle.get_tensor_desc();

    DML_ELEMENT_WISE_IF_OPERATOR_DESC if_logits_desc = {};
    if_logits_desc.ConditionTensor = &dml_condition_desc_ref;  // (from step 1)
    if_logits_desc.ATensor = &dml_input_desc_ref;  // Original logits
    if_logits_desc.BTensor =
        &dml_padding_value_desc_ref;  // (from step 2) -inf padding
    if_logits_desc.OutputTensor =
        &dml_masked_logits_desc_ref;  // Output to intermediate

    DML_OPERATOR_DESC if_logits_op_meta_desc = {DML_OPERATOR_ELEMENT_WISE_IF,
                                                &if_logits_desc};
    auto compiled_if_logits_op =
        dml::GetOrCreateCompiledOperatorApi(&if_logits_op_meta_desc);

    DML_BUFFER_BINDING masked_logits_buffer_binding =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(masked_logits_storage), 0,
            masked_logits_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC masked_logits_binding_if_output =
        dml::utils::create_binding_desc(&masked_logits_buffer_binding);

    std::vector<DML_BINDING_DESC> if_logits_op_input_bindings = {
        condition_binding_cmp_output,  // Condition (output of LESS_THAN)
        input_binding_desc,            // ATensor (Original input logits)
        padding_value_binding_desc     // BTensor (Padding values tensor)
    };
    std::vector<DML_BINDING_DESC> if_logits_op_output_bindings = {
        masked_logits_binding_if_output  // Output (masked_logits)
    };
    compiled_if_logits_op->Execute(if_logits_op_input_bindings,
                                   if_logits_op_output_bindings);

    // 4. Softmax or LogSoftmax on `masked_logits` into the final output
    DML_OPERATOR_DESC softmax_on_masked_desc = {};
#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      log_softmax_op_def_storage.InputTensor =
          &dml_masked_logits_desc_ref;  // Input is masked_logits
      log_softmax_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      log_softmax_op_def_storage.AxisCount = 1;
      log_softmax_op_def_storage.Axes = &axis;
      softmax_on_masked_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
      softmax_on_masked_desc.Desc = &log_softmax_op_def_storage;
    } else {
      softmax_op_def_storage.InputTensor =
          &dml_masked_logits_desc_ref;  // Input is masked_logits
      softmax_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      softmax_op_def_storage.AxisCount = 1;
      softmax_op_def_storage.Axes = &axis;
      softmax_on_masked_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
      softmax_on_masked_desc.Desc = &softmax_op_def_storage;
    }
#else
    if (_log) {
      log_softmax_old_op_def_storage.InputTensor = &dml_masked_logits_desc_ref;
      log_softmax_old_op_def_storage.OutputTensor = &dml_output_desc_ref;
      softmax_on_masked_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
      softmax_on_masked_desc.Desc = &log_softmax_old_op_def_storage;
    } else {
      softmax_old_op_def_storage.InputTensor = &dml_masked_logits_desc_ref;
      softmax_old_op_def_storage.OutputTensor = &dml_output_desc_ref;
      softmax_on_masked_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
      softmax_on_masked_desc.Desc = &softmax_old_op_def_storage;
    }
#endif
    auto compiled_softmax_on_masked_op =
        dml::GetOrCreateCompiledOperatorApi(&softmax_on_masked_desc);

    DML_BUFFER_BINDING final_output_buffer_binding_softmax =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(output), 0,
            output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC final_output_binding_softmax_output =
        dml::utils::create_binding_desc(&final_output_buffer_binding_softmax);

    std::vector<DML_BINDING_DESC> softmax_op_input_bindings = {
        masked_logits_binding_if_output};
    std::vector<DML_BINDING_DESC> softmax_op_output_bindings = {
        final_output_binding_softmax_output};

    compiled_softmax_on_masked_op->Execute(softmax_op_input_bindings,
                                           softmax_op_output_bindings);
  } else {
    // --- Original path: Softmax directly to final 'output' ---
    DML_OPERATOR_DESC softmax_op_desc_to_final = {};
#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      log_softmax_op_def_storage.InputTensor = &dml_input_desc_ref;
      log_softmax_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      log_softmax_op_def_storage.AxisCount = 1;
      log_softmax_op_def_storage.Axes = &axis;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
      softmax_op_desc_to_final.Desc = &log_softmax_op_def_storage;
    } else {
      softmax_op_def_storage.InputTensor = &dml_input_desc_ref;
      softmax_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      softmax_op_def_storage.AxisCount = 1;
      softmax_op_def_storage.Axes = &axis;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
      softmax_op_desc_to_final.Desc = &softmax_op_def_storage;
    }
#else
    if (_log) {
      log_softmax_old_op_def_storage.InputTensor = &dml_input_desc_ref;
      log_softmax_old_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
      softmax_op_desc_to_final.Desc = &log_softmax_old_op_def_storage;
    } else {
      softmax_old_op_def_storage.InputTensor = &dml_input_desc_ref;
      softmax_old_op_def_storage.OutputTensor =
          &dml_output_desc_ref;  // Final output
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
      softmax_op_desc_to_final.Desc = &softmax_old_op_def_storage;
    }
#endif
    auto compiled_softmax_to_final_op =
        dml::GetOrCreateCompiledOperatorApi(&softmax_op_desc_to_final);

    DML_BUFFER_BINDING output_buffer_binding_storage =
        dml::utils::create_buffer_binding(
            dml::utils::ResourceFromStorageView(output), 0,
            output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC output_binding_desc =
        dml::utils::create_binding_desc(&output_buffer_binding_storage);
    std::vector<DML_BINDING_DESC> softmax_final_output_bindings = {
        output_binding_desc};

    compiled_softmax_to_final_op->Execute(softmax_input_bindings,
                                          softmax_final_output_bindings);
  }
}

#define DECLARE_IMPL(T)                                     \
  template void SoftMax::compute<Device::DirectML, T>(      \
      const StorageView& input, const StorageView* lengths, \
      StorageView& output) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML