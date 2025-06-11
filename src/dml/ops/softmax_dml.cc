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
  auto* device_context = dml::get_device();
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

  dml::utils::DmlBufferBindingBundle input_buffer_binding_bundle(
      input_resource, 0,
      input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  const DML_BINDING_DESC input_binding_desc =
      input_buffer_binding_bundle.get_desc();
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

    dml::utils::DmlBufferBindingBundle seq_indices_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(sequence_indices_gpu), 0,
        sequence_indices_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC seq_indices_binding_fill_output =
        seq_indices_buffer_binding_bundle.get_desc();
    dml::utils::DmlBindingArrayBundle fill_seq_output_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            seq_indices_buffer_binding_bundle});
    compiled_fill_seq_op->Execute({}, fill_seq_output_bundle.get_descs());

    // 1.b Prepare broadcast-enabled tensor descriptors for the LESS_THAN
    // operation. Instead of physically tiling the `sequence_indices` and
    // `lengths` tensors, we create DML tensor descriptors that instruct DML to
    // broadcast them during the element-wise comparison. This is more efficient
    // as it avoids allocating and writing to large intermediate tensors.

    // For `sequence_indices_gpu` (physical shape [1, depth]):
    // Broadcast to [batch_size, depth] by setting stride for the batch
    // dimension to 0.
    std::vector<UINT> bcast_iota_dims = {static_cast<UINT>(batch_size),
                                         static_cast<UINT>(depth)};
    std::vector<UINT> bcast_iota_strides = {
        0, 1};  // Stride 0 for batch, 1 for depth
    dml::utils::DmlTensorDescBundle bcast_sequence_indices_desc_bundle(
        DML_TENSOR_DATA_TYPE_INT32, bcast_iota_dims, &bcast_iota_strides,
        sequence_indices_gpu.size() * sizeof(int32_t));
    const DML_TENSOR_DESC& dml_bcast_sequence_indices_desc_ref =
        bcast_sequence_indices_desc_bundle.get_tensor_desc();

    // For `lengths` tensor (physical shape [batch_size]):
    // Treat as [batch_size, 1] and broadcast to [batch_size, depth]
    // by setting the stride for the depth dimension to 0.
    std::vector<UINT> bcast_lengths_dims = {static_cast<UINT>(batch_size),
                                            static_cast<UINT>(depth)};
    std::vector<UINT> bcast_lengths_strides = {
        1, 0};  // Stride 1 for batch, 0 for depth
    dml::utils::DmlTensorDescBundle bcast_lengths_desc_bundle(
        DML_TENSOR_DATA_TYPE_INT32, bcast_lengths_dims, &bcast_lengths_strides,
        lengths->size() * sizeof(int32_t));
    const DML_TENSOR_DESC& dml_bcast_lengths_desc_ref =
        bcast_lengths_desc_bundle.get_tensor_desc();

    // The binding for the `lengths` tensor is straightforward, pointing to its
    // resource.
    dml::utils::DmlBufferBindingBundle lengths_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(*lengths), 0,
        lengths->size() * sizeof(int32_t));

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
    // Broadcasted iota [batch_size, depth]
    less_than_op_desc.ATensor = &dml_bcast_sequence_indices_desc_ref;
    // Broadcasted lengths [batch_size, depth]
    less_than_op_desc.BTensor = &dml_bcast_lengths_desc_ref;
    // Output [batch_size, depth]
    less_than_op_desc.OutputTensor = &dml_condition_desc_ref;
    DML_OPERATOR_DESC less_than_op_meta_desc = {
        DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN, &less_than_op_desc};
    auto compiled_less_than_op =
        dml::GetOrCreateCompiledOperatorApi(&less_than_op_meta_desc);

    dml::utils::DmlBufferBindingBundle
        condition_buffer_binding_for_cmp_output_bundle(
            dml::utils::ResourceFromStorageView(condition_tensor_gpu), 0,
            condition_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC condition_binding_cmp_output =
        condition_buffer_binding_for_cmp_output_bundle.get_desc();

    // Inputs to LESS_THAN are the original (but broadcastable) sequence_indices
    // and lengths tensors.
    dml::utils::DmlBindingArrayBundle less_than_inputs_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            seq_indices_buffer_binding_bundle, lengths_buffer_binding_bundle});
    dml::utils::DmlBindingArrayBundle less_than_output_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            condition_buffer_binding_for_cmp_output_bundle});
    compiled_less_than_op->Execute(less_than_inputs_bundle.get_descs(),
                                   less_than_output_bundle.get_descs());

    // 2. Prepare Padding Value Tensor (on GPU using DML_FILL_VALUE_CONSTANT)
    // For pre-softmax masking, padding with -infinity correctly excludes
    // elements.
    T padding_scalar_value = -std::numeric_limits<T>::infinity();

    StorageView padding_value_gpu_storage(output.dtype(), output.device());
    // Same shape as input/output
    padding_value_gpu_storage.resize(output.shape());
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

    dml::utils::DmlBufferBindingBundle padding_value_binding_bundle(  // Renamed
        dml::utils::ResourceFromStorageView(padding_value_gpu_storage), 0,
        padding_value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC padding_value_binding_desc =  // Renamed
        padding_value_binding_bundle.get_desc();
    dml::utils::DmlBindingArrayBundle fill_padding_output_bundle(  // Renamed
        std::vector<dml::utils::DmlBufferBindingBundle>{
            padding_value_binding_bundle});
    compiled_fill_padding_op->Execute({},
                                      fill_padding_output_bundle.get_descs());

    // 3. ELEMENT_WISE_IF to create `masked_logits`
    StorageView masked_logits_storage(output.dtype(), output.device());
    masked_logits_storage.resize(output.shape());
    dml::utils::DmlTensorDescBundle masked_logits_desc_bundle(
        masked_logits_storage.dtype(), dml_dims_vec, nullptr,
        masked_logits_storage.size() * sizeof(T));
    const DML_TENSOR_DESC& dml_masked_logits_desc_ref =
        masked_logits_desc_bundle.get_tensor_desc();

    // (from step 1)
    DML_ELEMENT_WISE_IF_OPERATOR_DESC if_logits_desc = {};
    if_logits_desc.ConditionTensor = &dml_condition_desc_ref;
    // Original logits
    if_logits_desc.ATensor = &dml_input_desc_ref;
    // (from step 2) -inf padding
    if_logits_desc.BTensor = &dml_padding_value_desc_ref;
    // Output to intermediate
    if_logits_desc.OutputTensor = &dml_masked_logits_desc_ref;

    DML_OPERATOR_DESC if_logits_op_meta_desc = {DML_OPERATOR_ELEMENT_WISE_IF,
                                                &if_logits_desc};
    auto compiled_if_logits_op =
        dml::GetOrCreateCompiledOperatorApi(&if_logits_op_meta_desc);

    dml::utils::DmlBufferBindingBundle masked_logits_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(masked_logits_storage), 0,
        masked_logits_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC masked_logits_binding_if_output =
        masked_logits_buffer_binding_bundle.get_desc();

    dml::utils::DmlBindingArrayBundle if_logits_op_input_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            condition_buffer_binding_for_cmp_output_bundle,  // Condition
                                                             // (output of
                                                             // LESS_THAN)
            input_buffer_binding_bundle,  // ATensor (Original input logits)
            padding_value_binding_bundle  // BTensor (Padding values tensor)
        });
    dml::utils::DmlBindingArrayBundle if_logits_op_output_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            masked_logits_buffer_binding_bundle  // Output (masked_logits)
        });
    compiled_if_logits_op->Execute(
        if_logits_op_input_bindings_bundle.get_descs(),
        if_logits_op_output_bindings_bundle.get_descs());

    // 4. Softmax or LogSoftmax on `masked_logits` into the final output
    DML_OPERATOR_DESC softmax_on_masked_desc = {};
#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      // Input is masked_logits
      log_softmax_op_def_storage.InputTensor = &dml_masked_logits_desc_ref;
      // Final output
      log_softmax_op_def_storage.OutputTensor = &dml_output_desc_ref;
      log_softmax_op_def_storage.AxisCount = 1;
      log_softmax_op_def_storage.Axes = &axis;
      softmax_on_masked_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
      softmax_on_masked_desc.Desc = &log_softmax_op_def_storage;
    } else {
      // Input is masked_logits
      softmax_op_def_storage.InputTensor = &dml_masked_logits_desc_ref;
      // Final output
      softmax_op_def_storage.OutputTensor = &dml_output_desc_ref;
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

    dml::utils::DmlBufferBindingBundle
        final_output_buffer_binding_softmax_bundle(
            dml::utils::ResourceFromStorageView(output), 0,
            output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC final_output_binding_softmax_output =
        final_output_buffer_binding_softmax_bundle.get_desc();

    dml::utils::DmlBindingArrayBundle softmax_op_input_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            masked_logits_buffer_binding_bundle});
    dml::utils::DmlBindingArrayBundle softmax_op_output_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            final_output_buffer_binding_softmax_bundle});

    compiled_softmax_on_masked_op->Execute(
        softmax_op_input_bindings_bundle.get_descs(),
        softmax_op_output_bindings_bundle.get_descs());
  } else {
    // --- Original path: Softmax directly to final 'output' ---
    DML_OPERATOR_DESC softmax_op_desc_to_final = {};
#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      log_softmax_op_def_storage.InputTensor = &dml_input_desc_ref;
      // Final output
      log_softmax_op_def_storage.OutputTensor = &dml_output_desc_ref;
      log_softmax_op_def_storage.AxisCount = 1;
      log_softmax_op_def_storage.Axes = &axis;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
      softmax_op_desc_to_final.Desc = &log_softmax_op_def_storage;
    } else {
      softmax_op_def_storage.InputTensor = &dml_input_desc_ref;
      // Final output
      softmax_op_def_storage.OutputTensor = &dml_output_desc_ref;
      softmax_op_def_storage.AxisCount = 1;
      softmax_op_def_storage.Axes = &axis;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
      softmax_op_desc_to_final.Desc = &softmax_op_def_storage;
    }
#else
    if (_log) {
      log_softmax_old_op_def_storage.InputTensor = &dml_input_desc_ref;
      // Final output
      log_softmax_old_op_def_storage.OutputTensor = &dml_output_desc_ref;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
      softmax_op_desc_to_final.Desc = &log_softmax_old_op_def_storage;
    } else {
      softmax_old_op_def_storage.InputTensor = &dml_input_desc_ref;
      // Final output
      softmax_old_op_def_storage.OutputTensor = &dml_output_desc_ref;
      softmax_op_desc_to_final.Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
      softmax_op_desc_to_final.Desc = &softmax_old_op_def_storage;
    }
#endif
    auto compiled_softmax_to_final_op =
        dml::GetOrCreateCompiledOperatorApi(&softmax_op_desc_to_final);

    dml::utils::DmlBufferBindingBundle output_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(output), 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    const DML_BINDING_DESC output_binding_desc =
        output_buffer_binding_bundle.get_desc();
    std::vector<DML_BINDING_DESC> softmax_final_output_bindings = {
        output_binding_desc};

    dml::utils::DmlBindingArrayBundle softmax_input_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            input_buffer_binding_bundle});
    dml::utils::DmlBindingArrayBundle softmax_final_output_bindings_bundle(
        std::vector<dml::utils::DmlBufferBindingBundle>{
            output_buffer_binding_bundle});

    compiled_softmax_to_final_op->Execute(
        softmax_input_bindings_bundle.get_descs(),
        softmax_final_output_bindings_bundle.get_descs());
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