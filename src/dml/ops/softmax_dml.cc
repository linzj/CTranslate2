#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/softmax.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void SoftMax::compute(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output) const {
  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  std::vector<UINT> dml_dims_vec = {static_cast<UINT>(batch_size),
                                    static_cast<UINT>(depth)};

  StorageView input_storage;
  ID3D12Resource* input_resource = dml::utils::ResourceFromStorageView(input);
  if (input.buffer() == output.buffer()) {
    input_storage = std::move(output);
    StorageView new_output(input_storage.shape(), input_storage.dtype(),
                           input_storage.device());
    output = std::move(new_output);
  }

  UINT axis = 1;

  UINT64 input_buffer_size = input.size() * sizeof(T);
  dml::utils::DmlBufferBindingBundle input_buffer_binding_bundle(
      input_resource, 0, input_buffer_size);
  if (lengths) {
    // --- Path with masking: MASK FIRST, THEN SOFTMAX ---

    // 1. Create Iota tensor: [0, 1, ..., depth-1]
    StorageView sequence_indices_gpu(DataType::INT32, output.device());
    sequence_indices_gpu.resize(Shape{1, depth});
    std::vector<UINT> seq_idx_tensor_dims = {1, static_cast<UINT>(depth)};
    UINT64 sequence_indices_size =
        sequence_indices_gpu.size() * sizeof(int32_t);

    dml::utils::DmlOperatorDescBundle fill_seq_op_bundle;
    auto& sequence_indices_desc = fill_seq_op_bundle.AddOutput(
        DML_TENSOR_DATA_TYPE_INT32, seq_idx_tensor_dims, nullptr,
        sequence_indices_size);

    auto& fill_seq_desc =
        fill_seq_op_bundle
            .GetOperatorDesc<DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC>();
    fill_seq_desc.OutputTensor = &sequence_indices_desc.get_tensor_desc();
    fill_seq_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
    fill_seq_desc.ValueStart.Int32 = 0;
    fill_seq_desc.ValueDelta.Int32 = 1;
    auto compiled_fill_seq_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(fill_seq_op_bundle));

    dml::utils::DmlBufferBindingBundle seq_indices_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(sequence_indices_gpu), 0,
        sequence_indices_size);
    dml::utils::DmlBindingArrayBundle fill_seq_output_bundle{
        {dml::utils::ResourceFromStorageView(sequence_indices_gpu), 0,
         sequence_indices_size}};
    compiled_fill_seq_op->Execute({}, fill_seq_output_bundle);

    // 2. Create Condition Tensor using
    // DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN
    std::vector<UINT> bcast_iota_dims = {static_cast<UINT>(batch_size),
                                         static_cast<UINT>(depth)};
    std::vector<UINT> bcast_iota_strides = {0, 1};
    std::vector<UINT> bcast_lengths_dims = {static_cast<UINT>(batch_size),
                                            static_cast<UINT>(depth)};
    std::vector<UINT> bcast_lengths_strides = {1, 0};

    dml::utils::DmlOperatorDescBundle less_than_op_bundle;
    auto& bcast_seq_indices_desc = less_than_op_bundle.AddInput(
        DML_TENSOR_DATA_TYPE_INT32, bcast_iota_dims, &bcast_iota_strides,
        sequence_indices_size);
    auto& bcast_lengths_desc = less_than_op_bundle.AddInput(
        DML_TENSOR_DATA_TYPE_INT32, bcast_lengths_dims, &bcast_lengths_strides,
        lengths->size() * sizeof(int32_t));

    StorageView condition_tensor_gpu(DataType::INT8, output.device());
    condition_tensor_gpu.resize(Shape{batch_size, depth});
    auto& condition_desc =
        less_than_op_bundle.AddOutput(DML_TENSOR_DATA_TYPE_UINT8, dml_dims_vec,
                                      nullptr, condition_tensor_gpu.size());

    auto& less_than_op_desc = less_than_op_bundle.GetOperatorDesc<
        DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>();
    less_than_op_desc.ATensor = &bcast_seq_indices_desc.get_tensor_desc();
    less_than_op_desc.BTensor = &bcast_lengths_desc.get_tensor_desc();
    less_than_op_desc.OutputTensor = &condition_desc.get_tensor_desc();
    auto compiled_less_than_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(less_than_op_bundle));

    dml::utils::DmlBufferBindingBundle lengths_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(*lengths), 0,
        lengths->size() * sizeof(int32_t));
    dml::utils::DmlBufferBindingBundle
        condition_buffer_binding_for_cmp_output_bundle(
            dml::utils::ResourceFromStorageView(condition_tensor_gpu), 0,
            condition_desc.get_buffer_desc().TotalTensorSizeInBytes);

    dml::utils::DmlBindingArrayBundle less_than_inputs_bundle{
        {dml::utils::ResourceFromStorageView(sequence_indices_gpu), 0,
         sequence_indices_size},
        {dml::utils::ResourceFromStorageView(*lengths), 0,
         lengths->size() * sizeof(int32_t)}};
    dml::utils::DmlBindingArrayBundle less_than_output_bundle{
        {dml::utils::ResourceFromStorageView(condition_tensor_gpu), 0,
         condition_desc.get_buffer_desc().TotalTensorSizeInBytes}};
    compiled_less_than_op->Execute(less_than_inputs_bundle,
                                   less_than_output_bundle);

    // 3. Prepare Padding Value Tensor
    T padding_scalar_value = -std::numeric_limits<T>::infinity();
    StorageView padding_value_gpu_storage(output.dtype(), output.device());
    padding_value_gpu_storage.resize(output.shape());
    UINT64 padding_value_size = padding_value_gpu_storage.size() * sizeof(T);

    dml::utils::DmlOperatorDescBundle fill_padding_op_bundle;
    auto& padding_value_desc = fill_padding_op_bundle.AddOutput(
        padding_value_gpu_storage.dtype(), dml_dims_vec, nullptr,
        padding_value_size);
    auto& fill_op_padding_desc =
        fill_padding_op_bundle
            .GetOperatorDesc<DML_FILL_VALUE_CONSTANT_OPERATOR_DESC>();
    fill_op_padding_desc.OutputTensor = &padding_value_desc.get_tensor_desc();
    fill_op_padding_desc.ValueDataType = padding_value_desc.get_data_type();
    static_assert(sizeof(T) <= sizeof(fill_op_padding_desc.Value.Bytes),
                  "Size of T exceeds DML_SCALAR_UNION's byte array");
    std::memcpy(fill_op_padding_desc.Value.Bytes, &padding_scalar_value,
                sizeof(T));
    if (sizeof(T) < sizeof(fill_op_padding_desc.Value.Bytes)) {
      std::memset(fill_op_padding_desc.Value.Bytes + sizeof(T), 0,
                  sizeof(fill_op_padding_desc.Value.Bytes) - sizeof(T));
    }
    auto compiled_fill_padding_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(fill_padding_op_bundle));

    dml::utils::DmlBufferBindingBundle padding_value_binding_bundle(
        dml::utils::ResourceFromStorageView(padding_value_gpu_storage), 0,
        padding_value_size);
    dml::utils::DmlBindingArrayBundle fill_padding_output_bundle{
        {dml::utils::ResourceFromStorageView(padding_value_gpu_storage), 0,
         padding_value_size}};
    compiled_fill_padding_op->Execute({}, fill_padding_output_bundle);

    // 4. ELEMENT_WISE_IF to create `masked_logits`
    StorageView masked_logits_storage(output.dtype(), output.device());
    masked_logits_storage.resize(output.shape());
    UINT64 masked_logits_size = masked_logits_storage.size() * sizeof(T);

    dml::utils::DmlOperatorDescBundle if_logits_op_bundle;
    auto& condition_tensor_desc_if =
        if_logits_op_bundle.AddInput(DML_TENSOR_DATA_TYPE_UINT8, dml_dims_vec,
                                     nullptr, condition_tensor_gpu.size());
    auto& input_tensor_desc_if = if_logits_op_bundle.AddInput(
        input.dtype(), dml_dims_vec, nullptr, input_buffer_size);
    auto& padding_tensor_desc_if =
        if_logits_op_bundle.AddInput(padding_value_gpu_storage.dtype(),
                                     dml_dims_vec, nullptr, padding_value_size);
    auto& masked_logits_desc = if_logits_op_bundle.AddOutput(
        masked_logits_storage.dtype(), dml_dims_vec, nullptr,
        masked_logits_size);

    auto& if_logits_desc =
        if_logits_op_bundle
            .GetOperatorDesc<DML_ELEMENT_WISE_IF_OPERATOR_DESC>();
    if_logits_desc.ConditionTensor =
        &condition_tensor_desc_if.get_tensor_desc();
    if_logits_desc.ATensor = &input_tensor_desc_if.get_tensor_desc();
    if_logits_desc.BTensor = &padding_tensor_desc_if.get_tensor_desc();
    if_logits_desc.OutputTensor = &masked_logits_desc.get_tensor_desc();
    auto compiled_if_logits_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(if_logits_op_bundle));

    dml::utils::DmlBufferBindingBundle masked_logits_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(masked_logits_storage), 0,
        masked_logits_size);
    dml::utils::DmlBindingArrayBundle if_logits_op_input_bindings_bundle{
        {dml::utils::ResourceFromStorageView(condition_tensor_gpu), 0,
         condition_tensor_gpu.size()},
        {input_resource, 0, input_buffer_size},
        {dml::utils::ResourceFromStorageView(padding_value_gpu_storage), 0,
         padding_value_size}};
    dml::utils::DmlBindingArrayBundle if_logits_op_output_bindings_bundle{
        {dml::utils::ResourceFromStorageView(masked_logits_storage), 0,
         masked_logits_size}};
    compiled_if_logits_op->Execute(if_logits_op_input_bindings_bundle,
                                   if_logits_op_output_bindings_bundle);

    // 5. Softmax or LogSoftmax on `masked_logits` into the final output.
    dml::utils::DmlOperatorDescBundle softmax_on_masked_op_bundle;
    auto& masked_logits_desc_final = softmax_on_masked_op_bundle.AddInput(
        masked_logits_storage.dtype(), dml_dims_vec, nullptr,
        masked_logits_size);
    auto& output_desc_final = softmax_on_masked_op_bundle.AddOutput(
        output.dtype(), dml_dims_vec, nullptr, output.size() * sizeof(T));

#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      auto& op_desc =
          softmax_on_masked_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC>();
      op_desc.InputTensor = &masked_logits_desc_final.get_tensor_desc();
      op_desc.OutputTensor = &output_desc_final.get_tensor_desc();
      op_desc.AxisCount = 1;
      op_desc.Axes = &axis;
    } else {
      auto& op_desc =
          softmax_on_masked_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC>();
      op_desc.InputTensor = &masked_logits_desc_final.get_tensor_desc();
      op_desc.OutputTensor = &output_desc_final.get_tensor_desc();
      op_desc.AxisCount = 1;
      op_desc.Axes = &axis;
    }
#else
    if (_log) {
      auto& op_desc =
          softmax_on_masked_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC>();
      op_desc.InputTensor = &masked_logits_desc_final.get_tensor_desc();
      op_desc.OutputTensor = &output_desc_final.get_tensor_desc();
    } else {
      auto& op_desc =
          softmax_on_masked_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_SOFTMAX_OPERATOR_DESC>();
      op_desc.InputTensor = &masked_logits_desc_final.get_tensor_desc();
      op_desc.OutputTensor = &output_desc_final.get_tensor_desc();
    }
#endif
    auto compiled_softmax_on_masked_op = dml::GetOrCreateCompiledOperatorApi(
        std::move(softmax_on_masked_op_bundle));

    dml::utils::DmlBufferBindingBundle final_output_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(output), 0,
        output_desc_final.get_buffer_desc().TotalTensorSizeInBytes);
    dml::utils::DmlBindingArrayBundle softmax_op_input_bindings_bundle{
        {dml::utils::ResourceFromStorageView(masked_logits_storage), 0,
         masked_logits_size}};
    dml::utils::DmlBindingArrayBundle softmax_op_output_bindings_bundle{
        {dml::utils::ResourceFromStorageView(output), 0,
         output_desc_final.get_buffer_desc().TotalTensorSizeInBytes}};

    compiled_softmax_on_masked_op->Execute(softmax_op_input_bindings_bundle,
                                           softmax_op_output_bindings_bundle);
  } else {
    // --- Original path: Softmax directly to final 'output' ---
    dml::utils::DmlOperatorDescBundle softmax_op_bundle;
    auto& input_desc = softmax_op_bundle.AddInput(
        input.dtype(), dml_dims_vec, nullptr, input.size() * sizeof(T));
    auto& output_desc = softmax_op_bundle.AddOutput(
        output.dtype(), dml_dims_vec, nullptr, output.size() * sizeof(T));

#if DML_TARGET_VERSION >= 0x5100
    if (_log) {
      auto& op_desc =
          softmax_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC>();
      op_desc.InputTensor = &input_desc.get_tensor_desc();
      op_desc.OutputTensor = &output_desc.get_tensor_desc();
      op_desc.AxisCount = 1;
      op_desc.Axes = &axis;
    } else {
      auto& op_desc =
          softmax_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC>();
      op_desc.InputTensor = &input_desc.get_tensor_desc();
      op_desc.OutputTensor = &output_desc.get_tensor_desc();
      op_desc.AxisCount = 1;
      op_desc.Axes = &axis;
    }
#else
    if (_log) {
      auto& op_desc =
          softmax_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC>();
      op_desc.InputTensor = &input_desc.get_tensor_desc();
      op_desc.OutputTensor = &output_desc.get_tensor_desc();
    } else {
      auto& op_desc =
          softmax_op_bundle
              .GetOperatorDesc<DML_ACTIVATION_SOFTMAX_OPERATOR_DESC>();
      op_desc.InputTensor = &input_desc.get_tensor_desc();
      op_desc.OutputTensor = &output_desc.get_tensor_desc();
    }
#endif
    auto compiled_softmax_to_final_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(softmax_op_bundle));

    dml::utils::DmlBufferBindingBundle output_buffer_binding_bundle(
        dml::utils::ResourceFromStorageView(output), 0,
        output_desc.get_buffer_desc().TotalTensorSizeInBytes);

    dml::utils::DmlBindingArrayBundle softmax_input_bindings_bundle{
        {input_resource, 0, input_buffer_size}};
    dml::utils::DmlBindingArrayBundle softmax_final_output_bindings_bundle{
        {dml::utils::ResourceFromStorageView(output), 0,
         output_desc.get_buffer_desc().TotalTensorSizeInBytes}};

    compiled_softmax_to_final_op->Execute(softmax_input_bindings_bundle,
                                          softmax_final_output_bindings_bundle);
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