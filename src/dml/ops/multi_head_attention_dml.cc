#include "ctranslate2/ops/multi_head_attention.h"

#ifdef CT2_WITH_DIRECTML
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <>
void MultiHeadAttention::compute<Device::DirectML, float>(
    const StorageView& queries,
    const StorageView& keys,
    const StorageView& values,
    const StorageView* bias,
    const StorageView* mask,
    const StorageView* relative_position_bias,
    StorageView& output,
    StorageView* present_keys,
    StorageView* present_values) const {
  dml::utils::DmlOperatorDescBundle op_bundle;
  DML_TENSOR_DATA_TYPE key_value_dtype =
      dml::utils::get_dml_data_type(keys.dtype());
  // remove the first dimension if is 1
  Shape keys_shape = keys.shape();

  if ((keys_shape[0] != 1 || keys.rank() != 4) &&
      (values.shape()[0] != 1 || values.rank() != 4)) {
    throw std::invalid_argument(
        "MultiHeadAttention requires the first dimension of keys and values to "
        "be 1, but got: " +
        std::to_string(keys_shape[0]));
  }

  std::vector<UINT> key_value_dims_original = {
      1, static_cast<UINT>(keys_shape[2]),
      static_cast<UINT>(keys_shape[1] * keys_shape[3])};

  auto& query_desc = op_bundle.AddInput(queries);
  auto& key_desc =
      op_bundle.AddInput(key_value_dtype, key_value_dims_original, nullptr);
  auto& value_desc =
      op_bundle.AddInput(key_value_dtype, key_value_dims_original, nullptr);
  const DML_TENSOR_DESC* bias_desc =
      bias ? &op_bundle.AddInput(*bias).get_tensor_desc() : nullptr;
  const DML_TENSOR_DESC* mask_desc =
      mask ? &op_bundle.AddInput(*mask).get_tensor_desc() : nullptr;
  const DML_TENSOR_DESC* relative_position_bias_desc =
      relative_position_bias
          ? &op_bundle.AddInput(*relative_position_bias).get_tensor_desc()
          : nullptr;

  auto& output_desc = op_bundle.AddOutput(output);
  const DML_TENSOR_DESC* present_key_desc =
      present_keys ? &op_bundle.AddOutput(*present_keys).get_tensor_desc()
                   : nullptr;
  const DML_TENSOR_DESC* present_value_desc =
      present_values ? &op_bundle.AddOutput(*present_values).get_tensor_desc()
                     : nullptr;

  auto& attn_desc =
      op_bundle.GetOperatorDesc<DML_MULTIHEAD_ATTENTION_OPERATOR_DESC>();
  attn_desc.QueryTensor = &query_desc.get_tensor_desc();
  attn_desc.KeyTensor = &key_desc.get_tensor_desc();
  attn_desc.ValueTensor = &value_desc.get_tensor_desc();
  attn_desc.BiasTensor = bias_desc;
  attn_desc.MaskTensor = mask_desc;
  attn_desc.RelativePositionBiasTensor = relative_position_bias_desc;
  attn_desc.OutputTensor = &output_desc.get_tensor_desc();
  attn_desc.OutputPresentKeyTensor = present_key_desc;
  attn_desc.OutputPresentValueTensor = present_value_desc;
  attn_desc.Scale = _scale;
  attn_desc.MaskFilterValue = _mask_filter_value;
  attn_desc.HeadCount = _head_count;
  attn_desc.MaskType =
      _is_causal ? DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH
                 : DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;

  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      std::move(op_bundle), DML_EXECUTION_FLAG_NONE);

  dml::utils::DmlBindingArrayBundle inputs({
      {dml::utils::ResourceFromStorageView(queries), 0, 0},
      {dml::utils::ResourceFromStorageView(keys), 0, 0},
      {dml::utils::ResourceFromStorageView(values), 0, 0},
  });
  if (bias)
    inputs.AddBinding(dml::utils::ResourceFromStorageView(*bias), 0, 0);
  if (mask)
    inputs.AddBinding(dml::utils::ResourceFromStorageView(*mask), 0, 0);
  if (relative_position_bias)
    inputs.AddBinding(
        dml::utils::ResourceFromStorageView(*relative_position_bias), 0, 0);

  dml::utils::DmlBindingArrayBundle outputs({
      {dml::utils::ResourceFromStorageView(output), 0, 0},
  });
  if (present_keys)
    outputs.AddBinding(dml::utils::ResourceFromStorageView(*present_keys), 0,
                       0);
  if (present_values)
    outputs.AddBinding(dml::utils::ResourceFromStorageView(*present_values), 0,
                       0);

  compiled_op->Execute(inputs, outputs);
}

}  // namespace ops
}  // namespace ctranslate2

#endif
