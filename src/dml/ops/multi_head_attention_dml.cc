#include "ctranslate2/ops/multi_head_attention.h"

#ifdef CT2_WITH_DIRECTML
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {
namespace {

// Transposes a tensor from [batch, seq_len, heads, depth] to [batch, heads,
// seq_len, depth]. The input StorageView has a shape of [1, batch, seq_len,
// heads * depth].
void transpose_kv_to_attention_layout(const StorageView& input,
                                      StorageView& output,
                                      const dim_t head_count,
                                      const dim_t head_size) {
  const Device device = input.device();
  if (device != Device::DirectML)
    throw std::invalid_argument(
        "transpose_to_attention_layout is only for DirectML device");

  const DataType dtype = input.dtype();
  if (dtype != DataType::FLOAT32)
    throw std::invalid_argument(
        "transpose_to_attention_layout only supports float32");

  DML_TENSOR_DATA_TYPE dml_type = dml::utils::get_dml_data_type(input.dtype());
  const Shape& shape = input.shape();
  const dim_t batch_size = shape[1];   // 20 from (1,20,1,64)
  const dim_t seq_len = shape[2];      // 1
  const dim_t hidden_size = shape[3];  // 64

  // Validate head_size parameter
  if (head_count * head_size > hidden_size)
    throw std::invalid_argument(
        "head_count * head_size exceeds hidden_size dimension");

  const UINT input_strides[4] = {
      static_cast<UINT>(seq_len * hidden_size),  // 1*64=64
      static_cast<UINT>(hidden_size),            // 64
      static_cast<UINT>(head_size),              // 3 (new parameter)
      static_cast<UINT>(1)};
  const std::vector<UINT> out_dims = {static_cast<UINT>(batch_size),  // 20
                                      static_cast<UINT>(head_count),  // 20
                                      static_cast<UINT>(seq_len),     // 1
                                      static_cast<UINT>(head_size)};  // 3
  const int perm[4] = {0, 2, 1, 3};

  output.resize({batch_size, head_count, seq_len, head_size});

  std::vector<UINT> perm_strides = {input_strides[perm[0]],   // 64 (stride[0])
                                    input_strides[perm[1]],   // 3  (stride[2])
                                    input_strides[perm[2]],   // 64 (stride[1])
                                    input_strides[perm[3]]};  // 1  (stride[3])

  ::ctranslate2::dml::utils::DmlOperatorDescBundle op_bundle;
  auto& src_desc = op_bundle.AddInput(dml_type, out_dims, &perm_strides, 0);
  auto& dst_desc = op_bundle.AddOutput(dml_type, out_dims, nullptr, 0);

  auto& identity_desc =
      op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
  identity_desc.InputTensor = &src_desc.get_tensor_desc();
  identity_desc.OutputTensor = &dst_desc.get_tensor_desc();

  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      std::move(op_bundle), DML_EXECUTION_FLAG_NONE,
      L"transpose_kv_to_attention_layout");

  IResourceWrapper* input_resource = dml::utils::ResourceFromStorageView(input);

  compiled_op->Execute({{input_resource, 0, 0}},
                       {{dml::utils::ResourceFromStorageView(output), 0, 0}});
}

// Transposes a tensor from [batch, heads, seq_len, depth] to [batch, seq_len,
// heads, depth]. The input StorageView has a shape of [1, batch, seq_len, heads
// * depth], but the data is in the transposed layout.
void transpose_kv_from_attention_layout(const StorageView& input,
                                        StorageView& output,
                                        const dim_t head_count,
                                        const dim_t hidden_size) {
  const Device device = input.device();
  DML_TENSOR_DATA_TYPE dml_type = dml::utils::get_dml_data_type(input.dtype());
  if (device != Device::DirectML)
    throw std::invalid_argument(
        "transpose_from_attention_layout is only for DirectML device");

  const DataType dtype = input.dtype();
  if (dtype != DataType::FLOAT32)
    throw std::invalid_argument(
        "transpose_from_attention_layout only supports float32");

  const Shape& shape = input.shape();
  const dim_t batch_size = shape[0];
  const dim_t head_count_ = shape[1];
  const dim_t seq_len = shape[2];
  const dim_t head_size = shape[3];
  if (head_count_ != head_count) {
    throw std::invalid_argument("Head count mismatch: expected " +
                                std::to_string(head_count) + ", got " +
                                std::to_string(head_count_));
  }

  // The input data is in [batch, heads, seq_len, head_size] layout.
  const dim_t perm[4] = {0, 2, 1, 3};
  const std::vector<UINT> out_dims = {
      static_cast<UINT>(1), static_cast<UINT>(batch_size),
      static_cast<UINT>(seq_len), static_cast<UINT>(hidden_size)};
  output.resize({1, batch_size, seq_len, hidden_size});
  const UINT input_strides[4] = {
      static_cast<UINT>(seq_len * head_size * head_count),
      static_cast<UINT>(head_size * head_count), static_cast<UINT>(head_size),
      static_cast<UINT>(1)};
  std::vector<UINT> perm_strides = {
      input_strides[perm[0]], input_strides[perm[1]], input_strides[perm[2]],
      input_strides[perm[3]]};
  ::ctranslate2::dml::utils::DmlOperatorDescBundle op_bundle;
  auto& src_desc = op_bundle.AddInput(dml_type, out_dims, &perm_strides, 0);

  auto& dst_desc = op_bundle.AddOutput(dml_type, out_dims, nullptr, 0);

  auto& identity_desc =
      op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
  identity_desc.InputTensor = &src_desc.get_tensor_desc();
  identity_desc.OutputTensor = &dst_desc.get_tensor_desc();

  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      std::move(op_bundle), DML_EXECUTION_FLAG_NONE,
      L"transpose_kv_from_attention");

  IResourceWrapper* input_resource = dml::utils::ResourceFromStorageView(input);

  compiled_op->Execute({{input_resource, 0, 0}},
                       {{dml::utils::ResourceFromStorageView(output), 0, 0}});
}

void transpose_mask_to_attention_layout(const StorageView& input,
                                        StorageView& output) {
  const Device device = input.device();
  if (device != Device::DirectML)
    throw std::invalid_argument(
        "transpose_mask_to_attention_layout is only for DirectML device");

  const DataType dtype = input.dtype();
  if (dtype != DataType::INT32)
    throw std::invalid_argument(
        "transpose_mask_to_attention_layout only supports float32");
  if (input.rank() != 3) {
    throw std::invalid_argument(
        "transpose_mask_to_attention_layout expects a 3D tensor, but got "
        "rank " +
        std::to_string(input.rank()));
  }
  if (input.dim(0) != 1) {
    throw std::invalid_argument(
        "transpose_mask_to_attention_layout expects a batch size of 1, but "
        "got " +
        std::to_string(input.dim(0)));
  }
  if (input.dim(2) != 2) {
    throw std::invalid_argument(
        "transpose_mask_to_attention_layout expects a mask with 2 dimensions, "
        "but got " +
        std::to_string(input.dim(2)));
  }

  const Shape& shape = input.shape();
  const dim_t batch_size = shape[1];

  output.resize({2, batch_size});
  const dim_t dims[2] = {batch_size, 2};

  primitives<Device::DirectML>::transpose_2d(input.data<int32_t>(), dims,
                                             output.data<int32_t>());
}

}  // anonymous namespace

template <>
void MultiHeadAttention::compute<Device::DirectML, float>(
    const StorageView& queries,
    const StorageView& keys,
    const StorageView& values,
    const StorageView* bias,
    const StorageView* mask,
    const StorageView* relative_position_bias,
    const StorageView* past_keys,
    const StorageView* past_values,
    StorageView& output,
    StorageView* present_keys,
    StorageView* present_values) const {
  dml::utils::DmlOperatorDescBundle op_bundle;
  DML_TENSOR_DATA_TYPE qkv_dtype = dml::utils::get_dml_data_type(keys.dtype());
  if (queries.rank() != 4) {
    throw std::invalid_argument("Queries must have rank 4, but got rank " +
                                std::to_string(queries.rank()));
  }
  const Shape query_shape = queries.shape();
  const Shape kv_shape = keys.shape();
  if (kv_shape != values.shape()) {
    throw std::invalid_argument(
        "Queries, keys, and values must have the same shape.");
  }
  if (present_keys && present_keys->shape() != keys.shape()) {
    throw std::invalid_argument(
        "Present keys must have the same shape as keys.");
  }
  if (present_values && present_values->shape() != values.shape()) {
    throw std::invalid_argument(
        "Present values must have the same shape as values.");
  }
  if (past_keys && past_keys->shape() != keys.shape()) {
    throw std::invalid_argument("Past keys must have the same shape as keys.");
  }
  if (past_values && past_values->shape() != values.shape()) {
    throw std::invalid_argument(
        "Past values must have the same shape as values.");
  }
  UINT head_size = query_shape[queries.rank() - 1] / _head_count;
  UINT hidden_size = head_size * _head_count;

  // Calculate the effective query shape and strides.
  std::vector<UINT> effective_query_shape =
      dml::utils::to_dml_dims(query_shape, queries.rank());
  effective_query_shape[queries.rank() - 1] = hidden_size;

  std::vector<UINT> effective_query_strides;
  for (dim_t i = 0; i < queries.rank(); ++i) {
    effective_query_strides.push_back(queries.stride(i));
  }
  std::vector<UINT>* const effective_query_strides_ptr =
      &effective_query_strides;

  //  Calculate the effective key/value shape and strides.
  std::vector<UINT> effective_kv_shape;
  for (dim_t i = 0; i < keys.rank(); ++i) {
    effective_kv_shape.push_back(kv_shape[i]);
  }
  effective_kv_shape[keys.rank() - 1] = hidden_size;
  std::vector<UINT> effective_kv_strides;
  for (dim_t i = 0; i < keys.rank(); ++i) {
    effective_kv_strides.push_back(keys.stride(i));
  }
  std::vector<UINT>* const effective_kv_strides_ptr = &effective_kv_strides;

  // Handle pass keys/values, present keys/values, which input in the shape
  // [1, batch_size, sequence_length, head_count * head_size] (treated as
  // [batch_size, sequence_length, head_count, head_size]). need to transpose
  // into [batch_size, head_count, sequence_length, head_size]. Use primitives
  // transpose_4d to handle this. and the end the present keys/values need to
  // transpose back to [1, batch_size, sequence_length, head_count * head_size].
  StorageView past_keys_processed(Device::DirectML, DataType::FLOAT32),
      past_values_processed(Device::DirectML, DataType::FLOAT32);
  if (past_keys) {
    transpose_kv_to_attention_layout(*past_keys, past_keys_processed,
                                     _head_count, head_size);
  }
  if (past_values) {
    transpose_kv_to_attention_layout(*past_values, past_values_processed,
                                     _head_count, head_size);
  }

  StorageView present_keys_processed(Device::DirectML, DataType::FLOAT32),
      present_values_processed(Device::DirectML, DataType::FLOAT32);
  if (present_keys) {
    present_keys_processed.resize({kv_shape[1], _head_count,
                                   kv_shape[2] + present_keys->dim(2),
                                   head_size});
  }
  if (present_values) {
    present_values_processed.resize({kv_shape[1], _head_count,
                                     kv_shape[2] + present_values->dim(2),
                                     head_size});
  }

  StorageView mask_processed(Device::DirectML, DataType::INT32);
  if (mask) {
    transpose_mask_to_attention_layout(*mask, mask_processed);
  }

  auto& query_desc = op_bundle.AddInput(qkv_dtype, effective_query_shape,
                                        effective_query_strides_ptr);
  auto& key_desc = op_bundle.AddInput(qkv_dtype, effective_kv_shape,
                                      effective_kv_strides_ptr);
  auto& value_desc = op_bundle.AddInput(qkv_dtype, effective_kv_shape,
                                        effective_kv_strides_ptr);
  const DML_TENSOR_DESC* bias_desc =
      bias ? &op_bundle.AddInput(*bias).get_tensor_desc() : nullptr;
  const DML_TENSOR_DESC* mask_desc =
      mask ? &op_bundle.AddInput(mask_processed).get_tensor_desc() : nullptr;
  const DML_TENSOR_DESC* relative_position_bias_desc =
      relative_position_bias
          ? &op_bundle.AddInput(*relative_position_bias).get_tensor_desc()
          : nullptr;
  const DML_TENSOR_DESC* past_key_desc =
      past_keys ? &op_bundle.AddInput(past_keys_processed).get_tensor_desc()
                : nullptr;
  const DML_TENSOR_DESC* past_value_desc =
      past_values ? &op_bundle.AddInput(past_values_processed).get_tensor_desc()
                  : nullptr;

  auto& output_desc = op_bundle.AddOutput(qkv_dtype, effective_query_shape,
                                          effective_query_strides_ptr);
  const DML_TENSOR_DESC* present_key_desc =
      present_keys
          ? &op_bundle.AddOutput(present_keys_processed).get_tensor_desc()
          : nullptr;
  const DML_TENSOR_DESC* present_value_desc =
      present_values
          ? &op_bundle.AddOutput(present_values_processed).get_tensor_desc()
          : nullptr;

  auto& attn_desc =
      op_bundle.GetOperatorDesc<DML_MULTIHEAD_ATTENTION_OPERATOR_DESC>();
  attn_desc.QueryTensor = &query_desc.get_tensor_desc();
  attn_desc.KeyTensor = &key_desc.get_tensor_desc();
  attn_desc.ValueTensor = &value_desc.get_tensor_desc();
  attn_desc.BiasTensor = bias_desc;
  attn_desc.MaskTensor = mask_desc;
  attn_desc.RelativePositionBiasTensor = relative_position_bias_desc;
  attn_desc.PastKeyTensor = past_key_desc;
  attn_desc.PastValueTensor = past_value_desc;
  attn_desc.OutputTensor = &output_desc.get_tensor_desc();
  attn_desc.OutputPresentKeyTensor = present_key_desc;
  attn_desc.OutputPresentValueTensor = present_value_desc;
  attn_desc.Scale = _scale;
  attn_desc.MaskFilterValue = _mask_filter_value;
  attn_desc.HeadCount = _head_count;
  attn_desc.MaskType =
      mask ? DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START
           : DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;

  dml::Operator* compiled_op = dml::GetOrCreateCompiledOperatorApi(
      std::move(op_bundle), DML_EXECUTION_FLAG_NONE);

  dml::utils::DmlBindingArrayBundle inputs({
      {dml::utils::ResourceFromStorageView(queries), 0, 0},
      {dml::utils::ResourceFromStorageView(keys), 0, 0},
      {dml::utils::ResourceFromStorageView(values), 0, 0},
      nullptr,  // StackedQueryKeyTensor
      nullptr,  // StackedKeyValueTensor
      nullptr,  // StackedQueryKeyValueTensor
  });
  if (bias) {
    inputs.AddBinding(dml::utils::ResourceFromStorageView(*bias), 0, 0);
  } else {
    inputs.AddBinding(nullptr, 0, 0);
  }
  if (mask) {
    inputs.AddBinding(dml::utils::ResourceFromStorageView(mask_processed), 0,
                      0);
  } else {
    inputs.AddBinding(nullptr, 0, 0);
  }
  if (relative_position_bias) {
    inputs.AddBinding(
        dml::utils::ResourceFromStorageView(*relative_position_bias), 0, 0);
  } else {
    inputs.AddBinding(nullptr, 0, 0);
  }
  if (past_keys) {
    inputs.AddBinding(dml::utils::ResourceFromStorageView(past_keys_processed),
                      0, 0);
  } else {
    inputs.AddBinding(nullptr, 0, 0);
  }
  if (past_values) {
    inputs.AddBinding(
        dml::utils::ResourceFromStorageView(past_values_processed), 0, 0);
  } else {
    inputs.AddBinding(nullptr, 0, 0);
  }

  dml::utils::DmlBindingArrayBundle outputs({
      {dml::utils::ResourceFromStorageView(output), 0, 0},
  });
  if (present_keys) {
    outputs.AddBinding(
        dml::utils::ResourceFromStorageView(present_keys_processed), 0, 0);
  } else {
    outputs.AddBinding(nullptr, 0, 0);
  }
  if (present_values) {
    outputs.AddBinding(
        dml::utils::ResourceFromStorageView(present_values_processed), 0, 0);
  } else {
    outputs.AddBinding(nullptr, 0, 0);
  }

  compiled_op->Execute(inputs, outputs);

  if (present_keys) {
    transpose_kv_from_attention_layout(present_keys_processed, *present_keys,
                                       _head_count,
                                       query_shape[queries.rank() - 1]);
  }
  if (present_values) {
    transpose_kv_from_attention_layout(present_values_processed,
                                       *present_values, _head_count,
                                       query_shape[queries.rank() - 1]);
  }
}

}  // namespace ops
}  // namespace ctranslate2

#endif
