#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"

#include "ctranslate2/storage_view.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

namespace {
// DML convolution expects 4D tensors. This helper reshapes CT2's 1D conv
// tensors (3D) to 4D. Input [N, C, W] -> [N, C, 1, W] Weight [Co, Ci, K] ->
// [Co, Ci, 1, K] Bias [Co] -> [1, Co, 1, 1] for broadcasting
std::vector<UINT> get_dml_tensor_shape_4d(const StorageView& view) {
  const auto& shape = view.shape();
  if (shape.size() == 1) {  // Bias
    return {1, static_cast<UINT>(shape[0]), 1, 1};
  }
  if (shape.size() == 3) {
    return {static_cast<UINT>(shape[0]), static_cast<UINT>(shape[1]), 1,
            static_cast<UINT>(shape[2])};
  }
  if (shape.size() == 4) {  // Already 4D
    std::vector<UINT> dims(4);
    for (size_t i = 0; i < 4; ++i)
      dims[i] = shape[i];
    return dims;
  }
  // Fallback for other cases.
  return dml::utils::to_dml_dims(view.shape(), view.size());
}
}  // anonymous namespace

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale) {
    // This implementation follows the CPU logic:
    // 1. Quantize input -> input_q8, input_scale
    // (DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR)
    // 2. Integer convolution -> output_q32 (DML_OPERATOR_CONVOLUTION_INTEGER)
    // 3. Dequantize output_q32 -> float output (CAST, two MULTIPLY)
    // 4. Add float bias (ADD)

    // 1. Quantize Input Tensor (Float -> INT8)
    StorageView input_q8({input.shape()}, int8_t(0), input.device());
    StorageView input_scale({1}, 0.0f, input.device());
    StorageView input_zero_point({1}, int8_t(0), input.device());
    {  // DYNAMIC_QUANTIZE_LINEAR
      dml::utils::DmlOperatorDescBundle op_bundle;

      auto& input_desc = op_bundle.AddInput(input);
      auto& input_q8_desc = op_bundle.AddOutput(input_q8);
      std::vector<UINT> scalar_dims(input.rank(), 1);
      auto& input_scale_desc =
          op_bundle.AddOutput(input_scale.dtype(), scalar_dims, nullptr,
                              input_scale.reserved_memory());
      auto& input_zero_point_desc = op_bundle.AddOutput(
          input_zero_point.dtype(), scalar_dims, nullptr, 0);

      auto& op_desc =
          op_bundle
              .GetOperatorDesc<DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC>();
      op_desc.InputTensor = &input_desc.get_tensor_desc();
      op_desc.OutputTensor = &input_q8_desc.get_tensor_desc();
      op_desc.OutputScaleTensor = &input_scale_desc.get_tensor_desc();
      op_desc.OutputZeroPointTensor = &input_zero_point_desc.get_tensor_desc();

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(input_q8)),
           dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(input_scale)),
           dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(input_zero_point))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 2. Integer Convolution
    StorageView output_q32(output.shape(), int32_t(0), output.device());
    StorageView weight_zero_point({1}, int8_t(0), weight.device());
    {  // CONVOLUTION_INTEGER
      dml::utils::DmlOperatorDescBundle op_bundle;

      std::vector<UINT> input_dims = get_dml_tensor_shape_4d(input_q8);
      auto& input_q8_desc = op_bundle.AddInput(
          input_q8.dtype(), input_dims, nullptr, input_q8.reserved_memory());
      auto& input_zp_desc = op_bundle.AddInput(input_zero_point);

      std::vector<UINT> weight_dims = get_dml_tensor_shape_4d(weight);
      auto& weight_desc = op_bundle.AddInput(weight.dtype(), weight_dims,
                                             nullptr, weight.reserved_memory());
      auto& weight_zp_desc = op_bundle.AddInput(weight_zero_point);

      std::vector<UINT> output_dims = get_dml_tensor_shape_4d(output_q32);
      auto& output_q32_desc =
          op_bundle.AddOutput(output_q32.dtype(), output_dims, nullptr,
                              output_q32.reserved_memory());

      auto& op_desc =
          op_bundle.GetOperatorDesc<DML_CONVOLUTION_INTEGER_OPERATOR_DESC>();
      op_desc.InputTensor = &input_q8_desc.get_tensor_desc();
      op_desc.InputZeroPointTensor = &input_zp_desc.get_tensor_desc();
      op_desc.FilterTensor = &weight_desc.get_tensor_desc();
      op_desc.FilterZeroPointTensor = &weight_zp_desc.get_tensor_desc();
      op_desc.OutputTensor = &output_q32_desc.get_tensor_desc();
      op_desc.DimensionCount = 2;
      UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
      op_desc.Strides = dml_strides;
      UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
      op_desc.Dilations = dml_dilations;
      UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
      op_desc.StartPadding = dml_start_padding;
      UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
      op_desc.EndPadding = dml_end_padding;
      op_desc.GroupCount = static_cast<UINT>(_groups);

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
      dml::utils::DmlBindingArrayBundle inputs({
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_q8)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_zero_point)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(weight)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(weight_zero_point)),
      });
      dml::utils::DmlBindingArrayBundle outputs(  //
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(output_q32))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 3. Dequantize
    StorageView final_float_output_storage;
    if (bias) {
      final_float_output_storage =
          StorageView(output.shape(), output.dtype(), output.device());
    }
    StorageView& final_float_output =
        bias ? final_float_output_storage : output;
    StorageView tmp_float(output.shape(), output.dtype(), output.device());

    // 3a. Cast to float
    {
      dml::utils::DmlOperatorDescBundle op_bundle;
      auto& in_desc = op_bundle.AddInput(output_q32);
      auto& out_desc = op_bundle.AddOutput(tmp_float);

      auto& op_desc = op_bundle.GetOperatorDesc<DML_CAST_OPERATOR_DESC>();
      op_desc.InputTensor = &in_desc.get_tensor_desc();
      op_desc.OutputTensor = &out_desc.get_tensor_desc();

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(output_q32))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(tmp_float))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 3b. Multiply by input_scale
    StorageView tmp_float2(output.shape(), output.dtype(), output.device());
    {
      dml::utils::DmlOperatorDescBundle op_bundle;

      auto& a_desc = op_bundle.AddInput(tmp_float);
      const auto target_dims =
          dml::utils::to_dml_dims(tmp_float.shape(), tmp_float.size());
      auto& b_desc =
          op_bundle.AddInputBroadcastFromSeach(input_scale, target_dims);
      auto& out_desc = op_bundle.AddOutput(tmp_float2);

      auto& op_desc =
          op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
      op_desc.ATensor = &a_desc.get_tensor_desc();
      op_desc.BTensor = &b_desc.get_tensor_desc();
      op_desc.OutputTensor = &out_desc.get_tensor_desc();

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

      dml::utils::DmlBindingArrayBundle inputs({
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(tmp_float)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_scale)),
      });
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(tmp_float2))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 3c. Multiply by weight_scale (*1/qscale) is DIVIDE
    {
      dml::utils::DmlOperatorDescBundle op_bundle;

      auto& a_desc = op_bundle.AddInput(tmp_float2);
      const auto target_dims =
          dml::utils::to_dml_dims(tmp_float2.shape(), tmp_float2.size());
      auto& b_desc = op_bundle.AddInputBroadcastFromSeach(*qscale, target_dims);
      auto& out_desc = op_bundle.AddOutput(final_float_output);

      auto& op_desc =
          op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>();
      op_desc.ATensor = &a_desc.get_tensor_desc();
      op_desc.BTensor = &b_desc.get_tensor_desc();
      op_desc.OutputTensor = &out_desc.get_tensor_desc();

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

      dml::utils::DmlBindingArrayBundle inputs({
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(tmp_float2)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(*qscale)),
      });
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(final_float_output))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 4. Add bias
    if (bias) {
      dml::utils::DmlOperatorDescBundle op_bundle;

      auto& a_desc = op_bundle.AddInput(final_float_output);
      const auto target_dims = dml::utils::to_dml_dims(
          final_float_output.shape(), final_float_output.size());
      auto& b_desc = op_bundle.AddInputBroadcastFromSeach(*bias, target_dims);
      auto& out_desc = op_bundle.AddOutput(output);

      auto& op_desc =
          op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
      op_desc.ATensor = &a_desc.get_tensor_desc();
      op_desc.BTensor = &b_desc.get_tensor_desc();
      op_desc.OutputTensor = &out_desc.get_tensor_desc();

      dml::Operator* dml_op =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

      dml::utils::DmlBindingArrayBundle inputs({
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(final_float_output)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(*bias)),
      });
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(output))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }
  } else {
    const DML_TENSOR_DATA_TYPE dml_data_type =
        dml::utils::get_dml_data_type(input.dtype());

    // 1. Prepare Tensor Descriptors
    dml::utils::DmlOperatorDescBundle op_bundle;
    std::vector<UINT> input_dml_dims_vec = get_dml_tensor_shape_4d(input);
    auto& input_desc_bundle = op_bundle.AddInput(
        input.dtype(), input_dml_dims_vec, nullptr, input.reserved_memory());

    std::vector<UINT> weight_dml_dims_vec = get_dml_tensor_shape_4d(weight);
    auto& weight_desc_bundle = op_bundle.AddInput(
        weight.dtype(), weight_dml_dims_vec, nullptr, weight.reserved_memory());

    const DML_TENSOR_DESC* bias_tensor_desc = nullptr;
    if (bias) {
      std::vector<UINT> bias_dml_dims_vec = get_dml_tensor_shape_4d(*bias);
      auto& bias_desc_bundle = op_bundle.AddInput(
          bias->dtype(), bias_dml_dims_vec, nullptr, bias->reserved_memory());
      bias_tensor_desc = &bias_desc_bundle.get_tensor_desc();
    }

    std::vector<UINT> output_dml_dims_vec = get_dml_tensor_shape_4d(output);
    auto& output_desc_bundle = op_bundle.AddOutput(
        output.dtype(), output_dml_dims_vec, nullptr, output.reserved_memory());

    // 2. Create Convolution Operator Descriptor
    auto& conv_op_desc =
        op_bundle.GetOperatorDesc<DML_CONVOLUTION_OPERATOR_DESC>();
    conv_op_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
    conv_op_desc.FilterTensor = &weight_desc_bundle.get_tensor_desc();
    conv_op_desc.BiasTensor = bias_tensor_desc;
    conv_op_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
    conv_op_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_op_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_op_desc.DimensionCount = 2;
    UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
    conv_op_desc.Strides = dml_strides;
    UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
    conv_op_desc.Dilations = dml_dilations;
    UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
    conv_op_desc.StartPadding = dml_start_padding;
    UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
    conv_op_desc.EndPadding = dml_end_padding;
    UINT dml_output_padding[] = {0, 0};
    conv_op_desc.OutputPadding = dml_output_padding;
    conv_op_desc.GroupCount = static_cast<UINT>(_groups);

    // 3. Get or Create Compiled Operator from Cache
    DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
    if (dml_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
      execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }

    dml::Operator* dml_convolution_operator =
        dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle),
                                            execution_flags);

    // 4. Prepare Bindings and Execute
    std::vector<dml::utils::DmlBufferBindingBundle> input_bundles;
    input_bundles.emplace_back(dml::utils::ResourceFromStorageView(input));
    input_bundles.emplace_back(dml::utils::ResourceFromStorageView(weight));
    if (bias) {
      input_bundles.emplace_back(dml::utils::ResourceFromStorageView(*bias));
    }

    dml::utils::DmlBindingArrayBundle input_bindings(std::move(input_bundles));
    dml::utils::DmlBindingArrayBundle output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});

    dml_convolution_operator->Execute(input_bindings.get_descs(),
                                      output_bindings.get_descs());
  }
}

#define DECLARE_CONV1D_DML_IMPL(T)                                             \
  template void Conv1D::compute<Device::DirectML, T>(                          \
      const StorageView& input, const StorageView& weight,                     \
      const StorageView* bias, StorageView& output, const StorageView* qscale) \
      const;

DECLARE_CONV1D_DML_IMPL(float)
DECLARE_CONV1D_DML_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
