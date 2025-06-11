#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"

#include "ctranslate2/storage_view.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

#include <numeric>

namespace ctranslate2 {
namespace ops {

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

    // 1. Quantize Input Tensor (Float -> INT8) - SYMMETRIC
    // This is a multi-step process to emulate the CPU's symmetric
    // quantization (zero-point = 0).
    // The scale is computed as max(abs(input)) / 127.
    // The quantized input is round(input/scale).
    // NOTE: This uses a global scale for the whole input tensor, which still
    // differs from the CPU's per-convolution-patch scale. However, it
    // corrects the zero-point issue which is a major source of divergence.
    StorageView input_q8({input.shape()}, int8_t(0), input.device());
    StorageView input_scale({1}, 0.0f, input.device());

    // 1a. input_abs = abs(input)
    StorageView input_abs(input.shape(), input.dtype(), input.device());
    {
      dml::utils::DmlTensorDescBundle input_desc(input);
      dml::utils::DmlTensorDescBundle input_abs_desc(input_abs);
      DML_ELEMENT_WISE_ABS_OPERATOR_DESC op_desc{
          &input_desc.get_tensor_desc(), &input_abs_desc.get_tensor_desc(),
          nullptr};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_ELEMENT_WISE_ABS,
                                            &op_desc};
      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_abs))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 1b. max_val = reduce_max(input_abs)
    StorageView max_val({1}, input.dtype(), input.device());
    {
      dml::utils::DmlTensorDescBundle input_abs_desc(input_abs);
      // The output of REDUCE must have the same rank as the input. We use the
      // advanced DmlTensorDescBundle constructor to create a view of `max_val`
      // that has the same rank as `input_abs` but with all dimensions set to 1.
      std::vector<UINT> reduce_output_dims(input_abs.rank(), 1);
      const auto physical_dims =
          dml::utils::to_dml_dims(max_val.shape(), max_val.size());
      dml::utils::DmlTensorDescBundle max_val_desc(
          dml::utils::get_dml_data_type(max_val.dtype()), reduce_output_dims,
          physical_dims, static_cast<int32_t>(reduce_output_dims.size()), 0, 0,
          static_cast<uint32_t>(reduce_output_dims.size()), 0);
      std::vector<UINT> axes(input.rank());
      std::iota(axes.begin(), axes.end(), 0);
      DML_REDUCE_OPERATOR_DESC op_desc{
          DML_REDUCE_FUNCTION_MAX, &input_abs_desc.get_tensor_desc(),
          &max_val_desc.get_tensor_desc(), static_cast<UINT>(axes.size()),
          axes.data()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_REDUCE, &op_desc};
      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_abs))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(max_val))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 1c. input_scale = max_val / 127.0
    StorageView divisor_cpu({1}, 127.f);
    StorageView divisor(divisor_cpu.shape(), divisor_cpu.dtype(),
                        input.device());
    divisor.copy_from(divisor_cpu);
    {
      dml::utils::DmlTensorDescBundle max_val_desc(max_val);
      dml::utils::DmlTensorDescBundle divisor_desc(divisor);
      dml::utils::DmlTensorDescBundle input_scale_desc(input_scale);
      DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC op_desc{
          &max_val_desc.get_tensor_desc(), &divisor_desc.get_tensor_desc(),
          &input_scale_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_ELEMENT_WISE_DIVIDE,
                                            &op_desc};
      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(max_val)),
           dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(divisor))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_scale))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 1d. temp_float = round(input / input_scale)
    StorageView temp_float(input.shape(), input.dtype(), input.device());
    StorageView temp_float_rounded(input.shape(), input.dtype(),
                                   input.device());
    {
      dml::utils::DmlTensorDescBundle input_desc(input);
      const auto target_dims =
          dml::utils::to_dml_dims(input.shape(), input.size());
      const auto physical_dims =
          dml::utils::to_dml_dims(input_scale.shape(), input_scale.size());
      dml::utils::DmlTensorDescBundle input_scale_desc_bcast(
          dml::utils::get_dml_data_type(input_scale.dtype()), target_dims,
          physical_dims, (int32_t)target_dims.size(), 0, 0,
          (uint32_t)target_dims.size(), 0);
      dml::utils::DmlTensorDescBundle temp_float_desc(temp_float);
      DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC div_op_desc{
          &input_desc.get_tensor_desc(),
          &input_scale_desc_bcast.get_tensor_desc(),
          &temp_float_desc.get_tensor_desc()};
      DML_OPERATOR_DESC div_dml_op_desc_wrapper{
          DML_OPERATOR_ELEMENT_WISE_DIVIDE, &div_op_desc};
      dml::Operator* div_dml_op = dml::GetOrCreateCompiledOperatorApi(
          &div_dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle div_inputs(
          {dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(input)),
           dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(input_scale))});
      dml::utils::DmlBindingArrayBundle div_outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(temp_float))});
      div_dml_op->Execute(div_inputs.get_descs(), div_outputs.get_descs());

      dml::utils::DmlTensorDescBundle temp_float_in_desc(temp_float);
      dml::utils::DmlTensorDescBundle temp_float_rounded_desc(
          temp_float_rounded);
      DML_ELEMENT_WISE_ROUND_OPERATOR_DESC round_op_desc{
          &temp_float_in_desc.get_tensor_desc(),
          &temp_float_rounded_desc.get_tensor_desc(),
          DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN};
      DML_OPERATOR_DESC round_dml_op_desc_wrapper{
          DML_OPERATOR_ELEMENT_WISE_ROUND, &round_op_desc};
      dml::Operator* round_dml_op = dml::GetOrCreateCompiledOperatorApi(
          &round_dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle round_inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(temp_float))});
      dml::utils::DmlBindingArrayBundle round_outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(temp_float_rounded))});
      round_dml_op->Execute(round_inputs.get_descs(),
                            round_outputs.get_descs());
    }

    // 1e. input_q8 = cast<int8>(temp_float_rounded)
    {
      dml::utils::DmlTensorDescBundle temp_float_rounded_desc(
          temp_float_rounded);
      dml::utils::DmlTensorDescBundle input_q8_desc(input_q8);
      DML_CAST_OPERATOR_DESC op_desc{&temp_float_rounded_desc.get_tensor_desc(),
                                     &input_q8_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_CAST, &op_desc};
      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
      dml::utils::DmlBindingArrayBundle inputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(temp_float_rounded))});
      dml::utils::DmlBindingArrayBundle outputs(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_q8))});
      dml_op->Execute(inputs.get_descs(), outputs.get_descs());
    }

    // 2. Integer Convolution
    StorageView output_q32(output.shape(), int32_t(0), output.device());
    StorageView weight_zero_point({1}, int8_t(0), weight.device());
    {
      std::vector<UINT> input_dims =
          dml::utils::get_dml_tensor_shape_4d(input_q8);
      dml::utils::DmlTensorDescBundle input_q8_desc(
          input_q8.dtype(), input_dims, nullptr, input_q8.reserved_memory());

      std::vector<UINT> weight_dims =
          dml::utils::get_dml_tensor_shape_4d(weight, true);
      dml::utils::DmlTensorDescBundle weight_desc(
          weight.dtype(), weight_dims, nullptr, weight.reserved_memory());
      dml::utils::DmlTensorDescBundle weight_zp_desc(weight_zero_point);

      std::vector<UINT> output_dims =
          dml::utils::get_dml_tensor_shape_4d(output_q32);
      dml::utils::DmlTensorDescBundle output_q32_desc(
          output_q32.dtype(), output_dims, nullptr,
          output_q32.reserved_memory());

      DML_CONVOLUTION_INTEGER_OPERATOR_DESC op_desc{};
      op_desc.InputTensor = &input_q8_desc.get_tensor_desc();
      op_desc.InputZeroPointTensor = nullptr;  // ZP is 0.
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
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_CONVOLUTION_INTEGER,
                                            &op_desc};

      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE,
          L"Conv1D::ConvolutionInteger");
      dml::utils::DmlBindingArrayBundle inputs({
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(input_q8)),
          nullptr,
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(weight)),
          dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(weight_zero_point)),
      });
      dml::utils::DmlBindingArrayBundle outputs(
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
      dml::utils::DmlTensorDescBundle in_desc(output_q32);
      dml::utils::DmlTensorDescBundle out_desc(tmp_float);
      DML_CAST_OPERATOR_DESC op_desc{&in_desc.get_tensor_desc(),
                                     &out_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_CAST, &op_desc};
      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
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
      dml::utils::DmlTensorDescBundle a_desc(tmp_float);

      const auto target_dims =
          dml::utils::to_dml_dims(tmp_float.shape(), tmp_float.size());
      const auto physical_dims =
          dml::utils::to_dml_dims(input_scale.shape(), input_scale.size());
      dml::utils::DmlTensorDescBundle b_desc(
          dml::utils::get_dml_data_type(input_scale.dtype()), target_dims,
          physical_dims, (int32_t)target_dims.size(), 0, 0,
          (uint32_t)target_dims.size(), 0);

      dml::utils::DmlTensorDescBundle out_desc(tmp_float2);
      DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC op_desc{
          &a_desc.get_tensor_desc(), &b_desc.get_tensor_desc(),
          &out_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                            &op_desc};

      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
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

    // 3c. Multiply by weight_scale (*qscale)
    {
      dml::utils::DmlTensorDescBundle a_desc(tmp_float2);
      const auto target_dims =
          dml::utils::to_dml_dims(tmp_float2.shape(), tmp_float2.size());
      const std::vector<UINT> physical_dims = {
          static_cast<UINT>(qscale->size()), 1};
      dml::utils::DmlTensorDescBundle b_desc(
          dml::utils::get_dml_data_type(qscale->dtype()), target_dims,
          physical_dims, (int32_t)target_dims.size(), 0, 0,
          (uint32_t)target_dims.size(), 0);
      dml::utils::DmlTensorDescBundle out_desc(final_float_output);
      DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC op_desc{
          &a_desc.get_tensor_desc(), &b_desc.get_tensor_desc(),
          &out_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                            &op_desc};

      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
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
      dml::utils::DmlTensorDescBundle a_desc(final_float_output);
      const auto target_dims = dml::utils::to_dml_dims(
          final_float_output.shape(), final_float_output.size());
      const std::vector<UINT> physical_dims = {static_cast<UINT>(bias->size()),
                                               1};
      dml::utils::DmlTensorDescBundle b_desc(
          dml::utils::get_dml_data_type(bias->dtype()), target_dims,
          physical_dims, (int32_t)target_dims.size(), 0, 0,
          (uint32_t)target_dims.size(), 0);
      dml::utils::DmlTensorDescBundle out_desc(output);

      DML_ELEMENT_WISE_ADD_OPERATOR_DESC op_desc{&a_desc.get_tensor_desc(),
                                                 &b_desc.get_tensor_desc(),
                                                 &out_desc.get_tensor_desc()};
      DML_OPERATOR_DESC dml_op_desc_wrapper{DML_OPERATOR_ELEMENT_WISE_ADD,
                                            &op_desc};

      dml::Operator* dml_op = dml::GetOrCreateCompiledOperatorApi(
          &dml_op_desc_wrapper, DML_EXECUTION_FLAG_NONE);
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
    std::vector<UINT> input_dml_dims_vec =
        dml::utils::get_dml_tensor_shape_4d(input);
    dml::utils::DmlTensorDescBundle input_desc_bundle(
        input.dtype(), input_dml_dims_vec, nullptr, input.reserved_memory());

    std::vector<UINT> weight_dml_dims_vec =
        dml::utils::get_dml_tensor_shape_4d(weight, true);
    dml::utils::DmlTensorDescBundle weight_desc_bundle(
        weight.dtype(), weight_dml_dims_vec, nullptr, weight.reserved_memory());

    std::vector<UINT> output_dml_dims_vec =
        dml::utils::get_dml_tensor_shape_4d(output);
    dml::utils::DmlTensorDescBundle output_desc_bundle(
        output.dtype(), output_dml_dims_vec, nullptr, output.reserved_memory());

    std::unique_ptr<dml::utils::DmlTensorDescBundle> bias_desc_bundle_ptr;
    if (bias) {
      std::vector<UINT> bias_dml_dims_vec =
          dml::utils::get_dml_tensor_shape_4d(*bias, true);
      bias_desc_bundle_ptr = std::make_unique<dml::utils::DmlTensorDescBundle>(
          bias->dtype(), bias_dml_dims_vec, nullptr, bias->reserved_memory());
    }

    // 2. Create Convolution Operator Descriptor
    DML_CONVOLUTION_OPERATOR_DESC conv_op_desc{};
    conv_op_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
    conv_op_desc.FilterTensor = &weight_desc_bundle.get_tensor_desc();
    conv_op_desc.BiasTensor =
        bias ? &bias_desc_bundle_ptr->get_tensor_desc() : nullptr;
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

    DML_OPERATOR_DESC dml_op_desc_wrapper{};
    dml_op_desc_wrapper.Type = DML_OPERATOR_CONVOLUTION;
    dml_op_desc_wrapper.Desc = &conv_op_desc;

    // 3. Get or Create Compiled Operator from Cache
    DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
    if (dml_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
      execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }

    dml::Operator* dml_convolution_operator =
        dml::GetOrCreateCompiledOperatorApi(&dml_op_desc_wrapper,
                                            execution_flags);

    // 4. Prepare Bindings and Execute
    std::vector<dml::utils::DmlBufferBindingBundle> input_bundles;
    input_bundles.emplace_back(dml::utils::ResourceFromStorageView(input));
    input_bundles.emplace_back(dml::utils::ResourceFromStorageView(weight));
    if (bias) {
      input_bundles.emplace_back(dml::utils::ResourceFromStorageView(*bias));
    } else {
      input_bundles
          .emplace_back();  // Add a "none" binding for the optional bias.
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
