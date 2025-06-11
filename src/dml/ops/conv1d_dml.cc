#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"

#include "ctranslate2/storage_view.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale) {
    dml::Device* dml_device = dml::get_device();
    DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
    if (output.dtype() == DataType::FLOAT16) {
      execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }

    // 1. Dynamically quantize the input tensor.
    StorageView q_input(weight.dtype(), input.device());
    q_input.resize(input.shape());
    Shape scale_shape(input.rank(), 1);
    StorageView q_input_scale(scale_shape, DataType::FLOAT32, input.device());
    Shape zeropoint_shape(input.rank(), 1);
    StorageView q_input_zeropoint(zeropoint_shape, weight.dtype(),
                                  input.device());

    dml::utils::DmlTensorDescBundle dq_input_desc(input);
    dml::utils::DmlTensorDescBundle dq_output_desc(q_input);
    dml::utils::DmlTensorDescBundle dq_output_scale_desc(q_input_scale);
    auto dml_zp_dtype =
        dml::utils::get_dml_data_type(q_input_zeropoint.dtype());
    auto dml_zp_sizes = dml::utils::to_dml_dims(q_input_zeropoint.shape(),
                                                q_input_zeropoint.size());
    dml::utils::DmlTensorDescBundle zeropoint_desc(dml_zp_dtype, dml_zp_sizes,
                                                   nullptr, 0);

    DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC dq_op_desc{};
    dq_op_desc.InputTensor = &dq_input_desc.get_tensor_desc();
    dq_op_desc.OutputTensor = &dq_output_desc.get_tensor_desc();
    dq_op_desc.OutputScaleTensor = &dq_output_scale_desc.get_tensor_desc();
    dq_op_desc.OutputZeroPointTensor = &zeropoint_desc.get_tensor_desc();

    DML_OPERATOR_DESC dml_dq_op_desc_wrapper{};
    dml_dq_op_desc_wrapper.Type = DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR;
    dml_dq_op_desc_wrapper.Desc = &dq_op_desc;
    dml::Operator* dml_dq_operator = dml::GetOrCreateCompiledOperatorApi(
        &dml_dq_op_desc_wrapper, execution_flags);

    dml::utils::DmlBindingArrayBundle dq_input_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(input))});
    dml::utils::DmlBindingArrayBundle dq_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(q_input)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(q_input_scale)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(q_input_zeropoint))});
    dml_dq_operator->Execute(dq_input_bindings.get_descs(),
                             dq_output_bindings.get_descs());

    // 2. Perform Integer Convolution. Output will be INT32.
    const dim_t batch_size = input.dim(0);
    const dim_t input_length = input.dim(2);
    const dim_t out_channels = weight.dim(0);
    const dim_t kernel_size = weight.dim(2);
    const dim_t output_length =
        (input_length + (2 * _padding) - (_dilation * (kernel_size - 1) + 1)) /
            _stride +
        1;

    StorageView conv_output_int32(DataType::INT32, output.device());
    conv_output_int32.resize({batch_size, out_channels, output_length});

    std::vector<UINT> q_input_dims = {
        (UINT)q_input.dim(0), (UINT)q_input.dim(1), 1, (UINT)q_input.dim(2)};
    dml::utils::DmlTensorDescBundle q_input_desc_bundle(
        q_input.dtype(), q_input_dims, nullptr, q_input.reserved_memory());

    std::vector<UINT> weight_dims = {(UINT)weight.dim(0), (UINT)weight.dim(1),
                                     1, (UINT)weight.dim(2)};
    dml::utils::DmlTensorDescBundle weight_desc_bundle(
        weight.dtype(), weight_dims, nullptr, weight.reserved_memory());

    std::vector<UINT> conv_output_dims = {(UINT)conv_output_int32.dim(0),
                                          (UINT)conv_output_int32.dim(1), 1,
                                          (UINT)conv_output_int32.dim(2)};
    dml::utils::DmlTensorDescBundle conv_output_int32_desc_bundle(
        conv_output_int32.dtype(), conv_output_dims, nullptr,
        conv_output_int32.reserved_memory());

    DML_CONVOLUTION_INTEGER_OPERATOR_DESC conv_op_desc{};
    conv_op_desc.InputTensor = &q_input_desc_bundle.get_tensor_desc();
    conv_op_desc.InputZeroPointTensor = &zeropoint_desc.get_tensor_desc();
    conv_op_desc.FilterTensor = &weight_desc_bundle.get_tensor_desc();
    conv_op_desc.FilterZeroPointTensor = nullptr;
    conv_op_desc.OutputTensor =
        &conv_output_int32_desc_bundle.get_tensor_desc();
    conv_op_desc.DimensionCount = 2;
    UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
    conv_op_desc.Strides = dml_strides;
    UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
    conv_op_desc.Dilations = dml_dilations;
    UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
    conv_op_desc.StartPadding = dml_start_padding;
    UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
    conv_op_desc.EndPadding = dml_end_padding;
    conv_op_desc.GroupCount = static_cast<UINT>(_groups);

    DML_OPERATOR_DESC dml_conv_op_wrapper{};
    dml_conv_op_wrapper.Type = DML_OPERATOR_CONVOLUTION_INTEGER;
    dml_conv_op_wrapper.Desc = &conv_op_desc;
    dml::Operator* dml_conv_operator = dml::GetOrCreateCompiledOperatorApi(
        &dml_conv_op_wrapper, execution_flags);
    dml::utils::DmlBindingArrayBundle conv_input_bindings({
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(q_input)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(q_input_zeropoint)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(weight)),
        {}  // FilterZeroPointTensor
    });
    dml::utils::DmlBindingArrayBundle conv_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(conv_output_int32))});
    dml_conv_operator->Execute(conv_input_bindings.get_descs(),
                               conv_output_bindings.get_descs());

    // 3. Dequantize the INT32 output to float.
    StorageView dequant_stage1(output.dtype(), output.device());
    dequant_stage1.resize(conv_output_int32.shape());

    // Dequant requires scale tensor to be broadcastable. The dynamic
    // quantization produces per-tensor scale, but DML op expects it to match
    // rank and be broadcastable. So tile it.
    StorageView tiled_input_scale(q_input_scale.dtype(),
                                  q_input_scale.device());
    tiled_input_scale.resize(conv_output_int32.shape());
    dml::utils::DmlTensorDescBundle tiled_input_scale_bundle(tiled_input_scale);
    std::vector<UINT> repeats_vec;
    for (dim_t i = 0; i < q_input_scale.rank(); ++i) {
      repeats_vec.push_back(conv_output_int32.dim(i) / q_input_scale.dim(i));
    }

    DML_TILE_OPERATOR_DESC tile_op_desc{};
    tile_op_desc.InputTensor = &dq_output_scale_desc.get_tensor_desc();
    tile_op_desc.OutputTensor = &tiled_input_scale_bundle.get_tensor_desc();
    tile_op_desc.RepeatsCount = static_cast<UINT>(repeats_vec.size());
    tile_op_desc.Repeats = repeats_vec.data();

    DML_OPERATOR_DESC dml_tile_op_wrapper{};
    dml_tile_op_wrapper.Type = DML_OPERATOR_TILE;
    dml_tile_op_wrapper.Desc = &tile_op_desc;
    dml::Operator* dml_tile_operator = dml::GetOrCreateCompiledOperatorApi(
        &dml_tile_op_wrapper, execution_flags);

    dml::utils::DmlBindingArrayBundle tile_input_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(q_input_scale))});
    dml::utils::DmlBindingArrayBundle tile_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(tiled_input_scale))});
    dml_tile_operator->Execute(tile_input_bindings.get_descs(),
                               tile_output_bindings.get_descs());

    std::vector<UINT> tiled_input_scale_dims_4d = {
        (UINT)tiled_input_scale.dim(0), (UINT)tiled_input_scale.dim(1), 1,
        (UINT)tiled_input_scale.dim(2)};
    dml::utils::DmlTensorDescBundle tiled_input_scale_4d_bundle(
        tiled_input_scale.dtype(), tiled_input_scale_dims_4d, nullptr,
        tiled_input_scale.reserved_memory());

    std::vector<UINT> dequant_stage1_dims_4d = {(UINT)dequant_stage1.dim(0),
                                                (UINT)dequant_stage1.dim(1), 1,
                                                (UINT)dequant_stage1.dim(2)};
    dml::utils::DmlTensorDescBundle dequant_stage1_desc_bundle(
        dequant_stage1.dtype(), dequant_stage1_dims_4d, nullptr,
        dequant_stage1.reserved_memory());
    DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequant_op_desc{};
    dequant_op_desc.InputTensor =
        &conv_output_int32_desc_bundle.get_tensor_desc();
    dequant_op_desc.ScaleTensor =
        &tiled_input_scale_4d_bundle.get_tensor_desc();
    dequant_op_desc.ZeroPointTensor = nullptr;
    dequant_op_desc.OutputTensor =
        &dequant_stage1_desc_bundle.get_tensor_desc();

    DML_OPERATOR_DESC dml_dequant_op_wrapper{};
    dml_dequant_op_wrapper.Type = DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR;
    dml_dequant_op_wrapper.Desc = &dequant_op_desc;
    dml::Operator* dml_dequant_operator = dml::GetOrCreateCompiledOperatorApi(
        &dml_dequant_op_wrapper, execution_flags);
    dml::utils::DmlBindingArrayBundle dequant_input_bindings({
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(conv_output_int32)),
        dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(tiled_input_scale)),
        {}  // ZeroPointTensor
    });
    dml::utils::DmlBindingArrayBundle dequant_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(dequant_stage1))});
    dml_dequant_operator->Execute(dequant_input_bindings.get_descs(),
                                  dequant_output_bindings.get_descs());

    StorageView device_qscale = qscale->to(output.device());
    if (device_qscale.rank() == 1) {
      // DML's TILE operator requires the input tensor to have the same rank
      // as the output tensor, but quantization scaling factors are often
      // per-channel (1D). So, we must reshape it to be broadcastable.
      device_qscale.reshape({1, device_qscale.dim(0), 1});
    }
    dml::utils::DmlTensorDescBundle device_qscale_desc_bundle(device_qscale);
    // Also tile the filter scale to be broadcastable with the dequantized
    // output.
    StorageView tiled_filter_scale(device_qscale.dtype(),
                                   device_qscale.device());
    tiled_filter_scale.resize(dequant_stage1.shape());
    dml::utils::DmlTensorDescBundle tiled_filter_scale_bundle(
        tiled_filter_scale);
    std::vector<UINT> filter_repeats_vec = {
        static_cast<UINT>(dequant_stage1.dim(0)), 1,
        static_cast<UINT>(dequant_stage1.dim(2))};

    DML_TILE_OPERATOR_DESC filter_tile_op_desc{};
    filter_tile_op_desc.InputTensor =
        &device_qscale_desc_bundle.get_tensor_desc();
    filter_tile_op_desc.OutputTensor =
        &tiled_filter_scale_bundle.get_tensor_desc();
    filter_tile_op_desc.RepeatsCount =
        static_cast<UINT>(filter_repeats_vec.size());
    filter_tile_op_desc.Repeats = filter_repeats_vec.data();

    DML_OPERATOR_DESC dml_filter_tile_op_wrapper{};
    dml_filter_tile_op_wrapper.Type = DML_OPERATOR_TILE;
    dml_filter_tile_op_wrapper.Desc = &filter_tile_op_desc;
    dml::Operator* dml_filter_tile_operator =
        dml::GetOrCreateCompiledOperatorApi(&dml_filter_tile_op_wrapper,
                                            execution_flags);
    dml::utils::DmlBindingArrayBundle filter_tile_input_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(device_qscale))});
    dml::utils::DmlBindingArrayBundle filter_tile_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(tiled_filter_scale))});
    dml_filter_tile_operator->Execute(filter_tile_input_bindings.get_descs(),
                                      filter_tile_output_bindings.get_descs());

    output.resize(dequant_stage1.shape());

    std::vector<UINT> output_dims_4d = {
        (UINT)output.dim(0), (UINT)output.dim(1), 1, (UINT)output.dim(2)};
    dml::utils::DmlTensorDescBundle output_desc_bundle(
        output.dtype(), output_dims_4d, nullptr, output.reserved_memory());

    std::vector<UINT> tiled_filter_scale_dims_4d = {
        (UINT)tiled_filter_scale.dim(0), (UINT)tiled_filter_scale.dim(1), 1,
        (UINT)tiled_filter_scale.dim(2)};
    dml::utils::DmlTensorDescBundle tiled_filter_scale_4d_bundle(
        tiled_filter_scale.dtype(), tiled_filter_scale_dims_4d, nullptr,
        tiled_filter_scale.reserved_memory());

    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_op_desc{};
    mul_op_desc.ATensor = &dequant_stage1_desc_bundle.get_tensor_desc();
    mul_op_desc.BTensor = &tiled_filter_scale_4d_bundle.get_tensor_desc();
    mul_op_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();

    DML_OPERATOR_DESC dml_mul_op_wrapper{};
    dml_mul_op_wrapper.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
    dml_mul_op_wrapper.Desc = &mul_op_desc;
    dml::Operator* dml_mul_operator = dml::GetOrCreateCompiledOperatorApi(
        &dml_mul_op_wrapper, execution_flags);
    dml::utils::DmlBindingArrayBundle mul_input_bindings(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(dequant_stage1)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(tiled_filter_scale))});
    dml::utils::DmlBindingArrayBundle mul_output_bindings(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    dml_mul_operator->Execute(mul_input_bindings.get_descs(),
                              mul_output_bindings.get_descs());

    if (bias) {
      // Tile bias to match output shape for element-wise add
      StorageView device_bias = bias->to(output.device());
      if (device_bias.rank() == 1) {
        device_bias.reshape({1, device_bias.dim(0), 1});
      }

      StorageView tiled_bias(device_bias.dtype(), device_bias.device());
      tiled_bias.resize(output.shape());

      dml::utils::DmlTensorDescBundle device_bias_desc_bundle(device_bias);
      dml::utils::DmlTensorDescBundle tiled_bias_desc_bundle(tiled_bias);

      std::vector<UINT> bias_repeats_vec = {static_cast<UINT>(output.dim(0)), 1,
                                            static_cast<UINT>(output.dim(2))};

      DML_TILE_OPERATOR_DESC bias_tile_op_desc{};
      bias_tile_op_desc.InputTensor =
          &device_bias_desc_bundle.get_tensor_desc();
      bias_tile_op_desc.OutputTensor =
          &tiled_bias_desc_bundle.get_tensor_desc();
      bias_tile_op_desc.RepeatsCount =
          static_cast<UINT>(bias_repeats_vec.size());
      bias_tile_op_desc.Repeats = bias_repeats_vec.data();

      DML_OPERATOR_DESC dml_bias_tile_op_wrapper{};
      dml_bias_tile_op_wrapper.Type = DML_OPERATOR_TILE;
      dml_bias_tile_op_wrapper.Desc = &bias_tile_op_desc;
      dml::Operator* dml_bias_tile_operator =
          dml::GetOrCreateCompiledOperatorApi(&dml_bias_tile_op_wrapper,
                                              execution_flags);

      dml::utils::DmlBindingArrayBundle bias_tile_input_bindings(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(device_bias))});
      dml::utils::DmlBindingArrayBundle bias_tile_output_bindings(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(tiled_bias))});

      dml_bias_tile_operator->Execute(bias_tile_input_bindings.get_descs(),
                                      bias_tile_output_bindings.get_descs());
      std::vector<UINT> tiled_bias_dims_4d = {(UINT)tiled_bias.dim(0),
                                              (UINT)tiled_bias.dim(1), 1,
                                              (UINT)tiled_bias.dim(2)};
      dml::utils::DmlTensorDescBundle bias_desc_bundle(
          tiled_bias.dtype(), tiled_bias_dims_4d, nullptr,
          tiled_bias.reserved_memory());

      DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_op_desc{};
      add_op_desc.ATensor = &output_desc_bundle.get_tensor_desc();
      add_op_desc.BTensor = &bias_desc_bundle.get_tensor_desc();
      add_op_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();

      DML_OPERATOR_DESC dml_add_op_desc_wrapper{};
      dml_add_op_desc_wrapper.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
      dml_add_op_desc_wrapper.Desc = &add_op_desc;
      dml::Operator* dml_add_operator = dml::GetOrCreateCompiledOperatorApi(
          &dml_add_op_desc_wrapper, execution_flags);
      dml::utils::DmlBindingArrayBundle add_input_bindings(
          {dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(output)),
           dml::utils::DmlBufferBindingBundle(
               dml::utils::ResourceFromStorageView(tiled_bias))});
      dml::utils::DmlBindingArrayBundle add_output_bindings(
          {dml::utils::DmlBufferBindingBundle(
              dml::utils::ResourceFromStorageView(output))});
      dml_add_operator->Execute(add_input_bindings.get_descs(),
                                add_output_bindings.get_descs());
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
