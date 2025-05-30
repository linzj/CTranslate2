#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/dequantize.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <>
void Dequantize::dequantize<Device::DirectML, int8_t, float>(
    const StorageView& input,
    const StorageView& scale,
    StorageView& output) const {
  auto device = dml::get_device();
  auto dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);

  // Step 1: Cast int8 input to float
  StorageView float_input(input.shape(), DataType::FLOAT32, Device::DirectML);

  // Create cast operation
  dml::utils::DmlTensorDescBundle input_desc_bundle(input);  // INT8
  // float_input is StorageView(input.shape(), DataType::FLOAT32,
  // Device::DirectML)
  dml::utils::DmlTensorDescBundle float_cast_output_desc_bundle(
      float_input);  // FLOAT32

  DML_CAST_OPERATOR_DESC cast_desc = {
      .InputTensor = &input_desc_bundle.get_tensor_desc(),
      .OutputTensor = &float_cast_output_desc_bundle.get_tensor_desc()};

  DML_OPERATOR_DESC cast_op_desc = {.Type = DML_OPERATOR_CAST,
                                    .Desc = &cast_desc};

  auto cast_compiled_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc);

  // Bind cast inputs and outputs
  // Bind cast inputs and outputs
  DML_BUFFER_BINDING cast_input_buffer_binding =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
          0, input.size() * sizeof(int8_t));
  DML_BINDING_DESC cast_input_binding_desc =
      dml::utils::create_binding_desc(&cast_input_buffer_binding);

  DML_BUFFER_BINDING cast_output_buffer_binding =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(float_input.buffer()), 0,
          float_input.size() * sizeof(float));
  DML_BINDING_DESC cast_output_binding_desc =
      dml::utils::create_binding_desc(&cast_output_buffer_binding);

  cast_compiled_op->Execute({cast_input_binding_desc},
                            {cast_output_binding_desc});

  // Step 2: Element-wise division (float_input / scale)

  // Handle scale broadcasting - scale is typically per-channel (depth
  // dimension)
  // dimension)
  // The float_cast_output_desc_bundle uses input_sizes for its .Sizes member
  std::vector<UINT> dml_input_shape_vec =
      dml::utils::to_dml_dims(input.shape(), input.size());
  std::vector<UINT> scale_sizes_vec;
  std::vector<UINT> scale_strides_vec;  // Keep this alive if used
  const std::vector<UINT>* scale_strides_ptr = nullptr;

  if (scale.size() == depth &&
      input.rank() > 0) {  // Ensure input is not scalar
    // Scale has same number of elements as the last dimension of input
    // Broadcast scale to match input shape
    scale_sizes_vec = dml_input_shape_vec;  // Target shape for broadcasting
    scale_strides_vec.resize(
        input.rank(), 0);  // Initialize all strides to 0 for broadcasting

    if (!scale_strides_vec
             .empty()) {             // Should not be empty if input.rank() > 0
      scale_strides_vec.back() = 1;  // Normal stride for the last dimension
                                     // that matches scale's actual data
    }
    scale_strides_ptr = &scale_strides_vec;
  } else {  // Scale has same shape as input or is scalar and will broadcast
    scale_sizes_vec = dml::utils::to_dml_dims(scale.shape(), scale.size());
    // if scale_strides_vec is empty (nullptr), DML implies contiguous based on
    // scale_sizes_vec. If scale is scalar, to_dml_dims makes it {1}, DML
    // broadcasts correctly.
  }

  // Important: DmlTensorDescBundle for scale needs to know the *original*
  // number of elements in scale storage for TotalTensorSizeInBytes, even if its
  // Sizes/Strides are for broadcasting.
  dml::utils::DmlTensorDescBundle scale_desc_bundle(
      scale.dtype(),    // FLOAT32
      scale_sizes_vec,  // This can be the broadcasted shape
      scale_strides_ptr,
      scale.size() * sizeof(float)  // Use original scale storage size
  );

  dml::utils::DmlTensorDescBundle div_output_desc_bundle(output);  // FLOAT32

  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC divide_desc = {
      .ATensor = &float_cast_output_desc_bundle
                      .get_tensor_desc(),  // Output from previous CAST
      .BTensor = &scale_desc_bundle.get_tensor_desc(),
      .OutputTensor = &div_output_desc_bundle.get_tensor_desc()};

  DML_OPERATOR_DESC divide_op_desc = {.Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE,
                                      .Desc = &divide_desc};

  auto divide_compiled_op =
      dml::GetOrCreateCompiledOperatorApi(&divide_op_desc);

  // Bind division inputs and outputs
  // Bind division inputs and outputs
  DML_BUFFER_BINDING div_input1_buffer_binding =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(float_input.buffer()), 0,
          float_input.size() * sizeof(float));

  DML_BUFFER_BINDING div_input2_buffer_binding =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(scale.buffer())),
          0, scale.size() * sizeof(float));

  DML_BUFFER_BINDING div_output_buffer_binding =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output.size() * sizeof(float));

  std::vector<DML_BINDING_DESC> div_input_bindings_vec_desc = {
      dml::utils::create_binding_desc(&div_input1_buffer_binding),
      dml::utils::create_binding_desc(&div_input2_buffer_binding)};

  std::vector<DML_BINDING_DESC> div_output_bindings_vec_desc = {
      dml::utils::create_binding_desc(&div_output_buffer_binding)};

  divide_compiled_op->Execute(div_input_bindings_vec_desc,
                              div_output_bindings_vec_desc);
}

template <>
void Dequantize::dequantize_gemm_output<Device::DirectML, float>(
    const StorageView& c,
    const StorageView& a_scale,
    const StorageView& b_scale,
    const bool transpose_a,
    const bool transpose_b,
    const StorageView* bias,
    StorageView& y) const {
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();
  auto* command_list = device->GetCommandList();

  const dim_t batch_size = a_scale.size();
  const dim_t depth = c.dim(-1);

  // Get input buffers
  auto* c_buffer = static_cast<ID3D12Resource*>(const_cast<void*>(c.buffer()));
  auto* a_scale_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(a_scale.buffer()));
  auto* b_scale_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(b_scale.buffer()));
  auto* y_buffer = static_cast<ID3D12Resource*>(y.buffer());

  // Create tensor descriptors
  // Note: DmlTensorDescBundle is better here to manage lifetimes of underlying
  // dimension/stride vectors
  dml::utils::DmlTensorDescBundle c_desc_bundle(c);              // INT32
  dml::utils::DmlTensorDescBundle a_scale_desc_bundle(a_scale);  // FLOAT32
  dml::utils::DmlTensorDescBundle b_scale_desc_bundle(b_scale);  // FLOAT32
  dml::utils::DmlTensorDescBundle y_desc_bundle(y);              // FLOAT32

  // For intermediate tensor used in cast/multiply/divide ops, needs to match
  // y_desc_bundle's properties
  dml::utils::DmlTensorDescBundle intermediate_desc_bundle(
      y.dtype(), y_desc_bundle.get_sizes_vec(), nullptr,
      y.size() * y.item_size());

  // Step 1: Cast int32 input to float
  DML_CAST_OPERATOR_DESC cast_desc_gemm = {};  // Renamed to avoid conflict
  cast_desc_gemm.InputTensor = &c_desc_bundle.get_tensor_desc();
  cast_desc_gemm.OutputTensor = &intermediate_desc_bundle.get_tensor_desc();

  DML_OPERATOR_DESC cast_op_desc = {};
  DML_OPERATOR_DESC cast_op_desc_gemm = {};  // Renamed
  cast_op_desc_gemm.Type = DML_OPERATOR_CAST;
  cast_op_desc_gemm.Desc = &cast_desc_gemm;

  auto cast_op = dml::GetOrCreateCompiledOperatorApi(&cast_op_desc_gemm);

  // Create temporary buffer for cast output
  auto cast_output =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 2: Multiply scales (a_scale * b_scale)
  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC scale_mult_desc = {};
  scale_mult_desc.ATensor = &a_scale_desc_bundle.get_tensor_desc();
  scale_mult_desc.BTensor = &b_scale_desc_bundle.get_tensor_desc();
  scale_mult_desc.OutputTensor = &intermediate_desc_bundle.get_tensor_desc();

  DML_OPERATOR_DESC scale_mult_op_desc = {};
  scale_mult_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
  scale_mult_op_desc.Desc = &scale_mult_desc;

  auto scale_mult_op = dml::GetOrCreateCompiledOperatorApi(&scale_mult_op_desc);

  // Create temporary buffer for scale multiplication
  auto combined_scale =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 3: Divide cast output by combined scale
  DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC divide_desc = {};
  divide_desc.ATensor =
      &intermediate_desc_bundle
           .get_tensor_desc();  // Output of previous cast becomes input here
                                // (if cast to temp) or result of scale mult (if
                                // scales multiplied first) Sequence of ops
                                // matters. Assuming Cast(c) -> Temp1
                                // Multiply(a_scale, b_scale) -> Temp2
                                // (CombinedScale) Divide(Temp1, Temp2) -> Temp3
                                // (or y if no bias/act)
  divide_desc.BTensor =
      &intermediate_desc_bundle
           .get_tensor_desc();  // This should be combined_scale tensor
  divide_desc.OutputTensor =
      &intermediate_desc_bundle.get_tensor_desc();  // Output of division

  DML_OPERATOR_DESC divide_op_desc = {};
  divide_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_DIVIDE;
  divide_op_desc.Desc = &divide_desc;

  auto divide_op = dml::GetOrCreateCompiledOperatorApi(&divide_op_desc);

  // Create temporary buffer for division result
  auto divide_output =
      device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));

  // Step 4: Add bias if provided
  dml::Operator* bias_add_op = nullptr;
  Microsoft::WRL::ComPtr<ID3D12Resource> bias_output;

  if (bias) {
    auto* bias_buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer()));

    // Bias tensor
    // Create DmlTensorDescBundle for bias, ensuring it can broadcast if its
    // rank is less than intermediate's
    std::vector<UINT> bias_dml_dims =
        dml::utils::to_dml_dims(bias->shape(), bias->size());
    if (bias_dml_dims.size() <
        intermediate_desc_bundle.get_sizes_vec().size()) {
      // Simplified broadcasting setup assuming bias is (depth) and intermediate
      // is (batch, depth) This matches the common case. A more general
      // broadcasting setup might be complex. For DML ADD, tensors must have
      // same DimensionCount or one of them is scalar-like and broadcast. If
      // bias is [depth], intermediate is [batch, depth], make bias [1, depth]
      if (bias_dml_dims.size() == 1 &&
          intermediate_desc_bundle.get_sizes_vec().size() == 2) {
        bias_dml_dims.insert(bias_dml_dims.begin(), 1);
      }
    }
    // Pass nullptr for strides to let DmlTensorDescBundle compute contiguous
    // ones or use existing ones Ensure total size in bytes uses the original
    // bias storage.
    dml::utils::DmlTensorDescBundle bias_desc_bundle(*bias, nullptr);

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC bias_add_desc = {};
    bias_add_desc.ATensor =
        &intermediate_desc_bundle
             .get_tensor_desc();  // Result from previous division
    bias_add_desc.BTensor = &bias_desc_bundle.get_tensor_desc();
    bias_add_desc.OutputTensor =
        &intermediate_desc_bundle
             .get_tensor_desc();  // Output of Add (in-place on intermediate or
                                  // to a new buffer if needed before
                                  // activation)

    DML_OPERATOR_DESC bias_add_op_desc = {};
    bias_add_op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
    bias_add_op_desc.Desc = &bias_add_desc;

    bias_add_op = dml::GetOrCreateCompiledOperatorApi(&bias_add_op_desc);
    bias_output =
        device->CreatePreferredDeviceMemoryBuffer(y.size() * sizeof(float));
  }

  // Step 5: Apply activation if specified
  dml::Operator* activation_op = nullptr;

  if (_activation_type) {
    DML_OPERATOR_DESC activation_op_desc = {};

    switch (*_activation_type) {
      case ActivationType::ReLU: {
        DML_ACTIVATION_RELU_OPERATOR_DESC relu_desc = {};
        relu_desc.InputTensor =
            &intermediate_desc_bundle
                 .get_tensor_desc();  // Input from previous step
        relu_desc.OutputTensor =
            &y_desc_bundle.get_tensor_desc();  // Final output

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_RELU;
        activation_op_desc.Desc = &relu_desc;
        break;
      }
      case ActivationType::GELU: {
        DML_ACTIVATION_GELU_OPERATOR_DESC gelu_desc = {};
        gelu_desc.InputTensor = &intermediate_desc_bundle.get_tensor_desc();
        gelu_desc.OutputTensor = &y_desc_bundle.get_tensor_desc();

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
        activation_op_desc.Desc = &gelu_desc;
        break;
      }
      case ActivationType::Sigmoid: {
        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid_desc = {};
        sigmoid_desc.InputTensor = &intermediate_desc_bundle.get_tensor_desc();
        sigmoid_desc.OutputTensor = &y_desc_bundle.get_tensor_desc();

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
        activation_op_desc.Desc = &sigmoid_desc;
        break;
      }
      case ActivationType::Tanh: {
        DML_ACTIVATION_TANH_OPERATOR_DESC tanh_desc = {};
        tanh_desc.InputTensor = &intermediate_desc_bundle.get_tensor_desc();
        tanh_desc.OutputTensor = &y_desc_bundle.get_tensor_desc();

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_TANH;
        activation_op_desc.Desc = &tanh_desc;
        break;
      }
      case ActivationType::Swish: {
        DML_ACTIVATION_SWISH_OPERATOR_DESC swish_desc = {};
        swish_desc.InputTensor = &intermediate_desc_bundle.get_tensor_desc();
        swish_desc.OutputTensor = &y_desc_bundle.get_tensor_desc();
        swish_desc.SigmoidInputScale = 1.0f;

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SWISH;
        activation_op_desc.Desc = &swish_desc;
        break;
      }
      default:
        // For unsupported activations, use identity
        DML_ACTIVATION_IDENTITY_OPERATOR_DESC identity_desc = {};
        identity_desc.InputTensor = &intermediate_desc_bundle.get_tensor_desc();
        identity_desc.OutputTensor = &y_desc_bundle.get_tensor_desc();

        activation_op_desc.Type = DML_OPERATOR_ACTIVATION_IDENTITY;
        activation_op_desc.Desc = &identity_desc;
        break;
    }

    activation_op = dml::GetOrCreateCompiledOperatorApi(&activation_op_desc);
  }

  // Execute the operators in sequence

  // 1. Cast int32 to float
  {
    DML_BUFFER_BINDING c_buffer_binding_storage =
        dml::utils::create_buffer_binding(c_buffer, 0,
                                          c.size() * sizeof(int32_t));
    DML_BUFFER_BINDING cast_output_buffer_binding_storage =
        dml::utils::create_buffer_binding(cast_output.Get(), 0,
                                          y.size() * sizeof(float));

    DML_BINDING_DESC input_binding_desc_gemm =
        dml::utils::create_binding_desc(&c_buffer_binding_storage);
    DML_BINDING_DESC output_binding_desc_gemm =
        dml::utils::create_binding_desc(&cast_output_buffer_binding_storage);

    cast_op->Execute({input_binding_desc_gemm}, {output_binding_desc_gemm});
  }

  // 2. Multiply scales
  {
    DML_BUFFER_BINDING a_scale_buffer_binding_storage =
        dml::utils::create_buffer_binding(a_scale_buffer, 0,
                                          a_scale.size() * sizeof(float));
    DML_BUFFER_BINDING b_scale_buffer_binding_storage =
        dml::utils::create_buffer_binding(b_scale_buffer, 0,
                                          b_scale.size() * sizeof(float));
    DML_BUFFER_BINDING scale_output_buffer_binding_storage =
        dml::utils::create_buffer_binding(combined_scale.Get(), 0,
                                          y.size() * sizeof(float));

    std::vector<DML_BINDING_DESC> scale_input_bindings_vec_desc = {
        dml::utils::create_binding_desc(&a_scale_buffer_binding_storage),
        dml::utils::create_binding_desc(&b_scale_buffer_binding_storage)};
    std::vector<DML_BINDING_DESC> scale_output_bindings_vec_desc = {
        dml::utils::create_binding_desc(&scale_output_buffer_binding_storage)};

    scale_mult_op->Execute(scale_input_bindings_vec_desc,
                           scale_output_bindings_vec_desc);
  }

  // 3. Divide cast output by combined scale
  {
    DML_BUFFER_BINDING cast_output_buffer_binding_for_div_storage =
        dml::utils::create_buffer_binding(
            cast_output.Get(), 0,
            y.size() * sizeof(float));  // Input1 to Divide
    DML_BUFFER_BINDING combined_scale_buffer_binding_storage =
        dml::utils::create_buffer_binding(
            combined_scale.Get(), 0,
            y.size() * sizeof(float));  // Input2 to Divide
    DML_BUFFER_BINDING divide_output_buffer_binding_storage =
        dml::utils::create_buffer_binding(divide_output.Get(), 0,
                                          y.size() * sizeof(float));

    std::vector<DML_BINDING_DESC> div_op_input_bindings_vec_desc = {
        dml::utils::create_binding_desc(
            &cast_output_buffer_binding_for_div_storage),
        dml::utils::create_binding_desc(
            &combined_scale_buffer_binding_storage)};
    std::vector<DML_BINDING_DESC> div_op_output_bindings_vec_desc = {
        dml::utils::create_binding_desc(&divide_output_buffer_binding_storage)};

    divide_op->Execute(div_op_input_bindings_vec_desc,
                       div_op_output_bindings_vec_desc);
  }

  // 4. Add bias if provided
  Microsoft::WRL::ComPtr<ID3D12Resource> current_output = divide_output;
  if (bias) {
    auto* bias_buffer =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer()));

    DML_BUFFER_BINDING prev_output_buffer_binding_storage =
        dml::utils::create_buffer_binding(
            divide_output.Get(), 0, y.size() * sizeof(float));  // Input1 to Add
    DML_BUFFER_BINDING bias_value_buffer_binding_storage =
        dml::utils::create_buffer_binding(
            bias_buffer, 0, bias->size() * sizeof(float));  // Input2 to Add
    DML_BUFFER_BINDING bias_add_output_buffer_binding_storage =
        dml::utils::create_buffer_binding(bias_output.Get(), 0,
                                          y.size() * sizeof(float));

    std::vector<DML_BINDING_DESC> bias_add_input_bindings_vec_desc = {
        dml::utils::create_binding_desc(&prev_output_buffer_binding_storage),
        dml::utils::create_binding_desc(&bias_value_buffer_binding_storage)};
    std::vector<DML_BINDING_DESC> bias_add_output_bindings_vec_desc = {
        dml::utils::create_binding_desc(
            &bias_add_output_buffer_binding_storage)};

    bias_add_op->Execute(bias_add_input_bindings_vec_desc,
                         bias_add_output_bindings_vec_desc);
    current_output = bias_output;
  }

  // 5. Apply activation or copy to final output
  if (activation_op) {
    DML_BUFFER_BINDING activation_input_buffer_binding_storage =
        dml::utils::create_buffer_binding(current_output.Get(), 0,
                                          y.size() * sizeof(float));
    DML_BUFFER_BINDING final_y_buffer_binding_storage =
        dml::utils::create_buffer_binding(y_buffer, 0,
                                          y.size() * sizeof(float));

    DML_BINDING_DESC activation_input_binding_desc =
        dml::utils::create_binding_desc(
            &activation_input_buffer_binding_storage);
    DML_BINDING_DESC final_y_output_binding_desc =
        dml::utils::create_binding_desc(&final_y_buffer_binding_storage);

    activation_op->Execute({activation_input_binding_desc},
                           {final_y_output_binding_desc});
  } else {
    // Copy current output to final output
    command_list->CopyBufferRegion(y_buffer, 0, current_output.Get(), 0,
                                   y.size() * sizeof(float));
    device->ExecuteCommandList();
  }
}

}  // namespace ops
}  // namespace ctranslate2
#endif
