#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/bias_add.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Local get_dml_data_type removed

template <Device D, typename T>
void BiasAdd::compute(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output) const {
  auto* dml_device_ptr = dml::get_dml_device();
  auto* ct2_dml_device = dml::get_device();
  std::unique_ptr<DML_OPERATOR_DESC> fused_activation_op_desc_holder;

  DataType ct2_data_type = value.dtype();

  const dim_t depth = bias.size();
  const dim_t batch_size = value.size() / depth;

  std::vector<UINT> value_output_dims_vec = {static_cast<UINT>(batch_size),
                                             static_cast<UINT>(depth)};
  std::vector<UINT> bias_dims_vec = {1, static_cast<UINT>(depth)};

  dml::utils::DmlTensorDescBundle value_desc_bundle(
      ct2_data_type, value_output_dims_vec, nullptr, value.size() * sizeof(T));
  dml::utils::DmlTensorDescBundle bias_desc_bundle(
      ct2_data_type, bias_dims_vec, nullptr, bias.size() * sizeof(T));
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      ct2_data_type, value_output_dims_vec, nullptr, output.size() * sizeof(T));

  const DML_TENSOR_DESC& dml_value_tensor_desc =
      value_desc_bundle.get_tensor_desc();
  const DML_TENSOR_DESC& dml_bias_tensor_desc =
      bias_desc_bundle.get_tensor_desc();
  const DML_TENSOR_DESC& dml_output_tensor_desc =
      output_desc_bundle
          .get_tensor_desc();  // Used as input to activation if not fused

  dml::Operator* compiled_op_ptr =
      nullptr;  // Renamed to avoid conflict if activation fallback creates
                // another one
  std::unique_ptr<void, void (*)(void*)> activation_desc_payload_storage(
      nullptr, [](void* p) {});

  bool perform_separate_activation = false;
  DML_OPERATOR_DESC separate_activation_op_desc = {};  // For fallback

  if (!_activation_type) {
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_op_payload = {};
    add_op_payload.ATensor = &dml_value_tensor_desc;
    add_op_payload.BTensor = &dml_bias_tensor_desc;
    add_op_payload.OutputTensor = &dml_output_tensor_desc;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                 &add_op_payload};
    compiled_op_ptr = dml::GetOrCreateCompiledOperatorApi(&op_desc);
  } else {
    fused_activation_op_desc_holder = std::make_unique<DML_OPERATOR_DESC>();

    // Lambda to set up activation description payload
    auto setup_activation_payload = [&](auto& payload_ptr,
                                        DML_OPERATOR_TYPE type,
                                        const DML_TENSOR_DESC* input_desc,
                                        const DML_TENSOR_DESC* output_desc) {
      payload_ptr->InputTensor = input_desc;
      payload_ptr->OutputTensor = output_desc;
      fused_activation_op_desc_holder->Type = type;
      fused_activation_op_desc_holder->Desc = payload_ptr.get();
      activation_desc_payload_storage.reset(payload_ptr.release());
      // The deleter type needs to be templated or use a common base if
      // possible, but for unique_ptr with custom deleter, type erasure isn't
      // direct. The lambda captures approach with individual delete statements
      // per type is safer for now.
    };

    switch (*_activation_type) {
      case ActivationType::ReLU: {
        auto payload = std::make_unique<DML_ACTIVATION_RELU_OPERATOR_DESC>();
        setup_activation_payload(payload, DML_OPERATOR_ACTIVATION_RELU,
                                 &dml_output_tensor_desc,
                                 &dml_output_tensor_desc);
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_RELU_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Sigmoid: {
        auto payload = std::make_unique<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
        setup_activation_payload(payload, DML_OPERATOR_ACTIVATION_SIGMOID,
                                 &dml_output_tensor_desc,
                                 &dml_output_tensor_desc);
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_SIGMOID_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Tanh: {
        auto payload = std::make_unique<DML_ACTIVATION_TANH_OPERATOR_DESC>();
        setup_activation_payload(payload, DML_OPERATOR_ACTIVATION_TANH,
                                 &dml_output_tensor_desc,
                                 &dml_output_tensor_desc);
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_TANH_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::GELU: {  // Assuming DML_OPERATOR_ACTIVATION_GELU
                                    // exists and is standard GELU
        auto payload = std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
        setup_activation_payload(payload, DML_OPERATOR_ACTIVATION_GELU,
                                 &dml_output_tensor_desc,
                                 &dml_output_tensor_desc);
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_GELU_OPERATOR_DESC*>(p);
        };
        break;
      }
      case ActivationType::Swish: {
        auto payload = std::make_unique<DML_ACTIVATION_SWISH_OPERATOR_DESC>();
        payload->SigmoidInputScale = 1.0f;  // Common default
        setup_activation_payload(payload, DML_OPERATOR_ACTIVATION_SWISH,
                                 &dml_output_tensor_desc,
                                 &dml_output_tensor_desc);
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_SWISH_OPERATOR_DESC*>(p);
        };
        break;
      }
      default:
        fused_activation_op_desc_holder
            .reset();  // No suitable DML fused activation
        perform_separate_activation = true;
        // Setup separate_activation_op_desc for fallback if an unsupported type
        // needs it For GELUTanh and GELUSigmoid specifically. Other unsupported
        // fall to error.
        if (*_activation_type == ActivationType::GELUTanh ||
            *_activation_type == ActivationType::GELUSigmoid) {
          auto gelu_payload_fallback =
              std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
          gelu_payload_fallback->InputTensor = &dml_output_tensor_desc;
          gelu_payload_fallback->OutputTensor = &dml_output_tensor_desc;
          separate_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
          separate_activation_op_desc.Desc = gelu_payload_fallback.get();
          // activation_desc_payload_storage will manage this if we proceed
          activation_desc_payload_storage.reset(
              gelu_payload_fallback.release());
          activation_desc_payload_storage.get_deleter() = [](void* p) {
            delete static_cast<DML_ACTIVATION_GELU_OPERATOR_DESC*>(p);
          };
        } else {
          throw std::runtime_error(
              "Unsupported activation type for DirectML BiasAdd fusion and no "
              "simple fallback for the specified activation type.");
        }
        break;
    }

    if (fused_activation_op_desc_holder && !perform_separate_activation) {
      DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add1_op_payload = {};
      add1_op_payload.ATensor = &dml_value_tensor_desc;
      add1_op_payload.BTensor = &dml_bias_tensor_desc;
      add1_op_payload.OutputTensor = &dml_output_tensor_desc;
      add1_op_payload.FusedActivation = fused_activation_op_desc_holder.get();
      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD1,
                                   &add1_op_payload};
      compiled_op_ptr = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    } else {  // Fallback to separate add + activation
      DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_op_payload = {};
      add_op_payload.ATensor = &dml_value_tensor_desc;
      add_op_payload.BTensor = &dml_bias_tensor_desc;
      add_op_payload.OutputTensor =
          &dml_output_tensor_desc;  // Add writes to final output
      DML_OPERATOR_DESC add_op_desc_separate = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                                &add_op_payload};
      dml::Operator* add_compiled_op =
          dml::GetOrCreateCompiledOperatorApi(&add_op_desc_separate);

      // Execute Add
      DML_BUFFER_BINDING value_binding_s = dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(value.buffer())),
          0, value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
      DML_BINDING_DESC value_binding_d =
          dml::utils::create_binding_desc(&value_binding_s);
      DML_BUFFER_BINDING bias_binding_s = dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(bias.buffer())),
          0, bias_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
      DML_BINDING_DESC bias_binding_d =
          dml::utils::create_binding_desc(&bias_binding_s);
      DML_BUFFER_BINDING output_binding_s_for_add =
          dml::utils::create_buffer_binding(
              reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
              output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
      DML_BINDING_DESC output_binding_d_for_add =
          dml::utils::create_binding_desc(&output_binding_s_for_add);
      add_compiled_op->Execute({value_binding_d, bias_binding_d},
                               {output_binding_d_for_add});

      // Now `compiled_op_ptr` will be the activation op
      compiled_op_ptr =
          dml::GetOrCreateCompiledOperatorApi(&separate_activation_op_desc);
      // Bindings for this separate activation op will be set up later in the
      // common binding section. Input for activation is the 'output' buffer,
      // which now contains result of ADD.
    }
  }

  // Common binding logic
  std::vector<DML_BINDING_DESC> final_input_bindings;
  if (!perform_separate_activation) {  // Case: No activation OR Fused
                                       // activation
    DML_BUFFER_BINDING val_b_s = dml::utils::create_buffer_binding(
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(value.buffer())), 0,
        value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC val_b_d = dml::utils::create_binding_desc(&val_b_s);
    final_input_bindings.push_back(val_b_d);

    DML_BUFFER_BINDING bias_b_s = dml::utils::create_buffer_binding(
        reinterpret_cast<ID3D12Resource*>(const_cast<void*>(bias.buffer())), 0,
        bias_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC bias_b_d = dml::utils::create_binding_desc(&bias_b_s);
    final_input_bindings.push_back(bias_b_d);
  } else {  // Case: Separate activation step; input is the result of previous
            // ADD (now in 'output' buffer)
    DML_BUFFER_BINDING act_in_s = dml::utils::create_buffer_binding(
        reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    DML_BINDING_DESC act_in_d = dml::utils::create_binding_desc(&act_in_s);
    final_input_bindings.push_back(act_in_d);
  }

  DML_BUFFER_BINDING final_out_s = dml::utils::create_buffer_binding(
      reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
      output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC final_out_d = dml::utils::create_binding_desc(&final_out_s);

  compiled_op_ptr->Execute(final_input_bindings, {final_out_d});
}

#define DECLARE_IMPL(T)                                                       \
  template void BiasAdd::compute<Device::DirectML, T>(                        \
      const StorageView& value, const StorageView& bias, StorageView& output) \
      const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif