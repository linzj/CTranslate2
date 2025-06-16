#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/bias_add.h"

#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

// Local get_dml_data_type removed
namespace {  // Anonymous namespace for file-local helper

// Attempts to prepare an activation operator's descriptor for fusion with
// DML_ELEMENT_WISE_ADD1. If successful, populates
// `out_fused_activation_op_desc`, manages the payload via `payload_storage`,
// and returns true. Otherwise, returns false.
// For fusion, the InputTensor and OutputTensor of the specific activation
// payload must be nullptr.
bool prepare_activation_descriptor_for_add1_fusion(
    ctranslate2::ops::ActivationType activation_type,
    dml::utils::DmlOperatorDescBundle& op_bundle) {
  switch (activation_type) {
    case ctranslate2::ops::ActivationType::ReLU: {
      op_bundle.GetFusedOperatorDesc<DML_ACTIVATION_RELU_OPERATOR_DESC>();
      return true;
    }
    case ctranslate2::ops::ActivationType::Sigmoid: {
      op_bundle.GetFusedOperatorDesc<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
      return true;
    }
    case ctranslate2::ops::ActivationType::Tanh: {
      op_bundle.GetFusedOperatorDesc<DML_ACTIVATION_TANH_OPERATOR_DESC>();
      return true;
    }
    case ctranslate2::ops::ActivationType::GELU: {
      op_bundle.GetFusedOperatorDesc<DML_ACTIVATION_GELU_OPERATOR_DESC>();
      return true;
    }
    case ctranslate2::ops::ActivationType::Swish: {
      auto& payload =
          op_bundle.GetFusedOperatorDesc<DML_ACTIVATION_SWISH_OPERATOR_DESC>();
      payload.SigmoidInputScale = 1.0f;  // Common default
      return true;
    }
      // GELUTanh and GELUSigmoid are typically not directly fused in DML with a
      // single op. Other types that aren't standard DML fused activations.
    default:
      return false;  // Cannot prepare a standard DML fused activation
                     // descriptor for this type.
  }
}

}  // anonymous namespace

template <Device D, typename T>
void BiasAdd::compute(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output) const {
  DataType ct2_data_type = value.dtype();

  const dim_t depth = bias.size();
  const dim_t batch_size = value.size() / depth;

  std::vector<UINT> value_output_dims_vec = {static_cast<UINT>(batch_size),
                                             static_cast<UINT>(depth)};
  std::vector<UINT> bias_dims_vec = {1, static_cast<UINT>(depth)};

  dml::Operator* compiled_op_ptr =
      nullptr;  // Renamed to avoid conflict if activation fallback creates
                // another one

  bool perform_separate_activation = false;

  if (_activation_type) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    bool can_fuse = prepare_activation_descriptor_for_add1_fusion(
        *_activation_type, op_bundle);

    if (can_fuse) {
      auto& value_tensor =
          op_bundle.AddInput(ct2_data_type, value_output_dims_vec, nullptr,
                             value.size() * sizeof(T));
      std::vector<UINT> broadcasted_bias_strides_for_op_vec = {0, 1};
      auto& bias_tensor = op_bundle.AddInput(
          ct2_data_type, value_output_dims_vec,
          &broadcasted_bias_strides_for_op_vec, bias.size() * sizeof(T));
      auto& output_tensor =
          op_bundle.AddOutput(ct2_data_type, value_output_dims_vec, nullptr,
                              output.size() * sizeof(T));

      auto& add1_op_payload =
          op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD1_OPERATOR_DESC>();
      add1_op_payload.ATensor = &value_tensor.get_tensor_desc();
      add1_op_payload.BTensor = &bias_tensor.get_tensor_desc();
      add1_op_payload.OutputTensor = &output_tensor.get_tensor_desc();
      add1_op_payload.FusedActivation = &op_bundle.get_fused_desc();

      compiled_op_ptr =
          dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    } else {
      perform_separate_activation = true;
      if (*_activation_type == ActivationType::GELUTanh ||
          *_activation_type == ActivationType::GELUSigmoid) {
        dml::utils::DmlOperatorDescBundle gelu_op_bundle;
        auto& input_tensor =
            gelu_op_bundle.AddInput(ct2_data_type, value_output_dims_vec,
                                    nullptr, output.size() * sizeof(T));
        auto& output_tensor =
            gelu_op_bundle.AddOutput(ct2_data_type, value_output_dims_vec,
                                     nullptr, output.size() * sizeof(T));
        auto& gelu_payload =
            gelu_op_bundle.GetOperatorDesc<DML_ACTIVATION_GELU_OPERATOR_DESC>();
        gelu_payload.InputTensor = &input_tensor.get_tensor_desc();
        gelu_payload.OutputTensor = &output_tensor.get_tensor_desc();
        compiled_op_ptr =
            dml::GetOrCreateCompiledOperatorApi(std::move(gelu_op_bundle));
      } else {  // Not fusable and not GELU fallback.
        throw std::runtime_error(
            "Unsupported activation type for DirectML BiasAdd fusion and no "
            "simple fallback (like separate GELU) is defined for: " +
            std::to_string(static_cast<int>(*_activation_type)));
      }
    }

  } else {  // No activation type specified, perform a simple ADD
    dml::utils::DmlOperatorDescBundle add_op_bundle;
    auto& value_tensor =
        add_op_bundle.AddInput(ct2_data_type, value_output_dims_vec, nullptr,
                               value.size() * sizeof(T));
    std::vector<UINT> broadcasted_bias_strides_for_op_vec = {0, 1};
    auto& bias_tensor = add_op_bundle.AddInput(
        ct2_data_type, value_output_dims_vec,
        &broadcasted_bias_strides_for_op_vec, bias.size() * sizeof(T));
    auto& output_tensor =
        add_op_bundle.AddOutput(ct2_data_type, value_output_dims_vec, nullptr,
                                output.size() * sizeof(T));
    auto& add_op_payload =
        add_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    add_op_payload.ATensor = &value_tensor.get_tensor_desc();
    add_op_payload.BTensor = &bias_tensor.get_tensor_desc();
    add_op_payload.OutputTensor = &output_tensor.get_tensor_desc();
    compiled_op_ptr =
        dml::GetOrCreateCompiledOperatorApi(std::move(add_op_bundle));
  }

  // If a separate activation is needed (either because fusion failed, or was
  // non-fusable type like GELU)
  if (perform_separate_activation) {
    // First, perform the ADD operation
    dml::utils::DmlOperatorDescBundle add_op_bundle;
    auto& value_tensor =
        add_op_bundle.AddInput(ct2_data_type, value_output_dims_vec, nullptr,
                               value.size() * sizeof(T));
    std::vector<UINT> broadcasted_bias_strides_for_op_vec = {0, 1};
    auto& bias_tensor = add_op_bundle.AddInput(
        ct2_data_type, value_output_dims_vec,
        &broadcasted_bias_strides_for_op_vec, bias.size() * sizeof(T));
    auto& output_tensor =
        add_op_bundle.AddOutput(ct2_data_type, value_output_dims_vec, nullptr,
                                output.size() * sizeof(T));
    auto& add_op_payload =
        add_op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    add_op_payload.ATensor = &value_tensor.get_tensor_desc();
    add_op_payload.BTensor = &bias_tensor.get_tensor_desc();
    add_op_payload.OutputTensor = &output_tensor.get_tensor_desc();
    dml::Operator* add_compiled_op =
        dml::GetOrCreateCompiledOperatorApi(std::move(add_op_bundle));

    // Execute the standalone Add
    dml::utils::DmlBindingArrayBundle inputs_add{
        {dml::utils::ResourceFromStorageView(value), 0,
         value.size() * sizeof(T)},
        {dml::utils::ResourceFromStorageView(bias), 0,
         bias.size() * sizeof(T)}};
    dml::utils::DmlBindingArrayBundle outputs_add{
        {dml::utils::ResourceFromStorageView(output), 0,
         output.size() * sizeof(T)}};

    add_compiled_op->Execute(inputs_add, outputs_add);
  }

  if (compiled_op_ptr) {
    dml::utils::DmlBindingArrayBundle final_outputs_main{
        {dml::utils::ResourceFromStorageView(output), 0,
         output.size() * sizeof(T)}};

    if (!perform_separate_activation) {  // Fused case
      dml::utils::DmlBindingArrayBundle inputs_fused{
          {dml::utils::ResourceFromStorageView(value), 0,
           value.size() * sizeof(T)},
          {dml::utils::ResourceFromStorageView(bias), 0,
           bias.size() * sizeof(T)}};
      compiled_op_ptr->Execute(inputs_fused, final_outputs_main);
    } else {  // Separate activation case
      dml::utils::DmlBindingArrayBundle inputs_separate_act{
          {dml::utils::ResourceFromStorageView(output), 0,
           output.size() * sizeof(T)}};
      compiled_op_ptr->Execute(inputs_separate_act, final_outputs_main);
    }
  }
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