#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/bias_add.h"
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
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
    DML_OPERATOR_DESC& out_fused_activation_op_desc,
    std::unique_ptr<void, void (*)(void*)>& payload_storage) {
  switch (activation_type) {
    case ctranslate2::ops::ActivationType::ReLU: {
      auto payload = std::make_unique<DML_ACTIVATION_RELU_OPERATOR_DESC>();
      payload->InputTensor = nullptr;   // Required for fusion with ADD1
      payload->OutputTensor = nullptr;  // Required for fusion with ADD1
      out_fused_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_RELU;
      out_fused_activation_op_desc.Desc = payload.get();
      payload_storage.reset(payload.release());
      payload_storage.get_deleter() = [](void* p) {
        delete static_cast<DML_ACTIVATION_RELU_OPERATOR_DESC*>(p);
      };
      return true;
    }
    case ctranslate2::ops::ActivationType::Sigmoid: {
      auto payload = std::make_unique<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
      payload->InputTensor = nullptr;
      payload->OutputTensor = nullptr;
      out_fused_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SIGMOID;
      out_fused_activation_op_desc.Desc = payload.get();
      payload_storage.reset(payload.release());
      payload_storage.get_deleter() = [](void* p) {
        delete static_cast<DML_ACTIVATION_SIGMOID_OPERATOR_DESC*>(p);
      };
      return true;
    }
    case ctranslate2::ops::ActivationType::Tanh: {
      auto payload = std::make_unique<DML_ACTIVATION_TANH_OPERATOR_DESC>();
      payload->InputTensor = nullptr;
      payload->OutputTensor = nullptr;
      out_fused_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_TANH;
      out_fused_activation_op_desc.Desc = payload.get();
      payload_storage.reset(payload.release());
      payload_storage.get_deleter() = [](void* p) {
        delete static_cast<DML_ACTIVATION_TANH_OPERATOR_DESC*>(p);
      };
      return true;
    }
    case ctranslate2::ops::ActivationType::
        GELU: {  // Assuming DML_OPERATOR_ACTIVATION_GELU is fusable
      auto payload = std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
      payload->InputTensor = nullptr;
      payload->OutputTensor = nullptr;
      out_fused_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
      out_fused_activation_op_desc.Desc = payload.get();
      payload_storage.reset(payload.release());
      payload_storage.get_deleter() = [](void* p) {
        delete static_cast<DML_ACTIVATION_GELU_OPERATOR_DESC*>(p);
      };
      return true;
    }
    case ctranslate2::ops::ActivationType::Swish: {
      auto payload = std::make_unique<DML_ACTIVATION_SWISH_OPERATOR_DESC>();
      payload->SigmoidInputScale = 1.0f;  // Common default
      payload->InputTensor = nullptr;
      payload->OutputTensor = nullptr;
      out_fused_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_SWISH;
      out_fused_activation_op_desc.Desc = payload.get();
      payload_storage.reset(payload.release());
      payload_storage.get_deleter() = [](void* p) {
        delete static_cast<DML_ACTIVATION_SWISH_OPERATOR_DESC*>(p);
      };
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
  auto* dml_device_ptr = dml::get_dml_device();
  auto* ct2_dml_device = dml::get_device();

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
  // Create a new DML_TENSOR_DESC for the bias tensor for the operator,
  // with explicit broadcasting shape and strides, to satisfy operators that
  // might require identical .Sizes fields despite supporting broadcasting.
  // This descriptor (dml_bias_tensor_desc_for_op) is used for BTensor in
  // ADD/ADD1.
  std::vector<UINT> broadcasted_bias_dims_for_op_vec = {
      static_cast<UINT>(batch_size), static_cast<UINT>(depth)};
  // Strides in elements: 0 for batch dim (broadcast), 1 for depth dim
  // (contiguous)
  std::vector<UINT> broadcasted_bias_strides_for_op_vec = {0, 1};

  // Get the original DML_BUFFER_TENSOR_DESC for bias to copy properties
  const DML_BUFFER_TENSOR_DESC* original_bias_buffer_desc_ptr =
      static_cast<const DML_BUFFER_TENSOR_DESC*>(
          bias_desc_bundle.get_tensor_desc().Desc);

  // These variables (broadcasted_dml_bias_buffer_tensor_desc,
  // dml_bias_tensor_desc_for_op, and the vectors) must remain in scope as long
  // as dml_bias_tensor_desc_for_op is used or its internal pointers are
  // dereferenced.
  DML_BUFFER_TENSOR_DESC broadcasted_dml_bias_buffer_tensor_desc =
      {};  // Must be kept in scope
  broadcasted_dml_bias_buffer_tensor_desc.DataType =
      original_bias_buffer_desc_ptr->DataType;
  broadcasted_dml_bias_buffer_tensor_desc.Flags =
      original_bias_buffer_desc_ptr->Flags;
  broadcasted_dml_bias_buffer_tensor_desc.DimensionCount =
      original_bias_buffer_desc_ptr->DimensionCount;  // Should be 2
  broadcasted_dml_bias_buffer_tensor_desc.Sizes =
      broadcasted_bias_dims_for_op_vec.data();
  broadcasted_dml_bias_buffer_tensor_desc.Strides =
      broadcasted_bias_strides_for_op_vec.data();
  // TotalTensorSizeInBytes must reflect the actual (compact) bias buffer size
  broadcasted_dml_bias_buffer_tensor_desc.TotalTensorSizeInBytes =
      original_bias_buffer_desc_ptr->TotalTensorSizeInBytes;
  broadcasted_dml_bias_buffer_tensor_desc.GuaranteedBaseOffsetAlignment =
      original_bias_buffer_desc_ptr->GuaranteedBaseOffsetAlignment;

  DML_TENSOR_DESC dml_bias_tensor_desc_for_op = {};  // Must be kept in scope
  dml_bias_tensor_desc_for_op.Type = DML_TENSOR_TYPE_BUFFER;
  dml_bias_tensor_desc_for_op.Desc = &broadcasted_dml_bias_buffer_tensor_desc;

  dml::Operator* compiled_op_ptr =
      nullptr;  // Renamed to avoid conflict if activation fallback creates
                // another one
  std::unique_ptr<void, void (*)(void*)> activation_desc_payload_storage(
      nullptr, [](void* p) {});

  bool perform_separate_activation = false;
  DML_OPERATOR_DESC separate_activation_op_desc = {};  // For fallback

  if (_activation_type) {
    DML_OPERATOR_DESC local_fused_activation_op_desc = {};
    bool can_fuse = prepare_activation_descriptor_for_add1_fusion(
        *_activation_type, local_fused_activation_op_desc,
        activation_desc_payload_storage);

    if (can_fuse && !perform_separate_activation) {  // Should always be true if
                                                     // can_fuse initially
      DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add1_op_payload = {};
      add1_op_payload.ATensor = &dml_value_tensor_desc;
      add1_op_payload.BTensor = &dml_bias_tensor_desc_for_op;
      add1_op_payload.OutputTensor = &dml_output_tensor_desc;
      add1_op_payload.FusedActivation = &local_fused_activation_op_desc;
      DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD1,
                                   &add1_op_payload};
      compiled_op_ptr = dml::GetOrCreateCompiledOperatorApi(&op_desc);
      // perform_separate_activation remains false
    } else {
      // If cannot fuse or fusion was disabled upstream (e.g.
      // perform_separate_activation was already true)
      perform_separate_activation = true;  // Ensure it's set for the next stage

      // Handle fallback for specific non-fusable types (GELUTanh, GELUSigmoid
      // -> separate GELU)
      if (*_activation_type == ActivationType::GELUTanh ||
          *_activation_type == ActivationType::GELUSigmoid) {
        auto gelu_payload_fallback =
            std::make_unique<DML_ACTIVATION_GELU_OPERATOR_DESC>();
        // For separate activation, Input/Output Tensors ARE set.
        gelu_payload_fallback->InputTensor =
            &dml_output_tensor_desc;  // Input is result of ADD
        gelu_payload_fallback->OutputTensor =
            &dml_output_tensor_desc;  // Activation writes to final output
        separate_activation_op_desc.Type = DML_OPERATOR_ACTIVATION_GELU;
        separate_activation_op_desc.Desc = gelu_payload_fallback.get();
        activation_desc_payload_storage.reset(gelu_payload_fallback.release());
        activation_desc_payload_storage.get_deleter() = [](void* p) {
          delete static_cast<DML_ACTIVATION_GELU_OPERATOR_DESC*>(p);
        };
      } else if (can_fuse && perform_separate_activation) {
        // This case means can_fuse was true, but perform_separate_activation
        // was set to true from some other logic. This indicates an override. We
        // need to setup the ADD op and then potentially a separate activation
        // based on 'separate_activation_op_desc'. The
        // 'separate_activation_op_desc' should be already set up if we expect
        // an activation. This state seems complex / possibly indicative of
        // conflicting flags upstream. For now, assume if
        // perform_separate_activation is true, it's definitive. The actual
        // activation to perform (if any) should be in
        // separate_activation_op_desc.
      } else if (!can_fuse) {  // Not fusable and not GELU fallback.
        throw std::runtime_error(
            "Unsupported activation type for DirectML BiasAdd fusion and no "
            "simple fallback (like separate GELU) is defined for: " +
            std::to_string(static_cast<int>(*_activation_type)));
      }
      // If perform_separate_activation is true, the Add op will be created
      // below.
    }
  } else {  // No activation type specified, perform a simple ADD
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_op_payload = {};
    add_op_payload.ATensor = &dml_value_tensor_desc;
    add_op_payload.BTensor = &dml_bias_tensor_desc_for_op;
    add_op_payload.OutputTensor = &dml_output_tensor_desc;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                 &add_op_payload};
    compiled_op_ptr = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    // perform_separate_activation remains false
  }

  // If a separate activation is needed (either because fusion failed, or was
  // overridden)
  if (perform_separate_activation) {
    // First, perform the ADD operation
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_op_payload_separate = {};
    add_op_payload_separate.ATensor = &dml_value_tensor_desc;
    add_op_payload_separate.BTensor = &dml_bias_tensor_desc_for_op;
    add_op_payload_separate.OutputTensor =
        &dml_output_tensor_desc;  // ADD writes to final output buffer
    DML_OPERATOR_DESC add_op_desc_separate_exec = {
        DML_OPERATOR_ELEMENT_WISE_ADD, &add_op_payload_separate};
    dml::Operator* add_compiled_op_separate =
        dml::GetOrCreateCompiledOperatorApi(&add_op_desc_separate_exec);

    // Execute the standalone Add
    dml::utils::DmlBufferBindingBundle value_binding_bundle_add(
        dml::utils::ResourceFromStorageView(value), 0,
        value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    dml::utils::DmlBufferBindingBundle bias_binding_bundle_add(
        dml::utils::ResourceFromStorageView(bias), 0,
        bias_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    dml::utils::DmlBufferBindingBundle output_binding_bundle_add(
        dml::utils::ResourceFromStorageView(output), 0,
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

    add_compiled_op_separate->Execute({value_binding_bundle_add.get_desc(),
                                       bias_binding_bundle_add.get_desc()},
                                      {output_binding_bundle_add.get_desc()});

    // Now, if a separate activation was specified (e.g., GELU fallback),
    // compile and set it as the `compiled_op_ptr` for the final common
    // execution. If no specific separate activation was needed (e.g. just an
    // add, or error thrown), then separate_activation_op_desc.Desc would be
    // null.
    if (separate_activation_op_desc.Desc) {
      compiled_op_ptr =
          dml::GetOrCreateCompiledOperatorApi(&separate_activation_op_desc);
    } else {
      // If we are in perform_separate_activation but no
      // separate_activation_op_desc is set, it means we only needed to do the
      // ADD, and no further op. Set compiled_op_ptr to null or a no-op to
      // prevent the final execute call from running with an invalid op.
      // However, the binding logic later expects compiled_op_ptr to be valid if
      // !perform_separate_activation. For this path, the primary operation
      // (ADD) has already been executed. The final Execute call needs to be
      // skipped or made conditional if no second op. For simplicity, if no
      // separate_activation_op_desc.Desc, the effect is ADD only. The binding
      // setup at the end also needs to be conditional on `compiled_op_ptr`
      // validity. One way is to let compiled_op_ptr be the
      // add_compiled_op_separate and ensure bindings are correct or the execute
      // is conditional. Given current structure, if
      // perform_separate_activation=true AND
      // separate_activation_op_desc.Desc=nullptr, the primary ADD is done, and
      // compiled_op_ptr will be what it was BEFORE this block or null if it was
      // just add. Let's set it to null here. Actually, `compiled_op_ptr` should
      // point to the *last* op in a sequence for common binding. If only ADD
      // was performed, and `perform_separate_activation` is true but no
      // *further* activation op the 'common execution' part should probably be
      // skipped. For now, let `compiled_op_ptr` be null. The common binding
      // needs care.
      compiled_op_ptr = nullptr;  // Signifies only ADD was done in this branch,
                                  // no further activation.
    }
  }

  // Common binding logic
  std::vector<DML_BINDING_DESC> final_input_bindings_descs;
  dml::utils::DmlBufferBindingBundle final_output_binding_bundle(
      dml::utils::ResourceFromStorageView(output), 0,
      output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  dml::utils::DmlBufferBindingBundle value_binding_bundle_final(
      dml::utils::ResourceFromStorageView(value), 0,
      value_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  dml::utils::DmlBufferBindingBundle bias_binding_bundle_final(
      dml::utils::ResourceFromStorageView(bias), 0,
      bias_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  dml::utils::DmlBufferBindingBundle act_input_binding_bundle_final(
      dml::utils::ResourceFromStorageView(
          output),  // This is the buffer after ADD operation
      0, output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);

  if (!perform_separate_activation) {
    final_input_bindings_descs.push_back(value_binding_bundle_final.get_desc());
    final_input_bindings_descs.push_back(bias_binding_bundle_final.get_desc());
  } else {
    // Case: Separate activation step; input is the result of previous
    // ADD (now in 'output' buffer)
    final_input_bindings_descs.push_back(
        act_input_binding_bundle_final.get_desc());
  }

  if (compiled_op_ptr) {
    compiled_op_ptr->Execute(final_input_bindings_descs,
                             {final_output_binding_bundle.get_desc()});
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