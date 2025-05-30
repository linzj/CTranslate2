#ifdef CT2_WITH_DIRECTML

#include "ctranslate2/ops/softmax.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void SoftMax::compute(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output) const {
  // Get DML device and context
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  // Tensor descriptors using DmlTensorDescBundle
  // Softmax is typically applied on the last dimension.
  // The input is conceptually [batch_size, depth].
  std::vector<UINT> dml_dims_vec = {static_cast<UINT>(batch_size),
                                    static_cast<UINT>(depth)};

  // DML_TENSOR_DATA_TYPE is derived from input.dtype() inside
  // DmlTensorDescBundle
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), dml_dims_vec, nullptr, input.size() * sizeof(T));
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), dml_dims_vec, nullptr, output.size() * sizeof(T));

  const DML_TENSOR_DESC& dml_input_desc_ref =
      input_desc_bundle.get_tensor_desc();
  const DML_TENSOR_DESC& dml_output_desc_ref =
      output_desc_bundle.get_tensor_desc();

  // Create the appropriate operator descriptor
  DML_OPERATOR_DESC op_desc = {};

// Use the newer softmax operators that support axis specification
#if DML_TARGET_VERSION >= 0x5100
  if (_log) {
    // Log softmax operator with axis support
    UINT axis = 1;  // Last dimension in our 2D tensor

    DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC log_softmax_desc = {};
    log_softmax_desc.InputTensor = &dml_input_desc_ref;
    log_softmax_desc.OutputTensor = &dml_output_desc_ref;
    log_softmax_desc.AxisCount = 1;
    log_softmax_desc.Axes = &axis;

    op_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1;
    op_desc.Desc = &log_softmax_desc;
  } else {
    // Regular softmax operator with axis support
    UINT axis = 1;  // Last dimension in our 2D tensor

    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmax_desc = {};
    softmax_desc.InputTensor = &dml_input_desc_ref;
    softmax_desc.OutputTensor = &dml_output_desc_ref;
    softmax_desc.AxisCount = 1;
    softmax_desc.Axes = &axis;

    op_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX1;
    op_desc.Desc = &softmax_desc;
  }
#else
  // Fallback to older operators (no axis support - operates on entire tensor)
  if (_log) {
    DML_ACTIVATION_LOG_SOFTMAX_OPERATOR_DESC log_softmax_desc = {};
    log_softmax_desc.InputTensor = &dml_input_desc_ref;
    log_softmax_desc.OutputTensor = &dml_output_desc_ref;

    op_desc.Type = DML_OPERATOR_ACTIVATION_LOG_SOFTMAX;
    op_desc.Desc = &log_softmax_desc;
  } else {
    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax_desc = {};
    softmax_desc.InputTensor = &dml_input_desc_ref;
    softmax_desc.OutputTensor = &dml_output_desc_ref;

    op_desc.Type = DML_OPERATOR_ACTIVATION_SOFTMAX;
    op_desc.Desc = &softmax_desc;
  }
#endif

  // Get or create compiled operator from cache
  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  // Setup input binding
  // Setup bindings using dml::utils
  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
          0,
          input_desc_bundle.get_buffer_desc()
              .TotalTensorSizeInBytes);  // This line should now work
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output_desc_bundle.get_buffer_desc()
              .TotalTensorSizeInBytes);  // This line should now work
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  std::vector<DML_BINDING_DESC> input_bindings_for_op_vec = {
      input_binding_desc_for_op};
  std::vector<DML_BINDING_DESC> output_bindings_for_op_vec = {
      output_binding_desc_for_op};
  compiled_op->Execute(input_bindings_for_op_vec, output_bindings_for_op_vec);

  // Handle lengths parameter if provided - mask output for out-of-sequence
  // positions
  if (lengths) {
#if 0
    // Use primitives to zero out positions beyond sequence length
    auto output_data = output.data<T>();
    auto length_data = lengths->data<int32_t>();

    for (dim_t batch = 0; batch < batch_size; ++batch) {
      dim_t seq_len = static_cast<dim_t>(length_data[batch]);
      if (seq_len < depth) {
        T* batch_output = output_data + batch * depth;
        // Zero out positions beyond sequence length
        for (dim_t i = seq_len; i < depth; ++i) {
          batch_output[i] = T(0);
        }
      }
    }
#else
    throw std::invalid_argument(
        "SoftMax with lengths parameter is not implemented for DirectML");
#endif
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