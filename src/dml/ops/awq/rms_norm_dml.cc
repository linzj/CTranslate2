#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rms_norm.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

namespace {}  // namespace

template <Device D, typename T>
void RMSNorm::compute(const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* device = dml::get_device();

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  const DML_TENSOR_DATA_TYPE data_type =
      dml::utils::get_dml_data_type(input.dtype());
  const auto device_type = Device::DirectML;

  device->ResetCommandList();

  // Tensor Descriptors
  dml::utils::DmlTensorDescBundle input_desc(input);
  dml::utils::DmlTensorDescBundle output_desc(output);
  std::vector<UINT> broadcast_dims = {(UINT)batch_size, (UINT)depth};

  // Describe Gamma as [1, depth] for broadcasting.
  std::vector<UINT> gamma_dims = {1, (UINT)depth};
  dml::utils::DmlTensorDescBundle gamma_desc(data_type, gamma_dims, nullptr,
                                             gamma.reserved_memory());

  // Step 1: Compute input_squared = input * input
  StorageView input_squared(input.shape(), input.dtype(), device_type);
  dml::utils::DmlTensorDescBundle input_squared_desc(input_squared);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc = {};
  multiply_desc.ATensor = &input_desc.get_tensor_desc();
  multiply_desc.BTensor = &input_desc.get_tensor_desc();
  multiply_desc.OutputTensor = &input_squared_desc.get_tensor_desc();

  DML_OPERATOR_DESC multiply_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                        &multiply_desc};
  auto* multiply_op = dml::GetOrCreateCompiledOperatorApi(&multiply_op_desc);

  dml::utils::DmlBindingArrayBundle mult_inputs(
      {dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(input)),
       dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(input))});
  dml::utils::DmlBindingArrayBundle mult_outputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(input_squared))});
  multiply_op->Execute(mult_inputs.get_descs(), mult_outputs.get_descs());

  // Step 2: Compute mean_squared = reduce_mean(input_squared, axis=1)
  StorageView mean_squared({batch_size, 1}, input.dtype(), device_type);
  dml::utils::DmlTensorDescBundle mean_squared_desc(mean_squared);

  const uint32_t reduce_axes[] = {1};  // reduce on depth
  DML_REDUCE_OPERATOR_DESC reduce_desc = {};
  reduce_desc.Function = DML_REDUCE_FUNCTION_AVERAGE;
  reduce_desc.InputTensor = &input_squared_desc.get_tensor_desc();
  reduce_desc.OutputTensor = &mean_squared_desc.get_tensor_desc();
  reduce_desc.AxisCount = 1;
  reduce_desc.Axes = reduce_axes;

  DML_OPERATOR_DESC reduce_op_desc = {DML_OPERATOR_REDUCE, &reduce_desc};
  auto* reduce_op = dml::GetOrCreateCompiledOperatorApi(&reduce_op_desc);

  dml::utils::DmlBindingArrayBundle reduce_inputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(input_squared))});
  dml::utils::DmlBindingArrayBundle reduce_outputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(mean_squared))});
  reduce_op->Execute(reduce_inputs.get_descs(), reduce_outputs.get_descs());

  // Step 3: Add epsilon
  StorageView epsilon_sv({batch_size, 1}, input.dtype(), device_type);
  epsilon_sv.fill(static_cast<T>(_epsilon));

  dml::utils::DmlTensorDescBundle epsilon_desc(epsilon_sv);

  StorageView variance(mean_squared.shape(), input.dtype(), device_type);
  dml::utils::DmlTensorDescBundle variance_desc(variance);

  DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
  add_desc.ATensor = &mean_squared_desc.get_tensor_desc();
  add_desc.BTensor = &epsilon_desc.get_tensor_desc();
  add_desc.OutputTensor = &variance_desc.get_tensor_desc();

  DML_OPERATOR_DESC add_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc};
  auto* add_op = dml::GetOrCreateCompiledOperatorApi(&add_op_desc);

  dml::utils::DmlBindingArrayBundle add_inputs(
      {dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(mean_squared)),
       dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(epsilon_sv))});
  dml::utils::DmlBindingArrayBundle add_outputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(variance))});
  add_op->Execute(add_inputs.get_descs(), add_outputs.get_descs());

  // Step 4: Compute inv_rms = 1/sqrt(variance)
  // Use variance buffer for sqrt output then recip output (in-place)
  StorageView& inv_rms = variance;
  dml::utils::DmlTensorDescBundle& inv_rms_desc = variance_desc;

  DML_ELEMENT_WISE_SQRT_OPERATOR_DESC sqrt_desc = {};
  sqrt_desc.InputTensor = &inv_rms_desc.get_tensor_desc();
  sqrt_desc.OutputTensor = &inv_rms_desc.get_tensor_desc();

  DML_OPERATOR_DESC sqrt_op_desc = {DML_OPERATOR_ELEMENT_WISE_SQRT, &sqrt_desc};
  auto* sqrt_op = dml::GetOrCreateCompiledOperatorApi(&sqrt_op_desc);

  dml::utils::DmlBindingArrayBundle sqrt_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(inv_rms))});
  sqrt_op->Execute(sqrt_bindings.get_descs(), sqrt_bindings.get_descs());

  DML_ELEMENT_WISE_RECIP_OPERATOR_DESC recip_desc = {};
  recip_desc.InputTensor = &inv_rms_desc.get_tensor_desc();
  recip_desc.OutputTensor = &inv_rms_desc.get_tensor_desc();  // In-place

  DML_OPERATOR_DESC recip_op_desc = {DML_OPERATOR_ELEMENT_WISE_RECIP,
                                     &recip_desc};
  auto* recip_op = dml::GetOrCreateCompiledOperatorApi(&recip_op_desc);
  recip_op->Execute(sqrt_bindings.get_descs(), sqrt_bindings.get_descs());

  // Step 5: Normalize input = input * inv_rms
  std::vector<UINT> inv_rms_broadcast_strides = {
      1, 0};  // Stride 0 for broadcasting
  dml::utils::DmlTensorDescBundle inv_rms_broadcast_desc(
      data_type, broadcast_dims, &inv_rms_broadcast_strides,
      inv_rms.reserved_memory());

  StorageView normalized(input.shape(), input.dtype(), device_type);
  dml::utils::DmlTensorDescBundle normalized_desc(normalized);

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC norm_multiply_desc = {};
  norm_multiply_desc.ATensor = &input_desc.get_tensor_desc();
  norm_multiply_desc.BTensor = &inv_rms_broadcast_desc.get_tensor_desc();
  norm_multiply_desc.OutputTensor = &normalized_desc.get_tensor_desc();

  DML_OPERATOR_DESC norm_multiply_op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                             &norm_multiply_desc};
  auto* norm_multiply_op =
      dml::GetOrCreateCompiledOperatorApi(&norm_multiply_op_desc);

  dml::utils::DmlBindingArrayBundle norm_mult_inputs(
      {dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(input)),
       dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(inv_rms))});
  dml::utils::DmlBindingArrayBundle norm_mult_outputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(normalized))});
  norm_multiply_op->Execute(norm_mult_inputs.get_descs(),
                            norm_mult_outputs.get_descs());

  // Step 6: Final scale: normalized * gamma or normalized * (1 + gamma)
  const StorageView* gamma_final = &gamma;
  StorageView gamma_plus_one;

  if (_use_residual) {
    gamma_plus_one.resize_as(gamma);
    StorageView ones({(dim_t)depth}, gamma.dtype(), device_type);
    ones.fill(static_cast<T>(1.0));

    dml::utils::DmlTensorDescBundle ones_desc(ones);
    // for broadcasting gamma, it is described as [1, depth]. "ones" should also
    // be broadcastable.
    dml::utils::DmlTensorDescBundle gamma_plus_one_desc(gamma_plus_one);

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC gamma_add_desc = {};
    gamma_add_desc.ATensor = &gamma_desc.get_tensor_desc();
    gamma_add_desc.BTensor = &ones_desc.get_tensor_desc();
    gamma_add_desc.OutputTensor = &gamma_plus_one_desc.get_tensor_desc();

    DML_OPERATOR_DESC gamma_add_op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD,
                                           &gamma_add_desc};
    auto* gamma_add_op =
        dml::GetOrCreateCompiledOperatorApi(&gamma_add_op_desc);

    dml::utils::DmlBindingArrayBundle gamma_add_inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(gamma)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(ones))});
    dml::utils::DmlBindingArrayBundle gamma_add_outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(gamma_plus_one))});
    gamma_add_op->Execute(gamma_add_inputs.get_descs(),
                          gamma_add_outputs.get_descs());

    gamma_final = &gamma_plus_one;
  }

  DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC final_multiply_desc = {};
  final_multiply_desc.ATensor = &normalized_desc.get_tensor_desc();
  final_multiply_desc.BTensor =
      &gamma_desc.get_tensor_desc();  // Re-use broadcast desc
  final_multiply_desc.OutputTensor = &output_desc.get_tensor_desc();

  DML_OPERATOR_DESC final_multiply_op_desc = {
      DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &final_multiply_desc};
  auto* final_multiply_op =
      dml::GetOrCreateCompiledOperatorApi(&final_multiply_op_desc);

  dml::utils::DmlBindingArrayBundle final_mult_inputs(
      {dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(normalized)),
       dml::utils::DmlBufferBindingBundle(
           dml::utils::ResourceFromStorageView(*gamma_final))});
  dml::utils::DmlBindingArrayBundle final_mult_outputs(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(output))});
  final_multiply_op->Execute(final_mult_inputs.get_descs(),
                             final_mult_outputs.get_descs());

  device->ExecuteCommandList();
}

#define DECLARE_IMPL(T)                                \
  template void RMSNorm::compute<Device::DirectML, T>( \
      const StorageView&, const StorageView&, StorageView&) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
