#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rms_norm.h"

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

  const dim_t depth = input.dim(-1);
  const dim_t batch_size = input.size() / depth;

  const DML_TENSOR_DATA_TYPE data_type =
      dml::utils::get_dml_data_type(input.dtype());
  const auto device_type = Device::DirectML;

  // Step 1: Compute input_squared = input * input
  StorageView input_squared(input.shape(), input.dtype(), device_type);
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input);
    auto& output_desc = op_bundle.AddOutput(input_squared);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
    op_desc.ATensor = &input_desc.get_tensor_desc();
    op_desc.BTensor = &input_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle mult_inputs{
        {{dml::utils::ResourceFromStorageView(input), 0, 0},
         {dml::utils::ResourceFromStorageView(input), 0, 0}}};
    dml::utils::DmlBindingArrayBundle mult_outputs{
        {{dml::utils::ResourceFromStorageView(input_squared), 0, 0}}};
    op->Execute(mult_inputs, mult_outputs);
  }

  // Step 2: Compute mean_squared = reduce_mean(input_squared, axis=1)
  StorageView mean_squared({batch_size, 1}, input.dtype(), device_type);
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input_squared);
    auto& output_desc = op_bundle.AddOutput(mean_squared);
    auto& op_desc = op_bundle.GetOperatorDesc<DML_REDUCE_OPERATOR_DESC>();
    const uint32_t reduce_axes[] = {1};  // reduce on depth
    op_desc.Function = DML_REDUCE_FUNCTION_AVERAGE;
    op_desc.InputTensor = &input_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    op_desc.AxisCount = 1;
    op_desc.Axes = reduce_axes;
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle reduce_inputs{
        {dml::utils::ResourceFromStorageView(input_squared), 0, 0}};
    dml::utils::DmlBindingArrayBundle reduce_outputs{
        {dml::utils::ResourceFromStorageView(mean_squared), 0, 0}};
    op->Execute(reduce_inputs, reduce_outputs);
  }

  // Step 3: Add epsilon
  StorageView epsilon_sv({batch_size, 1}, input.dtype(), device_type);
  epsilon_sv.fill(static_cast<T>(_epsilon));
  StorageView variance(mean_squared.shape(), input.dtype(), device_type);
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_a_desc = op_bundle.AddInput(mean_squared);
    auto& input_b_desc = op_bundle.AddInput(epsilon_sv);
    auto& output_desc = op_bundle.AddOutput(variance);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    op_desc.ATensor = &input_a_desc.get_tensor_desc();
    op_desc.BTensor = &input_b_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle add_inputs{
        {dml::utils::ResourceFromStorageView(mean_squared), 0, 0},
        {dml::utils::ResourceFromStorageView(epsilon_sv), 0, 0}};
    dml::utils::DmlBindingArrayBundle add_outputs{
        {dml::utils::ResourceFromStorageView(variance), 0, 0}};
    op->Execute(add_inputs, add_outputs);
  }

  // Step 4: Compute inv_rms = 1/sqrt(variance)
  // Use variance buffer for sqrt output then recip output (in-place)
  StorageView& inv_rms = variance;
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& desc = op_bundle.AddInput(inv_rms);
    op_bundle.AddOutput(inv_rms);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>();
    op_desc.InputTensor = &desc.get_tensor_desc();
    op_desc.OutputTensor = &desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle sqrt_bindings{
        {dml::utils::ResourceFromStorageView(inv_rms), 0, 0}};
    op->Execute(sqrt_bindings, sqrt_bindings);
  }
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& desc = op_bundle.AddInput(inv_rms);
    op_bundle.AddOutput(inv_rms);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>();
    op_desc.InputTensor = &desc.get_tensor_desc();
    op_desc.OutputTensor = &desc.get_tensor_desc();  // In-place
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle recip_bindings{
        {dml::utils::ResourceFromStorageView(inv_rms), 0, 0}};
    op->Execute(recip_bindings, recip_bindings);
  }

  // Step 5: Normalize input = input * inv_rms
  StorageView normalized(input.shape(), input.dtype(), device_type);
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input);
    std::vector<UINT> broadcast_dims = {(UINT)batch_size, (UINT)depth};
    std::vector<UINT> inv_rms_broadcast_strides = {
        1, 0};  // Stride 0 for broadcasting
    auto& inv_rms_broadcast_desc = op_bundle.AddInput(
        data_type, broadcast_dims, &inv_rms_broadcast_strides,
        inv_rms.reserved_memory());
    auto& output_desc = op_bundle.AddOutput(normalized);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
    op_desc.ATensor = &input_desc.get_tensor_desc();
    op_desc.BTensor = &inv_rms_broadcast_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle norm_mult_inputs{
        {dml::utils::ResourceFromStorageView(input), 0, 0},
        {dml::utils::ResourceFromStorageView(inv_rms), 0, 0}};
    dml::utils::DmlBindingArrayBundle norm_mult_outputs{
        {dml::utils::ResourceFromStorageView(normalized), 0, 0}};
    op->Execute(norm_mult_inputs, norm_mult_outputs);
  }

  // Step 6: Final scale: normalized * gamma or normalized * (1 + gamma)
  const StorageView* gamma_final = &gamma;
  StorageView gamma_plus_one;

  if (_use_residual) {
    gamma_plus_one.resize_as(gamma);
    StorageView ones({(dim_t)depth}, gamma.dtype(), device_type);
    ones.fill(static_cast<T>(1.0));
    dml::utils::DmlOperatorDescBundle op_bundle;
    std::vector<UINT> gamma_dims = {1, (UINT)depth};
    auto& gamma_desc = op_bundle.AddInput(data_type, gamma_dims, nullptr,
                                          gamma.reserved_memory());
    auto& ones_desc = op_bundle.AddInput(ones);
    auto& output_desc = op_bundle.AddOutput(gamma_plus_one);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    op_desc.ATensor = &gamma_desc.get_tensor_desc();
    op_desc.BTensor = &ones_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle gamma_add_inputs{
        {dml::utils::ResourceFromStorageView(gamma), 0, 0},
        {dml::utils::ResourceFromStorageView(ones), 0, 0}};
    dml::utils::DmlBindingArrayBundle gamma_add_outputs{
        {dml::utils::ResourceFromStorageView(gamma_plus_one), 0, 0}};
    op->Execute(gamma_add_inputs, gamma_add_outputs);
    gamma_final = &gamma_plus_one;
  }
  {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& norm_desc = op_bundle.AddInput(normalized);
    std::vector<UINT> gamma_dims = {1, (UINT)depth};
    auto& gamma_desc = op_bundle.AddInput(data_type, gamma_dims, nullptr, 0);
    auto& out_desc = op_bundle.AddOutput(output);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
    op_desc.ATensor = &norm_desc.get_tensor_desc();
    op_desc.BTensor = &gamma_desc.get_tensor_desc();
    op_desc.OutputTensor = &out_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle final_mult_inputs{
        {dml::utils::ResourceFromStorageView(normalized), 0, 0},
        {dml::utils::ResourceFromStorageView(*gamma_final), 0, 0}};
    dml::utils::DmlBindingArrayBundle final_mult_outputs{
        {dml::utils::ResourceFromStorageView(output), 0, 0}};
    op->Execute(final_mult_inputs, final_mult_outputs);
  }
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
