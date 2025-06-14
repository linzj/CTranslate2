#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rotary.h"

#include <memory>
#include <vector>
#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

class RotaryDMLCompute {
 private:
  void multiply_tensors(const StorageView& a,
                        const StorageView& b,
                        StorageView& output) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& a_desc = op_bundle.AddInput(a);
    auto& b_desc = op_bundle.AddInput(b);
    auto& output_desc = op_bundle.AddOutput(output);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
    op_desc.ATensor = &a_desc.get_tensor_desc();
    op_desc.BTensor = &b_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(a)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(b))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  void add_tensors(const StorageView& a,
                   const StorageView& b,
                   StorageView& output) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& a_desc = op_bundle.AddInput(a);
    auto& b_desc = op_bundle.AddInput(b);
    auto& output_desc = op_bundle.AddOutput(output);

    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    op_desc.ATensor = &a_desc.get_tensor_desc();
    op_desc.BTensor = &b_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();

    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(a)),
         dml::utils::DmlBufferBindingBundle(
             dml::utils::ResourceFromStorageView(b))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  void negate_tensor(const StorageView& input, StorageView& output) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input);
    auto& output_desc = op_bundle.AddOutput(output);

    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC>();
    op_desc.InputTensor = &input_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();

    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));

    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(input))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  void copy_tensor(const StorageView& input, StorageView& output) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input);
    auto& output_desc = op_bundle.AddOutput(output);
    auto& op_desc =
        op_bundle.GetOperatorDesc<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
    op_desc.InputTensor = &input_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    op_desc.ScaleBias = nullptr;

    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(input))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output), 0,
            output.size() * output.item_size())});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

  void slice_tensor_last_dim(const StorageView& input,
                             StorageView& output,
                             dim_t start_idx,
                             dim_t count) {
    dml::utils::DmlOperatorDescBundle op_bundle;
    auto& input_desc = op_bundle.AddInput(input);
    auto& output_desc = op_bundle.AddOutput(output);

    std::vector<UINT> offsets(input.rank(), 0);
    std::vector<UINT> strides(input.rank(), 1);

    offsets.back() = start_idx;

    auto& op_desc = op_bundle.GetOperatorDesc<DML_SLICE_OPERATOR_DESC>();
    op_desc.InputTensor = &input_desc.get_tensor_desc();
    op_desc.OutputTensor = &output_desc.get_tensor_desc();
    op_desc.DimensionCount = input.rank();
    op_desc.Offsets = offsets.data();
    op_desc.Sizes = output_desc.get_sizes_vec().data();
    op_desc.Strides = strides.data();

    auto* op = dml::GetOrCreateCompiledOperatorApi(std::move(op_bundle));
    dml::utils::DmlBindingArrayBundle inputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(input))});
    dml::utils::DmlBindingArrayBundle outputs(
        {dml::utils::DmlBufferBindingBundle(
            dml::utils::ResourceFromStorageView(output))});
    op->Execute(inputs.get_descs(), outputs.get_descs());
  }

 public:
  RotaryDMLCompute(dml::Device* device, DataType ct2_data_type) {}

  void compute(const StorageView& input,
               const StorageView& sin,
               const StorageView& cos,
               StorageView& output,
               dim_t rot_ndims,
               bool interleave) {
    if (interleave) {
      compute_interleave(input, sin, cos, output, rot_ndims);
    } else {
      compute_non_interleave(input, sin, cos, output, rot_ndims);
    }
  }

 private:
  void compute_non_interleave(const StorageView& input,
                              const StorageView& sin,
                              const StorageView& cos,
                              StorageView& output,
                              dim_t rot_ndims) {
    const dim_t middle = rot_ndims / 2;

    // temp1 = x * cos
    StorageView temp1(output.shape(), output.dtype(), Device::DirectML);
    multiply_tensors(input, cos, temp1);

    StorageView rot_part(input.shape(), input.dtype(), Device::DirectML);
    if (rot_ndims > 0) {
      // create a tensor of 0s that we can add the rotated parts to.
      rot_part.zero();

      Shape rot_part_shape(input.shape());
      rot_part_shape.back() = rot_ndims;

      Shape half_rot_part_shape = rot_part_shape;
      half_rot_part_shape.back() = middle;

      StorageView x_rot_part(rot_part_shape, input.dtype(), Device::DirectML);
      slice_tensor_last_dim(input, x_rot_part, 0, rot_ndims);

      StorageView x_first_half(half_rot_part_shape, input.dtype(),
                               Device::DirectML);
      StorageView x_second_half(half_rot_part_shape, input.dtype(),
                                Device::DirectML);
      slice_tensor_last_dim(x_rot_part, x_first_half, 0, middle);
      slice_tensor_last_dim(x_rot_part, x_second_half, middle, middle);

      StorageView neg_x_second_half(x_second_half.shape(),
                                    x_second_half.dtype(), Device::DirectML);
      negate_tensor(x_second_half, neg_x_second_half);

      // The sin tensor may be larger than needed. Slice it.
      Shape sin_cos_shape = sin.shape();
      sin_cos_shape.back() = rot_ndims;
      StorageView sin_rot(sin_cos_shape, sin.dtype(), Device::DirectML);
      slice_tensor_last_dim(sin, sin_rot, 0, rot_ndims);

      StorageView sin_first_half(half_rot_part_shape, sin.dtype(),
                                 Device::DirectML);
      StorageView sin_second_half(half_rot_part_shape, sin.dtype(),
                                  Device::DirectML);
      slice_tensor_last_dim(sin_rot, sin_first_half, 0, middle);
      slice_tensor_last_dim(sin_rot, sin_second_half, middle, middle);

      // -x2 * sin1
      StorageView first_half_rotated(x_first_half.shape(), x_first_half.dtype(),
                                     Device::DirectML);
      multiply_tensors(neg_x_second_half, sin_first_half, first_half_rotated);

      // x1 * sin2
      StorageView second_half_rotated(x_second_half.shape(),
                                      x_second_half.dtype(), Device::DirectML);
      multiply_tensors(x_first_half, sin_second_half, second_half_rotated);

      // Add first rotated part
      StorageView rot_part_first_half_view(half_rot_part_shape,
                                           rot_part.dtype(), Device::DirectML);
      slice_tensor_last_dim(rot_part, rot_part_first_half_view, 0,
                            middle);  // this is a view to scatter to
      add_tensors(rot_part_first_half_view, first_half_rotated,
                  rot_part_first_half_view);

      // Add second rotated part
      StorageView rot_part_second_half_view(half_rot_part_shape,
                                            rot_part.dtype(), Device::DirectML);
      slice_tensor_last_dim(rot_part, rot_part_second_half_view, middle,
                            middle);  // this is a view to scatter to
      add_tensors(rot_part_second_half_view, second_half_rotated,
                  rot_part_second_half_view);
    }

    add_tensors(temp1, rot_part, temp1);

    output.copy_from(temp1);
  }

  void compute_interleave(const StorageView& input,
                          const StorageView& sin,
                          const StorageView& cos,
                          StorageView& output,
                          dim_t rot_ndims) {
    // NOT IMPLEMENTED. Placeholder to compile.
    output.copy_from(input);
  }
};

template <Device D, typename T>
void Rotary::compute(const StorageView& input,
                     const StorageView& sin,
                     const StorageView& cos,
                     StorageView& output,
                     bool is_transposed) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* dml_device_wrapper = dml::get_device();
  if (!dml_device_wrapper) {
    throw std::runtime_error("DirectML device not available");
  }

  const dim_t feature_depth = input.dim(-1);
  const dim_t rot_ndims_param = _ndims == 0 ? feature_depth : _ndims;

  RotaryDMLCompute compute_kernel(dml_device_wrapper, input.dtype());

  compute_kernel.compute(input, sin, cos, output, rot_ndims_param, _interleave);
}

#define DECLARE_IMPL(T)                                           \
  template void Rotary::compute<Device::DirectML, T>(             \
      const StorageView&, const StorageView&, const StorageView&, \
      StorageView&, bool) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML