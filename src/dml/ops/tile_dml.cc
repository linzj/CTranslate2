#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/tile.h"

#include "dml/backend_dml.h"
#include "dml/dml_utils.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"
#include "type_dispatch.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename T>
void Tile::compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t inner_size,
                   StorageView& output) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  if constexpr (std::is_same_v<T, ctranslate2::bfloat16_t>) {
    throw std::invalid_argument(
        "DirectML does not support bfloat16 for Tile operation");
  }

  std::vector<UINT> input_dims_vec = {static_cast<UINT>(outer_size),
                                      static_cast<UINT>(inner_size)};
  dml::utils::DmlTensorDescBundle input_desc_bundle(
      input.dtype(), input_dims_vec,
      nullptr,                    // Default strides
      input.size() * sizeof(T));  // Corrected: removed 5th arg
  const DML_TENSOR_DESC& dml_input_desc_ref =
      input_desc_bundle.get_tensor_desc();

  std::vector<UINT> output_dims_vec = {
      static_cast<UINT>(outer_size),
      static_cast<UINT>(inner_size * _num_tiles)};
  dml::utils::DmlTensorDescBundle output_desc_bundle(
      output.dtype(), output_dims_vec,
      nullptr,                     // Default strides
      output.size() * sizeof(T));  // Corrected: removed 5th arg
  const DML_TENSOR_DESC& dml_output_desc_ref =
      output_desc_bundle.get_tensor_desc();

  UINT repeats[2] = {1, static_cast<UINT>(_num_tiles)};

  DML_TILE_OPERATOR_DESC tile_desc = {};
  tile_desc.InputTensor = &dml_input_desc_ref;
  tile_desc.OutputTensor = &dml_output_desc_ref;
  tile_desc.RepeatsCount = 2;
  tile_desc.Repeats = repeats;

  DML_OPERATOR_DESC op_desc = {};
  op_desc.Type = DML_OPERATOR_TILE;
  op_desc.Desc = &tile_desc;

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

  DML_BUFFER_BINDING input_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())),
          0, input_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC input_binding_desc_for_op =
      dml::utils::create_binding_desc(&input_buffer_binding_storage);

  DML_BUFFER_BINDING output_buffer_binding_storage =
      dml::utils::create_buffer_binding(
          reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
          output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
  DML_BINDING_DESC output_binding_desc_for_op =
      dml::utils::create_binding_desc(&output_buffer_binding_storage);

  std::vector<DML_BINDING_DESC> bindings_for_op_inputs = {
      input_binding_desc_for_op};
  std::vector<DML_BINDING_DESC> bindings_for_op_outputs = {
      output_binding_desc_for_op};

  compiled_op->Execute(bindings_for_op_inputs, bindings_for_op_outputs);
}

#define DECLARE_IMPL(T)                                 \
  template void Tile::compute<Device::DirectML, T>(     \
      const StorageView& input, const dim_t outer_size, \
      const dim_t inner_size, StorageView& output) const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML