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

  if constexpr (std::is_same_v<T, ctranslate2::bfloat16_t>) {
    throw std::invalid_argument(
        "DirectML does not support bfloat16 for Tile operation");
  }

  dml::utils::DmlOperatorDescBundle op_desc;

  std::vector<UINT> input_dims = {static_cast<UINT>(outer_size),
                                  static_cast<UINT>(inner_size)};
  const auto& input_desc = op_desc.AddInput(input.dtype(), input_dims, nullptr,
                                            input.size() * sizeof(T));

  std::vector<UINT> output_dims = {static_cast<UINT>(outer_size),
                                   static_cast<UINT>(inner_size * _num_tiles)};
  const auto& output_desc = op_desc.AddOutput(
      output.dtype(), output_dims, nullptr, output.size() * sizeof(T));

  UINT repeats[2] = {1, static_cast<UINT>(_num_tiles)};

  auto& tile_desc = op_desc.GetOperatorDesc<DML_TILE_OPERATOR_DESC>();
  tile_desc.InputTensor = &input_desc.get_tensor_desc();
  tile_desc.OutputTensor = &output_desc.get_tensor_desc();
  tile_desc.RepeatsCount = 2;
  tile_desc.Repeats = repeats;

  auto compiled_op = dml::GetOrCreateCompiledOperatorApi(std::move(op_desc));

  // Bindings
  dml::utils::DmlBindingArrayBundle input_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(input))});

  dml::utils::DmlBindingArrayBundle output_bindings(
      {dml::utils::DmlBufferBindingBundle(
          dml::utils::ResourceFromStorageView(output))});

  compiled_op->Execute(input_bindings.get_descs(), output_bindings.get_descs());
}

#define DECLARE_IMPL(T)                                 \
  template void Tile::compute<Device::DirectML, T>(     \
      const StorageView& input, const dim_t outer_size, \
      const dim_t inner_size, StorageView& output) const;

DECLARE_ALL_TYPES(DECLARE_IMPL)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML