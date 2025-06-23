#include "ctranslate2/ops/multi_head_attention.h"
#ifdef CT2_WITH_DIRECTML
#include <DirectML.h>
#include "dml/backend_dml.h"
#endif

namespace ctranslate2 {
namespace ops {

MultiHeadAttention::MultiHeadAttention(const float scale,
                                       const float mask_filter_value,
                                       const dim_t head_count,
                                       const bool is_causal)
    : _scale(scale),
      _mask_filter_value(mask_filter_value),
      _head_count(head_count),
      _is_causal(is_causal) {}

void MultiHeadAttention::operator()(const StorageView& queries,
                                    const StorageView& keys,
                                    const StorageView& values,
                                    const StorageView* bias,
                                    const StorageView* mask,
                                    const StorageView* relative_position_bias,
                                    const StorageView* past_keys,
                                    const StorageView* past_values,
                                    StorageView& output,
                                    StorageView* present_keys,
                                    StorageView* present_values) const {
  const Device device = queries.device();
  const DataType dtype = queries.dtype();
  output.resize(queries.shape());

  switch (device) {
    case Device::DirectML: {
#ifdef CT2_WITH_DIRECTML
      const auto& dxdevice = ctranslate2::dml::get_device();
      if (dxdevice->GetDmlFeatureLevel() >= DML_FEATURE_LEVEL_6_1) {
        compute<Device::DirectML, float>(
            queries, keys, values, bias, mask, relative_position_bias,
            past_keys, past_values, output, present_keys, present_values);
        break;
      }
#endif
      break;
    }
    default:
      THROW_INVALID_ARGUMENT(
          "MultiHeadAttention is not implemented for this device");
  }
}

}  // namespace ops
}  // namespace ctranslate2
