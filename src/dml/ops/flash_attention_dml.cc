#include "ctranslate2/ops/flash_attention.h"
#if defined(CT2_WITH_DIRECTML)

#if defined(CT2_WITH_FLASH_ATTN)

#endif

namespace ctranslate2 {
namespace ops {
template <>
void FlashAttention::compute<Device::DirectML>(StorageView& queries,
                                               StorageView& keys,
                                               StorageView& values,
                                               StorageView& output,
                                               StorageView* cached_keys,
                                               StorageView* cached_values,
                                               StorageView* attention,
                                               bool return_normalized_attention,
                                               StorageView* rotary_cos,
                                               StorageView* rotary_sin,
                                               const bool rotary_interleave,
                                               StorageView* alibi,
                                               dim_t offset) const {
  throw std::runtime_error("Flash attention 2 is not supported");
}

}  // namespace ops
}  // namespace ctranslate2
#endif