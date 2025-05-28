#ifdef CT2_WITH_DIRECTML

#include <ctranslate2/ops/awq/gemm.h>

namespace ctranslate2 {
namespace ops {
template <Device D, typename In, typename Out>
void GemmAwq::compute(const StorageView& a,
                      const StorageView& b,
                      const StorageView& scale,
                      const StorageView& zero,
                      StorageView& c) const {
  throw std::runtime_error(
      "GemmAwq::compute is not implemented for this device and data type.");
}

#define DECLARE_IMPL(T)                                           \
  template void GemmAwq::compute<Device::DirectML, T, int>(       \
      const StorageView&, const StorageView&, const StorageView&, \
      const StorageView&, StorageView&) const;

DECLARE_IMPL(float16_t)
}  // namespace ops

}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
