#pragma once

#include "ctranslate2/ops/op.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
namespace ops {

class MultiHeadAttention : public Op {
 public:
  MultiHeadAttention(const float scale,
                     const float mask_filter_value,
                     const dim_t head_count,
                     const bool is_causal);

  void operator()(const StorageView& queries,
                  const StorageView& keys,
                  const StorageView& values,
                  const StorageView* bias,
                  const StorageView* mask,
                  const StorageView* relative_position_bias,
                  StorageView& output,
                  StorageView* present_keys = nullptr,
                  StorageView* present_values = nullptr) const;

 private:
  template <Device D, typename T>
  void compute(const StorageView& queries,
               const StorageView& keys,
               const StorageView& values,
               const StorageView* bias,
               const StorageView* mask,
               const StorageView* relative_position_bias,
               StorageView& output,
               StorageView* present_keys = nullptr,
               StorageView* present_values = nullptr) const;

  float _scale;
  float _mask_filter_value;
  dim_t _head_count;
  bool _is_causal;
};

}  // namespace ops
}  // namespace ctranslate2
