#pragma once

#include "ctranslate2/allocator.h"
namespace ctranslate2 {
namespace dml {

class DMLAllocator : public Allocator {
 public:
  DMLAllocator();

  ~DMLAllocator() override;

  void* allocate(size_t size, int device_index) override;

  void free(void* ptr, int device_index) override;

};
}  // namespace dml
}  // namespace ctranslate2