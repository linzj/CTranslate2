#pragma once
#ifdef CT2_WITH_DIRECTML
#include "dxdevice.h"  // Added

namespace ctranslate2 {
namespace dml {

class Device;  // Forward declaration

bool has_directml_device();
void initialize_directml();
void release_directml();  // Added

Device* get_device();

class ScopedGraphRecording {
 public:
  ScopedGraphRecording() {
    // dml::get_device()->BeginGraphRecording();
  }
  ~ScopedGraphRecording() {
    // dml::get_device()->EndGraphRecording();
  }
};

inline void split_graph_recording() {
  Device* device = dml::get_device();
  if (device) {
    device->SplitGraphRecording();
  }
}
}  // namespace dml
}  // namespace ctranslate2

#endif