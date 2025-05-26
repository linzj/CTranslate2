#pragma once
#ifdef CT2_WITH_DIRECTML
#include <memory>      // Added for std::unique_ptr
#include "dxdevice.h"  // Added

namespace ctranslate2 {
namespace dml {

class Device;  // Forward declaration

bool has_directml_device();
void initialize_directml();
void release_directml();  // Added

Device* get_device();
IDMLDevice1* get_dml_device();
}  // namespace dml
}  // namespace ctranslate2

#endif