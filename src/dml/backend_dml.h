#pragma once
#ifdef CT2_WITH_DIRECTML
#include <DirectML.h>
#include <d3d12.h>

namespace ctranslate2 {
namespace dml {
bool has_directml_device();
void initialize_directml();
ID3D12Device* get_d3d12_device();

IDMLDevice* get_dml_device();

ID3D12CommandQueue* get_command_queue();
// We'll define dml::synchronize_device and dml::synchronize_stream here in the
// next step or inside dml/backend_dml.cc once core DML context/queue objects
// are accessible.
}  // namespace dml
}  // namespace ctranslate2

#endif