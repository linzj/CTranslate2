#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/awq/dequantize_awq.h"

#include <array>
#include "dml/backend_dml.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

namespace ctranslate2 {
namespace ops {

template <Device D, typename InT, typename OutT>
void DequantizeAwq::dequantize(const StorageView& input,
                               const StorageView& scale,
                               const StorageView& zero,
                               StorageView& output) const {
  // Get DML device and related objects
  auto* device = dml::get_device();
  auto* dml_device = dml::get_dml_device();

  // Calculate dimensions
  dim_t in_c = input.rank() == 2 ? input.dim(0) : input.dim(1);
  dim_t qout_c = input.rank() == 2 ? input.dim(1) : input.dim(2);
  int num_experts = input.rank() == 2 ? 1 : input.dim(0);
  int out_c = qout_c * 8;
  int G = in_c / (input.rank() == 2 ? scale.dim(0) : scale.dim(1));

  // Resize output tensor
  if (num_experts == 1) {
    output.resize({in_c, out_c});
  } else {
    output.resize({num_experts, in_c, out_c});
  }

  // Create tensor descriptors
  // Input tensor (quantized weights)
  std::vector<UINT> inputSizes, inputStrides;
  if (num_experts == 1) {
    inputSizes = {static_cast<UINT>(in_c), static_cast<UINT>(qout_c)};
    inputStrides = {static_cast<UINT>(qout_c), 1};
  } else {
    inputSizes = {static_cast<UINT>(num_experts), static_cast<UINT>(in_c),
                  static_cast<UINT>(qout_c)};
    inputStrides = {static_cast<UINT>(in_c * qout_c), static_cast<UINT>(qout_c),
                    1};
  }

  DML_BUFFER_TENSOR_DESC inputBufferDesc = {
      DML_TENSOR_DATA_TYPE_INT32,
      DML_TENSOR_FLAG_NONE,
      static_cast<UINT>(inputSizes.size()),
      inputSizes.data(),
      inputStrides.data(),
      input.size() * sizeof(InT),
      0};
  DML_TENSOR_DESC inputTensorDesc = {DML_TENSOR_TYPE_BUFFER, &inputBufferDesc};

  // Scale tensor - needs broadcasting to match output dimensions
  std::vector<UINT> scaleSizes, scaleStrides;
  if (num_experts == 1) {
    // Broadcast from [in_c/G, out_c] to [in_c, out_c]
    scaleSizes = {static_cast<UINT>(in_c), static_cast<UINT>(out_c)};
    scaleStrides = {static_cast<UINT>(out_c) * G, 1};  // Repeat every G rows
  } else {
    // Broadcast from [num_experts, in_c/G, out_c] to [num_experts, in_c, out_c]
    scaleSizes = {static_cast<UINT>(num_experts), static_cast<UINT>(in_c),
                  static_cast<UINT>(out_c)};
    scaleStrides = {static_cast<UINT>((in_c / G) * out_c),
                    static_cast<UINT>(out_c) * G, 1};
  }

  DML_BUFFER_TENSOR_DESC scaleBufferDesc = {
      DML_TENSOR_DATA_TYPE_FLOAT16,
      DML_TENSOR_FLAG_NONE,
      static_cast<UINT>(scaleSizes.size()),
      scaleSizes.data(),
      scaleStrides.data(),
      scale.size() * sizeof(OutT),
      0};
  DML_TENSOR_DESC scaleTensorDesc = {DML_TENSOR_TYPE_BUFFER, &scaleBufferDesc};

  // Zero tensor - needs broadcasting to match output dimensions
  std::vector<UINT> zeroSizes, zeroStrides;
  if (num_experts == 1) {
    // Broadcast from [in_c/G, qout_c] to [in_c, out_c] with 8x expansion
    zeroSizes = {static_cast<UINT>(in_c), static_cast<UINT>(out_c)};
    zeroStrides = {static_cast<UINT>(qout_c) * G,
                   8};  // Repeat every G rows, every 8 cols
  } else {
    // Broadcast from [num_experts, in_c/G, qout_c] to [num_experts, in_c,
    // out_c]
    zeroSizes = {static_cast<UINT>(num_experts), static_cast<UINT>(in_c),
                 static_cast<UINT>(out_c)};
    zeroStrides = {static_cast<UINT>((in_c / G) * qout_c),
                   static_cast<UINT>(qout_c) * G, 8};
  }

  DML_BUFFER_TENSOR_DESC zeroBufferDesc = {DML_TENSOR_DATA_TYPE_INT32,
                                           DML_TENSOR_FLAG_NONE,
                                           static_cast<UINT>(zeroSizes.size()),
                                           zeroSizes.data(),
                                           zeroStrides.data(),
                                           zero.size() * sizeof(InT),
                                           0};
  DML_TENSOR_DESC zeroTensorDesc = {DML_TENSOR_TYPE_BUFFER, &zeroBufferDesc};

  // Output tensor
  std::vector<UINT> outputSizes, outputStrides;
  if (num_experts == 1) {
    outputSizes = {static_cast<UINT>(in_c), static_cast<UINT>(out_c)};
    outputStrides = {static_cast<UINT>(out_c), 1};
  } else {
    outputSizes = {static_cast<UINT>(num_experts), static_cast<UINT>(in_c),
                   static_cast<UINT>(out_c)};
    outputStrides = {static_cast<UINT>(in_c * out_c), static_cast<UINT>(out_c),
                     1};
  }

  DML_BUFFER_TENSOR_DESC outputBufferDesc = {
      DML_TENSOR_DATA_TYPE_FLOAT16,
      DML_TENSOR_FLAG_NONE,
      static_cast<UINT>(outputSizes.size()),
      outputSizes.data(),
      outputStrides.data(),
      output.size() * sizeof(OutT),
      0};
  DML_TENSOR_DESC outputTensorDesc = {DML_TENSOR_TYPE_BUFFER,
                                      &outputBufferDesc};

  // Create dequantize linear operator
  DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequantizeDesc = {
      &inputTensorDesc, &scaleTensorDesc, &zeroTensorDesc, &outputTensorDesc};

  DML_OPERATOR_DESC opDesc = {DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR,
                              &dequantizeDesc};

  // Get compiled operator from cache
  auto compiledOp = dml::GetOrCreateCompiledOperatorApi(&opDesc);

  // Create input bindings
  std::array<DML_BUFFER_BINDING, 3> inputBindings = {
      {{reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer())), 0,
        input.size() * sizeof(InT)},
       {reinterpret_cast<ID3D12Resource*>(const_cast<void*>(scale.buffer())), 0,
        scale.size() * sizeof(OutT)},
       {reinterpret_cast<ID3D12Resource*>(const_cast<void*>(zero.buffer())), 0,
        zero.size() * sizeof(InT)}}};

  std::vector<DML_BINDING_DESC> inputBindingDescs = {
      {{DML_BINDING_TYPE_BUFFER, &inputBindings[0]},
       {DML_BINDING_TYPE_BUFFER, &inputBindings[1]},
       {DML_BINDING_TYPE_BUFFER, &inputBindings[2]}}};

  // Create output binding
  DML_BUFFER_BINDING outputBinding = {
      reinterpret_cast<ID3D12Resource*>(output.buffer()), 0,
      output.size() * sizeof(OutT)};
  DML_BINDING_DESC outputBindingDesc = {DML_BINDING_TYPE_BUFFER,
                                        &outputBinding};

  compiledOp->Execute(inputBindingDescs, {outputBindingDesc});
}

#define DECLARE_IMPL(T)                                              \
  template void DequantizeAwq::dequantize<Device::DirectML, int, T>( \
      const StorageView&, const StorageView&, const StorageView&,    \
      StorageView&) const;

DECLARE_IMPL(float16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
