#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/conv1d.h"  // Class declaration for Conv1D

#include "dml/backend_dml.h"
#include "dml/dxdevice.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
namespace ops {

namespace {
// Helper function to convert StorageView shape to DML compatible UINT
// dimensions (NCHW) For Conv1D, we map (Batch, Channels, Width) to (Batch,
// Channels, 1, Width) for input/output Weight: (OutChannels, InChannels/Groups,
// KernelW) -> (OutChannels, InChannels/Groups, 1, KernelW) Bias: (OutChannels)
// -> (1, OutChannels, 1, 1) for broadcasting
static std::vector<UINT> get_dml_tensor_shape_4d(
    const StorageView& tensor,
    bool is_filter_or_bias = false) {
  const auto& sv_shape = tensor.shape();
  std::vector<UINT> dml_dims;

  if (is_filter_or_bias) {
    if (tensor.rank() == 3) {  // Filter/Weight tensor
      // (OutChannels, InChannelsPerGroup, KernelWidth) -> (OutChannels,
      // InChannelsPerGroup, 1, KernelWidth)
      dml_dims = {static_cast<UINT>(sv_shape[0]),
                  static_cast<UINT>(sv_shape[1]), 1,
                  static_cast<UINT>(sv_shape[2])};
    } else if (tensor.rank() == 1) {  // Bias tensor
      // (OutChannels) -> (1, OutChannels, 1, 1)
      dml_dims = {1, static_cast<UINT>(sv_shape[0]), 1, 1};
    } else {
      throw std::runtime_error(
          "Conv1D DML: Unsupported rank for filter or bias tensor. Expected "
          "rank 3 for filter, 1 for bias.");
    }
  } else {  // Input or Output tensor
    if (tensor.rank() == 3) {
      // (Batch, Channels, Width) -> (Batch, Channels, 1, Width)
      dml_dims = {static_cast<UINT>(sv_shape[0]),
                  static_cast<UINT>(sv_shape[1]), 1,
                  static_cast<UINT>(sv_shape[2])};
    } else {
      throw std::runtime_error(
          "Conv1D DML: Unsupported rank for input or output tensor. Expected "
          "rank 3.");
    }
  }
  return dml_dims;
}

// Maps ctranslate2::DataType to DML_TENSOR_DATA_TYPE
inline DML_TENSOR_DATA_TYPE get_dml_data_type(DataType type) {
  switch (type) {
    case DataType::FLOAT32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case DataType::FLOAT16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    // TODO: Add other supported types like INT32 if necessary for other ops
    // case DataType::INT32: return DML_TENSOR_DATA_TYPE_INT32;
    default:
      throw std::runtime_error(
          "Unsupported ctranslate2::DataType for DirectML: " +
          dtype_name(type));
  }
}

// Calculates element size in bytes for DML data types
inline UINT get_dml_element_size_in_bytes(DML_TENSOR_DATA_TYPE data_type) {
  switch (data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
      return 2;
    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
      return 1;
    // Add other types as needed
    default:
      throw std::runtime_error(
          "Cannot determine element size for unknown DML_TENSOR_DATA_TYPE");
  }
}

// Helper to create a DML_BUFFER_TENSOR_DESC
// Sizes are DML dimensions (e.g., NCHW)
inline DML_BUFFER_TENSOR_DESC create_dml_buffer_tensor_desc(
    DML_TENSOR_DATA_TYPE data_type,
    const std::vector<UINT>& sizes,
    const std::vector<UINT>* strides = nullptr,
    UINT64 total_tensor_size_in_bytes = 0,
    DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE) {
  DML_BUFFER_TENSOR_DESC desc{};
  desc.DataType = data_type;
  desc.Flags = flags;
  desc.DimensionCount = static_cast<UINT>(sizes.size());
  desc.Sizes = sizes.data();  // sizes.data() must remain valid for the lifetime
                              // of desc or until DML consumes it

  if (strides) {
    desc.Strides = strides->data();  // strides->data() must also remain valid
  } else {
    desc.Strides = nullptr;  // DML will compute packed strides
  }

  if (total_tensor_size_in_bytes == 0) {
    UINT element_size_in_bytes = get_dml_element_size_in_bytes(data_type);
    UINT64 num_elements = 1;
    if (sizes.empty()) {  // Should not happen for valid DML tensors
      num_elements = 0;
    } else {
      for (UINT size : sizes) {
        if (size == 0) {  // A dimension of size 0 means 0 elements
          num_elements = 0;
          break;
        }
        num_elements *= size;
      }
    }
    desc.TotalTensorSizeInBytes = num_elements * element_size_in_bytes;
  } else {
    desc.TotalTensorSizeInBytes = total_tensor_size_in_bytes;
  }
  // desc.GuaranteedBaseOffsetAlignment can be set if specific alignment is
  // needed, otherwise 0 lets DML use default
  // (DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT which is 16).
  desc.GuaranteedBaseOffsetAlignment = 0;

  return desc;
}

// Helper to wrap DML_BUFFER_TENSOR_DESC into DML_TENSOR_DESC
inline DML_TENSOR_DESC wrap_dml_buffer_tensor_desc(
    const DML_BUFFER_TENSOR_DESC& buffer_desc) {
  DML_TENSOR_DESC desc{};
  desc.Type = DML_TENSOR_TYPE_BUFFER;
  desc.Desc = &buffer_desc;  // buffer_desc must remain valid
  return desc;
}

// Converts ctranslate2::Shape to std::vector<UINT> for DML
inline std::vector<UINT> to_dml_dims(const Shape& shape) {
  std::vector<UINT> dml_dims;
  dml_dims.reserve(shape.size());
  for (dim_t d : shape) {
    dml_dims.push_back(static_cast<UINT>(d));
  }
  return dml_dims;
}
}  // namespace

// Private helper method to encapsulate DML-specific logic for Conv1D

template <Device D, typename T>
void Conv1D::compute(const StorageView& input,
                     const StorageView& weight,
                     const StorageView* bias,
                     StorageView& output,
                     const StorageView* qscale) const {
  if (qscale)
    throw std::runtime_error(
        "Quantization is not supported in this Conv1D implementation");
  dml::Device* dml_device_handle =
      dml::get_device();  // ctranslate2::dml::Device wrapper
  // IDMLDevice1* dml_device = dml_device_handle->DML(); // Raw DML device, if
  // needed directly

  DML_TENSOR_DATA_TYPE dml_data_type = get_dml_data_type(input.dtype());

  // 1. Prepare Tensor Descriptors (input, weight, bias, output)
  // These descriptors and the dimension vectors must remain in scope until the
  // operator is created/compiled.

  std::vector<UINT> input_dml_dims = get_dml_tensor_shape_4d(input);
  DML_BUFFER_TENSOR_DESC input_buffer_desc =
      create_dml_buffer_tensor_desc(dml_data_type, input_dml_dims);
  DML_TENSOR_DESC input_tensor_desc =
      wrap_dml_buffer_tensor_desc(input_buffer_desc);

  std::vector<UINT> weight_dml_dims = get_dml_tensor_shape_4d(weight, true);
  DML_BUFFER_TENSOR_DESC weight_buffer_desc =
      create_dml_buffer_tensor_desc(dml_data_type, weight_dml_dims);
  DML_TENSOR_DESC weight_tensor_desc =
      wrap_dml_buffer_tensor_desc(weight_buffer_desc);

  std::vector<UINT> output_dml_dims = get_dml_tensor_shape_4d(output);
  DML_BUFFER_TENSOR_DESC output_buffer_desc =
      create_dml_buffer_tensor_desc(dml_data_type, output_dml_dims);
  DML_TENSOR_DESC output_tensor_desc =
      wrap_dml_buffer_tensor_desc(output_buffer_desc);

  // Optional Bias Tensor
  DML_TENSOR_DESC bias_tensor_desc_storage{};         // Zero-initialized
  DML_BUFFER_TENSOR_DESC bias_buffer_desc_storage{};  // Zero-initialized
  std::vector<UINT> bias_dml_dims;                    // Keep vector in scope
  const DML_TENSOR_DESC* bias_tensor_desc_ptr = nullptr;

  if (bias) {
    bias_dml_dims = get_dml_tensor_shape_4d(*bias, true);
    bias_buffer_desc_storage =
        create_dml_buffer_tensor_desc(dml_data_type, bias_dml_dims);
    bias_tensor_desc_storage =
        wrap_dml_buffer_tensor_desc(bias_buffer_desc_storage);
    bias_tensor_desc_ptr = &bias_tensor_desc_storage;
  }

  // 2. Create Convolution Operator Descriptor
  DML_CONVOLUTION_OPERATOR_DESC conv_op_desc{};
  conv_op_desc.InputTensor = &input_tensor_desc;
  conv_op_desc.FilterTensor = &weight_tensor_desc;
  conv_op_desc.BiasTensor = bias_tensor_desc_ptr;  // Null if no bias
  conv_op_desc.OutputTensor = &output_tensor_desc;

  conv_op_desc.Mode =
      DML_CONVOLUTION_MODE_CROSS_CORRELATION;  // Standard convolution behavior
  conv_op_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
  conv_op_desc.DimensionCount =
      2;  // Number of spatial dimensions (Height, Width). For 1D Conv, H=1.

  // Strides, Dilations, Paddings are for {H, W} dimensions.
  // For 1D convolution, H-dimension parameters are 1 (stride/dilation) or 0
  // (padding).
  UINT dml_strides[] = {1, static_cast<UINT>(_stride)};
  conv_op_desc.Strides = dml_strides;

  UINT dml_dilations[] = {1, static_cast<UINT>(_dilation)};
  conv_op_desc.Dilations = dml_dilations;

  // _padding is assumed to be symmetric padding for the width/length dimension.
  UINT dml_start_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.StartPadding = dml_start_padding;

  UINT dml_end_padding[] = {0, static_cast<UINT>(_padding)};
  conv_op_desc.EndPadding = dml_end_padding;

  // OutputPadding is mainly for deconvolution/transposed convolution. Usually
  // {0,0} for standard convolution.
  UINT dml_output_padding[] = {0, 0};
  conv_op_desc.OutputPadding = dml_output_padding;

  conv_op_desc.GroupCount = static_cast<UINT>(_groups);
  conv_op_desc.FusedActivation =
      nullptr;  // No fused activation in this basic implementation

  DML_OPERATOR_DESC dml_op_desc_wrapper{};
  dml_op_desc_wrapper.Type = DML_OPERATOR_CONVOLUTION;
  dml_op_desc_wrapper.Desc = &conv_op_desc;

  // 3. Get or Create Compiled Operator from Cache
  DML_EXECUTION_FLAGS execution_flags = DML_EXECUTION_FLAG_NONE;
  if (dml_data_type == DML_TENSOR_DATA_TYPE_FLOAT16) {
    execution_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  }

  // Use the global API which internally gets the device.
  dml::Operator* dml_convolution_operator = dml::GetOrCreateCompiledOperatorApi(
      &dml_op_desc_wrapper, execution_flags);

  // 4. Prepare Bindings for Execution
  // Input resources (order: Input, Filter, Bias (if present))
  // StorageView::buffer() returns const void*. We need non-const void* for
  // static_cast to ID3D12Resource*. Then, this ID3D12Resource* is used in
  // DML_BUFFER_BINDING.
  std::vector<DML_BINDING_DESC> input_bindings;

  ID3D12Resource* input_d3d_resource =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  DML_BUFFER_BINDING input_buffer_binding = dml::create_buffer_binding(
      input_d3d_resource, 0, input_buffer_desc.TotalTensorSizeInBytes);
  input_bindings.push_back(dml::create_binding_desc(input_buffer_binding));

  ID3D12Resource* weight_d3d_resource =
      static_cast<ID3D12Resource*>(const_cast<void*>(weight.buffer()));
  DML_BUFFER_BINDING weight_buffer_binding = dml::create_buffer_binding(
      weight_d3d_resource, 0, weight_buffer_desc.TotalTensorSizeInBytes);
  input_bindings.push_back(dml::create_binding_desc(weight_buffer_binding));

  if (bias) {
    ID3D12Resource* bias_d3d_resource =
        static_cast<ID3D12Resource*>(const_cast<void*>(bias->buffer()));
    DML_BUFFER_BINDING bias_buffer_binding = dml::create_buffer_binding(
        bias_d3d_resource, 0, bias_buffer_desc_storage.TotalTensorSizeInBytes);
    input_bindings.push_back(dml::create_binding_desc(bias_buffer_binding));
  } else {
    // If BiasTensor was null in DML_CONVOLUTION_OPERATOR_DESC,
    // the corresponding input binding must be DML_BINDING_TYPE_NONE.
    input_bindings.push_back({DML_BINDING_TYPE_NONE, nullptr});
  }

  // Output resources
  std::vector<DML_BINDING_DESC> output_bindings;
  ID3D12Resource* output_d3d_resource = static_cast<ID3D12Resource*>(
      output.buffer());  // Output buffer is non-const
  DML_BUFFER_BINDING output_buffer_binding = dml::create_buffer_binding(
      output_d3d_resource, 0, output_buffer_desc.TotalTensorSizeInBytes);
  output_bindings.push_back(dml::create_binding_desc(output_buffer_binding));

  // 5. Execute the Operator
  // This records the dispatch into the command list associated with
  // dml_device_handle. The command list needs to be executed later by the
  // framework.
  dml_convolution_operator->Execute(input_bindings, output_bindings);
}

// Template specializations for Conv1D::compute for Device::DirectML

template <>
void Conv1D::compute<Device::DirectML, float>(const StorageView& input,
                                              const StorageView& weight,
                                              const StorageView* bias,
                                              StorageView& output,
                                              const StorageView* qscale) const {
  if (qscale) {
    throw std::runtime_error(
        "Conv1D DML (float): Quantization with qscale is not supported.");
  }
  // Ensure output tensor is already allocated with correct shape and type.
  // CTranslate2's Op::operator() typically handles output allocation.
  compute<Device::DirectML, float>(input, weight, bias, output, qscale);
}

template <>
void Conv1D::compute<Device::DirectML, float16_t>(
    const StorageView& input,
    const StorageView& weight,
    const StorageView* bias,
    StorageView& output,
    const StorageView* qscale) const {
  if (qscale) {
    throw std::runtime_error(
        "Conv1D DML (float16): Quantization with qscale is not supported.");
  }
  compute<Device::DirectML, float16_t>(input, weight, bias, output, qscale);
}

// Note: If bfloat16_t is to be supported, a similar specialization would be
// needed. DML has limited or no native bfloat16 support; it might require
// conversion to/from float32 or rely on newer DML versions if support is added.
// For now, only float32 and float16.

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML
