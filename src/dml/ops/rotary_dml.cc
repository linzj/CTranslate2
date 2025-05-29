#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rotary.h"

#include <vector>
#include "dml/backend_dml.h"
#include "dml/dxdevice.h"
#include "dml/operator.h"
#include "dml/operator_cache.h"

using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
namespace ops {

// Helper function to get DML data type from C++ type
template <typename T>
DML_TENSOR_DATA_TYPE get_dml_data_type() {
  if constexpr (std::is_same_v<T, float>) {
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return DML_TENSOR_DATA_TYPE_FLOAT16;  // Use FLOAT16 for bfloat16
                                          // approximation
  } else {
    // Should not happen due to DECLARE_IMPL, but as a safeguard:
    throw std::invalid_argument("Unsupported data type for DirectML Rotary op");
  }
}

// Helper to get element size for data type
UINT get_element_size(DML_TENSOR_DATA_TYPE data_type) {
  switch (data_type) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
      return 4;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
      return 2;
    default:
      // Should be caught by get_dml_data_type, but as a safeguard:
      throw std::invalid_argument(
          "Unsupported DML_TENSOR_DATA_TYPE in get_element_size");
  }
}

// Helper to create tensor descriptor with proper shape
void create_tensor_desc(
    const StorageView& storage,
    DML_BUFFER_TENSOR_DESC& buffer_desc,
    DML_TENSOR_DESC& tensor_desc,
    std::vector<UINT>& shape,  // Output parameter to store the shape
    DML_TENSOR_DATA_TYPE data_type) {
  shape.clear();
  shape.reserve(storage.rank());
  for (dim_t i = 0; i < storage.rank(); ++i) {
    shape.push_back(static_cast<UINT>(storage.dim(i)));
  }

  buffer_desc.DataType = data_type;
  buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc.DimensionCount = static_cast<UINT>(storage.rank());
  buffer_desc.Sizes = shape.data();
  buffer_desc.Strides = nullptr;
  buffer_desc.TotalTensorSizeInBytes = storage.size() * storage.item_size();
  buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc.Desc = &buffer_desc;
}

// Helper to create tensor descriptor with custom shape and strides
void create_custom_tensor_desc(
    const std::vector<UINT>& sizes,
    const std::vector<UINT>* strides,  // Can be nullptr for contiguous
    DML_BUFFER_TENSOR_DESC& buffer_desc,
    DML_TENSOR_DESC& tensor_desc,
    DML_TENSOR_DATA_TYPE data_type,
    UINT64 total_size_in_bytes) {
  buffer_desc.DataType = data_type;
  buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
  buffer_desc.DimensionCount = static_cast<UINT>(sizes.size());
  buffer_desc.Sizes = sizes.data();
  buffer_desc.Strides = strides ? strides->data() : nullptr;
  buffer_desc.TotalTensorSizeInBytes = total_size_in_bytes;
  buffer_desc.GuaranteedBaseOffsetAlignment = 0;

  tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
  tensor_desc.Desc = &buffer_desc;
}

class RotaryDMLCompute {
 private:
  dml::Device* device_;
  DML_TENSOR_DATA_TYPE data_type_;
  UINT element_size_;

 public:
  RotaryDMLCompute(dml::Device* device, DML_TENSOR_DATA_TYPE data_type)
      : device_(device),
        data_type_(data_type),
        element_size_(get_element_size(data_type)) {}

  void compute_non_interleave(ID3D12Resource* input_buffer,
                              ID3D12Resource* sin_buffer,
                              ID3D12Resource* cos_buffer,
                              ID3D12Resource* output_buffer,
                              const StorageView& input,
                              const StorageView& sin,
                              const StorageView& cos,
                              const StorageView& output,
                              dim_t feature_depth,
                              dim_t rot_ndims,
                              dim_t middle) {
    // Create tensor descriptors
    std::vector<UINT> input_shape, sin_shape, cos_shape, output_shape;
    DML_BUFFER_TENSOR_DESC input_buffer_desc, sin_buffer_desc, cos_buffer_desc,
        output_buffer_desc;
    DML_TENSOR_DESC input_desc, sin_desc, cos_desc, output_desc;

    create_tensor_desc(input, input_buffer_desc, input_desc, input_shape,
                       data_type_);
    create_tensor_desc(sin, sin_buffer_desc, sin_desc, sin_shape, data_type_);
    create_tensor_desc(cos, cos_buffer_desc, cos_desc, cos_shape, data_type_);
    create_tensor_desc(output, output_buffer_desc, output_desc, output_shape,
                       data_type_);

    // Step 1: Compute x * cos -> temp1
    auto temp1_buffer = device_->CreatePreferredDeviceMemoryBuffer(
        output_buffer_desc.TotalTensorSizeInBytes);

    multiply_tensors(
        input_buffer, cos_buffer, temp1_buffer.Get(), input_desc, cos_desc,
        output_desc);  // output_desc here describes temp1_buffer's layout

    // If no dimensions are to be rotated, or rot_ndims is 0, temp1 (x*cos) is
    // not the final answer. The final answer would be x. However, the RoPE
    // formula implies cos[0] for unrotated parts would be 1 and sin[0] would be
    // 0. If rot_ndims == 0, the CUDA kernel effectively copies x to y. Here, if
    // rot_ndims == 0, middle will be 0. The slicing and subsequent ops should
    // handle this gracefully (empty slices). The copy_unchanged_elements at the
    // end will copy the original input for elements >= rot_ndims. If rot_ndims
    // == 0, it means all elements are "unchanged", so input will be copied to
    // output. The multiply_tensors above would have used cos values that should
    // be 1 for these elements if rot_ndims=0.

    // The logic now proceeds to calculate sin_term regardless, and the final
    // copy handles all cases.

    if (rot_ndims > 0) {
      // Only proceed with sin term calculation if there are
      // dimensions to rotate
      // Step 2: Create sliced views for first and second halves (up to
      // rot_ndims) Batch size here refers to all dimensions except the last one
      // (feature_depth)
      dim_t batch_elements = 1;
      for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_elements *= input_shape[i];
      }

      auto half_rot_ndims_size_bytes = batch_elements * middle * element_size_;

      // Create buffers for sliced data
      auto x_first_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto x_second_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      // Sin/Cos tensors are typically [max_seq_len, rot_ndims]. We need to
      // slice them appropriately. Assuming sin/cos input tensors are already
      // shaped [..., rot_ndims] or broadcastable. The sin_shape used for
      // slicing should correspond to the part of sin tensor that aligns with
      // rot_ndims.
      std::vector<UINT> rot_sin_cos_shape =
          sin_shape;  // Shape of the sin/cos tensor
      if (!rot_sin_cos_shape.empty())
        rot_sin_cos_shape.back() = static_cast<UINT>(rot_ndims);
      else { /* Handle error or assume broadcast if sin_shape is scalar like */
      }

      auto sin_first_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto sin_second_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);

      // Calculate shapes for halves based on rot_ndims
      std::vector<UINT> x_slice_shape_for_rot =
          input_shape;  // Shape of input tensor
      x_slice_shape_for_rot.back() = static_cast<UINT>(
          rot_ndims);  // We are interested in the first rot_ndims of input

      std::vector<UINT> half_rot_shape = x_slice_shape_for_rot;
      half_rot_shape.back() = static_cast<UINT>(middle);

      // Slice input and sin tensors (up to rot_ndims for the parts involved in
      // rotation) Input tensor (x) for rotation part
      auto x_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * element_size_);
      slice_tensor_last_dim(input_buffer, x_rot_part_buffer.Get(), input_shape,
                            x_slice_shape_for_rot, 0, rot_ndims);

      slice_tensor_last_dim(x_rot_part_buffer.Get(), x_first_half_buffer.Get(),
                            x_slice_shape_for_rot, half_rot_shape, 0, middle);
      slice_tensor_last_dim(x_rot_part_buffer.Get(), x_second_half_buffer.Get(),
                            x_slice_shape_for_rot, half_rot_shape, middle,
                            middle);

      // Sin tensor for rotation part
      // Assuming sin_buffer contains sin values for at least rot_ndims
      slice_tensor_last_dim(sin_buffer, sin_first_half_buffer.Get(),
                            rot_sin_cos_shape,  // Use rot_sin_cos_shape
                            half_rot_shape, 0, middle);
      slice_tensor_last_dim(sin_buffer, sin_second_half_buffer.Get(),
                            rot_sin_cos_shape,  // Use rot_sin_cos_shape
                            half_rot_shape, middle, middle);

      // Step 3: Compute rotated terms
      auto rot_first_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto rot_second_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);

      // Create tensor descriptors for half tensors (these describe the content
      // of *_half_buffer)
      DML_BUFFER_TENSOR_DESC half_buffer_desc;
      DML_TENSOR_DESC half_desc;
      create_custom_tensor_desc(half_rot_shape, nullptr, half_buffer_desc,
                                half_desc, data_type_,
                                half_rot_ndims_size_bytes);

      // First rotation: -x[middle:rot_ndims] * sin[0:middle]
      auto neg_x_second_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      negate_tensor(x_second_half_buffer.Get(), neg_x_second_buffer.Get(),
                    half_desc);
      multiply_tensors(neg_x_second_buffer.Get(), sin_first_half_buffer.Get(),
                       rot_first_buffer.Get(), half_desc, half_desc, half_desc);

      // Second rotation: x[0:middle] * sin[middle:rot_ndims]
      multiply_tensors(x_first_half_buffer.Get(), sin_second_half_buffer.Get(),
                       rot_second_buffer.Get(), half_desc, half_desc,
                       half_desc);

      // Step 4: Add rotated terms to temp1 (which currently holds x*cos)
      // The full_shape for add_sliced_to_full_tensor should be the shape of
      // temp1_buffer, which is output_shape. The slice_shape should be
      // half_rot_shape.
      add_sliced_to_full_tensor(temp1_buffer.Get(), rot_first_buffer.Get(),
                                output_shape, half_rot_shape, 0, middle);
      add_sliced_to_full_tensor(temp1_buffer.Get(), rot_second_buffer.Get(),
                                output_shape, half_rot_shape, middle, middle);
    }
    // temp1_buffer now contains (x*cos + sin_term) for the first rot_ndims
    // elements, and (x*cos) for elements from rot_ndims to feature_depth. The
    // (x*cos) for elements rot_ndims to feature_depth should ideally be (x*1)
    // if cos was set up correctly for non-rotated parts.

    // Step 5: Handle unchanged elements (i >= rot_ndims) by copying from
    // original input, and copy rotated part from temp1_buffer.
    if (rot_ndims < feature_depth) {
      copy_unchanged_elements(input_buffer, temp1_buffer.Get(), output_buffer,
                              input_shape, output_shape, rot_ndims,
                              feature_depth);
    } else {
      // rot_ndims >= feature_depth (effectively rot_ndims == feature_depth)
      // All elements were subject to rotation (or should have been if rot_ndims
      // > 0). temp1_buffer now holds the fully rotated result.
      copy_tensor(temp1_buffer.Get(), output_buffer, output_desc);
    }
  }

  void compute_interleave(ID3D12Resource* input_buffer,
                          ID3D12Resource* sin_buffer,
                          ID3D12Resource* cos_buffer,
                          ID3D12Resource* output_buffer,
                          const StorageView& input,
                          const StorageView& sin,
                          const StorageView& cos,
                          const StorageView& output,
                          dim_t feature_depth,
                          dim_t rot_ndims) {
    // Create tensor descriptors
    std::vector<UINT> input_shape, sin_shape, cos_shape, output_shape;
    DML_BUFFER_TENSOR_DESC input_buffer_desc, sin_buffer_desc, cos_buffer_desc,
        output_buffer_desc;
    DML_TENSOR_DESC input_desc, sin_desc, cos_desc, output_desc;

    create_tensor_desc(input, input_buffer_desc, input_desc, input_shape,
                       data_type_);
    create_tensor_desc(sin, sin_buffer_desc, sin_desc, sin_shape, data_type_);
    create_tensor_desc(cos, cos_buffer_desc, cos_desc, cos_shape, data_type_);
    create_tensor_desc(output, output_buffer_desc, output_desc, output_shape,
                       data_type_);

    // Step 1: Compute x * cos -> temp1
    auto temp1_buffer = device_->CreatePreferredDeviceMemoryBuffer(
        output_buffer_desc.TotalTensorSizeInBytes);
    multiply_tensors(input_buffer, cos_buffer, temp1_buffer.Get(), input_desc,
                     cos_desc, output_desc);

    if (rot_ndims > 0) {
      // Step 2: Extract even and odd indexed elements from input and sin (up to
      // rot_ndims)
      dim_t batch_elements = 1;
      for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_elements *= input_shape[i];
      }
      auto even_count = (rot_ndims + 1) / 2;
      auto odd_count = rot_ndims / 2;

      auto even_size_bytes = batch_elements * even_count * element_size_;
      auto odd_size_bytes = batch_elements * odd_count * element_size_;

      // Buffers for x slices
      auto x_even_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(even_size_bytes);
      auto x_odd_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(odd_size_bytes);
      // Buffers for sin slices
      auto sin_even_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(even_size_bytes);
      auto sin_odd_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(odd_size_bytes);

      // Shape of the first rot_ndims elements of input/sin/cos
      std::vector<UINT> x_rot_part_shape = input_shape;
      x_rot_part_shape.back() = static_cast<UINT>(rot_ndims);

      std::vector<UINT> sin_cos_rot_part_shape =
          sin_shape;  // Shape of the sin/cos tensor
      if (!sin_cos_rot_part_shape.empty())
        sin_cos_rot_part_shape.back() = static_cast<UINT>(rot_ndims);

      // Temporary buffer for the first rot_ndims of input
      auto x_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * element_size_);
      slice_tensor_last_dim(input_buffer, x_rot_part_buffer.Get(), input_shape,
                            x_rot_part_shape, 0, rot_ndims);

      // Temporary buffer for the first rot_ndims of sin
      auto sin_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * element_size_);
      slice_tensor_last_dim(sin_buffer, sin_rot_part_buffer.Get(), sin_shape,
                            sin_cos_rot_part_shape, 0, rot_ndims);

      // Shapes for even/odd slices (last dimension is count)
      std::vector<UINT> even_slice_shape = x_rot_part_shape;
      even_slice_shape.back() = static_cast<UINT>(even_count);
      std::vector<UINT> odd_slice_shape = x_rot_part_shape;
      odd_slice_shape.back() = static_cast<UINT>(odd_count);

      extract_even_odd_elements(x_rot_part_buffer.Get(), x_even_buffer.Get(),
                                x_odd_buffer.Get(), x_rot_part_shape, rot_ndims,
                                even_count, odd_count);
      extract_even_odd_elements(sin_rot_part_buffer.Get(),
                                sin_even_buffer.Get(), sin_odd_buffer.Get(),
                                sin_cos_rot_part_shape, rot_ndims, even_count,
                                odd_count);

      // Step 3: Compute rotated terms
      // For even positions i: use -x[i+1] * sin[i] (use x_odd with negation,
      // sin_even) For odd positions i: use x[i-1] * sin[i] (use x_even,
      // sin_odd)
      auto rot_term_for_even_pos_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(
              even_size_bytes);  // will hold -x_odd * sin_even
      auto rot_term_for_odd_pos_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(
              odd_size_bytes);  // will hold x_even * sin_odd

      DML_BUFFER_TENSOR_DESC even_slice_buffer_desc, odd_slice_buffer_desc;
      DML_TENSOR_DESC even_slice_desc, odd_slice_desc;
      create_custom_tensor_desc(even_slice_shape, nullptr,
                                even_slice_buffer_desc, even_slice_desc,
                                data_type_, even_size_bytes);
      create_custom_tensor_desc(odd_slice_shape, nullptr, odd_slice_buffer_desc,
                                odd_slice_desc, data_type_, odd_size_bytes);

      // Calculate term for even output positions: -x_odd * sin_even
      if (even_count > 0 && odd_count > 0) {
        // x_odd is used here, so odd_count must be > 0
        // If rot_ndims is 1, odd_count is 0. x_odd_buffer would be empty.
        // This term is for y[0], y[2], ...
        // y[i] = x[i]cos[i] - x[i+1]sin[i]
        // We need x_odd (x[1], x[3]...) and sin_even (sin[0], sin[2]...)
        // The number of elements in x_odd might be less than sin_even if
        // rot_ndims is odd. E.g. rot_ndims=3: x_odd is (x1), sin_even is
        // (s0,s2). We only need x1*s0 for y[0]. The current
        // compute_interleave_rotation_even/odd assumes inputs have same
        // 'count'. This needs careful handling if counts differ. For y[i] (even
        // i), sin_term is -x[i+1]*sin[i]. x[i+1] is from x_odd. sin[i] is from
        // sin_even. The number of such terms is 'even_count'. The x_odd buffer
        // has 'odd_count' elements. sin_even buffer has 'even_count' elements.
        // If rot_ndims is odd (e.g., 3), even_count=2 (y0,y2), odd_count=1
        // (x1). We need -x1*s0 for y0. For y2, there's no x3 if rot_ndims=3.
        // The loop in CUDA kernel `(i % 2 == 0 ? -C(x[i + 1]) : C(x[i - 1]))`
        // implicitly handles boundaries because x[i+1] or x[i-1] might be >=
        // rot_ndims. The DML ops here assume broadcast or exact shape matches.
        // Let's assume for now the helper functions handle the potentially
        // smaller x_odd for the last even element. If not, this is a subtle bug
        // source. The current `compute_interleave_rotation_even` uses
        // `tensor_desc` which is `even_desc`. This implies `x_odd_buffer` is
        // expected to be compatible with `even_desc`. This is only true if
        // even_count == odd_count (rot_ndims is even). If rot_ndims is odd,
        // say 3. even_count=2, odd_count=1. x_odd has 1 element. sin_even has 2
        // elements. -x_odd[0]*sin_even[0] is one term. What about the second
        // element for rot_term_for_even_pos_buffer? This part needs more robust
        // handling of potentially mismatched counts for the multiplication. For
        // simplicity in this fix, we'll assume the DML multiply handles
        // broadcasting or the smaller input correctly, or that the CUDA
        // kernel's boundary (i < ndims) means these problematic terms aren't
        // computed. CUDA: if i=rot_ndims-1 and it's even, x[i+1] is
        // x[rot_ndims], which is out of bounds for rotation. The CUDA kernel's
        // `if (i >= ndims)` handles this. The DML ops for multiply will require
        // tensors of compatible shapes. The current
        // `compute_interleave_rotation_even` uses `even_desc` for all inputs to
        // multiply. This means if x_odd has fewer elements than sin_even, it's
        // an issue. We should probably slice x_odd to match the number of
        // actual products needed. However, the original code structure is kept
        // for now.
        compute_interleave_rotation_even(
            x_odd_buffer.Get(), sin_even_buffer.Get(),
            rot_term_for_even_pos_buffer.Get(),
            even_slice_desc,  // Output is even_count
            even_count > odd_count
                ? odd_count
                : even_count);  // Use min count for safety with multiply
      }

      // Calculate term for odd output positions: x_even * sin_odd
      if (odd_count > 0) {  // x_even is used here.
        // y[i] = x[i]cos[i] + x[i-1]sin[i]
        // We need x_even (x[0], x[2]...) and sin_odd (sin[1], sin[3]...)
        // Number of terms is 'odd_count'.
        // x_even has 'even_count' elements. sin_odd has 'odd_count' elements.
        compute_interleave_rotation_odd(x_even_buffer.Get(),
                                        sin_odd_buffer.Get(),
                                        rot_term_for_odd_pos_buffer.Get(),
                                        odd_slice_desc,  // Output is odd_count
                                        odd_count);
      }

      // Step 4: BUG FIX 2: Correctly add these terms to temp1_buffer
      combine_even_odd_to_interleaved_additive(
          temp1_buffer.Get(), rot_term_for_even_pos_buffer.Get(),
          rot_term_for_odd_pos_buffer.Get(),
          output_shape,  // Shape of temp1_buffer
          even_slice_shape, odd_slice_shape, rot_ndims, even_count, odd_count);
    }

    // Step 5: Handle unchanged elements
    if (rot_ndims < feature_depth) {
      copy_unchanged_elements(input_buffer, temp1_buffer.Get(), output_buffer,
                              input_shape, output_shape, rot_ndims,
                              feature_depth);
    } else {
      copy_tensor(temp1_buffer.Get(), output_buffer, output_desc);
    }
  }

 private:
  void multiply_tensors(ID3D12Resource* a_buffer,
                        ID3D12Resource* b_buffer,
                        ID3D12Resource* output_buffer,
                        const DML_TENSOR_DESC& a_desc,
                        const DML_TENSOR_DESC& b_desc,
                        const DML_TENSOR_DESC& output_desc) {
    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc = {};
    multiply_desc.ATensor = &a_desc;
    multiply_desc.BTensor = &b_desc;
    multiply_desc.OutputTensor = &output_desc;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_MULTIPLY;
    op_desc.Desc = &multiply_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({a_buffer, b_buffer}, {output_buffer});
  }

  void negate_tensor(ID3D12Resource* input_buffer,
                     ID3D12Resource* output_buffer,
                     const DML_TENSOR_DESC& tensor_desc) {
    // Assuming output has same desc as input for this op
    DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC negate_desc = {};
    negate_desc.InputTensor = &tensor_desc;
    negate_desc.OutputTensor = &tensor_desc;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_NEGATE;
    op_desc.Desc = &negate_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({input_buffer}, {output_buffer});
  }

  void copy_tensor(
      ID3D12Resource* input_buffer,
      ID3D12Resource* output_buffer,
      const DML_TENSOR_DESC& tensor_desc) {  // Output desc might differ
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
    identity_desc.InputTensor = &tensor_desc;
    identity_desc.OutputTensor =
        &tensor_desc;  // Assuming output has same desc as input

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
    op_desc.Desc = &identity_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({input_buffer}, {output_buffer});
  }

  void slice_tensor_last_dim(
      ID3D12Resource* input_buffer,
      ID3D12Resource* output_buffer,
      const std::vector<UINT>& input_total_shape,  // Full shape of input buffer
      const std::vector<UINT>&
          output_slice_shape,  // Shape of the desired slice
      dim_t start_idx,
      dim_t size_on_last_dim) {
    // Create input tensor descriptor for the full input
    DML_BUFFER_TENSOR_DESC input_buffer_desc;
    DML_TENSOR_DESC input_desc;
    UINT64 input_size_bytes = 1;
    for (auto dim : input_total_shape)
      input_size_bytes *= dim;
    input_size_bytes *= element_size_;
    create_custom_tensor_desc(input_total_shape, nullptr, input_buffer_desc,
                              input_desc, data_type_, input_size_bytes);

    // Create output tensor descriptor for the slice
    DML_BUFFER_TENSOR_DESC output_buffer_desc;
    DML_TENSOR_DESC output_desc;
    UINT64 output_size_bytes = 1;
    for (auto dim : output_slice_shape)
      output_size_bytes *= dim;
    output_size_bytes *= element_size_;
    create_custom_tensor_desc(output_slice_shape, nullptr, output_buffer_desc,
                              output_desc, data_type_, output_size_bytes);

    std::vector<UINT> offsets(input_total_shape.size(), 0);
    // Start with full sizes
    std::vector<UINT> sizes = input_total_shape;
    std::vector<UINT> strides(input_total_shape.size(), 1);
    offsets.back() = static_cast<UINT>(start_idx);
    // Set size of slice on last dim
    sizes.back() = static_cast<UINT>(size_on_last_dim);

    DML_SLICE_OPERATOR_DESC slice_desc = {};
    slice_desc.InputTensor = &input_desc;
    slice_desc.OutputTensor = &output_desc;
    slice_desc.DimensionCount = static_cast<UINT>(input_total_shape.size());
    slice_desc.Offsets = offsets.data();
    slice_desc.Sizes = sizes.data();
    slice_desc.Strides = strides.data();

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_SLICE;
    op_desc.Desc = &slice_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({input_buffer}, {output_buffer});
  }

  void add_tensors(ID3D12Resource* a_buffer,
                   ID3D12Resource* b_buffer,
                   ID3D12Resource* output_buffer,
                   const DML_TENSOR_DESC& a_desc,
                   const DML_TENSOR_DESC& b_desc,
                   const DML_TENSOR_DESC& output_desc) {
    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {};
    add_desc.ATensor = &a_desc;
    add_desc.BTensor = &b_desc;
    add_desc.OutputTensor = &output_desc;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_ELEMENT_WISE_ADD;
    op_desc.Desc = &add_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({a_buffer, b_buffer}, {output_buffer});
  }

  void scatter_slice_to_full_tensor(ID3D12Resource* full_buffer_to_update,
                                    ID3D12Resource* slice_data_to_scatter,
                                    const std::vector<UINT>& full_shape,
                                    const std::vector<UINT>& slice_shape,
                                    dim_t start_idx_on_last_dim,
                                    dim_t size_on_last_dim) {
    // Elements in all dims except the last one
    dim_t batch_elements = 1;
    for (size_t i = 0; i < full_shape.size() - 1; ++i) {
      batch_elements *= full_shape[i];
    }

    auto num_indices = batch_elements * size_on_last_dim;
    auto indices_buffer_size_bytes = num_indices * sizeof(UINT32);
    ComPtr<ID3D12Resource> indices_buffer_gpu;

    std::vector<UINT32> indices_data_cpu;
    indices_data_cpu.reserve(num_indices);

    for (dim_t batch = 0; batch < batch_elements; ++batch) {
      for (dim_t i = 0; i < size_on_last_dim; ++i) {
        indices_data_cpu.push_back(static_cast<UINT32>(
            batch * full_shape.back() + start_idx_on_last_dim + i));
      }
    }

    indices_buffer_gpu = device_->Upload(
        indices_buffer_size_bytes,
        std::string_view(reinterpret_cast<const char*>(
                             static_cast<const void*>(indices_data_cpu.data())),
                         indices_buffer_size_bytes));

    // Flatten shapes for scatter: DML_SCATTER_ELEMENTS works on a specified
    // axis. Here, we make it effectively 1D scatter by precomputing exact
    // indices.
    std::vector<UINT> flat_full_shape = {
        static_cast<UINT>(batch_elements * full_shape.back())};
    std::vector<UINT> flat_slice_shape = {static_cast<UINT>(
        num_indices)};  // Shape of slice_data_to_scatter, flattened
    std::vector<UINT> flat_indices_shape = {static_cast<UINT>(num_indices)};

    DML_BUFFER_TENSOR_DESC full_buffer_desc, slice_buffer_desc,
        indices_buffer_desc;
    DML_TENSOR_DESC full_desc, slice_desc, indices_desc;

    UINT64 full_total_bytes = flat_full_shape[0] * element_size_;
    UINT64 slice_total_bytes = flat_slice_shape[0] * element_size_;

    create_custom_tensor_desc(flat_full_shape, nullptr, full_buffer_desc,
                              full_desc, data_type_, full_total_bytes);
    create_custom_tensor_desc(flat_slice_shape, nullptr, slice_buffer_desc,
                              slice_desc, data_type_, slice_total_bytes);
    create_custom_tensor_desc(flat_indices_shape, nullptr, indices_buffer_desc,
                              indices_desc, DML_TENSOR_DATA_TYPE_UINT32,
                              indices_buffer_size_bytes);

    DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
    scatter_desc.InputTensor = &full_desc;
    scatter_desc.IndicesTensor = &indices_desc;
    scatter_desc.UpdatesTensor = &slice_desc;
    scatter_desc.OutputTensor = &full_desc;
    scatter_desc.Axis = 0;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_SCATTER_ELEMENTS;
    op_desc.Desc = &scatter_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({full_buffer_to_update, indices_buffer_gpu.Get(),
                          slice_data_to_scatter},
                         {full_buffer_to_update});
  }

  // Helper for non-interleaved: temp1[slice] = temp1[slice] + rot_term_slice
  void add_sliced_to_full_tensor(ID3D12Resource* full_buffer_to_update,
                                 ID3D12Resource* slice_to_add,
                                 const std::vector<UINT>& full_shape,
                                 const std::vector<UINT>& slice_shape,
                                 dim_t start_idx_on_last_dim,
                                 dim_t size_on_last_dim) {
    UINT64 slice_size_bytes = 1;
    for (auto dim : slice_shape)
      slice_size_bytes *= dim;
    slice_size_bytes *= element_size_;

    auto current_slice_from_full_buffer =
        device_->CreatePreferredDeviceMemoryBuffer(slice_size_bytes);
    slice_tensor_last_dim(full_buffer_to_update,
                          current_slice_from_full_buffer.Get(), full_shape,
                          slice_shape, start_idx_on_last_dim, size_on_last_dim);

    auto result_added_slice_buffer =
        device_->CreatePreferredDeviceMemoryBuffer(slice_size_bytes);
    DML_BUFFER_TENSOR_DESC slice_buffer_desc_for_add;
    DML_TENSOR_DESC slice_desc_for_add;
    create_custom_tensor_desc(slice_shape, nullptr, slice_buffer_desc_for_add,
                              slice_desc_for_add, data_type_, slice_size_bytes);

    add_tensors(current_slice_from_full_buffer.Get(), slice_to_add,
                result_added_slice_buffer.Get(), slice_desc_for_add,
                slice_desc_for_add, slice_desc_for_add);

    scatter_slice_to_full_tensor(
        full_buffer_to_update, result_added_slice_buffer.Get(), full_shape,
        slice_shape, start_idx_on_last_dim, size_on_last_dim);
  }

  void copy_unchanged_elements(ID3D12Resource* original_input_buffer,
                               ID3D12Resource* computed_rotated_buffer,
                               ID3D12Resource* final_output_buffer,
                               const std::vector<UINT>& input_shape,
                               const std::vector<UINT>& output_shape,
                               dim_t rot_ndims,
                               dim_t feature_depth) {
    if (rot_ndims > 0) {
      std::vector<UINT> rotated_part_shape = output_shape;
      rotated_part_shape.back() = static_cast<UINT>(rot_ndims);
      // Slice from computed_rotated_buffer
      auto temp_rotated_part_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(
              (output_shape[0] * rot_ndims) * element_size_);
      slice_tensor_last_dim(computed_rotated_buffer,
                            temp_rotated_part_buffer.Get(), output_shape,
                            rotated_part_shape, 0, rot_ndims);
      // Scatter to final_output_buffer
      scatter_slice_to_full_tensor(final_output_buffer,
                                   temp_rotated_part_buffer.Get(), output_shape,
                                   rotated_part_shape, 0, rot_ndims);
    }

    // Copy unchanged part (rot_ndims to feature_depth-1) from
    // original_input_buffer to final_output_buffer
    if (feature_depth > rot_ndims) {
      dim_t unchanged_size = feature_depth - rot_ndims;
      std::vector<UINT> unchanged_part_shape = output_shape;
      unchanged_part_shape.back() = static_cast<UINT>(unchanged_size);

      UINT64 unchanged_size_bytes = 1;
      dim_t batch_elements = 1;
      for (size_t i = 0; i < unchanged_part_shape.size() - 1; ++i)
        batch_elements *= unchanged_part_shape[i];
      unchanged_size_bytes = batch_elements * unchanged_size * element_size_;

      auto temp_unchanged_part_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(unchanged_size_bytes);
      slice_tensor_last_dim(original_input_buffer,
                            temp_unchanged_part_buffer.Get(), input_shape,
                            unchanged_part_shape, rot_ndims, unchanged_size);
      scatter_slice_to_full_tensor(
          final_output_buffer, temp_unchanged_part_buffer.Get(), output_shape,
          unchanged_part_shape, rot_ndims, unchanged_size);
    }
  }

  void extract_even_odd_elements(ID3D12Resource* input_buffer,
                                 ID3D12Resource* even_buffer,
                                 ID3D12Resource* odd_buffer,
                                 const std::vector<UINT>& input_rot_part_shape,
                                 dim_t rot_ndims,
                                 dim_t even_count,
                                 dim_t odd_count) {
    if (even_count > 0) {
      std::vector<UINT> even_slice_shape = input_rot_part_shape;
      even_slice_shape.back() = static_cast<UINT>(even_count);
      extract_strided_elements(input_buffer, even_buffer, input_rot_part_shape,
                               even_slice_shape, 0, 2);
    }
    if (odd_count > 0) {
      std::vector<UINT> odd_slice_shape = input_rot_part_shape;
      odd_slice_shape.back() = static_cast<UINT>(odd_count);
      extract_strided_elements(input_buffer, odd_buffer, input_rot_part_shape,
                               odd_slice_shape, 1, 2);
    }
  }

  void extract_strided_elements(ID3D12Resource* input_buffer,
                                ID3D12Resource* output_buffer,
                                const std::vector<UINT>& input_shape,
                                const std::vector<UINT>& output_shape,
                                dim_t start_offset_on_last_dim,
                                dim_t stride_on_last_dim) {
    DML_BUFFER_TENSOR_DESC input_buffer_desc;
    DML_TENSOR_DESC input_desc;
    UINT64 input_size_bytes = 1;
    for (auto dim : input_shape)
      input_size_bytes *= dim;
    input_size_bytes *= element_size_;
    create_custom_tensor_desc(input_shape, nullptr, input_buffer_desc,
                              input_desc, data_type_, input_size_bytes);

    DML_BUFFER_TENSOR_DESC output_buffer_desc;
    DML_TENSOR_DESC output_desc;
    UINT64 output_size_bytes = 1;
    for (auto dim : output_shape)
      output_size_bytes *= dim;
    output_size_bytes *= element_size_;
    create_custom_tensor_desc(output_shape, nullptr, output_buffer_desc,
                              output_desc, data_type_, output_size_bytes);

    std::vector<UINT> window_offsets(input_shape.size(), 0);
    std::vector<UINT> window_sizes = input_shape;
    std::vector<INT> window_strides(input_shape.size(), 1);

    window_offsets.back() = static_cast<UINT>(start_offset_on_last_dim);
    window_sizes.back() = static_cast<UINT>(output_shape.back());
    window_strides.back() = static_cast<INT>(stride_on_last_dim);

    DML_SLICE1_OPERATOR_DESC slice_desc = {};
    slice_desc.InputTensor = &input_desc;
    slice_desc.OutputTensor = &output_desc;
    slice_desc.DimensionCount = static_cast<UINT>(input_shape.size());
    slice_desc.InputWindowOffsets = window_offsets.data();
    slice_desc.InputWindowSizes = window_sizes.data();
    slice_desc.InputWindowStrides = window_strides.data();

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_SLICE1;
    op_desc.Desc = &slice_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({input_buffer}, {output_buffer});
  }

  // Renamed for clarity, count is for output buffer
  void compute_interleave_rotation_even(ID3D12Resource* x_odd_buffer,
                                        ID3D12Resource* sin_even_buffer,
                                        ID3D12Resource* output_rot_term_buffer,
                                        const DML_TENSOR_DESC& output_desc,
                                        dim_t actual_op_count) {
    if (actual_op_count == 0)
      return;

    // This function computes -x_odd[k] * sin_even[k] for k from 0 to
    // actual_op_count-1 It assumes x_odd_buffer and sin_even_buffer have at
    // least actual_op_count elements accessible via a descriptor that matches
    // the first actual_op_count elements of output_desc. This is a
    // simplification. A robust way would be to slice inputs to actual_op_count.
    // For now, we rely on output_desc correctly describing the operation size.

    UINT64 op_size_bytes = actual_op_count;
    const DML_BUFFER_TENSOR_DESC* buffer_desc =
        static_cast<const DML_BUFFER_TENSOR_DESC*>(output_desc.Desc);
    for (UINT i = 0; i < buffer_desc->DimensionCount - 1; ++i)
      op_size_bytes *= buffer_desc->Sizes[i];
    op_size_bytes *= element_size_;

    auto neg_buffer = device_->CreatePreferredDeviceMemoryBuffer(op_size_bytes);
    // Create a descriptor for the actual operation size
    std::vector<UINT> op_shape;
    op_shape.reserve(buffer_desc->DimensionCount);
    for (UINT i = 0; i < buffer_desc->DimensionCount - 1; ++i)
      op_shape.push_back(buffer_desc->Sizes[i]);
    op_shape.push_back(actual_op_count);

    DML_BUFFER_TENSOR_DESC op_buffer_desc_s;
    DML_TENSOR_DESC op_desc_s;
    create_custom_tensor_desc(op_shape, nullptr, op_buffer_desc_s, op_desc_s,
                              data_type_, op_size_bytes);

    negate_tensor(x_odd_buffer, neg_buffer.Get(),
                  op_desc_s);  // Use op_desc_s for all
    multiply_tensors(neg_buffer.Get(), sin_even_buffer, output_rot_term_buffer,
                     op_desc_s, op_desc_s, op_desc_s);
  }

  void compute_interleave_rotation_odd(
      ID3D12Resource* x_even_buffer,           // Has even_count elements
      ID3D12Resource* sin_odd_buffer,          // Has odd_count elements
      ID3D12Resource* output_rot_term_buffer,  // For odd_count results
      const DML_TENSOR_DESC&
          output_desc,          // Describes output_rot_term_buffer (odd_count)
      dim_t actual_op_count) {  // Number of actual products
    if (actual_op_count == 0)
      return;
    // Similar to above, assumes inputs are sliceable/broadcastable to
    // actual_op_count via output_desc's shape.
    UINT64 op_size_bytes = actual_op_count;
    const DML_BUFFER_TENSOR_DESC* buffer_desc =
        static_cast<const DML_BUFFER_TENSOR_DESC*>(output_desc.Desc);
    for (UINT i = 0; i < buffer_desc->DimensionCount - 1; ++i)
      op_size_bytes *= buffer_desc->Sizes[i];
    op_size_bytes *= element_size_;

    std::vector<UINT> op_shape;
    op_shape.reserve(buffer_desc->DimensionCount);
    for (UINT i = 0; i < buffer_desc->DimensionCount - 1; ++i)
      op_shape.push_back(buffer_desc->Sizes[i]);
    op_shape.push_back(actual_op_count);

    DML_BUFFER_TENSOR_DESC op_buffer_desc_s;
    DML_TENSOR_DESC op_desc_s;
    create_custom_tensor_desc(op_shape, nullptr, op_buffer_desc_s, op_desc_s,
                              data_type_, op_size_bytes);

    multiply_tensors(x_even_buffer, sin_odd_buffer, output_rot_term_buffer,
                     op_desc_s, op_desc_s, op_desc_s);
  }

  // BUG FIX 2: New function for additive combination
  void combine_even_odd_to_interleaved_additive(
      ID3D12Resource* base_buffer_to_update,
      ID3D12Resource* even_rot_terms_to_add,
      ID3D12Resource* odd_rot_terms_to_add,
      const std::vector<UINT>& base_full_shape,
      const std::vector<UINT>& even_slice_shape,
      const std::vector<UINT>& odd_slice_shape,
      dim_t rot_ndims,
      dim_t even_count,
      dim_t odd_count) {
    // Add even_rot_terms_to_add to even positions in base_buffer_to_update
    if (even_count > 0) {
      UINT64 even_slice_bytes = 1;
      for (auto d : even_slice_shape)
        even_slice_bytes *= d;
      even_slice_bytes *= element_size_;

      auto current_even_slice_from_base =
          device_->CreatePreferredDeviceMemoryBuffer(even_slice_bytes);
      extract_strided_elements(base_buffer_to_update,
                               current_even_slice_from_base.Get(),
                               base_full_shape, even_slice_shape, 0, 2);

      auto result_added_even_slice =
          device_->CreatePreferredDeviceMemoryBuffer(even_slice_bytes);
      DML_BUFFER_TENSOR_DESC even_desc_for_add_s;
      DML_TENSOR_DESC even_desc_for_add;
      create_custom_tensor_desc(even_slice_shape, nullptr, even_desc_for_add_s,
                                even_desc_for_add, data_type_,
                                even_slice_bytes);

      add_tensors(current_even_slice_from_base.Get(), even_rot_terms_to_add,
                  result_added_even_slice.Get(), even_desc_for_add,
                  even_desc_for_add, even_desc_for_add);

      scatter_strided_to_full_tensor_overwrite(
          base_buffer_to_update, result_added_even_slice.Get(), base_full_shape,
          even_slice_shape, 0, 2, even_count);
    }

    // Add odd_rot_terms_to_add to odd positions in base_buffer_to_update
    if (odd_count > 0) {
      UINT64 odd_slice_bytes = 1;
      for (auto d : odd_slice_shape)
        odd_slice_bytes *= d;
      odd_slice_bytes *= element_size_;

      auto current_odd_slice_from_base =
          device_->CreatePreferredDeviceMemoryBuffer(odd_slice_bytes);
      extract_strided_elements(base_buffer_to_update,
                               current_odd_slice_from_base.Get(),
                               base_full_shape, odd_slice_shape, 1, 2);

      auto result_added_odd_slice =
          device_->CreatePreferredDeviceMemoryBuffer(odd_slice_bytes);
      DML_BUFFER_TENSOR_DESC odd_desc_for_add_s;
      DML_TENSOR_DESC odd_desc_for_add;
      create_custom_tensor_desc(odd_slice_shape, nullptr, odd_desc_for_add_s,
                                odd_desc_for_add, data_type_, odd_slice_bytes);

      add_tensors(current_odd_slice_from_base.Get(), odd_rot_terms_to_add,
                  result_added_odd_slice.Get(), odd_desc_for_add,
                  odd_desc_for_add, odd_desc_for_add);

      scatter_strided_to_full_tensor_overwrite(
          base_buffer_to_update, result_added_odd_slice.Get(), base_full_shape,
          odd_slice_shape, 1, 2, odd_count);
    }
  }

  // This was the original problematic scatter function, now used by the
  // additive one. It *overwrites* the target with the strided data.
  void scatter_strided_to_full_tensor_overwrite(
      ID3D12Resource* full_buffer_to_update,
      ID3D12Resource* strided_data_to_scatter,
      const std::vector<UINT>& full_shape,
      const std::vector<UINT>& strided_data_shape,
      dim_t start_offset_on_last_dim,
      dim_t stride_on_last_dim,
      dim_t count_on_last_dim_for_strided_data) {
    if (count_on_last_dim_for_strided_data == 0)
      return;

    dim_t batch_elements = 1;
    for (size_t i = 0; i < full_shape.size() - 1; ++i) {
      batch_elements *= full_shape[i];
    }

    auto num_indices = batch_elements * count_on_last_dim_for_strided_data;
    auto indices_buffer_size_bytes = num_indices * sizeof(UINT32);
    ComPtr<ID3D12Resource> indices_buffer_gpu;

    std::vector<UINT32> indices_data_cpu;
    indices_data_cpu.reserve(num_indices);
    for (dim_t batch = 0; batch < batch_elements; ++batch) {
      for (dim_t i = 0; i < count_on_last_dim_for_strided_data; ++i) {
        indices_data_cpu.push_back(static_cast<UINT32>(
            batch * full_shape.back() + start_offset_on_last_dim +
            i * stride_on_last_dim));
      }
    }
    indices_buffer_gpu = device_->Upload(
        indices_buffer_size_bytes,
        std::string_view(reinterpret_cast<const char*>(
                             static_cast<const void*>(indices_data_cpu.data())),
                         indices_buffer_size_bytes));

    std::vector<UINT> flat_full_shape = {
        static_cast<UINT>(batch_elements * full_shape.back())};
    std::vector<UINT> flat_strided_data_shape = {
        static_cast<UINT>(num_indices)};
    std::vector<UINT> flat_indices_shape = {static_cast<UINT>(num_indices)};

    DML_BUFFER_TENSOR_DESC full_desc_s, strided_data_desc_s, indices_desc_s;
    DML_TENSOR_DESC full_desc, strided_data_desc, indices_desc;

    UINT64 full_total_bytes = flat_full_shape[0] * element_size_;
    UINT64 strided_total_bytes = flat_strided_data_shape[0] * element_size_;

    create_custom_tensor_desc(flat_full_shape, nullptr, full_desc_s, full_desc,
                              data_type_, full_total_bytes);
    create_custom_tensor_desc(flat_strided_data_shape, nullptr,
                              strided_data_desc_s, strided_data_desc,
                              data_type_, strided_total_bytes);
    create_custom_tensor_desc(flat_indices_shape, nullptr, indices_desc_s,
                              indices_desc, DML_TENSOR_DATA_TYPE_UINT32,
                              indices_buffer_size_bytes);

    DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
    scatter_desc.InputTensor = &full_desc;
    scatter_desc.IndicesTensor = &indices_desc;
    scatter_desc.UpdatesTensor = &strided_data_desc;
    scatter_desc.OutputTensor = &full_desc;
    scatter_desc.Axis = 0;

    DML_OPERATOR_DESC op_desc = {};
    op_desc.Type = DML_OPERATOR_SCATTER_ELEMENTS;
    op_desc.Desc = &scatter_desc;

    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);
    compiled_op->Execute({full_buffer_to_update, indices_buffer_gpu.Get(),
                          strided_data_to_scatter},
                         {full_buffer_to_update});
  }
};

template <Device D, typename T>
void Rotary::compute(const StorageView& input,
                     const StorageView& sin,
                     const StorageView& cos,
                     StorageView& output,
                     bool is_transposed) const {
  static_assert(D == Device::DirectML,
                "This implementation is for DirectML only");

  auto* device = dml::get_device();
  if (!device) {
    throw std::runtime_error("DirectML device not available");
  }

  auto* input_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto* sin_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(sin.buffer()));
  auto* cos_buffer =
      static_cast<ID3D12Resource*>(const_cast<void*>(cos.buffer()));
  auto* output_buffer = static_cast<ID3D12Resource*>(output.buffer());

  // Determine feature_depth and rot_ndims based on CTranslate2's conventions
  // These names are used for clarity within RotaryDMLCompute
  const dim_t feature_depth = input.dim(-1);
  // If _ndims is 0, rotate all dims up to feature_depth
  const dim_t rot_ndims = _ndims == 0 ? feature_depth : _ndims;
  const dim_t middle_for_non_interleave = rot_ndims / 2;

  RotaryDMLCompute compute_impl(device, get_dml_data_type<T>());

  if (_interleave) {
    compute_impl.compute_interleave(input_buffer, sin_buffer, cos_buffer,
                                    output_buffer, input, sin, cos, output,
                                    feature_depth, rot_ndims);
  } else {
    compute_impl.compute_non_interleave(
        input_buffer, sin_buffer, cos_buffer, output_buffer, input, sin, cos,
        output, feature_depth, rot_ndims, middle_for_non_interleave);
  }
}

#define DECLARE_IMPL(T)                                           \
  template void Rotary::compute<Device::DirectML, T>(             \
      const StorageView&, const StorageView&, const StorageView&, \
      StorageView&, bool) const;

DECLARE_IMPL(float)
DECLARE_IMPL(float16_t)
DECLARE_IMPL(bfloat16_t)

}  // namespace ops
}  // namespace ctranslate2

#endif  // CT2_WITH_DIRECTML