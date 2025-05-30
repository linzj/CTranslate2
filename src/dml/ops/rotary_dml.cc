#ifdef CT2_WITH_DIRECTML
#include "ctranslate2/ops/rotary.h"

#include <memory>  // For std::unique_ptr
#include <vector>
#include "dml/backend_dml.h"
// #include "dml/dxdevice.h" // Included via dml_utils.h -> backend_dml.h
#include "dml/dml_utils.h"  // Added
#include "dml/operator.h"
#include "dml/operator_cache.h"

using Microsoft::WRL::ComPtr;

namespace ctranslate2 {
namespace ops {

// Local get_dml_data_type<T>(), get_element_size(), create_tensor_desc(),
// create_custom_tensor_desc() removed.

class RotaryDMLCompute {
 private:
  dml::Device* device_;     // ctranslate2::dml::Device
  DataType ct2_data_type_;  // Store original CTranslate2 DataType
  DML_TENSOR_DATA_TYPE dml_data_type_;
  UINT dml_element_size_bytes_;

 public:
  RotaryDMLCompute(dml::Device* device, DataType ct2_data_type)
      : device_(device),
        ct2_data_type_(ct2_data_type),
        dml_data_type_(dml::utils::get_dml_data_type(ct2_data_type)),
        dml_element_size_bytes_(
            dml::utils::get_dml_element_size_in_bytes(dml_data_type_)) {}

  // Helper to create DmlTensorDescBundle for various scenarios
  // Primarily for internal use by RotaryDMLCompute methods.
  dml::utils::DmlTensorDescBundle create_bundle_from_storage(
      const StorageView& storage) {
    return dml::utils::DmlTensorDescBundle(storage);
  }

  dml::utils::DmlTensorDescBundle create_bundle_custom(
      const std::vector<UINT>& sizes,
      const std::vector<UINT>* strides = nullptr,
      UINT64 total_size_bytes_override = 0) {
    UINT64 total_bytes = total_size_bytes_override;
    if (total_bytes == 0) {
      total_bytes = 1;
      bool has_zero_dim = false;
      for (UINT s : sizes) {
        if (s == 0) {
          has_zero_dim = true;
          break;
        }
        total_bytes *= s;
      }
      if (has_zero_dim)
        total_bytes = 0;
      else
        total_bytes *= dml_element_size_bytes_;
    }
    return dml::utils::DmlTensorDescBundle(dml_data_type_, sizes, strides,
                                           total_bytes);
  }

  void compute_non_interleave(
      ID3D12Resource* input_buffer,
      ID3D12Resource* sin_buffer,
      ID3D12Resource* cos_buffer,
      ID3D12Resource* output_buffer,
      const StorageView& input_sv,  // Renamed for clarity
      const StorageView& sin_sv,
      const StorageView& cos_sv,
      const StorageView& output_sv,
      dim_t feature_depth,
      dim_t rot_ndims,
      dim_t middle) {
    dml::utils::DmlTensorDescBundle input_desc_bundle =
        create_bundle_from_storage(input_sv);
    dml::utils::DmlTensorDescBundle sin_desc_bundle =
        create_bundle_from_storage(sin_sv);
    dml::utils::DmlTensorDescBundle cos_desc_bundle =
        create_bundle_from_storage(cos_sv);
    dml::utils::DmlTensorDescBundle output_desc_bundle =
        create_bundle_from_storage(output_sv);

    const std::vector<UINT>& output_dml_shape =
        output_desc_bundle.get_sizes_vec();

    // Step 1: Compute x * cos -> temp1
    auto temp1_buffer_res = device_->CreatePreferredDeviceMemoryBuffer(
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    device_->KeepAliveUntilNextCommandListDispatch(temp1_buffer_res);

    multiply_tensors(
        input_buffer, cos_buffer, temp1_buffer_res.Get(),
        input_desc_bundle.get_tensor_desc(), cos_desc_bundle.get_tensor_desc(),
        output_desc_bundle.get_tensor_desc());  // output_desc here matches
                                                // temp1_buffer layout

    if (rot_ndims > 0) {
      dim_t batch_elements = 1;
      const auto& input_dml_shape = input_desc_bundle.get_sizes_vec();
      for (size_t i = 0; i < input_dml_shape.size() - 1; ++i) {
        batch_elements *= input_dml_shape[i];
      }
      auto half_rot_ndims_size_bytes =
          batch_elements * middle * dml_element_size_bytes_;

      auto x_first_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto x_second_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(x_first_half_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(x_second_half_buffer);

      std::vector<UINT> rot_sin_cos_dml_shape = sin_desc_bundle.get_sizes_vec();
      if (!rot_sin_cos_dml_shape.empty())
        rot_sin_cos_dml_shape.back() = static_cast<UINT>(rot_ndims);

      auto sin_first_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto sin_second_half_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(sin_first_half_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(sin_second_half_buffer);

      std::vector<UINT> x_slice_shape_for_rot_dml = input_dml_shape;
      if (!x_slice_shape_for_rot_dml.empty())
        x_slice_shape_for_rot_dml.back() = static_cast<UINT>(rot_ndims);

      std::vector<UINT> half_rot_dml_shape = x_slice_shape_for_rot_dml;
      if (!half_rot_dml_shape.empty())
        half_rot_dml_shape.back() = static_cast<UINT>(middle);

      auto x_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * dml_element_size_bytes_);
      device_->KeepAliveUntilNextCommandListDispatch(x_rot_part_buffer);

      slice_tensor_last_dim(input_buffer, x_rot_part_buffer.Get(),
                            input_dml_shape, x_slice_shape_for_rot_dml, 0,
                            rot_ndims);
      slice_tensor_last_dim(x_rot_part_buffer.Get(), x_first_half_buffer.Get(),
                            x_slice_shape_for_rot_dml, half_rot_dml_shape, 0,
                            middle);
      slice_tensor_last_dim(x_rot_part_buffer.Get(), x_second_half_buffer.Get(),
                            x_slice_shape_for_rot_dml, half_rot_dml_shape,
                            middle, middle);

      slice_tensor_last_dim(sin_buffer, sin_first_half_buffer.Get(),
                            rot_sin_cos_dml_shape, half_rot_dml_shape, 0,
                            middle);
      slice_tensor_last_dim(sin_buffer, sin_second_half_buffer.Get(),
                            rot_sin_cos_dml_shape, half_rot_dml_shape, middle,
                            middle);

      auto rot_first_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      auto rot_second_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(rot_first_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(rot_second_buffer);

      dml::utils::DmlTensorDescBundle half_desc_bundle = create_bundle_custom(
          half_rot_dml_shape, nullptr, half_rot_ndims_size_bytes);

      auto neg_x_second_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(half_rot_ndims_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(neg_x_second_buffer);
      negate_tensor(x_second_half_buffer.Get(), neg_x_second_buffer.Get(),
                    half_desc_bundle.get_tensor_desc());
      multiply_tensors(neg_x_second_buffer.Get(), sin_first_half_buffer.Get(),
                       rot_first_buffer.Get(),
                       half_desc_bundle.get_tensor_desc(),
                       half_desc_bundle.get_tensor_desc(),
                       half_desc_bundle.get_tensor_desc());
      multiply_tensors(x_first_half_buffer.Get(), sin_second_half_buffer.Get(),
                       rot_second_buffer.Get(),
                       half_desc_bundle.get_tensor_desc(),
                       half_desc_bundle.get_tensor_desc(),
                       half_desc_bundle.get_tensor_desc());

      add_sliced_to_full_tensor(temp1_buffer_res.Get(), rot_first_buffer.Get(),
                                output_dml_shape, half_rot_dml_shape, 0,
                                middle);
      add_sliced_to_full_tensor(temp1_buffer_res.Get(), rot_second_buffer.Get(),
                                output_dml_shape, half_rot_dml_shape, middle,
                                middle);
    }

    if (rot_ndims < feature_depth) {
      copy_unchanged_elements(input_buffer, temp1_buffer_res.Get(),
                              output_buffer, input_desc_bundle.get_sizes_vec(),
                              output_dml_shape, rot_ndims, feature_depth);
    } else {
      copy_tensor(temp1_buffer_res.Get(), output_buffer,
                  output_desc_bundle.get_tensor_desc());
    }
  }

  void compute_interleave(ID3D12Resource* input_buffer,
                          ID3D12Resource* sin_buffer,
                          ID3D12Resource* cos_buffer,
                          ID3D12Resource* output_buffer,
                          const StorageView& input_sv,
                          const StorageView& sin_sv,
                          const StorageView& cos_sv,
                          const StorageView& output_sv,
                          dim_t feature_depth,
                          dim_t rot_ndims) {
    dml::utils::DmlTensorDescBundle input_desc_bundle =
        create_bundle_from_storage(input_sv);
    dml::utils::DmlTensorDescBundle sin_desc_bundle =
        create_bundle_from_storage(sin_sv);
    dml::utils::DmlTensorDescBundle cos_desc_bundle =
        create_bundle_from_storage(cos_sv);
    dml::utils::DmlTensorDescBundle output_desc_bundle =
        create_bundle_from_storage(output_sv);
    const auto& input_dml_shape = input_desc_bundle.get_sizes_vec();
    const auto& output_dml_shape = output_desc_bundle.get_sizes_vec();
    const auto& sin_dml_shape = sin_desc_bundle.get_sizes_vec();

    auto temp1_buffer_res = device_->CreatePreferredDeviceMemoryBuffer(
        output_desc_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    device_->KeepAliveUntilNextCommandListDispatch(temp1_buffer_res);
    multiply_tensors(input_buffer, cos_buffer, temp1_buffer_res.Get(),
                     input_desc_bundle.get_tensor_desc(),
                     cos_desc_bundle.get_tensor_desc(),
                     output_desc_bundle.get_tensor_desc());

    if (rot_ndims > 0) {
      dim_t batch_elements = 1;
      for (size_t i = 0; i < input_dml_shape.size() - 1; ++i) {
        batch_elements *= input_dml_shape[i];
      }
      auto even_count = (rot_ndims + 1) / 2;
      auto odd_count = rot_ndims / 2;

      auto even_size_bytes =
          batch_elements * even_count * dml_element_size_bytes_;
      auto odd_size_bytes =
          batch_elements * odd_count * dml_element_size_bytes_;

      auto x_even_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(even_size_bytes);
      auto x_odd_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(odd_size_bytes);
      auto sin_even_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(even_size_bytes);
      auto sin_odd_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(odd_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(x_even_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(x_odd_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(sin_even_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(sin_odd_buffer);

      std::vector<UINT> x_rot_part_dml_shape = input_dml_shape;
      if (!x_rot_part_dml_shape.empty())
        x_rot_part_dml_shape.back() = static_cast<UINT>(rot_ndims);

      std::vector<UINT> sin_cos_rot_part_dml_shape = sin_dml_shape;
      if (!sin_cos_rot_part_dml_shape.empty())
        sin_cos_rot_part_dml_shape.back() = static_cast<UINT>(rot_ndims);

      auto x_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * dml_element_size_bytes_);
      device_->KeepAliveUntilNextCommandListDispatch(x_rot_part_buffer);
      slice_tensor_last_dim(input_buffer, x_rot_part_buffer.Get(),
                            input_dml_shape, x_rot_part_dml_shape, 0,
                            rot_ndims);

      auto sin_rot_part_buffer = device_->CreatePreferredDeviceMemoryBuffer(
          batch_elements * rot_ndims * dml_element_size_bytes_);
      device_->KeepAliveUntilNextCommandListDispatch(sin_rot_part_buffer);
      slice_tensor_last_dim(sin_buffer, sin_rot_part_buffer.Get(),
                            sin_dml_shape, sin_cos_rot_part_dml_shape, 0,
                            rot_ndims);

      std::vector<UINT> even_slice_dml_shape = x_rot_part_dml_shape;
      if (!even_slice_dml_shape.empty())
        even_slice_dml_shape.back() = static_cast<UINT>(even_count);
      std::vector<UINT> odd_slice_dml_shape = x_rot_part_dml_shape;
      if (!odd_slice_dml_shape.empty())
        odd_slice_dml_shape.back() = static_cast<UINT>(odd_count);

      extract_even_odd_elements(x_rot_part_buffer.Get(), x_even_buffer.Get(),
                                x_odd_buffer.Get(), x_rot_part_dml_shape,
                                rot_ndims, even_count, odd_count);
      extract_even_odd_elements(sin_rot_part_buffer.Get(),
                                sin_even_buffer.Get(), sin_odd_buffer.Get(),
                                sin_cos_rot_part_dml_shape, rot_ndims,
                                even_count, odd_count);

      auto rot_term_for_even_pos_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(even_size_bytes);
      auto rot_term_for_odd_pos_buffer =
          device_->CreatePreferredDeviceMemoryBuffer(odd_size_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(
          rot_term_for_even_pos_buffer);
      device_->KeepAliveUntilNextCommandListDispatch(
          rot_term_for_odd_pos_buffer);

      dml::utils::DmlTensorDescBundle even_slice_desc_bundle =
          create_bundle_custom(even_slice_dml_shape, nullptr, even_size_bytes);
      dml::utils::DmlTensorDescBundle odd_slice_desc_bundle =
          create_bundle_custom(odd_slice_dml_shape, nullptr, odd_size_bytes);

      if (even_count > 0 &&
          odd_count > 0) {  // Ensure odd_count > 0 for x_odd_buffer usage
        compute_interleave_rotation_even(
            x_odd_buffer.Get(), sin_even_buffer.Get(),
            rot_term_for_even_pos_buffer.Get(),
            even_slice_desc_bundle.get_tensor_desc(),
            std::min(even_count, odd_count));
      } else if (even_count > 0 &&
                 rot_ndims ==
                     1) {  // Special case: rot_ndims=1, only x[0]cos[0] -
                           // 0*sin[0] happens, effectively handled if
                           // compute_interleave_rotation_even is skipped
        // Or if compute_interleave_rotation_even is robust to zero size
        // odd_count passed to min
      }

      if (odd_count > 0) {
        compute_interleave_rotation_odd(
            x_even_buffer.Get(), sin_odd_buffer.Get(),
            rot_term_for_odd_pos_buffer.Get(),
            odd_slice_desc_bundle.get_tensor_desc(), odd_count);
      }

      combine_even_odd_to_interleaved_additive(
          temp1_buffer_res.Get(), rot_term_for_even_pos_buffer.Get(),
          rot_term_for_odd_pos_buffer.Get(), output_dml_shape,
          even_slice_dml_shape, odd_slice_dml_shape, rot_ndims, even_count,
          odd_count);
    }

    if (rot_ndims < feature_depth) {
      copy_unchanged_elements(input_buffer, temp1_buffer_res.Get(),
                              output_buffer, input_dml_shape, output_dml_shape,
                              rot_ndims, feature_depth);
    } else {
      copy_tensor(temp1_buffer_res.Get(), output_buffer,
                  output_desc_bundle.get_tensor_desc());
    }
  }

 private:  // Helper methods using DML ops. Bindings need to use dml::utils
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
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_MULTIPLY,
                                 &multiply_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[3];
    bindings_storage[0] = dml::utils::create_buffer_binding(a_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(b_buffer);
    bindings_storage[2] = dml::utils::create_buffer_binding(output_buffer);

    std::vector<DML_BINDING_DESC> inputs = {
        dml::utils::create_binding_desc(&bindings_storage[0]),
        dml::utils::create_binding_desc(&bindings_storage[1])};
    std::vector<DML_BINDING_DESC> outputs = {
        dml::utils::create_binding_desc(&bindings_storage[2])};
    compiled_op->Execute(inputs, outputs);
  }

  void negate_tensor(ID3D12Resource* input_buffer,
                     ID3D12Resource* output_buffer,
                     const DML_TENSOR_DESC& tensor_desc) {
    DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC negate_desc = {};
    negate_desc.InputTensor = &tensor_desc;
    negate_desc.OutputTensor = &tensor_desc;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_NEGATE,
                                 &negate_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[2];
    bindings_storage[0] = dml::utils::create_buffer_binding(input_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(output_buffer);
    compiled_op->Execute(
        {dml::utils::create_binding_desc(&bindings_storage[0])},
        {dml::utils::create_binding_desc(&bindings_storage[1])});
  }

  void copy_tensor(ID3D12Resource* input_buffer,
                   ID3D12Resource* output_buffer,
                   const DML_TENSOR_DESC& tensor_desc) {
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc = {};
    identity_desc.InputTensor = &tensor_desc;
    identity_desc.OutputTensor = &tensor_desc;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY,
                                 &identity_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[2];
    bindings_storage[0] = dml::utils::create_buffer_binding(input_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(output_buffer);
    compiled_op->Execute(
        {dml::utils::create_binding_desc(&bindings_storage[0])},
        {dml::utils::create_binding_desc(&bindings_storage[1])});
  }

  void slice_tensor_last_dim(ID3D12Resource* input_buffer,
                             ID3D12Resource* output_buffer,
                             const std::vector<UINT>& input_total_shape_dml,
                             const std::vector<UINT>& output_slice_shape_dml,
                             dim_t start_idx,
                             dim_t size_on_last_dim) {
    dml::utils::DmlTensorDescBundle input_full_desc =
        create_bundle_custom(input_total_shape_dml);
    dml::utils::DmlTensorDescBundle output_slice_desc =
        create_bundle_custom(output_slice_shape_dml);

    std::vector<UINT> offsets(input_total_shape_dml.size(), 0);
    std::vector<UINT> sizes = input_total_shape_dml;
    std::vector<UINT> strides(input_total_shape_dml.size(), 1);
    if (!offsets.empty())
      offsets.back() = static_cast<UINT>(start_idx);
    if (!sizes.empty())
      sizes.back() = static_cast<UINT>(size_on_last_dim);

    DML_SLICE_OPERATOR_DESC slice_desc = {};
    slice_desc.InputTensor = &input_full_desc.get_tensor_desc();
    slice_desc.OutputTensor = &output_slice_desc.get_tensor_desc();
    slice_desc.DimensionCount = static_cast<UINT>(input_total_shape_dml.size());
    slice_desc.Offsets = offsets.data();
    slice_desc.Sizes = sizes.data();
    slice_desc.Strides = strides.data();
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE, &slice_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[2];
    bindings_storage[0] = dml::utils::create_buffer_binding(input_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(output_buffer);
    compiled_op->Execute(
        {dml::utils::create_binding_desc(&bindings_storage[0])},
        {dml::utils::create_binding_desc(&bindings_storage[1])});
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
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[3];
    bindings_storage[0] = dml::utils::create_buffer_binding(a_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(b_buffer);
    bindings_storage[2] = dml::utils::create_buffer_binding(output_buffer);
    std::vector<DML_BINDING_DESC> inputs = {
        dml::utils::create_binding_desc(&bindings_storage[0]),
        dml::utils::create_binding_desc(&bindings_storage[1])};
    std::vector<DML_BINDING_DESC> outputs = {
        dml::utils::create_binding_desc(&bindings_storage[2])};
    compiled_op->Execute(inputs, outputs);
  }

  void scatter_slice_to_full_tensor(
      ID3D12Resource* full_buffer_to_update,
      ID3D12Resource* slice_data_to_scatter,
      const std::vector<UINT>& full_shape_dml,
      const std::vector<UINT>& slice_shape_dml,
      dim_t start_idx_on_last_dim,
      dim_t size_on_last_dim) {  // size_on_last_dim is count of elements in
                                 // slice's last dim
    dim_t batch_elements = 1;
    for (size_t i = 0; i < full_shape_dml.size() - 1; ++i) {
      batch_elements *= full_shape_dml[i];
    }

    auto num_indices = batch_elements * size_on_last_dim;
    auto indices_buffer_size_bytes = num_indices * sizeof(UINT32);
    ComPtr<ID3D12Resource> indices_buffer_gpu;

    std::vector<UINT32> indices_data_cpu;
    indices_data_cpu.reserve(num_indices);
    for (dim_t batch = 0; batch < batch_elements; ++batch) {
      for (dim_t i = 0; i < size_on_last_dim; ++i) {
        indices_data_cpu.push_back(static_cast<UINT32>(
            batch * full_shape_dml.back() + start_idx_on_last_dim + i));
      }
    }
    indices_buffer_gpu = device_->Upload(
        indices_buffer_size_bytes,
        std::string_view(reinterpret_cast<const char*>(
                             static_cast<const void*>(indices_data_cpu.data())),
                         indices_buffer_size_bytes));
    device_->KeepAliveUntilNextCommandListDispatch(indices_buffer_gpu);

    std::vector<UINT> flat_full_dml_shape = {
        static_cast<UINT>(batch_elements * full_shape_dml.back())};
    std::vector<UINT> flat_slice_dml_shape = {static_cast<UINT>(num_indices)};
    std::vector<UINT> flat_indices_dml_shape = {static_cast<UINT>(num_indices)};

    dml::utils::DmlTensorDescBundle full_desc_bundle =
        create_bundle_custom(flat_full_dml_shape);
    dml::utils::DmlTensorDescBundle slice_desc_bundle =
        create_bundle_custom(flat_slice_dml_shape);
    dml::utils::DmlTensorDescBundle indices_desc_bundle(
        DML_TENSOR_DATA_TYPE_UINT32, flat_indices_dml_shape, nullptr,
        indices_buffer_size_bytes);

    DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
    scatter_desc.InputTensor =
        &full_desc_bundle.get_tensor_desc();  // Input to scatter is the tensor
                                              // being updated
    scatter_desc.IndicesTensor = &indices_desc_bundle.get_tensor_desc();
    scatter_desc.UpdatesTensor =
        &slice_desc_bundle.get_tensor_desc();  // Data to write
    scatter_desc.OutputTensor =
        &full_desc_bundle.get_tensor_desc();  // Output is the updated tensor
    scatter_desc.Axis = 0;  // Scatter into flattened 1D tensor
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SCATTER_ELEMENTS, &scatter_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[4];  // input, indices, updates, output
    bindings_storage[0] = dml::utils::create_buffer_binding(
        full_buffer_to_update);  // Input binding for SCATTER_ELEMENTS is the
                                 // "current state"
    bindings_storage[1] =
        dml::utils::create_buffer_binding(indices_buffer_gpu.Get());
    bindings_storage[2] =
        dml::utils::create_buffer_binding(slice_data_to_scatter);
    bindings_storage[3] = dml::utils::create_buffer_binding(
        full_buffer_to_update);  // Output binding is the same buffer

    std::vector<DML_BINDING_DESC> scatter_inputs = {
        dml::utils::create_binding_desc(&bindings_storage[0]),
        dml::utils::create_binding_desc(&bindings_storage[1]),
        dml::utils::create_binding_desc(&bindings_storage[2])};
    std::vector<DML_BINDING_DESC> scatter_outputs = {
        dml::utils::create_binding_desc(&bindings_storage[3])};
    compiled_op->Execute(scatter_inputs, scatter_outputs);
  }

  void add_sliced_to_full_tensor(ID3D12Resource* full_buffer_to_update,
                                 ID3D12Resource* slice_to_add,
                                 const std::vector<UINT>& full_shape_dml,
                                 const std::vector<UINT>& slice_shape_dml,
                                 dim_t start_idx_on_last_dim,
                                 dim_t size_on_last_dim) {
    UINT64 slice_total_bytes = 1;
    bool slice_empty = false;
    for (UINT s : slice_shape_dml) {
      if (s == 0) {
        slice_empty = true;
        break;
      }
      slice_total_bytes *= s;
    }
    if (slice_empty)
      slice_total_bytes = 0;
    else
      slice_total_bytes *= dml_element_size_bytes_;
    if (slice_total_bytes == 0)
      return;

    auto current_slice_from_full_buffer =
        device_->CreatePreferredDeviceMemoryBuffer(slice_total_bytes);
    device_->KeepAliveUntilNextCommandListDispatch(
        current_slice_from_full_buffer);
    slice_tensor_last_dim(full_buffer_to_update,
                          current_slice_from_full_buffer.Get(), full_shape_dml,
                          slice_shape_dml, start_idx_on_last_dim,
                          size_on_last_dim);

    auto result_added_slice_buffer =
        device_->CreatePreferredDeviceMemoryBuffer(slice_total_bytes);
    device_->KeepAliveUntilNextCommandListDispatch(result_added_slice_buffer);

    dml::utils::DmlTensorDescBundle slice_desc_for_add_bundle =
        create_bundle_custom(slice_shape_dml, nullptr, slice_total_bytes);

    add_tensors(current_slice_from_full_buffer.Get(), slice_to_add,
                result_added_slice_buffer.Get(),
                slice_desc_for_add_bundle.get_tensor_desc(),
                slice_desc_for_add_bundle.get_tensor_desc(),
                slice_desc_for_add_bundle.get_tensor_desc());

    scatter_slice_to_full_tensor(
        full_buffer_to_update, result_added_slice_buffer.Get(), full_shape_dml,
        slice_shape_dml, start_idx_on_last_dim, size_on_last_dim);
  }

  void copy_unchanged_elements(
      ID3D12Resource* original_input_buffer,
      ID3D12Resource* computed_rotated_buffer,  // Holds the result for rotated
                                                // part (0 to rot_ndims-1)
      ID3D12Resource* final_output_buffer,  // Final output being constructed
      const std::vector<UINT>&
          input_full_dml_shape,  // Full shape of original_input_buffer &
                                 // final_output_buffer
      const std::vector<UINT>&,  // output_full_dml_shape (same as
                                 // input_full_dml_shape)
      dim_t rot_ndims,
      dim_t feature_depth) {
    // Copy the already computed rotated part (0 to rot_ndims-1) from
    // computed_rotated_buffer to final_output_buffer
    if (rot_ndims > 0) {
      std::vector<UINT> rotated_part_dml_shape = input_full_dml_shape;
      if (!rotated_part_dml_shape.empty())
        rotated_part_dml_shape.back() = static_cast<UINT>(rot_ndims);

      UINT64 rotated_part_bytes = 1;
      bool rotated_empty = false;
      for (UINT s : rotated_part_dml_shape) {
        if (s == 0) {
          rotated_empty = true;
          break;
        }
        rotated_part_bytes *= s;
      }
      if (rotated_empty)
        rotated_part_bytes = 0;
      else
        rotated_part_bytes *= dml_element_size_bytes_;

      if (rotated_part_bytes > 0) {  // only copy if there is data
        auto temp_rotated_part_buffer =
            device_->CreatePreferredDeviceMemoryBuffer(rotated_part_bytes);
        device_->KeepAliveUntilNextCommandListDispatch(
            temp_rotated_part_buffer);
        slice_tensor_last_dim(
            computed_rotated_buffer, temp_rotated_part_buffer.Get(),
            input_full_dml_shape, rotated_part_dml_shape, 0, rot_ndims);
        scatter_slice_to_full_tensor(
            final_output_buffer, temp_rotated_part_buffer.Get(),
            input_full_dml_shape, rotated_part_dml_shape, 0, rot_ndims);
      }
    }

    // Copy unchanged part (rot_ndims to feature_depth-1) from
    // original_input_buffer to final_output_buffer
    if (feature_depth > rot_ndims) {
      dim_t unchanged_size = feature_depth - rot_ndims;
      std::vector<UINT> unchanged_part_dml_shape = input_full_dml_shape;
      if (!unchanged_part_dml_shape.empty())
        unchanged_part_dml_shape.back() = static_cast<UINT>(unchanged_size);

      UINT64 unchanged_part_bytes = 1;
      bool unchanged_empty = false;
      for (UINT s : unchanged_part_dml_shape) {
        if (s == 0) {
          unchanged_empty = true;
          break;
        }
        unchanged_part_bytes *= s;
      }
      if (unchanged_empty)
        unchanged_part_bytes = 0;
      else
        unchanged_part_bytes *= dml_element_size_bytes_;

      if (unchanged_part_bytes > 0) {
        auto temp_unchanged_part_buffer =
            device_->CreatePreferredDeviceMemoryBuffer(unchanged_part_bytes);
        device_->KeepAliveUntilNextCommandListDispatch(
            temp_unchanged_part_buffer);
        slice_tensor_last_dim(original_input_buffer,
                              temp_unchanged_part_buffer.Get(),
                              input_full_dml_shape, unchanged_part_dml_shape,
                              rot_ndims, unchanged_size);
        scatter_slice_to_full_tensor(
            final_output_buffer, temp_unchanged_part_buffer.Get(),
            input_full_dml_shape, unchanged_part_dml_shape, rot_ndims,
            unchanged_size);
      }
    }
  }

  void extract_even_odd_elements(
      ID3D12Resource* input_buffer,  // Contains elements up to rot_ndims
      ID3D12Resource* even_buffer,   // Output for even indexed elements
      ID3D12Resource* odd_buffer,    // Output for odd indexed elements
      const std::vector<UINT>&
          input_rot_part_dml_shape,  // Shape of input_buffer (e.g.
                                     // [B,H,S,rot_ndims])
      dim_t rot_ndims,  // Last dimension size of input_buffer relevant for
                        // even/odd extraction
      dim_t even_count,
      dim_t odd_count) {
    if (even_count > 0) {
      std::vector<UINT> even_slice_dml_shape = input_rot_part_dml_shape;
      if (!even_slice_dml_shape.empty())
        even_slice_dml_shape.back() = static_cast<UINT>(even_count);
      extract_strided_elements(input_buffer, even_buffer,
                               input_rot_part_dml_shape, even_slice_dml_shape,
                               0, 2);
    }
    if (odd_count > 0) {
      std::vector<UINT> odd_slice_dml_shape = input_rot_part_dml_shape;
      if (!odd_slice_dml_shape.empty())
        odd_slice_dml_shape.back() = static_cast<UINT>(odd_count);
      extract_strided_elements(input_buffer, odd_buffer,
                               input_rot_part_dml_shape, odd_slice_dml_shape, 1,
                               2);
    }
  }

  void extract_strided_elements(
      ID3D12Resource* input_buffer,
      ID3D12Resource* output_buffer,
      const std::vector<UINT>& input_shape_dml,  // Full shape of input_buffer
      const std::vector<UINT>&
          output_shape_dml,  // Shape of output_buffer (e.g. [B,H,S,even_count])
      dim_t start_offset_on_last_dim,  // 0 for even, 1 for odd
      dim_t stride_on_last_dim) {      // 2 for interleaved

    dml::utils::DmlTensorDescBundle input_desc_bundle =
        create_bundle_custom(input_shape_dml);
    dml::utils::DmlTensorDescBundle output_desc_bundle =
        create_bundle_custom(output_shape_dml);

    std::vector<UINT> window_offsets(input_shape_dml.size(), 0);
    std::vector<UINT> window_sizes =
        output_shape_dml;  // OutputTensor sizes must match InputWindowSizes.
    std::vector<INT> window_strides(input_shape_dml.size(), 1);
    // For dimensions other than the slice axis, stride must be 1.
    // The effective number of elements copied from input must match elements in
    // output.

    if (!input_shape_dml.empty()) {
      window_offsets.back() = static_cast<UINT>(start_offset_on_last_dim);
      // Window_sizes.back() should be output_shape_dml.back() which is
      // even_count or odd_count.
      window_strides.back() = static_cast<INT>(stride_on_last_dim);
    }

    DML_SLICE1_OPERATOR_DESC slice_desc = {};
    slice_desc.InputTensor = &input_desc_bundle.get_tensor_desc();
    slice_desc.OutputTensor = &output_desc_bundle.get_tensor_desc();
    slice_desc.DimensionCount = static_cast<UINT>(input_shape_dml.size());
    slice_desc.InputWindowOffsets = window_offsets.data();
    slice_desc.InputWindowSizes =
        window_sizes.data();  // Must be same as outputTensor sizes.
    slice_desc.InputWindowStrides = window_strides.data();
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SLICE1, &slice_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[2];
    bindings_storage[0] = dml::utils::create_buffer_binding(input_buffer);
    bindings_storage[1] = dml::utils::create_buffer_binding(output_buffer);
    compiled_op->Execute(
        {dml::utils::create_binding_desc(&bindings_storage[0])},
        {dml::utils::create_binding_desc(&bindings_storage[1])});
  }

  void compute_interleave_rotation_even(
      ID3D12Resource* x_odd_buffer,
      ID3D12Resource* sin_even_buffer,
      ID3D12Resource* output_rot_term_buffer,
      const DML_TENSOR_DESC&
          output_op_desc,  // Describes output_rot_term_buffer, implies op_count
      dim_t
          actual_op_count) {  // Unused due to DML_TENSOR_DESC controlling size
    if (static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
            ->Sizes[static_cast<const DML_BUFFER_TENSOR_DESC*>(
                        output_op_desc.Desc)
                        ->DimensionCount -
                    1] == 0)
      return;

    dml::utils::DmlTensorDescBundle x_odd_bundle =
        create_bundle_custom(  // x_odd has shape corresponding to
                               // output_op_desc
            std::vector<UINT>(
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->Sizes,
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                        ->Sizes +
                    static_cast<const DML_BUFFER_TENSOR_DESC*>(
                        output_op_desc.Desc)
                        ->DimensionCount));
    dml::utils::DmlTensorDescBundle sin_even_bundle =
        create_bundle_custom(  // sin_even has shape corresponding to
                               // output_op_desc
            std::vector<UINT>(
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->Sizes,
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                        ->Sizes +
                    static_cast<const DML_BUFFER_TENSOR_DESC*>(
                        output_op_desc.Desc)
                        ->DimensionCount));

    auto neg_buffer = device_->CreatePreferredDeviceMemoryBuffer(
        x_odd_bundle.get_buffer_desc().TotalTensorSizeInBytes);
    device_->KeepAliveUntilNextCommandListDispatch(neg_buffer);
    negate_tensor(
        x_odd_buffer, neg_buffer.Get(),
        x_odd_bundle.get_tensor_desc());  // Negate x_odd (using its inferred
                                          // descriptor from output_op_desc)
    multiply_tensors(
        neg_buffer.Get(), sin_even_buffer, output_rot_term_buffer,
        x_odd_bundle.get_tensor_desc(),     // use desc matching x_odd_buffer's
                                            // content size after negation
        sin_even_bundle.get_tensor_desc(),  // use desc matching
                                            // sin_even_buffer's content size
        output_op_desc);  // Output uses the provided descriptor
  }

  void compute_interleave_rotation_odd(
      ID3D12Resource* x_even_buffer,
      ID3D12Resource* sin_odd_buffer,
      ID3D12Resource* output_rot_term_buffer,
      const DML_TENSOR_DESC&
          output_op_desc,  // Describes output_rot_term_buffer, implies op_count
      dim_t actual_op_count) {  // Unused
    if (static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
            ->Sizes[static_cast<const DML_BUFFER_TENSOR_DESC*>(
                        output_op_desc.Desc)
                        ->DimensionCount -
                    1] == 0)
      return;

    dml::utils::DmlTensorDescBundle x_even_bundle =
        create_bundle_custom(std::vector<UINT>(
            static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                ->Sizes,
            static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->Sizes +
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->DimensionCount));
    dml::utils::DmlTensorDescBundle sin_odd_bundle =
        create_bundle_custom(std::vector<UINT>(
            static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                ->Sizes,
            static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->Sizes +
                static_cast<const DML_BUFFER_TENSOR_DESC*>(output_op_desc.Desc)
                    ->DimensionCount));

    multiply_tensors(x_even_buffer, sin_odd_buffer, output_rot_term_buffer,
                     x_even_bundle.get_tensor_desc(),
                     sin_odd_bundle.get_tensor_desc(), output_op_desc);
  }

  void combine_even_odd_to_interleaved_additive(
      ID3D12Resource* base_buffer_to_update,
      ID3D12Resource* even_rot_terms_to_add,
      ID3D12Resource* odd_rot_terms_to_add,
      const std::vector<UINT>& base_full_dml_shape,
      const std::vector<UINT>& even_slice_dml_shape,
      const std::vector<UINT>& odd_slice_dml_shape,
      dim_t rot_ndims,  // Total rotated dimension
      dim_t even_count,
      dim_t odd_count) {
    if (even_count > 0) {
      UINT64 even_slice_bytes = 1;
      for (auto d : even_slice_dml_shape)
        even_slice_bytes *= d;
      even_slice_bytes *= dml_element_size_bytes_;

      auto current_even_slice_from_base =
          device_->CreatePreferredDeviceMemoryBuffer(even_slice_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(
          current_even_slice_from_base);
      extract_strided_elements(base_buffer_to_update,
                               current_even_slice_from_base.Get(),
                               base_full_dml_shape, even_slice_dml_shape, 0, 2);

      auto result_added_even_slice =
          device_->CreatePreferredDeviceMemoryBuffer(even_slice_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(result_added_even_slice);
      dml::utils::DmlTensorDescBundle even_slice_bundle =
          create_bundle_custom(even_slice_dml_shape, nullptr, even_slice_bytes);

      add_tensors(current_even_slice_from_base.Get(), even_rot_terms_to_add,
                  result_added_even_slice.Get(),
                  even_slice_bundle.get_tensor_desc(),
                  even_slice_bundle.get_tensor_desc(),
                  even_slice_bundle.get_tensor_desc());

      scatter_strided_to_full_tensor_overwrite(
          base_buffer_to_update, result_added_even_slice.Get(),
          base_full_dml_shape, even_slice_dml_shape, 0, 2, even_count);
    }

    if (odd_count > 0) {
      UINT64 odd_slice_bytes = 1;
      for (auto d : odd_slice_dml_shape)
        odd_slice_bytes *= d;
      odd_slice_bytes *= dml_element_size_bytes_;

      auto current_odd_slice_from_base =
          device_->CreatePreferredDeviceMemoryBuffer(odd_slice_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(
          current_odd_slice_from_base);
      extract_strided_elements(base_buffer_to_update,
                               current_odd_slice_from_base.Get(),
                               base_full_dml_shape, odd_slice_dml_shape, 1, 2);

      auto result_added_odd_slice =
          device_->CreatePreferredDeviceMemoryBuffer(odd_slice_bytes);
      device_->KeepAliveUntilNextCommandListDispatch(result_added_odd_slice);
      dml::utils::DmlTensorDescBundle odd_slice_bundle =
          create_bundle_custom(odd_slice_dml_shape, nullptr, odd_slice_bytes);

      add_tensors(current_odd_slice_from_base.Get(), odd_rot_terms_to_add,
                  result_added_odd_slice.Get(),
                  odd_slice_bundle.get_tensor_desc(),
                  odd_slice_bundle.get_tensor_desc(),
                  odd_slice_bundle.get_tensor_desc());

      scatter_strided_to_full_tensor_overwrite(
          base_buffer_to_update, result_added_odd_slice.Get(),
          base_full_dml_shape, odd_slice_dml_shape, 1, 2, odd_count);
    }
  }

  void scatter_strided_to_full_tensor_overwrite(
      ID3D12Resource* full_buffer_to_update,
      ID3D12Resource* strided_data_to_scatter,
      const std::vector<UINT>& full_shape_dml,
      const std::vector<UINT>&
          strided_data_shape_dml,               // e.g. [B, H, S, even_count]
      dim_t start_offset_on_last_dim,           // 0 for even, 1 for odd
      dim_t dml_stride_on_last_dim,             // 2 for interleaved
      dim_t count_on_last_dim_for_strided_data  // even_count or odd_count
  ) {
    if (count_on_last_dim_for_strided_data == 0)
      return;

    dim_t batch_elements = 1;  // Elements before the last dimension
    for (size_t i = 0; i < full_shape_dml.size() - 1; ++i) {
      batch_elements *= full_shape_dml[i];
    }

    auto num_indices_to_scatter =
        batch_elements * count_on_last_dim_for_strided_data;
    auto indices_buffer_size_bytes = num_indices_to_scatter * sizeof(UINT32);
    ComPtr<ID3D12Resource> indices_buffer_gpu;

    std::vector<UINT32> indices_data_cpu;
    indices_data_cpu.reserve(num_indices_to_scatter);
    for (dim_t batch_idx = 0; batch_idx < batch_elements; ++batch_idx) {
      for (dim_t i = 0; i < count_on_last_dim_for_strided_data; ++i) {
        indices_data_cpu.push_back(static_cast<UINT32>(
            batch_idx * full_shape_dml.back() + start_offset_on_last_dim +
            i * dml_stride_on_last_dim));
      }
    }
    indices_buffer_gpu = device_->Upload(
        indices_buffer_size_bytes,
        std::string_view(reinterpret_cast<const char*>(
                             static_cast<const void*>(indices_data_cpu.data())),
                         indices_buffer_size_bytes));
    device_->KeepAliveUntilNextCommandListDispatch(indices_buffer_gpu);

    // Scatter operates on flattened input/updates/indices based on axis.
    // Here Axis=0 for flattened.
    std::vector<UINT> flat_full_dml_shape = {
        static_cast<UINT>(batch_elements * full_shape_dml.back())};
    std::vector<UINT> flat_updates_dml_shape = {
        static_cast<UINT>(num_indices_to_scatter)};
    std::vector<UINT> flat_indices_dml_shape = {
        static_cast<UINT>(num_indices_to_scatter)};

    dml::utils::DmlTensorDescBundle input_for_scatter_desc =
        create_bundle_custom(
            flat_full_dml_shape);  // Describes full_buffer_to_update (as Input)
    dml::utils::DmlTensorDescBundle updates_desc = create_bundle_custom(
        flat_updates_dml_shape);  // Describes strided_data_to_scatter
    dml::utils::DmlTensorDescBundle indices_desc(
        DML_TENSOR_DATA_TYPE_UINT32, flat_indices_dml_shape, nullptr,
        indices_buffer_size_bytes);
    dml::utils::DmlTensorDescBundle output_for_scatter_desc =
        create_bundle_custom(
            flat_full_dml_shape);  // Describes full_buffer_to_update (as
                                   // Output)

    DML_SCATTER_ELEMENTS_OPERATOR_DESC scatter_desc = {};
    scatter_desc.InputTensor = &input_for_scatter_desc.get_tensor_desc();
    scatter_desc.IndicesTensor = &indices_desc.get_tensor_desc();
    scatter_desc.UpdatesTensor = &updates_desc.get_tensor_desc();
    scatter_desc.OutputTensor = &output_for_scatter_desc.get_tensor_desc();
    scatter_desc.Axis = 0;
    DML_OPERATOR_DESC op_desc = {DML_OPERATOR_SCATTER_ELEMENTS, &scatter_desc};
    auto* compiled_op = dml::GetOrCreateCompiledOperatorApi(&op_desc);

    DML_BUFFER_BINDING bindings_storage[4];
    bindings_storage[0] =
        dml::utils::create_buffer_binding(full_buffer_to_update);
    bindings_storage[1] =
        dml::utils::create_buffer_binding(indices_buffer_gpu.Get());
    bindings_storage[2] =
        dml::utils::create_buffer_binding(strided_data_to_scatter);
    bindings_storage[3] =
        dml::utils::create_buffer_binding(full_buffer_to_update);

    std::vector<DML_BINDING_DESC> scatter_inputs = {
        dml::utils::create_binding_desc(&bindings_storage[0]),
        dml::utils::create_binding_desc(&bindings_storage[1]),
        dml::utils::create_binding_desc(&bindings_storage[2])};
    std::vector<DML_BINDING_DESC> scatter_outputs = {
        dml::utils::create_binding_desc(&bindings_storage[3])};
    compiled_op->Execute(scatter_inputs, scatter_outputs);
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

  auto* dml_device_wrapper = dml::get_device();
  if (!dml_device_wrapper) {
    throw std::runtime_error("DirectML device not available");
  }

  auto* input_buffer_res =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(input.buffer()));
  auto* sin_buffer_res =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(sin.buffer()));
  auto* cos_buffer_res =
      reinterpret_cast<ID3D12Resource*>(const_cast<void*>(cos.buffer()));
  auto* output_buffer_res = reinterpret_cast<ID3D12Resource*>(output.buffer());

  const dim_t feature_depth = input.dim(-1);
  const dim_t rot_ndims_param = _ndims == 0 ? feature_depth : _ndims;
  const dim_t middle_non_interleave = rot_ndims_param / 2;

  RotaryDMLCompute compute_kernel(dml_device_wrapper, input.dtype());

  if (_interleave) {
    compute_kernel.compute_interleave(
        input_buffer_res, sin_buffer_res, cos_buffer_res, output_buffer_res,
        input, sin, cos, output, feature_depth, rot_ndims_param);
  } else {
    compute_kernel.compute_non_interleave(
        input_buffer_res, sin_buffer_res, cos_buffer_res, output_buffer_res,
        input, sin, cos, output, feature_depth, rot_ndims_param,
        middle_non_interleave);
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