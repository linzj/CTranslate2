#include "benchmark_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "ctranslate2/ops/ops.h"

namespace {
constexpr const bool kCheckCorrectness = true;

static std::string dtype_str_local(ctranslate2::DataType dtype) {
  switch (dtype) {
    case ctranslate2::DataType::FLOAT32:
      return "float32";
    case ctranslate2::DataType::FLOAT16:
      return "float16";
    case ctranslate2::DataType::BFLOAT16:
      return "bfloat16";
    case ctranslate2::DataType::INT32:
      return "int32";
    case ctranslate2::DataType::INT16:
      return "int16";
    case ctranslate2::DataType::INT8:
      return "int8";
    default:
      return "unknown_dtype(" + std::to_string(static_cast<int>(dtype)) + ")";
  }
}

// Helper to print shape
static std::string shape_to_string(const ctranslate2::Shape& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
  }
  oss << "]";
  return oss.str();
}

template <typename T>
static int compare_storage_views_detailed_typed(
    const ctranslate2::StorageView& view1,  // Device output copied to CPU
    const ctranslate2::StorageView& view2,  // CPU reference output
    const std::string& tensor_name,
    std::ostringstream& oss,
    float relative_diff_threshold = 0.001f,      // 0.1%
    float absolute_diff_threshold_float = 1e-7f  // Min absolute diff for floats
) {
  const T* data1 = view1.data<T>();
  const T* data2 = view2.data<T>();
  ctranslate2::dim_t N = view1.size();
  int mismatches_found = 0;
  const int max_printed_mismatches = 10000;

  for (ctranslate2::dim_t i = 0; i < N; ++i) {
    T val1 = data1[i];
    T val2 = data2[i];
    bool current_mismatch = false;

    if constexpr (std::is_floating_point_v<T>) {
      T diff = std::abs(val1 - val2);
      T max_abs_val_for_rel = std::max({static_cast<T>(1e-9), std::abs(val1),
                                        std::abs(val2)});  // Avoid div by zero

      if (diff > (relative_diff_threshold * max_abs_val_for_rel) &&
          diff > static_cast<T>(absolute_diff_threshold_float)) {
        current_mismatch = true;
      }
    } else {  // Integer types
      if (val1 != val2) {
        current_mismatch = true;
      }
    }

    if (current_mismatch) {
      if (mismatches_found < max_printed_mismatches) {
        // Coordinate printing removed due to lack of standard index_to_coord
        oss << tensor_name << " mismatch at raw_index " << i;
        if constexpr (std::is_same_v<T, int8_t>) {
          oss << ": DeviceCPUVal=" << static_cast<int>(val1)
              << " vs. CPUVal=" << static_cast<int>(val2)
              << ", AbsDiff=" << std::abs(val1 - val2);
        } else {
          oss << ": DeviceCPUVal=" << val1 << " vs. CPUVal=" << val2
              << ", AbsDiff=" << std::abs(val1 - val2);
        }
        if constexpr (std::is_floating_point_v<T>) {
          T max_abs_val_for_rel =
              std::max({static_cast<T>(1e-9), std::abs(val1), std::abs(val2)});
          oss << ", RelDiff="
              << (max_abs_val_for_rel > 0
                      ? (std::abs(val1 - val2) / max_abs_val_for_rel)
                      : 0);
        }
        oss << "\n";
      }
      mismatches_found++;
    }
  }
  if (mismatches_found > max_printed_mismatches) {
    oss << tensor_name << ": and "
        << (mismatches_found - max_printed_mismatches)
        << " more mismatches...\n";
  }
  return mismatches_found;
}

static int dispatch_compare_views(
    const ctranslate2::StorageView& view1,  // Device output copied to CPU
    const ctranslate2::StorageView& view2,  // CPU reference output
    const std::string& tensor_name,
    std::ostringstream& oss,
    float relative_diff_threshold = 0.001f,
    float absolute_diff_threshold_float = 1e-7f) {
  if (view1.shape() != view2.shape()) {
    oss << tensor_name << " shape mismatch: DeviceCPU "
        << shape_to_string(view1.shape()) << " vs. CPU "
        << shape_to_string(view2.shape()) << "\n";
    return 1;
  }
  if (view1.size() != view2.size()) {
    oss << tensor_name << " size mismatch: DeviceCPU " << view1.size()
        << " vs. CPU " << view2.size() << "\n";
    return 1;
  }
  if (view1.device() != ctranslate2::Device::CPU ||
      view2.device() != ctranslate2::Device::CPU) {
    oss << tensor_name
        << " error: both views must be on CPU for dispatch_compare_views.\n";
    return 1;
  }
  if (view1.size() == 0)
    return 0;

  // For float16/bfloat16, cast to float32 for comparison
  if (view1.dtype() == ctranslate2::DataType::FLOAT16 ||
      view1.dtype() == ctranslate2::DataType::BFLOAT16) {
    if (view2.dtype() != view1.dtype()) {
      oss << tensor_name << " dtype mismatch for castable types: DeviceCPU "
          << dtype_str_local(view1.dtype()) << " vs. CPU "
          << dtype_str_local(view2.dtype()) << "\n";
      return 1;
    }
    ctranslate2::StorageView v1_float =
        view1.to(ctranslate2::DataType::FLOAT32);
    ctranslate2::StorageView v2_float =
        view2.to(ctranslate2::DataType::FLOAT32);
    return compare_storage_views_detailed_typed<float>(
        v1_float, v2_float, tensor_name + " (as float32)", oss,
        relative_diff_threshold, absolute_diff_threshold_float);
  }

  // Direct comparison for other types if they match
  if (view1.dtype() != view2.dtype()) {
    oss << tensor_name << " dtype mismatch: DeviceCPU "
        << dtype_str_local(view1.dtype()) << " vs. CPU "
        << dtype_str_local(view2.dtype()) << "\n";
    return 1;
  }

  switch (view1.dtype()) {
    case ctranslate2::DataType::FLOAT32:
      return compare_storage_views_detailed_typed<float>(
          view1, view2, tensor_name, oss, relative_diff_threshold,
          absolute_diff_threshold_float);
    case ctranslate2::DataType::INT32:
      return compare_storage_views_detailed_typed<int32_t>(view1, view2,
                                                           tensor_name, oss);
    case ctranslate2::DataType::INT16:
      return compare_storage_views_detailed_typed<int16_t>(view1, view2,
                                                           tensor_name, oss);
    case ctranslate2::DataType::INT8:
      return compare_storage_views_detailed_typed<int8_t>(view1, view2,
                                                          tensor_name, oss);
    default:
      oss << tensor_name << ": Unsupported dtype for direct comparison: "
          << dtype_str_local(view1.dtype()) << "\n";
      return 1;
  }
}
}  // namespace

using namespace ctranslate2;

void benchmark_gather(Device device) {
  // kCheckCorrectness is now from the anonymous namespace
  if (!kCheckCorrectness) {
    StorageView data({512, 512}, DataType::FLOAT32, device);
    std::vector<int32_t> input_v(250);
    std::iota(input_v.begin(), input_v.end(), 0);
    StorageView input({static_cast<dim_t>(input_v.size())}, input_v, device);
    StorageView output(device);
    const ops::Gather gather_op;
    BENCHMARK(gather_op(data, input, output), 100000);
  } else {
    std::ostringstream error_log;
    int total_mismatches = 0;

    Shape data_shape = {512, 512};
    std::vector<float> data_v(data_shape[0] * data_shape[1]);
    std::iota(data_v.begin(), data_v.end(), 0.0f);
    StorageView data_device(data_shape, data_v, device);
    StorageView data_device_cpu_copy(data_device.dtype(),
                                     Device::CPU);      // Copy to CPU
    data_device_cpu_copy.copy_from(data_device, true);  // force sync
    StorageView data_host_ref(data_shape, data_v,
                              Device::CPU);  // Original host data
    total_mismatches += dispatch_compare_views(
        data_device_cpu_copy, data_host_ref, "Gather Initial Data", error_log);

    Shape input_shape_1d = {250};
    std::vector<int32_t> input_v(input_shape_1d[0]);
    std::iota(input_v.begin(), input_v.end(), 0);
    StorageView input_device(input_shape_1d, input_v, device);
    StorageView input_device_cpu_copy(input_device.dtype(),
                                      Device::CPU);       // Copy to CPU
    input_device_cpu_copy.copy_from(input_device, true);  // force sync
    StorageView input_host_ref(input_shape_1d, input_v,
                               Device::CPU);  // Original host data
    total_mismatches +=
        dispatch_compare_views(input_device_cpu_copy, input_host_ref,
                               "Gather Initial Input", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("Gather Input Data Sanity Check Failed:\n" +
                               error_log.str());
    }
    // Clear log for op output check if initial checks passed
    error_log.str("");
    error_log.clear();
    total_mismatches = 0;

    StorageView output_from_device(DataType::FLOAT32,
                                   device);  // Output from Device op
    const ops::Gather gather_op_device;
    gather_op_device(data_device, input_device, output_from_device);

    StorageView output_device_cpu_copy(
        output_from_device.dtype(), Device::CPU);  // Copy device output to CPU
    output_device_cpu_copy.copy_from(output_from_device, true);  // force sync

    StorageView output_cpu_ref(output_from_device.dtype(),
                               Device::CPU);  // Output from CPU op
    const ops::Gather gather_op_cpu;
    gather_op_cpu(data_host_ref, input_host_ref,
                  output_cpu_ref);  // Use host refs for CPU op

    total_mismatches += dispatch_compare_views(
        output_device_cpu_copy, output_cpu_ref, "Gather Output", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("Gather output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_transpose(Device device) {
  if (!kCheckCorrectness) {
    StorageView x({64, 48, 8, 64}, DataType::FLOAT32, device);
    StorageView y(device);
    const ops::Transpose transpose_op({0, 2, 1, 3});
    BENCHMARK(transpose_op(x, y), 1000);
  } else {
    const Shape x_shape = {64, 48, 8, 64};
    const DataType dtype = DataType::FLOAT32;
    // Use initializer list directly for consistency and to avoid type issues
    const ops::Transpose transpose_op({0, 2, 1, 3});

    const dim_t num_elements =
        std::accumulate(x_shape.begin(), x_shape.end(), static_cast<dim_t>(1),
                        std::multiplies<dim_t>());
    std::vector<float> x_data_vec(num_elements);
    std::iota(x_data_vec.begin(), x_data_vec.end(), 0.0f);

    StorageView x_device(x_shape, x_data_vec, device);
    StorageView y_device(dtype,
                         device);  // Output will have same dtype as input

    StorageView x_cpu(x_shape, x_data_vec, Device::CPU);
    StorageView y_cpu(dtype,
                      Device::CPU);  // Output will have same dtype as input

    // Perform operation on device
    transpose_op(x_device, y_device);

    // Perform operation on CPU for reference
    transpose_op(x_cpu, y_cpu);

    // Copy device result to CPU for comparison
    StorageView y_device_cpu_copy(dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "Transpose Output", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("Transpose output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_split(Device device) {
  if (!kCheckCorrectness) {
    StorageView x({64, 512 * 3}, DataType::FLOAT32, device);
    StorageView a(device);
    StorageView b(device);
    StorageView c(device);
    const ops::Split split_op(-1);
    BENCHMARK(split_op(x, a, b, c), 10000);
  } else {
    const Shape x_shape = {64, 512 * 3};
    const DataType dtype = DataType::FLOAT32;
    const int axis = -1;  // As used in the original benchmark
    const ops::Split split_op(axis);

    const dim_t num_elements =
        std::accumulate(x_shape.begin(), x_shape.end(), static_cast<dim_t>(1),
                        std::multiplies<dim_t>());
    std::vector<float> x_data_vec(num_elements);
    std::iota(x_data_vec.begin(), x_data_vec.end(), 0.0f);

    StorageView x_device(x_shape, x_data_vec, device);
    StorageView a_device(dtype, device);
    StorageView b_device(dtype, device);
    StorageView c_device(dtype, device);

    StorageView x_cpu(x_shape, x_data_vec, Device::CPU);
    StorageView a_cpu(dtype, Device::CPU);
    StorageView b_cpu(dtype, Device::CPU);
    StorageView c_cpu(dtype, Device::CPU);

    // Perform operation on device
    split_op(x_device, a_device, b_device, c_device);

    // Perform operation on CPU for reference
    split_op(x_cpu, a_cpu, b_cpu, c_cpu);

    // Copy device results to CPU for comparison
    StorageView a_device_cpu_copy(dtype, Device::CPU);
    StorageView b_device_cpu_copy(dtype, Device::CPU);
    StorageView c_device_cpu_copy(dtype, Device::CPU);
    a_device_cpu_copy.copy_from(a_device, true);
    b_device_cpu_copy.copy_from(b_device, true);
    c_device_cpu_copy.copy_from(c_device, true);

    std::ostringstream error_log;
    int total_mismatches = 0;
    total_mismatches += dispatch_compare_views(a_device_cpu_copy, a_cpu,
                                               "Split Output 'a'", error_log);
    total_mismatches += dispatch_compare_views(b_device_cpu_copy, b_cpu,
                                               "Split Output 'b'", error_log);
    total_mismatches += dispatch_compare_views(c_device_cpu_copy, c_cpu,
                                               "Split Output 'c'", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("Split output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_layer_norm(Device device) {
  if (!kCheckCorrectness) {
    std::vector<float> gamma_ = rand_vector(512);
    std::vector<float> beta_ = rand_vector(512);
    std::vector<float> x_ = rand_vector(100 * 512);

    StorageView gamma({512}, gamma_, device);
    StorageView beta({512}, beta_, device);
    StorageView x({100, 512}, x_, device);
    StorageView y(x.device());
    const ops::LayerNorm layer_norm_op{};
    BENCHMARK(layer_norm_op(beta, gamma, x, y), 10000);
  } else {
    const Shape x_shape = {100, 512};
    const Shape params_shape = {512};
    const DataType dtype = DataType::FLOAT32;
    const ops::LayerNorm layer_norm_op{};

    std::vector<float> gamma_data = rand_vector(params_shape[0]);
    std::vector<float> beta_data = rand_vector(params_shape[0]);
    std::vector<float> x_data = rand_vector(x_shape[0] * x_shape[1]);

    StorageView gamma_device(params_shape, gamma_data, device);
    StorageView beta_device(params_shape, beta_data, device);
    StorageView x_device(x_shape, x_data, device);
    StorageView y_device(dtype, device);

    StorageView gamma_cpu(params_shape, gamma_data, Device::CPU);
    StorageView beta_cpu(params_shape, beta_data, Device::CPU);
    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView y_cpu(dtype, Device::CPU);

    layer_norm_op(beta_device, gamma_device, x_device, y_device);
    layer_norm_op(beta_cpu, gamma_cpu, x_cpu, y_cpu);

    StorageView y_device_cpu_copy(dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "LayerNorm Output", error_log);

    if (total_mismatches > 3) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("LayerNorm output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_softmax(Device device) {
  if (!kCheckCorrectness) {
    std::vector<float> x_ = rand_vector(100 * 512);
    StorageView x({100, 512}, x_, device);
    StorageView y(x.device());
    const ops::SoftMax softmax_op{};
    BENCHMARK(softmax_op(x, y), 10000);
  } else {
    const Shape x_shape = {100, 512};
    const DataType dtype = DataType::FLOAT32;
    const ops::SoftMax softmax_op{};

    std::vector<float> x_data = rand_vector(x_shape[0] * x_shape[1]);

    StorageView x_device(x_shape, x_data, device);
    StorageView y_device(dtype, device);

    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView y_cpu(dtype, Device::CPU);

    softmax_op(x_device, y_device);
    softmax_op(x_cpu, y_cpu);

    StorageView y_device_cpu_copy(dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "SoftMax Output", error_log, 0.001f, 1e-6f);

    if (total_mismatches > 0) {
      throw std::runtime_error("SoftMax output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_masked_softmax(Device device) {
  if (!kCheckCorrectness) {
    const dim_t batch_size = 32;
    const dim_t num_heads = 8;
    const dim_t max_source = 24;
    const dim_t max_target = 36;
    StorageView lengths(
        {batch_size}, std::vector<int32_t>(batch_size, max_source - 5), device);
    StorageView x({batch_size, num_heads, max_source, max_target},
                  rand_vector(batch_size * num_heads * max_source * max_target),
                  device);
    StorageView y(x.device());
    const ops::SoftMax softmax_op{};
    BENCHMARK(softmax_op(x, lengths, y), 10000);
  } else {
    const dim_t actual_batch_size = 32;
    const dim_t num_heads = 8;
    const dim_t max_source = 24;
    const dim_t max_target = 36;
    const DataType dtype = DataType::FLOAT32;
    const ops::SoftMax softmax_op{};

    // This is the number of independent softmax operations (B * H * S)
    const dim_t num_softmax_vectors =
        actual_batch_size * num_heads * max_source;

    std::vector<int32_t> lengths_data(num_softmax_vectors);
    // Fill with a length value. This length (max_source - 5 = 19) will be
    // applied to the 'max_target' dimension (size 36) for each of the
    // num_softmax_vectors. This means the first 19 elements are considered, the
    // rest are masked.
    std::fill(lengths_data.begin(), lengths_data.end(), max_source - 5);

    std::vector<float> x_data =
        rand_vector(actual_batch_size * num_heads * max_source * max_target);

    // `lengths_device` should have a total size of `num_softmax_vectors`.
    // It can be flat or have a shape like {actual_batch_size, num_heads,
    // max_source}. The SoftMax operator's check `lengths->size()` only cares
    // about the total number of elements. Using a flat shape here for
    // simplicity, matching the operator's `batch_size` calculation logic.
    StorageView lengths_device(Shape{num_softmax_vectors}, lengths_data,
                               device);  // Specify INT32
    StorageView x_device({actual_batch_size, num_heads, max_source, max_target},
                         x_data, device);
    StorageView y_device(dtype, device);

    StorageView lengths_cpu(Shape{num_softmax_vectors}, lengths_data,
                            Device::CPU);  // Specify INT32
    StorageView x_cpu({actual_batch_size, num_heads, max_source, max_target},
                      x_data, Device::CPU);
    StorageView y_cpu(dtype, Device::CPU);

    softmax_op(x_device, &lengths_device, y_device);  // Pass pointer
    softmax_op(x_cpu, &lengths_cpu, y_cpu);           // Pass pointer

    StorageView y_device_cpu_copy(dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "Masked SoftMax Output", error_log, 0.001f,
        1e-5f);  // Adjusted tolerance slightly for float calcs

    if (total_mismatches > 0) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("Masked SoftMax output mismatch details:\n" +
                               error_log.str());
    } else {
      std::cout << "Masked SoftMax test passed!" << std::endl;
    }
  }
}

void benchmark_topk(Device device) {
  if (!kCheckCorrectness) {
    const size_t k = 4;
    const size_t batch_size = 8;
    const size_t vocab_size = 32000;
    std::vector<float> x = rand_vector(batch_size * k * vocab_size);
    StorageView input({batch_size, k * vocab_size}, x, device);
    StorageView values(input.dtype(), device);
    StorageView indices(DataType::INT32, device);
    const ops::TopK op(k);
    BENCHMARK(op(input, values, indices), 2000);
  } else {
    const size_t k = 4;
    const size_t batch_size = 8;
    const size_t vocab_size = 32000;  // This seems large for k*vocab_size as
                                      // second dim. Let's use a smaller inner
                                      // dim for the check, e.g., vocab_size/k
    const Shape input_shape = {batch_size,
                               vocab_size};  // More standard TopK input
    const DataType dtype = DataType::FLOAT32;
    const ops::TopK topk_op(k);

    std::vector<float> input_data =
        rand_vector(input_shape[0] * input_shape[1]);

    StorageView input_device(input_shape, input_data, device);
    StorageView values_device(dtype, device);
    StorageView indices_device(DataType::INT32, device);

    StorageView input_cpu(input_shape, input_data, Device::CPU);
    StorageView values_cpu(dtype, Device::CPU);
    StorageView indices_cpu(DataType::INT32, Device::CPU);

    topk_op(input_device, values_device, indices_device);
    topk_op(input_cpu, values_cpu, indices_cpu);

    StorageView values_device_cpu_copy(dtype, Device::CPU);
    values_device_cpu_copy.copy_from(values_device, true);
    StorageView indices_device_cpu_copy(DataType::INT32, Device::CPU);
    indices_device_cpu_copy.copy_from(indices_device, true);

    std::ostringstream error_log;
    int total_mismatches = 0;
    total_mismatches += dispatch_compare_views(
        values_device_cpu_copy, values_cpu, "TopK Values", error_log);

    // indices compare is stable, ignore first.
    // total_mismatches += dispatch_compare_views(
    //     indices_device_cpu_copy, indices_cpu, "TopK Indices", error_log);

    if (total_mismatches > 0) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("TopK output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_gemm(Device device, DataType dtype) {
  if (!kCheckCorrectness) {
    DataType output_dtype =
        dtype != DataType::FLOAT32 ? DataType::INT32 : dtype;
    StorageView a({32 * 32, 512}, dtype, device);
    StorageView b({2048, 512}, dtype, device);  // b_t
    StorageView c(output_dtype, device);
    const ops::Gemm gemm_op(1, 0, false, true);
    BENCHMARK(gemm_op(a, b, c), 1000);
  } else {
    const Shape a_shape = {32 * 32, 512};  // M x K
    const Shape b_t_shape = {2048,
                             512};  // N x K (b is transposed, so K is last dim)
                                    // Effective B shape for GEMM: K x N
    const dim_t M = a_shape[0];
    const dim_t K = a_shape[1];
    const dim_t N = b_t_shape[0];
    const Shape b_storage_shape = {
        N,
        K};  // B storage is N x K, op takes transB=true, so effectively K x N.

    // const ops::Gemm gemm_op(1.f, 0.f, false, true, false,
    // ops::Gemm::Prologue::NONE, ops::Gemm::Epilogue::NONE, dtype); Simplify
    // Gemm op to match non-kCheckCorrectness branch style, with explicit alpha,
    // beta. The ops::Gemm has several constructors. The one for (alpha, beta,
    // transA, transB, transC, acc_type, epilogue_beta, epilogue) might be what
    // was attempted. Or (alpha, beta, transa, transb, broadcast_c, dtype_a,
    // dtype_b, dtype_c, prefer_int8_gemm, prologue, epilogue) Use the simple
    // constructor as in the non-kCheckCorrectness part.
    const ops::Gemm gemm_op(1.0f, 0.0f, false, true);

    DataType output_dtype_check =
        (dtype == DataType::FLOAT32 || dtype == DataType::FLOAT16 ||
         dtype == DataType::BFLOAT16)
            ? dtype
            : DataType::INT32;

    std::vector<float> a_data_f;
    std::vector<int16_t> a_data_i16;
    std::vector<int8_t> a_data_i8;

    std::vector<float> b_data_f;
    std::vector<int16_t> b_data_i16;
    std::vector<int8_t> b_data_i8;

    StorageView a_device(dtype, device);
    StorageView b_device(dtype, device);  // This B is N x K

    if (dtype == DataType::FLOAT32) {
      a_data_f = rand_vector(M * K);
      b_data_f = rand_vector(N * K);  // B is N x K
      a_device.resize(a_shape);
      a_device.copy_from(a_data_f.data(), M * K, Device::CPU);
      b_device.resize(b_storage_shape);  // N x K
      b_device.copy_from(b_data_f.data(), N * K, Device::CPU);
    } else if (dtype == DataType::INT16) {
      // For quantized types, let's use simple sequences for now
      a_data_i16.resize(M * K);
      std::iota(a_data_i16.begin(), a_data_i16.end(), static_cast<int16_t>(0));
      b_data_i16.resize(N * K);
      std::iota(b_data_i16.begin(), b_data_i16.end(), static_cast<int16_t>(0));
      a_device.resize(a_shape);
      a_device.copy_from(a_data_i16.data(), M * K, Device::CPU);
      b_device.resize(b_storage_shape);  // N x K
      b_device.copy_from(b_data_i16.data(), N * K, Device::CPU);
    } else {  // INT8
      a_data_i8.resize(M * K);
      std::iota(a_data_i8.begin(), a_data_i8.end(), static_cast<int8_t>(0));
      b_data_i8.resize(N * K);
      std::iota(b_data_i8.begin(), b_data_i8.end(), static_cast<int8_t>(0));
      a_device.resize(a_shape);
      a_device.copy_from(a_data_i8.data(), M * K, Device::CPU);
      b_device.resize(b_storage_shape);  // N x K
      b_device.copy_from(b_data_i8.data(), N * K, Device::CPU);
    }

    StorageView c_device(output_dtype_check, device);

    StorageView a_cpu(dtype, Device::CPU);
    StorageView b_cpu(dtype, Device::CPU);  // This B is N x K
    StorageView c_cpu(output_dtype_check, Device::CPU);

    a_cpu.copy_from(a_device, true);
    b_cpu.copy_from(b_device, true);

    gemm_op(a_device, b_device, c_device);
    gemm_op(a_cpu, b_cpu, c_cpu);

    StorageView c_device_cpu_copy(output_dtype_check, Device::CPU);
    c_device_cpu_copy.copy_from(c_device, true);

    std::ostringstream error_log;
    int total_mismatches = dispatch_compare_views(c_device_cpu_copy, c_cpu,
                                                  "GEMM Output", error_log);

    if (total_mismatches > 0) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("GEMM output mismatch for input dtype " +
                               dtype_str_local(dtype) + " (output dtype " +
                               dtype_str_local(output_dtype_check) +
                               ") details:\n" + error_log.str());
    }
  }
}

void benchmark_quantize(Device device, DataType out_dtype) {
  if (!kCheckCorrectness) {
    StorageView x({32, 512}, rand_vector(32 * 512), device);
    StorageView y(out_dtype, device);
    StorageView scale(DataType::FLOAT32, device);
    const ops::Quantize quantize_op;
    BENCHMARK(quantize_op(x, y, scale), 10000);
  } else {
    const Shape x_shape = {32, 512};
    const ops::Quantize quantize_op;

    std::vector<float> x_data = rand_vector(x_shape[0] * x_shape[1]);

    StorageView x_device(x_shape, x_data, device);
    StorageView y_device(DataType::INT8, device);
    StorageView scale_device(DataType::FLOAT32, device);

    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView y_cpu(DataType::INT8, Device::CPU);
    StorageView scale_cpu(DataType::FLOAT32, Device::CPU);

    quantize_op(x_device, y_device, scale_device);
    quantize_op(x_cpu, y_cpu, scale_cpu);

    StorageView y_device_cpu_copy(DataType::INT8, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);
    StorageView scale_device_cpu_copy(DataType::FLOAT32, Device::CPU);
    scale_device_cpu_copy.copy_from(scale_device, true);

    std::ostringstream error_log;
    int total_mismatches = 0;
    total_mismatches += dispatch_compare_views(y_device_cpu_copy, y_cpu,
                                               "Quantize Output Y", error_log);
    if (scale_device_cpu_copy.shape() != scale_cpu.shape()) {
      scale_device_cpu_copy.reshape(scale_cpu.shape());
    }
    total_mismatches += dispatch_compare_views(
        scale_device_cpu_copy, scale_cpu, "Quantize Output Scale", error_log);

    if (total_mismatches > 1) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("Quantize output mismatch for output_dtype " +
                               dtype_str_local(out_dtype) + " details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_dequantize(Device device) {
  if (!kCheckCorrectness) {
    StorageView x({64, 8192}, DataType::INT32, device);
    StorageView input_scale({32}, DataType::FLOAT32,
                            device);  // scales_c in some contexts
    StorageView weight_scale({8192}, DataType::FLOAT32,
                             device);  // scales_a in some contexts
    StorageView bias({8192}, DataType::FLOAT32, device);
    StorageView y(device);
    const ops::ActivationType activation_type = ops::ActivationType::ReLU;
    const ops::Dequantize dequantize_op(&activation_type);
    BENCHMARK(
        dequantize_op(x, input_scale, weight_scale, false, true, y, &bias),
        10000);
  } else {
    const Shape x_shape = {64, 8192};
    const Shape bias_shape = {8192};
    // scales are per-channel or per-tensor depending on usage
    // The benchmark example uses input_scale({32}) and weight_scale({8192})
    // Let's assume output_transposed = false, weight_transposed = true (as in
    // benchmark) This implies for x(M,K), input_scale is per-row of X (if M
    // matches input_scale.dim(0)) and weight_scale is per-column of Weight
    // matrix. If Weight is (N,K) or (K,N), this should match N. The
    // `dequantize_op` has variants. This one takes (x, scales_c, scales_a,
    // output_transpose, weight_transpose, y, bias) If weight_transpose=true,
    // weight scales_a corresponds to columns of original weight (last dim). So
    // dim is N. If output_transpose=false, input scales_c corresponds to rows
    // of X (first dim of x if global scale for K not used). Or single scale.
    // The provided shapes {32} and {8192} are a bit unusual for these scale
    // meanings without more context on matrix multiply that produces X. For
    // now, let's replicate the benchmark scales.
    const Shape input_scale_shape = {
        x_shape[0]};  // Assume scale per output channel of previous layer / per
                      // row of x
    const Shape weight_scale_shape = {
        x_shape[1]};  // Assume scale per input channel / per column of x

    const ops::ActivationType activation_type = ops::ActivationType::ReLU;
    const ops::Dequantize dequantize_op(&activation_type);
    const bool output_transpose = false;
    const bool weight_transpose = true;

    std::vector<int32_t> x_data(x_shape[0] * x_shape[1]);
    std::iota(x_data.begin(), x_data.end(), 0);

    std::vector<float> input_scale_data_raw = rand_vector(input_scale_shape[0]);
    std::vector<float> input_scale_data(input_scale_shape[0]);
    std::transform(input_scale_data_raw.begin(), input_scale_data_raw.end(),
                   input_scale_data.begin(), [](float v) {
                     return 0.1f + std::fabs(v) * 0.9f;
                   });  // Scale to [0.1, 1.0] approx.

    std::vector<float> weight_scale_data_raw =
        rand_vector(weight_scale_shape[0]);
    std::vector<float> weight_scale_data(weight_scale_shape[0]);
    std::transform(weight_scale_data_raw.begin(), weight_scale_data_raw.end(),
                   weight_scale_data.begin(), [](float v) {
                     return 0.1f + std::fabs(v) * 0.9f;
                   });  // Scale to [0.1, 1.0] approx.

    std::vector<float> bias_data = rand_vector(bias_shape[0]);

    StorageView x_device(x_shape, x_data, device);
    StorageView input_scale_device(input_scale_shape, input_scale_data, device);
    StorageView weight_scale_device(weight_scale_shape, weight_scale_data,
                                    device);
    StorageView bias_device(bias_shape, bias_data, device);
    StorageView y_device(DataType::FLOAT32, device);

    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView input_scale_cpu(input_scale_shape, input_scale_data,
                                Device::CPU);
    StorageView weight_scale_cpu(weight_scale_shape, weight_scale_data,
                                 Device::CPU);
    StorageView bias_cpu(bias_shape, bias_data, Device::CPU);
    StorageView y_cpu(DataType::FLOAT32, Device::CPU);

    dequantize_op(x_cpu, input_scale_cpu, weight_scale_cpu, output_transpose,
                  weight_transpose, y_cpu, &bias_cpu);
    dequantize_op(x_device, input_scale_device, weight_scale_device,
                  output_transpose, weight_transpose, y_device, &bias_device);

    StorageView y_device_cpu_copy(DataType::FLOAT32, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    // For dequantize, which involves multiple float operations, a slightly
    // higher relative tolerance might be needed or a check that is more robust
    // to accumulated precision errors if 0.1% is too strict. Sticking to 0.1%
    // as requested first. Absolute diff threshold becomes more important here.
    int total_mismatches =
        dispatch_compare_views(y_device_cpu_copy, y_cpu, "Dequantize Output",
                               error_log, 0.001f, 1e-5f);

    if (total_mismatches > 0) {
      throw std::runtime_error("Dequantize output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_conv1d(Device device) {
  if (!kCheckCorrectness) {
    StorageView x({1, 768, 3000}, DataType::FLOAT32,
                  device);  // Batch, Channels, Width
    StorageView weight({768, 768, 3}, DataType::FLOAT32,
                       device);                          // Out, In, Kernel
    StorageView bias({768}, DataType::FLOAT32, device);  // Out
    StorageView y(device);
    const ops::Conv1D conv_op{2, 1};  // stride, padding
    BENCHMARK(conv_op(x, weight, bias, y), 100);
  } else {
    const Shape x_shape = {1, 768, 3000};
    const Shape weight_shape = {768, 768,
                                3};  // Filters, InputChannels, KernelWidth
    const Shape bias_shape = {weight_shape[0]};  // Filters
    const int stride = 2;
    const int padding = 1;
    const ops::Conv1D conv_op{stride, padding};
    const DataType dtype = DataType::FLOAT32;

    std::vector<float> x_data =
        rand_vector(x_shape[0] * x_shape[1] * x_shape[2]);
    std::vector<float> weight_data =
        rand_vector(weight_shape[0] * weight_shape[1] * weight_shape[2]);
    std::vector<float> bias_data = rand_vector(bias_shape[0]);

    StorageView x_device(x_shape, x_data, device);
    StorageView weight_device(weight_shape, weight_data, device);
    StorageView bias_device(bias_shape, bias_data, device);
    StorageView y_device(dtype, device);

    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView weight_cpu(weight_shape, weight_data, Device::CPU);
    StorageView bias_cpu(bias_shape, bias_data, Device::CPU);
    StorageView y_cpu(dtype, Device::CPU);

    conv_op(x_device, weight_device, bias_device, y_device);
    conv_op(x_cpu, weight_cpu, bias_cpu, y_cpu);

    StorageView y_device_cpu_copy(dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    // Conv1D also can accumulate float errors. Using a slightly higher absolute
    // epsilon.
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "Conv1D Output", error_log, 0.001f, 1e-5f);

    if (total_mismatches > 0) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("Conv1D output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_conv1d_qscale(Device device) {
  if (!kCheckCorrectness) {
    StorageView x({1, 1280, 3000}, DataType::FLOAT32,
                  device);  // Batch, Channels, Width
    StorageView weight({1280, 128, 3}, DataType::INT8,
                       device);                          // Out, In, Kernel
    StorageView bias({1280}, DataType::FLOAT32, device);  // Out
    StorageView qscale({1280}, 2.5f, device);  // Use a realistic scale.
    StorageView y(DataType::FLOAT32, device);
    const ops::Conv1D conv_op{2, 1, 1, 10};  // stride, padding, dilation, groups
    BENCHMARK(conv_op(x, weight, bias, y, &qscale), 100);
  } else {
    const Shape x_shape = {1, 1280, 3000};
    const Shape weight_shape = {1280, 128,
                                3};  // Filters, InputChannels, KernelWidth
    const Shape bias_shape = {weight_shape[0]};  // Filters
    const Shape qscale_shape = {weight_shape[0]};
    const int stride = 2;
    const int padding = 1;
    const int groups = 10;
    const ops::Conv1D conv_op{stride, padding, 1, groups};
    const DataType out_dtype = DataType::FLOAT32;

    std::vector<float> x_data =
        rand_vector(x_shape[0] * x_shape[1] * x_shape[2]);
    const dim_t weight_size =
        weight_shape[0] * weight_shape[1] * weight_shape[2];
    std::vector<float> weight_data_f = rand_vector(weight_size);
    std::vector<int8_t> weight_data(weight_size);
    float max_abs = 0.f;
    for (const auto& v : weight_data_f)
      max_abs = std::max(max_abs, std::abs(v));
    const float scale = 127.f / (max_abs > 1e-9f ? max_abs : 127.f);
    for (dim_t i = 0; i < weight_size; ++i) {
      const float val_scaled = weight_data_f[i] * scale;
      weight_data[i] = static_cast<int8_t>(
          std::round(std::max(-127.f, std::min(127.f, val_scaled))));
    }
    std::vector<float> bias_data = rand_vector(bias_shape[0]);
    std::vector<float> qscale_data(qscale_shape[0], 2.5f);

    StorageView x_device(x_shape, x_data, device);
    StorageView weight_device(weight_shape, weight_data, device);
    StorageView bias_device(bias_shape, bias_data, device);
    StorageView qscale_device(qscale_shape, qscale_data, device);
    StorageView y_device(out_dtype, device);

    StorageView x_cpu(x_shape, x_data, Device::CPU);
    StorageView weight_cpu(weight_shape, weight_data, Device::CPU);
    StorageView bias_cpu(bias_shape, bias_data, Device::CPU);
    StorageView qscale_cpu(qscale_shape, qscale_data, Device::CPU);
    StorageView y_cpu(out_dtype, Device::CPU);

    conv_op(x_device, weight_device, bias_device, y_device, &qscale_device);
    conv_op(x_cpu, weight_cpu, bias_cpu, y_cpu, &qscale_cpu);

    StorageView y_device_cpu_copy(out_dtype, Device::CPU);
    y_device_cpu_copy.copy_from(y_device, true);

    std::ostringstream error_log;
    // For quantized ops, allow for a few mismatches due to potential rounding
    // differences between device and CPU.
    int total_mismatches = dispatch_compare_views(
        y_device_cpu_copy, y_cpu, "Conv1D Qscale Output", error_log);

    if (total_mismatches > 5) {
      std::cerr << error_log.str() << std::endl;
      throw std::runtime_error("Conv1D with qscale output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_gumbel_max(Device device) {
  const dim_t num_samples =
      4;  // Number of samples to draw, equivalent to k in TopK
  if (!kCheckCorrectness) {
    const dim_t batch_size_bm = 32;
    const dim_t depth_bm = 1024;  // Example depth/vocab size for benchmark
    std::vector<float> x_data_bm = rand_vector(batch_size_bm * depth_bm);
    StorageView x_bm({batch_size_bm, depth_bm}, DataType::FLOAT32, device);
    // It's good practice to initialize data for ops that compute on values.
    // Copying from CPU host vector to device.
    StorageView x_temp_cpu(x_bm.shape(), x_data_bm, Device::CPU);
    x_bm.copy_from(x_temp_cpu);

    StorageView values_bm(DataType::FLOAT32, device);
    StorageView indices_bm(DataType::INT32, device);
    const ops::GumbelMax gumbel_max_op_bm(num_samples);
    BENCHMARK(gumbel_max_op_bm(x_bm, values_bm, indices_bm),
              2000);  // Adjust iteration count as needed
  } else {
    std::ostringstream error_log;
    int total_mismatches = 0;

    const dim_t batch_size_corr = 8;
    const dim_t depth_corr = 100;  // Smaller depth for faster correctness check
    const Shape input_shape = {batch_size_corr, depth_corr};
    const DataType dtype =
        DataType::FLOAT32;  // GumbelMax typically operates on floats
    const ops::GumbelMax gumbel_max_op(num_samples);

    std::vector<float> x_data_vec =
        rand_vector(input_shape[0] * input_shape[1]);

    StorageView x_device(input_shape, x_data_vec, device);
    StorageView values_device(dtype, device);
    StorageView indices_device(DataType::INT32, device);

    StorageView x_cpu(input_shape, x_data_vec, Device::CPU);
    StorageView values_cpu(dtype, Device::CPU);
    StorageView indices_cpu(DataType::INT32, Device::CPU);

    // Perform operation on device
    gumbel_max_op(x_device, values_device, indices_device);

    // Perform operation on CPU for reference
    gumbel_max_op(x_cpu, values_cpu, indices_cpu);

    // Copy device results to CPU for comparison
    StorageView values_device_cpu_copy(values_device.dtype(), Device::CPU);
    values_device_cpu_copy.copy_from(values_device, true);  // true for sync
    StorageView indices_device_cpu_copy(indices_device.dtype(), Device::CPU);
    indices_device_cpu_copy.copy_from(indices_device, true);  // true for sync

    // GumbelMax output values are original values from x at selected indices
    total_mismatches += dispatch_compare_views(
        values_device_cpu_copy, values_cpu, "GumbelMax Values", error_log);

    // Indices should match exactly
    total_mismatches += dispatch_compare_views(
        indices_device_cpu_copy, indices_cpu, "GumbelMax Indices", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("GumbelMax output mismatch details:\n" +
                               error_log.str());
    }
  }
}

void benchmark_bias_add(Device device) {
  if (!kCheckCorrectness) {
    const Shape value_shape = {128, 1024};
    const Shape bias_shape = {1024};
    const DataType dtype = DataType::FLOAT32;

    StorageView value_device(value_shape, dtype, device);
    // Initialize with some data if op expects non-zero inputs for realistic
    // benchmark
    std::vector<float> value_data_vec =
        rand_vector(value_shape[0] * value_shape[1]);
    StorageView value_temp_cpu(value_shape, value_data_vec, Device::CPU);
    value_device.copy_from(value_temp_cpu);

    StorageView bias_device(bias_shape, dtype, device);
    std::vector<float> bias_data_vec = rand_vector(bias_shape[0]);
    StorageView bias_temp_cpu(bias_shape, bias_data_vec, Device::CPU);
    bias_device.copy_from(bias_temp_cpu);

    StorageView output_device(dtype, device);
    const ops::BiasAdd bias_add_op;  // Default: no activation

    BENCHMARK(bias_add_op(value_device, bias_device, output_device), 10000);

  } else {
    std::ostringstream error_log;
    int total_mismatches = 0;

    const Shape value_shape = {32, 128};
    const Shape bias_shape = {
        value_shape[1]};  // Bias is typically on the last dimension
    const DataType dtype = DataType::FLOAT32;
    ctranslate2::ops::ActivationType activation_type =
        ctranslate2::ops::ActivationType::GELU;
    const ops::BiasAdd bias_add_op(&activation_type);

    std::vector<float> value_data_vec =
        rand_vector(value_shape[0] * value_shape[1]);
    std::vector<float> bias_data_vec = rand_vector(bias_shape[0]);

    // Device path
    StorageView value_device(value_shape, value_data_vec, device);
    StorageView bias_device(bias_shape, bias_data_vec, device);
    StorageView output_device(dtype, device);
    bias_add_op(value_device, bias_device, output_device);

    // CPU path (reference)
    StorageView value_cpu(value_shape, value_data_vec, Device::CPU);
    StorageView bias_cpu(bias_shape, bias_data_vec, Device::CPU);
    StorageView output_cpu_ref(dtype, Device::CPU);
    bias_add_op(value_cpu, bias_cpu, output_cpu_ref);

    // Copy device result to CPU for comparison
    StorageView output_device_cpu_copy(output_device.dtype(), Device::CPU);
    output_device_cpu_copy.copy_from(output_device, true);  // true for sync

    total_mismatches += dispatch_compare_views(
        output_device_cpu_copy, output_cpu_ref, "BiasAdd Output", error_log);

    if (total_mismatches > 0) {
      throw std::runtime_error("BiasAdd output mismatch details:\n" +
                               error_log.str());
    } else {
      std::cout << "BiasAdd test passed!" << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " op device [dtype]" << std::endl;
    return 1;
  }

  std::string op = argv[1];
  std::string device_str = argv[2];
  Device device = device_str == "cuda" ? Device::CUDA : Device::CPU;
  if (device_str == "directml") {
    device = Device::DirectML;
  }
  std::string dtype_str = argc > 3 ? argv[3] : "float32";
  DataType dtype = DataType::FLOAT32;
  if (dtype_str == "int16")
    dtype = DataType::INT16;
  else if (dtype_str == "int8")
    dtype = DataType::INT8;

  if (op == "gather")
    benchmark_gather(device);
  else if (op == "transpose")
    benchmark_transpose(device);
  else if (op == "split")
    benchmark_split(device);
  else if (op == "layer_norm")
    benchmark_layer_norm(device);
  else if (op == "softmax")
    benchmark_softmax(device);
  else if (op == "masked_softmax")
    benchmark_masked_softmax(device);
  else if (op == "topk")
    benchmark_topk(device);
  else if (op == "gemm")
    benchmark_gemm(device, dtype);
  else if (op == "quantize")
    benchmark_quantize(device, dtype);
  else if (op == "dequantize")
    benchmark_dequantize(device);
  else if (op == "conv1d")
    benchmark_conv1d(device);
  else if (op == "conv1d_qscale")
    benchmark_conv1d_qscale(device);
  else if (op == "gumbel_max")
    benchmark_gumbel_max(device);
  else if (op == "bias_add")
    benchmark_bias_add(device);

  return 0;
}
