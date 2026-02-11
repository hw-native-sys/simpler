/**
 * Tensor Descriptor - Overlap detection implementations
 */

#include "tensor.h"

#include <algorithm>
#include <sstream>

std::string to_str(OverlapType overlap_type) {
    switch (overlap_type) {
#define CASE(X)            \
    case OverlapType::X: { \
        return #X;         \
    }
        CASE(Accurate)
        CASE(Fuzzy)
#undef CASE
        default:
            always_assert(false);
    }
    return "";
}

// ContiguousMemSegIterator implementations
Tensor::ContiguousMemSegIterator::ContiguousMemSegIterator(const Tensor& tensor)
    : tensor_(tensor), cur_seg({tensor.start_offset, tensor.start_offset + tensor.repeats[tensor.ndims - 1]}) {
    for (uint64_t i = 0; i < tensor.ndims; i++) {
        indexes_[i] = 0;
    }
}

void Tensor::ContiguousMemSegIterator::operator++() {
    debug_assert(!is_end());
    debug_assert(tensor_.ndims > 1 || (tensor_.ndims == 1 && indexes_[0] == 0));
    indexes_[tensor_.ndims - 1] += tensor_.repeats[tensor_.ndims - 1];
    cur_seg.begin += tensor_.repeats[tensor_.ndims - 1];
    for (int32_t i = tensor_.ndims - 1; i >= 1; i--) {
        debug_assert(indexes_[i] <= tensor_.repeats[i]);
        if (indexes_[i] == tensor_.repeats[i]) {
            indexes_[i - 1]++;
            indexes_[i] = 0;
            // Jump to next outer dimension iteration:
            // outer_stride - (inner_stride * inner_repeats)
            cur_seg.begin += tensor_.strides[i - 1] - tensor_.strides[i] * tensor_.repeats[i];
        }
    }
    cur_seg.end = cur_seg.begin + tensor_.repeats[tensor_.ndims - 1];
}

// Tensor constructors
Tensor::Tensor(uint64_t addr,
    uint64_t buffer_size_bytes,
    uint64_t start_offset,
    uint64_t strides[],
    uint64_t repeats[],
    uint64_t ndims,
    DataType dtype,
    int32_t version,
    OverlapType overlap_type)
    : buffer{addr, buffer_size_bytes},
      start_offset(start_offset),
      ndims(ndims),
      dtype(dtype),
      version(version),
      overlap_type(overlap_type) {
    for (uint64_t i = 0; i < ndims; i++) {
        this->strides[i] = strides[i];
        this->repeats[i] = repeats[i];
    }
    debug_assert(Tensor(*this).optimize().is_valid_tensor());
}

Tensor::Tensor(Tensor&& other)
    : buffer(other.buffer),
      start_offset(other.start_offset),
      ndims(other.ndims),
      dtype(other.dtype),
      version(other.version),
      overlap_type(other.overlap_type) {
    for (uint64_t i = 0; i < ndims; i++) {
        strides[i] = other.strides[i];
        repeats[i] = other.repeats[i];
    }
}

Tensor::Tensor(const Tensor& other)
    : buffer(other.buffer),
      start_offset(other.start_offset),
      ndims(other.ndims),
      dtype(other.dtype),
      version(other.version),
      overlap_type(other.overlap_type) {
    for (uint64_t i = 0; i < ndims; i++) {
        strides[i] = other.strides[i];
        repeats[i] = other.repeats[i];
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    buffer = other.buffer;
    start_offset = other.start_offset;
    ndims = other.ndims;
    dtype = other.dtype;
    version = other.version;
    overlap_type = other.overlap_type;
    for (uint64_t i = 0; i < ndims; i++) {
        strides[i] = other.strides[i];
        repeats[i] = other.repeats[i];
    }
    return *this;
}

std::string Tensor::dump() const {
    std::stringstream ss;
    std::string indent = "    ";
    ss << "{" << std::endl;
    ss << indent << "buffer.addr: " << buffer.addr << std::endl;
    ss << indent << "buffer.size: " << buffer.size << " bytes" << std::endl;
    ss << indent << "dtype: " << get_dtype_name(dtype) << std::endl;
    ss << indent << "start_offset: " << start_offset << " elements" << std::endl;
    ss << indent << "ndims: " << ndims << std::endl;
    ss << indent << "version: " << version << std::endl;
    ss << indent << "overlap_type: " << to_str(overlap_type) << std::endl;

    ss << indent << "strides: [";
    for (uint64_t i = 0; i < ndims; i++) {
        if (i > 0) {
            ss << ", ";
        }
        ss << strides[i];
    }
    ss << "] (elements)" << std::endl;
    ss << indent << "repeats: [";
    for (uint64_t i = 0; i < ndims; i++) {
        if (i > 0) {
            ss << ", ";
        }
        ss << repeats[i];
    }
    ss << "]" << std::endl;
    ss << "}" << std::endl;
    return ss.str();
}

bool Tensor::is_valid_tensor() const {
    if (strides[ndims - 1] != 1) {
        return false;
    }
    // After resort_strides, strides are sorted in descending order:
    // strides[0] >= strides[1] >= ... >= strides[ndims-1] = 1
    for (uint64_t i = 1; i < ndims; i++) {
        // Check descending order
        if (strides[i] > strides[i - 1]) {
            return false;
        }
        // Outer stride must be divisible by inner stride
        if (strides[i - 1] % strides[i] != 0) {
            return false;
        }
        // Inner block must not exceed outer stride
        if (strides[i - 1] < strides[i] * repeats[i]) {
            return false;
        }
    }
    // get_fuzzy_seg() returns element offsets, convert to bytes and check against buffer.size
    Segment fuzzy_seg = get_fuzzy_seg();
    uint64_t end_byte_offset = fuzzy_seg.end * get_element_size(dtype);
    if (end_byte_offset > (uint64_t)buffer.size) {
        return false;
    }
    return true;
}

Tensor& Tensor::optimize() {
#ifndef NDEBUG
    // 仅在 debug 模式下保存原始数据用于验证
    uint64_t original_strides[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t original_repeats[RUNTIME_MAX_TENSOR_DIMS];
    int32_t original_ndims = ndims;
    for (uint64_t i = 0; i < ndims; i++) {
        original_strides[i] = this->strides[i];
        original_repeats[i] = this->repeats[i];
    }
#endif
    resort_strides();

#ifndef NDEBUG
    debug_assert(validate_memory_access_preserved(original_strides, original_repeats, original_ndims));
#endif
    return *this;
}

void Tensor::resort_strides() {
    for (uint64_t i = 0; i < ndims; i++) {
        for (uint64_t j = i + 1; j < ndims; j++) {
            if (strides[i] < strides[j] || (strides[i] == strides[j] && repeats[i] < repeats[j])) {
                std::swap(strides[i], strides[j]);
                std::swap(repeats[i], repeats[j]);
            }
        }
    }
}

#ifndef NDEBUG
bool Tensor::validate_memory_access_preserved(
    uint64_t original_strides[], uint64_t original_repeats[], int32_t original_ndims) const {
    // 收集原始 tensor 访问的所有 offset
    auto original_offsets = collect_all_offsets(original_strides, original_repeats, original_ndims);

    // 收集处理后 tensor 访问的所有 offset
    auto processed_offsets = collect_all_offsets(strides, repeats, ndims);

    // 排序后比较
    std::sort(original_offsets.begin(), original_offsets.end());
    std::sort(processed_offsets.begin(), processed_offsets.end());

    return original_offsets == processed_offsets;
}

std::vector<uint64_t> Tensor::collect_all_offsets(
    const uint64_t strides_arr[], const uint64_t repeats_arr[], int32_t dims) const {
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> idx(dims, 0);
    while (true) {
        // 计算当前索引对应的 offset
        uint64_t offset = start_offset;
        for (int32_t i = 0; i < dims; i++) {
            offset += idx[i] * strides_arr[i];
        }
        offsets.push_back(offset);

        // 递增多维索引（从最内层开始）
        int32_t dim = dims - 1;
        while (dim >= 0) {
            idx[dim]++;
            if (idx[dim] < repeats_arr[dim]) {
                break;
            }
            idx[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;  // 遍历完成
        }
    }
    return offsets;
}
#endif

bool Tensor::valid_view(const uint64_t shapes[], const uint64_t offsets[]) const {
    for (size_t i = 0; i < ndims; i++) {
        if (shapes[i] + offsets[i] > repeats[i]) {
            return false;
        }
    }
    return true;
}

bool Tensor::valid_reshape(const uint64_t shapes[], uint64_t new_ndims) const {
    uint64_t x = 1;
    for (size_t i = 0; i < ndims; i++) {
        x *= repeats[i];
    }
    uint64_t y = 1;
    for (size_t i = 0; i < new_ndims; i++) {
        y *= shapes[i];
    }
    return x == y;
}

Tensor Tensor::view(const uint64_t shapes[], const uint64_t offsets[]) const {
    debug_assert(valid_view(shapes, offsets));
    Tensor result(*this);

    // 计算新的 start_offset: 原 offset + sum(offsets[i] * strides[i])
    result.start_offset = start_offset + offset_ndim_to_1d(offsets);

    // strides 保持不变，repeats 更新为新的 shapes
    for (size_t i = 0; i < ndims; i++) {
        result.repeats[i] = shapes[i];
    }

    return result;
}

bool Tensor::is_contiguous() const {
    if (ndims == 0) {
        return true;
    }
    if (strides[ndims - 1] != 1) {
        return false;
    }
    for (int32_t i = ndims - 2; i >= 0; i--) {
        if (strides[i] != strides[i + 1] * repeats[i + 1]) {
            return false;
        }
    }
    return true;
}

uint64_t Tensor::numel() const {
    if (ndims == 0) {
        return 0;
    }
    uint64_t total = 1;
    for (uint64_t i = 0; i < ndims; i++) {
        total *= repeats[i];
    }
    return total;
}

Tensor Tensor::reshape(const uint64_t shapes[], uint64_t new_ndims) const {
    debug_assert(valid_reshape(shapes, new_ndims));

    // 目前我们期望reshape的对象一定是连续内存，先限制使用场景
    always_assert(is_contiguous());

    // 计算新的 strides（row-major 布局）和 repeats
    uint64_t new_strides[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t new_repeats[RUNTIME_MAX_TENSOR_DIMS];
    // int32_t new_ndims = static_cast<int32_t>(shapes.size());

    uint64_t stride = 1;
    for (int i = new_ndims - 1; i >= 0; i--) {
        new_strides[i] = stride;
        new_repeats[i] = shapes[i];
        stride *= shapes[i];
    }

    return Tensor(
        buffer.addr, buffer.size, start_offset, new_strides, new_repeats, new_ndims, dtype, version, overlap_type);
}

Tensor Tensor::transpose(uint64_t x, uint64_t y) const {
    debug_assert(valid_transpose(x, y));
    Tensor result(*this);

    // 直接交换 strides 和 repeats
    std::swap(result.strides[x], result.strides[y]);
    std::swap(result.repeats[x], result.repeats[y]);

    return result;
}

Segment Tensor::get_fuzzy_seg() const {
    uint64_t end_offset = start_offset;
    for (uint64_t i = 0; i < ndims; i++) {
        end_offset += strides[i] * repeats[i];
    }
    return {start_offset, end_offset};
}

bool Tensor::is_same_strides(const Tensor& other) const {
    for (uint64_t i = 0; i < ndims; i++) {
        if (strides[i] != other.strides[i]) {
            return false;
        }
    }
    return true;
}

void Tensor::offset_to_ndims(uint64_t offset_ndims[]) const {
    uint64_t cur_offset = start_offset;
    for (uint64_t i = 0; i < ndims; i++) {
        offset_ndims[i] = cur_offset / strides[i];
        cur_offset %= strides[i];
    }
}

uint64_t Tensor::offset_ndim_to_1d(const uint64_t offset_ndims[]) const {
    uint64_t result = 0;
    for (uint64_t i = 0; i < ndims; i++) {
        result += offset_ndims[i] * strides[i];
    }
    return result;
}

OverlapStatus Tensor::is_overlap(const Tensor& pre_task_output) const {
    printf("input: %s\n", dump().c_str());
    printf("output: %s\n", pre_task_output.dump().c_str());
    if (!is_same_memref(pre_task_output)) {
        return OverlapStatus::NO_OVERLAP;
    }
    debug_assert(version >= pre_task_output.version);
    if (version > pre_task_output.version) {
        return OverlapStatus::OTHER;
    }

    // Convert element offsets to byte offsets for comparison
    // This handles cases where the two descriptors have different dtypes
    uint64_t elem_size_input = get_element_size(dtype);
    uint64_t elem_size_output = get_element_size(pre_task_output.dtype);

    Segment input_memory_fuzzy_seg = get_fuzzy_seg();
    Segment output_memory_fuzzy_seg = pre_task_output.get_fuzzy_seg();

    // Convert to byte offsets
    Segment input_byte_seg{
        input_memory_fuzzy_seg.begin * elem_size_input, input_memory_fuzzy_seg.end * elem_size_input};
    Segment output_byte_seg{
        output_memory_fuzzy_seg.begin * elem_size_output, output_memory_fuzzy_seg.end * elem_size_output};

    if (!input_byte_seg.line_segment_intersection(output_byte_seg)) {
        return OverlapStatus::NO_OVERLAP;
    }

    // 只做模糊判断
    if (pre_task_output.overlap_type == OverlapType::Fuzzy) {
        return OverlapStatus::OTHER;
    }

    // 一维场景
    if (ndims == 1 && pre_task_output.ndims == 1) {
        debug_assert(strides[0] == 1);
        debug_assert(pre_task_output.strides[0] == 1);
        if (input_byte_seg.contains(output_byte_seg)) {
            return OverlapStatus::COVERED;
        } else {
            return OverlapStatus::OTHER;
        }
    }

    // 精准判断 - only if same dtype and strides
    // For different dtypes, we fall back to complex_overlap
    if (dtype == pre_task_output.dtype && ndims == pre_task_output.ndims && is_same_strides(pre_task_output)) {
        uint64_t input_offset_ndims[RUNTIME_MAX_TENSOR_DIMS];
        uint64_t output_offset_ndims[RUNTIME_MAX_TENSOR_DIMS];
        offset_to_ndims(input_offset_ndims);
        pre_task_output.offset_to_ndims(output_offset_ndims);
        // O(ndims) 判断超矩形间overlap
        bool need_complex_compare = false;
        bool contains = true;
        bool overlap = true;
        for (uint64_t i = 0; i < ndims; i++) {
            Segment input_range_dim_i{input_offset_ndims[i], input_offset_ndims[i] + repeats[i]};
            Segment output_range_dim_i{output_offset_ndims[i], output_offset_ndims[i] + pre_task_output.repeats[i]};
            // Skip outermost dimension (i == 0), check inner dimensions
            // With descending strides, strides[i-1] is the outer dimension's stride
            if (i > 0) {
                // input不是超矩形
                if (input_range_dim_i.end * strides[i] > strides[i - 1]) {
                    need_complex_compare = true;
                    break;
                }
                // output不是超矩形
                if (output_range_dim_i.end * pre_task_output.strides[i] > pre_task_output.strides[i - 1]) {
                    need_complex_compare = true;
                    break;
                }
            }
            if (!input_range_dim_i.line_segment_intersection(output_range_dim_i)) {
                overlap = false;
            } else if (!input_range_dim_i.contains(output_range_dim_i)) {
                contains = false;
            }
        }
        if (!need_complex_compare) {
            if (contains) {
                return OverlapStatus::COVERED;
            } else if (overlap) {
                return OverlapStatus::OTHER;
            } else {
                return OverlapStatus::NO_OVERLAP;
            }
        }
    }
    // O(\prod repeats[i]) 判断线段相交
    return complex_overlap(pre_task_output) ? OverlapStatus::OTHER : OverlapStatus::NO_OVERLAP;
}

bool Tensor::complex_overlap(const Tensor& pre_task_output) const {
#ifndef NDEBUG
    OverlapPathTracker::record_complex_call();
#endif
    // Convert element offsets to byte offsets for comparison when dtypes differ
    uint64_t elem_size_input = get_element_size(dtype);
    uint64_t elem_size_output = get_element_size(pre_task_output.dtype);

    ContiguousMemSegIterator input_segs_iter(*this);
    ContiguousMemSegIterator output_segs_iter(pre_task_output);
    while (!input_segs_iter.is_end() && !output_segs_iter.is_end()) {
        const Segment& cur_input_memory_seg = *input_segs_iter;
        const Segment& cur_output_memory_seg = *output_segs_iter;

        // Convert to byte offsets for comparison
        Segment input_byte_seg{
            cur_input_memory_seg.begin * elem_size_input, cur_input_memory_seg.end * elem_size_input};
        Segment output_byte_seg{
            cur_output_memory_seg.begin * elem_size_output, cur_output_memory_seg.end * elem_size_output};

        if (input_byte_seg.end <= output_byte_seg.begin) {
            input_segs_iter++;
            continue;
        } else if (output_byte_seg.end <= input_byte_seg.begin) {
            output_segs_iter++;
            continue;
        }
        return true;
    }
    return false;
}