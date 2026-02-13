/**
 * Tensor methods needed by orchestration .so
 *
 * Contains constructors/operator= and the methods they transitively call
 * (optimize → resort_strides → is_valid_tensor → get_fuzzy_seg, plus
 * debug-only validate_memory_access_preserved / collect_all_offsets).
 *
 * Both runtime targets (aicore/aicpu/host) and the orchestration .so
 * compile this file. The remaining Tensor methods stay in tensor.cpp
 * and are only compiled into runtime targets.
 */

#include "tensor.h"

#include <algorithm>
#include <sstream>

// =============================================================================
// Constructors and assignment
// =============================================================================

Tensor::Tensor(uint64_t addr,
    uint64_t buffer_size_bytes,
    uint64_t start_offset,
    const uint64_t strides[],
    const uint64_t repeats[],
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

// =============================================================================
// Validation and optimization (called by constructor's debug_assert)
// =============================================================================

Segment Tensor::get_fuzzy_seg() const {
    uint64_t end_offset = start_offset;
    for (uint64_t i = 0; i < ndims; i++) {
        end_offset += strides[i] * (repeats[i] - 1);
    }
    return {start_offset, end_offset + 1};
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

Tensor& Tensor::optimize() {
#ifndef NDEBUG
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

#ifndef NDEBUG
bool Tensor::validate_memory_access_preserved(
    uint64_t original_strides[], uint64_t original_repeats[], int32_t original_ndims) const {
    auto original_offsets = collect_all_offsets(original_strides, original_repeats, original_ndims);
    auto processed_offsets = collect_all_offsets(strides, repeats, ndims);

    std::sort(original_offsets.begin(), original_offsets.end());
    std::sort(processed_offsets.begin(), processed_offsets.end());

    return original_offsets == processed_offsets;
}

std::vector<uint64_t> Tensor::collect_all_offsets(
    const uint64_t strides_arr[], const uint64_t repeats_arr[], int32_t dims) const {
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> idx(dims, 0);
    while (true) {
        uint64_t offset = start_offset;
        for (int32_t i = 0; i < dims; i++) {
            offset += idx[i] * strides_arr[i];
        }
        offsets.push_back(offset);

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
            break;
        }
    }
    return offsets;
}
#endif

// =============================================================================
// Methods needed by orchestration .so (view, reshape, transpose, etc.)
// =============================================================================

uint64_t Tensor::offset_ndim_to_1d(const uint64_t offset_ndims[]) const {
    uint64_t result = 0;
    for (uint64_t i = 0; i < ndims; i++) {
        result += offset_ndims[i] * strides[i];
    }
    return result;
}

uint64_t Tensor::offset_ndim_to_1d(const std::vector<uint64_t>& offset_ndims) const {
    uint64_t result = 0;
    for (uint64_t i = 0; i < ndims; i++) {
        result += offset_ndims[i] * strides[i];
    }
    return result;
}

bool Tensor::valid_view(const uint64_t shapes[], const uint64_t offsets[]) const {
    for (size_t i = 0; i < ndims; i++) {
        if (shapes[i] + offsets[i] > repeats[i]) {
            return false;
        }
    }
    return true;
}

Tensor Tensor::view(const uint64_t shapes[], const uint64_t offsets[]) const {
    debug_assert(valid_view(shapes, offsets));
    Tensor result(*this);
    result.start_offset = start_offset + offset_ndim_to_1d(offsets);
    for (size_t i = 0; i < ndims; i++) {
        result.repeats[i] = shapes[i];
    }
    return result;
}

Tensor Tensor::view(const std::vector<uint64_t>& shapes, const std::vector<uint64_t>& offsets) const {
    Tensor result(*this);
    result.start_offset = start_offset + offset_ndim_to_1d(offsets);
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

Tensor Tensor::reshape(const uint64_t shapes[], uint64_t new_ndims) const {
    debug_assert(valid_reshape(shapes, new_ndims));
    always_assert(is_contiguous());
    uint64_t new_strides[RUNTIME_MAX_TENSOR_DIMS];
    uint64_t new_repeats[RUNTIME_MAX_TENSOR_DIMS];
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
    std::swap(result.strides[x], result.strides[y]);
    std::swap(result.repeats[x], result.repeats[y]);
    return result;
}

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
