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

// =============================================================================
// Constructors and assignment
// =============================================================================

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
