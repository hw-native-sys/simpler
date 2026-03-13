#pragma once

#include <stdint.h>
#include <memory.h>

#include <sstream>

#include "common.h"
#include "data_type.h"

constexpr int RUNTIME_MAX_TENSOR_DIMS = 5;

/**
 * Buffer Handle
 *
 * Represents a device memory buffer with address and total size in bytes.
 * This is the underlying memory allocation that a Tensor describes access patterns for.
 */
struct PTOBufferHandle {
    uint64_t addr;  // Device memory address (bytes)
    uint64_t size;  // Total buffer size in bytes
};

enum class OverlapStatus {
    NO_OVERLAP,
    COVERED,
    OTHER,
};

struct Segment {
    uint64_t begin;
    uint64_t end;

    bool line_segment_intersection(const Segment& other) const { return end > other.begin && other.end > begin; }
    bool contains(const Segment& other) const { return begin <= other.begin && other.end <= end; }
};

/**
 * Tensor descriptor for Task input/output (128B = 2 cache lines)
 *
 * Describes a memory access pattern on Global Memory (GM) using
 * raw_shapes (underlying buffer dimensions), shapes (current view dimensions),
 * and offsets (multi-dimensional offset into the buffer).
 *
 * - `buffer` contains the underlying memory allocation (addr in bytes, size in bytes)
 * - `raw_shapes[]`, `shapes[]`, `offsets[]` are in ELEMENTS
 * - `dtype` specifies element type for interpreting buffer contents
 *
 * Layout: cache line 1 holds hot-path fields (buffer, start_offset, version,
 * dtype, ndims, shapes); cache line 2 holds warm-path fields (raw_shapes, offsets).
 */
struct alignas(64) Tensor {
    // === Cache line 1 (64B) — hot path ===
    PTOBufferHandle buffer;                        // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t start_offset;                         // Cached 1D element offset (precomputed from raw_shapes + offsets)
    int32_t version;                               // Tensor version for overlap detection
    DataType dtype;                                // Data type of tensor elements
    uint32_t ndims;                                // Number of dimensions used
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];      // Current view shape per dimension

    // === Cache line 2 (64B) — warm path ===
    uint32_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];  // Underlying buffer shape per dimension
    uint32_t offsets[RUNTIME_MAX_TENSOR_DIMS];     // Multi-dimensional offset per dimension

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    Tensor(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t raw_shapes[],
        const uint32_t shapes[],
        const uint32_t offsets[],
        uint32_t ndims,
        DataType dtype,
        int32_t version) {
        init(addr, buffer_size_bytes, raw_shapes, shapes, offsets, ndims, dtype, version);
    }

    // --- Initialization ---
    void init(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t in_raw_shapes[],
        const uint32_t in_shapes[],
        const uint32_t in_offsets[],
        uint32_t in_ndims,
        DataType in_dtype,
        int32_t in_version) {
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        version = in_version;
        for (uint32_t i = 0; i < in_ndims; i++) {
            raw_shapes[i] = in_raw_shapes[i];
            shapes[i] = in_shapes[i];
            offsets[i] = in_offsets[i];
        }
    }

    void init(const Tensor& other) {
        buffer = other.buffer;
        start_offset = other.start_offset;
        version = other.version;
        ndims = other.ndims;
        dtype = other.dtype;
        for (uint32_t i = 0; i < ndims; i++) {
            raw_shapes[i] = other.raw_shapes[i];
            shapes[i] = other.shapes[i];
            offsets[i] = other.offsets[i];
        }
    }

    void init_with_view(const Tensor& other, const uint32_t view_shapes[], const uint32_t view_offsets[]) {
        buffer = other.buffer;
        ndims = other.ndims;
        dtype = other.dtype;
        version = other.version;
        for (uint32_t i = 0; i < ndims; i++) {
            raw_shapes[i] = other.raw_shapes[i];
            shapes[i] = view_shapes[i];
            offsets[i] = other.offsets[i] + view_offsets[i];
        }
    }

    // --- Operations ---
    void update_start_offset() {
        uint64_t result = 0;
        uint64_t stride = 1;
        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            result += offsets[i] * stride;
            stride *= raw_shapes[i];
        }
        start_offset = result;
    }

    void copy(const Tensor &other) {
        init(other);
    }

    Tensor view(const uint32_t view_shapes[], const uint32_t view_offsets[]) const {
        Tensor result;
        result.init_with_view(*this, view_shapes, view_offsets);
        return result;
    }

    bool is_contiguous() const {
        if (ndims == 0) {
            return true;
        }
        for (uint32_t i = 1; i < ndims; i++) {
            if (shapes[i] != raw_shapes[i]) {
                return false;
            }
        }
        return true;
    }

    bool valid_reshape(const uint32_t new_shapes[], uint32_t new_ndims) const {
        uint64_t x = numel();
        uint64_t y = 1;
        for (uint32_t i = 0; i < new_ndims; i++) {
            y *= new_shapes[i];
        }
        return x == y;
    }

    Tensor reshape(const uint32_t new_shapes[], uint32_t new_ndims) const {
        debug_assert(valid_reshape(new_shapes, new_ndims));
        always_assert(is_contiguous());
        Tensor result;
        result.copy(*this);
        result.ndims = new_ndims;
        for (uint32_t i = 0; i < new_ndims; i++) {
            result.raw_shapes[i] = new_shapes[i];
            result.shapes[i] = new_shapes[i];
            result.offsets[i] = 0;
        }
        return result;
    }

    bool valid_transpose(uint32_t x, uint32_t y) const { return x < ndims && y < ndims; }

    Tensor transpose(uint32_t x, uint32_t y) const {
        debug_assert(valid_transpose(x, y));
        Tensor result;
        result.copy(*this);
        std::swap(result.raw_shapes[x], result.raw_shapes[y]);
        std::swap(result.shapes[x], result.shapes[y]);
        std::swap(result.offsets[x], result.offsets[y]);
        return result;
    }

    uint64_t numel() const {
        if (ndims == 0) {
            return 0;
        }
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++) {
            total *= shapes[i];
        }
        return total;
    }

    bool is_same_memref(const Tensor& other) const { return buffer.addr == other.buffer.addr; }

    OverlapStatus is_overlap(const Tensor& pre_task_output) const {
        debug_assert(is_same_memref(pre_task_output));
        debug_assert(version >= pre_task_output.version);
        if (version > pre_task_output.version) {
            return OverlapStatus::OTHER;
        }
        bool contains = true;
        for (uint32_t i = 0; i < ndims; i++) {
            Segment input_range_dim_i{offsets[i], offsets[i] + (uint64_t)shapes[i]};
            Segment output_range_dim_i{
                pre_task_output.offsets[i], pre_task_output.offsets[i] + (uint64_t)pre_task_output.shapes[i]};
            if (!input_range_dim_i.line_segment_intersection(output_range_dim_i)) {
                return OverlapStatus::NO_OVERLAP;
            } else if (!input_range_dim_i.contains(output_range_dim_i)) {
                contains = false;
            }
        }
        if (contains) {
            return OverlapStatus::COVERED;
        }
        return OverlapStatus::OTHER;
    }

    std::string dump() const {
        std::stringstream ss;
        std::string indent = "    ";
        ss << "{" << std::endl;
        ss << indent << "buffer.addr: " << buffer.addr << std::endl;
        ss << indent << "buffer.size: " << buffer.size << " bytes" << std::endl;
        ss << indent << "dtype: " << get_dtype_name(dtype) << std::endl;
        ss << indent << "ndims: " << ndims << std::endl;
        ss << indent << "version: " << version << std::endl;

        ss << indent << "raw_shapes: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << raw_shapes[i];
        }
        ss << "]" << std::endl;
        ss << indent << "shapes: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << shapes[i];
        }
        ss << "]" << std::endl;
        ss << indent << "offsets: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << offsets[i];
        }
        ss << "]" << std::endl;
        ss << "}" << std::endl;
        return ss.str();
    }
};

static_assert(sizeof(Tensor) == 128, "Tensor must be exactly 2 cache lines (128 bytes)");

using TensorData = Tensor;

// =============================================================================
// Factory Helpers
// =============================================================================
/**
 * Create a Tensor for pre-allocated external memory.
 */
static inline Tensor make_tensor_external(void* addr,
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(addr, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version);
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * NO memory allocation: only records dtype, shape, and buffer.size in the Tensor struct.
 * The runtime allocates from the heap ring and fills buffer.addr during pto2_submit_task
 * when this tensor is passed as OUTPUT param. No buffer content is ever copied.
 */
static inline Tensor make_tensor(const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(0, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version);
}
