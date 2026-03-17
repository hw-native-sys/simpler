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
 * Fast-path flags (both on cache line 1):
 * - is_all_offset_zero: when true, offsets[] are implicitly zero — skip offset read/write
 * - is_raw_eq_shapes: when true, raw_shapes[] == shapes[] — skip raw_shapes read/write,
 *   use shapes[] wherever raw_shapes would be needed
 *
 * When BOTH flags are true, cache line 2 is never accessed.
 *
 * Layout: cache line 1 holds hot-path fields (buffer, start_offset, version,
 * dtype, ndims, flags, shapes); cache line 2 holds warm-path fields (raw_shapes, offsets).
 */
struct alignas(64) Tensor {
    // === Cache line 1 (64B) — hot path ===
    PTOBufferHandle buffer;                        // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t start_offset;                         // Cached 1D element offset (precomputed from raw_shapes + offsets)
    int32_t version;                               // Tensor version for overlap detection
    DataType dtype;                                // Data type of tensor elements
    uint32_t ndims;                                // Number of dimensions used
    bool is_all_offset_zero;                       // True when all offsets[] are zero (skip offset read/write)
    bool is_raw_eq_shapes;                         // True when raw_shapes[] == shapes[] (skip raw_shapes read/write)
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

    /// Return the effective raw_shapes pointer (shapes[] when is_raw_eq_shapes).
    /// Avoids cache line 2 access for the common case.
    const uint32_t* get_raw_shapes() const {
        return is_raw_eq_shapes ? shapes : raw_shapes;
    }

    Tensor(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t raw_shapes[],
        const uint32_t shapes[],
        const uint32_t offsets[],
        uint32_t ndims,
        DataType dtype,
        int32_t version,
        bool is_all_offset_zero = false,
        bool is_raw_eq_shapes = false) {
        init(addr, buffer_size_bytes, raw_shapes, shapes, offsets, ndims, dtype, version,
             is_all_offset_zero, is_raw_eq_shapes);
    }

    // --- Initialization ---
    void init(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t in_raw_shapes[],
        const uint32_t in_shapes[],
        const uint32_t in_offsets[],
        uint32_t in_ndims,
        DataType in_dtype,
        int32_t in_version,
        bool in_is_all_offset_zero = false,
        bool in_is_raw_eq_shapes = false) {
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        version = in_version;
        is_all_offset_zero = in_is_all_offset_zero;
        is_raw_eq_shapes = in_is_raw_eq_shapes;
        for (uint32_t i = 0; i < in_ndims; i++) {
            shapes[i] = in_shapes[i];
        }
        if (!in_is_raw_eq_shapes) {
            for (uint32_t i = 0; i < in_ndims; i++) {
                raw_shapes[i] = in_raw_shapes[i];
            }
        }
        if (!in_is_all_offset_zero) {
            for (uint32_t i = 0; i < in_ndims; i++) {
                offsets[i] = in_offsets[i];
            }
        }
    }

    void init(const Tensor& other) {
        buffer = other.buffer;
        start_offset = other.start_offset;
        version = other.version;
        ndims = other.ndims;
        dtype = other.dtype;
        is_all_offset_zero = other.is_all_offset_zero;
        is_raw_eq_shapes = other.is_raw_eq_shapes;
        for (uint32_t i = 0; i < ndims; i++) {
            shapes[i] = other.shapes[i];
        }
        if (!other.is_raw_eq_shapes) {
            for (uint32_t i = 0; i < ndims; i++) {
                raw_shapes[i] = other.raw_shapes[i];
            }
        }
        if (!other.is_all_offset_zero) {
            for (uint32_t i = 0; i < ndims; i++) {
                offsets[i] = other.offsets[i];
            }
        }
    }

    void init_with_view(const Tensor& other, const uint32_t view_shapes[], const uint32_t view_offsets[]) {
        buffer = other.buffer;
        ndims = other.ndims;
        dtype = other.dtype;
        version = other.version;
        // view always diverges shapes from raw_shapes, so is_raw_eq_shapes = false.
        // Read parent's effective raw_shapes (avoids parent cache line 2 when parent is_raw_eq_shapes).
        is_raw_eq_shapes = false;
        const uint32_t* parent_raw = other.get_raw_shapes();
        for (uint32_t i = 0; i < ndims; i++) {
            raw_shapes[i] = parent_raw[i];
            shapes[i] = view_shapes[i];
        }
        // Compute offsets and zero-flag
        bool all_zero = true;
        if (other.is_all_offset_zero) {
            for (uint32_t i = 0; i < ndims; i++) {
                if (view_offsets[i] != 0) { all_zero = false; break; }
            }
            if (!all_zero) {
                for (uint32_t i = 0; i < ndims; i++) {
                    offsets[i] = view_offsets[i];
                }
            }
        } else {
            all_zero = false;
            for (uint32_t i = 0; i < ndims; i++) {
                offsets[i] = other.offsets[i] + view_offsets[i];
            }
        }
        is_all_offset_zero = all_zero;
    }

    // --- Operations ---
    void update_start_offset() {
        if (is_all_offset_zero) {
            start_offset = 0;
            return;
        }
        const uint32_t* rs = get_raw_shapes();
        uint64_t result = 0;
        uint64_t stride = 1;
        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            result += offsets[i] * stride;
            stride *= rs[i];
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
        if (is_raw_eq_shapes || ndims == 0) {
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
        result.is_all_offset_zero = true;
        result.is_raw_eq_shapes = true;
        for (uint32_t i = 0; i < new_ndims; i++) {
            result.shapes[i] = new_shapes[i];
        }
        return result;
    }

    bool valid_transpose(uint32_t x, uint32_t y) const { return x < ndims && y < ndims; }

    Tensor transpose(uint32_t x, uint32_t y) const {
        debug_assert(valid_transpose(x, y));
        Tensor result;
        result.copy(*this);
        // transpose swaps the same dims in both arrays, so equality is preserved
        std::swap(result.shapes[x], result.shapes[y]);
        if (!result.is_raw_eq_shapes) {
            std::swap(result.raw_shapes[x], result.raw_shapes[y]);
        }
        if (!result.is_all_offset_zero) {
            std::swap(result.offsets[x], result.offsets[y]);
        }
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
        // Fast path: both tensors have zero offsets — ranges are [0, shape[i])
        // Overlap is guaranteed; COVERED iff this->shapes[i] >= other.shapes[i] for all dims.
        // Avoids reading offsets[] entirely (stays on cache line 1).
        if (is_all_offset_zero && pre_task_output.is_all_offset_zero) {
            bool contains = true;
            for (uint32_t i = 0; i < ndims; i++) {
                if (shapes[i] < pre_task_output.shapes[i]) {
                    contains = false;
                    break;
                }
            }
            return contains ? OverlapStatus::COVERED : OverlapStatus::OTHER;
        }
        // Slow path: at least one tensor has non-zero offsets
        bool contains = true;
        for (uint32_t i = 0; i < ndims; i++) {
            uint64_t this_off = is_all_offset_zero ? 0 : offsets[i];
            uint64_t other_off = pre_task_output.is_all_offset_zero ? 0 : pre_task_output.offsets[i];
            Segment input_range_dim_i{this_off, this_off + (uint64_t)shapes[i]};
            Segment output_range_dim_i{other_off, other_off + (uint64_t)pre_task_output.shapes[i]};
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

        const uint32_t* rs = get_raw_shapes();
        ss << indent << "raw_shapes: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << rs[i];
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
            ss << (is_all_offset_zero ? 0u : offsets[i]);
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
    return Tensor(addr, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true);
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
    return Tensor(0, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true);
}
