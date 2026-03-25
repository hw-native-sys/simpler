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
 * - `layout` specifies the relationship between logical (shapes) and physical (raw_shapes) dimensions
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
    uint64_t start_offset;                         // Cached 1D element offset (precomputed from raw_shapes + offsets), only calc before incore, useless in orch
    int32_t version;                               // Tensor version for overlap detection
    DataType dtype;                                // Data type of tensor elements
    TensorLayout layout;                           // Tensor layout (ND or DN) for logical-to-physical dimension mapping
    uint32_t ndims;                                // Number of dimensions used
    bool is_all_offset_zero;                       // True when all offsets[] are zero (skip offset read/write)
    bool is_raw_eq_shapes;                         // True when raw_shapes[] == shapes[] (skip raw_shapes read/write)
    bool manual_dep;                               // True when dependency is managed manually (skip tensormap lookup/insert)
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
        TensorLayout layout = TensorLayout::ND,
        bool is_all_offset_zero = false,
        bool is_raw_eq_shapes = false,
        bool manual_dep = false) {
        init(addr, buffer_size_bytes, raw_shapes, shapes, offsets, ndims, dtype, version,
             layout, is_all_offset_zero, is_raw_eq_shapes, manual_dep);
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
        TensorLayout in_layout = TensorLayout::ND,
        bool in_is_all_offset_zero = false,
        bool in_is_raw_eq_shapes = false,
        bool in_manual_dep = false) {
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        version = in_version;
        layout = in_layout;
        is_all_offset_zero = in_is_all_offset_zero;
        is_raw_eq_shapes = in_is_raw_eq_shapes;
        manual_dep = in_manual_dep;
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
        memcpy(this, &other, 64); // fast copy cache line 1
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

    void init_with_view(const Tensor& other, const uint32_t view_shapes[], const uint32_t view_offsets[], bool in_manual_dep = false) {
        buffer = other.buffer;
        ndims = other.ndims;
        dtype = other.dtype;
        version = other.version;
        manual_dep = in_manual_dep;
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

    Tensor view(const uint32_t view_shapes[], const uint32_t view_offsets[], bool manual_dep = false) const {
        Tensor result;
        if (layout == TensorLayout::DN && ndims >= 2) {
            always_assert(ndims <= RUNTIME_MAX_TENSOR_DIMS);
            uint32_t adjusted_offsets[RUNTIME_MAX_TENSOR_DIMS];
            for (uint32_t i = 0; i < ndims - 2; i++) {
                adjusted_offsets[i] = view_offsets[i];
            }
            // Swap only the last two dimensions (DN reverses row/col order)
            adjusted_offsets[ndims - 2] = view_offsets[ndims - 1];
            adjusted_offsets[ndims - 1] = view_offsets[ndims - 2];
            result.init_with_view(*this, view_shapes, adjusted_offsets, manual_dep);
        } else {
            result.init_with_view(*this, view_shapes, view_offsets, manual_dep);
        }
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

    Tensor reshape(const uint32_t new_shapes[], uint32_t new_ndims, bool manual_dep = false) const {
        debug_assert(valid_reshape(new_shapes, new_ndims));
        always_assert(is_contiguous());
        Tensor result;
        result.copy(*this);
        result.ndims = new_ndims;
        result.is_all_offset_zero = true;
        result.is_raw_eq_shapes = true;
        result.manual_dep = manual_dep;
        for (uint32_t i = 0; i < new_ndims; i++) {
            result.shapes[i] = new_shapes[i];
        }
        return result;
    }

    bool valid_transpose(uint32_t x, uint32_t y) const { return x < ndims && y < ndims; }

    Tensor transpose(uint32_t x, uint32_t y, bool manual_dep = false) const {
        debug_assert(valid_transpose(x, y));
        Tensor result;
        result.copy(*this);
        result.manual_dep = manual_dep;
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

    std::string dump() const {
        std::stringstream ss;
        std::string indent = "    ";
        ss << "{" << std::endl;
        ss << indent << "buffer.addr: " << buffer.addr << std::endl;
        ss << indent << "buffer.size: " << buffer.size << " bytes" << std::endl;
        ss << indent << "dtype: " << get_dtype_name(dtype) << std::endl;
        ss << indent << "ndims: " << ndims << std::endl;
        ss << indent << "version: " << version << std::endl;
        ss << indent << "layout: " << (layout == TensorLayout::ND ? "ND" : "DN") << std::endl;

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
static_assert(offsetof(Tensor, raw_shapes) == 64);

using TensorData = Tensor;

// =============================================================================
// Factory Helpers
// =============================================================================
/**
 * Create a Tensor for pre-allocated external memory.
 *
 * For DN layout, raw_shapes (physical layout) will have the last two
 * dimensions swapped compared to the logical shapes.
 */
template<TensorLayout Layout = TensorLayout::ND>
static inline Tensor make_tensor_external(void* addr,
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    bool manual_dep = false,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    uint64_t buf_size = total * get_element_size(dtype);

    if constexpr (Layout == TensorLayout::DN) {
        always_assert(ndims >= 2 && ndims <= RUNTIME_MAX_TENSOR_DIMS);
        uint32_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];
        for (uint32_t i = 0; i < ndims - 2; i++) raw_shapes[i] = shapes[i];
        raw_shapes[ndims - 2] = shapes[ndims - 1];
        raw_shapes[ndims - 1] = shapes[ndims - 2];
        return Tensor(addr, buf_size, raw_shapes, shapes, zero_offsets,
                      ndims, dtype, version, TensorLayout::DN,
                      /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/false, manual_dep);
    } else {
        return Tensor(addr, buf_size, shapes, shapes, zero_offsets,
                      ndims, dtype, version, TensorLayout::ND,
                      /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep);
    }
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * NO memory allocation: only records dtype, shape, and buffer.size in the Tensor struct.
 * The runtime allocates from the heap ring and fills buffer.addr during pto2_submit_task
 * when this tensor is passed as OUTPUT param. No buffer content is ever copied.
 *
 * For DN layout, raw_shapes (physical layout) will have the last two
 * dimensions swapped compared to the logical shapes.
 */
template<TensorLayout Layout = TensorLayout::ND>
static inline Tensor make_tensor(const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    bool manual_dep = false,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    uint64_t buf_size = total * get_element_size(dtype);

    if constexpr (Layout == TensorLayout::DN) {
        always_assert(ndims >= 2 && ndims <= RUNTIME_MAX_TENSOR_DIMS);
        uint32_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];
        for (uint32_t i = 0; i < ndims - 2; i++) raw_shapes[i] = shapes[i];
        raw_shapes[ndims - 2] = shapes[ndims - 1];
        raw_shapes[ndims - 1] = shapes[ndims - 2];
        return Tensor(0, buf_size, raw_shapes, shapes, zero_offsets,
                      ndims, dtype, version, TensorLayout::DN,
                      /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/false, manual_dep);
    } else {
        return Tensor(0, buf_size, shapes, shapes, zero_offsets,
                      ndims, dtype, version, TensorLayout::ND,
                      /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep);
    }
}

// =============================================================================
// Layout Reinterpretation
// =============================================================================

/**
 * Zero-cost layout reinterpretation.
 *
 * Returns a copy of src with the layout field changed to NewLayout.
 * raw_shapes (physical layout) are unchanged; shapes[] last two dimensions
 * are swapped to reflect the new logical interpretation.
 *
 * Uses Tensor::copy() for cache-line-aware copying — avoids touching
 * cache line 2 when is_raw_eq_shapes and is_all_offset_zero are both true.
 *
 * Safe for both addr=0 (make_tensor) and addr!=0 (make_tensor_external)
 * tensors — this is pure metadata reinterpretation, no data is moved.
 */
template<TensorLayout NewLayout>
static inline Tensor reinterpret_layout(const Tensor& src) {
    if (src.layout == NewLayout) {
        return src;
    }

    always_assert(src.ndims >= 2 && src.ndims <= RUNTIME_MAX_TENSOR_DIMS);

    Tensor dst;
    dst.copy(src);
    dst.layout = NewLayout;

    if (src.is_raw_eq_shapes) {
        // src is ND full tensor → dst is DN → materialize raw_shapes, set false
        for (uint32_t i = 0; i < dst.ndims; i++) {
            dst.raw_shapes[i] = src.shapes[i];
        }
        dst.is_raw_eq_shapes = false;
    }

    std::swap(dst.shapes[dst.ndims - 2], dst.shapes[dst.ndims - 1]);

    // DN→ND: check if swap restores shapes == raw_shapes (full tensor round-trip)
    if constexpr (NewLayout == TensorLayout::ND) {
        if (!src.is_raw_eq_shapes) {
            bool eq = true;
            for (uint32_t i = 0; i < dst.ndims; i++) {
                if (dst.shapes[i] != dst.raw_shapes[i]) { eq = false; break; }
            }
            dst.is_raw_eq_shapes = eq;
        }
    }

    if (!dst.is_all_offset_zero) {
        std::swap(dst.offsets[dst.ndims - 2], dst.offsets[dst.ndims - 1]);
    }

    return dst;
}
