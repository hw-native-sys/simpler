/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

#include <memory.h>
#include <stdint.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "common.h"
#include "data_type.h"
#include "pto_task_id.h"

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

    bool line_segment_intersection(const Segment &other) const { return end > other.begin && other.end > begin; }
    bool contains(const Segment &other) const { return begin <= other.begin && other.end <= end; }
};

/**
 * TensorCreateInfo — submit-time create-info for runtime-allocated outputs.
 *
 * Carries the metadata required to materialize a fresh contiguous output:
 * dtype, ndims, shapes, manual_dep, and an optional initial value fill.
 *
 * Layout (64B) is aligned with Tensor cache line 1 so that
 * init_from_create_info() can copy the entire cache line with a single memcpy,
 * then overwrite buffer/owner metadata and compute the contiguous stride in
 * cache line 2.
 *
 * Arg::add_output() stores a pointer to this object, so the original
 * must remain valid (not a temporary) until after the submit call.
 */
class alignas(64) TensorCreateInfo {
public:
    TensorCreateInfo(
        const uint32_t shapes_in[], uint32_t ndims_in, DataType dtype_in = DataType::FLOAT32, bool manual_dep_in = false
    ) :
        initial_value(0),
        has_initial_value(false),
        __pad2__(0),
        start_offset(0),  // mirrors Tensor::start_offset; pre-zeroed for create-info outputs
        version(0),
        ndims(ndims_in),
        dtype(dtype_in),
        manual_dep(manual_dep_in),
        is_contiguous(true),  // mirrors Tensor::is_contiguous; pre-set for create-info outputs
        __pad_flags__(0) {
        for (uint32_t i = 0; i < ndims_in; i++) {
            shapes[i] = shapes_in[i];
        }
    }

    void copy(const TensorCreateInfo &other) { memcpy(this, &other, sizeof(other)); }

    template <typename T = uint64_t>
    void set_initial_value(T value) {
        has_initial_value = true;
        initial_value = to_u64(value);
    }

    uint64_t buffer_size_bytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++) {
            total *= shapes[i];
        }
        return total * get_element_size(dtype);
    }

public:
    // --- Bytes [0, 32): TensorCreateInfo-only fields ---
    // These occupy the same positions as Tensor::buffer, Tensor::owner_task_id,
    // and Tensor::start_offset. The runtime overwrites owner metadata after the
    // memcpy and recomputes start_offset / stride during payload materialization.
    uint64_t initial_value;
    bool has_initial_value;
    uint8_t __pad1__[7];
    uint64_t __pad2__;      // → Tensor::owner_task_id (overwritten post-memcpy)
    uint64_t start_offset;  // mirrors Tensor::start_offset; always 0 for create-info outputs

    // --- Bytes [32, 64): Matches Tensor cache line 1 layout ---
    int32_t version;  // Always 0 for create-info outputs
    uint32_t ndims;
    DataType dtype;
    bool manual_dep;
    bool is_contiguous;  // Always true for create-info outputs
    uint8_t __pad_flags__;
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // → Tensor::shapes

    TensorCreateInfo() = default;

    friend struct Arg;
};

static_assert(sizeof(TensorCreateInfo) == 64);

/**
 * Tensor descriptor for Task input/output (128B = 2 cache lines)
 *
 * Describes a strided memory access pattern on Global Memory (GM) using:
 *   - `buffer`: underlying memory allocation (addr/size in bytes)
 *   - `start_offset`: 1D element offset of the view origin from `buffer.addr`
 *   - `shapes[i]`, `strides[i]`: per-dim view shape and **element** stride
 *
 * Stride semantics:
 *   - Element-granularity (matches start_offset). Byte offset of element
 *     `coords[]` is `(start_offset + Σ coords[i] · strides[i]) · dtype_bytes`.
 *   - strides[i] > 0 STRICTLY. Broadcast (stride=0) and negative slice step
 *     (stride<0) are NOT supported.
 *
 * Fast-path flags on cache line 1:
 *   - manual_dep: when true, dependency tracking is creator-only (skip OverlapMap)
 *   - is_contiguous: cached PyTorch-style contiguous flag — i.e.
 *     `strides[i] == prod(shapes[i+1..ndims-1])`. When true AND start_offset==0,
 *     all hot paths can compute extent_elem from `shapes` alone and never read
 *     cache line 2. NOTE: this is strictly tighter than the pre-#808
 *     `shapes[i] == raw_shapes[i]` test, but equivalent on every view the old
 *     (raw_shapes-based) encoding could express; the two only diverge on
 *     post-#808-only views (transpose / permute / slice-with-step results).
 *
 * Layout: cache line 1 holds hot-path fields (buffer, owner_task_id,
 * start_offset, version, ndims, dtype, flags, shapes); cache line 2 holds
 * stride + cached extent_elem.
 *
 * Construction:
 * Users cannot default-construct or directly construct a Tensor.
 * Valid Tensors are obtained only through controlled entry points:
 *   - make_tensor_external(...)
 *   - from_tensor_arg(...)
 *   - TaskOutputTensors returned by submit(...)
 *   - Tensor::view() / reshape() / transpose() / permute() / slice() on an existing valid Tensor
 */
struct alignas(64) Tensor {
    // === Cache line 1 (64B) — hot path ===
    PTOBufferHandle buffer;    // Underlying memory buffer (addr in bytes, size in bytes)
    PTO2TaskId owner_task_id;  // Creator task; PTO2TaskId::invalid() for external tensors
    uint64_t start_offset;     // 1D ELEMENT offset of the view origin into `buffer`
    int32_t version;           // Tensor version for overlap detection
    uint32_t ndims;            // Number of dimensions used
    DataType dtype;            // Data type of tensor elements
    bool manual_dep;           // True when dependency tracking is creator-only (skip OverlapMap lookup/insert)
    bool is_contiguous;        // Cached: strides[] == row_major_stride(shapes)
    uint8_t _pad_cl1;          // Pad to align shapes[5] at byte 44
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // Current view shape per dimension (elements)

    // === Cache line 2 (64B) — warm path (view metadata) ===
    // Field order: place the 8B-aligned cache before the 4B-aligned strides[]
    // to avoid 4B padding between them (sizeof(Tensor) must stay 128).
    uint64_t extent_elem_cache;                 // Cached extent_elem (see extent_elem()); maintained by ops
    uint32_t strides[RUNTIME_MAX_TENSOR_DIMS];  // Element stride per dimension; ALWAYS > 0 (type-enforced)
    uint8_t _pad_cl2[36];                       // Reserved for future extension

    // --- Copy / move / destroy ---
    // Kept trivially copyable (default copy = byte-for-byte) so other modules
    // (PTO2TensorMapEntry::copy_from_tensor, TensorCreateInfo memcpy path)
    // can rely on memcpy semantics. The contiguous fast-path optimization
    // lives in `init(const Tensor&)`; call sites that care should use
    // `result.init(*this)` instead of the default copy ctor.
    Tensor(const Tensor &) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor(Tensor &&) = default;
    Tensor &operator=(Tensor &&) = default;
    ~Tensor() = default;

    // ========================================================================
    // Accessors / helpers
    // ========================================================================

    /// Number of logical elements covered by the view (NOT the extent).
    /// ndims > 0 is a construction-time invariant (see init_external /
    /// init_from_create_info), so the loop always runs at least once.
    uint64_t numel() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++)
            total *= shapes[i];
        return total;
    }

    /// Element extent — the smallest M such that every reachable element lies in [start_offset, start_offset+M).
    /// For strides[i]>0: extent_elem = 1 + Σ (shapes[i]-1) · strides[i].
    uint64_t extent_elem() const {
        if (is_contiguous) return numel();  // fast path: line 2 not needed when contiguous
        return extent_elem_cache;
    }

    // ========================================================================
    // Initialization (operates on already-constructed Tensor)
    // ========================================================================

    /// Initialize as a contiguous tensor that covers `shapes[]` starting at `addr`.
    /// stride is set to row_major(shapes); start_offset = 0; is_contiguous = true.
    /// Enforces the ndims > 0 invariant relied upon by every downstream op.
    void init_external(
        void *addr, uint64_t buffer_size_bytes, const uint32_t in_shapes[], uint32_t in_ndims, DataType in_dtype,
        int32_t in_version, bool in_manual_dep = false
    ) {
        always_assert(in_ndims > 0 && in_ndims <= RUNTIME_MAX_TENSOR_DIMS);
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        version = in_version;
        manual_dep = in_manual_dep;
        is_contiguous = true;
        _pad_cl1 = 0;
        start_offset = 0;
        owner_task_id = PTO2TaskId::invalid();
        // Single reverse pass: write shapes, accumulate row-major stride, and
        // track numel — `s` ends as prod(shapes) which is also extent_elem
        // for a contiguous view.
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(in_ndims) - 1; i >= 0; --i) {
            shapes[i] = in_shapes[i];
            strides[i] = s;
            s *= in_shapes[i];
        }
        extent_elem_cache = s;
    }

    /// Deep copy with contiguous fast-path optimization.
    ///
    /// Always copies cache line 1 (always needed: buffer, shapes, dtype, ...).
    /// When `other` is in canonical contiguous form (is_contiguous &&
    /// start_offset == 0), cache line 2 (stride / extent_elem_cache) is fully
    /// derivable from line 1, so we **skip reading other's cache line 2** and
    /// write dst's line 2 from the local shapes instead. Non-contiguous source
    /// pays one line 2 read; contiguous source does not.
    void init_from(const Tensor &other) {
        init_from_line1(other);
        if (other.is_contiguous && other.start_offset == 0) {
            // Derive line 2 from line 1: stride = row-major of shapes; extent = numel.
            uint32_t s = 1;
            for (int32_t i = static_cast<int32_t>(ndims) - 1; i >= 0; --i) {
                strides[i] = s;
                s *= shapes[i];
            }
            extent_elem_cache = s;
        } else {
            extent_elem_cache = other.extent_elem_cache;
            for (uint32_t i = 0; i < other.ndims; i++) {
                strides[i] = other.strides[i];
            }
            // _pad_cl2 left stale on purpose — reserved bytes are not
            // semantically read by any consumer.
        }
    }

    /// View ops use this: copy cache line 1 only, leaving cache line 2 (stride,
    /// extent_elem_cache) untouched. The op then mutates shapes / start_offset
    /// in place and calls `refresh_derived()` to recompute line 2 once. This
    /// avoids the wasted line 2 writes that `init_from()` would do just before
    /// the op overwrites them.
    void init_from_line1(const Tensor &other) { memcpy(this, &other, 64); }

    /// Backward-compat alias used by orchestrator hot paths that need a full
    /// deep copy. Equivalent to `init_from(other)`.
    void copy(const Tensor &other) { init_from(other); }

    /// Materialize a TensorCreateInfo into this Tensor (fresh contiguous output).
    /// Single 64B memcpy covers cache line 1; ci pre-initialises start_offset (=0)
    /// and is_contiguous (=true) in its line-1 slots so they need no reset here.
    /// Cache line 2 (stride/extent) is computed from `ci.shapes` in a single reverse pass.
    void init_from_create_info(const TensorCreateInfo &ci, void *addr, uint64_t buffer_size) {
        always_assert(ci.ndims > 0 && ci.ndims <= RUNTIME_MAX_TENSOR_DIMS);
        memcpy(this, &ci, 64);
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size};
        owner_task_id = PTO2TaskId::invalid();  // caller (orchestrator) overwrites with actual task_id
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(ndims) - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shapes[i];
        }
        extent_elem_cache = s;
        if (ci.has_initial_value) {
            fill_initial_value(ci.initial_value);
        }
    }

    void fill_initial_value(uint64_t initial_value) {
        always_assert(reinterpret_cast<char *>(buffer.addr) != nullptr);
        uint64_t elem_size = get_element_size(dtype);
        char *dst = reinterpret_cast<char *>(buffer.addr);
        constexpr uint64_t blk_size = 64;
        uint64_t blk = (buffer.size < blk_size) ? buffer.size : blk_size;
        for (uint64_t b = 0; b < blk; b += elem_size) {
            memcpy(dst + b, &initial_value, elem_size);
        }
        uint64_t filled = blk;
        while (filled < buffer.size) {
            uint64_t copy_size = ((buffer.size - filled) < filled) ? (buffer.size - filled) : filled;
            memcpy(dst + filled, dst, copy_size);
            filled += copy_size;
        }
    }

    // ========================================================================
    // Address / offset computation
    // ========================================================================

    /// Compute 1D flat ELEMENT offset of `indices[]` from `buffer.addr`.
    /// Callers multiply by `get_element_size(dtype)` to obtain a byte offset.
    /// Works for any view (transpose / permute / slice / reshape).
    uint64_t compute_flat_offset(const uint32_t indices[], uint32_t in_ndims) const {
        uint64_t elem_off = start_offset;
        for (uint32_t d = 0; d < in_ndims; d++) {
            elem_off += static_cast<uint64_t>(indices[d]) * static_cast<uint64_t>(strides[d]);
        }
        return elem_off;
    }

    // ========================================================================
    // View operations (zero-copy metadata rewrites)
    // ========================================================================

    /// Sub-tensor at per-dim offsets, with new per-dim shape.
    /// Updates start_offset += Σ off[i]·strides[i]; shapes := new_shape; stride unchanged.
    /// Each (offset[i], new_shape[i]) must stay within the current shapes[i] —
    /// i.e. a view cannot expand any dimension beyond what the parent view sees.
    Tensor view(const uint32_t view_shapes[], const uint32_t view_offsets[], bool in_manual_dep = false) const {
        Tensor result;
        // Copy line 1 only; stride from *this is still in result's line 2 garbage
        // — we need to bring it forward explicitly since view keeps stride.
        result.init_from_line1(*this);
        for (uint32_t i = 0; i < ndims; i++) {
            debug_assert(view_offsets[i] + view_shapes[i] <= shapes[i]);
            result.start_offset += static_cast<uint64_t>(view_offsets[i]) * static_cast<uint64_t>(strides[i]);
            result.shapes[i] = view_shapes[i];
            result.strides[i] = strides[i];
        }
        result.manual_dep = in_manual_dep;
        result.refresh_derived();
        result.assert_in_buffer_bounds();
        return result;
    }

    bool valid_transpose(uint32_t x, uint32_t y) const { return x < ndims && y < ndims; }

    /// Swap two dimensions: shapes/stride swapped together. start_offset unchanged.
    Tensor transpose(uint32_t x, uint32_t y, bool in_manual_dep = false) const {
        debug_assert(valid_transpose(x, y));
        Tensor result;
        result.init_from_line1(*this);
        // Carry forward source's stride before swapping (line 2 was not memcpy'd).
        for (uint32_t i = 0; i < ndims; i++)
            result.strides[i] = strides[i];
        std::swap(result.shapes[x], result.shapes[y]);
        std::swap(result.strides[x], result.strides[y]);
        result.manual_dep = in_manual_dep;
        result.refresh_derived();
        return result;
    }

    /// Permute dimensions according to `order[]` (length = ndims).
    /// Both shapes and stride are reordered in-place; start_offset unchanged.
    Tensor permute(const uint32_t order[], bool in_manual_dep = false) const {
        Tensor result;
        result.init_from_line1(*this);
        for (uint32_t i = 0; i < ndims; i++) {
            debug_assert(order[i] < ndims);
            result.shapes[i] = shapes[order[i]];
            result.strides[i] = strides[order[i]];
        }
        result.manual_dep = in_manual_dep;
        result.refresh_derived();
        return result;
    }

    /// Slice dimension `dim` with `[start, end)` and positive `step`.
    /// strides[dim] *= step; shapes[dim] = ⌈(end-start)/step⌉; start_offset += start·strides[dim_old].
    Tensor slice(uint32_t dim, uint32_t start, uint32_t end, uint32_t step = 1, bool in_manual_dep = false) const {
        debug_assert(dim < ndims);
        debug_assert(step >= 1);
        debug_assert(end > start);
        debug_assert(end <= shapes[dim]);
        Tensor result;
        result.init_from_line1(*this);
        // Carry forward source's stride before patching the sliced dim.
        for (uint32_t i = 0; i < ndims; i++)
            result.strides[i] = strides[i];
        const uint32_t old_stride_d = strides[dim];
        result.start_offset += static_cast<uint64_t>(start) * static_cast<uint64_t>(old_stride_d);
        const uint32_t new_len = (end - start + step - 1) / step;
        result.shapes[dim] = new_len;
        result.strides[dim] = static_cast<uint32_t>(static_cast<uint64_t>(old_stride_d) * step);
        result.manual_dep = in_manual_dep;
        result.refresh_derived();
        result.assert_in_buffer_bounds();
        return result;
    }

    bool valid_reshape(const uint32_t new_shapes[], uint32_t new_ndims) const {
        uint64_t x = numel();
        uint64_t y = 1;
        for (uint32_t i = 0; i < new_ndims; i++)
            y *= new_shapes[i];
        return x == y;
    }

    /// Reshape — zero-copy only if source is_contiguous; otherwise asserts.
    /// Materialize fallback (allocating a contiguous copy) is NOT in this op;
    /// callers must reach contiguous via a copy before calling reshape on a
    /// non-contiguous view.
    Tensor reshape(const uint32_t new_shapes[], uint32_t new_ndims, bool in_manual_dep = false) const {
        debug_assert(valid_reshape(new_shapes, new_ndims));
        always_assert(is_contiguous);
        Tensor result;
        result.init_from_line1(*this);
        result.ndims = new_ndims;
        result.manual_dep = in_manual_dep;
        // Single reverse pass: write new shapes, accumulate row-major stride, track numel.
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(new_ndims) - 1; i >= 0; --i) {
            result.shapes[i] = new_shapes[i];
            result.strides[i] = s;
            s *= new_shapes[i];
        }
        result.is_contiguous = true;
        result.extent_elem_cache = s;
        return result;
    }

    // ========================================================================
    // Dump for diagnostics
    // ========================================================================

    std::string dump() const {
        std::stringstream ss;
        std::string indent = "    ";
        ss << "{" << '\n';
        ss << indent << "buffer.addr: " << buffer.addr << '\n';
        ss << indent << "buffer.size: " << buffer.size << " bytes" << '\n';
        ss << indent << "dtype: " << get_dtype_name(dtype) << '\n';
        ss << indent << "ndims: " << ndims << '\n';
        ss << indent << "version: " << version << '\n';
        ss << indent << "start_offset: " << start_offset << " (elements)" << '\n';
        ss << indent << "is_contiguous: " << (is_contiguous ? "true" : "false") << '\n';
        ss << indent << "shapes: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) ss << ", ";
            ss << shapes[i];
        }
        ss << "]" << '\n';
        ss << indent << "strides: [";
        for (uint32_t i = 0; i < ndims; i++) {
            if (i > 0) ss << ", ";
            ss << strides[i];
        }
        ss << "]" << '\n';
        ss << "}" << '\n';
        return ss.str();
    }

private:
    // Default and parameterized constructors are private.
    // Valid Tensors come only from controlled entry points.
    Tensor() = default;

    Tensor(
        void *addr, uint64_t buffer_size_bytes, const uint32_t in_shapes[], uint32_t in_ndims, DataType in_dtype,
        int32_t in_version, bool in_manual_dep = false
    ) {
        init_external(addr, buffer_size_bytes, in_shapes, in_ndims, in_dtype, in_version, in_manual_dep);
    }

    // ------------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------------

    /// Recompute extent_elem_cache and is_contiguous from current shapes / stride.
    /// Called after any op that mutates view metadata. Single reverse pass:
    ///   extent_elem += (shapes[i] - 1) · strides[i]
    ///   is_contiguous &&= (strides[i] == prod(shapes[i+1..]))
    void refresh_derived() {
        uint64_t e = 1;
        uint64_t expected = 1;
        bool contig = true;
        for (int32_t i = static_cast<int32_t>(ndims) - 1; i >= 0; --i) {
            if (strides[i] != expected) contig = false;
            if (shapes[i] > 0) {
                e += static_cast<uint64_t>(shapes[i] - 1) * static_cast<uint64_t>(strides[i]);
            }
            expected *= shapes[i];
        }
        extent_elem_cache = e;
        is_contiguous = contig;
    }

    /// Assert the view stays inside the underlying buffer (byte-range safety).
    void assert_in_buffer_bounds() const {
        const uint64_t elem_size = get_element_size(dtype);
        const uint64_t buffer_elems = buffer.size / elem_size;
        debug_assert(start_offset + extent_elem_cache <= buffer_elems);
    }

    // Friends that need to construct Tensors
    friend struct PTO2TaskPayload;
    friend inline Tensor make_tensor_external(
        void *addr, const uint32_t shapes[], uint32_t ndims, DataType dtype, bool manual_dep, int32_t version
    );
};

static_assert(sizeof(Tensor) == 128, "Tensor must be exactly 2 cache lines (128 bytes)");
static_assert(offsetof(Tensor, owner_task_id) == 16, "owner_task_id must be at bytes 16-23 (cacheline 1)");
static_assert(offsetof(Tensor, start_offset) == 24, "start_offset must be at bytes 24-31 (cacheline 1)");
static_assert(offsetof(Tensor, version) == 32);
static_assert(offsetof(Tensor, ndims) == 36);
static_assert(offsetof(Tensor, dtype) == 40);
static_assert(offsetof(Tensor, manual_dep) == 41);
static_assert(offsetof(Tensor, is_contiguous) == 42);
static_assert(offsetof(Tensor, shapes) == 44, "shapes must start at byte 44 (cacheline 1)");
static_assert(offsetof(Tensor, extent_elem_cache) == 64, "extent_elem_cache must start at byte 64 (cacheline 2)");
static_assert(offsetof(Tensor, strides) == 72);

// TensorCreateInfo layout must match Tensor cacheline 1 for memcpy optimization
static_assert(sizeof(TensorCreateInfo) == 64, "TensorCreateInfo must match Tensor cacheline 1 size (64 bytes)");
static_assert(offsetof(TensorCreateInfo, start_offset) == offsetof(Tensor, start_offset));
static_assert(offsetof(TensorCreateInfo, version) == offsetof(Tensor, version));
static_assert(offsetof(TensorCreateInfo, ndims) == offsetof(Tensor, ndims));
static_assert(offsetof(TensorCreateInfo, dtype) == offsetof(Tensor, dtype));
static_assert(offsetof(TensorCreateInfo, manual_dep) == offsetof(Tensor, manual_dep));
static_assert(offsetof(TensorCreateInfo, is_contiguous) == offsetof(Tensor, is_contiguous));
static_assert(offsetof(TensorCreateInfo, shapes) == offsetof(Tensor, shapes));
