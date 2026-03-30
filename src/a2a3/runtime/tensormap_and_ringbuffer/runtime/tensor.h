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

#include "common.h"     // NOLINT(build/include_subdir)
#include "data_type.h"  // NOLINT(build/include_subdir)

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
 * TensorCreateInfo — submit-time create-info for runtime-allocated outputs.
 *
 * Carries the metadata required to materialize a fresh contiguous output:
 * dtype, ndims, raw_shapes (== shapes), manual_dep, and an optional
 * initial value fill.
 *
 * Layout (64B) is aligned with Tensor cacheline 1 so that
 * init_from_create_info() can copy the entire cacheline with a single memcpy,
 * then overwrite buffer.addr and buffer.size separately.
 *
 * Arg::add_output() stores a pointer to this object, so the original
 * must remain valid (not a temporary) until after the submit call.
 */
class TensorCreateInfo {
public:  // NOLINT(whitespace/indent)
    TensorCreateInfo(
        const uint32_t shapes[], uint32_t ndims, DataType dtype = DataType::FLOAT32, bool manual_dep = false)
        : initial_value(0),
          has_initial_value(false),
          version(0),
          dtype(dtype),
          ndims(ndims),
          is_all_offset_zero(true),
          is_raw_eq_shapes(true),
          manual_dep(manual_dep) {
        for (uint32_t i = 0; i < ndims; i++) {
            raw_shapes[i] = shapes[i];
        }
    }

    void copy(const TensorCreateInfo& other) { memcpy(this, &other, sizeof(other)); }

    template <typename T = uint64_t>
    void set_initial_value(T value) {
        has_initial_value = true;
        initial_value = to_u64(value);
    }

    uint64_t buffer_size_bytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++) {
            total *= raw_shapes[i];
        }
        return total * get_element_size(dtype);
    }

public:  // NOLINT(whitespace/indent)
    // --- Bytes [0, 24): TensorCreateInfo-only fields ---
    // These occupy the same positions as Tensor::buffer and Tensor::start_offset,
    // which are overwritten after the memcpy in init_from_create_info().
    uint64_t initial_value;
    bool has_initial_value;
    uint8_t __pad1__[7];
    uint64_t __pad2__;  // → Tensor::start_offset (zeroed)

    // --- Bytes [24, 64): Matches Tensor cacheline 1 layout ---
    int32_t version;  // Always 0 for create-info outputs
    DataType dtype;
    uint32_t ndims;
    bool is_all_offset_zero;  // Always true for create-info outputs
    bool is_raw_eq_shapes;    // Always true for create-info outputs
    bool manual_dep;
    uint8_t __pad3__;
    uint32_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];  // → Tensor::shapes
    uint32_t __pad4__;

    TensorCreateInfo() = default;

    friend struct Arg;
};

static_assert(sizeof(TensorCreateInfo) == 64);

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
 * Fast-path flags (all on cache line 1):
 * - is_all_offset_zero: when true, offsets[] are implicitly zero — skip offset read/write
 * - is_raw_eq_shapes: when true, raw_shapes[] == shapes[] — skip raw_shapes read/write,
 *   use shapes[] wherever raw_shapes would be needed
 * - manual_dep: when true, dependency tracking is managed manually
 *
 * When BOTH flags are true, cache line 2 is never accessed.
 *
 * Layout: cache line 1 holds hot-path fields (buffer, start_offset, version,
 * dtype, ndims, flags, shapes); cache line 2 holds warm-path fields (raw_shapes, offsets).
 *
 * Construction:
 * Users cannot default-construct or directly construct a Tensor.
 * Valid Tensors are obtained only through controlled entry points:
 *   - make_tensor_external(...)
 *   - from_tensor_arg(...)
 *   - TaskOutputTensors returned by submit(...)
 *   - Tensor::view() / reshape() / transpose() on an existing valid Tensor
 */
struct alignas(64) Tensor {
    // === Cache line 1 (64B) — hot path ===
    PTOBufferHandle buffer;   // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t start_offset;    // Cached 1D element offset (precomputed from raw_shapes + offsets), only calc before
                              // incore, useless in orch
    int32_t version;          // Tensor version for overlap detection
    DataType dtype;           // Data type of tensor elements
    uint32_t ndims;           // Number of dimensions used
    bool is_all_offset_zero;  // True when all offsets[] are zero (skip offset read/write)
    bool is_raw_eq_shapes;    // True when raw_shapes[] == shapes[] (skip raw_shapes read/write)
    bool manual_dep;          // True when dependency is managed manually (skip tensormap lookup/insert)
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // Current view shape per dimension
    uint32_t __padding__;

    // === Cache line 2 (64B) — warm path ===
    uint32_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];  // Underlying buffer shape per dimension
    uint32_t offsets[RUNTIME_MAX_TENSOR_DIMS];     // Multi-dimensional offset per dimension

    // --- Copy / move / destroy are public (valid tensors can be freely copied) ---
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    /// Return the effective raw_shapes pointer (shapes[] when is_raw_eq_shapes).
    /// Avoids cache line 2 access for the common case.
    const uint32_t* get_raw_shapes() const { return is_raw_eq_shapes ? shapes : raw_shapes; }

    // --- Initialization (operates on already-constructed Tensor) ---
    void init(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t in_raw_shapes[],
        const uint32_t in_shapes[],
        const uint32_t in_offsets[],
        uint32_t in_ndims,
        DataType in_dtype,
        int32_t in_version,
        bool in_is_all_offset_zero = false,
        bool in_is_raw_eq_shapes = false,
        bool in_manual_dep = false) {
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        version = in_version;
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
        memcpy(this, &other, 64);  // fast copy cache line 1
        if (!other.is_raw_eq_shapes) {
            for (uint32_t i = 0; i < other.ndims; i++) {
                raw_shapes[i] = other.raw_shapes[i];
            }
        }
        if (!other.is_all_offset_zero) {
            for (uint32_t i = 0; i < other.ndims; i++) {
                offsets[i] = other.offsets[i];
            }
        }
    }

    void init_with_view(
        const Tensor& other, const uint32_t view_shapes[], const uint32_t view_offsets[], bool in_manual_dep = false) {
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
                if (view_offsets[i] != 0) {
                    all_zero = false;
                    break;
                }
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

    /// Compute 1D flat element offset from multi-dimensional indices.
    /// Uses Horner's method (forward traversal, no stride variable).
    uint64_t compute_flat_offset(const uint32_t indices[], uint32_t in_ndims) const {
        if (in_ndims == 0) return 0;
        const uint32_t* rs = get_raw_shapes();
        uint64_t offset = 0;
        if (is_all_offset_zero) {
            for (uint32_t d = 0; d < in_ndims; d++) offset = offset * rs[d] + indices[d];
        } else {
            for (uint32_t d = 0; d < in_ndims; d++) offset = offset * rs[d] + indices[d] + offsets[d];
        }
        return offset;
    }

    /// Materialize a TensorCreateInfo into this Tensor (fresh contiguous output).
    /// Single 64B memcpy covers the entire cacheline 1, then buffer is overwritten.
    void init_from_create_info(const TensorCreateInfo& ci, void* addr, uint64_t buffer_size) {
        memcpy(this, &ci, 64);
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size};
        if (ci.has_initial_value) {
            fill_initial_value(ci.initial_value);
        }
    }

    void fill_initial_value(uint64_t initial_value) {
        always_assert(reinterpret_cast<char*>(buffer.addr) != nullptr);
        uint64_t elem_size = get_element_size(dtype);
        char* dst = reinterpret_cast<char*>(buffer.addr);
        constexpr uint64_t BLK = 64;
        uint64_t blk = (buffer.size < BLK) ? buffer.size : BLK;
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

    void copy(const Tensor& other) { init(other); }

    Tensor view(const uint32_t view_shapes[], const uint32_t view_offsets[], bool manual_dep = false) const {
        Tensor result;
        result.init_with_view(*this, view_shapes, view_offsets, manual_dep);
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

private:
    // Default and parameterized constructors are private.
    // Valid Tensors come only from controlled entry points.
    Tensor() = default;

    Tensor(void* addr,
        uint64_t buffer_size_bytes,
        const uint32_t raw_shapes[],
        const uint32_t shapes[],
        const uint32_t offsets[],
        uint32_t ndims,
        DataType dtype,
        int32_t version,
        bool is_all_offset_zero = false,
        bool is_raw_eq_shapes = false,
        bool manual_dep = false) {
        init(addr,
            buffer_size_bytes,
            raw_shapes,
            shapes,
            offsets,
            ndims,
            dtype,
            version,
            is_all_offset_zero,
            is_raw_eq_shapes,
            manual_dep);
    }

    // Friends that need to construct Tensors
    friend struct PTO2TaskPayload;
    friend inline Tensor make_tensor_external(
        void* addr, const uint32_t shapes[], uint32_t ndims, DataType dtype, bool manual_dep, int32_t version);
};

static_assert(sizeof(Tensor) == 128, "Tensor must be exactly 2 cache lines (128 bytes)");
static_assert(offsetof(Tensor, raw_shapes) == 64);

// TensorCreateInfo layout must match Tensor cacheline 1 for memcpy optimization
static_assert(sizeof(TensorCreateInfo) == 64, "TensorCreateInfo must match Tensor cacheline 1 size (64 bytes)");
static_assert(offsetof(TensorCreateInfo, version) == offsetof(Tensor, version));
static_assert(offsetof(TensorCreateInfo, dtype) == offsetof(Tensor, dtype));
static_assert(offsetof(TensorCreateInfo, ndims) == offsetof(Tensor, ndims));
static_assert(offsetof(TensorCreateInfo, is_all_offset_zero) == offsetof(Tensor, is_all_offset_zero));
static_assert(offsetof(TensorCreateInfo, is_raw_eq_shapes) == offsetof(Tensor, is_raw_eq_shapes));
static_assert(offsetof(TensorCreateInfo, manual_dep) == offsetof(Tensor, manual_dep));
static_assert(offsetof(TensorCreateInfo, raw_shapes) == offsetof(Tensor, shapes));
