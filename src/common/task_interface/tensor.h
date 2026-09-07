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
#include <type_traits>
#include <utility>

#include "assert_compat.h"
#include "data_type.h"

/**
 * Buffer Handle
 *
 * Represents a device memory buffer with address and total size in bytes.
 * This is the underlying memory allocation that a ChipTensor describes access patterns for.
 */
struct PTOBufferHandle {
    uint64_t addr;  // Device memory address (bytes)
    uint64_t size;  // Total buffer size in bytes
};

/**
 * TensorArgType - Distinguishes inputs, outputs, and in-place updates.
 *
 * A per-tensor tag carried by TaskArgs (drives dependency inference at submit
 * time; stripped before the args cross the dispatch boundary).
 */
enum class TensorArgType : int32_t {
    INPUT = 0,            // Read-only input buffer
    OUTPUT = 1,           // Write-only output buffer (runtime allocates)
    INOUT = 2,            // Read-then-write: modifier for downstream
    OUTPUT_EXISTING = 3,  // Write-only existing tensor: skips OverlapMap lookup, depends on creator
    NO_DEP = 4,           // No-dependency existing tensor: skips OverlapMap lookup, no publish
};

// `OverlapStatus` / `Segment` (overlap geometry) live in the runtime
// tensormap.h. `TensorCreateInfo` (submit-time create-info for
// runtime-allocated outputs) and its materialization helpers live in the
// runtime tensor_create_info.h. Both are runtime-only and intentionally not
// part of the wire/host-facing ChipTensor definition.

/**
 * ChipTensor — a task argument as it arrives at the chip runtime (72 B).
 *
 * Names a resolved buffer and a strided view of it, and nothing else. A caller
 * knows where the memory is and what shape it is read in; it has no basis for
 * saying which task produced it or how its dependencies should be tracked, so
 * those fields are not here to be filled in. `Runtime::set_orch_args` adopts each
 * argument into the runtime's own `Tensor`, which adds them.
 *
 * The two other tensor types this repository has:
 *   - `Tensor` (src/common/task_interface/buffer.h) — the L3+ wire element. Carries
 *     an embedded buffer *descriptor* and no address, because at submit time none
 *     exists. See docs/buffer-abi.md.
 *   - `simpler::{hbg,tmr}::Tensor` — one runtime's working form, described above.
 *
 * Stride semantics:
 *   - Element-granularity (matches start_offset). Byte offset of element
 *     `coords[]` is `(start_offset + Σ coords[i] · strides[i]) · dtype_bytes`.
 *   - strides[i] > 0 STRICTLY. Broadcast (stride=0) and negative slice step
 *     (stride<0) are NOT supported.
 *
 * Construction:
 * Default construction is public (ChipTensor doubles as ChipStorageTaskArgs
 * storage, which needs a default-constructible element) but yields an
 * UNINITIALIZED object. A *valid* ChipTensor comes from make_tensor_external() or
 * make_tensor_strided().
 */
struct ChipTensor {
    PTOBufferHandle buffer;             // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t start_offset;              // 1D ELEMENT offset of the view origin into `buffer`
    uint32_t shapes[MAX_TENSOR_DIMS];   // View shape per dimension (elements)
    uint32_t strides[MAX_TENSOR_DIMS];  // Element stride per dimension; ALWAYS > 0
    uint32_t ndims;                     // Number of dimensions used
    DataType dtype;                     // Data type of tensor elements
    AddressSpace address_space;         // HOST (default) or DEVICE (child-managed device memory; skips H2D copy)

    ChipTensor() = default;

    /// Number of logical elements covered by the view (NOT the extent).
    /// ndims > 0 is a construction-time invariant, so the loop always runs once.
    [[nodiscard]] uint64_t numel() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++)
            total *= shapes[i];
        return total;
    }

    /// Element extent — the smallest M such that every reachable element lies in
    /// [start_offset, start_offset+M). For strides[i]>0 that is
    /// 1 + Σ (shapes[i]-1) · strides[i]. Computed, not cached: a runtime that reads
    /// this per task caches it on its own Tensor.
    [[nodiscard]] uint64_t extent_elem() const {
        uint64_t last = 0;
        for (uint32_t i = 0; i < ndims; i++) {
            if (shapes[i] == 0) continue;
            last += static_cast<uint64_t>(shapes[i] - 1) * static_cast<uint64_t>(strides[i]);
        }
        return last + 1;
    }

    /// PyTorch-style contiguity: only dimensions of size > 1 must have row-major strides.
    [[nodiscard]] bool is_contiguous() const {
        uint64_t expected = 1;
        for (int32_t i = static_cast<int32_t>(ndims) - 1; i >= 0; --i) {
            if (shapes[i] != 1 && strides[i] != expected) return false;
            expected *= shapes[i];
        }
        return true;
    }

    /// True when `buffer.addr` is a device pointer allocated by the child process
    /// (the host then skips the H2D copy).
    [[nodiscard]] bool is_device_memory() const { return address_space == AddressSpace::DEVICE; }

    /// Logical byte size of the view (numel * element size). For a contiguous
    /// host-constructed tensor this equals buffer.size.
    [[nodiscard]] uint64_t nbytes() const { return numel() * get_element_size(dtype); }

    /// Typed pointer to the tensor's buffer base (== buffer.addr).
    template <typename T>
    T *data_as() const {
        return reinterpret_cast<T *>(static_cast<uintptr_t>(buffer.addr));
    }

    /// Initialize as a contiguous tensor covering `shapes[]` starting at `addr`.
    /// strides become row-major; start_offset = 0.
    void init_external(
        void *addr, uint64_t buffer_size_bytes, const uint32_t in_shapes[], uint32_t in_ndims, DataType in_dtype,
        AddressSpace in_address_space = AddressSpace::HOST
    ) {
        always_assert(in_ndims > 0 && in_ndims <= MAX_TENSOR_DIMS);
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        ndims = in_ndims;
        dtype = in_dtype;
        address_space = in_address_space;
        start_offset = 0;
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(in_ndims) - 1; i >= 0; --i) {
            shapes[i] = in_shapes[i];
            strides[i] = s;
            s *= in_shapes[i];
        }
    }

    [[nodiscard]] std::string dump() const {
        std::stringstream ss;
        const std::string indent = "    ";
        ss << "{" << '\n';
        ss << indent << "buffer.addr: " << buffer.addr << '\n';
        ss << indent << "buffer.size: " << buffer.size << " bytes" << '\n';
        ss << indent << "dtype: " << get_dtype_name(dtype) << '\n';
        ss << indent << "ndims: " << ndims << '\n';
        ss << indent << "start_offset: " << start_offset << " (elements)" << '\n';
        ss << indent << "is_contiguous: " << (is_contiguous() ? "true" : "false") << '\n';
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
};

static_assert(
    std::is_trivially_copyable_v<ChipTensor> && std::is_standard_layout_v<ChipTensor>,
    "ChipTensor crosses the runtime.so ABI as raw bytes"
);
static_assert(sizeof(ChipTensor) == 72, "ChipTensor is geometry plus a resolved address, and nothing else");

// =============================================================================
// ChipTensor factories — the controlled construction entries. Host-side consumers
// (the nanobind binding, make_chip_tensor_arg) and the L2 leaf's materialization
// build arguments through these.
// =============================================================================

/// Contiguous view over pre-allocated external memory: start_offset == 0 and
/// strides == row_major(shapes).
inline ChipTensor make_tensor_external(
    void *addr, const uint32_t shapes[], uint32_t ndims, DataType dtype = DataType::FLOAT32,
    AddressSpace address_space = AddressSpace::HOST
) {
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    ChipTensor t{};
    t.init_external(addr, total * get_element_size(dtype), shapes, ndims, dtype, address_space);
    return t;
}

/// A possibly-strided external view. `addr` is the view origin (start_offset == 0);
/// `strides[]` are element strides and may be non-row-major, as from a transpose /
/// permute / step-sliced wire Tensor. buffer.size is the element extent in bytes.
inline ChipTensor make_tensor_strided(
    void *addr, const uint32_t shapes[], const uint32_t strides[], uint32_t ndims, DataType dtype = DataType::FLOAT32,
    AddressSpace address_space = AddressSpace::HOST
) {
    always_assert(ndims > 0 && ndims <= MAX_TENSOR_DIMS);
    ChipTensor t{};
    t.buffer.addr = reinterpret_cast<uint64_t>(addr);
    t.ndims = ndims;
    t.dtype = dtype;
    t.address_space = address_space;
    t.start_offset = 0;
    for (uint32_t i = 0; i < ndims; i++) {
        t.shapes[i] = shapes[i];
        t.strides[i] = strides[i];
    }
    t.buffer.size = t.extent_elem() * get_element_size(dtype);
    return t;
}
