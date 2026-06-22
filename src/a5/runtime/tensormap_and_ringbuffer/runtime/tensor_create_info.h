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
/**
 * TensorCreateInfo — submit-time create-info for runtime-allocated outputs.
 *
 * Runtime-only: this header (and the materialization helpers below) are NOT
 * part of the wire/host-facing Tensor in src/common/task_interface/tensor.h.
 * It carries the metadata required to materialize a fresh contiguous output:
 * dtype, ndims, shapes, manual_dep, and an optional initial value fill. Its
 * 64B layout mirrors Tensor cache line 1 so init_tensor_from_create_info() can
 * copy the whole line with a single memcpy.
 */

#pragma once

#include <cstring>
#include <memory.h>
#include <stdint.h>

#include "data_type.h"
#include "tensor.h"

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
        // Bound the write below: shapes[] holds MAX_TENSOR_DIMS, and ndims_in
        // comes from user-submitted output shapes — guard before the loop so an
        // oversized rank can't overrun the fixed array.
        always_assert(ndims_in > 0 && ndims_in <= MAX_TENSOR_DIMS);
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
    bool is_contiguous;                // Always true for create-info outputs
    uint8_t __pad_flags__;             // → Tensor::child_memory (always 0 for create-info outputs)
    uint32_t shapes[MAX_TENSOR_DIMS];  // → Tensor::shapes

    TensorCreateInfo() = default;

    friend struct Arg;
};

// TensorCreateInfo layout must match Tensor cacheline 1 for memcpy optimization
static_assert(sizeof(TensorCreateInfo) == 64, "TensorCreateInfo must match Tensor cacheline 1 size (64 bytes)");
static_assert(offsetof(TensorCreateInfo, start_offset) == offsetof(Tensor, start_offset));
static_assert(offsetof(TensorCreateInfo, version) == offsetof(Tensor, version));
static_assert(offsetof(TensorCreateInfo, ndims) == offsetof(Tensor, ndims));
static_assert(offsetof(TensorCreateInfo, dtype) == offsetof(Tensor, dtype));
static_assert(offsetof(TensorCreateInfo, manual_dep) == offsetof(Tensor, manual_dep));
static_assert(offsetof(TensorCreateInfo, is_contiguous) == offsetof(Tensor, is_contiguous));
static_assert(offsetof(TensorCreateInfo, __pad_flags__) == offsetof(Tensor, child_memory));
static_assert(offsetof(TensorCreateInfo, shapes) == offsetof(Tensor, shapes));

// ============================================================================
// Materialization helpers — operate on a Tensor& through its public members.
// Factored out of Tensor (which now lives in the wire/host-facing common
// header) so the create-info dependency stays runtime-only.
// ============================================================================

/// Fill the entire backing buffer of `t` with `initial_value` (doubling memcpy).
inline void fill_tensor_initial_value(Tensor &t, uint64_t initial_value) {
    always_assert(reinterpret_cast<char *>(t.buffer.addr) != nullptr);
    uint64_t elem_size = get_element_size(t.dtype);
    char *dst = reinterpret_cast<char *>(t.buffer.addr);
    constexpr uint64_t blk_size = 64;
    uint64_t blk = (t.buffer.size < blk_size) ? t.buffer.size : blk_size;
    for (uint64_t b = 0; b < blk; b += elem_size) {
        memcpy(dst + b, &initial_value, elem_size);
    }
    uint64_t filled = blk;
    while (filled < t.buffer.size) {
        uint64_t copy_size = ((t.buffer.size - filled) < filled) ? (t.buffer.size - filled) : filled;
        memcpy(dst + filled, dst, copy_size);
        filled += copy_size;
    }
}

/// Materialize a TensorCreateInfo into `t` (fresh contiguous output).
/// Single 64B memcpy covers cache line 1; `ci` pre-initialises start_offset (=0)
/// and is_contiguous (=true) in its line-1 slots so they need no reset here.
/// Cache line 2 (stride/extent) is computed from `ci.shapes` in a single reverse pass.
inline void init_tensor_from_create_info(Tensor &t, const TensorCreateInfo &ci, void *addr, uint64_t buffer_size) {
    always_assert(ci.ndims > 0 && ci.ndims <= MAX_TENSOR_DIMS);
    memcpy(&t, &ci, 64);
    t.buffer = {reinterpret_cast<uint64_t>(addr), buffer_size};
    t.owner_task_id = PTO2TaskId::invalid();  // caller (orchestrator) overwrites with actual task_id
    uint32_t s = 1;
    for (int32_t i = static_cast<int32_t>(t.ndims) - 1; i >= 0; --i) {
        t.strides[i] = s;
        s *= t.shapes[i];
    }
    t.extent_elem_cache = s;
    if (ci.has_initial_value) {
        fill_tensor_initial_value(t, ci.initial_value);
    }
}
