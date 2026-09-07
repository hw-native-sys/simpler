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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "host_build_graph/task_id.h"
#include "task_interface/buffer_handle.h"
#include "task_interface/data_type.h"

namespace simpler::hbg {

/**
 * TensorData — every field a Tensor has (96B), and the form one travels in.
 *
 * A Definition image is built in a byte vector, which has nowhere to put a
 * 64-byte-aligned element, so the image stores this instead: the same fields in the
 * same types, packed to 8-byte alignment. Tensor derives from it, so one field set
 * serves both forms and a tensor moves between them by base-class assignment.
 *
 * The field order is the aligned form's, not the packing-optimal one: the hot set
 * (buffer, owner, origin, version, flags, shapes) ends exactly at offset 64, so a
 * Tensor's first cache line holds all of it and the warm view metadata sits past it.
 *
 * A Tensor array is NOT a TensorData array: the element strides are 128 and 96.
 * Convert one element at a time; never cast the array base and index it.
 */
struct TensorData {
    // === Cache line 1 (64B) — hot path ===
    PTOBufferHandle buffer;  // Underlying memory buffer (addr in bytes, size in bytes)
    // Creator task; TaskId::invalid() for external tensors. On a tensor a modular task's
    // body recorded, this says where the tensor CAME FROM rather than which run-time task
    // owns it: the recorder stamps PARAM with a boundary parameter's index, or SUB_TASK
    // with the producing block's index within that body. A replay rebinds the tensor
    // without reminting the field (graph_rebind_tensor), so a materialized tensor still
    // carries the recording's value — nothing on the device reads it back.
    TaskId owner_task_id;
    uint64_t start_offset;             // 1D ELEMENT offset of the view origin into `buffer`
    int32_t version;                   // Tensor version for overlap detection
    uint8_t ndims;                     // Number of dimensions used; MAX_TENSOR_DIMS bounds it
    DataType dtype;                    // Data type of tensor elements
    bool manual_dep;                   // True when dependency tracking is creator-only (skip OverlapMap lookup/insert)
    bool is_contiguous;                // Cached row-major contiguity, ignoring singleton strides
    AddressSpace address_space;        // Physical location after argument binding
    uint32_t shapes[MAX_TENSOR_DIMS];  // Current view shape per dimension (elements)

    // === Cache line 2 (64B) — warm path (view metadata) ===
    // Field order: place the 8B-aligned cache before the 4B-aligned strides[]
    // to avoid 4B padding between them.
    uint64_t extent_elem_cache;         // Cached extent_elem (see extent_elem()); maintained by ops
    uint32_t strides[MAX_TENSOR_DIMS];  // Element stride per dimension; ALWAYS > 0 (type-enforced)

    /// Deep copy. `strides` / `extent_elem_cache` are derived from the geometry,
    /// so a canonically contiguous source (is_contiguous && start_offset == 0)
    /// has them recomputed from `shapes` rather than read across — which leaves the
    /// second cache line untouched entirely for the common case.
    ///
    /// This is also how a tensor crosses between its working and image forms: both are
    /// TensorData, so neither direction needs a conversion of its own.
    void init_from(const TensorData &other) {
        init_geometry_from(other);
        if (other.is_contiguous && other.start_offset == 0) {
            refresh_row_major_derived();
        } else {
            extent_elem_cache = other.extent_elem_cache;
            for (uint32_t i = 0; i < other.ndims; i++) {
                strides[i] = other.strides[i];
            }
        }
    }

    /// Copy everything except the derived `strides` / `extent_elem_cache`. A view
    /// op then mutates `shapes` / `start_offset` and calls `refresh_derived()`
    /// once, so writing the derived pair here would only be overwritten.
    ///
    /// Only the `ndims` slots in use are copied. A slot past ndims describes no axis and
    /// every reader bounds its loop by ndims; the one op that raises ndims (reshape)
    /// writes the slots it adds before anything can read them.
    void init_geometry_from(const TensorData &other) {
        buffer = other.buffer;
        owner_task_id = other.owner_task_id;
        start_offset = other.start_offset;
        version = other.version;
        ndims = other.ndims;
        dtype = other.dtype;
        manual_dep = other.manual_dep;
        is_contiguous = other.is_contiguous;
        address_space = other.address_space;
        for (uint32_t i = 0; i < other.ndims; i++) {
            shapes[i] = other.shapes[i];
        }
    }

    /// Set `strides` to the row-major stride of `shapes` and `extent_elem_cache`
    /// to the resulting element count — the derived pair of a contiguous view.
    void refresh_row_major_derived() {
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(ndims) - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shapes[i];
        }
        extent_elem_cache = s;
    }
};

static_assert(
    std::is_trivially_copyable_v<TensorData> && std::is_standard_layout_v<TensorData>,
    "a Definition image holds these as raw bytes"
);
static_assert(sizeof(TensorData) == 96, "the wire layout is fixed: host and device read the same bytes");
static_assert(MAX_TENSOR_DIMS <= UINT8_MAX, "a tensor's rank must fit the byte that records it");
static_assert(offsetof(TensorData, extent_elem_cache) == 64, "the hot set must end at the first cache line");

}  // namespace simpler::hbg
