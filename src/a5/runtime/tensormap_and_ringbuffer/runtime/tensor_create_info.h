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
 * part of the boundary simpler::tmr::Tensor in src/common/task_interface/tensor.h.
 * It carries the metadata required to materialize a fresh contiguous output:
 * dtype, ndims, shapes, manual_dep.
 *
 * There is no initial-value fill. That fill ran inside the hidden alloc task
 * where nothing could race it, but it bound the AICPU orchestrator to writing
 * the buffer itself; an orchestration that needs a defined starting content
 * writes it with set_tensor_data or has a task write it. See
 * docs/SCALAR_DATA_ACCESS.md.
 */

#pragma once

#include <cstring>
#include <memory.h>
#include <stdint.h>

#include "data_type.h"
#include "tensor.h"

class TensorCreateInfo {
public:
    TensorCreateInfo(
        const uint32_t shapes_in[], uint32_t ndims_in, DataType dtype_in = DataType::FLOAT32, bool manual_dep_in = false
    ) :
        ndims(ndims_in),
        dtype(dtype_in),
        manual_dep(manual_dep_in) {
        // Bound the write below: shapes[] holds MAX_TENSOR_DIMS, and ndims_in
        // comes from user-submitted output shapes — guard before the loop so an
        // oversized rank can't overrun the fixed array.
        always_assert(ndims_in > 0 && ndims_in <= MAX_TENSOR_DIMS);
        for (uint32_t i = 0; i < ndims_in; i++) {
            shapes[i] = shapes_in[i];
        }
    }

    void copy(const TensorCreateInfo &other) { *this = other; }

    uint64_t buffer_size_bytes() const {
        uint64_t total = 1;
        for (uint32_t i = 0; i < ndims; i++) {
            total *= shapes[i];
        }
        return total * get_element_size(dtype);
    }

public:
    uint32_t ndims{0};
    DataType dtype{DataType::FLOAT32};
    bool manual_dep{false};
    uint32_t shapes[MAX_TENSOR_DIMS]{};

    TensorCreateInfo() = default;
};

// ============================================================================
// Materialization helpers — operate on a simpler::tmr::Tensor& through its public members.
// Factored out of simpler::tmr::Tensor (which now lives in the wire/host-facing common
// header) so the create-info dependency stays runtime-only.
// ============================================================================

/// Materialize a TensorCreateInfo into `t` (fresh contiguous output).
inline void
init_tensor_from_create_info(simpler::tmr::Tensor &t, const TensorCreateInfo &ci, void *addr, uint64_t buffer_size) {
    always_assert(ci.ndims > 0 && ci.ndims <= MAX_TENSOR_DIMS);
    t.buffer = {reinterpret_cast<uint64_t>(addr), buffer_size};
    t.owner_task_id = TaskId::invalid();  // caller (orchestrator) overwrites with actual task_id
    t.start_offset = 0;
    t.version = 0;
    t.ndims = ci.ndims;
    t.dtype = ci.dtype;
    t.manual_dep = ci.manual_dep;
    t.is_contiguous = true;
    t.address_space = AddressSpace::DEVICE;
    for (uint32_t i = 0; i < ci.ndims; ++i) {
        t.shapes[i] = ci.shapes[i];
    }
    t.refresh_row_major_derived();
}
