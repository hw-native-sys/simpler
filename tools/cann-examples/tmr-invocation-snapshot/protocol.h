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
#include <cstdint>
#include <type_traits>
#include "task_interface/kernel_invocation_validation.h"

// Test fixtures, published by a separate init task before any invocation.
struct SnapshotProbeInit {
    uint64_t results_addr;
    uint64_t context_generation;
    uint64_t gate_addr;
    uint32_t capacity;
};
struct SnapshotProbeResult {
    uint64_t sum;
    int32_t callable_id;
    int32_t status;
};
static_assert(std::is_trivially_copyable_v<SnapshotProbeInit> && std::is_standard_layout_v<SnapshotProbeInit>);
static_assert(std::is_trivially_copyable_v<SnapshotProbeResult> && std::is_standard_layout_v<SnapshotProbeResult>);

inline simpler::kernel::PreparedInvocationView snapshot_probe_callable(int32_t id) {
    switch (id) {
    case 0:
        return {0, 0, 0, 1};
    case 1:
        return {1, 1, 1, 2};
    case 2:
        return {2, CHIP_MAX_TENSOR_ARGS, 0, 3};
    default:
        return {-1, 0, 0, 0};
    }
}
