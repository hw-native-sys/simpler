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

#include "common/lane_mask.h"
#include "common/platform_config.h"
#include "kernel_args.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

constexpr int32_t SCHEDULER_TIMEOUT_MS = PLATFORM_SCHEDULER_TIMEOUT_MS;
constexpr uint64_t SCHEDULER_TIMEOUT_CYCLES =
    static_cast<uint64_t>(SCHEDULER_TIMEOUT_MS) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);

inline constexpr uint64_t SCHEDULER_STATE_ALIGNMENT = 64;
// CANN's SIMT GM-atomic model uses 128B sectors within a 512B L2 line.
// AIC/AIV scalar atomicAdd/ld_dev probes support this separation for layout;
// it is not a scalar ordering guarantee. See the data-layout contract.
inline constexpr uint64_t SCHEDULER_GM_ATOMIC_SECTOR_BYTES = 128;
inline constexpr uint64_t SCHEDULER_WORKER_CAPACITY = 108;

inline constexpr uint32_t SCHEDULER_CLUSTER_CAPACITY = SCHEDULER_WORKER_CAPACITY / 3;

inline constexpr uint32_t SCHEDULER_SCALAR_RECORD_CAPACITY = 262144;

struct SchedulerScalarRecord {
    uint64_t start_time;
    uint64_t end_time;
    int64_t task_ref;
    uint32_t kind;
    uint32_t count;
};
static_assert(sizeof(SchedulerScalarRecord) == 32);

// The Host validates submit-time shape before binding calendar rows.
struct SchedulerTaskMetadata {
    uint16_t kernel_ids[3];
    LaneMask active_mask;
    bool inline_task;
    uint16_t logical_block_num;
};

inline __aicore__ bool
scheduler_subtask_count_checked(uint64_t logical_block_num, uint32_t active_subtasks, uint32_t *out) {
    if (out == nullptr || logical_block_num == 0 || active_subtasks == 0 || active_subtasks > 3 ||
        logical_block_num > static_cast<uint64_t>(INT16_MAX) / active_subtasks)
        return false;
    *out = static_cast<uint32_t>(logical_block_num * active_subtasks);
    return true;
}

template <typename T>
inline __aicore__ __gm__ T *scheduler_state_at(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

inline bool scheduler_layout_checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool scheduler_layout_checked_mul(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool scheduler_layout_checked_align(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!scheduler_layout_checked_add(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool scheduler_layout_reserve(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !scheduler_layout_checked_align(*cursor, alignment, &aligned))
        return false;
    uint64_t end = 0;
    if (!scheduler_layout_checked_add(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}
