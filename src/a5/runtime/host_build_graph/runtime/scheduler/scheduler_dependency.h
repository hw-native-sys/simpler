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

#include "scheduler_memory.h"

// All references are relative to the scheduler allocation. Remaining counts
// occupy atomic-only cache lines; the CSR arrays are immutable for the run.
struct alignas(64) SchedulerDependencyHeader {
    uint64_t version{2};
    uint64_t task_count{0};
    uint64_t edge_capacity{0};
    uint64_t edge_count{0};
    uint64_t remaining_offset{0};
    uint64_t fanout_offsets_offset{0};
    uint64_t fanout_refs_offset{0};
    uint64_t storage_end_offset{0};
};

static_assert(sizeof(SchedulerDependencyHeader) == 64);
static_assert(
    std::is_standard_layout_v<SchedulerDependencyHeader> && std::is_trivially_copyable_v<SchedulerDependencyHeader>
);

inline __aicore__ bool
scheduler_dependency_validate(__gm__ void *base, uint64_t state_bytes, uint64_t offset, uint64_t task_count) {
    if (base == nullptr || offset == 0 || offset % SCHEDULER_STATE_ALIGNMENT != 0 || offset > state_bytes ||
        sizeof(SchedulerDependencyHeader) > state_bytes - offset || task_count > INT32_MAX)
        return false;
    auto *header = scheduler_state_at<SchedulerDependencyHeader>(base, offset);
    for (uint64_t byte = 0; byte < sizeof(*header); byte += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(header) + byte);
    if (header->version != 2 || header->task_count != task_count || header->edge_count > header->edge_capacity ||
        header->storage_end_offset > state_bytes)
        return false;
    uint64_t cursor = offset + sizeof(*header);
    const uint64_t offsets[] = {header->remaining_offset, header->fanout_offsets_offset, header->fanout_refs_offset};
    const uint64_t counts[] = {task_count, task_count + 1, header->edge_capacity};
    const uint64_t sizes[] = {sizeof(uint64_t), sizeof(uint64_t), sizeof(int32_t)};
    for (uint32_t i = 0; i < 3; ++i) {
        if (offsets[i] < cursor || offsets[i] % SCHEDULER_STATE_ALIGNMENT != 0 ||
            offsets[i] > header->storage_end_offset || counts[i] > (header->storage_end_offset - offsets[i]) / sizes[i])
            return false;
        cursor = offsets[i] + counts[i] * sizes[i];
    }
    return true;
}

inline __aicore__ bool scheduler_dependency_successor(
    __gm__ void *base, const __gm__ SchedulerDependencyHeader &header, int64_t producer, uint32_t index,
    int64_t *successor
) {
    if (producer < 0 || static_cast<uint64_t>(producer) >= header.task_count || successor == nullptr) return false;
    const auto *offsets = scheduler_state_at<uint64_t>(base, header.fanout_offsets_offset);
    const uint64_t begin = offsets[producer], end = offsets[producer + 1];
    if (begin > end || end > header.edge_count || end - begin > header.task_count || index > end - begin) return false;
    if (index == end - begin) {
        *successor = SCHEDULER_WAKE_LIST_OPEN;
        return true;
    }
    const int32_t ref = scheduler_state_at<int32_t>(base, header.fanout_refs_offset)[begin + index];
    if (ref <= producer || static_cast<uint64_t>(ref) >= header.task_count) return false;
    *successor = ref;
    return true;
}

enum class SchedulerDependencyResult : uint8_t { WAITING, READY, ERROR };

inline __aicore__ SchedulerDependencyResult
scheduler_dependency_notify(__gm__ void *base, const __gm__ SchedulerDependencyHeader &header, int64_t successor) {
    if (successor < 0 || static_cast<uint64_t>(successor) >= header.task_count) return SchedulerDependencyResult::ERROR;
    auto &remaining = scheduler_state_at<uint64_t>(base, header.remaining_offset)[successor];
    const uint64_t previous = scheduler_gm_fetch_add(remaining, UINT64_MAX);
    if (previous == 0 || previous > static_cast<uint64_t>(successor)) return SchedulerDependencyResult::ERROR;
    return previous == 1 ? SchedulerDependencyResult::READY : SchedulerDependencyResult::WAITING;
}
