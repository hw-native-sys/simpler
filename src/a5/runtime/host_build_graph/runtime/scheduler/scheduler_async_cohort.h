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

#include "scheduler_async_layout.h"

inline constexpr uint64_t SCHEDULER_ASYNC_COHORT_VERSION = 1;

// The scratch prefix is scheduler-owned; auxiliary backend scratch follows it.
// Maps are immutable and counters occupy separate atomic-only cache lines.
struct alignas(128) SchedulerAsyncCohortHeader {
    uint64_t version{SCHEDULER_ASYNC_COHORT_VERSION};
    uint64_t total_bytes{0};
    uint64_t run_epoch{0};
    uint64_t task_count{0};
    uint64_t cohort_count{0};
    uint64_t maps_offset{0};
    uint64_t counters_offset{0};
    uint64_t auxiliary_offset{0};
    uint64_t auxiliary_bytes{0};
    uint64_t reserved[7]{};
};

struct SchedulerAsyncCohortTaskMap {
    uint64_t cohort_base{SCHEDULER_ASYNC_INDEX_NONE};
    uint32_t block_count{0};
    uint32_t lane_mask{0};
    uint32_t expected{0};
    uint32_t cohort_count{0};
    uint32_t flags{0};
    uint32_t reserved{0};
};

struct alignas(128) SchedulerAsyncCohortCounter {
    volatile uint64_t ready{0};
    uint8_t reserved[120]{};
};

static_assert(sizeof(SchedulerAsyncCohortHeader) == 128 && sizeof(SchedulerAsyncCohortTaskMap) == 32);
static_assert(sizeof(SchedulerAsyncCohortCounter) == 128);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncCohortHeader> && std::is_trivially_copyable_v<SchedulerAsyncCohortHeader>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncCohortTaskMap> && std::is_trivially_copyable_v<SchedulerAsyncCohortTaskMap>
);
static_assert(
    std::is_standard_layout_v<SchedulerAsyncCohortCounter> && std::is_trivially_copyable_v<SchedulerAsyncCohortCounter>
);

namespace scheduler_async_cohort_detail {
template <typename T>
inline __aicore__ T load(__gm__ const T *source) {
    static_assert(sizeof(T) % 8 == 0 && alignof(T) >= 8 && std::is_trivially_copyable_v<T>);
    T value{};
    for (uint32_t offset = 0; offset < sizeof(T); offset += 8) {
        uint64_t word = 0;
#if defined(__CCE_AICORE__)
        auto *address = reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ uint8_t *>(reinterpret_cast<__gm__ const uint8_t *>(source) + offset)
        );
        word = static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
#else
        __builtin_memcpy(&word, reinterpret_cast<const uint8_t *>(source) + offset, 8);
#endif
        __builtin_memcpy(reinterpret_cast<uint8_t *>(&value) + offset, &word, 8);
    }
    return value;
}
inline __aicore__ bool region(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t *end) {
    if (count == 0) return offset == 0;
    if (offset < *end || offset % 128 != 0 || offset > bytes || count > (bytes - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}
}  // namespace scheduler_async_cohort_detail

struct SchedulerAsyncCohortView {
    __gm__ uint8_t *base{nullptr};
    SchedulerAsyncCohortHeader header{};

    static inline __aicore__ bool
    mount(__gm__ const uint8_t *async_base, const SchedulerAsyncLayout &layout, SchedulerAsyncCohortView *out) {
        using namespace scheduler_async_cohort_detail;
        if (async_base == nullptr || out == nullptr || reinterpret_cast<uint64_t>(async_base) % 128 != 0 ||
            layout.scratch_bytes < sizeof(SchedulerAsyncCohortHeader) ||
            layout.scratch_offset < sizeof(SchedulerAsyncLayout) || layout.scratch_offset % 128 != 0 ||
            layout.scratch_offset > layout.total_bytes ||
            layout.scratch_bytes > layout.total_bytes - layout.scratch_offset ||
            layout.total_bytes > UINT64_MAX - reinterpret_cast<uint64_t>(async_base))
            return false;
        SchedulerAsyncCohortView view{};
        view.base = const_cast<__gm__ uint8_t *>(async_base) + layout.scratch_offset;
        view.header = load(reinterpret_cast<__gm__ const SchedulerAsyncCohortHeader *>(view.base));
        const auto &h = view.header;
        if (h.version != SCHEDULER_ASYNC_COHORT_VERSION || h.total_bytes != layout.scratch_bytes || h.run_epoch == 0 ||
            h.run_epoch != layout.run_epoch || h.task_count != layout.task_count || h.task_count == 0 ||
            h.auxiliary_offset % 128 != 0 || h.auxiliary_offset > h.total_bytes ||
            h.auxiliary_bytes != h.total_bytes - h.auxiliary_offset)
            return false;
        for (uint64_t reserved : h.reserved)
            if (reserved != 0) return false;
        uint64_t end = sizeof(h);
        if (!region(h.auxiliary_offset, h.maps_offset, h.task_count, sizeof(SchedulerAsyncCohortTaskMap), &end) ||
            !region(h.auxiliary_offset, h.counters_offset, h.cohort_count, sizeof(SchedulerAsyncCohortCounter), &end))
            return false;
        *out = view;
        return true;
    }
};

inline __aicore__ bool scheduler_async_cohort_execution_matches(
    const SchedulerAsyncCohortView &view, int64_t task, const SchedulerTaskMetadata &metadata
) {
    if (view.base == nullptr || task < 0 || static_cast<uint64_t>(task) >= view.header.task_count) return false;
    const auto map = scheduler_async_cohort_detail::load(
        reinterpret_cast<__gm__ const SchedulerAsyncCohortTaskMap *>(view.base + view.header.maps_offset) + task
    );
    uint32_t lanes = 0;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (metadata.active_mask.has(static_cast<SubtaskSlot>(lane))) lanes |= 1u << lane;
    return map.flags == metadata.flags && map.block_count == metadata.logical_block_num && map.lane_mask == lanes;
}

// One local cursor belongs to the existing slot/cohort generation. Once arrived
// it cannot be moved, revoked, or reused after another business kernel writes.
struct SchedulerAsyncCohortCursor {
    uint64_t run_epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t dispatch_generation{0};
    uint64_t core_write_serial{0};
    uint64_t cohort_index{SCHEDULER_ASYNC_INDEX_NONE};
    uint32_t block_idx{0};
    uint32_t subtask_slot{0};
    bool arrived{false};
};

enum class SchedulerAsyncCohortResult : uint8_t { PENDING, READY, ERROR };

// A unit performs at most one atomic operation, with no retry loop. The caller
// pins this physical execution after its first contribution until kernel return.
template <typename Identity>
inline __aicore__ SchedulerAsyncCohortResult scheduler_async_cohort_prepare(
    const SchedulerAsyncCohortView &view, const Identity &identity, SchedulerAsyncCohortCursor &cursor, uint32_t &budget
) {
    using Result = SchedulerAsyncCohortResult;
    if (view.base == nullptr || identity.run_epoch != view.header.run_epoch || identity.task_ref < 0 ||
        static_cast<uint64_t>(identity.task_ref) >= view.header.task_count || identity.subtask_slot >= 3 ||
        identity.dispatch_generation == 0 || identity.core_write_serial == UINT64_MAX)
        return Result::ERROR;
    const auto map = scheduler_async_cohort_detail::load(
        reinterpret_cast<__gm__ const SchedulerAsyncCohortTaskMap *>(view.base + view.header.maps_offset) +
        identity.task_ref
    );
    const uint32_t lanes = __builtin_popcount(map.lane_mask);
    if (map.reserved != 0 || map.flags > UINT8_MAX || map.block_count == 0 ||
        map.block_count > SCHEDULER_ASYNC_JOIN_LIMIT || (map.lane_mask & ~7u) != 0 ||
        (map.lane_mask & (1u << identity.subtask_slot)) == 0 || identity.block_idx >= map.block_count ||
        !scheduler_task_is_executable(map.flags) || scheduler_task_is_inline(map.flags) ||
        scheduler_task_is_mix(map.flags) != (lanes > 1) || scheduler_task_is_spmd(map.flags) != (map.block_count > 1))
        return Result::ERROR;
    const bool sync = scheduler_task_requires_sync_start(map.flags);
    const bool mixed = lanes > 1;
    if (!sync && !mixed) {
        return map.cohort_base == SCHEDULER_ASYNC_INDEX_NONE && map.cohort_count == 0 && map.expected == 0 &&
                       !cursor.arrived ?
                   Result::READY :
                   Result::ERROR;
    }
    const uint32_t expected = sync ? map.block_count * lanes : lanes;
    if ((sync && map.block_count == 1) || expected > SCHEDULER_ASYNC_JOIN_LIMIT || map.expected != expected ||
        map.cohort_count != (sync ? 1u : map.block_count) || map.cohort_base >= view.header.cohort_count ||
        map.cohort_count > view.header.cohort_count - map.cohort_base)
        return Result::ERROR;
    const uint64_t index = map.cohort_base + (sync ? 0 : identity.block_idx);
    if (cursor.arrived && (cursor.run_epoch != identity.run_epoch || cursor.task_ref != identity.task_ref ||
                           cursor.block_idx != identity.block_idx || cursor.subtask_slot != identity.subtask_slot ||
                           cursor.dispatch_generation != identity.dispatch_generation ||
                           cursor.core_write_serial != identity.core_write_serial || cursor.cohort_index != index))
        return Result::ERROR;
    if (budget == 0) return Result::PENDING;
    --budget;
    auto &counter =
        (reinterpret_cast<__gm__ SchedulerAsyncCohortCounter *>(view.base + view.header.counters_offset) + index)
            ->ready;
    uint64_t count = 0;
    if (!cursor.arrived) {
        cursor = {
            identity.run_epoch,
            identity.task_ref,
            identity.dispatch_generation,
            identity.core_write_serial,
            index,
            identity.block_idx,
            identity.subtask_slot,
            true
        };
        scheduler_cache_barrier();
        const uint64_t previous = scheduler_gm_fetch_add(counter, UINT64_C(1));
        if (previous >= expected) return Result::ERROR;
        count = previous + 1;
    } else {
        count = scheduler_gm_fetch_add(counter, UINT64_C(0));
        if (count > expected) return Result::ERROR;
    }
    if (count != expected) return Result::PENDING;
    scheduler_cache_barrier();
    return Result::READY;
}
