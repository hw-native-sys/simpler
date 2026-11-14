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
#include <cstdint>
#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_task_catalog.h"
#include "tensor.h"

namespace {
constexpr uint32_t kLineWords = 32;

// Bootstrap publishes each background task once. A strong READY observation
// after CHECK witnesses a backlog without depending on a shared queue head.
inline __aicore__ bool background_is_ready(__gm__ void *base, uint32_t count) {
    const auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(base);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, run->graph_summary_offset);
    const auto *pool = scheduler_state_at<SchedulerTaskCatalogHeader>(base, summary->task_catalog_offset);
    if (pool->version != SCHEDULER_TASK_CATALOG_VERSION || count >= pool->task_count) return false;
    auto *states = scheduler_state_at<SchedulerTaskAvailability>(base, pool->availability_offset);
    for (uint32_t ref = 1; ref <= count; ++ref)
        if (scheduler_gm_fetch_add(states[ref].phase, UINT64_C(0)) ==
            static_cast<uint64_t>(SchedulerAvailabilityPhase::READY))
            return true;
    return false;
}

// Only the waiter's FINISH can publish this consumer. Its monotonic phase
// retains that fact after a worker has claimed it.
inline __aicore__ bool consumer_was_published(__gm__ void *base, uint32_t count) {
    const auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(base);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, run->graph_summary_offset);
    const auto *pool = scheduler_state_at<SchedulerTaskCatalogHeader>(base, summary->task_catalog_offset);
    if (pool->version != SCHEDULER_TASK_CATALOG_VERSION || static_cast<uint64_t>(count) + 1 >= pool->task_count)
        return false;
    auto *states = scheduler_state_at<SchedulerTaskAvailability>(base, pool->availability_offset);
    const uint64_t phase = scheduler_gm_fetch_add(states[count + 1].phase, UINT64_C(0));
    return phase == static_cast<uint64_t>(SchedulerAvailabilityPhase::READY) ||
           phase == static_cast<uint64_t>(SchedulerAvailabilityPhase::OWNED);
}

inline __aicore__ bool wait_was_checked(__gm__ void *base) {
    const auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(base);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, run->graph_summary_offset);
    const auto *layout = scheduler_state_at<SchedulerAsyncLayout>(base, summary->async_layout_offset);
    const auto *map = scheduler_state_at<uint32_t>(base, layout->task_map_offset);
    if (map[0] == pto2::resident::ASYNC_INVALID_INDEX || map[0] >= layout->async_task_count) return false;
    auto *join = scheduler_state_at<SchedulerAsyncTaskJoin>(base, layout->joins_offset) + map[0];
    return scheduler_async_pending::ready_to_finish(scheduler_gm_fetch_add(join->word, UINT64_C(0)));
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *out = reinterpret_cast<__gm__ uint32_t *>(tensor->buffer.addr) + tensor->start_offset;
    const uint32_t role = static_cast<uint32_t>(args[1]);
    const uint32_t index = static_cast<uint32_t>(args[2]);
    const uint32_t count = static_cast<uint32_t>(args[3]);
    auto ctx = get_async_ctx(args);
    if (role == 0) {
        save_expected_notification_counter(ctx, reinterpret_cast<__gm__ int32_t *>(out), 1);
        scheduler_gm_store(out[kLineWords], UINT32_C(1));
    } else if (role == 1) {
        // Every invocation does bounded work and returns. Publishing from any
        // background task avoids blocking on which core executes the waiter.
        if (scheduler_gm_query_u32_pair(out + kLineWords) == 1) scheduler_gm_store(out[0], UINT32_C(1));
        if (ctx.resident_enabled == 1 && wait_was_checked(ctx.resident.image) &&
            consumer_was_published(ctx.resident.image, count) && background_is_ready(ctx.resident.image, count))
            scheduler_gm_store(out[2 * kLineWords], UINT32_C(1));
        scheduler_gm_store(out[(index + 4) * kLineWords], index + 1);
    } else if (role == 2) {
        scheduler_gm_store(out[3 * kLineWords], static_cast<uint32_t>(scheduler_gm_query_u32_pair(out)));
    }
}
