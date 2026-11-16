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
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_prepared_layout.h"
#include "tensor.h"

namespace {
constexpr uint32_t kLineWords = 32;

// Independent single-block background tasks are ready while their Host-built
// execution remains unclaimed. This observes the current immutable catalog.
inline __aicore__ bool background_is_ready(__gm__ void *base, uint32_t count) {
    const auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(base);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, run->graph_summary_offset);
    const auto *prepared = scheduler_state_at<SchedulerPreparedLayout>(base, summary->prepared_layout_offset);
    if (count >= prepared->task_count) return false;
    const auto *plans = scheduler_state_at<SchedulerExecutionPlan>(base, prepared->plans_offset);
    auto *controls = scheduler_state_at<SchedulerPreparedControl>(base, prepared->controls_offset);
    for (uint32_t ref = 1; ref <= count; ++ref)
        if (scheduler_prepared_phase(scheduler_gm_fetch_add(controls[plans[ref].execution_begin].state, UINT64_C(0))) ==
            SchedulerPreparedPhase::BUILT)
            return true;
    return false;
}

// The canonical task gate is published only after all registered conditions settle.
inline __aicore__ bool wait_was_finished(__gm__ void *base) {
    const auto *run = reinterpret_cast<__gm__ SchedulerRunControl *>(base);
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, run->graph_summary_offset);
    const auto *prepared = scheduler_state_at<SchedulerPreparedLayout>(base, summary->prepared_layout_offset);
    auto *control = scheduler_state_at<SchedulerTaskControl>(base, prepared->task_controls_offset);
    return scheduler_completion_done(scheduler_gm_fetch_add(control[0].completion_state, UINT64_C(0)));
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
        if (ctx.resident_enabled == 1 && wait_was_finished(ctx.resident.image) &&
            background_is_ready(ctx.resident.image, count))
            scheduler_gm_store(out[2 * kLineWords], UINT32_C(1));
        scheduler_gm_store(out[(index + 4) * kLineWords], index + 1);
    } else if (role == 2) {
        scheduler_gm_store(out[3 * kLineWords], static_cast<uint32_t>(scheduler_gm_query_u32_pair(out)));
    }
}
