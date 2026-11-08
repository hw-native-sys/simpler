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

#include "async_observer.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_async_check.h"

struct AsyncObservationBarrier {
    inline __aicore__ void finish() { scheduler_cache_barrier(); }
};

template <typename Reader = AsyncCompletionScalarReader, typename Cache = AsyncObservationBarrier>
class AsyncCompletionBackend {
public:
    inline __aicore__ explicit AsyncCompletionBackend(Reader reader = {}, Cache cache = {}) :
        reader_(reader),
        cache_(cache) {}

    inline __aicore__ SchedulerAsyncConditionOutcome
    observe_once(const SchedulerAsyncGroupIdentity &, uint32_t, const DeferredCompletionEntry &entry) {
        const auto result = async_observe_once(entry, reader_);
        if (result.state == CompletionPollState::FAILED)
            return {SchedulerAsyncConditionState::FAILED, result.error_code};
        if (result.state == CompletionPollState::READY) cache_.finish();
        return {
            result.state == CompletionPollState::READY ? SchedulerAsyncConditionState::READY :
                                                         SchedulerAsyncConditionState::PENDING,
            0
        };
    }

private:
    Reader reader_;
    Cache cache_;
};

inline __aicore__ bool async_sdma_guard_address(const AsyncCtx &ctx, uint64_t *guard_address) {
    if (guard_address == nullptr) return false;
    *guard_address = 0;
    if (ctx.execution_context == nullptr) return true;
    const auto &execution = *ctx.execution_context;
    if (execution.magic != ASYNC_EXECUTION_CONTEXT_MAGIC || execution.region_address == 0 ||
        execution.region_address % 128 != 0 || execution.subtask_slot >= 3)
        return false;
    const auto *layout = reinterpret_cast<__gm__ const SchedulerAsyncLayout *>(execution.region_address);
    return layout->version == SCHEDULER_ASYNC_LAYOUT_VERSION && layout->run_epoch != 0 &&
           layout->total_bytes >= sizeof(*layout) && layout->total_bytes <= UINT64_MAX - execution.region_address &&
           execution.task_ref < layout->task_count &&
           async_sdma_guard_region(
               execution.region_address, layout->total_bytes, layout->run_epoch, layout->sdma_guard_context_offset,
               layout->sdma_guard_count, guard_address
           );
}
