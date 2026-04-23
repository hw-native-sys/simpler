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

#ifndef PTO_ASYNC_KERNEL_API_H
#define PTO_ASYNC_KERNEL_API_H

#include <stdint.h>

#include "intrinsic.h"
#include "pto_completion_ingress.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

struct PTO2AsyncCtx {
    volatile __gm__ PTO2DeferredCompletionEntry *entries;
    volatile __gm__ uint32_t *entry_count;
    uint32_t entry_capacity;
    PTO2TaskId task_token;
};

inline __aicore__ PTO2AsyncCtx pto2_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    PTO2AsyncCtx ctx;
    ctx.entries = lc->deferred_completion_entries;
    ctx.entry_count = lc->deferred_completion_count;
    ctx.entry_capacity = lc->deferred_completion_capacity;
    ctx.task_token.raw = lc->task_token.raw;
    return ctx;
}

inline __aicore__ bool pto2_async_ctx_is_deferred(const PTO2AsyncCtx &ctx) { return ctx.task_token.is_valid(); }

inline __aicore__ void pto2_defer_counter(PTO2AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected) {
    if (!ctx.task_token.is_valid() || ctx.entries == nullptr || ctx.entry_count == nullptr) {
        return;
    }

    uint32_t idx = *ctx.entry_count;
    if (idx >= ctx.entry_capacity) return;

    volatile __gm__ PTO2DeferredCompletionEntry *slot = &ctx.entries[idx];
    slot->addr = reinterpret_cast<uint64_t>(counter_addr);
    slot->expected_value = expected;
    slot->engine = PTO2_COMPLETION_ENGINE_SDMA;
    slot->completion_type = PTO2_COMPLETION_TYPE_COUNTER;
    slot->_pad = 0;
    *ctx.entry_count = idx + 1;
}

inline __aicore__ void pto2_defer_flush(PTO2AsyncCtx &ctx) {
    if (!ctx.task_token.is_valid() || ctx.entries == nullptr || ctx.entry_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    dcci((__gm__ int32_t *)ctx.entries, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    dcci((__gm__ int32_t *)ctx.entry_count, SINGLE_CACHE_LINE, CACHELINE_OUT);
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
#else
    (void)ctx;
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

#endif  // PTO_ASYNC_KERNEL_API_H
