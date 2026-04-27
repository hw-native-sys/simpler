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

#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#include "intrinsic.h"
#include "pto_completion_ingress.h"
#include "pto_runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

struct PTO2AsyncCtx {
    volatile __gm__ PTO2DeferredCompletionIngressBuffer *ingress;
    uint32_t entry_capacity;
    PTO2TaskId task_token;
};

inline __aicore__ PTO2AsyncCtx pto2_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    PTO2AsyncCtx ctx;
    ctx.ingress = lc->deferred_ingress;
    ctx.entry_capacity = lc->deferred_completion_capacity;
    ctx.task_token.raw = lc->task_token.raw;
    return ctx;
}

inline __aicore__ void pto2_defer_condition(
    PTO2AsyncCtx &ctx, volatile __gm__ void *addr, uint32_t expected, uint32_t engine, int32_t completion_type
) {
    if (ctx.task_token.is_invalid() || ctx.ingress == nullptr) {
        return;
    }

    uint32_t idx = ctx.ingress->count;
    if (idx >= ctx.entry_capacity) {
        ctx.ingress->error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
        return;
    }

    volatile __gm__ PTO2DeferredCompletionEntry *slot = &ctx.ingress->entries[idx];
    slot->addr = reinterpret_cast<uint64_t>(addr);
    slot->expected_value = expected;
    slot->engine = engine;
    slot->completion_type = completion_type;
    slot->_pad = 0;
    ctx.ingress->count = idx + 1;
}

inline __aicore__ void pto2_defer_flush(PTO2AsyncCtx &ctx) {
    if (ctx.task_token.is_invalid() || ctx.ingress == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    dcci((__gm__ int32_t *)ctx.ingress->entries, ENTIRE_DATA_CACHE, CACHELINE_OUT);
    dcci((__gm__ int32_t *)ctx.ingress, SINGLE_CACHE_LINE, CACHELINE_OUT);
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
#else
    (void)ctx;
    __asm__ __volatile__("" ::: "memory");
#endif
}

inline __aicore__ void
pto2_send_notification(volatile __gm__ void *remote_counter_addr, int32_t value, pto::comm::NotifyOp notify_op) {
    __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ void *>(remote_counter_addr));
    pto::comm::Signal signal(counter);
    pto::comm::TNOTIFY(signal, value, notify_op);
}

inline __aicore__ void pto2_save_expected_notification_counter(
    __gm__ int64_t *args, volatile __gm__ void *counter_addr, uint32_t expected_value
) {
    PTO2AsyncCtx ctx = pto2_async_ctx(args);
    pto2_defer_condition(ctx, counter_addr, expected_value, PTO2_COMPLETION_ENGINE_SDMA, PTO2_COMPLETION_TYPE_COUNTER);
    pto2_defer_flush(ctx);
}

#endif  // PTO_ASYNC_KERNEL_API_H
