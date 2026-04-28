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

inline __aicore__ bool pto2_async_ctx_is_deferred(const PTO2AsyncCtx &ctx) { return ctx.task_token.is_valid(); }

inline __aicore__ void pto2_defer_error(PTO2AsyncCtx &ctx, int32_t error_code) {
    if (ctx.task_token.is_valid() && ctx.ingress != nullptr) {
        ctx.ingress->error_code = error_code;
    }
}

inline __aicore__ void
pto2_defer_completion(PTO2AsyncCtx &ctx, uint64_t addr, uint32_t expected, uint32_t engine, int32_t completion_type) {
    if (!ctx.task_token.is_valid() || ctx.ingress == nullptr) {
        return;
    }

    uint32_t idx = ctx.ingress->count;
    if (idx >= ctx.entry_capacity) {
        ctx.ingress->error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
        return;
    }

    volatile __gm__ PTO2DeferredCompletionEntry *slot = &ctx.ingress->entries[idx];
    slot->addr = addr;
    slot->expected_value = expected;
    slot->engine = engine;
    slot->completion_type = completion_type;
    slot->_pad = 0;
    ctx.ingress->count = idx + 1;
}

inline __aicore__ void pto2_defer_counter(PTO2AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected) {
    pto2_defer_completion(
        ctx, reinterpret_cast<uint64_t>(counter_addr), expected, PTO2_COMPLETION_ENGINE_SDMA,
        PTO2_COMPLETION_TYPE_COUNTER
    );
}

inline __aicore__ void pto2_defer_sdma_event_record(PTO2AsyncCtx &ctx, volatile __gm__ void *record_addr) {
    pto2_defer_completion(
        ctx, reinterpret_cast<uint64_t>(record_addr), 0, PTO2_COMPLETION_ENGINE_SDMA,
        PTO2_COMPLETION_TYPE_SDMA_EVENT_RECORD
    );
}

#if defined(PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP)
template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void pto2_defer_pto_async_event(PTO2AsyncCtx &ctx, const PtoAsyncEvent &event,
                                                  const PtoAsyncSession &session) {
    if (!ctx.task_token.is_valid() || ctx.ingress == nullptr) {
        (void)event.Wait(session);
        return;
    }
    if (event.handle == 0) {
        return;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::SDMA)) {
        pto2_defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }

    ::pto::comm::sdma::detail::UbTmpBuf tmp_buf;
    uint32_t sync_id = 0;
    __gm__ uint8_t *recv_workspace = nullptr;
    uint32_t queue_num = 0;
    if (!::pto::comm::sdma::detail::PrepareEventCheck(
            session.sdmaSession, tmp_buf, sync_id, recv_workspace, queue_num
        )) {
        pto2_defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    for (uint32_t queue_id = 0; queue_id < queue_num; ++queue_id) {
        pto2_defer_sdma_event_record(ctx, ::pto::comm::sdma::detail::GetEventRecord(recv_workspace, queue_id));
    }
}
#else
template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void pto2_defer_pto_async_event(PTO2AsyncCtx &ctx, const PtoAsyncEvent &,
                                                  const PtoAsyncSession &) {
    pto2_defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}
#endif

inline __aicore__ void pto2_defer_flush(PTO2AsyncCtx &ctx) {
    if (!ctx.task_token.is_valid() || ctx.ingress == nullptr) return;
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

#endif  // PTO_ASYNC_KERNEL_API_H
