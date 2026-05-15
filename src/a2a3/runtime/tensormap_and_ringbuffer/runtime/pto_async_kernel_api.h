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
#include "pto_completion_token.h"
#include "pto_runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

inline __aicore__ void defer_load_ingress(AsyncCtx &ctx) {
    if (ctx.completion_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t line = reinterpret_cast<uintptr_t>(ctx.completion_count) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
    dcci((__gm__ int32_t *)line, SINGLE_CACHE_LINE);
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx{};
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    ctx.task_token.raw = lc->async_ctx.task_token.raw;
    defer_load_ingress(ctx);
    return ctx;
}

inline __aicore__ bool async_ctx_is_deferred(const AsyncCtx &ctx) { return ctx.task_token.is_valid(); }

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) {
    if (ctx.task_token.is_valid() && ctx.completion_error_code != nullptr) {
        *ctx.completion_error_code = error_code;
    }
}

// Canonical writer: backend submit handlers build a CompletionToken and pass
// it here. Writes one DeferredCompletionEntry to the AsyncCtx ingress slab and
// bumps completion_count. Returns false on overflow (also stores
// PTO2_ERROR_ASYNC_WAIT_OVERFLOW in ctx.completion_error_code) or when ctx is
// not currently a deferred context.
inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        return false;
    }

    uint32_t idx = *ctx.completion_count;
    if (idx >= ctx.completion_capacity) {
        if (ctx.completion_error_code != nullptr) {
            *ctx.completion_error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
        }
        return false;
    }

    volatile __gm__ DeferredCompletionEntry *slot = &ctx.completion_entries[idx];
    slot->addr = token.addr;
    slot->expected_value = token.expected_value;
    slot->engine = token.engine;
    slot->completion_type = token.completion_type;
    slot->_pad = 0;
    *ctx.completion_count = idx + 1;
    return true;
}

inline __aicore__ void
defer_condition(AsyncCtx &ctx, uint64_t addr, uint32_t expected, uint32_t engine, int32_t completion_type) {
    CompletionToken token{addr, expected, engine, completion_type, 0};
    (void)register_completion_condition(ctx, token);
}

inline __aicore__ void defer_condition(
    AsyncCtx &ctx, volatile __gm__ void *addr, uint32_t expected, uint32_t engine, int32_t completion_type
) {
    defer_condition(ctx, reinterpret_cast<uint64_t>(addr), expected, engine, completion_type);
}

inline __aicore__ void defer_counter(AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected) {
    defer_condition(
        ctx, reinterpret_cast<uint64_t>(counter_addr), expected, COMPLETION_ENGINE_SDMA,
        COMPLETION_TYPE_COUNTER
    );
}

inline __aicore__ void defer_sdma_event_record(AsyncCtx &ctx, volatile __gm__ void *record_addr) {
    defer_condition(
        ctx, reinterpret_cast<uint64_t>(record_addr), 0, COMPLETION_ENGINE_SDMA,
        COMPLETION_TYPE_SDMA_EVENT_RECORD
    );
}

#if defined(PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP)
template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void
defer_pto_async_event(AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        (void)event.Wait(session);
        return;
    }
    if (event.handle == 0) {
        return;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::SDMA)) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }

    ::pto::comm::sdma::detail::UbTmpBuf tmp_buf;
    uint32_t sync_id = 0;
    __gm__ uint8_t *recv_workspace = nullptr;
    uint32_t queue_num = 0;
    if (!::pto::comm::sdma::detail::PrepareEventCheck(
            session.sdmaSession, tmp_buf, sync_id, recv_workspace, queue_num
        )) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    for (uint32_t queue_id = 0; queue_id < queue_num; ++queue_id) {
        defer_sdma_event_record(ctx, ::pto::comm::sdma::detail::GetEventRecord(recv_workspace, queue_id));
    }
}
#else
template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void defer_pto_async_event(AsyncCtx &ctx, const PtoAsyncEvent &, const PtoAsyncSession &) {
    defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
}
#endif

inline __aicore__ void defer_flush_range(volatile __gm__ void *addr, uint32_t size_bytes) {
    if (addr == nullptr || size_bytes == 0) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
    uintptr_t end =
        (reinterpret_cast<uintptr_t>(addr) + size_bytes + PTO2_ALIGN_SIZE - 1u) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
    for (uintptr_t p = start; p < end; p += PTO2_ALIGN_SIZE) {
        dcci((__gm__ int32_t *)p, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#else
    (void)addr;
    (void)size_bytes;
#endif
}

inline __aicore__ void defer_flush(AsyncCtx &ctx) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uint32_t count = *ctx.completion_count;
    if (count > ctx.completion_capacity) {
        count = ctx.completion_capacity;
    }
    uint32_t flush_bytes = static_cast<uint32_t>(sizeof(*ctx.completion_count));
    if (ctx.completion_error_code != nullptr) {
        flush_bytes += static_cast<uint32_t>(sizeof(*ctx.completion_error_code));
    }
    if (ctx.completion_entries != nullptr) {
        flush_bytes += count * static_cast<uint32_t>(sizeof(DeferredCompletionEntry));
    }
    defer_flush_range(ctx.completion_count, flush_bytes);
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
send_notification(volatile __gm__ void *remote_counter_addr, int32_t value, pto::comm::NotifyOp notify_op) {
    __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ void *>(remote_counter_addr));
    pto::comm::Signal signal(counter);
    pto::comm::TNOTIFY(signal, value, notify_op);
}

inline __aicore__ void
save_expected_notification_counter(AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected_value) {
    defer_condition(ctx, counter_addr, expected_value, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER);
    defer_flush(ctx);
}

#endif  // PTO_ASYNC_KERNEL_API_H
