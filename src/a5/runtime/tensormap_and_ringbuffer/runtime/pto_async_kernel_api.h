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

// Public surface: get_async_ctx, register_completion_condition,
// send_notification, save_expected_notification_counter. Everything else
// lives in pto2::detail and is reserved for backend adapters / internal use.
namespace pto2::detail {

inline __aicore__ void defer_load_ingress(AsyncCtx &ctx) {
    if (ctx.completion_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t line = reinterpret_cast<uintptr_t>(ctx.completion_count) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
    dcci((__gm__ int32_t *)line, SINGLE_CACHE_LINE);
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

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

}  // namespace pto2::detail

inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx = lc->async_ctx;
    pto2::detail::defer_load_ingress(ctx);
    return ctx;
}

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
send_notification(volatile __gm__ void *remote_counter_addr, int32_t value, pto::comm::NotifyOp notify_op) {
    __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ void *>(remote_counter_addr));
    pto::comm::Signal signal(counter);
    pto::comm::TNOTIFY(signal, value, notify_op);
}

inline __aicore__ void
save_expected_notification_counter(AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected_value) {
    CompletionToken token{reinterpret_cast<uint64_t>(counter_addr), expected_value, COMPLETION_ENGINE_SDMA,
                          COMPLETION_TYPE_COUNTER, 0};
    (void)register_completion_condition(ctx, token);
    pto2::detail::defer_flush(ctx);
}

#endif  // PTO_ASYNC_KERNEL_API_H
