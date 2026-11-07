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

#include <stdint.h>

#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#include "intrinsic.h"
#include "async_completion_writer.h"
#include "async_sdma_guard.h"
#include "async_submission_context.h"
#include "scheduler/scheduler_async_layout.h"
#include "dispatch_payload.h"
#include "aicore_completion_mailbox_types.h"
#include "host_build_graph/completion_token.h"
#include "host_build_graph/runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// Public surface: get_async_ctx, async_ctx_is_deferred,
// register_completion_condition, send_notification,
// save_expected_notification_counter. Everything else lives in
// pto2::detail and is reserved for backend adapters / internal use.
namespace pto2::detail {

inline __aicore__ void defer_load_slab(AsyncCtx &ctx) {
    if (ctx.completion_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t line = reinterpret_cast<uintptr_t>(ctx.completion_count) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
    dcci((__gm__ int32_t *)line, SINGLE_CACHE_LINE);
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) { async_completion_error(ctx, error_code); }

inline __aicore__ void defer_flush_range(volatile __gm__ void *addr, uint32_t size_bytes) {
    if (addr == nullptr || size_bytes == 0) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
    uintptr_t end =
        (reinterpret_cast<uintptr_t>(addr) + size_bytes + CHIP_ALIGN_SIZE - 1u) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
    for (uintptr_t p = start; p < end; p += CHIP_ALIGN_SIZE) {
        dcci((__gm__ int32_t *)p, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
#else
    (void)addr;
    (void)size_bytes;
#endif
}

struct CompletionCache {
    inline __aicore__ void flush_range(volatile __gm__ void *address, uint32_t bytes) {
        defer_flush_range(address, bytes);
    }
    inline __aicore__ void finish() {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
#if defined(__CPU_SIM)
        dsb(0);
#else
        dsb(DSB_DDR);
#endif
        pipe_barrier(PIPE_ALL);
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
};

inline __aicore__ void defer_flush(AsyncCtx &ctx) {
    CompletionCache cache;
    async_completion_flush(ctx, cache);
}

}  // namespace pto2::detail

inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx{};
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    TaskId::assign(ctx.task_token, lc->async_ctx.task_token);
    pto2::detail::defer_load_slab(ctx);
    const uint64_t argument_address = reinterpret_cast<uint64_t>(args);
    if (argument_address < offsetof(DispatchPayload, args)) return ctx;
    const uint64_t payload_address = argument_address - offsetof(DispatchPayload, args);
    if (payload_address % alignof(DispatchPayload) != 0) return ctx;
    const auto *payload = reinterpret_cast<__gm__ const DispatchPayload *>(payload_address);
    if (payload->args[SPMD_LOCAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&payload->local_context) &&
        payload->args[SPMD_GLOBAL_CONTEXT_INDEX] == reinterpret_cast<uint64_t>(&payload->global_context)) {
        const uint64_t address = payload->async_submission_context;
        if (address != 0 && address % 128 == 0 &&
            reinterpret_cast<__gm__ const AsyncSdmaGuardContext *>(address)->magic == ASYNC_SDMA_GUARD_MAGIC)
            ctx.sdma_guard_context = address;
        else if (auto *qualified = get_async_submission_context(args)) {
            const auto &header = qualified->header;
            if (header.region_bytes < sizeof(SchedulerAsyncLayout)) {
                pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
                return ctx;
            }
            const auto *layout = reinterpret_cast<__gm__ const SchedulerAsyncLayout *>(header.region_address);
            if (layout->version != SCHEDULER_ASYNC_LAYOUT_VERSION ||
                !async_sdma_guard_region(
                    header.region_address, header.region_bytes, header.run_epoch, layout->sdma_guard_context_offset,
                    layout->sdma_guard_count, &ctx.sdma_guard_context
                ))
                pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        }
    }
    return ctx;
}

inline __aicore__ bool async_ctx_is_deferred(const AsyncCtx &ctx) { return ctx.task_token.is_valid(); }

// Canonical writer: backend submit handlers build a CompletionToken and pass
// it here. Writes one DeferredCompletionEntry to the AsyncCtx slab and
// bumps completion_count. Returns false on overflow (also stores
// SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW in ctx.completion_error_code) or when ctx is
// not currently a deferred context.
inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    return async_completion_append(ctx, token);
}

inline __aicore__ void
send_notification(volatile __gm__ void *remote_counter_addr, int32_t value, pto::comm::NotifyOp notify_op) {
    __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ void *>(remote_counter_addr));
    pto::comm::Signal signal(counter);
    pto::comm::TNOTIFY(signal, value, notify_op);
}

inline __aicore__ void
save_expected_notification_counter(AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected_value) {
    CompletionToken token{
        reinterpret_cast<uint64_t>(counter_addr), expected_value, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    (void)register_completion_condition(ctx, token);
    pto2::detail::defer_flush(ctx);
}
