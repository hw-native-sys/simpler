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
#include "async_condition_writer.h"
#include "aicore_completion_mailbox_types.h"
#include "host_build_graph/completion_token.h"
#include "host_build_graph/runtime_status.h"
#include "platform_comm/ccu/ccu_device_submission.h"
#include "platform_comm/comm_context.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// Kernel-local view. The legacy wire prefix lives in LocalContext; resident
// state is addressed explicitly through the GlobalContext fields.
struct AsyncCtx : LegacyAsyncCtx {
    uint32_t resident_enabled{0};
    uint32_t urma_sender_id{UINT32_MAX};
    pto2::resident::AsyncWriter resident{};

    static inline AsyncCtx make(TaskId task_token, volatile __gm__ DeferredCompletionSlab *buffer) {
        AsyncCtx ctx{};
        const auto legacy = LegacyAsyncCtx::make(task_token, buffer);
        ctx.completion_count = legacy.completion_count;
        ctx.completion_error_code = legacy.completion_error_code;
        ctx.completion_entries = legacy.completion_entries;
        ctx.completion_capacity = legacy.completion_capacity;
        ctx.task_token = legacy.task_token;
        return ctx;
    }
};

// Cache and error adapters for the existing address-based registration API.
namespace pto2::detail {

inline __aicore__ bool resident_context_is_live(const AsyncCtx &ctx) {
    if (ctx.resident_enabled != 1) return false;
    const auto *data = pto2::resident::detail::group(ctx.resident);
    if (data == nullptr) return false;
    // A sticky registration error does not erase the context's identity.
    return data->state == pto2::resident::AsyncPublicationState::OPEN &&
           pto2::resident::detail::layout_valid(ctx.resident, *data);
}

inline __aicore__ void defer_load_slab(AsyncCtx &ctx) {
    if (ctx.resident_enabled != 0 || ctx.completion_count == nullptr) return;
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    uintptr_t line = reinterpret_cast<uintptr_t>(ctx.completion_count) & ~(uintptr_t(CHIP_ALIGN_SIZE) - 1u);
    dcci((__gm__ int32_t *)line, SINGLE_CACHE_LINE);
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) {
    if (ctx.resident_enabled != 0) {
        pto2::resident::detail::fail(ctx.resident, error_code);
        return;
    }
    if (ctx.task_token.is_valid() && ctx.completion_error_code != nullptr) {
        *ctx.completion_error_code = error_code;
    }
}

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

inline __aicore__ void defer_flush(AsyncCtx &ctx) {
    if (ctx.resident_enabled != 0 || !ctx.task_token.is_valid() || ctx.completion_count == nullptr) return;
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
    AsyncCtx ctx{};
    const auto *global =
        reinterpret_cast<__gm__ const GlobalContext *>(static_cast<uintptr_t>(args[PAYLOAD_GLOBAL_CONTEXT_INDEX]));
    ctx.resident_enabled = global->async_present;
    if (ctx.resident_enabled != 0) {
        ctx.task_token = TaskId::invalid();
        ctx.resident.image = reinterpret_cast<__gm__ uint8_t *>(global->async_image_address);
        ctx.resident.image_bytes = global->async_image_bytes;
        ctx.resident.run_epoch = global->async_run_epoch;
        ctx.resident.group_offset = global->async_group_offset;
        ctx.resident.group_index = global->async_group_index;
        ctx.urma_sender_id = global->urma_sender_id == UINT16_MAX ? UINT32_MAX : global->urma_sender_id;
        if (ctx.resident_enabled != 1 || global->async_output_capture_refused != 0) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
            ctx.resident_enabled = UINT32_MAX;
            return ctx;
        }
        if (!pto2::detail::resident_context_is_live(ctx)) {
            (void)pto2::resident::detail::valid(ctx.resident);
            return ctx;
        }
        auto *data = pto2::resident::detail::group(ctx.resident);
        ctx.completion_count = &data->count;
        ctx.completion_error_code = &data->error_code;
        ctx.completion_entries =
            reinterpret_cast<volatile __gm__ DeferredCompletionEntry *>(ctx.resident.image + data->entries_offset);
        ctx.completion_capacity = data->capacity;
        return ctx;
    }
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    TaskId::assign(ctx.task_token, lc->async_ctx.task_token);
    pto2::detail::defer_load_slab(ctx);
    return ctx;
}

inline __aicore__ bool async_ctx_is_deferred(const AsyncCtx &ctx) {
    return ctx.resident_enabled == 1 ? pto2::detail::resident_context_is_live(ctx) :
           ctx.resident_enabled == 0 ? ctx.task_token.is_valid() :
                                       false;
}

// Canonical writer: backend submit handlers build a CompletionToken and pass
// it here. Writes one DeferredCompletionEntry to the AsyncCtx slab and
// bumps completion_count. Returns false on overflow (also stores
// SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW in ctx.completion_error_code) or when ctx is
// not currently a deferred context.
inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    if (ctx.resident_enabled != 0) return pto2::resident::register_completion_condition(ctx.resident, token);
    if (!ctx.task_token.is_valid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        return false;
    }

    uint32_t idx = *ctx.completion_count;
    if (idx >= ctx.completion_capacity) {
        if (ctx.completion_error_code != nullptr) {
            *ctx.completion_error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
        }
        return false;
    }

    volatile __gm__ DeferredCompletionEntry *slot = &ctx.completion_entries[idx];
    slot->addr = token.addr;
    slot->backend_cookie = token.backend_cookie;
    slot->expected_value = token.expected_value;
    slot->engine = token.engine;
    slot->completion_type = token.completion_type;
    slot->_pad = 0;
    *ctx.completion_count = idx + 1;
    return true;
}

inline __aicore__ bool submit_ccu_mission(
    AsyncCtx &ctx, const __gm__ CommContext *domain, uint32_t ccu_id, const void *parameters, uint64_t parameter_size
) {
    const auto *workspace = simpler::ccu::ccu_workspace(domain);
    const auto *mission = simpler::ccu::ccu_control(workspace, ccu_id);
    auto register_ccu_completion = [&](uint64_t address, uint64_t mask) __aicore__ {
        const CompletionToken token{address, 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, mask};
        if (!register_completion_condition(ctx, token)) return false;
        pto2::detail::defer_flush(ctx);
        return true;
    };
    const bool submitted = simpler::ccu::submit_mission(mission, parameters, parameter_size, register_ccu_completion);
    if (!submitted) pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return submitted;
}

template <size_t N>
inline __aicore__ bool submit_ccu_mission(
    AsyncCtx &ctx, const __gm__ CommContext *domain, uint32_t ccu_id, const simpler::ccu::CcuCallArg (&call_args)[N]
) {
    uint64_t words[N * 2]{};
    uint64_t bytes = 0;
    if (!simpler::ccu::resolve_call_args(domain, call_args, words, bytes)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return submit_ccu_mission(ctx, domain, ccu_id, words, bytes);
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
