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
#include "host_build_graph/completion_types.h"
#include "host_build_graph/constants.h"
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

struct AsyncCtx {
    volatile __gm__ uint32_t *completion_count{nullptr};
    volatile __gm__ int32_t *completion_error_code{nullptr};
    volatile __gm__ DeferredCompletionEntry *completion_entries{nullptr};
    uint32_t completion_capacity{0};
    uint32_t resident_enabled{0};
    pto2::resident::AsyncWriter resident{};
};

// Validation and sticky errors for the resident registration API.
namespace pto2::detail {

inline __aicore__ bool resident_context_is_live(const AsyncCtx &ctx) {
    if (ctx.resident_enabled != 1) return false;
    const auto *data = pto2::resident::detail::group(ctx.resident);
    if (data == nullptr) return false;
    // A sticky registration error does not erase the context's identity.
    return data->state == pto2::resident::AsyncPublicationState::OPEN &&
           pto2::resident::detail::layout_valid(ctx.resident, *data);
}

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) {
    pto2::resident::detail::fail(ctx.resident, error_code);
}

// Address-based kernels use this for data publication independently of async storage.
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

// Publish the current resident group; the executor still owns sealing and joining.
inline __aicore__ void defer_flush(AsyncCtx &ctx) {
    if (!resident_context_is_live(ctx)) return;
    auto *data = pto2::resident::detail::group(ctx.resident);
    const uint32_t count = data->count > data->capacity ? data->capacity : data->count;
    defer_flush_range(ctx.completion_entries, count * sizeof(DeferredCompletionEntry));
    defer_flush_range(data, sizeof(*data));
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
}

}  // namespace pto2::detail

inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    AsyncCtx ctx{};
    const auto *global =
        reinterpret_cast<__gm__ const GlobalContext *>(static_cast<uintptr_t>(args[PAYLOAD_GLOBAL_CONTEXT_INDEX]));
    ctx.resident_enabled = global->async_present;
    if (ctx.resident_enabled != 0) {
        ctx.resident.image = reinterpret_cast<__gm__ uint8_t *>(global->async_image_address);
        ctx.resident.image_bytes = global->async_image_bytes;
        ctx.resident.run_epoch = global->async_run_epoch;
        ctx.resident.group_offset = global->async_group_offset;
        ctx.resident.group_index = global->async_group_index;
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
    return ctx;
}

inline __aicore__ bool async_ctx_is_deferred(const AsyncCtx &ctx) {
    return pto2::detail::resident_context_is_live(ctx);
}

inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
    return ctx.resident_enabled == 1 && pto2::resident::register_completion_condition(ctx.resident, token);
}

inline __aicore__ bool submit_ccu_mission(
    AsyncCtx &ctx, const __gm__ CommContext *domain, uint32_t ccu_id, const void *parameters, uint64_t parameter_size
) {
    const auto *workspace = simpler::ccu::ccu_workspace(domain);
    const auto *mission = simpler::ccu::ccu_control(workspace, ccu_id);
    auto register_ccu_completion = [&](uint64_t address, uint64_t mask) __aicore__ {
        const CompletionToken token{address, 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, mask};
        return register_completion_condition(ctx, token);
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
}
