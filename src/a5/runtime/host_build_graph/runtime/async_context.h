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

#include "intrinsic.h"
#include "async_condition_writer.h"
#include "host_build_graph/constants.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

struct AsyncCtx {
    uint32_t urma_sender_id{UINT32_MAX};
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
    // Mark before validation: failed or unfinished registration still needs sealing.
    auto *invocation = simpler::percounter::percounter_current_context();
    invocation->async_touched = 1;
    AsyncCtx ctx{};
    ctx.urma_sender_id = invocation->urma_sender_id;
    const GlobalContext global = load_global_context(args);
    ctx.resident_enabled = global.async_present;
    if (ctx.resident_enabled != 0) {
        ctx.resident.image = reinterpret_cast<__gm__ uint8_t *>(global.async_image_address);
        ctx.resident.image_bytes = global.async_image_bytes;
        ctx.resident.run_epoch = global.async_run_epoch;
        ctx.resident.group_offset = global.async_group_offset;
        ctx.resident.group_index = global.async_group_index;
        if (ctx.resident_enabled != 1 || global.async_output_capture_refused != 0) {
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
