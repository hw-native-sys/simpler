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

#include "async_completion_format.h"
#include "host_build_graph/completion_token.h"

template <typename Context>
inline __aicore__ void async_completion_error(Context &ctx, int32_t error) {
    if (ctx.completion_sealed != nullptr && *ctx.completion_sealed != 0) return;
    if (ctx.completion_error_code != nullptr && *ctx.completion_error_code == 0) *ctx.completion_error_code = error;
}

template <typename Context>
inline __aicore__ bool async_completion_writer_valid(const Context &ctx) {
    return ctx.task_token.is_valid() && ctx.completion_count != nullptr && ctx.completion_error_code != nullptr &&
           ctx.completion_entries != nullptr && ctx.completion_capacity >= 1 &&
           ctx.completion_capacity <= MAX_COMPLETIONS_PER_TASK;
}

template <typename Context>
inline __aicore__ bool async_completion_open(Context &ctx) {
    if (!async_completion_writer_valid(ctx)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return *ctx.completion_error_code == 0 && (ctx.completion_sealed == nullptr || *ctx.completion_sealed == 0);
}

// A kernel is the sole writer until return. SDMA checks its maximum record count
// before entering the synchronous SDK submit; no ticket survives that call.
template <typename Context>
inline __aicore__ bool async_completion_has_capacity(Context &ctx, uint32_t count) {
    if (!async_completion_open(ctx)) return false;
    if (*ctx.completion_count > ctx.completion_capacity || count > ctx.completion_capacity - *ctx.completion_count) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return false;
    }
    return true;
}

template <typename Context>
inline __aicore__ bool async_completion_append(Context &ctx, const CompletionToken &token) {
    if (!async_completion_has_capacity(ctx, 1)) return false;
    if (!async_completion_format_valid(token)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    auto &entry = ctx.completion_entries[*ctx.completion_count];
    entry.addr = token.addr;
    entry.backend_cookie = token.backend_cookie;
    entry.expected_value = token.expected_value;
    entry.engine = token.engine;
    entry.completion_type = token.completion_type;
    entry._pad = 0;
    *ctx.completion_count = *ctx.completion_count + 1;
    return true;
}

// Copies share count, error and seal state. CHECK sees them only after return.
template <typename Context>
inline __aicore__ bool async_completion_seal(Context &ctx) {
    if (ctx.completion_sealed == nullptr || *ctx.completion_sealed != 0) return false;
    const bool valid = async_completion_has_capacity(ctx, 0);
    *ctx.completion_sealed = 1;
    return valid;
}

// Count/error and tokens can occupy separate allocations/cache lines. Never
// flush the gap: it may hold atomic-only leases and directory words belonging
// to other cores. Cache supplies flush_range(address,bytes) and finish().
template <typename Context, typename Cache>
inline __aicore__ void async_completion_flush(Context &ctx, Cache &cache) {
    // Resident writers publish once at seal, before their CHECK lease is visible.
    // Legacy mailbox registration still needs publication at the API boundary.
    if (!ctx.task_token.is_valid() || ctx.completion_sealed != nullptr) return;
    uint32_t count = ctx.completion_count != nullptr ? *ctx.completion_count : 0;
    if (count > ctx.completion_capacity) count = ctx.completion_capacity;
    if (count > MAX_COMPLETIONS_PER_TASK) count = MAX_COMPLETIONS_PER_TASK;
    if (ctx.completion_count != nullptr) cache.flush_range(ctx.completion_count, sizeof(*ctx.completion_count));
    if (ctx.completion_error_code != nullptr)
        cache.flush_range(ctx.completion_error_code, sizeof(*ctx.completion_error_code));
    if (ctx.completion_entries != nullptr && count != 0)
        cache.flush_range(ctx.completion_entries, count * sizeof(DeferredCompletionEntry));
    cache.finish();
}
