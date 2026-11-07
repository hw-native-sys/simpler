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
#include "host_build_graph/completion_token.h"

template <typename Context>
inline __aicore__ void async_completion_error(Context &ctx, int32_t error) {
    if (ctx.task_token.is_valid() && ctx.completion_error_code != nullptr && *ctx.completion_error_code == 0)
        *ctx.completion_error_code = error;
}

template <typename Context>
inline __aicore__ bool async_completion_writer_valid(const Context &ctx) {
    return ctx.task_token.is_valid() && ctx.completion_count != nullptr && ctx.completion_error_code != nullptr &&
           ctx.completion_entries != nullptr && ctx.completion_capacity >= 1 &&
           ctx.completion_capacity <= MAX_COMPLETIONS_PER_TASK;
}

// Every context has one kernel writer. This local grant reserves part of that
// writer's already-exclusive capacity across a synchronous backend submit call;
// it is not a shared queue reservation or permission to reuse a DMA channel.
struct AsyncCompletionAllowance {
    uint32_t begin{0};
    uint32_t limit{0};
    bool valid{false};
};

template <typename Context>
inline __aicore__ bool
async_completion_reserve(Context &ctx, uint32_t upper_bound, AsyncCompletionAllowance *allowance) {
    if (allowance != nullptr) *allowance = {};
    if (allowance == nullptr || !async_completion_writer_valid(ctx)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (*ctx.completion_error_code != 0) return false;
    const uint32_t count = *ctx.completion_count;
    if (count > ctx.completion_capacity || upper_bound > ctx.completion_capacity - count) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return false;
    }
    *allowance = {count, upper_bound, true};
    return true;
}

template <typename Context>
inline __aicore__ bool
async_completion_accept_count(Context &ctx, const AsyncCompletionAllowance &allowance, uint32_t actual_count) {
    if (!async_completion_writer_valid(ctx) || !allowance.valid || actual_count > allowance.limit ||
        *ctx.completion_count != allowance.begin || allowance.begin > ctx.completion_capacity ||
        allowance.limit > ctx.completion_capacity - allowance.begin) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return *ctx.completion_error_code == 0;
}

template <typename Context>
inline __aicore__ bool async_completion_append(Context &ctx, const CompletionToken &token) {
    if (!async_completion_writer_valid(ctx)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (*ctx.completion_error_code != 0) return false;
    const uint32_t index = *ctx.completion_count;
    if (index >= ctx.completion_capacity) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return false;
    }
    auto &entry = ctx.completion_entries[index];
    entry.addr = token.addr;
    entry.backend_cookie = token.backend_cookie;
    entry.expected_value = token.expected_value;
    entry.engine = token.engine;
    entry.completion_type = token.completion_type;
    entry._pad = 0;
    *ctx.completion_count = index + 1;
    return true;
}

// Count/error and tokens can occupy separate allocations/cache lines. Never
// flush the gap: it may hold atomic-only leases and directory words belonging
// to other cores. Cache supplies flush_range(address,bytes) and finish().
template <typename Context, typename Cache>
inline __aicore__ void async_completion_flush(Context &ctx, Cache &cache) {
    if (!ctx.task_token.is_valid()) return;
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
