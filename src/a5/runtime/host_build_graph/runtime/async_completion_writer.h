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
    if (ctx.completion_reservation != nullptr && ctx.completion_reservation->sealed != 0) return;
    if (ctx.completion_error_code != nullptr && *ctx.completion_error_code == 0) *ctx.completion_error_code = error;
}

template <typename Context>
inline __aicore__ bool async_completion_writer_valid(const Context &ctx) {
    return ctx.task_token.is_valid() && ctx.completion_count != nullptr && ctx.completion_error_code != nullptr &&
           ctx.completion_entries != nullptr && ctx.completion_capacity >= 1 &&
           ctx.completion_capacity <= MAX_COMPLETIONS_PER_TASK;
}

// The reservation is a kernel-local handle. Its live serial and progress belong
// to the shared writer, so copying AsyncCtx cannot bypass an outstanding submit.
struct AsyncCompletionReservation {
    uint64_t writer{0};
    uint32_t serial{0};
    uint32_t begin{0};
    uint32_t limit{0};
    bool valid{false};
};

template <typename Context>
inline __aicore__ bool async_completion_open(Context &ctx) {
    if (!async_completion_writer_valid(ctx)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (*ctx.completion_error_code != 0) return false;
    if (ctx.completion_reservation != nullptr && ctx.completion_reservation->sealed != 0) return false;
    return true;
}

template <typename Context>
inline __aicore__ bool
async_completion_reserve(Context &ctx, uint32_t upper_bound, AsyncCompletionReservation *reservation) {
    if (reservation != nullptr) *reservation = {};
    if (reservation == nullptr || !async_completion_open(ctx)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    const uint32_t count = *ctx.completion_count;
    auto *state = ctx.completion_reservation;
    if (upper_bound == 0 || (state != nullptr && (state->limit != 0 || state->serial == UINT32_MAX))) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (count > ctx.completion_capacity || upper_bound > ctx.completion_capacity - count) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return false;
    }
    if (state != nullptr) {
        state->serial = state->serial + 1;
        state->begin = count;
        state->limit = upper_bound;
        state->written = 0;
    }
    *reservation = {
        reinterpret_cast<uint64_t>(ctx.completion_count), state == nullptr ? 0 : state->serial, count, upper_bound, true
    };
    return true;
}

template <typename Context>
inline __aicore__ bool async_completion_reservation_valid(Context &ctx, const AsyncCompletionReservation &reservation) {
    if (!async_completion_open(ctx)) return false;
    const auto *state = ctx.completion_reservation;
    const uint32_t count = *ctx.completion_count;
    if (!reservation.valid || reservation.writer != reinterpret_cast<uint64_t>(ctx.completion_count) ||
        reservation.limit == 0 || reservation.begin > ctx.completion_capacity ||
        reservation.limit > ctx.completion_capacity - reservation.begin || count < reservation.begin ||
        count - reservation.begin > reservation.limit ||
        (state != nullptr &&
         (reservation.serial == 0 || state->serial != reservation.serial || state->begin != reservation.begin ||
          state->limit != reservation.limit || state->written != count - reservation.begin))) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return true;
}

template <typename Context>
inline __aicore__ bool
async_completion_accept_count(Context &ctx, const AsyncCompletionReservation &reservation, uint32_t actual_count) {
    if (!async_completion_reservation_valid(ctx, reservation)) return false;
    if (actual_count > reservation.limit || *ctx.completion_count != reservation.begin) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return true;
}

template <typename Context>
inline __aicore__ bool
async_completion_commit(Context &ctx, const AsyncCompletionReservation &reservation, const CompletionToken &token) {
    if (!async_completion_reservation_valid(ctx, reservation)) return false;
    if (!async_completion_format_valid(token)) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    const uint32_t index = *ctx.completion_count;
    if (index - reservation.begin == reservation.limit) {
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
    if (ctx.completion_reservation != nullptr)
        ctx.completion_reservation->written = ctx.completion_reservation->written + 1;
    return true;
}

template <typename Context>
inline __aicore__ bool
async_completion_finish(Context &ctx, const AsyncCompletionReservation &reservation, uint32_t actual_count) {
    if (!async_completion_reservation_valid(ctx, reservation)) return false;
    if (*ctx.completion_count - reservation.begin != actual_count) {
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    if (ctx.completion_reservation != nullptr) {
        ctx.completion_reservation->limit = 0;
        ctx.completion_reservation->begin = 0;
        ctx.completion_reservation->written = 0;
    }
    return true;
}

template <typename Context>
inline __aicore__ bool async_completion_append(Context &ctx, const CompletionToken &token) {
    AsyncCompletionReservation reservation{};
    return async_completion_reserve(ctx, 1, &reservation) && async_completion_commit(ctx, reservation, token) &&
           async_completion_finish(ctx, reservation, 1);
}

// Seal only publishes the writer's decision. Lease activation and the execution
// contribution remain with the scheduler after token storage is made visible.
template <typename Context>
inline __aicore__ bool async_completion_seal(Context &ctx) {
    auto *state = ctx.completion_reservation;
    if (state == nullptr || state->sealed != 0) return false;
    bool valid = async_completion_writer_valid(ctx);
    if (!valid || state->limit != 0 || state->reserved != 0)
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    else if (*ctx.completion_count > ctx.completion_capacity)
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    valid = valid && *ctx.completion_error_code == 0;
    state->sealed = 1;
    return valid;
}

// Count/error and tokens can occupy separate allocations/cache lines. Never
// flush the gap: it may hold atomic-only leases and directory words belonging
// to other cores. Cache supplies flush_range(address,bytes) and finish().
template <typename Context, typename Cache>
inline __aicore__ void async_completion_flush(Context &ctx, Cache &cache) {
    // Resident writers publish once at seal, before their CHECK lease is visible.
    // Legacy mailbox registration still needs publication at the API boundary.
    if (!ctx.task_token.is_valid() || ctx.completion_reservation != nullptr) return;
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
