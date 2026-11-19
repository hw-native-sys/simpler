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
#include "aicore_completion_mailbox_types.h"
#include "completion_token.h"
#include "completion_context.h"
#include "runtime_status.h"

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

inline __aicore__ bool async_ctx_is_deferred(const AsyncCtx &ctx) { return ctx.task_token.is_valid(); }

// Canonical writer: backend submit handlers build a CompletionToken and pass
// it here. Writes one DeferredCompletionEntry to the AsyncCtx slab and
// bumps completion_count. Returns false on overflow (also stores
// SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW in ctx.completion_error_code) or when ctx is
// not currently a deferred context.
inline __aicore__ bool register_completion_condition(AsyncCtx &ctx, const CompletionToken &token) {
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

#endif  // PTO_ASYNC_KERNEL_API_H
