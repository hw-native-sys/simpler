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

#if !defined(__CPU_SIM)
#include "async_kernel_api.h"
#endif
#include "aicore_completion_mailbox_types.h"
#include "intrinsic.h"
#include "completion_token.h"
#include "runtime_status.h"
#include "urma/urma_request.h"

#if defined(__CPU_SIM)
inline __aicore__ AsyncCtx get_async_ctx(__gm__ int64_t *args) {
    __gm__ LocalContext *lc =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uintptr_t>(args[PAYLOAD_LOCAL_CONTEXT_INDEX]));
    AsyncCtx ctx{};
    ctx.completion_count = lc->async_ctx.completion_count;
    ctx.completion_error_code = lc->async_ctx.completion_error_code;
    ctx.completion_entries = lc->async_ctx.completion_entries;
    ctx.completion_capacity = lc->async_ctx.completion_capacity;
    ctx.urma_sender_id = lc->async_ctx.urma_sender_id;
    TaskId::assign(ctx.task_token, lc->async_ctx.task_token);
    return ctx;
}

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
#endif

namespace pto2::detail {

#if defined(__CPU_SIM)
inline __aicore__ void defer_load_slab(AsyncCtx & /*ctx*/) { __asm__ __volatile__("" ::: "memory"); }

inline __aicore__ void defer_error(AsyncCtx &ctx, int32_t error_code) {
    if (ctx.task_token.is_valid() && ctx.completion_error_code != nullptr) {
        *ctx.completion_error_code = error_code;
    }
}

inline __aicore__ void defer_flush(AsyncCtx & /*ctx*/) { __asm__ __volatile__("" ::: "memory"); }
#endif

inline __aicore__ void set_urma_error(AsyncCtx &ctx, int32_t error_code) {
    if (ctx.completion_error_code != nullptr && *ctx.completion_error_code == SIMPLER_ERROR_NONE) {
        defer_error(ctx, error_code);
    }
}

}  // namespace pto2::detail

namespace simpler::urma {

class TmrRequestAdapter {
public:
    explicit __aicore__ TmrRequestAdapter(AsyncCtx &ctx) :
        ctx_(ctx) {}

    inline __aicore__ uint32_t sender_id() const { return ctx_.urma_sender_id; }

    inline __aicore__ bool capacity_available(uint64_t required) const {
        return ctx_.task_token.is_valid() && ctx_.completion_count != nullptr && ctx_.completion_entries != nullptr &&
               ctx_.completion_error_code != nullptr && *ctx_.completion_error_code == SIMPLER_ERROR_NONE &&
               *ctx_.completion_count <= ctx_.completion_capacity &&
               required <= ctx_.completion_capacity - *ctx_.completion_count;
    }

    inline __aicore__ bool reserve(uint64_t required) const { return capacity_available(required); }
    inline __aicore__ bool finish() const { return true; }

    inline __aicore__ void set_error(RequestError error) {
        pto2::detail::set_urma_error(
            ctx_, error == RequestError::INVALID ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID :
                                                   SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
        );
    }

    inline __aicore__ bool register_target(const CompletionTarget &target) {
        const CompletionToken token{
            (static_cast<uint64_t>(target.queue_row) << 32) | target.target_bb, target.target_cqe,
            COMPLETION_ENGINE_URMA, COMPLETION_TYPE_URMA_EVENT_HANDLE, reinterpret_cast<uint64_t>(target.workspace)
        };
        return register_completion_condition(ctx_, token);
    }

    inline __aicore__ void publish() { pto2::detail::defer_flush(ctx_); }

private:
    AsyncCtx &ctx_;
};

}  // namespace simpler::urma

// Kernel callers use this namespace for remote MR addressing.
namespace pto2::urma_backend {
using simpler::urma::peer_mr_base_addr;
using simpler::urma::peer_mr_ptr;
}  // namespace pto2::urma_backend

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
    simpler::urma::TmrRequestAdapter runtime(ctx);
    const bool submitted = simpler::urma::submit_chunked_urma_request(runtime, desc);
    runtime.publish();
    return submitted;
}
