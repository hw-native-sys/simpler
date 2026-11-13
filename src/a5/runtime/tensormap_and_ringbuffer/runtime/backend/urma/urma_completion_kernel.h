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
#include "platform_comm/comm_context.h"
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

#if defined(PTO_URMA_SUPPORTED)
// Hold the error until the caller knows whether all published DMA is tracked.
class TmrBatchRequestAdapter : public TmrRequestAdapter {
public:
    explicit __aicore__ TmrBatchRequestAdapter(AsyncCtx &ctx) :
        TmrRequestAdapter(ctx) {}

    inline __aicore__ void set_error(RequestError error) {
        error_code = error == RequestError::INVALID ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID :
                                                      SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
    }

    int32_t error_code{SIMPLER_ERROR_NONE};
};

// Flush events carry cumulative targets. Remove the untouched suffix so the
// ordinary event registrar waits only on this batch's participating jetties.
inline __aicore__ bool trim_urma_batch_event(
    pto::comm::AsyncEvent &event, const pto::comm::AsyncSession &session, const QueueProducer *initial
) {
    const uint32_t count = event.urmaJettyCount;
    if (event.engine != session.engine || event.urmaJettyBase != session.qpIdxBase ||
        count != session.cfg.urma.qpCount) {
        return false;
    }
    uint32_t active = 0;
    for (uint32_t j = 0; j < count; ++j) {
        const uint32_t bb = event.urmaTargetBbPerJetty[j] - initial[j].bb;
        const uint32_t cqe = event.urmaTargetCqePerJetty[j] - initial[j].cqe;
        if (bb == 0 && cqe == 0) continue;
        if (active != j || bb >= (1u << 31) || cqe >= (1u << 31)) return false;
        ++active;
    }
    event.urmaJettyCount = active;
    return active != 0;
}

#endif

inline __aicore__ bool urma_range_contains(uint64_t base, uint64_t size, uint64_t addr, uint64_t bytes) {
    return base != 0 && addr >= base && addr - base < size && bytes <= size - (addr - base) &&
           bytes - 1 <= UINT64_MAX - addr;
}

template <typename DstTensor, typename SrcTensor>
inline __aicore__ int32_t submit_urma_put_batch(
    AsyncCtx &ctx, const __gm__ GlobalContext *global, const __gm__ CommContext &domain,
    const UrmaRequestDescriptor<DstTensor, SrcTensor> *requests, uint32_t count, uint32_t batch_size, bool &accounted
) {
    using RawDType = typename DstTensor::RawDType;
    if (batch_size == 0 || (count != 0 && requests == nullptr)) return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    if (count == 0) return SIMPLER_ERROR_NONE;
    const uint32_t peer = requests[0].remote_rank;
    const uint32_t rank = domain.rankId;
    const auto &urma_workspace = comm_workspace<CommWorkspaceKind::URMA>(domain);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(urma_workspace.addr);
    // Ordinary Device domains publish a mapped peer window. The PUT itself uses the peer MR address.
    if (domain.rankNum > COMM_MAX_RANK_NUM || rank >= domain.rankNum || peer >= domain.rankNum || peer == rank ||
        domain.windowsIn[peer] == 0 || workspace == nullptr || urma_workspace.size < sizeof(UrmaInfo)) {
        return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    }
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(workspace);
    if (info->layout != UrmaLayout::PER_PEER || info->rankCount != domain.rankNum || info->rowsPerPeer == 0 ||
        info->railCycleLen != 0 || static_cast<uint64_t>(info->rankCount) * info->rowsPerPeer != info->jettyCount ||
        info->memPtr == 0) {
        return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    }
    // All PER_PEER channels for a peer register the same symmetric domain window.
    auto *peer_mr = reinterpret_cast<__gm__ pto::comm::urma::UrmaMemInfo *>(info->memPtr) + peer * info->rowsPerPeer;
    const uint64_t src_base = domain.windowsIn[rank];
    const uint64_t dst_base = peer_mr->addr;
    const uint64_t dst_size = domain.winSize < peer_mr->len ? domain.winSize : peer_mr->len;
    uint32_t last = 0;
    uint64_t chunks = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto desc = requests[i];
        uint64_t elems = 0;
        // The ISA's deferred PUT validation reads each extent and stride as an int.
        if (desc.op != UrmaOp::TPUT || desc.workspace != workspace || desc.remote_rank != peer ||
            !validate_urma_transfer(desc, elems) || elems > INT32_MAX) {
            return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        }
        if (elems == 0) continue;
        const uint64_t bytes = elems * sizeof(RawDType);
        const uint64_t src = reinterpret_cast<uint64_t>(static_cast<const __gm__ void *>(desc.src.data()));
        const uint64_t dst = reinterpret_cast<uint64_t>(static_cast<const __gm__ void *>(desc.dst.data()));
        if (!urma_range_contains(src_base, domain.winSize, src, bytes) ||
            !urma_range_contains(dst_base, dst_size, dst, bytes)) {
            return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        }
        const uint64_t request_chunks = chunk_count(bytes);
        if (request_chunks > UINT32_MAX - chunks) return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        chunks += request_chunks;
        last = i;
    }
    if (chunks == 0) return SIMPLER_ERROR_NONE;
    if (global == nullptr || global->urma_batch_guard != URMA_BATCH_GUARD)
        return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
#if defined(PTO_URMA_SUPPORTED)
    pto::comm::AsyncSession session;
    if (!build_urma_session(ctx.urma_sender_id, workspace, peer, session))
        return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;

    QueueProducer initial[pto::comm::urma::kUrmaMaxJettiesPerCore]{};
    // Preflight the whole list before staging. AICPU owns CQ consumption.
    TmrBatchRequestAdapter runtime(ctx);
    if (!preflight_urma_request(runtime, session, chunks, initial)) return runtime.error_code;

    session.batchSize = batch_size;
    uint32_t staged = 0;
    pto::comm::AsyncEvent published{};
    published.handle = 0;
    int32_t error = SIMPLER_ERROR_NONE;
    for (uint32_t i = 0; i <= last; ++i) {
        auto desc = requests[i];
        const uint64_t elems = desc.src.GetShape(pto::GlobalTensorDim::DIM_4);
        if (elems == 0) continue;
        const bool flush = ++staged == batch_size || i == last;
        session.submitMode =
            i == last ? pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT : pto::comm::AsyncSubmitMode::DEFER;
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(desc.dst, desc.src, session);
        if (event.handle == 0 || event.postFailed) {
            // At pto_isa.pin, deferred validation and flush event construction can fail
            // only before the commit loop. Device interruption has no such return.
            error = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
            break;
        }
        if (flush) {
            if (!trim_urma_batch_event(event, session, initial)) {
                accounted = false;
                error = SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
                break;
            }
            published = event;
            staged = 0;
        }
    }
    // The latest confirmed event includes every earlier publication on this
    // session, even if a later attempt failed before publishing its tail.
    if (published.handle != 0 && !register_urma_async_event(runtime, published, session, workspace)) {
        accounted = false;
        return SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
    }
    return error;
#else
    (void)ctx;
    (void)last;
    (void)accounted;
    return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
#endif
}

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

// The calling AIV owns the request array. Nonempty PUTs name the registered windows of one Device domain.
// Successful return publishes every request; the scheduler observes completion.
template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool send_urma_put_batch(
    __gm__ int64_t *args, const __gm__ CommContext &domain, const UrmaRequestDescriptor<DstTensor, SrcTensor> *requests,
    uint32_t count, uint32_t batch_size
) {
    auto ctx = get_async_ctx(args);
    auto *global =
        reinterpret_cast<const __gm__ GlobalContext *>(static_cast<uintptr_t>(args[PAYLOAD_GLOBAL_CONTEXT_INDEX]));
    bool accounted = true;
    const int32_t error =
        simpler::urma::submit_urma_put_batch(ctx, global, domain, requests, count, batch_size, accounted);
    if (error != SIMPLER_ERROR_NONE) {
        pto2::detail::set_urma_error(ctx, accounted ? error | DEFERRED_ERROR_DMA_ACCOUNTED : error);
    }
    pto2::detail::defer_flush(ctx);
    return error == SIMPLER_ERROR_NONE;
}
