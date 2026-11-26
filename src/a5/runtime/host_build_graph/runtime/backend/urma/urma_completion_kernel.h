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
#include <pto/pto-inst.hpp>
#endif

#include "async_context.h"

#if defined(PTO_URMA_SUPPORTED)
#include "backend/urma/urma_completion.h"
#include "scheduler/wire_types.h"
#include "urma/urma_request.h"

namespace simpler::urma {

inline __aicore__ const __gm__ UrmaRunBinding *
registered_sender_domain(const AsyncCtx &ctx, uint64_t address, const __gm__ CommWorkspace *desc, uint32_t peer) {
    if (!async_ctx_is_deferred(ctx)) return nullptr;
    const auto *control = reinterpret_cast<const __gm__ simpler::percounter::PercounterControl *>(ctx.resident.image);
    const auto *domain = simpler::percounter::find_urma_binding(control->urma, address);
    if (!domain || (desc && (desc->addr != address || desc->size != domain->workspace_bytes)) ||
        peer >= domain->rank_count ||
        !simpler::percounter::urma_sender_valid(*domain, uint64_t{ctx.urma_sender_id} + 1))
        return nullptr;
    return simpler::percounter::urma_workspace_valid(*domain) ? domain : nullptr;
}

inline __aicore__ bool build_urma_session(
    const AsyncCtx &ctx, const __gm__ CommWorkspace &workspace, UrmaSession &session, uint32_t peer = 0
) {
    const auto *domain = registered_sender_domain(ctx, workspace.addr, &workspace, peer);
    if (!domain) {
        session.valid = false;
        session.contextGm = nullptr;
        session.domain_workspace = nullptr;
        return false;
    }
    return detail::bind_urma_session(ctx.urma_sender_id, domain->workspace, domain->workspace_bytes, session, peer);
}

class PercounterRequestAdapter {
public:
    explicit __aicore__ PercounterRequestAdapter(AsyncCtx &ctx) :
        ctx_(ctx) {}
    inline __aicore__ uint32_t sender_id() const { return ctx_.urma_sender_id; }

    template <typename Dst, typename Src>
    inline __aicore__ bool submit(
        UrmaRequestDescriptor<Dst, Src> desc, UrmaSession &session, bool reuse, const __gm__ CommWorkspace *workspace
    ) {
        if (desc.domain && workspace && (desc.domain->addr != workspace->addr || desc.domain->size != workspace->size))
            return invalid();
        auto *bound = reuse ? &session : nullptr;
        return select_domain(desc.workspace, desc.remote_rank, workspace ? workspace : desc.domain, bound) &&
               submit_chunked_urma_request(*this, desc, bound, domain_->workspace_bytes);
    }

    template <typename Remote, typename Local>
    inline __aicore__ bool submit(
        UrmaSegmentedRequest<Remote, Local> desc, UrmaSession &session, bool reuse,
        const __gm__ CommWorkspace *workspace
    ) {
        if (desc.remote_count != 1 || desc.remote == nullptr) return invalid();
        auto *root = reuse ? session.domain_workspace : reinterpret_cast<__gm__ uint8_t *>(workspace->addr);
        const uint32_t peer = load_urma_counter(reinterpret_cast<uint64_t>(&desc.remote->peer));
        if (!select_domain(root, peer, workspace, reuse ? &session : nullptr)) return false;
        if (!reuse &&
            !detail::bind_urma_session(sender_id(), domain_->workspace, domain_->workspace_bytes, session, peer))
            return invalid();
        return submit_segmented_request(*this, desc, session);
    }

    inline __aicore__ bool capacity_available(uint64_t required) const {
        if (required == 0 || !pto2::resident::detail::valid(ctx_.resident)) return false;
        const auto *group = pto2::resident::detail::group(ctx_.resident);
        if (reservation_.serial)
            return pto2::resident::detail::reservation_valid(ctx_.resident, reservation_) &&
                   required <= reservation_.capacity - group->reservation_written;
        if (group->reservation_limit != 0) return false;
        if (required > group->capacity - group->count)
            return pto2::resident::detail::fail(ctx_.resident, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        return true;
    }

    inline __aicore__ bool reserve(uint64_t required) {
        if (!capacity_available(required)) return false;
        reservation_ = pto2::resident::reserve_conditions(ctx_.resident, static_cast<uint32_t>(required));
        return reservation_.serial != 0;
    }

    inline __aicore__ bool register_target(const CompletionTarget &target) {
        const CompletionToken token{
            (uint64_t{target.queue_row} << 32) | target.target_bb, target.target_cqe, COMPLETION_ENGINE_URMA,
            COMPLETION_TYPE_URMA_EVENT_HANDLE, reinterpret_cast<uint64_t>(target.workspace)
        };
        const bool valid = simpler::percounter::urma_token_valid(token, *domain_, uint64_t{sender_id()} + 1);
        if (!pto2::resident::detail::commit_validated_condition(ctx_.resident, reservation_, token, valid))
            return false;
        ++written_;
        return true;
    }

    inline __aicore__ void set_error(RequestError error) {
        if (!error_)
            error_ = error == RequestError::INVALID ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID :
                                                      SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
        if (!reservation_.serial) pto2::detail::defer_error(ctx_, error_);
    }

    inline __aicore__ bool finish() {
        const bool ok = pto2::resident::finish_reservation(ctx_.resident, reservation_, written_);
        if (error_) pto2::detail::defer_error(ctx_, error_);
        return ok && !error_;
    }

private:
    inline __aicore__ bool invalid() {
        set_error(RequestError::INVALID);
        return false;
    }

    inline __aicore__ bool select_domain(
        __gm__ uint8_t *workspace, uint32_t peer, const __gm__ CommWorkspace *desc, const UrmaSession *bound
    ) {
        const uint64_t address = reinterpret_cast<uint64_t>(workspace);
        domain_ = registered_sender_domain(ctx_, address, desc, peer);
        if (!domain_ || peer == domain_->local_rank) return invalid();
        if (bound) {
            const uint64_t offset = static_cast<uint32_t>(get_block_idx()) == sender_id() ?
                                        0 :
                                        sender_view_offset(domain_->workspace_bytes, sender_id());
            if (bound->domain_workspace != workspace ||
                reinterpret_cast<uint64_t>(bound->contextGm) != address + offset)
                return invalid();
        }
        return true;
    }

    AsyncCtx &ctx_;
    const __gm__ UrmaRunBinding *domain_{};
    pto2::resident::AsyncReservation reservation_{};
    uint32_t written_{};
    int32_t error_{};
};

}  // namespace simpler::urma

namespace pto2::urma_backend {
using simpler::urma::peer_mr_base_addr;
using simpler::urma::peer_mr_ptr;
}  // namespace pto2::urma_backend

template <typename Dst, typename Src>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, UrmaRequestDescriptor<Dst, Src> desc) {
    simpler::urma::PercounterRequestAdapter runtime(ctx);
    simpler::urma::UrmaSession session;
    return runtime.submit(desc, session, false, nullptr);
}

template <typename Request>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, Request desc, simpler::urma::UrmaSession &session) {
    simpler::urma::PercounterRequestAdapter runtime(ctx);
    return runtime.submit(desc, session, true, nullptr);
}

template <typename Request>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, Request desc, const __gm__ CommWorkspace &workspace) {
    simpler::urma::PercounterRequestAdapter runtime(ctx);
    simpler::urma::UrmaSession session;
    return runtime.submit(desc, session, false, &workspace);
}
#endif
