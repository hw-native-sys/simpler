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

#include "async_condition_writer.h"
#include <pto/comm/async/urma/urma_types.hpp>
#if defined(__CCE_AICORE__)
#include <pto/pto-inst.hpp>
#include <pto/comm/async/urma/urma_async_intrin.hpp>
#endif

namespace pto2::urma_backend {

// The sender AIV is also the only consumer of its queues, across all groups.
// Returning PENDING on other workers leaves the group available to its owner.
template <typename Poll>
inline __aicore__ CompletionPollResult
poll_owned_queue(const CompletionToken &token, uint32_t observer_sender, const Poll &poll) {
    const CompletionPollResult invalid{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (token.engine != COMPLETION_ENGINE_URMA || !resident::detail::token_valid(token)) return invalid;
    if (observer_sender == UINT32_MAX) return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    using namespace pto::comm::urma;
    auto *info = reinterpret_cast<__gm__ UrmaInfo *>(token.backend_cookie);
    const uint32_t row = static_cast<uint32_t>(token.addr >> 32);
    if (info->layout != UrmaLayout::PER_PEER || info->rankCount != 2 || info->rowsPerPeer == 0 ||
        info->jettiesPerCore != 1 || static_cast<uint64_t>(info->rowsPerPeer) * info->rankCount != info->jettyCount ||
        row >= info->jettyCount || info->sqPtr == 0 || info->scqPtr == 0 || info->sqPtr % alignof(UrmaWQCtx) != 0 ||
        info->scqPtr % alignof(UrmaCqCtx) != 0)
        return invalid;
    const uint32_t peer = row / info->rowsPerPeer;
    const uint32_t jetty = row % info->rowsPerPeer;
    if (observer_sender != jetty) return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    auto *wq = reinterpret_cast<__gm__ UrmaWQCtx *>(info->sqPtr) + row;
    auto *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(info->scqPtr) + row;
    const auto power_of_two = [](uint32_t n) __aicore__ {
        return n && !(n & (n - 1)) && n < (1u << 31);
    };
    if (!power_of_two(wq->depth) || !power_of_two(cq->depth) || cq->cqeShiftSize != 6 || cq->bufAddr == 0 ||
        cq->bufAddr % 64 != 0 || cq->tailAddr == 0 || cq->dbAddr == 0 || wq->headAddr == 0 || wq->tailAddr == 0 ||
        wq->dbAddr == 0 || (cq->tailAddr | cq->dbAddr | wq->headAddr | wq->tailAddr | wq->dbAddr) % 4 != 0)
        return invalid;
    return poll(info, peer, jetty, static_cast<uint32_t>(token.addr), token.expected_value);
}

struct AicoreCqPoll {
    inline __aicore__ CompletionPollResult
    operator()(__gm__ pto::comm::urma::UrmaInfo *info, uint32_t peer, uint32_t jetty, uint32_t bb, uint32_t cqe) const {
#if defined(__CCE_AICORE__) && defined(PTO_URMA_SUPPORTED)
        using namespace pto::comm::urma;
        auto *wq = reinterpret_cast<__gm__ UrmaWQCtx *>(info->sqPtr) + peer * info->rowsPerPeer + jetty;
        auto *cq = reinterpret_cast<__gm__ UrmaCqCtx *>(info->scqPtr) + peer * info->rowsPerPeer + jetty;
        const uint32_t cq_tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(cq->tailAddr), 0);
        const uint32_t sq_tail = ld_dev(reinterpret_cast<__gm__ uint32_t *>(wq->tailAddr), 0);
        if (!pto::comm::urma::detail::SequenceReached(cq_tail, cqe) &&
            (cqe - cq_tail > cq->depth || bb - sq_tail > wq->depth))
            return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        pto::comm::AsyncSession session;
        session.valid = true;
        session.engine = pto::comm::DmaEngine::URMA;
        session.contextGm = reinterpret_cast<__gm__ uint8_t *>(info);
        session.destRankId = peer;
        session.qpIdxBase = jetty;
        session.cfg.urma.qpCount = 1;
        bool completed = false;
        if (!pto::comm::urma::detail::UrmaPollCqAt(session, peer, jetty, bb, cqe, false, completed))
            return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
        return {completed ? CompletionPollState::READY : CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
#else
        (void)info;
        (void)peer;
        (void)jetty;
        (void)bb;
        (void)cqe;
        return {CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
#endif
    }
};
}  // namespace pto2::urma_backend
