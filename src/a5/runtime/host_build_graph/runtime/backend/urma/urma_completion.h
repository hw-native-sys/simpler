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

#include <pto/comm/async/urma/urma_types.hpp>

#include "host_build_graph/completion_token.h"
#include "host_build_graph/completion_types.h"
#include "platform_comm/urma_run_binding.h"
#include "scheduler/gm.h"
#include "../../../../urma/urma_cq.h"

namespace simpler::percounter {

inline __aicore__ const __gm__ UrmaRunBinding *
find_urma_binding(const __gm__ UrmaRunBindings &bindings, uint64_t workspace) {
    if (workspace == 0 || bindings.address == 0) return nullptr;
    const auto *domains = reinterpret_cast<const __gm__ UrmaRunBinding *>(bindings.address);
    for (uint32_t i = 0; i < bindings.count; ++i) {
        if (domains[i].workspace == workspace) return &domains[i];
    }
    return nullptr;
}

inline __aicore__ bool urma_sender_valid(const __gm__ UrmaRunBinding &domain, uint64_t sender_plus_one) {
    return domain.workspace != 0 && domain.row_owner != 0 && domain.local_rank < domain.rank_count &&
           domain.jetties_per_core != 0 && domain.rows_per_peer % domain.jetties_per_core == 0 &&
           sender_plus_one != 0 && sender_plus_one <= domain.rows_per_peer / domain.jetties_per_core;
}

inline __aicore__ bool
urma_sender_can_target(const __gm__ UrmaRunBinding &domain, uint32_t peer, uint64_t sender_plus_one) {
    return peer < domain.rank_count && peer != domain.local_rank && urma_sender_valid(domain, sender_plus_one);
}

inline __aicore__ bool urma_workspace_valid(const __gm__ UrmaRunBinding &domain) {
    using namespace pto::comm::urma;
    if (domain.workspace == 0 || domain.workspace_bytes < sizeof(UrmaInfo)) return false;
    const auto *info = reinterpret_cast<const __gm__ UrmaInfo *>(domain.workspace);
    return info->layout == UrmaLayout::PER_PEER && info->rankCount == domain.rank_count &&
           info->rowsPerPeer == domain.rows_per_peer && info->jettiesPerCore == domain.jetties_per_core &&
           uint64_t{info->rowsPerPeer} * info->rankCount == info->jettyCount && info->sqPtr != 0 && info->scqPtr != 0;
}

inline __aicore__ bool
urma_token_valid(const CompletionToken &token, const __gm__ UrmaRunBinding &domain, uint64_t sender_plus_one) {
    if (token.engine != COMPLETION_ENGINE_URMA || token.completion_type != COMPLETION_TYPE_URMA_EVENT_HANDLE ||
        token.backend_cookie != domain.workspace || domain.rows_per_peer == 0)
        return false;
    const uint32_t row = static_cast<uint32_t>(token.addr >> 32);
    return urma_sender_can_target(domain, row / domain.rows_per_peer, sender_plus_one) &&
           (row % domain.rows_per_peer) / domain.jetties_per_core == sender_plus_one - 1;
}

struct CoreCqAccess {
    static inline __aicore__ uint32_t load(uint64_t address) {
        return percounter_gm_read(*reinterpret_cast<__gm__ uint32_t *>(address));
    }

    static inline __aicore__ void store(uint64_t address, uint32_t value) {
        auto *ptr = reinterpret_cast<__gm__ uint32_t *>(address);
#if defined(__CCE_AICORE__)
        __builtin_cce_st_dev(value, ptr, 0);
        dsb(DSB_DDR);
#else
        __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif
    }
};

inline __aicore__ CompletionPollResult
poll_urma(const CompletionToken &token, const __gm__ UrmaRunBinding &domain, uint64_t sender_plus_one) {
    using namespace pto::comm::urma;
    const CompletionPollResult invalid{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (!urma_token_valid(token, domain, sender_plus_one) || !urma_workspace_valid(domain)) return invalid;
    const auto *info = reinterpret_cast<const __gm__ UrmaInfo *>(domain.workspace);
    const uint32_t row = static_cast<uint32_t>(token.addr >> 32);
    auto &owner = reinterpret_cast<__gm__ UrmaRowOwner *>(domain.row_owner)[row].value;
    // Different completion records can cover the same row. A busy observer yields to the calendar.
    if (percounter_gm_compare_exchange(owner, uint64_t{0}, uint64_t{1}) != 0)
        return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    percounter_memory_barrier();
    const auto *sq = reinterpret_cast<const __gm__ UrmaWQCtx *>(info->sqPtr) + row;
    const auto *cq = reinterpret_cast<const __gm__ UrmaCqCtx *>(info->scqPtr) + row;
    const auto result = simpler::urma::poll_completion_row<CoreCqAccess, CompletionPollState>(
        sq, cq, static_cast<uint32_t>(token.addr), token.expected_value
    );
    percounter_memory_barrier();
    percounter_gm_store(owner, uint64_t{0});
    return {
        result, result == CompletionPollState::FAILED ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID : SIMPLER_ERROR_NONE
    };
}

}  // namespace simpler::percounter
