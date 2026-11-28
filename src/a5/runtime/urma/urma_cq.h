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

#include <cstdint>

#if defined(__CCE_AICORE__)
#define SIMPLER_URMA_DEVICE [aicore]
#else
#define SIMPLER_URMA_DEVICE
#endif

namespace simpler::urma {

SIMPLER_URMA_DEVICE inline bool cq_sequence_reached(uint32_t current, uint32_t target) {
    return static_cast<int32_t>(current - target) >= 0;
}

template <typename SendQueue, typename CompletionQueue>
SIMPLER_URMA_DEVICE inline bool valid_completion_row(SendQueue wq, CompletionQueue cq) {
    const uint32_t cq_depth = cq->depth;
    const uint32_t sq_depth = wq->depth;
    return cq->bufAddr != 0 && cq->tailAddr != 0 && cq->dbAddr != 0 && wq->tailAddr != 0 && cq->cqeShiftSize == 6 &&
           cq_depth != 0 && (cq_depth & (cq_depth - 1)) == 0 && cq_depth < (1u << 31) && sq_depth != 0 &&
           (sq_depth & (sq_depth - 1)) == 0 && sq_depth < (1u << 31);
}

template <typename PollState, typename Memory>
SIMPLER_URMA_DEVICE inline PollState
inspect_cq_entries(const Memory &memory, uint64_t buffer, uint32_t depth, uint32_t completed, uint32_t target) {
    for (uint32_t seq = completed; seq != target; ++seq) {
        const uint64_t address = buffer + 64 * static_cast<uint64_t>(seq & (depth - 1));
        uint32_t status;
        if (!memory.read(address, status)) return PollState::FAILED;
        if (((status >> 2) & 1u) == ((seq / depth) & 1u)) return PollState::PENDING;
        if ((status >> 16) != 0) return PollState::FAILED;
    }
    return PollState::READY;
}

template <typename Access>
struct DirectCqReader {
    SIMPLER_URMA_DEVICE static bool read(uint64_t address, uint32_t &value) {
        value = Access::load(address);
        return true;
    }
};

// The caller serializes consumers of this row. Only the submitting AIV writes the send doorbell.
template <typename Access, typename PollState, typename SendQueue, typename CompletionQueue>
SIMPLER_URMA_DEVICE inline PollState
poll_completion_row(SendQueue wq, CompletionQueue cq, uint32_t target_bb, uint32_t target_cqe) {
    if (!valid_completion_row(wq, cq)) return PollState::FAILED;
    const uint32_t completed_cqe = Access::load(cq->tailAddr);
    const uint32_t completed_bb = Access::load(wq->tailAddr);
    if (cq_sequence_reached(completed_cqe, target_cqe)) {
        return cq_sequence_reached(completed_bb, target_bb) ? PollState::READY : PollState::PENDING;
    }
    if (target_cqe - completed_cqe > cq->depth || target_bb - completed_bb > wq->depth) return PollState::FAILED;
    const auto state =
        inspect_cq_entries<PollState>(DirectCqReader<Access>{}, cq->bufAddr, cq->depth, completed_cqe, target_cqe);
    if (state != PollState::READY) return state;
    Access::store(cq->tailAddr, target_cqe);
    Access::store(cq->dbAddr, target_cqe & 0xffffffu);
    Access::store(wq->tailAddr, target_bb);
    return PollState::READY;
}

}  // namespace simpler::urma

#undef SIMPLER_URMA_DEVICE
