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
#include "../../../runtime/urma/urma_cq.h"
#include "platform_comm/urma_run_lease.h"

namespace simpler::urma {

template <typename Memory>
UrmaQueueState publish_urma_receipt(const Memory &memory, UrmaQueueReceipt &receipt) {
    if (!receipt.cq_published) {
        if (!memory.write(receipt.cq_tail, receipt.target_cqe)) return UrmaQueueState::FAILED;
        receipt.cq_published = true;
    }
    if (!receipt.doorbell_published) {
        const auto state = memory.ring_doorbell(receipt.cq_doorbell, receipt.target_cqe & 0xffffffu);
        if (state != UrmaQueueState::READY) return state;
        receipt.doorbell_published = true;
    }
    if (!memory.write(receipt.sq_tail, receipt.target_bb)) return UrmaQueueState::FAILED;
    receipt = {};
    return UrmaQueueState::READY;
}

// Domain close observes tails. Only a Run that proved chip stop and still owns these queues supplies a receipt.
template <typename Memory>
UrmaQueueState
check_urma_queues(const Memory &memory, uint64_t workspace, uint32_t rank, uint32_t ranks, UrmaQueueReceipt *receipt) {
    using namespace pto::comm::urma;
    if (receipt != nullptr && receipt->cq_tail != 0) {
        const auto state = publish_urma_receipt(memory, *receipt);
        if (state != UrmaQueueState::READY) return state;
    }
    UrmaInfo info{};
    if (!memory.read(workspace, info) || info.layout != UrmaLayout::PER_PEER || rank >= ranks ||
        info.rankCount != ranks || info.rowsPerPeer == 0 ||
        static_cast<uint64_t>(info.rankCount) * info.rowsPerPeer != info.jettyCount || info.sqPtr == 0 ||
        info.scqPtr == 0)
        return UrmaQueueState::FAILED;
    for (uint32_t row = 0; row < info.jettyCount; ++row) {
        if (row / info.rowsPerPeer == rank) continue;
        UrmaWQCtx sq{};
        UrmaCqCtx cq{};
        uint32_t submitted_bb, completed_bb, completed_cqe;
        if (!memory.read(info.sqPtr + sizeof(sq) * row, sq) || !memory.read(info.scqPtr + sizeof(cq) * row, cq) ||
            !memory.read(sq.headAddr, submitted_bb) || !memory.read(sq.tailAddr, completed_bb) ||
            !memory.read(cq.tailAddr, completed_cqe))
            return UrmaQueueState::FAILED;
        const uint32_t bb_count = submitted_bb - completed_bb;
        const uint32_t cqe_count = sq.submittedWqeCount - completed_cqe;
        if (bb_count == 0 && cqe_count == 0) continue;
        if (receipt == nullptr || !valid_completion_row(&sq, &cq) || cq.dbMode != UrmaDbMode::SW_DB || cqe_count == 0 ||
            cqe_count > bb_count || cqe_count > cq.depth || bb_count > sq.depth)
            return UrmaQueueState::FAILED;
        const auto checked = memory.inspect_prefix(cq.bufAddr, cq.depth, completed_cqe, sq.submittedWqeCount);
        if (checked != UrmaQueueState::READY) return checked;
        *receipt = {cq.tailAddr, cq.dbAddr, sq.tailAddr, sq.submittedWqeCount, submitted_bb};
        const auto published = publish_urma_receipt(memory, *receipt);
        if (published != UrmaQueueState::READY) return published;
    }
    return UrmaQueueState::READY;
}

}  // namespace simpler::urma
