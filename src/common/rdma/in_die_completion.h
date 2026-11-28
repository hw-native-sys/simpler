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

#include <cstddef>
#include <cstdint>

#include <pto/comm/async/rdma/backends/in_die/in_die_types.hpp>
#include <pto/comm/async/rdma/rdma_workspace_types.hpp>

namespace simpler::rdma {

enum class InDiePollState { Pending, Ready, Failed };

struct InDiePollResult {
    InDiePollState state;
    uint32_t error;
};

inline constexpr uint32_t kInvalidInDieCompletion = 0x22001;

struct InDieCompletionCursor {
    uint64_t workspace = 0;
    uint64_t owner_epoch = 0;
    uint64_t tail = 0;
    uint64_t head = 0;
    uint64_t queue_error = 0;
    uint64_t buffer = 0;
    uint32_t queue_index = 0;
    uint32_t depth = 0;
    uint32_t prefix = 0;
    uint32_t failed_at = 0;
    uint32_t error = 0;
};

inline bool in_die_span(uint64_t address, uint64_t bytes, uint64_t alignment) {
    return address != 0 && address % alignment == 0 && bytes != 0 && address <= UINT64_MAX - bytes;
}

template <typename Reader>
inline bool init_in_die_cursor(
    uint64_t workspace, uint32_t queue_index, uint64_t epoch, InDieCompletionCursor &cursor, Reader &reader
) {
    using pto::comm::rdma::RdmaInfo;
    using pto::comm::rdma::in_die::Queue;
    using pto::comm::rdma::in_die::Ring;
    if (!in_die_span(workspace, sizeof(RdmaInfo), alignof(RdmaInfo)) || epoch == 0) return false;
    const uint32_t magic = reader.read32(workspace + offsetof(RdmaInfo, magic));
    const uint32_t version = reader.read32(workspace + offsetof(RdmaInfo, version));
    const uint32_t backend = reader.read32(workspace + offsetof(RdmaInfo, backend));
    const uint32_t ranks = reader.read32(workspace + offsetof(RdmaInfo, rankCount));
    const uint32_t qps = reader.read32(workspace + offsetof(RdmaInfo, qpNum));
    if (magic != pto::comm::rdma::kRdmaWorkspaceMagic || version != pto::comm::rdma::kRdmaWorkspaceVersion ||
        backend != static_cast<uint32_t>(pto::comm::RdmaBackend::IN_DIE) || ranks == 0 || ranks > 64 || qps == 0 ||
        qps > 8 || queue_index >= ranks * qps)
        return false;
    const uint64_t queues = reader.read64(workspace + offsetof(RdmaInfo, sqPtr));
    const uint64_t cq_queues = reader.read64(workspace + offsetof(RdmaInfo, scqPtr));
    const uint64_t queue_bytes = uint64_t(ranks) * qps * sizeof(Queue);
    if (!in_die_span(workspace, sizeof(RdmaInfo) + queue_bytes, alignof(RdmaInfo)) ||
        queues != workspace + sizeof(RdmaInfo) || cq_queues != queues ||
        reader.read64(workspace + offsetof(RdmaInfo, memPtr)) != queues + queue_bytes)
        return false;
    const uint64_t queue = queues + uint64_t(queue_index) * sizeof(Queue);
    const uint64_t cq = queue + offsetof(Queue, cq);
    const uint64_t sq = queue + offsetof(Queue, sq);
    const uint64_t tail = reader.read64(cq + offsetof(Ring, tail));
    const uint64_t head = reader.read64(sq + offsetof(Ring, head));
    const uint64_t buffer = reader.read64(cq + offsetof(Ring, buffer));
    const uint32_t depth = reader.read32(cq + offsetof(Ring, depth));
    if (depth < 2 || depth >= (uint32_t{1} << 31) || (depth & (depth - 1)) != 0 ||
        reader.read32(cq + offsetof(Ring, stride)) != 32 || !in_die_span(buffer, uint64_t(depth) * 32, 4) ||
        !in_die_span(tail, sizeof(uint32_t), 4) || !in_die_span(head, sizeof(uint32_t), 4))
        return false;
    cursor = {};
    cursor.workspace = workspace;
    cursor.owner_epoch = epoch;
    cursor.queue_index = queue_index;
    cursor.tail = tail;
    cursor.head = head;
    cursor.queue_error = queue + offsetof(Queue, error);
    cursor.buffer = buffer;
    cursor.depth = depth;
    return true;
}

// Reader only exposes acquire loads. The provider alone advances CQ/SQ tails and doorbells.
template <typename Reader>
inline InDiePollResult poll_in_die_completion(
    uint64_t handle, uint64_t workspace, uint64_t epoch, InDieCompletionCursor &cursor, Reader &reader
) {
    const uint32_t queue = static_cast<uint32_t>(handle >> 32);
    const uint32_t target = static_cast<uint32_t>(handle);
    if (queue == UINT32_MAX || target == 0 ||
        ((cursor.depth == 0 || cursor.workspace != workspace || cursor.owner_epoch != epoch ||
          cursor.queue_index != queue) &&
         !init_in_die_cursor(workspace, queue, epoch, cursor, reader))) {
        return {InDiePollState::Failed, kInvalidInDieCompletion};
    }
    if (cursor.error != 0 && cursor.failed_at < target) return {InDiePollState::Failed, cursor.error};
    uint32_t tail = reader.read32(cursor.tail);
    const uint32_t head = reader.read32(cursor.head);
    if (head < target || tail > head) return {InDiePollState::Failed, kInvalidInDieCompletion};
    if (tail >= target || cursor.prefix >= target) return {InDiePollState::Ready, 0};
    uint32_t position = tail > cursor.prefix ? tail : cursor.prefix;
    for (uint32_t scanned = 0; scanned < 32 && position < target; ++scanned) {
        const uint64_t entry = cursor.buffer + uint64_t(position & (cursor.depth - 1)) * 32;
        const uint32_t word = reader.read32(entry);
        tail = reader.read32(cursor.tail);
        if (tail > position) {
            position = tail;
            continue;
        }
        const uint32_t owner = (position & cursor.depth) == 0 ? 1U : 0U;
        if (((word >> 7) & 1U) != owner) break;
        const uint32_t status = (word >> 8) & 0xffU;
        if (status != 0) {
            cursor.error = status;
            cursor.failed_at = position;
            return {InDiePollState::Failed, status};
        }
        ++position;
    }
    cursor.prefix = position;
    if (position >= target) return {InDiePollState::Ready, 0};
    const uint32_t error = reader.read32(cursor.queue_error);
    if (error != 0) {
        tail = reader.read32(cursor.tail);
        if (tail >= target) return {InDiePollState::Ready, 0};
        cursor.error = error;
        cursor.failed_at = tail > cursor.prefix ? tail : cursor.prefix;
        return {InDiePollState::Failed, error};
    }
    return {InDiePollState::Pending, 0};
}

}  // namespace simpler::rdma
