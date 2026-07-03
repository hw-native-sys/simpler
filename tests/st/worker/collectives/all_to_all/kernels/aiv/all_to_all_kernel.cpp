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
/**
 * AllToAll kernel — symmetric, 2-phase push-based pattern.
 *
 * Phase 1 (push):     for dest in 0..nranks-1:
 *                       TPUT input[dest*C..) → peer dest's scratch[my_rank*C..)
 *                     Self-rank handled naturally: CommRemotePtr to self
 *                     returns the local pointer, so TPUT degenerates to a
 *                     local GM write.
 * Phase 2 (barrier):  signal matrix + TWAIT cross-rank sync.
 *                     No post-exchange barrier needed — input is read-only
 *                     and the scratch is write-only from peers, so there is
 *                     no WAR hazard on the scratch.
 * Phase 3 (copy-out): for src in 0..nranks-1:
 *                       TLOAD(local scratch[src*C..)) → TSTORE(output[src*C..))
 *                     Purely local copy — the scratch already holds the
 *                     result after the barrier.
 *
 * Scratch at offset src*C holds the chunk that rank src pushed here.
 *
 * args layout:
 *   tensor(0) = input    nranks*COUNT_PER_RANK floats  (INPUT)
 *   tensor(1) = output   nranks*COUNT_PER_RANK floats  (OUTPUT_EXISTING)
 *   tensor(2) = scratch  HCCL window slot              (INOUT)
 *   scalar(0) = nranks
 *   scalar(1) = CommContext device pointer
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

static constexpr size_t COUNT_PER_RANK = 64;
static constexpr int kMaxSupportedRanks = 16;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int nranks = static_cast<int>(args[3]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *scratch =
        reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, COUNT_PER_RANK, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // signal_base follows the nranks * COUNT_PER_RANK float data region.
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + nranks * COUNT_PER_RANK);

    ShapeDyn shape(1, 1, 1, 1, COUNT_PER_RANK);
    StrideDyn stride(COUNT_PER_RANK, COUNT_PER_RANK, COUNT_PER_RANK, COUNT_PER_RANK, 1);

    // Single push tile — reused for both Phase 1 push and Phase 3 copy-out.
    TileData pushTile(1, COUNT_PER_RANK);
    TASSIGN(pushTile, 0x0);

    // ------------------------------------------------------------------
    // Phase 1: push — TPUT each destination chunk directly into the
    // corresponding peer's scratch at offset my_rank*C. When dest == my_rank
    // CommRemotePtr returns the local pointer, so TPUT is a local GM write.
    // ------------------------------------------------------------------
    __gm__ float *scratch_dst_local = scratch + my_rank * COUNT_PER_RANK;
    for (int dest = 0; dest < nranks; ++dest) {
        __gm__ float *scratch_dst_remote = CommRemotePtr(commCtx, scratch_dst_local, dest);

        Global inputChunkG(input + dest * COUNT_PER_RANK, shape, stride);
        Global scratchChunkG(scratch_dst_remote, shape, stride);

        pto::comm::TPUT(scratchChunkG, inputChunkG, pushTile);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier — notify every peer that our pushes are done,
    // then wait until every peer has notified us.
    // ------------------------------------------------------------------
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(commCtx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 3: copy-out — every peer has pushed into our scratch at their
    // rank's offset. Copy each slot into the output tensor. Purely local
    // reads — no CommRemotePtr needed here.
    // ------------------------------------------------------------------
    for (int src = 0; src < nranks; ++src) {
        Global scratchSlotG(scratch + src * COUNT_PER_RANK, shape, stride);
        Global outputSlotG(output + src * COUNT_PER_RANK, shape, stride);
        TLOAD(pushTile, scratchSlotG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputSlotG, pushTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
