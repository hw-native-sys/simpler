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
 * AllReduce kernel — symmetric, 4-phase, HCCL-window scratch pattern.
 *
 * Phase 1 (stage-in):   input → my scratch slot (in window)
 * Phase 2 (barrier):    signal matrix + TWAIT cross-rank sync
 * Phase 3 (compute):    for peer in nranks: TLOAD(peer_scratch), TADD(acc)
 * Phase 4 (stage-out):  TSTORE(output, acc)
 *
 * input / output are per-rank host tensors passed through TaskArgs (the
 * runtime handles the H2D / D2H).  scratch is the single HCCL-window
 * buffer, shared across ranks for cross-rank reads.  The signal area
 * lives at the tail of scratch: nranks int32 slots where peer r writes a
 * counter and my_rank waits on slot[r] before reading.
 *
 * args layout (passed as ContinuousTensor arg slots — see allreduce_orch.cpp):
 *   tensor(0) = input    (host-backed, framework-supplied device addr)
 *   tensor(1) = output   (host-backed, framework-supplied device addr)
 *   tensor(2) = scratch  (HCCL window slot, cross-rank addressable)
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

static constexpr size_t ALLREDUCE_COUNT = 256;
static constexpr int kMaxSupportedRanks = 16;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

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
    // Signal area sits at the tail of the scratch buffer: nranks int32 slots.
    // Peer r writes into my_rank's signal[r] when its stage-in is done.
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + ALLREDUCE_COUNT);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, ALLREDUCE_COUNT);
    StrideDyn stride(ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, 1);

    TileData stageTile(1, ALLREDUCE_COUNT);
    TileData accTile(1, ALLREDUCE_COUNT);
    TileData recvTile(1, ALLREDUCE_COUNT);
    TASSIGN(stageTile, 0x0);
    TASSIGN(accTile, 0x10000);
    TASSIGN(recvTile, 0x20000);

    Global inputG(input, shape, stride);
    Global scratchG(scratch, shape, stride);
    Global outputG(output, shape, stride);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy local input (device mem) into my scratch
    // slot (HCCL window), so peers can TLOAD it in Phase 3.
    // ------------------------------------------------------------------
    TLOAD(stageTile, inputG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(scratchG, stageTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: device barrier — each rank notifies every peer that its
    // stage-in is visible, then waits until every peer has notified us.
    // After this point the scratch data on all ranks is readable.
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
    // Phase 3: compute — sum every rank's scratch slot into accTile.
    // Start from my local scratch (no remote pointer needed), then add
    // peers via CommRemotePtr.
    // ------------------------------------------------------------------
    TLOAD(accTile, scratchG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ float *remote_scratch = CommRemotePtr(commCtx, scratch, peer);
        Global remoteG(remote_scratch, shape, stride);
        TLOAD(recvTile, remoteG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(accTile, accTile, recvTile);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // ------------------------------------------------------------------
    // Phase 4: stage-out — write the reduced accumulator into the local
    // output (plain device mem), no remote traffic involved.
    // ------------------------------------------------------------------
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputG, accTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    pipe_barrier(PIPE_ALL);
}
