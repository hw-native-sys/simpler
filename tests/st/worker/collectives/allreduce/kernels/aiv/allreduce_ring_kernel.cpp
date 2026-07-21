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
 * Ring AllReduce kernel — chunked reduce-scatter + allgather, HCCL-window scratch.
 *
 * Phase 1 (stage-in):       partition input → P chunk slots in window
 * Phase 2 (reduce-scatter): (P-1) ring steps; rank r owns reduced chunk r
 * Phase 3 (allgather):      (P-1) ring steps; collect all reduced chunks
 * Phase 4 (stage-out):      chunks → output
 *
 * input / output are per-rank host tensors passed through TaskArgs (the
 * runtime handles the H2D / D2H).  scratch is the HCCL-window buffer:
 *   [0 .. P*chunk)           P chunk slots owned by this rank
 *   tail                     2*(P-1)*kMaxSupportedRanks int32 barrier slots
 *                            (fresh-window zero init)
 *
 * After each per-round barrier, peers read directly from each other's
 * chunks[] slots via CommRemotePtr — no separate exchange publish buffer.
 * The ``pipe_barrier(PIPE_ALL)`` at the end of each RS/AG step body drains
 * MTE pipes before ``NeighborBarrier`` fires ``TNOTIFY`` (hand-written
 * equivalent of PTOAS v0.45's automatic ``emitTNotifyMteDrain``).
 *
 * args layout (passed as Tensor arg slots — see allreduce_ring_orch.cpp):
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

// Per-round neighbor barrier for ring topology.
// Rank r TLOADs (reads) from left = (r-1)%P; right = (r+1)%P TLOADs from r.
// Each rank notifies its right neighbor (who reads from this rank's chunks)
// and waits on its left neighbor (whose chunks this rank reads from).
// Signal rows used exactly once (AtomicAdd 0→1, TWAIT GE 1) —
// matches RoundBarrier zero-init semantics (fresh HCCL window).
AICORE inline void
NeighborBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int left, int nranks) {
    int right = (my_rank + 1) % nranks;
    // Notify right neighbor: "I'm done with this step, you can TLOAD from my chunks."
    __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_row + my_rank, right);
    pto::comm::Signal sig_out(remote_signal);
    pto::comm::TNOTIFY(sig_out, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    // Wait on left neighbor: "Are you done? I need to TLOAD your chunks."
    pto::comm::Signal sig_in(signal_row + left);
    pto::comm::TWAIT(sig_in, (int32_t)1, pto::comm::WaitCmp::GE);
    pipe_barrier(PIPE_ALL);
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

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));
    __gm__ float *chunks = scratch;
    // Signal rows after float region: 2*(P-1) rounds, kMaxSupportedRanks stride.
    __gm__ int32_t *signal_base =
        reinterpret_cast<__gm__ int32_t *>(scratch + static_cast<size_t>(nranks * chunk_elems));

    TileData chunkTile(1, chunk_elems);
    TileData recvTile(1, chunk_elems);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — partition local input into P chunk slots in the
    // HCCL window so peers can TLOAD them in later ring steps.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        __gm__ float *dst = chunks + static_cast<size_t>(chunk * chunk_elems);
        __gm__ float *src = input + static_cast<size_t>(chunk * chunk_elems);
        Global srcG(src, chunkShape, chunkStride);
        Global dstG(dst, chunkShape, chunkStride);
        TLOAD(chunkTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, chunkTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);

    int round = 0;

    // ------------------------------------------------------------------
    // Phase 2: reduce-scatter — (P-1) ring steps; rank r ends with fully
    // reduced chunk r.  Barrier, then TLOAD left neighbour's chunks[] slot.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int recv_add_idx = (my_rank - step - 1 + nranks) % nranks;
        const int left = (my_rank - 1 + nranks) % nranks;

        NeighborBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, left, nranks);
        ++round;

        const int left_send_idx = (left - step + nranks) % nranks;
        {
            __gm__ float *remote_chunk =
                CommRemotePtr(commCtx, chunks + static_cast<size_t>(left_send_idx * chunk_elems), left);
            Global remoteG(remote_chunk, chunkShape, chunkStride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        Global accG(chunks + static_cast<size_t>(recv_add_idx * chunk_elems), chunkShape, chunkStride);
        TLOAD(chunkTile, accG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(chunkTile, chunkTile, recvTile);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(accG, chunkTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 3: allgather — (P-1) ring steps; every rank collects all
    // reduced chunks from left neighbour chunks[] after each barrier.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int recv_idx = (my_rank - step + nranks) % nranks;
        const int left = (my_rank - 1 + nranks) % nranks;

        NeighborBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, left, nranks);
        ++round;

        const int left_send_idx = (left - step + 1 + nranks) % nranks;
        {
            __gm__ float *remote_chunk =
                CommRemotePtr(commCtx, chunks + static_cast<size_t>(left_send_idx * chunk_elems), left);
            Global remoteG(remote_chunk, chunkShape, chunkStride);
            TLOAD(recvTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        Global dstG(chunks + static_cast<size_t>(recv_idx * chunk_elems), chunkShape, chunkStride);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, recvTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 4: stage-out — write concatenated chunks into local output.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        __gm__ float *dst = output + static_cast<size_t>(chunk * chunk_elems);
        __gm__ float *src = chunks + static_cast<size_t>(chunk * chunk_elems);
        Global srcG(src, chunkShape, chunkStride);
        Global dstG(dst, chunkShape, chunkStride);
        TLOAD(chunkTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, chunkTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
