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
 * Two-Phase Mesh AllReduce kernel — reduce-scatter + allgather with mesh barriers.
 *
 * Phase 1 (stage-in):   partition input → P chunk slots in window
 * Phase 2 (RS barrier): mesh barrier (all-to-all notify/wait)
 * Phase 3 (reduce):     acc[my_rank] = sum over peers of peer.scratch[my_rank]
 * Phase 4 (AG barrier): mesh barrier
 * Phase 5 (gather):     for r in P: read peer[r].scratch[r] → output[r*C]
 *
 * This is bandwidth-optimal for medium P: each rank only reduces its owned
 * chunk, then gathers all reduced chunks. Total remote data: 2*(P-1) chunks
 * of size N/P vs one-phase's (P-1) full vectors.
 *
 * args layout (same as onephase and ring):
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

AICORE inline void MeshBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_row + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        pto::comm::Signal sig(signal_row + peer);
        pto::comm::TWAIT(sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
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

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));

    // Signal rows: 2 rows (RS barrier + AG barrier), each with kMaxSupportedRanks slots.
    // Located after the nranks * chunk_elems float staging area.
    __gm__ int32_t *signal_rs = reinterpret_cast<__gm__ int32_t *>(scratch + nranks * chunk_elems);
    __gm__ int32_t *signal_ag = signal_rs + kMaxSupportedRanks;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    TileData chunkTile(1, chunk_elems);
    TileData accTile(1, chunk_elems);
    TileData recvTile(1, chunk_elems);
    TileData stageTile(1, ALLREDUCE_COUNT);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(accTile, 0x10000);
    TASSIGN(recvTile, 0x20000);
    TASSIGN(stageTile, 0x0);

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);
    ShapeDyn fullShape(1, 1, 1, 1, ALLREDUCE_COUNT);
    StrideDyn fullStride(ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, 1);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy local input into P contiguous chunk slots
    // in the HCCL window so all peers can read any chunk in Phase 3/5.
    // ------------------------------------------------------------------
    Global inputG(input, fullShape, fullStride);
    Global scratchG(scratch, fullShape, fullStride);
    TLOAD(stageTile, inputG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(scratchG, stageTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: RS barrier — wait until all ranks have staged their input.
    // ------------------------------------------------------------------
    MeshBarrier(commCtx, signal_rs, my_rank, nranks);

    // ------------------------------------------------------------------
    // Phase 3: reduce-scatter — each rank reduces chunk[my_rank] from
    // all peers into its local scratch[my_rank]. After this phase,
    // rank r owns the fully reduced chunk r.
    // ------------------------------------------------------------------
    Global myChunkG(scratch + my_rank * chunk_elems, chunkShape, chunkStride);
    TLOAD(accTile, myChunkG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) continue;
        __gm__ float *remote_chunk = CommRemotePtr(commCtx, scratch + my_rank * chunk_elems, peer);
        Global remoteG(remote_chunk, chunkShape, chunkStride);
        TLOAD(recvTile, remoteG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(accTile, accTile, recvTile);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // Write reduced chunk back to scratch for allgather phase.
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(myChunkG, accTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 4: AG barrier — wait until all ranks have completed their
    // reduce-scatter. After this, scratch[r] on rank r holds the fully
    // reduced chunk r.
    // ------------------------------------------------------------------
    MeshBarrier(commCtx, signal_ag, my_rank, nranks);

    // ------------------------------------------------------------------
    // Phase 5: allgather — each rank reads every peer's reduced chunk
    // and writes it to the corresponding output slice. Rank r reads
    // peer[p].scratch[p] and writes to output[p * chunk_elems].
    // ------------------------------------------------------------------
    for (int r = 0; r < nranks; ++r) {
        // Read chunk r from rank r (the owner of the reduced chunk r).
        __gm__ float *remote_chunk = CommRemotePtr(commCtx, scratch + r * chunk_elems, r);
        Global remoteG(remote_chunk, chunkShape, chunkStride);
        Global outputSlotG(output + r * chunk_elems, chunkShape, chunkStride);
        TLOAD(recvTile, remoteG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputSlotG, recvTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
