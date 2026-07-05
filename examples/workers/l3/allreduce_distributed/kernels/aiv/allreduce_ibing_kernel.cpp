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
 * IBing Interleaved Bidirectional Ring AllReduce — faithful implementation
 * of the algorithm by Zong et al., ACM TACO 2025.
 *
 * Unlike the two-ring bidirectional_ring variant (which runs separate RS
 * and AG phases, 2(P-1)+1 barriers), this kernel implements the true
 * IBing interleaved schedule: P-1 bidirectional rounds, each round
 * pushing one chunk clockwise and one counter-clockwise, with the
 * first ⌊P/2⌋ steps using AtomicAdd (reduce-scatter) and the remaining
 * steps using AtomicNone (allgather-forward).
 *
 * Right-bound:  r pushes chunks[(r - s + P) % P]         → rank r+1
 * Left-bound:   r pushes chunks[(r + s + 1 + P) % P]    → rank r-1
 *
 * On CPU sim (POSIX shared memory): exchange buffers snapshot source
 * chunks before pushing, avoiding intra-round read/write races.
 * On NPU hardware (HCCL window): direct TPUT<AtomicAdd/AtomicNone>.
 *
 * Scratch layout (per rank, in HCCL window):
 *   [0 .. P*chunk_elems)                   P working chunks
 *   [P*chunk .. (P+1)*chunk)               exchange_right (CPU sim snapshot)
 *   [(P+1)*chunk .. (P+2)*chunk)          exchange_left  (CPU sim snapshot)
 *   tail                   (2*(P-1)+1) * kMaxSupportedRanks int32 signals
 *
 * Divisibility: requires ALLREDUCE_COUNT % nranks == 0.
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

AICORE inline void RoundBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
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

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GT = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));
    __gm__ float *chunks = scratch;
    __gm__ float *exchange_right = scratch + static_cast<size_t>(nranks * chunk_elems);
    __gm__ float *exchange_left = exchange_right + static_cast<size_t>(chunk_elems);
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(exchange_left + static_cast<size_t>(chunk_elems));

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

#ifndef __CPU_SIM
    using TilePush = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;
    TilePush pushTile(1, chunk_elems);
    TASSIGN(pushTile, 0x10000);
#endif

    using TileStage = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;
    TileStage stageTile(1, chunk_elems);
    TASSIGN(stageTile, 0x0);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — copy P chunks from input to scratch.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        __gm__ float *dst = chunks + static_cast<size_t>(c * chunk_elems);
        __gm__ float *src = input + static_cast<size_t>(c * chunk_elems);
        GT srcG(src, chunkShape, chunkStride);
        GT dstG(dst, chunkShape, chunkStride);
        TLOAD(stageTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, stageTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);

    // ------------------------------------------------------------------
    // Phase 2: IBing interleaved RS+AG — P−1 rounds.
    //
    // In each step s ∈ [1, P−1]:
    //   - Push chunk (r - s + P) % P to right neighbour.
    //   - Push chunk (r + s + 1 + P) % P to left neighbour.
    //
    //   Steps 1..⌊P/2⌋:     reduce  — AtomicAdd pushes from exchange buffers.
    //   Steps ⌊P/2⌋+1..P−1: forward — AtomicNone pushes (allgather).
    //
    // CPU sim uses a double-barrier scheme:
    //   1. Barrier A: ensure prior writes are globally visible.
    //   2. Snapshot:  copy src chunks → exchange buffers.
    //   3. Barrier B: ensure all snapshots complete.
    //   4. Push:      write exchange buffers → neighbour's chunks[].
    //
    // This replicates the MPI exchange-buffer semantics of the original
    // IBing paper in shared memory.  On NPU hardware the snapshots and
    // second barrier are unnecessary (HCCL provides ordering guarantees).
    // ------------------------------------------------------------------
    const int left = (my_rank - 1 + nranks) % nranks;
    const int right = (my_rank + 1) % nranks;
    const int reduce_steps = nranks / 2;

    for (int step = 1; step < nranks; ++step) {
#ifdef __CPU_SIM
        // Barrier A: all prior writes globally visible.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1)) * kMaxSupportedRanks, my_rank, nranks);
#else
        RoundBarrier(commCtx, signal_base + (step - 1) * kMaxSupportedRanks, my_rank, nranks);
#endif

        const int idx_r = (my_rank - step + nranks) % nranks;
        const int idx_l = (my_rank + step + nranks + 1) % nranks;
        const bool fwd = (step > reduce_steps);

#ifdef __CPU_SIM
        // Snapshot source chunks into exchange buffers.
        for (int i = 0; i < chunk_elems; ++i)
            exchange_right[i] = chunks[static_cast<size_t>(idx_r * chunk_elems) + i];
        for (int i = 0; i < chunk_elems; ++i)
            exchange_left[i] = chunks[static_cast<size_t>(idx_l * chunk_elems) + i];

        // Barrier B: all snapshots complete — safe to push.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1) + 1) * kMaxSupportedRanks, my_rank, nranks);
#endif

        // Right-bound push.
        {
#ifdef __CPU_SIM
            __gm__ float *src = exchange_right;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_r * chunk_elems), right);
            if (fwd) {
                for (int i = 0; i < chunk_elems; ++i)
                    dst[i] = src[i];
            } else {
                for (int i = 0; i < chunk_elems; ++i)
                    dst[i] += src[i];
            }
#else
            __gm__ float *src = chunks + static_cast<size_t>(idx_r * chunk_elems);
            __gm__ float *dst = CommRemotePtr(commCtx, src, right);
            GT srcG(src, chunkShape, chunkStride);
            GT dstG(dst, chunkShape, chunkStride);
            if (fwd) pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
            else pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
        }

        // Left-bound push.
        {
#ifdef __CPU_SIM
            __gm__ float *src = exchange_left;
            __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_l * chunk_elems), left);
            if (fwd) {
                for (int i = 0; i < chunk_elems; ++i)
                    dst[i] = src[i];
            } else {
                for (int i = 0; i < chunk_elems; ++i)
                    dst[i] += src[i];
            }
#else
            __gm__ float *src = chunks + static_cast<size_t>(idx_l * chunk_elems);
            __gm__ float *dst = CommRemotePtr(commCtx, src, left);
            GT srcG(src, chunkShape, chunkStride);
            GT dstG(dst, chunkShape, chunkStride);
            if (fwd) pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstG, srcG, pushTile);
            else pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
#endif
        }

        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Final sync: ensure all pushes are globally visible before stage-out.
    // ------------------------------------------------------------------
    if (nranks > 1) {
#ifdef __CPU_SIM
        RoundBarrier(commCtx, signal_base + (2 * (nranks - 1)) * kMaxSupportedRanks, my_rank, nranks);
#else
        RoundBarrier(commCtx, signal_base + (nranks - 1) * kMaxSupportedRanks, my_rank, nranks);
#endif
    }

    // ------------------------------------------------------------------
    // Phase 3: stage-out — copy P fully-reduced chunks to output.
    // ------------------------------------------------------------------
    for (int c = 0; c < nranks; ++c) {
        __gm__ float *dst = output + static_cast<size_t>(c * chunk_elems);
        __gm__ float *src = chunks + static_cast<size_t>(c * chunk_elems);
        GT srcG(src, chunkShape, chunkStride);
        GT dstG(dst, chunkShape, chunkStride);
        TLOAD(stageTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, stageTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
}
