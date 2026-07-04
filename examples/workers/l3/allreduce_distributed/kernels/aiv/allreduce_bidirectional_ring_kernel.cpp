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
 * Bidirectional Ring AllReduce kernel — interleaved bidirectional RS+AG
 * implementing the IBing algorithm (Zong et al., ACM TACO 2025).
 *
 * Communication steps halved from 2(P-1) to P-1 by sending/receiving in
 * both HCCS directions simultaneously.  Reuses TLOAD/TNOTIFY/TWAIT/TADD
 * primitives unchanged — innovation at the schedule level.
 *
 * Phase 1 (stage-in):      partition input → P chunk slots in window
 * Phase 2 (bidir RS+AG):   P-1 ring steps; first floor((P-1)/2) steps
 *                          TADD both received chunks; remaining steps
 *                          TSTORE-forward (allgather).
 * Phase 3 (stage-out):     chunks → output
 *
 * Scratch layout (per rank, in HCCL window):
 *   [0 .. P*chunk)              P chunk slots owned by this rank
 *   [P*chunk .. (P+1)*chunk)    exchange_left  — left-bound  publish buffer
 *   [(P+1)*chunk .. (P+2)*chunk) exchange_right — right-bound publish buffer
 *   tail                        (P-1) * kMaxSupportedRanks int32 barrier slots
 *
 * Two exchange buffers are required because each rank publishes two
 * different chunks per step — one TSTORE would overwrite the other before
 * the barrier.  The barrier signals both are ready.
 *
 * args layout (passed as Tensor arg slots):
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

// Per-round barrier row: used exactly once (AtomicAdd 0→1, TWAIT GE 1).
AICORE inline void RoundBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_row + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
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
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (ALLREDUCE_COUNT % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(ALLREDUCE_COUNT / static_cast<size_t>(nranks));
    __gm__ float *chunks = scratch;
    __gm__ float *exchange_left = scratch + static_cast<size_t>(nranks * chunk_elems);
    __gm__ float *exchange_right = exchange_left + static_cast<size_t>(chunk_elems);
    // Signal rows after float region: (P-1) rounds, kMaxSupportedRanks stride.
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(exchange_right + static_cast<size_t>(chunk_elems));

    TileData chunkTile(1, chunk_elems);
    TileData publishTile(1, chunk_elems);
    TileData recvLeftTile(1, chunk_elems);
    TileData recvRightTile(1, chunk_elems);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(publishTile, 0x10000);
    TASSIGN(recvLeftTile, 0x20000);
    TASSIGN(recvRightTile, 0x30000);

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

    // ------------------------------------------------------------------
    // Phase 1: stage-in — partition local input into P chunk slots.
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

    // ------------------------------------------------------------------
    // Phase 2: bidirectional RS+AG — P-1 ring steps.
    //
    // Per step s (1..P-1):
    //   1. Publish two chunks to exchange buffers
    //   2. RoundBarrier — notify all peers both chunks are ready
    //   3. TLOAD from left neighbor's exchange_right + right neighbor's exchange_left
    //   4. First floor((P-1)/2) steps: TADD both into chunks[]
    //   5. Remaining steps: TSTORE both into chunks[] (allgather forward)
    //
    // Index formulas from IBing (Zong et al. 2025):
    //   Right-bound publish:  idx = (r - s + P) % P  → exchange_right
    //   Left-bound  publish:  idx = (r + s + P + 1) % P  → exchange_left
    //   Receive from left:    accumulate at (left - s + P) % P
    //   Receive from right:   accumulate at (right + s + P + 1) % P
    // ------------------------------------------------------------------
    int round = 0;
    int reduce_steps = (nranks - 1) / 2;  // floor((P-1)/2)
    if (nranks == 2) {
        reduce_steps = 1;  // P=2: single step must always reduce (no forwarding needed)
    }

    for (int step = 1; step < nranks; ++step) {
        const int left = (my_rank - 1 + nranks) % nranks;
        const int right = (my_rank + 1) % nranks;

        // Indices for the two chunks to publish this step.
        const int right_send_idx = (my_rank - step + nranks) % nranks;
        const int left_send_idx = (my_rank + step + nranks + 1) % nranks;

        // Publish right-bound chunk → exchange_right.
        {
            Global srcG(chunks + static_cast<size_t>(right_send_idx * chunk_elems), chunkShape, chunkStride);
            Global dstG(exchange_right, chunkShape, chunkStride);
            TLOAD(publishTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG, publishTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // Publish left-bound chunk → exchange_left.
        {
            Global srcG(chunks + static_cast<size_t>(left_send_idx * chunk_elems), chunkShape, chunkStride);
            Global dstG(exchange_left, chunkShape, chunkStride);
            TLOAD(publishTile, srcG);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstG, publishTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        }
        pipe_barrier(PIPE_ALL);

        RoundBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, nranks);
        ++round;

        // Indices where received data accumulates locally.
        const int acc_from_left_idx = (left - step + nranks) % nranks;
        const int acc_from_right_idx = (right + step + nranks + 1) % nranks;

        // TLOAD from left neighbor's exchange_right.
        {
            __gm__ float *remote_buf = CommRemotePtr(commCtx, exchange_right, left);
            Global remoteG(remote_buf, chunkShape, chunkStride);
            TLOAD(recvLeftTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        // TLOAD from right neighbor's exchange_left.
        {
            __gm__ float *remote_buf = CommRemotePtr(commCtx, exchange_left, right);
            Global remoteG(remote_buf, chunkShape, chunkStride);
            TLOAD(recvRightTile, remoteG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }

        if (step <= reduce_steps) {
            // --- Reduce phase: TADD both received chunks ---

            // TADD from left into acc_from_left_idx.
            {
                Global accG(chunks + static_cast<size_t>(acc_from_left_idx * chunk_elems), chunkShape, chunkStride);
                TLOAD(chunkTile, accG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
                TADD(chunkTile, chunkTile, recvLeftTile);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(accG, chunkTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }

            // TADD from right into acc_from_right_idx.
            {
                Global accG(chunks + static_cast<size_t>(acc_from_right_idx * chunk_elems), chunkShape, chunkStride);
                TLOAD(chunkTile, accG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
                TADD(chunkTile, chunkTile, recvRightTile);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(accG, chunkTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            }
        } else {
            // --- Allgather phase: TSTORE-forward both received chunks ---

            {
                Global dstG(chunks + static_cast<size_t>(acc_from_left_idx * chunk_elems), chunkShape, chunkStride);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstG, recvLeftTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }

            {
                Global dstG(chunks + static_cast<size_t>(acc_from_right_idx * chunk_elems), chunkShape, chunkStride);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstG, recvRightTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            }
        }

        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Phase 3: stage-out — write concatenated chunks into local output.
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
