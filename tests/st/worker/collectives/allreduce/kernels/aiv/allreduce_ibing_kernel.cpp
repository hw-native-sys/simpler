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
 * Right-bound:  r pushes chunks[(r - s + P) % P]         → rank r+1
 * Left-bound:   r pushes chunks[(r + s + 1 + P) % P]    → rank r-1
 *
 * Exchange buffers snapshot source chunks before pushing on all platforms,
 * avoiding intra-round read/write races where a remote TPUT can modify a
 * chunk between the right-bound and left-bound TLOAD within the same step.
 * The double-barrier scheme ensures (a) prior writes are globally visible,
 * (b) all snapshots complete, before any push reads the exchange buffers.
 *
 * **Async allgather (forward) phase**: steps ⌊P/2⌋+1..P−1 use TPUT_ASYNC
 * instead of synchronous TPUT<AtomicNone>.  Both CW and CCW pushes are
 * issued to the SDMA engine back-to-back per round, then drained with a
 * single WaitAll().  Reduce-scatter steps (1..⌊P/2⌋) still use synchronous
 * TPUT<AtomicAdd> because TPUT_ASYNC does not support atomic operations.
 *
 * Scratch layout (per rank, in HCCL window):
 *   [0 .. P*chunk_elems)                   P working chunks
 *   [P*chunk .. (P+1)*chunk)               exchange_right snapshot buffer
 *   [(P+1)*chunk .. (P+2)*chunk)          exchange_left  snapshot buffer
 *   tail                   (2*(P-1)+1) * kMaxSupportedRanks int32 signals
 *   tail + ...                             kSdmaWorkspaceSize bytes for SDMA
 *                                          control structures
 *
 * Divisibility: requires ALLREDUCE_COUNT % nranks == 0.
 */
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/comm/async_common/async_types.hpp"
#ifndef __CPU_SIM
#include "pto/comm/async_common/async_event_impl.hpp"
#endif
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
static constexpr size_t kSdmaWorkspaceSize = 16 * 1024;
static constexpr int kMaxAsyncEvents = 8;

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

// ---------------------------------------------------------------------------
// Async session wrapper (self-contained; inline for homogeneity with all
// other AIV kernel files in this directory).
// ---------------------------------------------------------------------------

struct AivAsyncSession {
    pto::comm::AsyncSession session{};
    pto::comm::AsyncEvent events[kMaxAsyncEvents];
    int numPending = 0;

#if defined(__CPU_SIM) || defined(__COSTMODEL)
    AICORE bool Init(__gm__ uint8_t *workspace,
                     const pto::comm::sdma::SdmaBaseConfig& = {}) {
        (void)workspace;
        session = {};
        numPending = 0;
        return true;
    }
#else
    template <int kRows = 1, int kCols = 256>
    AICORE bool Init(__gm__ uint8_t *workspace,
                     const pto::comm::sdma::SdmaBaseConfig& baseConfig =
                         {pto::comm::sdma::kDefaultSdmaBlockBytes, 0, 1}) {
        using ScratchTile =
            pto::Tile<pto::TileType::Vec, uint8_t, kRows, kCols, pto::BLayout::RowMajor, -1, -1>;
        ScratchTile scratchTile(kRows, kCols);
        TASSIGN(scratchTile, static_cast<uint8_t>(0));
        bool ok = pto::comm::BuildAsyncSession(scratchTile, workspace, session, 0, baseConfig);
        if (!ok) return false;
        numPending = 0;
        return true;
    }
#endif

    AICORE void Reset() { numPending = 0; }

    AICORE void Issue(const pto::comm::AsyncEvent& ev) {
        events[numPending++] = ev;
    }

    AICORE bool WaitAll() {
        for (int i = 0; i < numPending; ++i) {
            if (!events[i].Wait(session)) {
                numPending = 0;
                return false;
            }
        }
        numPending = 0;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

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

    // SDMA workspace: carved from the end of the scratch buffer, after the
    // signal rows.  On CPUSIM, TPUT_ASYNC always completes immediately
    // (handle=0), so the workspace content is irrelevant.
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(signal_base + (static_cast<size_t>(2 * (nranks - 1) + 1) * kMaxSupportedRanks));

    // Build the async session (one per kernel invocation).
    AivAsyncSession asyncS;
    if (!asyncS.Init(workspace)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

    using TilePush = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;
    TilePush pushTile(1, chunk_elems);
    TASSIGN(pushTile, 0x10000);

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
    // Steps 1..⌊P/2⌋:     reduce  — AtomicAdd pushes from exchange buffers.
    // Steps ⌊P/2⌋+1..P−1: forward — TPUT_ASYNC AtomicNone pushes.
    //
    // Double-barrier + exchange-buffer snapshot scheme replicates MPI
    // ISend/IRecv semantics (plan 44).  Forward steps issue both CW and CCW
    // pushes to the SDMA engine via TPUT_ASYNC and drain with a single
    // WaitAll() instead of pipe_barrier(PIPE_ALL).
    // ------------------------------------------------------------------
    const int left = (my_rank - 1 + nranks) % nranks;
    const int right = (my_rank + 1) % nranks;
    const int reduce_steps = nranks / 2;

    for (int step = 1; step < nranks; ++step) {
        // Barrier A: all prior writes globally visible.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1)) * kMaxSupportedRanks, my_rank, nranks);

        const int idx_r = (my_rank - step + nranks) % nranks;
        const int idx_l = (my_rank + step + nranks + 1) % nranks;
        const bool fwd = (step > reduce_steps);

        // Snapshot source chunks into exchange buffers.
#ifdef __CPU_SIM
        for (int i = 0; i < chunk_elems; ++i)
            exchange_right[i] = chunks[static_cast<size_t>(idx_r * chunk_elems) + i];
        for (int i = 0; i < chunk_elems; ++i)
            exchange_left[i] = chunks[static_cast<size_t>(idx_l * chunk_elems) + i];
#else
        // exchange_right ← chunks[idx_r]
        {
            __gm__ float *src_r = chunks + static_cast<size_t>(idx_r * chunk_elems);
            GT srcG_r(src_r, chunkShape, chunkStride);
            GT dstG_r(exchange_right, chunkShape, chunkStride);
            TLOAD(stageTile, srcG_r);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstG_r, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        // exchange_left ← chunks[idx_l]
        {
            __gm__ float *src_l = chunks + static_cast<size_t>(idx_l * chunk_elems);
            GT srcG_l(src_l, chunkShape, chunkStride);
            GT dstG_l(exchange_left, chunkShape, chunkStride);
            TLOAD(stageTile, srcG_l);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstG_l, stageTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        }
        pipe_barrier(PIPE_ALL);
#endif

        // Barrier B: all snapshots complete — safe to push.
        RoundBarrier(commCtx, signal_base + (2 * (step - 1) + 1) * kMaxSupportedRanks, my_rank, nranks);

        if (fwd) {
            // Forward step (allgather): TPUT_ASYNC for both CW and CCW pushes.
            asyncS.Reset();

            // Right-bound async push.
            {
                __gm__ float *src = exchange_right;
                __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_r * chunk_elems), right);
                GT srcG(src, chunkShape, chunkStride);
                GT dstG(dst, chunkShape, chunkStride);
                pto::comm::AsyncEvent ev = pto::comm::TPUT_ASYNC(dstG, srcG, asyncS.session);
                asyncS.Issue(ev);
            }

            // Left-bound async push.
            {
                __gm__ float *src = exchange_left;
                __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_l * chunk_elems), left);
                GT srcG(src, chunkShape, chunkStride);
                GT dstG(dst, chunkShape, chunkStride);
                pto::comm::AsyncEvent ev = pto::comm::TPUT_ASYNC(dstG, srcG, asyncS.session);
                asyncS.Issue(ev);
            }

            // Wait for both SDMA transfers to complete.
            asyncS.WaitAll();
        } else {
            // Reduce step: synchronous TPUT<AtomicAdd> (async does not
            // support atomic operations).
            // Right-bound reduce push.
            {
                __gm__ float *src = exchange_right;
                __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_r * chunk_elems), right);
                GT srcG(src, chunkShape, chunkStride);
                GT dstG(dst, chunkShape, chunkStride);
                pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
            }

            // Left-bound reduce push.
            {
                __gm__ float *src = exchange_left;
                __gm__ float *dst = CommRemotePtr(commCtx, chunks + static_cast<size_t>(idx_l * chunk_elems), left);
                GT srcG(src, chunkShape, chunkStride);
                GT dstG(dst, chunkShape, chunkStride);
                pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pushTile);
            }

            pipe_barrier(PIPE_ALL);
        }
    }

    // ------------------------------------------------------------------
    // Final sync: ensure all pushes are globally visible before stage-out.
    // ------------------------------------------------------------------
    if (nranks > 1) {
        RoundBarrier(commCtx, signal_base + (2 * (nranks - 1)) * kMaxSupportedRanks, my_rank, nranks);
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
