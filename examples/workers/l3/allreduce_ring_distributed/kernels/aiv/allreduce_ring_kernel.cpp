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
 * Ring AllReduce kernel — chunked RS + AG over a logical ring.
 *
 * Contrasts with mesh ``allreduce_kernel.cpp`` (O(P) full-vector remote reads):
 * each of the (P-1) reduce-scatter and (P-1) allgather rounds moves one chunk
 * to the right neighbour; left neighbour reads the published exchange slot via
 * CommRemotePtr + TLOAD (same local-write / barrier / remote-read pattern as mesh).
 *
 * Scratch layout (float region then int32 signals):
 *   [0 .. P*chunk)           P chunk slots owned by this rank
 *   [P*chunk .. (P+1)*chunk) single-chunk exchange publish area
 *   tail                     kMaxSupportedRanks int32 barrier slots (one row)
 *
 * Multi-round barriers reuse the same signal row with monotonically increasing
 * generation counters (AtomicAdd + TWAIT GE generation).  Do not zero signal
 * slots between rounds — a faster rank's TNOTIFY can arrive while a slower
 * rank is still resetting, which loses the wakeup and deadlocks TWAIT.
 *
 * args layout (see allreduce_ring_orch.cpp):
 *   tensor(0) = input    (ALLREDUCE_COUNT floats)
 *   tensor(1) = output   (ALLREDUCE_COUNT floats)
 *   tensor(2) = scratch  (HCCL window slot)
 *   scalar(0) = nranks
 *   scalar(1) = CommContext device pointer
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "platform_comm/comm_context.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t ALLREDUCE_COUNT = 256;
static constexpr int kMaxSupportedRanks = 16;
// Minimum rank count is 2, so the largest chunk is ALLREDUCE_COUNT / 2.
static constexpr size_t CHUNK_MAX = ALLREDUCE_COUNT / 2;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, float, 1, CHUNK_MAX, pto::BLayout::RowMajor, -1, -1>;

// Mesh-style device barrier with cumulative generation counter on a single signal row.
AICORE inline void
DeviceBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_base, int my_rank, int nranks, int32_t generation) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ int32_t *remote_signal = CommRemotePtr(ctx, signal_base + my_rank, peer);
        pto::comm::Signal sig(remote_signal);
        pto::comm::TNOTIFY(sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        pto::comm::Signal sig(signal_base + peer);
        pto::comm::TWAIT(sig, generation, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);
}

AICORE inline void CopyChunkGm(__gm__ float *dst, __gm__ float *src, int chunk_elems, TileData &tile) {
    ShapeDyn shape(1, 1, 1, 1, chunk_elems);
    StrideDyn stride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);
    Global srcG(src, shape, stride);
    Global dstG(dst, shape, stride);
    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

AICORE inline void RecvExchangeFromLeft(
    __gm__ CommContext *ctx, __gm__ float *exchange_local, int my_rank, int nranks, int chunk_elems, TileData &recvTile
) {
    int left = (my_rank - 1 + nranks) % nranks;
    __gm__ float *remote_exchange = CommRemotePtr(ctx, exchange_local, left);
    ShapeDyn shape(1, 1, 1, 1, chunk_elems);
    StrideDyn stride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);
    Global remoteG(remote_exchange, shape, stride);
    TLOAD(recvTile, remoteG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
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
    __gm__ float *chunks = scratch;
    __gm__ float *exchange = scratch + static_cast<size_t>(nranks * chunk_elems);
    __gm__ int32_t *signal_base =
        reinterpret_cast<__gm__ int32_t *>(scratch + static_cast<size_t>((nranks + 1) * chunk_elems));

    // Zero-init once per invocation (handles stale HCCL window state from a prior run).
    for (int i = 0; i < nranks; ++i) {
        signal_base[i] = 0;
    }
    pipe_barrier(PIPE_ALL);

    TileData chunkTile(1, chunk_elems);
    TileData recvTile(1, chunk_elems);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    ShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    StrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

    int32_t generation = 0;

    // ------------------------------------------------------------------
    // Stage-in: partition local input into P chunk slots in the window.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        CopyChunkGm(
            chunks + static_cast<size_t>(chunk * chunk_elems), input + static_cast<size_t>(chunk * chunk_elems),
            chunk_elems, chunkTile
        );
    }
    pipe_barrier(PIPE_ALL);

    DeviceBarrier(commCtx, signal_base, my_rank, nranks, ++generation);

    // ------------------------------------------------------------------
    // Reduce-scatter: (P-1) ring steps; rank r ends with fully reduced chunk r.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int send_idx = (my_rank - step + nranks) % nranks;
        const int recv_add_idx = (my_rank - step - 1 + nranks) % nranks;

        CopyChunkGm(exchange, chunks + static_cast<size_t>(send_idx * chunk_elems), chunk_elems, chunkTile);
        pipe_barrier(PIPE_ALL);

        DeviceBarrier(commCtx, signal_base, my_rank, nranks, ++generation);

        RecvExchangeFromLeft(commCtx, exchange, my_rank, nranks, chunk_elems, recvTile);

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
    // Allgather: (P-1) ring steps; every rank collects all reduced chunks.
    // ------------------------------------------------------------------
    for (int step = 1; step < nranks; ++step) {
        const int send_idx = (my_rank - step + 1 + nranks) % nranks;
        const int recv_idx = (my_rank - step + nranks) % nranks;

        CopyChunkGm(exchange, chunks + static_cast<size_t>(send_idx * chunk_elems), chunk_elems, chunkTile);
        pipe_barrier(PIPE_ALL);

        DeviceBarrier(commCtx, signal_base, my_rank, nranks, ++generation);

        RecvExchangeFromLeft(commCtx, exchange, my_rank, nranks, chunk_elems, recvTile);

        Global dstG(chunks + static_cast<size_t>(recv_idx * chunk_elems), chunkShape, chunkStride);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, recvTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    // ------------------------------------------------------------------
    // Stage-out: concatenated chunks → local output tensor.
    // ------------------------------------------------------------------
    for (int chunk = 0; chunk < nranks; ++chunk) {
        CopyChunkGm(
            output + static_cast<size_t>(chunk * chunk_elems), chunks + static_cast<size_t>(chunk * chunk_elems),
            chunk_elems, chunkTile
        );
    }
    pipe_barrier(PIPE_ALL);
}
