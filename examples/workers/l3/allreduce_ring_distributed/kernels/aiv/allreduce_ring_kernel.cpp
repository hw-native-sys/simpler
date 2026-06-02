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
 * Ring AllReduce kernel — chunked reduce-scatter + allgather on a logical ring.
 *
 * Contrasts with mesh ``allreduce_kernel.cpp`` (O(P) full-vector remote reads):
 * each of the (P-1) reduce-scatter and (P-1) allgather rounds moves one chunk
 * to the right neighbour; left neighbour reads the published exchange slot via
 * CommRemotePtr + TLOAD.
 *
 * Scratch layout (float region then int32 signals):
 *   [0 .. P*chunk)           P chunk slots owned by this rank
 *   [P*chunk .. (P+1)*chunk) single-chunk exchange publish area
 *   tail                     2*(P-1)*kMaxSupportedRanks int32 barrier slots
 *                            (one row per round, indexed by rank)
 */

#include "allreduce_ring_common.hpp"
#include "tensor.h"

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

    if (nranks <= 1 || nranks > kMaxSupportedRanks || (kAllReduceCount % static_cast<size_t>(nranks)) != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int chunk_elems = static_cast<int>(kAllReduceCount / static_cast<size_t>(nranks));
    __gm__ float *chunks = nullptr;
    __gm__ float *exchange = nullptr;
    __gm__ int32_t *signal_base = nullptr;
    int signal_slots = 0;
    RingBindScratch(scratch, nranks, chunk_elems, chunks, exchange, signal_base, signal_slots);

    RingZeroSignals(signal_base, signal_slots);

    RingTileData chunkTile(1, chunk_elems);
    RingTileData recvTile(1, chunk_elems);
    TASSIGN(chunkTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    RingShapeDyn chunkShape(1, 1, 1, 1, chunk_elems);
    RingStrideDyn chunkStride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);

    // Stage-in: partition local input into P chunk slots in the window.
    for (int chunk = 0; chunk < nranks; ++chunk) {
        RingCopyChunkGm(
            chunks + static_cast<size_t>(chunk * chunk_elems), input + static_cast<size_t>(chunk * chunk_elems),
            chunk_elems, chunkTile
        );
    }
    pipe_barrier(PIPE_ALL);

    int round = 0;

    // Reduce-scatter: (P-1) ring steps; rank r ends with fully reduced chunk r.
    for (int step = 1; step < nranks; ++step) {
        const int send_idx = (my_rank - step + nranks) % nranks;
        const int recv_add_idx = (my_rank - step - 1 + nranks) % nranks;

        RingCopyChunkGm(exchange, chunks + static_cast<size_t>(send_idx * chunk_elems), chunk_elems, chunkTile);
        pipe_barrier(PIPE_ALL);

        RingRoundBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, nranks);
        ++round;

        RingRecvExchangeFromLeft(commCtx, exchange, my_rank, nranks, chunk_elems, recvTile);

        RingGlobal accG(chunks + static_cast<size_t>(recv_add_idx * chunk_elems), chunkShape, chunkStride);
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

    // Allgather: (P-1) ring steps; every rank collects all reduced chunks.
    for (int step = 1; step < nranks; ++step) {
        const int send_idx = (my_rank - step + 1 + nranks) % nranks;
        const int recv_idx = (my_rank - step + nranks) % nranks;

        RingCopyChunkGm(exchange, chunks + static_cast<size_t>(send_idx * chunk_elems), chunk_elems, chunkTile);
        pipe_barrier(PIPE_ALL);

        RingRoundBarrier(commCtx, signal_base + round * kMaxSupportedRanks, my_rank, nranks);
        ++round;

        RingRecvExchangeFromLeft(commCtx, exchange, my_rank, nranks, chunk_elems, recvTile);

        RingGlobal dstG(chunks + static_cast<size_t>(recv_idx * chunk_elems), chunkShape, chunkStride);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, recvTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    // Stage-out: concatenated chunks -> local output tensor.
    for (int chunk = 0; chunk < nranks; ++chunk) {
        RingCopyChunkGm(
            output + static_cast<size_t>(chunk * chunk_elems), chunks + static_cast<size_t>(chunk * chunk_elems),
            chunk_elems, chunkTile
        );
    }
    pipe_barrier(PIPE_ALL);
}
