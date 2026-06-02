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

#ifndef ALLREDUCE_RING_COMMON_HPP
#define ALLREDUCE_RING_COMMON_HPP

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "platform_comm/comm_context.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t kAllReduceCount = 256;
static constexpr int kMaxSupportedRanks = 16;
static constexpr size_t kChunkMax = kAllReduceCount / 2;

template <typename T>
AICORE inline __gm__ T *RingCommRemotePtr(__gm__ CommContext *ctx, __gm__ T *localPtr, int pe) {
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

using RingShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RingStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RingGlobal = pto::GlobalTensor<float, RingShapeDyn, RingStrideDyn, pto::Layout::ND>;
using RingTileData = pto::Tile<pto::TileType::Vec, float, 1, kChunkMax, pto::BLayout::RowMajor, -1, -1>;

AICORE inline void RingZeroSignals(__gm__ int32_t *signal_base, int signal_slots) {
    for (int i = 0; i < signal_slots; ++i) {
        signal_base[i] = 0;
    }
    pipe_barrier(PIPE_ALL);
}

AICORE inline void RingRoundBarrier(__gm__ CommContext *ctx, __gm__ int32_t *signal_row, int my_rank, int nranks) {
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ int32_t *remote_signal = RingCommRemotePtr(ctx, signal_row + my_rank, peer);
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

AICORE inline void RingCopyChunkGm(__gm__ float *dst, __gm__ float *src, int chunk_elems, RingTileData &tile) {
    RingShapeDyn shape(1, 1, 1, 1, chunk_elems);
    RingStrideDyn stride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);
    RingGlobal srcG(src, shape, stride);
    RingGlobal dstG(dst, shape, stride);
    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

AICORE inline void RingRecvExchangeFromLeft(
    __gm__ CommContext *ctx, __gm__ float *exchange_local, int my_rank, int nranks, int chunk_elems,
    RingTileData &recvTile
) {
    int left = (my_rank - 1 + nranks) % nranks;
    __gm__ float *remote_exchange = RingCommRemotePtr(ctx, exchange_local, left);
    RingShapeDyn shape(1, 1, 1, 1, chunk_elems);
    RingStrideDyn stride(chunk_elems, chunk_elems, chunk_elems, chunk_elems, 1);
    RingGlobal remoteG(remote_exchange, shape, stride);
    TLOAD(recvTile, remoteG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

AICORE inline void RingBindScratch(
    __gm__ float *scratch, int nranks, int chunk_elems, __gm__ float *&chunks, __gm__ float *&exchange,
    __gm__ int32_t *&signal_base, int &signal_slots
) {
    chunks = scratch;
    exchange = scratch + static_cast<size_t>(nranks * chunk_elems);
    signal_base = reinterpret_cast<__gm__ int32_t *>(scratch + static_cast<size_t>((nranks + 1) * chunk_elems));
    signal_slots = 2 * (nranks - 1) * kMaxSupportedRanks;
}

#endif  // ALLREDUCE_RING_COMMON_HPP
