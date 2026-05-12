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
 * AllReduce-sum kernel — publish/notify/wait/accumulate over partial_local.
 *
 * Stage 2 of the FFN tensor-parallel demo.  Each rank holds a per-rank
 * ``partial_local`` (= x_shard @ w_shard, written by the AIC matmul kernel
 * in the previous stage); we sum it across ranks into ``y``.  Cross-rank
 * exchange goes through ``scratch``, which is laid out as:
 *
 *   [ mailbox: nranks * M*N floats | signal tail: nranks int32 slots ]
 *
 * Each rank publishes its partial_local into peer's mailbox slot
 * mailbox[my_rank], notifies the peer's signal[my_rank], waits until its
 * own signal tail has been bumped by every peer, then sums local +
 * mailbox[peer] for each peer into sum_tile and stores into y.
 *
 * args layout (ChipStorageTaskArgs — see allreduce_sum_orch.cpp):
 *   tensor(0) = partial_local INPUT           (M x N, per-rank device mem)
 *   tensor(1) = y             OUTPUT_EXISTING (M x N, host-backed)
 *   tensor(2) = scratch       INOUT           (HCCL-window slot, cross-rank)
 *   scalar(0) = nranks
 *   scalar(1) = CommContext device pointer
 */

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <cstdint>

#include <pto/pto-inst.hpp>
#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

#include "platform_comm/comm_context.h"
#include "tensor.h"

using namespace pto;

static constexpr int kRows = 64;
static constexpr int kCols = 64;
static constexpr int kElemsPerPartial = kRows * kCols;
static constexpr int kMaxSupportedRanks = 16;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *partial_local_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *y_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *scratch_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    int nranks = static_cast<int>(args[3]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    __gm__ float *partial_local_ptr =
        reinterpret_cast<__gm__ float *>(partial_local_tensor->buffer.addr) + partial_local_tensor->start_offset;
    __gm__ float *y_ptr = reinterpret_cast<__gm__ float *>(y_tensor->buffer.addr) + y_tensor->start_offset;
    __gm__ float *mailbox_ptr =
        reinterpret_cast<__gm__ float *>(scratch_tensor->buffer.addr) + scratch_tensor->start_offset;
    // Signal slots sit at the tail of the scratch buffer, after nranks * M*N floats.
    __gm__ int32_t *signal_base = reinterpret_cast<__gm__ int32_t *>(mailbox_ptr + nranks * kElemsPerPartial);

    using MatrixGlobal = GlobalTensor<float, Shape<1, 1, 1, kRows, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using MatrixTile = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(comm_ctx->rankId);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    MatrixGlobal partial_local_global(partial_local_ptr);

    MatrixTile sum_tile(kRows, kCols);
    MatrixTile tmp_tile(kRows, kCols);
    MatrixTile staging_tile(kRows, kCols);
    TASSIGN(sum_tile, 0x0);
    TASSIGN(tmp_tile, 0x10000);
    TASSIGN(staging_tile, 0x20000);

    TLOAD(sum_tile, partial_local_global);
    pipe_barrier(PIPE_ALL);

    // Phase 1: publish my partial_local into every peer's mailbox slot mailbox[my_rank].
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ float *remote_mailbox_base = CommRemotePtr(comm_ctx, mailbox_ptr, peer);
        __gm__ float *remote_slot_ptr = remote_mailbox_base + my_rank * kElemsPerPartial;
#if defined(__CPU_SIM)
        for (int i = 0; i < kElemsPerPartial; ++i) {
            remote_slot_ptr[i] = partial_local_ptr[i];
        }
#else
        MatrixGlobal remote_slot(remote_slot_ptr);
        pto::comm::TPUT(remote_slot, partial_local_global, staging_tile);
#endif
    }
    pipe_barrier(PIPE_ALL);

    // Phase 2: notify peer's signal[my_rank] slot, then wait for every peer to notify ours.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ int32_t *remote_signal_slot = CommRemotePtr(comm_ctx, signal_base + my_rank, peer);
        pto::comm::Signal remote_sig(remote_signal_slot);
        pto::comm::TNOTIFY(remote_sig, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
    }
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        pto::comm::Signal local_sig(signal_base + peer);
        pto::comm::TWAIT(local_sig, (int32_t)1, pto::comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);

    // Phase 3: accumulate every peer's mailbox slot into sum_tile.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ float *mailbox_slot_ptr = mailbox_ptr + peer * kElemsPerPartial;
        MatrixGlobal mailbox_slot(mailbox_slot_ptr);
        TLOAD(tmp_tile, mailbox_slot);
        pipe_barrier(PIPE_ALL);
        TADD(sum_tile, sum_tile, tmp_tile);
        pipe_barrier(PIPE_ALL);
    }

    // Phase 4: store sum_tile into y (per-rank device output).
    MatrixGlobal y_global(y_ptr);
    TSTORE(y_global, sum_tile);
    pipe_barrier(PIPE_ALL);
}
