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

#include <cstdint>

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommDeviceContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *partial_local_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *partial_window_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *y_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *notify_counter_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ CommDeviceContext *comm_ctx = reinterpret_cast<__gm__ CommDeviceContext *>(args[4]);

    __gm__ float *partial_local_ptr =
        reinterpret_cast<__gm__ float *>(partial_local_tensor->buffer.addr) + partial_local_tensor->start_offset;
    __gm__ float *partial_window_ptr =
        reinterpret_cast<__gm__ float *>(partial_window_tensor->buffer.addr) + partial_window_tensor->start_offset;
    __gm__ float *y_ptr = reinterpret_cast<__gm__ float *>(y_tensor->buffer.addr) + y_tensor->start_offset;
    __gm__ int32_t *notify_counter_ptr =
        reinterpret_cast<__gm__ int32_t *>(notify_counter_tensor->buffer.addr) + notify_counter_tensor->start_offset;

    constexpr int kRows = 64;
    constexpr int kCols = 64;
    constexpr int kElemsPerPartial = kRows * kCols;

    using MatrixGlobal = GlobalTensor<float, Shape<1, 1, 1, kRows, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using MatrixTile = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);

    MatrixGlobal partial_local_global(partial_local_ptr);

    MatrixTile sum_tile(kRows, kCols);
    MatrixTile tmp_tile(kRows, kCols);
    MatrixTile staging_tile(kRows, kCols);
    TASSIGN(sum_tile, 0x0);
    TASSIGN(tmp_tile, 0x10000);
    TASSIGN(staging_tile, 0x20000);

    TLOAD(sum_tile, partial_local_global);
    pipe_barrier(PIPE_ALL);

    // First publish this rank's local partial into every peer's mailbox slot.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ float *remote_mailbox_base = CommRemotePtr(comm_ctx, partial_window_ptr, peer);
        __gm__ float *remote_slot_ptr = remote_mailbox_base + my_rank * kElemsPerPartial;
        MatrixGlobal remote_slot(remote_slot_ptr);
        pto::comm::TPUT(remote_slot, partial_local_global, staging_tile);
    }
    pipe_barrier(PIPE_ALL);

    // Only notify peers after the TPUT sequence above has been issued.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ int32_t *remote_counter = CommRemotePtr(comm_ctx, notify_counter_ptr, peer);
        pto::comm::Signal remote_signal(remote_counter);
        pto::comm::TNOTIFY(remote_signal, 1, pto::comm::NotifyOp::AtomicAdd);
    }
    pipe_barrier(PIPE_ALL);

    pto::comm::Signal local_counter(notify_counter_ptr);
    pto::comm::TWAIT(local_counter, nranks - 1, pto::comm::WaitCmp::GE);
    pipe_barrier(PIPE_ALL);

    // After all peers have published, accumulate the mailbox slots that were
    // written into this rank's local comm window.
    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == my_rank) {
            continue;
        }
        __gm__ float *mailbox_slot_ptr = partial_window_ptr + peer * kElemsPerPartial;
        MatrixGlobal mailbox_slot(mailbox_slot_ptr);
        TLOAD(tmp_tile, mailbox_slot);
        pipe_barrier(PIPE_ALL);
        TADD(sum_tile, sum_tile, tmp_tile);
        pipe_barrier(PIPE_ALL);
    }

    MatrixGlobal y_global(y_ptr);
    TSTORE(y_global, sum_tile);
    pipe_barrier(PIPE_ALL);
}
