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
 * AllReduce kernel using PR #522's publish/notify/wait design.
 *
 * Each rank publishes its local input into every peer's recv_window via TPUT,
 * notifies peers once the publish is issued, waits until all peers have done
 * the same, then accumulates the slots that landed in its own recv_window.
 *
 * args layout:
 *   args[0] = __gm__ Tensor* input_tensor
 *   args[1] = __gm__ Tensor* recv_window_tensor
 *   args[2] = __gm__ Tensor* output_tensor
 *   args[3] = __gm__ Tensor* notify_counter_tensor
 *   args[4] = __gm__ CommContext* ctx
 */

#include <cstdint>

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include "pto/comm/pto_comm_inst.hpp"
#include "tensor.h"
#include "platform_comm/comm_context.h"

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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *inputTensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *recvWindowTensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *outputTensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *notifyCounterTensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    __gm__ float *inputPtr =
        reinterpret_cast<__gm__ float *>(inputTensor->buffer.addr) + inputTensor->start_offset;
    __gm__ float *recvWindowPtr =
        reinterpret_cast<__gm__ float *>(recvWindowTensor->buffer.addr) + recvWindowTensor->start_offset;
    __gm__ float *outputPtr =
        reinterpret_cast<__gm__ float *>(outputTensor->buffer.addr) + outputTensor->start_offset;
    __gm__ int32_t *notifyCounterPtr =
        reinterpret_cast<__gm__ int32_t *>(notifyCounterTensor->buffer.addr) + notifyCounterTensor->start_offset;

    using VectorGlobal = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, ALLREDUCE_COUNT>,
        pto::Stride<1, 1, 1, ALLREDUCE_COUNT, 1>>;
    using VectorTile = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int myRank = static_cast<int>(commCtx->rankId);
    int nranks = static_cast<int>(commCtx->rankNum);

    VectorGlobal inputGlobal(inputPtr);
    VectorTile sumTile(1, ALLREDUCE_COUNT);
    VectorTile recvTile(1, ALLREDUCE_COUNT);
    VectorTile stagingTile(1, ALLREDUCE_COUNT);
    TASSIGN(sumTile, 0x0);
    TASSIGN(recvTile, 0x10000);
    TASSIGN(stagingTile, 0x20000);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    TLOAD(sumTile, inputGlobal);
    pipe_barrier(PIPE_ALL);

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == myRank) {
            continue;
        }
        __gm__ float *remoteRecvWindowBase = CommRemotePtr(commCtx, recvWindowPtr, peer);
        __gm__ float *remoteSlotPtr = remoteRecvWindowBase + myRank * ALLREDUCE_COUNT;
        VectorGlobal remoteSlot(remoteSlotPtr);
        pto::comm::TPUT(remoteSlot, inputGlobal, stagingTile);
    }
    pipe_barrier(PIPE_ALL);

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == myRank) {
            continue;
        }
        __gm__ int32_t *remoteCounter = CommRemotePtr(commCtx, notifyCounterPtr, peer);
        pto::comm::Signal remoteSignal(remoteCounter);
        pto::comm::TNOTIFY(remoteSignal, 1, pto::comm::NotifyOp::AtomicAdd);
    }
    pipe_barrier(PIPE_ALL);

    pto::comm::Signal localCounter(notifyCounterPtr);
    pto::comm::TWAIT(localCounter, nranks - 1, pto::comm::WaitCmp::GE);
    pipe_barrier(PIPE_ALL);

    for (int peer = 0; peer < nranks; ++peer) {
        if (peer == myRank) {
            continue;
        }
        __gm__ float *recvSlotPtr = recvWindowPtr + peer * ALLREDUCE_COUNT;
        VectorGlobal recvSlot(recvSlotPtr);
        TLOAD(recvTile, recvSlot);
        pipe_barrier(PIPE_ALL);
        TADD(sumTile, sumTile, recvTile);
        pipe_barrier(PIPE_ALL);
    }

    VectorGlobal outputGlobal(outputPtr);
    TSTORE(outputGlobal, sumTile);

    pipe_barrier(PIPE_ALL);
}
