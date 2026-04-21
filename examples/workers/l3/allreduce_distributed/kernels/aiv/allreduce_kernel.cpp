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
 * AllReduce kernel for simpler's kernel_entry signature.
 *
 * Every rank independently reads all ranks' inputs from the RDMA window,
 * computes the element-wise sum, and writes the result to its own output.
 * This is a symmetric allreduce — no designated root, all ranks active.
 *
 * args layout (all uint64_t, cast as needed):
 *   args[0] = __gm__ float* input   (device addr in RDMA window)
 *   args[1] = __gm__ float* output  (device addr, local)
 *   args[2] = int nranks
 *   args[3] = (unused, kept for ABI compatibility)
 *   args[4] = __gm__ CommContext* ctx  (device addr)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
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
    __gm__ float *input = reinterpret_cast<__gm__ float *>(args[0]);
    __gm__ float *output = reinterpret_cast<__gm__ float *>(args[1]);
    int nranks = static_cast<int>(args[2]);
    int root = static_cast<int>(args[3]);
    __gm__ CommContext *commCtx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, ALLREDUCE_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, ALLREDUCE_COUNT);
    StrideDyn stride(ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, ALLREDUCE_COUNT, 1);

    TileData accTile(1, ALLREDUCE_COUNT);
    TileData recvTile(1, ALLREDUCE_COUNT);
    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    if (nranks <= 0 || nranks > kMaxSupportedRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // Every rank reads all inputs and sums them into its own output.
    Global outputG(output, shape, stride);

    __gm__ float *firstInput = CommRemotePtr(commCtx, input, 0);
    Global firstG(firstInput, shape, stride);
    TLOAD(accTile, firstG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int r = 1; r < nranks; ++r) {
        __gm__ float *remoteInput = CommRemotePtr(commCtx, input, r);
        Global remoteG(remoteInput, shape, stride);
        TLOAD(recvTile, remoteG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(accTile, accTile, recvTile);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputG, accTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    pipe_barrier(PIPE_ALL);
}
