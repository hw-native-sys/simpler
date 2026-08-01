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
#include <pto/pto-inst.hpp>

#include "platform_comm/comm_context.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t kCount = 256;
static constexpr int kMaxRanks = 16;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    int rank_count = static_cast<int>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[3]);

    if (rank_count <= 0 || rank_count > kMaxRanks) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    using Shape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, Shape, Stride, pto::Layout::ND>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, kCount, pto::BLayout::RowMajor, -1, -1>;

    Shape shape(1, 1, 1, 1, kCount);
    Stride stride(kCount, kCount, kCount, kCount, 1);
    Tile accumulator(1, kCount);
    Tile peer_tile(1, kCount);
    TASSIGN(accumulator, 0x0);
    TASSIGN(peer_tile, 0x10000);

    Global local_input(input, shape, stride);
    TLOAD(accumulator, local_input);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    int my_rank = static_cast<int>(comm_ctx->rankId);
    for (int peer = 0; peer < rank_count; ++peer) {
        if (peer == my_rank) continue;
        __gm__ float *peer_input = CommRemotePtr(comm_ctx, input, peer);
        Global peer_global(peer_input, shape, stride);
        TLOAD(peer_tile, peer_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TADD(accumulator, accumulator, peer_tile);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    Global result_global(result, shape, stride);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(result_global, accumulator);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}
