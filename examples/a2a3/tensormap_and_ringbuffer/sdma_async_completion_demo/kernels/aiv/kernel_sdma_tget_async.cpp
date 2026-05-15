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

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/npu/comm/async/sdma/sdma_types.hpp>
#include <pto/pto-inst.hpp>

#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "pto_async_kernel_api.h"
#include "tensor.h"

using namespace pto;

template <typename T>
static inline __aicore__ __gm__ T *comm_remote_ptr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);

    __gm__ float *local_in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *local_out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);
    if (nranks != 2 || comm_ctx->workSpace == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int peer_rank = 1 - rank;

    constexpr int kElems = 128 * 128;
    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    __gm__ float *remote_in = comm_remote_ptr(comm_ctx, local_in, peer_rank);
    GlobalData remote_global(remote_in);
    GlobalData local_global(local_out);

    ScratchTile scratch_tile;
    TASSIGN(scratch_tile, 0x0);

    pto::comm::AsyncSession session;
    __gm__ uint8_t *sdma_workspace = reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace);
    if (!pto::comm::BuildAsyncSession(scratch_tile, sdma_workspace, session, 0)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(local_global, remote_global, session);
    AsyncCtx async_ctx = get_async_ctx(args);
    defer_pto_async_event(async_ctx, event, session);
    defer_flush(async_ctx);
}
