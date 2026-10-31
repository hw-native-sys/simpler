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

#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif

#include <pto/pto-inst.hpp>

#include "backend/sdma/sdma_completion_kernel.h"
#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *out = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *ctx = reinterpret_cast<__gm__ CommContext *>(args[2]);
    if (ctx == nullptr || ctx->rankNum != 2 || ctx->rankId >= 2 ||
        comm_workspace<CommWorkspaceKind::SDMA>(*ctx).addr == 0 ||
        comm_workspace<CommWorkspaceKind::URMA>(*ctx).addr == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    constexpr int kHalfElems = 128 * 128 / 2;
    using Shape = pto::Shape<1, 1, 1, 1, kHalfElems>;
    using Stride = pto::Stride<kHalfElems, kHalfElems, kHalfElems, kHalfElems, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Scratch = pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT>;

    auto *src = reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset;
    auto *dst = reinterpret_cast<__gm__ float *>(out->buffer.addr) + out->start_offset;
    const uint32_t peer = 1u - ctx->rankId;
    const uint64_t offset = reinterpret_cast<uint64_t>(src) - ctx->windowsIn[ctx->rankId];
    auto *sdma_src = reinterpret_cast<__gm__ float *>(ctx->windowsIn[peer] + offset);
    auto *urma_workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*ctx).addr);
    auto *urma_src = pto2::urma_backend::peer_mr_ptr<float>(urma_workspace, peer, offset + kHalfElems * sizeof(float));
    Scratch scratch;
    pto::TASSIGN(scratch, 0x0);
    AsyncCtx async_ctx = get_async_ctx(args);
    // The two engines write disjoint halves and share one deferred-completion slab.
    // Opposite posting orders on the two ranks exercise both registration orders.
    if (ctx->rankId == 0) {
        (void)send_request_entry(
            async_ctx, SdmaTget(
                           Global(dst), Global(sdma_src), scratch,
                           reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*ctx).addr)
                       )
        );
        (void)send_request_entry(async_ctx, UrmaTget(Global(dst + kHalfElems), Global(urma_src), urma_workspace, peer));
    } else {
        (void)send_request_entry(async_ctx, UrmaTget(Global(dst + kHalfElems), Global(urma_src), urma_workspace, peer));
        (void)send_request_entry(
            async_ctx, SdmaTget(
                           Global(dst), Global(sdma_src), scratch,
                           reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*ctx).addr)
                       )
        );
    }
}
