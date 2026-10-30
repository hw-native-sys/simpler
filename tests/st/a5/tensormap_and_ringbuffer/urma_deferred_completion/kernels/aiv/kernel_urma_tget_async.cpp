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

#include "backend/urma/urma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {

using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using GlobalData = pto::GlobalTensor<float, Shape, Stride>;
constexpr uint32_t kChunkedElems = 256 * 1024 * 1024 / sizeof(float) + 8192;

inline __aicore__ GlobalData global_data(__gm__ float *ptr, uint32_t elems) {
    return GlobalData(ptr, Shape(1, 1, 1, 1, elems), Stride(elems, elems, elems, elems, 1));
}

inline __aicore__ __gm__ float *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename Tensor>
inline __aicore__ void submit_repeated(
    AsyncCtx &ctx, Tensor dst, Tensor src, __gm__ uint8_t *workspace, uint32_t peer, bool put, uint32_t repeats
) {
    const uint32_t before = ctx.completion_count == nullptr ? 0 : *ctx.completion_count;
    uint32_t entries_per_request = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
        const bool ok = put ? send_request_entry(ctx, UrmaTput(dst, src, workspace, peer)) :
                              send_request_entry(ctx, UrmaTget(dst, src, workspace, peer));
        if (!ok) return;
        if (i == 0) entries_per_request = *ctx.completion_count - before;
    }
    if (entries_per_request == 0 || ctx.completion_count == nullptr ||
        static_cast<uint64_t>(*ctx.completion_count - before) != static_cast<uint64_t>(entries_per_request) * repeats) {
        pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
    }
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *window_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *remote_window_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint32_t elems = static_cast<uint32_t>(args[5]);
    const bool put = args[6] == 1;
    const uint32_t repeats = static_cast<uint32_t>(args[7]);
    AsyncCtx ctx = get_async_ctx(args);
    if (comm == nullptr || comm->rankNum != 2 || comm->rankId >= 2 ||
        comm_workspace<CommWorkspaceKind::URMA>(*comm).addr == 0 || comm->windowsIn[comm->rankId] == 0 || elems == 0 ||
        repeats == 0) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::URMA>(*comm).addr);
    const uint32_t peer = 1u - comm->rankId;
    auto *local = tensor_data(window_tensor);
    const uint64_t remote_offset =
        reinterpret_cast<uint64_t>(tensor_data(remote_window_tensor)) - comm->windowsIn[comm->rankId];
    auto *remote = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, remote_offset);
    if (!put && elems == kChunkedElems) {
        using StaticShape = pto::Shape<1, 1, 1, 1, kChunkedElems>;
        using StaticStride = pto::Stride<kChunkedElems, kChunkedElems, kChunkedElems, kChunkedElems, 1>;
        using StaticData = pto::GlobalTensor<float, StaticShape, StaticStride>;
        submit_repeated(ctx, StaticData(tensor_data(out_tensor)), StaticData(remote), workspace, peer, false, repeats);
        return;
    }
    auto src = global_data(put ? local : remote, elems);
    auto dst = global_data(put ? remote : tensor_data(out_tensor), elems);
    submit_repeated(ctx, dst, src, workspace, peer, put, repeats);
}
