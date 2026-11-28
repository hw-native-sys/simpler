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

#include "backend/urma/urma_completion_kernel.h"

#include <pto/pto-inst.hpp>

#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {

using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using GlobalData = pto::GlobalTensor<float, Shape, Stride>;

inline __aicore__ GlobalData global_data(__gm__ float *ptr, uint32_t elems) {
    return GlobalData(ptr, Shape(1, 1, 1, 1, elems), Stride(elems, elems, elems, elems, 1));
}

inline __aicore__ __gm__ float *tensor_data(__gm__ Tensor *tensor) {
    return reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset;
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
    const uint32_t ranks = 2;
    if (comm == nullptr || comm->rankNum != ranks || comm->rankId >= ranks ||
        comm_workspace<CommWorkspaceKind::URMA>(*comm).addr == 0 || comm->windowsIn[comm->rankId] == 0 || elems == 0 ||
        repeats != 1) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
        return;
    }
    const auto &urma = comm_workspace<CommWorkspaceKind::URMA>(*comm);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(urma.addr);
    const uint32_t peer = (comm->rankId + 1) % ranks;
    auto *local = tensor_data(window_tensor);
    const uint64_t remote_offset =
        reinterpret_cast<uint64_t>(tensor_data(remote_window_tensor)) - comm->windowsIn[comm->rankId];
    auto *remote = pto2::urma_backend::peer_mr_ptr<float>(workspace, peer, remote_offset);
    constexpr uint32_t guard = 32;
    auto *out = tensor_data(out_tensor);
    const bool readback = args[6] == 2;
    if (elems <= 2 * guard) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    if (!put && !readback) {
        for (uint32_t i = 0; i < guard; ++i) {
            out[i] = -77.0f;
            out[elems - guard + i] = -77.0f;
        }
        pto2::detail::defer_flush_range(out, guard * sizeof(float));
        pto2::detail::defer_flush_range(out + elems - guard, guard * sizeof(float));
        dsb(DSB_DDR);
    }
    const uint32_t offset = readback ? 0 : guard;
    const uint32_t count = elems - 2 * offset;
    auto dst = global_data((put ? remote : out) + offset, count);
    auto src = global_data((put ? local : remote) + offset, count);
    if (put) (void)send_request_entry(ctx, UrmaTput(dst, src, urma, peer));
    else (void)send_request_entry(ctx, UrmaTget(dst, src, urma, peer));
}
