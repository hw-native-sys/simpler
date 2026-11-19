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

#include "../device.h"
#include "async_kernel_api.h"
#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto ctx = get_async_ctx(args);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    auto *metadata = sdma_reuse::data<uint64_t>(args[3]);
    const uint32_t lane = get_block_idx(args);
    const uint32_t posts = args[6];
    const uint32_t lanes = args[7];
    if (!async_ctx_is_deferred(ctx) || ctx.resident_enabled != 1 || posts > sdma_reuse::kPostsPerLane ||
        lane >= lanes || lanes != get_block_num(args) || comm->rankNum < 2) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    if (lane == 0) {
        auto *header = metadata + sdma_reuse::kMaxPosts * sdma_reuse::kWords;
        header[0] = lanes;
        header[1] = posts;
        header[2] = get_block_num(args);
        sdma_reuse::flush(header);
    }
    auto *source = sdma_reuse::data<float>(args[0]);
    auto *destination = sdma_reuse::data<float>(args[1]);
    const uint32_t peer = args[5];
    if (peer >= comm->rankNum || peer == comm->rankId) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    const uint64_t relative = reinterpret_cast<uint64_t>(source) - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + relative);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    pto::TASSIGN(scratch, SDMA_SCRATCH_ALIGNMENT);
    for (uint32_t i = 0; i < posts; ++i) {
        const uint32_t index = lane * posts + i;
        const uint32_t n = sdma_reuse::elements(index);
        const uint32_t start = sdma_reuse::offset(index);
        auto *row = metadata + index * sdma_reuse::kWords;
        Global src(remote + start, Shape(1, 1, 1, 1, n), Stride(n, n, n, n, 1));
        Global dst(destination + start, Shape(1, 1, 1, 1, n), Stride(n, n, n, n, 1));
        if (!send_request_entry(ctx, SdmaTget(dst, src, scratch, workspace))) return;
        if (*ctx.completion_count != i + 1) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return;
        }
        const auto &entry = ctx.completion_entries[i];
        row[0] = 1;
        row[1] = entry.addr;
        row[2] = entry.backend_cookie;
        row[3] = get_block_idx();
        row[4] = lane;
        row[5] = i;
        sdma_reuse::flush(row);
    }
}
