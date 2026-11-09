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
#include "backend/sdma/sdma_completion_kernel.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace mixed_async;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[9]);
    const uint32_t elements = args[10], mask = args[11];
    const uint32_t lane = mask == 2 ? 1 : 1 + get_sub_block_id(args);
    const uint32_t execution = get_block_idx(args) * 3 + lane;
    const uint32_t stride = elements + kLine;
    auto *meta = data<uint64_t>(args, 8) + execution * kMeta;
    auto *signals = data<int32_t>(args, 1) + execution * comm->rankNum * kLine;
    auto ctx = get_async_ctx(args);
    meta[0] = 1;
    meta[7] = get_block_idx();
    if (!async_ctx_is_deferred(ctx) || comm->rankNum < 2 || comm->rankNum > 4 || *ctx.completion_count != 0 ||
        !active(execution, mask)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT> scratch;
    pto::TASSIGN(scratch, 0);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    auto *source = data<float>(args, 0) + execution * stride;
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
        Global src(
            remote(comm, source, peer), Shape(1, 1, 1, 1, elements), Stride(elements, elements, elements, elements, 1)
        );
        Global dst(
            data<float>(args, 4) + (execution * comm->rankNum + peer) * stride, Shape(1, 1, 1, 1, elements),
            Stride(elements, elements, elements, elements, 1)
        );
        const uint32_t before = *ctx.completion_count;
        if (!send_request_entry(ctx, SdmaTget(dst, src, scratch, workspace))) return;
        if (*ctx.completion_count != before + 1) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return;
        }
        const uint32_t record = static_cast<uint32_t>(meta[3]);
        meta[8 + record * 2] = ctx.completion_entries[before].addr;
        meta[9 + record * 2] = ctx.completion_entries[before].backend_cookie;
        ++meta[3];
        meta[14] += static_cast<uint32_t>(signals[peer * kLine]);
        save_expected_notification_counter(ctx, signals + peer * kLine, 1);
        ++meta[4];
    }
    meta[2] = *ctx.completion_count;
    meta[5] = *ctx.completion_error_code;
    publish(meta, kMeta * sizeof(uint64_t));
    auto *registered = data<int32_t>(args, 7) + execution * kLine;
    *registered = 1;
    publish(registered, kLine * sizeof(int32_t));
}
