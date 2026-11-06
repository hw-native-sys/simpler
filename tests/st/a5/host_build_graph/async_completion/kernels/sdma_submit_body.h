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

#pragma once

#include "sdma.h"
#include "async_kernel_api.h"
#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"

using namespace pto;
using namespace async_case;

namespace {
struct PublicRecordReceipt {
    __gm__ uint64_t *meta;
    inline __aicore__ void operator()(uint32_t index, uint32_t count, const SdmaCompletionReceiptRecord &record) const {
        meta[kSdmaAddresses + 2 * index] = record.address;
        meta[kSdmaAddresses + 2 * index + 1] = record.expected;
        flush(meta + kSdmaAddresses + 2 * index);
        if (index + 1 == count) add(meta + 112, count);
    }
};

template <int N, bool Declared>
__aicore__ void submit(__gm__ int64_t *args, AsyncCtx &ctx, __gm__ uint64_t *meta) {
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    auto *local = data<float>(args[0]);
    const uint64_t offset = reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId];
    const uint64_t peer = static_cast<uint64_t>(args[6]);
    bool failed = comm->rankNum != 2 || peer >= comm->rankNum || peer == comm->rankId;
    if (!failed) {
        auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + offset);
        using ShapeType = Shape<1, 1, 1, 1, N>;
        using StrideType = pto::Stride<N, N, N, N, 1>;
        using Global = GlobalTensor<float, ShapeType, StrideType>;
        Tile<TileType::Vec, uint8_t, 1, 32> scratch;
        TASSIGN(scratch, SDMA_SCRATCH_ALIGNMENT);
        auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
        const auto request = SdmaTget(Global(data<float>(args[1])), Global(remote), scratch, workspace);
        if constexpr (Declared) {
            failed = !send_request_entry(args, request, PublicRecordReceipt{meta});
        } else {
            failed = !send_request_entry(ctx, request);
            if (!failed) {
                const uint32_t count = *ctx.completion_count;
                for (uint32_t index = 0; index < count; ++index) {
                    const auto &entry = ctx.completion_entries[index];
                    PublicRecordReceipt{meta}(index, count, {entry.addr, entry.backend_cookie});
                }
            }
        }
    }
    if (failed) {
        add(meta + 32, 1);
        // Preserve the first error from the real SDK/AsyncCtx writer.
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
    }
}

}  // namespace

template <bool Declared>
__aicore__ void run(__gm__ int64_t *args) {
    auto *meta = data<uint64_t>(args[3]);
    auto ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx)) {
        add(meta + 16, 1);
    } else if (args[5] == 8) {
        submit<8, Declared>(args, ctx, meta);
    } else if (args[5] == 1024 * 1024) {
        submit<1024 * 1024, Declared>(args, ctx, meta);
    } else if (args[5] != 0) {
        add(meta + 32, 1);
    }
    add(meta, 1);
}
