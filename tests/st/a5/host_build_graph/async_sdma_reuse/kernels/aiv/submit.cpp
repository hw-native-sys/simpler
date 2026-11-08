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
using namespace pto;
using namespace sdma_series;
namespace {
struct Receipt {
    __gm__ uint64_t *meta;
    inline __aicore__ void operator()(uint32_t index, uint32_t count, const SdmaCompletionReceiptRecord &record) const {
        if (count == 0 || count > kQueueNum || index >= count) {
            add(meta + ERROR, 1);
            return;
        }
        meta[kReceiptBase + 2 * index] = record.address;
        meta[kReceiptBase + 2 * index + 1] = record.expected;
        flush(meta + kReceiptBase + 2 * index);
        if (index + 1 == count) add(meta + RECEIPT_COUNT, count);
    }
};
template <int N>
inline __aicore__ bool post(__gm__ int64_t *args, __gm__ uint64_t *meta) {
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint64_t peer = static_cast<uint64_t>(args[5]);
    if (comm->rankNum != 2 || peer >= comm->rankNum || peer == comm->rankId) return false;
    auto *source = data<float>(args[0]);
    const uint64_t relative = reinterpret_cast<uint64_t>(source) - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + relative);
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    Tile<TileType::Vec, uint8_t, 1, 32> scratch;
    TASSIGN(scratch, SDMA_SCRATCH_ALIGNMENT);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    return send_request_entry(
        args, SdmaTget(Global(data<float>(args[1])), Global(remote), scratch, workspace, 0, kQueueNum), Receipt{meta}
    );
}
}  // namespace
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *all = data<uint64_t>(args[3]);
    auto *meta = all + args[6] * kMetaWords;
    add(meta + SUBMIT_VISITS, 1);
    auto ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx)) {
        add(meta + ERROR, 1);
        return;
    }
    meta[TASK_TOKEN] = TaskId::to_uint64(ctx.task_token);
    flush(meta + TASK_TOKEN);
    sample_prior(all, args[6], args[8]);
    const bool ok = args[7] == kLargeElements ? post<kLargeElements>(args, meta) :
                    args[7] == 8              ? post<8>(args, meta) :
                                                false;
    if (!ok) {
        add(meta + ERROR, 1);
        async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        pto2::detail::defer_flush(ctx);
    }
}
