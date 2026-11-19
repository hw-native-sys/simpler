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
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace mixed_async;
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[9]);
    const uint32_t execution = get_block_idx(args) * 3;
    auto *meta = data<uint64_t>(args, 8) + execution * kMeta;
    auto *signals = data<int32_t>(args, 1) + execution * comm->rankNum * kLine;
    auto ctx = get_async_ctx(args);
    meta[0] = 1;
    meta[7] = get_block_idx();
    if (!async_ctx_is_deferred(ctx) || *ctx.completion_count != 0) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
        if (peer == comm->rankId) continue;
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
