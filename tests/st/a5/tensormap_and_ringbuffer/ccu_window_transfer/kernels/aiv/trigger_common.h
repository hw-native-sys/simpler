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
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "tensor.h"

inline __aicore__ void trigger_window(__gm__ int64_t *args, uint32_t offset, uint32_t ccu_id) {
    const auto *comm = reinterpret_cast<const __gm__ CommContext *>(args[offset + 1]);
    AsyncCtx ctx = get_async_ctx(args);
    if (args[offset + 3] == 0) {
        // Length zero is this test kernel's exit-only request.
        const uint64_t words[7]{};
        (void)submit_ccu_mission(ctx, comm, ccu_id, words, sizeof(words));
        return;
    }
    const uint64_t round = args[offset + 2] - (ccu_id == 1 ? 2 : 1);
    const uint64_t displacement = (round % 8) * 64;
    const uint64_t length = ((round % 16) + 1) * 64;
    using Arg = simpler::ccu::CcuCallArg;
    const uint32_t target = (comm->rankId + comm->rankNum - 1) % comm->rankNum;
    const uint32_t source = (comm->rankId + 1) % comm->rankNum;
    const Arg call_args[]{
        {Arg::Kind::Window, comm->rankId, ccu_id * 8192 + displacement, length},
        {Arg::Kind::Window, target, 16384 + ccu_id * 8192 + displacement, length},
        {Arg::Kind::Scalar, 0, length, 0},
        {Arg::Kind::Scalar, 0, target, 0},
        {Arg::Kind::Scalar, 0, source, 0}
    };
    uint64_t words[10]{};
    uint64_t bytes = 0;
    const bool resolved = simpler::ccu::resolve_call_args(comm, call_args, words, bytes);
    auto *audit = reinterpret_cast<__gm__ Tensor *>(args[offset]);
    auto *out = reinterpret_cast<__gm__ uint64_t *>(audit->buffer.addr) + audit->start_offset;
    for (uint32_t i = 0; i < 7; ++i)
        out[i] = words[i];
    out[7] = resolved && bytes == 56 && submit_ccu_mission(ctx, comm, ccu_id, call_args);
    dcci(out, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
