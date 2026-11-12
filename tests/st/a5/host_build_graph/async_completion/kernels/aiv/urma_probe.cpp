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

#include "../urma.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace async_case;
    auto *meta = data<uint64_t>(args[3]);
    const bool returned = await_at_least(meta, 1);
    if (!returned) add(meta + 32, 1);
    const uint64_t started = now();
    while (returned) {
        const uint64_t sampled_at = now();
        if (urma_ready(meta)) break;
        add(meta + 64, 1);
        // Sample-start time makes a post-return observation conservative.
        meta[80] = sampled_at;
        flush(meta + 80);
        if (now() - started > 2000 * ticks_per_ms()) {
            add(meta + 32, 1);
            break;
        }
    }
    auto *cq = reinterpret_cast<__gm__ pto::comm::urma::UrmaCqCtx *>(observe(meta + 128));
    if (cq != nullptr) {
        auto *sq = reinterpret_cast<__gm__ pto::comm::urma::UrmaWQCtx *>(observe(meta + 192));
        const uint32_t sequence = static_cast<uint32_t>(observe(meta + 144)) - 1;
        auto *word = reinterpret_cast<__gm__ uint32_t *>(cq->bufAddr + 64 * (sequence & (cq->depth - 1)));
        meta[256] = ld_dev(word, 0);
        meta[257] = ld_dev(reinterpret_cast<__gm__ uint32_t *>(cq->tailAddr), 0);
        meta[258] = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->headAddr), 0);
        meta[259] = ld_dev(reinterpret_cast<__gm__ uint32_t *>(sq->tailAddr), 0);
        meta[260] = cq->depth;
        if (args[5] != 0) {
            auto *destination = data<uint32_t>(args[1]);
            meta[261] = ld_dev(destination, 0);
            meta[262] = ld_dev(destination + args[5] - 1, 0);
        }
        flush(meta + 256);
    }
}
