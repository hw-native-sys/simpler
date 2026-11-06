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

#include "../sdma.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    using namespace async_case;
    auto *meta = data<uint64_t>(args[0]);
    if (!await_at_least(meta, 1)) {
        add(meta + 32, 1);
        return;
    }
    const uint64_t started = now();
    while (true) {
        const uint64_t sampled_at = now();
        if (sdma_ready(meta)) break;
        add(meta + 64, 1);
        // Sample-start time makes a post-return observation conservative.
        meta[80] = sampled_at;
        flush(meta + 80);
        if (now() - started > 2000 * ticks_per_ms()) {
            add(meta + 32, 1);
            break;
        }
    }
}
