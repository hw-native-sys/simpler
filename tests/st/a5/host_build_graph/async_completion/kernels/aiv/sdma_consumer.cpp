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

using namespace pto;
using namespace async_case;

template <int N>
__aicore__ void consume(__gm__ float *source, __gm__ float *output, int64_t elements) {
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    using Vec = Tile<TileType::Vec, float, 1, N>;
    Vec input, result;
    TASSIGN(input, 0);
    TASSIGN(result, 0x10000);
    for (int64_t base = 0; base < elements; base += N) {
        Global src(source + base), dst(output + base);
        TLOAD(input, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(result, input, 7.0f);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst, result);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *meta = data<uint64_t>(args[3]);
    if (observe(meta) == 0 || !sdma_ready(meta)) {
        add(meta + 48, 1);
        return;
    }
    if (args[5] == 8) consume<8>(data<float>(args[1]), data<float>(args[2]), args[5]);
    else if (args[5] != 0) consume<256>(data<float>(args[1]), data<float>(args[2]), args[5]);
}
