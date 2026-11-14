// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// O-B: MXFP8 [M, K] x [K, 5120], with a bounded BF16 output row count.
#include "mxfp8_projection.hpp"

using namespace mxfp8_projection;

template <int K, int MT = 80>
__aicore__ inline void project_ob(__gm__ int64_t* args) {
    constexpr int N_TILE = K == 8192 ? 160 : 256;
    constexpr int COLUMNS = 5120 / N_TILE;
    const int tasks = (args[5] + MT - 1) / MT * COLUMNS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        const int row = task / COLUMNS * MT;
        const int n = task % COLUMNS * N_TILE;
        const int rows = args[5] - row < MT ? args[5] - row : MT;
        project<K, 5120, N_TILE, MT>(
            data<F>(args, 0) + row * K, data<F>(args, 1),
            data<S>(args, 2) + row * (K / 32), data<S>(args, 3),
            data<bfloat16_t>(args, 4) + row * 5120, rows, 5120, n);
    }
}

template <int MT>
__aicore__ inline void project_width(__gm__ int64_t* args) {
    if (args[6] > 4096) {
        project_ob<8192, MT>(args);
    } else if (args[6] > 2048) {
        project_ob<4096, MT>(args);
    } else if (args[6] > 1024) {
        project_ob<2048, MT>(args);
    } else {
        project_ob<1024, MT>(args);
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    if (args[7] == 256) {
        project_width<256>(args);
    } else {
        project_width<80>(args);
    }
}
