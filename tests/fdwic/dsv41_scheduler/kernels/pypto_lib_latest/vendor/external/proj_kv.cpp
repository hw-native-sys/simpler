// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// MXFP8 projection with resident scales and bounded BF16 output rows.
#include "mxfp8_projection.hpp"

using namespace mxfp8_projection;

template <int MT>
__aicore__ inline void project_kv(__gm__ int64_t* args) {
    constexpr int NT = MT == 80 ? 64 : 256;
    constexpr int COLUMNS = 512 / NT;
    const int tasks = (args[5] + MT - 1) / MT * COLUMNS;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        const int row = task / COLUMNS * MT;
        const int n = task % COLUMNS * NT;
        const int rows = args[5] - row < MT ? args[5] - row : MT;
        project<5120, 512, NT, MT>(
            data<F>(args, 0) + row * 5120, data<F>(args, 1),
            data<S>(args, 2) + row * (5120 / 32), data<S>(args, 3),
            data<bfloat16_t>(args, 4) + args[6] * 512 + row * 512, rows, 512, n);
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    if (args[7] == 256) {
        project_kv<256>(args);
    } else {
        project_kv<80>(args);
    }
}
