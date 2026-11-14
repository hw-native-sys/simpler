// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Merged MXFP8 gate/up projection: one 80-row tile per (projection, N tile) work item.
#include "expert_shared_mxfp8.hpp"

using namespace expert_shared_mxfp8;

namespace {
constexpr int D = 5120;
constexpr int MOE_INTER = 2304;
constexpr int N_OUT = 2 * MOE_INTER;
constexpr int NT = 256;
constexpr int NTILES = MOE_INTER / NT;
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    const int rows = static_cast<int>(args[7]);
    const int block = get_block_idx(args);
    const int blocks = get_block_num(args);
    const int tasks = (rows + 79) / 80 * 2 * NTILES;
    for (int work = block; work < tasks; work += blocks) {
        const int row = work / (2 * NTILES) * 80;
        const int projection = work / NTILES % 2;
        const int n = work % NTILES * NT;
        const int active = rows - row < 80 ? rows - row : 80;
        __gm__ F* weight = data<F>(args, projection ? 4 : 2);
        __gm__ S* weight_scale = data<S>(args, projection ? 5 : 3);
        __gm__ bfloat16_t* output = data<bfloat16_t>(args, 6) + row * N_OUT + projection * MOE_INTER;
        project<D, MOE_INTER, NT>(
            data<F>(args, 0) + row * D, weight, data<S>(args, 1) + row * (D / 32), weight_scale, output, active, N_OUT, n);
    }
}
