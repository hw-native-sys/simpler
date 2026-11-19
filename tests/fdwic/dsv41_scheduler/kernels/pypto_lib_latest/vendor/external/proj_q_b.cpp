// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// NZ attention Q-B projection.
#include "q_lora_projection.hpp"


#if defined(__DAV_CUBE__)
template <int N, int Workers, int MT>
__aicore__ inline void dispatch_qb(__gm__ int64_t* args) {
    int rows = args[7];
    if (rows <= MT) {
        project_qb<N, Workers, false, true, MT>(args);
        return;
    }
    int tasks = (rows + MT - 1) / MT * 8;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int first = task / 8 * MT;
        int count = rows - first < MT ? rows - first : MT;
        project_qb<N, 0, false, true, MT>(args, first, count, task % 8, 8);
    }
}

template <int MT>
__aicore__ inline void dispatch_width(__gm__ int64_t* args) {
    if (args[8] > 16384) dispatch_qb<32768, 32, MT>(args);
    else if (args[8] > 8192) dispatch_qb<16384, 32, MT>(args);
    else if (args[8] > 4096) dispatch_qb<8192, 32, MT>(args);
    else dispatch_qb<4096, 16, MT>(args);
}
#elif defined(__DAV_VEC__)
template <int MT>
__aicore__ inline void dispatch_width(__gm__ int64_t* args) {
    int rows = args[7];
    if (rows <= MT) {
        project_qb_rope<MT>(args);
        return;
    }
    int tasks = (rows + MT - 1) / MT * 8;
    for (int task = get_block_idx(args); task < tasks; task += get_block_num(args)) {
        int first = task / 8 * MT;
        int count = rows - first < MT ? rows - first : MT;
        project_qb_rope<MT>(args, first, count, task % 8, 8);
    }
}
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    if (args[10] == 128) {
        dispatch_width<128>(args);
    } else {
        dispatch_width<80>(args);
    }
}
