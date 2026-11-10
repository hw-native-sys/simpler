// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// O-A: grouped MXFP8 [80, 4096] x [4096, 1024] to concatenated BF16 columns.
#include "mxfp8_projection.hpp"

using namespace mxfp8_projection;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    int workers = get_block_num(args);
    for (int tile = get_block_idx(args); tile < args[5] / 256; tile += workers) {
        int group = tile / 4;
        project<4096, 1024>(
            data<F>(args, 0) + group * 80 * 4096,
            data<F>(args, 1) + group * 4096 * 1024,
            data<S>(args, 2) + group * 80 * 128,
            data<S>(args, 3) + group * 128 * 1024,
            data<bfloat16_t>(args, 4) + group * 1024, 80, args[5], tile % 4 * 256);
    }
}
