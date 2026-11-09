// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// O-B: MXFP8 [80, K] x [K, 5120], with a bounded BF16 output row count.
#include "mxfp8_projection.hpp"

using namespace mxfp8_projection;

template <int K>
__aicore__ inline void project_ob(__gm__ int64_t* args) {
    project<K, 5120>(
        data<F>(args, 0), data<F>(args, 1), data<S>(args, 2), data<S>(args, 3),
        data<bfloat16_t>(args, 4), args[5], 5120, get_block_idx(args) * 256);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    // Ordered TP width ranges keep the entry free of jump tables.
    if (args[6] > 4096) {
        project_ob<8192>(args);
    } else if (args[6] > 2048) {
        project_ob<4096>(args);
    } else if (args[6] > 1024) {
        project_ob<2048>(args);
    } else {
        project_ob<1024>(args);
    }
}
