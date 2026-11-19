// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// M80 MXFP8 projection with an N64 tile and resident scales.
#include "mxfp8_projection.hpp"

using namespace mxfp8_projection;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    project<5120, 1280, 64>(
        data<F>(args, 0), data<F>(args, 1), data<S>(args, 2), data<S>(args, 3),
        data<bfloat16_t>(args, 4), args[5], 1280, get_block_idx(args) * 64);
}
