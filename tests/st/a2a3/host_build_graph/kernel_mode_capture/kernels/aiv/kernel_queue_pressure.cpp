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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "tensor.h"  // NOLINT(build/include_subdir)

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *data = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    const int32_t task = static_cast<int32_t>(args[2]);
    const int32_t block = get_block_idx(args);
    const int32_t offset = (task * static_cast<int32_t>(args[3]) + block) * 16;
    const float value = static_cast<float>(args[1] + task * 256 + block);
    for (int i = 0; i < 16; ++i)
        data[offset + i] = value;
    dcci(data + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
