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

/**
 * One slot per (block, lane) of a Mix task: a block's two AIV subtasks write
 * neighbouring cache lines, so the result says both which blocks ran and which
 * lane each subtask was dispatched as.
 *
 * sub_block_id is the lane within the cluster, which for a Mix task the
 * scheduler derives from the cluster lane its subtask was placed on rather than
 * from any hardware register.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

constexpr int32_t FLOATS_PER_CACHE_LINE = 16;
constexpr int32_t LANES_PER_BLOCK = 2;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    const int32_t block_idx = get_block_idx(args);
    const int32_t block_num = get_block_num(args);
    const int32_t lane = get_sub_block_id(args);
    const int32_t offset = (block_idx * LANES_PER_BLOCK + lane) * FLOATS_PER_CACHE_LINE;

    out[offset] = static_cast<float>(block_idx);
    out[offset + 1] = static_cast<float>(block_num);
    out[offset + 2] = static_cast<float>(lane);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
