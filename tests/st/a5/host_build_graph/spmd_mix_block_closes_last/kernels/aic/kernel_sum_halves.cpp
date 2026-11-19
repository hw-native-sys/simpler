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
 * Reads both halves of every block of a Mix producer and writes their sum.
 *
 * Reading both halves is what makes the consumer sensitive to how a block is
 * closed. A block is done only when all of its subtasks have reported, and the
 * task only when every block has; a consumer released on a subtask rather than
 * a block would find the fast halves written and the slow ones not.
 *
 * Args: [0] out tensor, [1] block_num.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

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
constexpr int32_t SLOTS_PER_BLOCK = 2;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t block_num = static_cast<int32_t>(args[1]);

    float total = 0.0f;
    // Both halves of every block were written by other cores, so this one's
    // view of them is whatever it last had; the reader invalidates before
    // reading, the same way fanout_worker.cpp reads its producer's slot.
    for (int32_t slot = 0; slot < block_num * SLOTS_PER_BLOCK; ++slot)
        dcci(&out[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
    for (int32_t slot = 0; slot < block_num * SLOTS_PER_BLOCK; ++slot)
        total += out[slot * FLOATS_PER_CACHE_LINE];

    const int32_t offset = block_num * SLOTS_PER_BLOCK * FLOATS_PER_CACHE_LINE;
    out[offset] = total;
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
