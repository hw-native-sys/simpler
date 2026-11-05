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
 * One block of one stage of an SPMD chain: sums the whole previous stage and
 * writes that sum plus its own index into its own slot.
 *
 * Reading every slot of the previous stage is what makes the stage boundary
 * observable: one slot still holding the sentinel changes this block's output,
 * and the output feeds the next stage, so the corruption reaches the golden
 * rather than being overwritten. Stage 0 has no previous stage and sums
 * nothing, which makes the same code the whole chain.
 *
 * The spin is proportional to the block index, so the blocks of a stage finish
 * spread out instead of together. A stage declared done on any block but its
 * last then has a window in which the next stage is already reading slots that
 * are still sentinel.
 *
 * Args: [0] out tensor, [1] block_num, [2] stage, [3] repeats.
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t block_num = static_cast<int32_t>(args[1]);
    const int32_t stage = static_cast<int32_t>(args[2]);
    const int64_t repeats = args[3];

    const int32_t block_idx = get_block_idx(args);

    float previous = 0.0f;
    if (stage > 0) {
        const int32_t base = (stage - 1) * block_num;
        // The previous stage's slots were written by other cores, so this one's
        // view of them is whatever it last had. Invalidating before the read is
        // the reader's job here, the same way fanout_worker.cpp reads its
        // producer's slot.
        for (int32_t block = 0; block < block_num; ++block)
            dcci(&out[(base + block) * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
        dsb((mem_dsb_t)0);
        for (int32_t block = 0; block < block_num; ++block)
            previous += out[(base + block) * FLOATS_PER_CACHE_LINE];
    }

    volatile int64_t sink = 0;
    for (int32_t round = 0; round < block_idx; ++round)
        for (int64_t i = 0; i < repeats; ++i)
            sink += i;
    (void)sink;

    const int32_t offset = (stage * block_num + block_idx) * FLOATS_PER_CACHE_LINE;
    out[offset] = previous + static_cast<float>(block_idx);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
