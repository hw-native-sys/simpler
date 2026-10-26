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
 * kernel_stamp_block, but the block holds its core until the whole cohort has
 * been claimed.
 *
 * The one-block-per-core rule is enforced by a core scanning its own dispatch
 * slots for the task it is about to claim another block of, and a slot is
 * cleared the moment its block completes. So a block that returns immediately
 * stops being visible to that scan, and its core is free to take a second
 * block of the same cohort -- which is correct, because a cohort whose blocks
 * do not wait on each other has nothing to deadlock on.
 *
 * That makes "every core took exactly one" untestable with a kernel that
 * returns at once: the answer depends on whether the claim round finished
 * before the first block did. The spin here is what a real cohort's barrier
 * would do -- hold the slot -- so the scan sees every block that has been
 * placed and the distribution is a property rather than a race.
 *
 * Args: [0] out tensor.
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

// One 64-byte line per block.
constexpr int32_t FLOATS_PER_CACHE_LINE = 16;
// Orders of magnitude longer than a claim round, which is one pass per core
// over a queue of one task. Nothing waits on this, so it costs the case its
// own duration and nothing else.
constexpr int64_t HOLD_ITERATIONS = 200000;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    const int32_t block_idx = get_block_idx(args);
    const int32_t block_num = get_block_num(args);
    const int32_t offset = block_idx * FLOATS_PER_CACHE_LINE;

    volatile int64_t sink = 0;
    for (int64_t i = 0; i < HOLD_ITERATIONS; ++i) sink += i;
    (void)sink;

    // The count goes in alongside the index: a dispatch that lost block_num
    // would still stamp the right indices while telling every block it was
    // alone, and nothing downstream would notice.
    out[offset] = static_cast<float>(block_idx);
    out[offset + 1] = static_cast<float>(block_num);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
