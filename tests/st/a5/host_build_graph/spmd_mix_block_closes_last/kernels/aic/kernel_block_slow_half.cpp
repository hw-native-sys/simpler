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
 * The slow half of a Mix block: spins, then marks its slot.
 *
 * A Mix block closes only when all of its subtasks have reported, and a block
 * whose halves finish together cannot tell that from closing on the first one.
 * This kernel is paired with one that returns immediately, so the gap between
 * the first subtask of a block finishing and the block itself finishing is
 * wide enough for a successor released too early to be caught inside it.
 *
 * Args: [0] out tensor, [1] repeats.
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
constexpr int32_t SLOTS_PER_BLOCK = 2;
constexpr int32_t SLOW_SLOT = 0;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int64_t repeats = args[1];

    volatile int64_t sink = 0;
    for (int64_t i = 0; i < repeats; ++i)
        sink += i;
    (void)sink;

    const int32_t block_idx = get_block_idx(args);
    const int32_t offset = (block_idx * SLOTS_PER_BLOCK + SLOW_SLOT) * FLOATS_PER_CACHE_LINE;
    out[offset] = static_cast<float>(block_idx);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
