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
 * A block of the producer: writes its own index into its own slot, after a
 * delay that every block but the first pays.
 *
 * The delay is what gives the test its window. A consumer released when the
 * first block finished would start while the rest are still spinning, and read
 * the sentinel they have not yet overwritten. Without it every block completes
 * at about the same moment and a consumer released too early still reads the
 * finished values.
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int64_t repeats = args[1];

    const int32_t block_idx = get_block_idx(args);
    if (block_idx != 0) {
        volatile int64_t sink = 0;
        for (int64_t i = 0; i < repeats; ++i) sink += i;
        (void)sink;
    }

    const int32_t offset = block_idx * FLOATS_PER_CACHE_LINE;
    out[offset] = static_cast<float>(block_idx);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
