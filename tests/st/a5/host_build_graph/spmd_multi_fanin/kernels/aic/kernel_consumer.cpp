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
 * One block of the SPMD consumer: sums every producer slot and writes that sum
 * plus its own index.
 *
 * Summing all of them is what gives the four dependencies one observable
 * answer: a block released before any producer finished reads that producer's
 * sentinel, and every block of the consumer carries the error into the golden.
 *
 * Args: [0] out tensor, [1] producer slot count, [2] consumer base slot.
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
    const int32_t producer_slots = static_cast<int32_t>(args[1]);
    const int32_t consumer_base = static_cast<int32_t>(args[2]);

    const int32_t block_idx = get_block_idx(args);

    float total = 0.0f;
    // Every producer slot was written by another core, so this one's view of
    // them is whatever it last had; the reader invalidates before reading, the
    // same way fanout_worker.cpp reads its producer's slot.
    for (int32_t slot = 0; slot < producer_slots; ++slot)
        dcci(&out[slot * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
    for (int32_t slot = 0; slot < producer_slots; ++slot)
        total += out[slot * FLOATS_PER_CACHE_LINE];

    const int32_t offset = (consumer_base + block_idx) * FLOATS_PER_CACHE_LINE;
    out[offset] = total + static_cast<float>(block_idx);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
