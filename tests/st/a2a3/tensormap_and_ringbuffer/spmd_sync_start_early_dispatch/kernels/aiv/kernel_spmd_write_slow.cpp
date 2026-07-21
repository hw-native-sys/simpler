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
 * SPMD Multi-Block Write Kernel with a spin delay (AIV)
 *
 * Same result as the plain write kernel, but spins for `spin_iters` before the
 * store. A slow producer keeps its cores occupied long enough that a dependent
 * consumer is processed as an early-dispatch candidate WHILE the producer is still
 * running — the only way a trivial-kernel scene exercises the speculative
 * gated-dispatch path (a fast producer finishes before the consumer is ever staged).
 *
 *   out[(base_cl + block_idx) * FLOATS_PER_CACHE_LINE] = float(block_idx)
 *
 * Args:
 *   args[0] = output Tensor* (INOUT)
 *   args[1] = scalar: base_cl (starting cache line index for this task)
 *   args[2] = scalar: spin_iters (0 = no delay)
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

#include "intrinsic.h"

static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;

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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int32_t base_cl = static_cast<int32_t>(args[1]);
    int32_t spin_iters = static_cast<int32_t>(args[2]);
    int32_t block_idx = get_block_idx(args);
    int32_t offset = (base_cl + block_idx) * FLOATS_PER_CACHE_LINE;

    volatile int32_t acc = 0;
    for (int32_t i = 0; i < spin_iters; i++) {
        acc++;  // ++ not += i: += i overflows int32 for large spin_iters (UB); volatile keeps the loop
    }
    (void)acc;

    out[offset] = static_cast<float>(block_idx);

    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
