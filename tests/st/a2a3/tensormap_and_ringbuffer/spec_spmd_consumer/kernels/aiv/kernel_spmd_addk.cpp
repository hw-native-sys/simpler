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
 * SPMD consumer: block i computes out[i] = in[i] + 10.
 *
 * Each block reads its own element of the producer's output. As a pre-staged
 * SPMD consumer some blocks are gated on the doorbell and some dispatch
 * normally — all must read post-producer data, so the result is correct only if
 * every block runs after the producer completes.
 *
 * Args:
 *   args[0] = input  Tensor* (INPUT)
 *   args[1] = output Tensor* (OUTPUT)
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

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef ENTIRE_DATA_CACHE
#define ENTIRE_DATA_CACHE 0
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int32_t block_idx = get_block_idx(args);

    // Invalidate so this load sees the producer's post-doorbell data.
    dcci(in, ENTIRE_DATA_CACHE);
    // Each SPMD block writes its own cache line: two cores must never write the
    // same line (each flushes the whole line -> last-writer-wins). The producer
    // output `in` is compact (single-block writer), so reads stay compact.
    // CL = 64B / sizeof(float) on a2a3. See docs/aicore-kernel-programming.md.
    constexpr int CL = 16;
    out[block_idx * CL] = in[block_idx] + 10.0f;
    dcci(&out[block_idx * CL], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
