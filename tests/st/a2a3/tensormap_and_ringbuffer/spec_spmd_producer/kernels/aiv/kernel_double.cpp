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
 * Single-block consumer: out[i] = 2 * in[i] for i in [0, count).
 *
 * Reads the ENTIRE producer output. If released before the SPMD producer's
 * last block runs, some in[i] are still zero and the result is wrong — so this
 * doubles as the premature-release detector for SPMD-as-producer speculation.
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
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    // Invalidate the producer's output so this load sees post-doorbell data.
    dcci(in, ENTIRE_DATA_CACHE);

    // The SPMD producer wrote block i at stride CL (its own cache line); read
    // back the per-block values. out is single-block (this kernel), so it can
    // stay compact. CL = 64B / sizeof(float) on a2a3.
    constexpr int CL = 16;
    uint32_t n = in_tensor->shapes[0] / CL;
    for (uint32_t i = 0; i < n; i++) {
        out[i] = 2.0f * in[i * CL];
    }
    dcci(out, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}
