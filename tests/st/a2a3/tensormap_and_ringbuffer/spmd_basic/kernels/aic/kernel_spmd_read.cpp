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
 * SPMD Context Read Kernel (AIC version)
 *
 * Reads SPMD local context (block_idx, block_num) and writes values to
 * cache line 0 of the shared output tensor.  AIC does not use
 * get_sub_block_id (sub_block_id is only meaningful for AIV).
 *
 * Args:
 *   args[0] = output Tensor* (OUTPUT, 48 float32 elements)
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

// Cache line = 64B = 16 float32.  Each slot owns one cache line.
static constexpr int32_t FLOATS_PER_CACHE_LINE = 16;

// dcci + constants: CCEC provides these as builtins; provide fallbacks for sim.
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

    // AIC writes at fixed cache line 0 (no sub_block_id needed)
    out[0] = static_cast<float>(get_block_idx(args));
    out[1] = static_cast<float>(get_block_num(args));

    // Flush this cache line to HBM so host can read the output.
    dcci(&out[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
