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
 * SPMD producer kernel: block i writes out[i] = i + 1.
 *
 * Each block fills exactly one element of the output, so the full output is
 * complete only after EVERY block has run. A consumer that reads the whole
 * output is correct only if it is released after the last block finishes —
 * which is what the speculative completion-path doorbell must guarantee.
 *
 * Args:
 *   args[0] = output Tensor* (OUTPUT, block_num float32 elements)
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
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    // Isolate each block on its own cache line: two AICore cores must never
    // write the same line (each flushes the whole line -> last-writer-wins).
    // See docs/aicore-kernel-programming.md "Each block must write to its own
    // cache line". CL = 64B / sizeof(float) on a2a3.
    constexpr int CL = 16;
    int32_t block_idx = get_block_idx(args);
    out[block_idx * CL] = static_cast<float>(block_idx + 1);

    dcci(&out[block_idx * CL], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
