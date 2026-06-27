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
 * SPMD Context Read Kernel (AIV version)
 *
 * Reads SPMD context via Get* accessors and writes values to the shared
 * output tensor.  AIV uses get_sub_block_id to determine its lane (0=left,
 * 1=right) and writes at cache line (1 + sub_block_id) to avoid
 * overlapping with the AIC slot at cache line 0.
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

    int32_t sub_block_id = get_sub_block_id(args);
    // AIV writes at cache line (1 + sub_block_id), skipping AIC's cache line 0
    int32_t offset = (1 + sub_block_id) * FLOATS_PER_CACHE_LINE;

    out[offset + 0] = static_cast<float>(get_block_idx(args));
    out[offset + 1] = static_cast<float>(get_block_num(args));
    out[offset + 2] = static_cast<float>(sub_block_id);

    // Flush this cache line to HBM so host can read the output.
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
