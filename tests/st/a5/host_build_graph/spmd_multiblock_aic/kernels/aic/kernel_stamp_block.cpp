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
 * Each block stamps its own index into its own slot of args[0], so the result
 * is sensitive both to how many blocks ran and to which index each one carried.
 * A kernel that writes one fixed cell from every block cannot tell a run that
 * dispatched every block from one that dispatched a single block twice.
 *
 * Slots are a cache line apart because blocks land on different cores and each
 * core flushes the line it wrote; packing them would make neighbouring blocks
 * race for one line.
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    const int32_t block_idx = get_block_idx(args);
    const int32_t block_num = get_block_num(args);
    const int32_t offset = block_idx * FLOATS_PER_CACHE_LINE;

    // The count goes in alongside the index: a dispatch that lost block_num
    // would still stamp the right indices while telling every block it was
    // alone, and nothing downstream would notice.
    out[offset] = static_cast<float>(block_idx);
    out[offset + 1] = static_cast<float>(block_num);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
