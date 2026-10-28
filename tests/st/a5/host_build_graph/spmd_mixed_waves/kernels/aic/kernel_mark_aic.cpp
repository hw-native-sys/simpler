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
 * The AIC half of a wave task: block 0 marks its task's AIC slot.
 *
 * Only block 0 writes, so a task of any width leaves exactly one mark and the
 * expected output does not depend on how many blocks the topology gave it --
 * which is what keeps this graph's result computable while its block counts
 * are sized from the run. How many blocks actually ran is asserted by the
 * cases built for that; here the question is whether the scheduler survives
 * the mix.
 *
 * Args: [0] out tensor, [1] task index.
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
constexpr int32_t SLOTS_PER_TASK = 4;
constexpr int32_t AIC_SLOT = 0;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    if (get_block_idx(args) != 0) return;
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t task_index = static_cast<int32_t>(args[1]);

    const int32_t offset = (task_index * SLOTS_PER_TASK + AIC_SLOT) * FLOATS_PER_CACHE_LINE;
    out[offset] = static_cast<float>(task_index);
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
