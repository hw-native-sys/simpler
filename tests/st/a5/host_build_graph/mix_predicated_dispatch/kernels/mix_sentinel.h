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
 * Mix subtask sentinel writer. The three subtasks of one Mix task share one
 * argument list and each writes only its own slot, so a per-subtask sentinel
 * identifies exactly which subtasks ran.
 *
 * args[0..2] = the three sentinel tensors (INOUT, FP32).
 * MIX_SENTINEL_SLOT selects which one this translation unit writes, and
 * MIX_SENTINEL_VALUE is the value it writes.
 */

#pragma once

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

static __aicore__ void mix_sentinel_write(__gm__ int64_t *args, int32_t slot, float value) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[slot]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    out[0] = value;
    dcci(&out[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
