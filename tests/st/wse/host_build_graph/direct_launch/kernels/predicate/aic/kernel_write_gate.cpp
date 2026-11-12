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
 * Producer of the predicate value for the predicated_dispatch scene tests.
 * Writes the scalar arg into gate[0] (INT32). A downstream task carries a
 * dispatch predicate on gate[0]; the scheduler reads this value at the dispatch
 * point (after this producer has completed) to decide whether to dispatch.
 *
 * args[0] = gate tensor (INOUT, INT32); args[1] = scalar gate value.
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
    int64_t gate_value = args[1];  // scalar arg follows the tensor args

    __gm__ int32_t *out = reinterpret_cast<__gm__ int32_t *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    out[0] = static_cast<int32_t>(gate_value);
    dcci(&out[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
