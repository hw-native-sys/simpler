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
// Kernel Function: partials_reduce_hidden
//
// Reducer over a hidden-width accumulator (row stride 5120). Used for
// down_acc_all, whose bands have no writer other than the split-K partials this
// sums, so the result is stored rather than accumulated.

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

#include <pto/pto-inst.hpp>
#include "tensor.h"

#include "partials_reduce.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *part_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *part = reinterpret_cast<__gm__ float *>(part_tensor->buffer.addr) + part_tensor->start_offset;

    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    union {
        uint64_t u64;
        int64_t val;
    } splits_conv;
    splits_conv.u64 = args[2];
    int64_t splits = splits_conv.val;

    union {
        uint64_t u64;
        int64_t val;
    } band_width_conv;
    band_width_conv.u64 = args[3];
    int64_t band_width = band_width_conv.val;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    reduce_partials_into_band<5120, pto::AtomicType::AtomicNone>(part, out, splits, band_width);
#else
    (void)part;
    (void)out;
    (void)splits;
    (void)band_width;
#endif  // __DAV_VEC__

    pipe_barrier(PIPE_ALL);
}
