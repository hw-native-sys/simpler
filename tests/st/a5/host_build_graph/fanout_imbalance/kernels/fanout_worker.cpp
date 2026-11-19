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
 * Fanout worker. Each task writes `task_id + 1` into its own stride-8 slot, or
 * the negation of it when its declared producer has not published yet.
 *
 * The task then burns `repeats` iterations of arithmetic before returning. The
 * backlog this scene test creates is only observable while tasks are still
 * queued, so a task that returns immediately would drain the queue faster than
 * the distribution can be sampled.
 *
 * args[0] = state tensor (INOUT, INT64, stride 8 per task)
 * args[1] = task_id, args[2] = producer id (-1 for the root), args[3] = repeats
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int64_t *state = reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
    const int64_t task_id = args[1];
    const int64_t producer = args[2];
    const int64_t repeats = args[3];

    bool ready = true;
    if (producer >= 0) {
        dcci(&state[producer * 8], cache_line_t::SINGLE_CACHE_LINE);
        dsb((mem_dsb_t)0);
        ready = state[producer * 8] == producer + 1;
    }

    const int64_t slot = task_id * 8;
    int64_t acc = task_id;
    for (int64_t r = 0; r < repeats; ++r) {
        acc = acc * 1103515245 + r;
    }
    // The accumulator exists to keep the loop, not to be read: this store is
    // unreachable for every value the loop can produce.
    if (acc == INT64_MIN) state[slot + 1] = acc;

    state[slot] = ready ? task_id + 1 : -(task_id + 1);
    dcci(&state[slot], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
