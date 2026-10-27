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
 * Holds a core for a while, then records that it ran.
 *
 * The competing work in a case that asks whether a cohort still assembles
 * while ordinary tasks are taking cores. It has to occupy a core rather than
 * return at once: a task that finishes immediately frees its slot before the
 * cohort's claim pass looks, and then there was never any competition.
 *
 * Args: [0] out tensor, [1] slot, [2] how long to hold the core, in ms.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#ifdef PTO_CPUSTUB_HPP
#include <chrono>
#endif

#include "common/platform_config.h"
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
constexpr float RAN = 1.0f;

// Held for a duration, not a spin count. An iteration costs a different amount
// in the CPU-stub build than on silicon, so a count that occupies a core for
// milliseconds in one occupies it for microseconds in the other -- and this
// kernel exists only to still be running at a particular moment.
#ifdef PTO_CPUSTUB_HPP
static uint64_t busy_now_ticks() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#else
static __aicore__ uint64_t busy_now_ticks() { return get_sys_cnt(); }
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t slot = static_cast<int32_t>(args[1]);
    const uint64_t hold_ticks =
        static_cast<uint64_t>(args[2]) * (static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) / 1000u);

    const uint64_t started = busy_now_ticks();
    volatile int64_t sink = 0;
    while (busy_now_ticks() - started < hold_ticks) sink += 1;
    (void)sink;

    const int32_t offset = slot * FLOATS_PER_CACHE_LINE;
    out[offset] = RAN;
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
