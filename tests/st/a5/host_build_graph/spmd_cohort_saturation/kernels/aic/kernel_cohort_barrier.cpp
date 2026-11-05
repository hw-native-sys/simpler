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
 * A block of a cohort that actually waits for its siblings.
 *
 * Every other sync-start kernel here writes its slot and returns, which makes
 * co-residency unobservable: the guarantee sync-start buys is that all of a
 * task's blocks hold cores at the same time, and a block that never waits
 * cannot tell whether that held. Those cases pin placement -- which core got
 * which block -- and nothing else.
 *
 * This one arrives, then spins until it has seen every sibling arrive. A
 * cohort that is not co-resident therefore cannot finish, which is the whole
 * point; the wait is bounded so that failure surfaces as a wrong number rather
 * than as a device-side timeout forty-five seconds later. A block that gives
 * up writes POISON, and no golden can contain it.
 *
 * Arrival flags and results live in separate slots: a block must not overwrite
 * the flag a sibling is still polling. Each is its own cache line because the
 * blocks are on different cores.
 *
 * A Mix block is several subtasks on several cores, and co-residency is a claim
 * about all of them, so each subtask arrives at a flag of its own and waits for
 * every subtask of every block. `lanes_per_block` says how many a block has and
 * COHORT_LANE says which one this file is; a single-subtask cohort passes 1 and
 * uses lane 0.
 *
 * Args: [0] out tensor, [1] arrive base slot, [2] result base slot,
 *       [3] spin limit, [4] lanes per block.
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
constexpr float ARRIVED = 1.0f;
// Negative and far from any block index, so a sum or an equality check on the
// result slots rejects it however the caller reads them.
constexpr float POISON = -99.0f;

#ifndef COHORT_LANE
#define COHORT_LANE 0
#endif

// The wait is bounded in ticks, not iterations. An iteration costs a different
// amount in the CPU-stub build than it does on silicon, so a limit tuned as a
// count is either far too short on one of them or long enough on the other to
// outlast the device's own op-execute timeout -- and then the failure is a
// killed op rather than the poison this kernel writes, which is the one
// outcome the case cannot read. Ticks are nanoseconds on both.
#ifdef PTO_CPUSTUB_HPP
static uint64_t cohort_now_ticks() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#else
static __aicore__ uint64_t cohort_now_ticks() { return get_sys_cnt(); }
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    const int32_t arrive_base = static_cast<int32_t>(args[1]);
    const int32_t result_base = static_cast<int32_t>(args[2]);
    // Milliseconds, converted here so the caller never has to know the tick
    // rate of the platform it is running on.
    const uint64_t wait_ticks =
        static_cast<uint64_t>(args[3]) * (static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) / 1000u);
    const int32_t lanes_per_block = static_cast<int32_t>(args[4]);

    const int32_t block_idx = get_block_idx(args);
    const int32_t block_num = get_block_num(args);
    const int32_t lane_count = block_num * lanes_per_block;
    const int32_t own_lane = block_idx * lanes_per_block + COHORT_LANE;

    out[(arrive_base + own_lane) * FLOATS_PER_CACHE_LINE] = ARRIVED;
    dcci(&out[(arrive_base + own_lane) * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE, CACHELINE_OUT);

    bool assembled = false;
    // The flags are read through a volatile view. Without it the compiler is
    // free to hoist the load out of the spin -- `out` is an ordinary pointer
    // and dcci is a no-op in the CPU-stub build -- so a lane that misses a
    // sibling on its first pass would re-read a register for the rest of the
    // wait and never see the arrival. That turns staggered dispatch into a
    // false "never co-resident", which is precisely the verdict this kernel
    // exists to make trustworthy.
    volatile __gm__ float *flags = out;
    const uint64_t started = cohort_now_ticks();
    uint64_t waited = 0;
    while (!assembled && waited < wait_ticks) {
        // Invalidate every flag, then one barrier, then read. An invalidate
        // without a barrier before the load is not guaranteed to have taken
        // effect, so a sibling's arrival can stay invisible for the whole wait.
        for (int32_t lane = 0; lane < lane_count; ++lane)
            dcci(&out[(arrive_base + lane) * FLOATS_PER_CACHE_LINE], SINGLE_CACHE_LINE);
        dsb((mem_dsb_t)0);
        assembled = true;
        for (int32_t lane = 0; lane < lane_count; ++lane) {
            if (flags[(arrive_base + lane) * FLOATS_PER_CACHE_LINE] != ARRIVED) {
                assembled = false;
                break;
            }
        }
        waited = cohort_now_ticks() - started;
    }

    const int32_t offset = (result_base + own_lane) * FLOATS_PER_CACHE_LINE;
    out[offset] = assembled ? static_cast<float>(block_idx) : POISON;
    // How long this lane waited, in milliseconds, written only when it gave
    // up. It is the number that separates "the siblings never came" from "the
    // wait was too short", and a failure without it sends the reader back to
    // re-run with a probe. Written only on the give-up path so that a correct
    // run leaves the slot untouched and a fixed golden still describes the
    // whole buffer.
    if (!assembled)
        out[offset + 1] = static_cast<float>(waited / (static_cast<uint64_t>(PLATFORM_PROF_SYS_CNT_FREQ) / 1000u));
    dcci(&out[offset], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
