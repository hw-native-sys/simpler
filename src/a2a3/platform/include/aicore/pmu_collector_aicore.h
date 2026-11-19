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
 * @file pmu_collector_aicore.h
 * @brief AICore-side PMU counter gate (per-task enable/disable).
 *
 * Purpose:
 *   AICPU programs the PMU event selectors and enables the PMU framework once
 *   at init (pmu_aicpu_start). During the task loop, AICore toggles the CTRL
 *   SPR bit0 around each kernel execution so the PMU counters only accumulate
 *   during actual kernel work, excluding dispatch / idle polling overhead.
 *
 * Usage (from aicore_executor.cpp main loop):
 *     if (pmu_enabled) pmu_aicore_begin();
 *     execute_task(...);
 *     if (pmu_enabled) pmu_aicore_end();
 */

#ifndef PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_

#include "aicore/aicore.h"
#include "common/platform_config.h"

// PMU enable bit in the AICore CTRL SPR (bit0 = GLB_PMU_EN).
constexpr uint64_t PMU_AICORE_CTRL_ENABLE_BIT = 0x1ULL;

/**
 * Begin PMU counting window: set CTRL bit0 so hardware counters start accruing.
 *
 * The toggle is fenced on both sides because that is how the toolchain's own
 * equivalent does it. `metrics_prof_start` /`metrics_prof_stop` in
 * `__clang_cce_aicore_functions.h` (CANN 9.0.0 BiSheng, guarded for
 * `__DAV_C220_VEC__` / `__DAV_C220_CUBE__` and `__NPU_ARCH__ == 3510`, which
 * covers both architectures here) set and clear **this same bit 0** and wrap
 * each write in `pipe_barrier(PIPE_ALL)`. `set_ctrl` is a scalar-pipe write
 * while the work being measured runs on the other pipes, so without those
 * fences the enable is not ordered against the kernel body in either
 * direction. This is contract conformance: it is not evidence about what the
 * counters then hold, and it does not explain any particular measurement.
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_begin() {
    pipe_barrier(PIPE_ALL);
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) | PMU_AICORE_CTRL_ENABLE_BIT);
    pipe_barrier(PIPE_ALL);
}

/**
 * End PMU counting window: clear CTRL bit0.
 *
 * Whether the counter registers hold their accumulated value after this, or
 * stop presenting it, is not stated by any contract this repo can cite. Both
 * executors on this architecture read the counters from the AICPU, released by
 * the per-task FIN, and publishing FIN is a notification rather than an
 * acknowledgement that the read happened — so where this call sits relative to
 * FIN does not decide whether a read lands inside the window, and no ordering
 * here should be read as establishing that it does.
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_end() {
    pipe_barrier(PIPE_ALL);
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) & ~PMU_AICORE_CTRL_ENABLE_BIT);
    pipe_barrier(PIPE_ALL);
}

#endif  // PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_
