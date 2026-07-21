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
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_begin() {
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) | PMU_AICORE_CTRL_ENABLE_BIT);
}

/**
 * End PMU counting window: clear CTRL bit0 so counters freeze until next begin.
 */
__aicore__ __attribute__((always_inline)) static inline void pmu_aicore_end() {
    write_reg(RegId::CTRL, read_reg(RegId::CTRL) & ~PMU_AICORE_CTRL_ENABLE_BIT);
}

#endif  // PLATFORM_AICORE_PMU_COLLECTOR_AICORE_H_
