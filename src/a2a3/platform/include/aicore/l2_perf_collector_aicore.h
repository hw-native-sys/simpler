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
 * @file l2_perf_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#ifndef PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_

#include "common/l2_perf_profiling.h"
#include "aicore/aicore.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/a2a3/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/a2a3/platform/sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * Record task execution performance data
 *
 * Writes timing metrics into the per-core staging ring at
 * `dual_issue_slots[task_id % PLATFORM_L2_AICORE_RING_SIZE]`. The ring is
 * stable for the entire run (its address never changes), so AICore is fully
 * decoupled from the AICPU's records-buffer rotation.
 *
 * AICore writes L2PerfRecord.task_id as the register dispatch token (low 32 bits, zero-extended).
 * For tensormap_and_ringbuffer, AICPU overwrites with the full (ring_id << 32) | local_id
 * encoding after handshake match.
 *
 * @param ring Per-core staging ring pointer (resolved at AICore kernel
 *             entry from KernelArgs::aicore_ring_addr[block_idx] via
 *             set_aicore_l2_perf_ring()/get_aicore_l2_perf_ring())
 * @param task_id Register dispatch id (DATA_MAIN_BASE), stored in task_id low 32 bits
 * @param start_time Start timestamp
 * @param end_time End timestamp
 */
__aicore__ __attribute__((always_inline)) static inline void
l2_perf_aicore_record_task(__gm__ L2PerfAicoreRing *ring, uint32_t task_id, uint64_t start_time, uint64_t end_time) {
    __gm__ L2PerfRecord *record = &ring->dual_issue_slots[task_id % PLATFORM_L2_AICORE_RING_SIZE];

    record->start_time = start_time;
    record->end_time = end_time;
    record->task_id = static_cast<uint64_t>(task_id);

    // Flush cache to make data visible to AICPU
    dcci(record, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}

#endif  // PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
