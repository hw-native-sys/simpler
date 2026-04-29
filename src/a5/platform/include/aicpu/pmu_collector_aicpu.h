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
 * @file pmu_collector_aicpu.h
 * @brief AICPU-side PMU collection interface (a5)
 *
 * Lifecycle (called from aicpu_executor.cpp):
 *   pmu_aicpu_init()              — resolve per-core PMU MMIO bases + buffer
 *                                   pointers, program events, start counters,
 *                                   pop initial PmuBuffers from free_queues,
 *                                   publish (pmu_buffer_addr, pmu_reg_base)
 *                                   to each Handshake.
 *   [task loop]
 *     pmu_aicpu_complete_record()     — copy the dual-issue slot AICore wrote
 *                                   into PmuBuffer::records[count], filling
 *                                   func_id + core_type. Switches buffer
 *                                   when full.
 *   pmu_aicpu_flush_buffers()     — per-thread: flush each of this thread's
 *                                   non-empty PmuBuffers to the ready_queue
 *                                   (mirrors a2a3 pmu_aicpu_flush_buffers)
 *   pmu_aicpu_finalize()          — per-thread: restore CTRL registers.
 */

#ifndef PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_

#include <cstdint>

#include "common/core_type.h"
#include "common/pmu_profiling.h"
#include "runtime.h"  // Handshake

extern "C" void set_platform_pmu_base(uint64_t pmu_data_base);
extern "C" uint64_t get_platform_pmu_base();
extern "C" void set_pmu_enabled(bool enable);
extern "C" bool is_pmu_enabled();

/**
 * Initialize PMU for all cores.
 *
 * For each logical core i in [0, num_cores):
 *   - Resolve the PMU MMIO base from physical_core_ids[i] via the platform's
 *     PMU reg-addr table.
 *   - Program event selectors (PMU_CNT0_IDX..CNT9_IDX).
 *   - Start counters (set PMU_CTRL_0 and PMU_CTRL_1).
 *   - Pop an initial PmuBuffer from the per-core free_queue.
 *   - Publish (pmu_buffer_addr, pmu_reg_base) into handshakes[i] so the
 *     matching AICore can read PMU MMIO and write the dual-issue slot.
 *
 * On sim (or when a core has no PMU reg addr), the core is skipped for MMIO
 * programming. The handshake fields still carry whatever reg_base the
 * platform reg table returns (0 on sim for missing entries), so AICore
 * no-ops the read if reg_base is 0.
 *
 * Must be called after the host has published pmu_data_base (via
 * set_platform_pmu_base) and after every active core has reported its
 * physical_core_id via handshake. Must be called BEFORE the caller
 * sets aicpu_regs_ready=1 on each handshake.
 *
 * @param handshakes         Handshake array (one per core)
 * @param physical_core_ids  Array of hardware physical core ids
 * @param num_cores          Number of active cores
 */
void pmu_aicpu_init(Handshake *handshakes, const uint32_t *physical_core_ids, int num_cores);

/**
 * Commit one PmuRecord from the dual-issue staging slot.
 * Switches buffer via SPSC free_queue/ready_queue when full.
 *
 * @param core_id     Logical core index
 * @param thread_idx  AICPU thread index (selects ready_queue)
 * @param reg_task_id Register dispatch token (slot match key)
 * @param task_id     Full task_id to store in the PmuRecord
 * @param func_id     kernel_id from the completed task slot
 * @param core_type   AIC or AIV
 */
void pmu_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type
);

/**
 * Per-thread PMU buffer flush. Mirrors a2a3 pmu_aicpu_flush_buffers().
 *
 * For each core in cur_thread_cores, enqueue its non-empty PmuBuffer into the
 * thread's ready_queue so the host collector can pick it up.
 *
 * @param thread_idx        AICPU thread index (selects ready_queue)
 * @param cur_thread_cores  Array of logical core ids owned by this thread
 * @param core_num          Entries in cur_thread_cores
 */
void pmu_aicpu_flush_buffers(int thread_idx, const int *cur_thread_cores, int core_num);

/**
 * Per-thread PMU finalize: restore CTRL registers for this thread's cores.
 *
 * @param cur_thread_cores  Array of logical core ids owned by this thread
 * @param core_num          Entries in cur_thread_cores
 */
void pmu_aicpu_finalize(const int *cur_thread_cores, int core_num);

#endif  // PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_
