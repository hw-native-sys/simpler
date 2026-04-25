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
 * Split of duties:
 *   - AICPU owns init (event selectors, PMU_CTRL_0/1 start) and finalize
 *     (CTRL restore). It also publishes per-core pmu_buffer_addr /
 *     pmu_reg_base into Handshake at init time so AICore can do the
 *     MMIO read itself.
 *   - AICore reads PMU counters + PMU_CNT_TOTAL via MMIO after each task
 *     (pmu_aicore_record_task), writing into PmuBuffer::dual_issue_slots.
 *   - AICPU, on COND FIN, validates the slot and commits a full PmuRecord
 *     into PmuBuffer::records[] (pmu_aicpu_complete_record).
 *
 * Lifecycle (called from aicpu_executor.cpp):
 *   pmu_aicpu_init()              — resolve per-core PMU MMIO bases + buffer
 *                                   pointers, program events, start counters,
 *                                   publish (pmu_buffer_addr, pmu_reg_base)
 *                                   to each Handshake.
 *   [task loop]
 *     pmu_aicpu_complete_record() — copy the dual-issue slot AICore wrote
 *                                   into PmuBuffer::records[count], filling
 *                                   func_id + core_type. Drops the record
 *                                   silently if the buffer is full.
 *   pmu_aicpu_finalize()          — per-thread: restore CTRL registers.
 *
 * a5 uses a single pre-allocated PmuBuffer per core; the host drains it via
 * rtMemcpy after stream sync (see src/a5/platform/src/host/pmu_collector.cpp).
 * There is no SPSC queue and no per-thread flush step.
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
 * sets aicpu_regs_ready=1 on each handshake, so AICore observes the
 * new fields via the same release/acquire boundary.
 *
 * @param handshakes         Handshake array (one per core). This function
 *                           writes pmu_buffer_addr and pmu_reg_base into
 *                           handshakes[0..num_cores). Caller owns lifetime.
 * @param physical_core_ids  Array of hardware physical core ids, indexed by
 *                           logical core_id. Caller owns the memory; this
 *                           function does not retain the pointer.
 * @param num_cores          Number of active cores (logical core_id range is [0, num_cores))
 */
void pmu_aicpu_init(Handshake *handshakes, const uint32_t *physical_core_ids, int num_cores);

/**
 * Commit one PmuRecord from the dual-issue staging slot that AICore wrote
 * into PmuBuffer::dual_issue_slots[task_id & 1]. Copies register state
 * (pmu_counters + pmu_total_cycles) and fills AICPU-owned metadata
 * (task_id, func_id, core_type). When the buffer is full the record is
 * dropped and the core's PmuBufferState::dropped_record_count is incremented.
 * Every call bumps PmuBufferState::total_record_count so host can cross-check
 * collected + dropped against the AICPU's attempted-commit count.
 * No-op if PMU is not enabled or the core has no PMU buffer bound.
 *
 * @param core_id     Logical core index
 * @param thread_idx  AICPU thread index (reserved; not used on a5 memcpy path)
 * @param reg_task_id Register dispatch token (DATA_MAIN_BASE value). AICore
 *                    wrote this 32-bit value into dual_issue_slots[...].task_id,
 *                    so AICPU uses it to locate the slot and validate its
 *                    freshness. Callers should pass the same register token
 *                    they observed on COND / wrote via DATA_MAIN_BASE.
 * @param task_id     Full task_id to store in the PmuRecord (e.g. PTO2's
 *                    (ring_id<<32)|local_id). May differ from reg_task_id.
 * @param func_id     kernel_id from the completed task slot
 * @param core_type   AIC or AIV
 */
void pmu_aicpu_complete_record(
    int core_id, int thread_idx, uint32_t reg_task_id, uint64_t task_id, uint32_t func_id, CoreType core_type
);

/**
 * Per-thread PMU finalize: restore CTRL registers for this thread's cores.
 *
 * @param cur_thread_cores  Array of logical core ids owned by this thread
 * @param core_num          Entries in cur_thread_cores
 */
void pmu_aicpu_finalize(const int *cur_thread_cores, int core_num);

#endif  // PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_
