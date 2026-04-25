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
 * @brief AICPU-side AICore PMU counter collection interface
 *
 * Lifecycle (called from aicpu_executor.cpp):
 *   pmu_aicpu_init()            — resolve per-core PMU MMIO bases from the
 *                                 caller-supplied physical_core_id array,
 *                                 program events, start counters, and pop
 *                                 initial PmuBuffers from free_queues
 *   [task loop]
 *     pmu_aicpu_record_task()   — read counters, write one PmuRecord; switch buffer when full
 *   pmu_aicpu_flush_buffers()   — per-thread: flush each of this thread's non-empty
 *                                 PmuBuffers to the ready_queue
 *   pmu_aicpu_finalize()        — per-thread: restore CTRL registers on shutdown
 */

#ifndef PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_
#define PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_

#include <cstdint>

#include "common/core_type.h"
#include "common/pmu_profiling.h"

extern "C" void set_platform_pmu_base(uint64_t pmu_data_base);
extern "C" uint64_t get_platform_pmu_base();
extern "C" void set_pmu_enabled(bool enable);
extern "C" bool is_pmu_enabled();

/**
 * Initialize PMU for all cores.
 *
 * For each logical core i in [0, num_cores):
 *   - Resolve the PMU MMIO base from physical_core_ids[i] via the platform's
 *     PMU reg-addr table, cache it in the collector's file-local state.
 *   - Program event selectors and start CTRL_0.
 *   - Pop an initial PmuBuffer from the per-core free_queue.
 *
 * On sim (or when a core has no PMU reg addr), the core is skipped for CTRL
 * programming and subsequent pmu_aicpu_record_task() calls for that core
 * become no-ops.
 *
 * Must be called after host has initialized PMU shared memory (pmu_data_base
 * set) and after every active core has reported its physical_core_id via
 * handshake (i.e. after handshake_all_cores returns).
 *
 * @param physical_core_ids  Array of hardware physical core ids, indexed by
 *                           logical core_id. Caller owns the memory; this
 *                           function does not retain the pointer.
 * @param num_cores          Number of active cores (logical core_id range is [0, num_cores))
 */
void pmu_aicpu_init(const uint32_t *physical_core_ids, int num_cores);

/**
 * Read PMU counters for one completed task and append a PmuRecord to the
 * per-core buffer. Switches to a fresh buffer (via the SPSC free_queue /
 * ready_queue protocol) when the current buffer is full.
 * No-op if PMU is not enabled or the core has no PMU address bound.
 *
 * @param core_id    Logical core index
 * @param thread_idx AICPU thread index (used to select the per-thread ready_queue)
 * @param task_id    task_id.raw from the completed task slot
 * @param func_id    kernel_id from the completed task slot
 * @param core_type  AIC or AIV
 */
void pmu_aicpu_record_task(int core_id, int thread_idx, uint64_t task_id, uint32_t func_id, CoreType core_type);

/**
 * Per-thread PMU buffer flush.
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
 * Called after pmu_aicpu_flush_buffers() during shutdown.
 *
 * @param cur_thread_cores  Array of logical core ids owned by this thread
 * @param core_num          Entries in cur_thread_cores
 */
void pmu_aicpu_finalize(const int *cur_thread_cores, int core_num);

#endif  // PLATFORM_AICPU_PMU_COLLECTOR_AICPU_H_
