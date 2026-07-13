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
 * @file dep_gen_collector_aicpu.h
 * @brief AICPU-side dep_gen (SubmitTrace) capture interface
 *
 * Lifecycle (called from scheduler cold path + aicpu_executor.cpp):
 *   dep_gen_aicpu_init()                — pop the initial DepGenBuffer from
 *                                         the (single) instance's free_queue.
 *                                         Runs in SchedulerContext::init() on
 *                                         the single-threaded cold path.
 *   dep_gen_aicpu_set_orch_thread_idx() — record which AICPU thread runs the
 *                                         orchestrator (used to select the
 *                                         per-thread ready_queue on flush).
 *                                         Runs later on the orchestrator thread.
 *   [submit_task loop]
 *     dep_gen_aicpu_record_submit()     — append one DepGenRecord; rotate
 *                                         buffer when full.
 *   dep_gen_aicpu_flush()               — push current buffer (if non-empty)
 *                                         to ready_queue.
 *   dep_gen_aicpu_finalize()            — clear bookkeeping.
 *
 * All-primitive interface (no runtime types in platform header):
 *   - task_id passed as raw uint64 (PTO2TaskId::raw)
 *   - tensor data passed via opaque void* pointers (memcpy'd into the
 *     DEP_GEN_TENSOR_SIZE-byte slot; static_asserted against sizeof(Tensor)
 *     in the .cpp)
 *   - explicit_deps passed as uint64*
 *
 * No-op when dep_gen is disabled (is_dep_gen_enabled() returns false).
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEP_GEN_COLLECTOR_AICPU_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEP_GEN_COLLECTOR_AICPU_H_

#include <cstdint>

#include "common/dep_gen.h"

extern "C" void set_platform_dep_gen_base(uint64_t dep_gen_data_base);
extern "C" uint64_t get_platform_dep_gen_base();
extern "C" void set_dep_gen_enabled(bool enable);
extern "C" bool is_dep_gen_enabled();

/**
 * Register the AICPU thread index that hosts the orchestrator. Used to select
 * the per-thread ready_queue when buffers fill or on flush. Must be called by
 * aicpu_executor.cpp before any dep_gen_aicpu_record_submit() can fire.
 *
 * Mirrors l2_swimlane_aicpu_set_orch_thread_idx().
 */
void dep_gen_aicpu_set_orch_thread_idx(int thread_idx);

/**
 * Initialize dep_gen capture: pop the initial DepGenBuffer from the (single)
 * orchestrator instance's free_queue and stash it as the current buffer.
 *
 * Pre-conditions:
 *   - Host has set the data base via set_platform_dep_gen_base()
 *   - dep_gen is enabled via set_dep_gen_enabled(true)
 *
 * The initial pop only touches instance 0's free_queue, so it does not depend
 * on the orchestrator thread index; dep_gen_aicpu_set_orch_thread_idx() may run
 * after this (and does — on the orchestrator thread), as long as it precedes
 * the first record_submit.
 *
 * If the free_queue is empty at init (host bug), the function leaves the
 * current buffer as null and subsequent record_submit calls will bump
 * dropped_record_count.
 */
void dep_gen_aicpu_init();

/**
 * Append a base DepGenRecord (and zero or more DepGenOverflowRecord chain
 * records) for a completed submit_task call. Switches buffer via the SPSC
 * free_queue / ready_queue protocol when the current buffer cannot hold the
 * full chain. No-op if dep_gen is disabled.
 *
 * Tensor handling: for slot i, if tensor_ptrs[i] is non-null, its first
 * DEP_GEN_TENSOR_SIZE bytes are memcpy'd into record.tensors[i]. If null
 * (e.g. arg_types[i] == OUTPUT, where the Tensor is materialized later by
 * the runtime), the slot is left zeroed. Replay decides what to do with
 * each slot based on arg_types[i].
 *
 * Dep handling: the first DEP_GEN_MAX_EXPLICIT_DEPS deps land in the base
 * record; any excess spills into a chain of DepGenOverflowRecord slots. A
 * submit whose chain would exceed the buffer's remaining capacity (even
 * after switch) is truncated to fit; the dropped tail is logged.
 *
 * @param task_id_raw         PTO2TaskId::raw (the assigned task_id for this submit)
 * @param in_manual_scope     true iff the submit happened inside a manual scope
 * @param tensor_count        Number of slots in tensor_ptrs / arg_types (≤ CORE_MAX_TENSOR_ARGS)
 * @param tensor_ptrs         Per-slot Tensor pointer (nullptr to skip the slot)
 * @param arg_types           Per-slot TensorArgType (interpreted as raw byte)
 * @param explicit_dep_count  Number of explicit_deps — no static cap; truncated only when the
 *                            chain would not fit in a single DepGenBuffer
 * @param explicit_deps_raw   Per-dep PTO2TaskId::raw (length = explicit_dep_count)
 * @param block_num           SPMD logical block count; 1 means non-SPMD
 * @param kernel_ids          Per-subslot kernel id triple {AIC, AIV0, AIV1};
 *                            inactive subslots use INVALID_KERNEL_ID (-1).
 *                            Captured here so the host swimlane post-processor
 *                            can resolve (task_id → kernel) without the AICore
 *                            hot path writing identity fields itself.
 */
void dep_gen_aicpu_record_submit(
    uint64_t task_id_raw, bool in_manual_scope, bool early_dispatch, int tensor_count, const void *const *tensor_ptrs,
    const uint8_t *arg_types, int explicit_dep_count, const uint64_t *explicit_deps_raw, int block_num,
    const int32_t kernel_ids[3]
);

/**
 * Push the current (partially-filled) DepGenBuffer to the orchestrator
 * thread's ready_queue so the host can pick it up. Called once at end of
 * run, after the orchestrator's last submit.
 */
void dep_gen_aicpu_flush();

/**
 * Clear file-local bookkeeping (current_buf cache, etc.). Called at shutdown.
 */
void dep_gen_aicpu_finalize();

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_DEP_GEN_COLLECTOR_AICPU_H_
