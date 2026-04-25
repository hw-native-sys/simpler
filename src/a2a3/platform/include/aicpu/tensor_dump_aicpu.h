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
 * @file tensor_dump_aicpu.h
 * @brief AICPU tensor dump collection interface
 *
 * Provides tensor dump management for AICPU side.
 * Handles dump shared-memory base propagation plus buffer initialization,
 * tensor data copying to arenas, metadata recording, and flushing.
 */

#ifndef PLATFORM_AICPU_TENSOR_DUMP_AICPU_H_
#define PLATFORM_AICPU_TENSOR_DUMP_AICPU_H_

#include <cinttypes>

#include "common/memory_barrier.h"
#include "common/tensor_dump.h"
#include "data_type.h"

#ifdef __cplusplus
#include "callable.h"
#include "common/unified_log.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the tensor dump shared-memory base address.
 * Called by the platform layer before AICPU execution starts.
 *
 * @param dump_data_base Device pointer (as uint64_t) to dump shared memory
 */
void set_platform_dump_base(uint64_t dump_data_base);

/**
 * Get the tensor dump shared-memory base address.
 *
 * @return Device pointer (as uint64_t) to dump shared memory
 */
uint64_t get_platform_dump_base();

/**
 * Set whether tensor dump is enabled for this execution.
 * Called by the platform layer before AICPU execution starts.
 *
 * @param enable true to enable tensor dump, false to disable
 */
void set_dump_tensor_enabled(bool enable);

/**
 * Get whether tensor dump is enabled for this execution.
 *
 * @return true if tensor dump is enabled
 */
bool is_dump_tensor_enabled();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
bool get_tensor_dump_role_from_direction(ArgDirection dir, TensorDumpRole *role);
int32_t count_callable_tensor_args(const CoreCallable &callable);
bool should_dump_tensor_at_stage(TensorDumpRole role, TensorDumpStage stage);
bool try_log_tensor_dump_layout_mismatch();
int dump_tensor_record(int thread_idx, const TensorDumpInfo &info);

template <int MaxSubtaskSlots, typename SlotStateT, typename IsSubtaskActiveFn, typename GetFunctionBinAddrFn>
inline void dump_tensors_for_task(
    int32_t thread_idx, const SlotStateT &slot_state, TensorDumpStage stage, IsSubtaskActiveFn is_subtask_active,
    GetFunctionBinAddrFn get_function_bin_addr
) {
    const auto &pl = *slot_state.payload;
    const CoreCallable *callables[MaxSubtaskSlots] = {};
    int32_t total_tensor_args = 0;

    for (int raw_subtask_id = 0; raw_subtask_id < MaxSubtaskSlots; raw_subtask_id++) {
        if (!is_subtask_active(slot_state.active_mask, raw_subtask_id)) {
            continue;
        }
        int32_t slot_idx = raw_subtask_id;
        uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
        if (callable_addr == 0) {
            return;
        }
        callables[slot_idx] = reinterpret_cast<const CoreCallable *>(callable_addr);
        total_tensor_args += count_callable_tensor_args(*callables[slot_idx]);
    }

    if (total_tensor_args != pl.tensor_count) {
        if (try_log_tensor_dump_layout_mismatch()) {
            LOG_WARN(
                "Thread %d: tensor dump skipped for task 0x%" PRIx64
                ": active callable tensor count (%d) does not match payload tensor count (%d). "
                "Task-level dump assumes payload tensors are concatenated by active subtask order.",
                thread_idx, static_cast<uint64_t>(slot_state.task->task_id.raw), total_tensor_args, pl.tensor_count
            );
        }
        return;
    }

    rmb();

    int32_t payload_index = 0;
    for (int raw_subtask_id = 0; raw_subtask_id < MaxSubtaskSlots; raw_subtask_id++) {
        if (!is_subtask_active(slot_state.active_mask, raw_subtask_id)) {
            continue;
        }
        int32_t slot_idx = raw_subtask_id;
        const CoreCallable &callable = *callables[slot_idx];
        for (int32_t sig_idx = 0; sig_idx < callable.sig_count(); sig_idx++) {
            ArgDirection dir = callable.sig(sig_idx);
            if (dir == ArgDirection::SCALAR) {
                continue;
            }
            TensorDumpRole role;
            if (get_tensor_dump_role_from_direction(dir, &role) && should_dump_tensor_at_stage(role, stage)) {
                const auto &t = pl.tensors[payload_index];
                TensorDumpInfo info = {};
                info.buffer_addr = t.buffer.addr;
                info.dtype = static_cast<uint8_t>(t.dtype);
                info.ndims = static_cast<uint8_t>(t.ndims);
                const uint32_t *raw_shapes = t.get_raw_shapes();
                for (uint32_t d = 0; d < t.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
                    info.shapes[d] = t.shapes[d];
                    info.offsets[d] = t.is_all_offset_zero ? 0 : t.offsets[d];
                    info.raw_shapes[d] = raw_shapes[d];
                }
                info.task_id = slot_state.task->task_id.raw;
                info.subtask_id = raw_subtask_id;
                info.func_id = slot_state.task->kernel_id[slot_idx];
                info.arg_index = static_cast<uint32_t>(payload_index);
                info.role = role;
                info.stage = stage;
                dump_tensor_record(thread_idx, info);
            }
            payload_index++;
        }
    }
}

template <typename TensorInfoT>
inline void dump_tensors_for_task(
    int32_t thread_idx, uint64_t task_id, uint8_t subtask_id, int32_t task_arg_count, int32_t func_id,
    const CoreCallable &callable, const TensorInfoT *tensor_info, int32_t tensor_info_count,
    const uint64_t *buffer_addrs, int32_t buffer_count, TensorDumpStage stage
) {
    int32_t sig_count = callable.sig_count();
    if (task_arg_count < sig_count) {
        static bool logged_task_signature_mismatch = false;
        if (!logged_task_signature_mismatch) {
            logged_task_signature_mismatch = true;
            LOG_WARN(
                "Thread %d: tensor dump skipped for task 0x%" PRIx64
                ": task args (%d) smaller than callable signature (%d)",
                thread_idx, task_id, task_arg_count, sig_count
            );
        }
        return;
    }

    int32_t tensor_arg_count = count_callable_tensor_args(callable);
    if (tensor_info == nullptr || tensor_info_count != tensor_arg_count) {
        if (tensor_arg_count == 0) {
            return;
        }
        if (try_log_tensor_dump_layout_mismatch()) {
            LOG_WARN(
                "Thread %d: tensor dump skipped for task 0x%" PRIx64
                ": callable tensor args (%d) do not match registered tensor info (%d)",
                thread_idx, task_id, tensor_arg_count, tensor_info_count
            );
        }
        return;
    }

    if (buffer_addrs == nullptr || buffer_count != tensor_arg_count) {
        static bool logged_task_tensor_addr_mismatch = false;
        if (!logged_task_tensor_addr_mismatch) {
            logged_task_tensor_addr_mismatch = true;
            LOG_WARN(
                "Thread %d: tensor dump skipped for task 0x%" PRIx64
                ": reconstructed tensor buffers (%d) do not match callable tensor args (%d)",
                thread_idx, task_id, buffer_count, tensor_arg_count
            );
        }
        return;
    }

    rmb();

    int32_t tensor_arg_index = 0;
    for (int32_t sig_idx = 0; sig_idx < sig_count; sig_idx++) {
        ArgDirection dir = callable.sig(sig_idx);
        if (dir == ArgDirection::SCALAR) {
            continue;
        }

        TensorDumpRole role;
        if (!get_tensor_dump_role_from_direction(dir, &role) || !should_dump_tensor_at_stage(role, stage)) {
            tensor_arg_index++;
            continue;
        }

        const auto &t = tensor_info[tensor_arg_index];
        TensorDumpInfo info = {};
        info.task_id = task_id;
        info.subtask_id = subtask_id;
        info.role = role;
        info.stage = stage;
        info.dtype = static_cast<uint8_t>(t.dtype);
        info.ndims = t.ndims;
        info.func_id = static_cast<uint32_t>(func_id);
        info.arg_index = static_cast<uint32_t>(tensor_arg_index);
        info.buffer_addr = buffer_addrs[tensor_arg_index];
        for (uint32_t d = 0; d < t.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
            info.shapes[d] = t.shapes[d];
            info.offsets[d] = t.offsets[d];
            info.raw_shapes[d] = t.raw_shapes[d];
        }
        dump_tensor_record(thread_idx, info);
        tensor_arg_index++;
    }
}
#endif

/**
 * Initialize tensor dump.
 *
 * Sets up per-thread DumpBufferState pointers and pops initial
 * metadata buffers from each thread's free_queue.
 *
 * @param num_dump_threads Number of scheduling threads that will dump tensors
 */
void dump_tensor_init(int num_dump_threads);

/**
 * Record a single tensor dump.
 *
 * Copies tensor data from GM to the thread's arena, appends a
 * TensorDumpRecord to the current metadata buffer. Switches
 * buffers when full via the SPSC free_queue.
 *
 * When metadata buffers are temporarily exhausted, old dump metadata may be
 * overwritten so execution can continue without losing the active buffer.
 *
 * @param thread_idx Scheduling thread index
 * @param info Tensor metadata and identification
 * @return 0 on success or intentional drop, -1 only when dump state is unavailable
 */
int dump_tensor_record(int thread_idx, const TensorDumpInfo &info);

/**
 * Flush remaining tensor dump data for a thread.
 *
 * Marks non-empty metadata buffers as ready and enqueues them
 * for host collection.
 *
 * @param thread_idx Thread index
 */
void dump_tensor_flush(int thread_idx);

#endif  // PLATFORM_AICPU_TENSOR_DUMP_AICPU_H_
