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
 * @file args_dump_aicpu.h
 * @brief AICPU args dump collection interface
 *
 * Provides args dump management for AICPU side.
 * Handles dump shared-memory base propagation plus buffer initialization,
 * tensor data copying to arenas, metadata recording, and flushing.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_AICPU_ARGS_DUMP_AICPU_H_
#define SRC_COMMON_PLATFORM_INCLUDE_AICPU_ARGS_DUMP_AICPU_H_

#include <stdint.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
#include <cinttypes>
#endif

#include "common/memory_barrier.h"
#include "common/args_dump.h"
#include "data_type.h"

#ifdef __cplusplus
#include "callable.h"
#include "common/unified_log.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the args dump shared-memory base address.
 * Called by the platform layer before AICPU execution starts.
 *
 * @param dump_data_base Device pointer (as uint64_t) to dump shared memory
 */
void set_platform_dump_base(uint64_t dump_data_base);

/**
 * Get the args dump shared-memory base address.
 *
 * @return Device pointer (as uint64_t) to dump shared memory
 */
uint64_t get_platform_dump_base();

/**
 * Set whether args dump is enabled for this execution.
 * Called by the platform layer before AICPU execution starts.
 *
 * @param enable true to enable args dump, false to disable
 */
void set_dump_args_enabled(bool enable);

/**
 * Get whether args dump is enabled for this execution.
 *
 * @return true if args dump is enabled
 */
bool is_dump_args_enabled();
bool is_dump_args_selective_mode();
void set_dump_args_task_mask(uint64_t task_id, ArgsDumpArgMask mask, ArgsDumpArgMask flags);
void get_dump_args_task_masks(uint64_t task_id, ArgsDumpArgMask *mask, ArgsDumpArgMask *flags);
void set_dump_args_task_scalar_dtypes(uint64_t task_id, uint32_t scalar_count, const uint8_t *scalar_dtypes);
bool get_dump_args_task_scalar_dtypes(uint64_t task_id, uint32_t *scalar_count, uint8_t *scalar_dtypes);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
bool get_dump_arg_role_from_direction(ArgDirection dir, ArgsDumpRole *role);
int32_t count_callable_tensor_args(const CoreCallable &callable);
bool should_dump_arg_at_stage(ArgsDumpRole role, ArgsDumpStage stage);
bool should_dump_task(ArgsDumpArgMask arg_mask);
bool should_dump_arg(ArgsDumpArgMask arg_mask, int32_t arg_index);
bool has_dump_arg_flag(ArgsDumpArgMask arg_mask, int32_t arg_index);
bool try_log_dump_args_layout_mismatch();
int dump_arg_record(int thread_idx, const ArgsDumpInfo &info);

template <int MaxSubtaskSlots, typename SlotStateT, typename IsSubtaskActiveFn, typename GetFunctionBinAddrFn>
inline void dump_args_for_task(
    int32_t thread_idx, const SlotStateT &slot_state, ArgsDumpStage stage, IsSubtaskActiveFn is_subtask_active,
    GetFunctionBinAddrFn get_function_bin_addr
) {
    // The record's func_ids[] must hold every active subtask's id. MaxSubtaskSlots
    // is PTO2_SUBTASK_SLOT_COUNT at every call site, so this ties the record array
    // size (platform layer) to the runtime subtask cap and catches any drift.
    static_assert(MaxSubtaskSlots <= ARGS_DUMP_MAX_FUNC_IDS, "ARGS_DUMP_MAX_FUNC_IDS must cover MaxSubtaskSlots");
    const auto &pl = *slot_state.payload;
    ArgsDumpArgMask dump_arg_mask = ARGS_DUMP_ARG_MASK_NONE;
    ArgsDumpArgMask dump_arg_flags = ARGS_DUMP_ARG_MASK_NONE;
    if (is_dump_args_selective_mode()) {
        get_dump_args_task_masks(slot_state.task->task_id.raw, &dump_arg_mask, &dump_arg_flags);
    }
    if (!should_dump_task(dump_arg_mask)) {
        return;
    }
    // Gather the task's active-subtask set. The widest active subtask's (completed)
    // signature drives direction + positional slot mapping; active_fids is the
    // task's mix membership, stamped on every record so a mix is recoverable as
    // the func set on each slot (no per-subtask duplication of the geometry).
    int32_t active_fids[ARGS_DUMP_MAX_FUNC_IDS] = {};
    int32_t active_count = 0;
    const CoreCallable *sig_src = nullptr;

    for (int raw_subtask_id = 0; raw_subtask_id < MaxSubtaskSlots; raw_subtask_id++) {
        if (!is_subtask_active(slot_state.active_mask, raw_subtask_id)) {
            continue;
        }
        uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[raw_subtask_id]);
        if (callable_addr == 0) {
            return;
        }
        const CoreCallable *cand = reinterpret_cast<const CoreCallable *>(callable_addr);
        if (sig_src == nullptr || cand->sig_count() > sig_src->sig_count()) {
            sig_src = cand;
        }
        if (active_count < ARGS_DUMP_MAX_FUNC_IDS) {
            active_fids[active_count++] = slot_state.task->kernel_id[raw_subtask_id];
        }
    }
    if (sig_src == nullptr) {
        return;  // no active subtask
    }

    // Reject negative too: tensor_count is signed and would wrap when cast below.
    if (pl.tensor_count < 0 || pl.tensor_count > CORE_MAX_TENSOR_ARGS) {
        return;
    }

    rmb();

    // Dump each payload tensor ONCE, driven by the widest active subtask's
    // (completed) signature: signature entry i maps to payload slot i positionally.
    // Every record carries the task's full active-subtask set in func_ids — the
    // slot's geometry is no longer duplicated per subtask; a mix is recoverable as
    // the func set on each slot.
    bool covered[CORE_MAX_TENSOR_ARGS] = {};
    int32_t covered_count = 0;
    const CoreCallable &sig = *sig_src;

    for (int32_t sig_idx = 0; sig_idx < sig.sig_count(); sig_idx++) {
        ArgDirection dir = sig.sig(sig_idx);
        if (dir == ArgDirection::SCALAR) {
            continue;
        }
        uint32_t slot = static_cast<uint32_t>(sig_idx);
        if (slot >= static_cast<uint32_t>(pl.tensor_count)) {
            // Slot not in this task's payload (a task that dispatched a prefix of
            // a wider completed signature). Skip — memory safety.
            continue;
        }
        if (!covered[slot]) {
            covered[slot] = true;
            covered_count++;
        }
        ArgsDumpRole role;
        if (!get_dump_arg_role_from_direction(dir, &role) || !should_dump_arg_at_stage(role, stage) ||
            !should_dump_arg(dump_arg_mask, slot)) {
            continue;
        }
        const auto &t = pl.tensors[slot];
        ArgsDumpInfo info = {};
        info.buffer_addr = t.buffer.addr;
        info.dtype = static_cast<uint8_t>(t.dtype);
        info.ndims = static_cast<uint8_t>(t.ndims);
        info.start_offset = t.start_offset;
        for (uint32_t d = 0; d < t.ndims && d < PLATFORM_DUMP_MAX_DIMS; d++) {
            info.shapes[d] = t.shapes[d];
            info.strides[d] = t.strides[d];
        }
        info.task_id = slot_state.task->task_id.raw;
        info.arg_index = slot;
        info.role = role;
        info.stage = stage;
        info.kind = static_cast<uint8_t>(ArgsDumpKind::TENSOR);
        info.func_count = active_count;
        for (int32_t i = 0; i < active_count; i++) {
            info.func_ids[i] = active_fids[i];
        }
        dump_arg_record(thread_idx, info);
    }

    if (covered_count != pl.tensor_count && try_log_dump_args_layout_mismatch()) {
        // Soft: some payload tensor slots are not covered by the signature (a
        // narrower-than-signature payload, or a signature shorter than the
        // payload), so they are not dumped.
        LOG_WARN(
            "Thread %d: task 0x%" PRIx64
            ": signature covers %d tensor slots but payload has %d; the rest are not dumped.",
            thread_idx, static_cast<uint64_t>(slot_state.task->task_id.raw), covered_count, pl.tensor_count
        );
    }

    // Scalars are stored once in the task payload; dump them once with the task's
    // full active-subtask set (same func_ids as the tensor records).
    if (stage == ArgsDumpStage::BEFORE_DISPATCH && pl.scalar_count > 0) {
        uint8_t scalar_dtypes[CORE_MAX_SCALAR_ARGS] = {};
        uint32_t dtype_scalar_count = 0;
        bool has_scalar_dtypes =
            get_dump_args_task_scalar_dtypes(slot_state.task->task_id.raw, &dtype_scalar_count, scalar_dtypes);
        for (int32_t scalar_index = 0; scalar_index < pl.scalar_count; scalar_index++) {
            int32_t scalar_arg_index = pl.tensor_count + scalar_index;
            if (!should_dump_arg(dump_arg_mask, scalar_arg_index)) {
                continue;
            }
            ArgsDumpInfo info = {};
            info.task_id = slot_state.task->task_id.raw;
            info.role = ArgsDumpRole::INPUT;
            info.stage = stage;
            info.dtype = (has_scalar_dtypes && scalar_index < static_cast<int32_t>(dtype_scalar_count)) ?
                             scalar_dtypes[scalar_index] :
                             static_cast<uint8_t>(DataType::UINT64);
            info.ndims = 0;
            info.arg_index = static_cast<uint32_t>(scalar_arg_index);
            info.kind = static_cast<uint8_t>(ArgsDumpKind::SCALAR);
            info.func_count = active_count;
            for (int32_t i = 0; i < active_count; i++) {
                info.func_ids[i] = active_fids[i];
            }
            info.scalar_value = pl.scalars[scalar_index];
            if (has_dump_arg_flag(dump_arg_flags, scalar_arg_index)) {
                info.flags = ARGS_DUMP_RECORD_FLAG_ARG_INDEX_AMBIGUOUS;
            }
            dump_arg_record(thread_idx, info);
        }
    }
}

// Dump the OUTPUT tensors of every task still RUNNING on a core, for the
// scheduler-hang / timeout path. These tasks never reached completion, so their
// AFTER_COMPLETION output was never captured; reading the current GM contents
// shows how far each stuck kernel got and how much output it wrote.
//
// Best-effort: only AICore writes already drained to GM are visible (the stuck
// core issued no pipe_barrier, so writes still in its cache are not), and a read
// may be torn if the core is mid-write. Records go into thread_idx's dump buffer
// and ride out on its dump_args_flush.
//
// The runtime supplies its own core/slot accessors so this stays platform-side
// and reusable across runtimes:
//   get_running_slot(cid)   -> SlotState* for the task running on core cid, or
//                              nullptr if idle. A cluster's cores share one
//                              SlotState pointer; pointer identity is used to
//                              emit each running task exactly once.
//   is_subtask_active / get_function_bin_addr — same callbacks as
//   dump_args_for_task.
template <int MaxSubtaskSlots, typename GetRunningSlotFn, typename IsSubtaskActiveFn, typename GetFunctionBinAddrFn>
inline void dump_running_task_outputs(
    int32_t thread_idx, int32_t cores_total_num, GetRunningSlotFn get_running_slot, IsSubtaskActiveFn is_subtask_active,
    GetFunctionBinAddrFn get_function_bin_addr
) {
    for (int32_t cid = 0; cid < cores_total_num; cid++) {
        auto *running = get_running_slot(cid);
        if (running == nullptr) {
            continue;
        }
        // Dedup: let the lowest-id core of each running task drive the dump.
        bool already_dumped = false;
        for (int32_t prev = 0; prev < cid; prev++) {
            if (get_running_slot(prev) == running) {
                already_dumped = true;
                break;
            }
        }
        if (already_dumped) {
            continue;
        }
        dump_args_for_task<MaxSubtaskSlots>(
            thread_idx, *running, ArgsDumpStage::AFTER_COMPLETION, is_subtask_active, get_function_bin_addr
        );
    }
}

template <typename TensorInfoT>
inline void dump_args_for_task(
    int32_t thread_idx, uint64_t task_id, int32_t task_arg_count, const CoreCallable &callable,
    const TensorInfoT *tensor_info, int32_t tensor_info_count, const uint64_t *buffer_addrs, int32_t buffer_count,
    ArgsDumpStage stage
) {
    int32_t sig_count = callable.sig_count();
    if (task_arg_count < sig_count) {
        static bool logged_task_signature_mismatch = false;
        if (!logged_task_signature_mismatch) {
            logged_task_signature_mismatch = true;
            LOG_WARN(
                "Thread %d: args dump skipped for task 0x%" PRIx64
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
        if (try_log_dump_args_layout_mismatch()) {
            LOG_WARN(
                "Thread %d: args dump skipped for task 0x%" PRIx64
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
                "Thread %d: args dump skipped for task 0x%" PRIx64
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

        ArgsDumpRole role;
        if (!get_dump_arg_role_from_direction(dir, &role) || !should_dump_arg_at_stage(role, stage)) {
            tensor_arg_index++;
            continue;
        }

        const auto &t = tensor_info[tensor_arg_index];
        ArgsDumpInfo info = {};
        info.task_id = task_id;
        info.func_count = 1;    // host_build_graph overload does not thread func_id
        info.func_ids[0] = -1;  // -> unknown
        info.role = role;
        info.stage = stage;
        info.dtype = static_cast<uint8_t>(t.dtype);
        info.ndims = t.ndims;
        info.arg_index = static_cast<uint32_t>(sig_idx);
        info.buffer_addr = buffer_addrs[tensor_arg_index];
        // TensorInfo (host_build_graph) still carries (raw_shapes, offsets)
        // implicitly describing a row-major-aligned sub-region. Translate to
        // (start_offset, strides[]) on the fly:
        //   strides[d] = prod(raw_shapes[d+1..])
        //   start_offset = Σ offsets[d] · strides[d]
        uint64_t s = 1;
        uint64_t start = 0;
        for (int32_t d = static_cast<int32_t>(t.ndims) - 1; d >= 0 && d < PLATFORM_DUMP_MAX_DIMS; --d) {
            info.shapes[d] = t.shapes[d];
            info.strides[d] = static_cast<int32_t>(s);
            start += static_cast<uint64_t>(t.offsets[d]) * s;
            s *= t.raw_shapes[d];
        }
        info.start_offset = start;
        dump_arg_record(thread_idx, info);
        tensor_arg_index++;
    }
}

/**
 * Initialize args dump.
 *
 * Sets up per-thread DumpBufferState pointers and pops initial
 * metadata buffers from each thread's free_queue.
 *
 * @param num_dump_threads Number of scheduling threads that will dump args
 */
void dump_args_init(int num_dump_threads);

/**
 * Record a single dumped arg.
 *
 * Copies tensor data from GM to the thread's arena, appends a
 * ArgsDumpRecord to the current metadata buffer. Switches
 * buffers when full via the SPSC free_queue.
 *
 * When metadata buffers are temporarily exhausted, old dump metadata may be
 * overwritten so execution can continue without losing the active buffer.
 *
 * @param thread_idx Scheduling thread index
 * @param info Tensor metadata and identification
 * @return 0 on success or intentional drop, -1 only when dump state is unavailable
 */
int dump_arg_record(int thread_idx, const ArgsDumpInfo &info);

/**
 * Flush remaining args dump data for a thread.
 *
 * Marks non-empty metadata buffers as ready and enqueues them
 * for host collection.
 *
 * @param thread_idx Thread index
 */
void dump_args_flush(int thread_idx);

#endif

#endif  // SRC_COMMON_PLATFORM_INCLUDE_AICPU_ARGS_DUMP_AICPU_H_
