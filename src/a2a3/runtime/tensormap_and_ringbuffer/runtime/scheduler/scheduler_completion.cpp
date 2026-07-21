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

#include "scheduler_context.h"

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "common.h"
#include "callable.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"

SlotTransition SchedulerContext::decide_slot_transition(
    int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id
) {
    SlotTransition t;
    if (pending_id != AICPU_TASK_INVALID && reg_task_id == pending_id) {
        t.matched = true;
        t.running_done = true;  // Serial execution: pending event implies running done
        t.running_freed = true;
        t.pending_freed = true;
        if (reg_state == TASK_FIN_STATE) t.pending_done = true;  // Case 1: pending FIN
        // else: Case 2: pending ACK (pending_done stays false)
    } else if (reg_task_id == running_id) {
        if (reg_state == TASK_FIN_STATE) {
            if (pending_id == AICPU_TASK_INVALID) {
                // Case 3.2: running FIN, no pending -> core goes idle
                t.matched = true;
                t.running_done = true;
                t.running_freed = true;
            }
            // Case 3.1: running FIN, pending exists -> skip (transient state).
            // Case 1/2 (pending ACK/FIN) will complete running implicitly via running_done=true.
        } else {
            // Case 4: running ACK -- only pending_freed (slot now hardware-latched)
            t.matched = true;
            t.pending_freed = true;
        }
    }
    return t;
}

void SchedulerContext::complete_slot_task(
    PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, int32_t core_id, int32_t &completed_this_turn
) {
    AICoreCompletionMailbox *mailbox = rt_ != nullptr ? rt_->aicore_mailbox : nullptr;
    bool defer_completion_to_consumer = false;

    if (slot_state.payload != nullptr) {
        volatile DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][expected_reg_task_id & 1];
        // (q) Read count first. AICore only writes error_code as part of a
        // condition-registration attempt that also increments count, so
        // count == 0 ⇒ no error and no conditions to forward. This is the
        // common path for kernels that don't use async waits (paged
        // attention, GEMM, etc.) and saves an L1 load + branch per call.
        uint32_t cond_count = deferred_slab->count;
        if (cond_count != 0) {
            int32_t slab_err = deferred_slab->error_code;
            if (slab_err != PTO2_ERROR_NONE) {
                int32_t expected = PTO2_ERROR_NONE;
                sched_->sm_header->sched_error_code.compare_exchange_strong(
                    expected, slab_err, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                return;
            }
            if (cond_count > MAX_COMPLETIONS_PER_TASK) {
                int32_t expected = PTO2_ERROR_NONE;
                sched_->sm_header->sched_error_code.compare_exchange_strong(
                    expected, PTO2_ERROR_ASYNC_REGISTRATION_FAILED, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                return;
            }

            slot_state.any_subtask_deferred.store(true, std::memory_order_release);

            const PTO2TaskId token = slot_state.task->task_id;
            for (uint32_t i = 0; i < cond_count; ++i) {
                volatile DeferredCompletionEntry *e = &deferred_slab->entries[i];
                while (!mailbox->try_push_condition(token, e->addr, e->expected_value, e->engine, e->completion_type)) {
                    sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
                    SPIN_WAIT_HINT();
                }
            }
        }
    }

    bool mixed_complete = sched_->on_subtask_complete(slot_state);

    if (mixed_complete && slot_state.payload != nullptr &&
        slot_state.any_subtask_deferred.load(std::memory_order_acquire)) {
        // Some subtask of this task registered conditions; finish the
        // registration by handing the slot_state off to the consumer.
        while (!mailbox->try_push_normal_done(slot_state.task->task_id, reinterpret_cast<uint64_t>(&slot_state))) {
            sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
            SPIN_WAIT_HINT();
        }
        defer_completion_to_consumer = true;
    }

    if (mixed_complete && !defer_completion_to_consumer) {
        sched_->on_mixed_task_complete(slot_state);
        completed_this_turn++;
    }
}

void SchedulerContext::promote_pending_to_running(CoreExecState &core) {
    core.running_slot_state = core.pending_slot_state;
    core.running_reg_task_id = core.pending_reg_task_id;
    core.running_subslot = core.pending_subslot;
    core.pending_slot_state = nullptr;
    core.pending_reg_task_id = AICPU_TASK_INVALID;
}

void SchedulerContext::clear_running_slot(CoreExecState &core) {
    core.running_slot_state = nullptr;
    core.running_reg_task_id = AICPU_TASK_INVALID;
}

void SchedulerContext::check_running_cores_for_completion(
    int32_t thread_idx, int32_t &completed_this_turn, int32_t &cur_thread_completed, bool &made_progress
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto running_core_states = tracker.get_all_running_cores();
    while (running_core_states.has_value()) {
        int32_t bit_pos = running_core_states.pop_first();
        int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
        CoreExecState &core = core_exec_states_[core_id];

        uint64_t reg_val = static_cast<uint64_t>(*core.cond_ptr);
        rmb();
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

        SlotTransition t =
            decide_slot_transition(reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id);
        if (!t.matched) continue;

        // --- Apply phase: execute actions based on transition ---

        // 1. Complete finished tasks (capture pointers before modifying core state)
        if (t.pending_done) {
            // Task-timing finish: latest FIN observation for a tagged task, folded
            // as max. Sampled before complete_slot_task clears pending_slot_state.
            if (core.pending_slot_state->task->task_timing_slot != TASK_TIMING_SLOT_NONE)
                aicpu_task_timing_finish(core.pending_slot_state->task->task_timing_slot, thread_idx);
            complete_slot_task(*core.pending_slot_state, core.pending_reg_task_id, core_id, completed_this_turn);
            cur_thread_completed++;
        }
        if (t.running_done) {
            if (core.running_slot_state->task->task_timing_slot != TASK_TIMING_SLOT_NONE)
                aicpu_task_timing_finish(core.running_slot_state->task->task_timing_slot, thread_idx);
            complete_slot_task(*core.running_slot_state, core.running_reg_task_id, core_id, completed_this_turn);
            cur_thread_completed++;
        }

        // 2. Update slot data
        if (t.running_freed) {
            if (core.pending_slot_state != nullptr && !t.pending_done) {
                promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
            } else {
                clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                if (t.pending_done) {
                    core.pending_slot_state = nullptr;
                    core.pending_reg_task_id = AICPU_TASK_INVALID;
                }
            }
        }

        // 3. Update tracker bitmap
        bool is_idle = (core.running_reg_task_id == AICPU_TASK_INVALID);
        if (is_idle) {
            tracker.change_core_state(bit_pos);       // Mark idle
            tracker.clear_pending_occupied(bit_pos);  // Idle safeguard: no payload to protect
        } else if (t.pending_freed && core.pending_reg_task_id == AICPU_TASK_INVALID) {
            tracker.clear_pending_occupied(bit_pos);
        }

        // 4. Progress signal (only when running task completes)
        if (t.running_done) made_progress = true;
    }
}

bool SchedulerContext::enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num) {
    int32_t expected = 0;
    if (!drain_state_.sync_start_pending.compare_exchange_strong(
            expected, -1, std::memory_order_relaxed, std::memory_order_relaxed
        ))
        return false;  // Another thread already holds the drain slot.
    // We own the drain slot.  Store the task and reset election flag before making it visible.
    drain_state_.pending_task.store(slot_state, std::memory_order_release);
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    // Release store: all stores above are now visible to any thread that
    // acquire-loads sync_start_pending and sees block_num > 0.
    drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
    return true;
}

int32_t SchedulerContext::count_global_available(PTO2ResourceShape shape) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++)
        total += core_trackers_[t].get_idle_core_offset_states(shape).count();
    return total;
}

void SchedulerContext::drain_worker_dispatch(int32_t block_num) {
    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    if (!slot_state) {
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
        return;
    }
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();

    for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++) {
        auto valid = core_trackers_[t].get_idle_core_offset_states(shape);
        int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
        int32_t claim = std::min(valid.count(), remaining);
        int32_t start = slot_state->next_block_idx;
        slot_state->next_block_idx += claim;
        PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
        int handle_count = 0;
        for (int32_t b = 0; b < claim; b++) {
            auto core_offset = valid.pop_first();
            handle_count += prepare_block_for_dispatch(
                t, core_offset, *slot_state, shape, false, start + b, &handles[handle_count]
            );
        }
        wmb();
        uint64_t dispatch_ts = 0;
        for (int i = 0; i < handle_count; i++)
            publish_subtask_to_core(handles[i], dispatch_ts, t);
    }

    std::atomic_thread_fence(std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    drain_state_.sync_start_pending.store(0, std::memory_order_release);
}

void SchedulerContext::handle_drain_mode(int32_t thread_idx) {
    // Spin until drain is fully initialized (sentinel -1 -> block_num > 0).
    int32_t block_num;
    do {
        block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
    } while (block_num < 0);
    if (block_num == 0) return;

    uint32_t all_acked = (1u << active_sched_threads_) - 1;

    // Ack barrier -- signal this thread has stopped dispatch.
    drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    // Spin until all threads have acked.
    // If our bit is cleared while waiting, elected reset due to insufficient resources.
    while (true) {
        uint32_t ack = drain_state_.drain_ack_mask.load(std::memory_order_acquire);
        if ((ack & all_acked) == all_acked) break;
        if ((ack & (1u << thread_idx)) == 0) return;
        SPIN_WAIT_HINT();
    }

    // Election -- exactly one thread wins the CAS.
    int32_t expected = 0;
    drain_state_.drain_worker_elected.compare_exchange_strong(
        expected, thread_idx + 1, std::memory_order_acquire, std::memory_order_relaxed
    );

    if (drain_state_.drain_worker_elected.load(std::memory_order_relaxed) != thread_idx + 1) {
        // Non-elected: spin-wait for drain completion or resource-insufficient reset.
        while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        return;
    }

    // Elected: check if global resources are sufficient.
    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    if (slot_state == nullptr) {
        drain_state_.drain_worker_elected.store(0, std::memory_order_release);
        return;
    }
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();
    int32_t available = count_global_available(shape);

    if (available < block_num) {
        // Insufficient resources -- reset drain fields so threads can resume
        // completion polling to free running cores, then retry.
        drain_state_.drain_ack_mask.store(0, std::memory_order_release);
        drain_state_.drain_worker_elected.store(0, std::memory_order_release);
        return;
    }

    // Dispatch -- all other threads are spinning, elected thread has exclusive tracker access.
    drain_worker_dispatch(block_num);
}
