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

#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"

// =============================================================================
// Dual-slot state machine helpers
// =============================================================================

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;
}

// Pure function: read register result -> SlotTransition (no side effects).
SlotTransition SchedulerContext::decide_slot_transition(
    int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id
) {
    SlotTransition t;
    if (pending_id != AICPU_TASK_INVALID && reg_task_id == pending_id) {
        t.matched = true;
        t.running_done = true;  // Serial execution: pending event implies running done
        t.running_freed = true;
        t.pending_freed = true;
        if (reg_state == TASK_FIN_STATE) {
            t.pending_done = true;  // Case 1: pending FIN
        }
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

// Complete one slot's task: subtask counting, mixed completion, deferred release, profiling.
void SchedulerContext::complete_slot_task(
    PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, PTO2SubtaskSlot subslot, int32_t thread_idx,
    int32_t core_id, Handshake *hank, int32_t &completed_this_turn, PTO2TaskSlotState *deferred_release_slot_states[],
    int32_t &deferred_release_count, PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
    ,
    uint64_t dispatch_ts
#endif
) {
#if PTO2_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#else
    (void)hank;
#endif
    bool mixed_complete = sched_->on_subtask_complete(slot_state);
    if (slot_state.payload != nullptr) {
        int32_t reg_err = PTO2_ERROR_NONE;
        PTO2AsyncWaitList::RegisterResult reg_result;
        volatile PTO2DeferredCompletionIngressBuffer *deferred_ingress =
            &deferred_ingress_per_core_[core_id][expected_reg_task_id & 1];
        AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_ingress);
        do {
            reg_result = sched_->async_wait_list.register_deferred(slot_state, async_ctx, mixed_complete, reg_err);
            if (reg_result == PTO2AsyncWaitList::RegisterResult::Skipped) {
                SPIN_WAIT_HINT();
            }
        } while (reg_result == PTO2AsyncWaitList::RegisterResult::Skipped);

        if (reg_result == PTO2AsyncWaitList::RegisterResult::Error) {
            int32_t expected = PTO2_ERROR_NONE;
            sched_->sm_header->sched_error_code.compare_exchange_strong(
                expected, reg_err, std::memory_order_acq_rel, std::memory_order_acquire
            );
            completed_.store(true, std::memory_order_release);
            return;
        }

        if (mixed_complete && reg_result == PTO2AsyncWaitList::RegisterResult::Registered) {
            return;
        }
    }
    if (mixed_complete) {
#if PTO2_PROFILING
        if (is_dump_tensor_enabled()) {
            dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
                thread_idx, slot_state, TensorDumpStage::AFTER_COMPLETION,
                [](ActiveMask active_mask, int raw_subtask_id) {
                    return active_mask.subtask_active(static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                },
                [this](int32_t func_id) {
                    return get_function_bin_addr(func_id);
                }
            );
        }
#endif
#if PTO2_SCHED_PROFILING
        PTO2CompletionStats cstats = sched_->on_mixed_task_complete(slot_state, thread_idx, local_bufs);
        l2_perf.notify_edges_total += cstats.fanout_edges;
        if (cstats.fanout_edges > l2_perf.notify_max_degree) l2_perf.notify_max_degree = cstats.fanout_edges;
        l2_perf.notify_tasks_enqueued += cstats.tasks_enqueued;
        l2_perf.phase_complete_count++;
#else
        sched_->on_mixed_task_complete(slot_state, local_bufs);
#if PTO2_PROFILING
        l2_perf.phase_complete_count++;
#endif
#endif
        if (deferred_release_count < PTO2_DEFERRED_RELEASE_CAP) {
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        } else {
            LOG_INFO_V9("Thread %d: release", thread_idx);
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                int32_t fe =
                    sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
                l2_perf.fanin_edges_total += fe;
                if (fe > l2_perf.fanin_max_degree) l2_perf.fanin_max_degree = fe;
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        }
        completed_this_turn++;
    }

#if PTO2_PROFILING
    if (l2_perf.l2_perf_enabled) {
#if PTO2_SCHED_PROFILING
        uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif
        Handshake *h = &hank[core_id];
        uint64_t finish_ts = get_sys_cnt_aicpu();
        L2PerfBuffer *pbuf = reinterpret_cast<L2PerfBuffer *>(h->l2_perf_records_addr);

        uint64_t fanout_arr[RUNTIME_MAX_FANOUT];
        int32_t fanout_n = 0;
        PTO2DepListEntry *cur = slot_state.fanout_head;
        while (cur != nullptr && fanout_n < RUNTIME_MAX_FANOUT) {
            fanout_arr[fanout_n++] = cur->slot_state->task->task_id.raw;
            cur = cur->next;
        }

        int32_t perf_slot_idx = static_cast<int32_t>(subslot);
        if (l2_perf_aicpu_complete_record(
                pbuf, static_cast<uint32_t>(expected_reg_task_id), slot_state.task->task_id.raw,
                slot_state.task->kernel_id[perf_slot_idx], hank[core_id].core_type, dispatch_ts, finish_ts, fanout_arr,
                fanout_n
            ) != 0) {
            LOG_ERROR(
                "Core %d: l2_perf_aicpu_complete_record failed for task 0x%" PRIx64, core_id,
                static_cast<uint64_t>(slot_state.task->task_id.raw)
            );
        }
#if PTO2_SCHED_PROFILING
        l2_perf.sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
    }
#endif

#if PTO2_PROFILING
    if (is_pmu_enabled()) {
        pmu_aicpu_record_task(
            core_id, thread_idx, slot_state.task->task_id.raw,
            slot_state.task->kernel_id[static_cast<int32_t>(subslot)], hank[core_id].core_type
        );
    }
#endif
}

// Promote pending slot data to running slot. Clears pending fields.
void SchedulerContext::promote_pending_to_running(CoreExecState &core) {
    core.running_slot_state = core.pending_slot_state;
    core.running_reg_task_id = core.pending_reg_task_id;
    core.running_subslot = core.pending_subslot;
#if PTO2_PROFILING
    core.running_dispatch_timestamp = core.pending_dispatch_timestamp;
#endif
    core.pending_slot_state = nullptr;
    core.pending_reg_task_id = AICPU_TASK_INVALID;
}

// Clear running slot (core becomes idle).
void SchedulerContext::clear_running_slot(CoreExecState &core) {
    core.running_slot_state = nullptr;
    core.running_reg_task_id = AICPU_TASK_INVALID;
}

void SchedulerContext::check_running_cores_for_completion(
    int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
    bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
    PTO2LocalReadyBuffer *local_bufs
) {
#if PTO2_SCHED_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#endif
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto running_core_states = tracker.get_all_running_cores();
    while (running_core_states.has_value()) {
        int32_t bit_pos = running_core_states.pop_first();
        int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
        CoreExecState &core = core_exec_states_[core_id];

        // --- Judgment phase: read register, derive transition ---
        uint64_t reg_val = read_reg(core.reg_addr, RegId::COND);
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

#if PTO2_SCHED_PROFILING
        if (l2_perf.l2_perf_enabled) {
            l2_perf.complete_probe_count++;
        }
#endif

        SlotTransition t =
            decide_slot_transition(reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id);
        if (!t.matched) continue;

#if PTO2_SCHED_PROFILING
        if (l2_perf.l2_perf_enabled && (t.running_done || t.pending_done)) {
            l2_perf.complete_hit_count++;
        }
#endif

        // --- Apply phase: execute actions based on transition ---

        // 1. Complete finished tasks (capture pointers before modifying core state)
        if (t.pending_done) {
            complete_slot_task(
                *core.pending_slot_state, core.pending_reg_task_id, core.pending_subslot, thread_idx, core_id, hank,
                completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                core.pending_dispatch_timestamp
#endif
            );
            cur_thread_completed++;
        }
        if (t.running_done) {
            complete_slot_task(
                *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                core.running_dispatch_timestamp
#endif
            );
            cur_thread_completed++;
        }

        // 2. Update slot data
        if (t.running_freed) {
            if (core.pending_slot_state != nullptr && !t.pending_done) {
                promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
            } else {
                clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                if (t.pending_done) {
                    // Case 1: pending FIN observed directly -- clear stale pending fields.
                    // Without this, pending_reg_task_id retains a stale value that blocks
                    // clear_pending_occupied and permanently degrades pipelining.
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
            // Case 4 (running ACK) or Case 2 (pending ACK): clear pending_occupied only
            // when no pending task is currently held. Otherwise pending slot is occupied
            // by a pre-loaded task and must stay protected.
            tracker.clear_pending_occupied(bit_pos);
        }

        // 4. Progress signal (only when running task completes)
        if (t.running_done) {
            made_progress = true;
        }
    }
}

// =============================================================================
// sync_start drain protocol
// =============================================================================

// Take ownership of slot_state and signal all threads to enter drain mode.
// Returns true if this thread won the CAS and owns the drain slot.
// Returns false if another thread already holds drain; caller must re-push slot_state.
//
// Two-phase protocol: CAS 0 -> -1 (sentinel) to claim ownership, store task and
// reset election flag, then release-store block_num.  Other threads acquire-load
// sync_start_pending; seeing block_num > 0 ensures all relaxed stores are visible.
bool SchedulerContext::enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num) {
    int32_t expected = 0;
    if (!drain_state_.sync_start_pending.compare_exchange_strong(
            expected, -1, std::memory_order_relaxed, std::memory_order_relaxed
        )) {
        return false;  // Another thread already holds the drain slot.
    }
    // We own the drain slot.  Store the task and reset election flag before making it visible.
    drain_state_.pending_task = slot_state;
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    // Release store: all stores above are now visible to any thread that
    // acquire-loads sync_start_pending and sees block_num > 0.
    drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
    return true;
}

// Count total available resources across all scheduler threads for a given shape.
int32_t SchedulerContext::count_global_available(PTO2ResourceShape shape) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        total += core_trackers_[t].get_idle_core_offset_states(shape).count();
    }
    return total;
}

// Drain worker: dispatch all blocks in one pass across all threads' trackers.
// Called only when global resources >= block_num, so one pass always suffices.
// All other threads are spinning -- the drain worker has exclusive tracker access.
void SchedulerContext::drain_worker_dispatch(Runtime *runtime, int32_t block_num) {
    PTO2TaskSlotState *slot_state = drain_state_.pending_task;
    if (!slot_state) {
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
        return;
    }
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();

    for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++) {
        auto valid = core_trackers_[t].get_idle_core_offset_states(shape);
        while (valid.has_value() && slot_state->next_block_idx < block_num) {
            dispatch_block(runtime, t, valid.pop_first(), *slot_state, shape, false);
            slot_state->next_block_idx++;
        }
    }

    // All blocks dispatched -- clear drain state.
    // Release fence ensures tracker mutations are visible to threads that
    // acquire-load sync_start_pending == 0 and resume normal operation.
    std::atomic_thread_fence(std::memory_order_release);
    drain_state_.pending_task = nullptr;
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    drain_state_.sync_start_pending.store(0, std::memory_order_release);
}

// Called by each scheduler thread when drain_state_.sync_start_pending != 0.
//
// Protocol (single-stage ack barrier):
//   1. Ack barrier: all threads signal they've stopped dispatch, then spin
//      until all ack bits are set.
//      If this thread's bit gets cleared while waiting, a reset occurred -- return.
//   2. Election: one thread wins the CAS and becomes the drain worker.
//      If resources are insufficient, reset ack/election fields and return --
//      all threads resume completion polling to free running cores, then retry.
//   3. Dispatch: elected thread dispatches all blocks (one pass, resources guaranteed).
//      Non-elected threads spin-wait until sync_start_pending == 0.
//      During dispatch the elected thread has exclusive tracker access.
void SchedulerContext::handle_drain_mode(Runtime *runtime, int32_t thread_idx) {
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
    PTO2TaskSlotState *slot_state = drain_state_.pending_task;
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
    drain_worker_dispatch(runtime, block_num);
}
