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

#include <tracr/tracr.hpp>
#include <tracr_simpler_markers.hpp>

#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/l2_swimlane_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_swimlane_collector_aicpu.h"
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
    int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id, bool pending_gated
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
            } else if (pending_gated) {
                // Case 3.3: running FIN, pending is a SPECULATIVE GATED task. The
                // Case 3.1 "wait for the pending's ack" shortcut assumes the AICore
                // immediately runs the pending task; a gated task instead spins on
                // its doorbell and never acks until its producer completes — and
                // that producer's completion depends on collecting THIS running FIN.
                // Waiting would deadlock. Complete the running FIN now and promote
                // the gated task (it then skip-gates until its doorbell). pending is
                // NOT freed (it promotes, not retires) so the bitmap update keeps the
                // core off-limits — no second gated block, no doorbell overwrite.
                t.matched = true;
                t.running_done = true;
                t.running_freed = true;
            }
            // Case 3.1: running FIN, NON-gated pending exists -> skip (transient
            // state). Case 1/2 (pending ack/FIN) completes running implicitly.
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
    PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, [[maybe_unused]] PTO2SubtaskSlot subslot,
    int32_t thread_idx, int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
    PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count, PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
    ,
    uint64_t dispatch_ts, uint64_t finish_ts
#endif
) {
#if PTO2_PROFILING
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#else
    (void)hank;
#endif
    // MPSC fast-path is opt-in per task: only tasks with at least one subtask
    // that registered a deferred condition route through the mailbox. Pure
    // non-deferred tasks complete inline on this thread (matching pre-MPSC
    // behavior — keeps the common case parallelized across scheduler threads
    // instead of serializing through the single consumer). The
    // any_subtask_deferred flag on slot_state is the discriminator; it's set
    // (release) before on_subtask_complete and read (acquire) after, so the
    // last subtask sees flag writes from any earlier subtask of the same task.
    AICoreCompletionMailbox *mailbox = rt_ != nullptr ? rt_->aicore_mailbox : nullptr;
    bool defer_completion_to_consumer = false;

    if (slot_state.payload != nullptr) {
        volatile DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][expected_reg_task_id & 1];
        int32_t slab_err = deferred_slab->error_code;
        if (slab_err != PTO2_ERROR_NONE) {
            int32_t expected = PTO2_ERROR_NONE;
            sched_->sm_header->sched_error_code.compare_exchange_strong(
                expected, slab_err, std::memory_order_acq_rel, std::memory_order_acquire
            );
            completed_.store(true, std::memory_order_release);
            return;
        }

        uint32_t cond_count = deferred_slab->count;
        if (cond_count > MAX_COMPLETIONS_PER_TASK) {
            int32_t expected = PTO2_ERROR_NONE;
            sched_->sm_header->sched_error_code.compare_exchange_strong(
                expected, PTO2_ERROR_ASYNC_REGISTRATION_FAILED, std::memory_order_acq_rel, std::memory_order_acquire
            );
            completed_.store(true, std::memory_order_release);
            return;
        }

        if (cond_count > 0) {
            // Publish "this task is deferred" before on_subtask_complete so the
            // acq_rel fetch_add inside on_subtask_complete makes the flag
            // visible to whichever subtask sees mixed_complete=true (which may
            // be this thread or a later one).
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

#if PTO2_PROFILING
    // Sub-block retire that did not finish the slot: record it so the poll
    // iteration becomes visible on the scheduler lane (the SPMD harvest tail).
    if (!mixed_complete && l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        l2_swimlane.phase_subretire_count++;
    }
#endif

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
#if PTO2_PROFILING
        if (is_dump_args_enabled()) {
            dump_args_for_task<PTO2_SUBTASK_SLOT_COUNT>(
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
#if PTO2_PROFILING
        // Time the fanout (fanin release + consumer doorbells) so it renders as
        // a child bar nested inside this iteration's Complete bar — surfacing how
        // much of a long complete is the 50-consumer walk + doorbell vs scan.
        uint64_t fanout_t0 = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
#if PTO2_SCHED_PROFILING
        // SCHED_PROFILING variant takes thread_idx for its per-thread atomic
        // counter side-effects (g_sched_*_atomic_count[thread_idx], consumed
        // by the otc_* log lines). Its return value is unused.
        (void)sched_->on_mixed_task_complete(slot_state, thread_idx, local_bufs);
#else
        sched_->on_mixed_task_complete(slot_state, local_bufs);
#endif
#if PTO2_PROFILING
        if (fanout_t0 != 0) {
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::Fanout, fanout_t0, get_sys_cnt_aicpu(),
                l2_swimlane.sched_loop_count, /*tasks_processed=*/0
            );
        }
        l2_swimlane.phase_complete_count++;
#endif
        if (deferred_release_count < PTO2_DEFERRED_RELEASE_CAP) {
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        } else {
            LOG_INFO_V9("Thread %d: release", thread_idx);
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                // SCHED_PROFILING variant takes thread_idx for the per-thread
                // atomic counter side-effects. The return value is unused.
                (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        }
        completed_this_turn++;
    }

#if PTO2_PROFILING
    // Level gate: at AICORE_TIMING (level=1) the AICore record alone carries
    // {start, end, task_token_raw}, host resolves func_id/core_type from
    // dep_gen / per-core mapping, and AICPU has nothing to write. Only at
    // AICPU_TIMING (level=2) and above does AICPU contribute dispatch/finish
    // timestamps via complete_task. Bypassing here saves the per-completion
    // hot-path cost (counter inc + ring lookup + record store + wmb + buffer
    // rotation bookkeeping) for runs that only want AICore timing.
    if (l2_swimlane.l2_swimlane_enabled && l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
#if PTO2_SCHED_PROFILING
        uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif

        if (l2_swimlane_aicpu_complete_task(
                core_id, thread_idx, static_cast<uint32_t>(expected_reg_task_id), dispatch_ts, finish_ts
            ) != 0) {
            LOG_ERROR(
                "Core %d: l2_swimlane_aicpu_complete_task failed for task 0x%" PRIx64, core_id,
                static_cast<uint64_t>(slot_state.task->task_id.raw)
            );
        }
#if PTO2_SCHED_PROFILING
        l2_swimlane.sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
    }

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
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#endif
#if PTO2_PROFILING
    // Time this whole pass and count cores whose MMIO COND we actually read
    // (skip-gated cores excluded) — emit a Scan bar so the per-pass MMIO cost
    // and effective re-poll cadence are visible directly.
    bool scan_record = l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES;
    uint64_t scan_t0 = scan_record ? get_sys_cnt_aicpu() : 0;
    uint32_t scan_cores = 0;
#endif
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto running_core_states = tracker.get_all_running_cores();
    while (running_core_states.has_value()) {
        int32_t bit_pos = running_core_states.pop_first();
        int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
        CoreExecState &core = core_exec_states_[core_id];

        // Skip gated speculative cores. A STAGED task is parked on this core
        // waiting for its doorbell — it physically cannot ACK/FIN yet, so
        // reading its COND (MMIO, and the core is hot-spinning on its own SPR)
        // every poll is pure waste that drags out the completion phase. The
        // doorbell (try_speculative_release) flips spec_state to DISPATCHED, at
        // which point the core becomes pollable again and its FIN is caught.
        // Cheap cacheable load; no MMIO. Pending slot is empty while gated.
        {
            PTO2TaskSlotState *rs = core.running_slot_state;
            if (rs != nullptr && rs->payload != nullptr &&
                rs->payload->spec_state.load(std::memory_order_relaxed) == PTO2_SPEC_STAGING) {
                continue;
            }
        }

        // --- Judgment phase: read register, derive transition ---
        // Use the precomputed cond_ptr (resolved once in handshake) to skip
        // the reg_offset switch and reg_addr addition on every poll.
#if PTO2_PROFILING
        scan_cores++;  // counts cores whose MMIO COND we actually read this pass
#endif
        uint64_t reg_val = static_cast<uint64_t>(*core.cond_ptr);
        // ARM64 allows Device-nGnRnE -> Normal-cacheable load reorder; the
        // rmb() pins any AICore-published cacheable reads downstream of the
        // FIN observation. Replaces the post-`__sync_synchronize` that the
        // old read_reg() helper carried implicitly.
        rmb();
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

#if PTO2_SCHED_PROFILING
        if (l2_swimlane.l2_swimlane_enabled) {
            l2_swimlane.complete_probe_count++;
        }
#endif

        // A pending task is "gated" when it is a speculative pre-stage still
        // waiting on its doorbell (STAGED): it will not ack on the producer's FIN,
        // so the Case 3.1 wait-for-pending-ack shortcut would deadlock. Detect it
        // so decide_slot_transition completes the running FIN and promotes it.
        bool pending_gated =
            (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr &&
             core.pending_slot_state->payload->spec_state.load(std::memory_order_relaxed) == PTO2_SPEC_STAGING);
        SlotTransition t = decide_slot_transition(
            reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id, pending_gated
        );
        if (!t.matched) continue;

#if PTO2_SCHED_PROFILING
        if (l2_swimlane.l2_swimlane_enabled && (t.running_done || t.pending_done)) {
            l2_swimlane.complete_hit_count++;
        }
#endif

#if PTO2_PROFILING
        // Capture finish_ts at the FIN observation point — right after rmb()
        // above pinned the cacheable AICore reads downstream of the register
        // load, and BEFORE any fanin / deferred-release work. Anything later
        // (slot transition apply, complete_slot_task fanin processing) would
        // charge AICPU completion-processing cost to the (end → finish)
        // span, masking the actual FIN-delivery latency.
        uint64_t finish_ts = 0;
        if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING && (t.pending_done || t.running_done)) {
            finish_ts = get_sys_cnt_aicpu();
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
                core.pending_dispatch_timestamp, finish_ts
#endif
            );
            cur_thread_completed++;
            INSTRUMENTATION_MARK_RESET(sched_thread_num_ + 1 + core_id);
        }
        if (t.running_done) {
            complete_slot_task(
                *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                core.running_dispatch_timestamp, finish_ts
#endif
            );
            cur_thread_completed++;
            INSTRUMENTATION_MARK_RESET(sched_thread_num_ + 1 + core_id);
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
#if PTO2_PROFILING
    if (scan_record && scan_cores > 0) {
        l2_swimlane_aicpu_record_sched_phase(
            thread_idx, L2SwimlaneSchedPhaseKind::Scan, scan_t0, get_sys_cnt_aicpu(),
            sched_l2_swimlane_[thread_idx].sched_loop_count, scan_cores
        );
    }
#endif
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
    drain_state_.pending_task.store(slot_state, std::memory_order_release);
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
void SchedulerContext::drain_worker_dispatch(int32_t block_num) {
    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    if (!slot_state) {
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
        return;
    }
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();

    for (int32_t t = 0;
         t < active_sched_threads_ && slot_state->next_block_idx.load(std::memory_order_relaxed) < block_num; t++) {
        auto valid = core_trackers_[t].get_idle_core_offset_states(shape);
        int32_t start = slot_state->next_block_idx.load(std::memory_order_relaxed);
        int32_t remaining = slot_state->logical_block_num - start;
        int32_t claim = std::min(valid.count(), remaining);
        slot_state->next_block_idx.store(static_cast<int16_t>(start + claim), std::memory_order_relaxed);
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
#if PTO2_PROFILING
        if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
            dispatch_ts = get_sys_cnt_aicpu();
        }
#endif
        for (int i = 0; i < handle_count; i++) {
            publish_subtask_to_core(handles[i], dispatch_ts);
        }
    }

    // All blocks dispatched -- clear drain state.
    // Release fence ensures tracker mutations are visible to threads that
    // acquire-load sync_start_pending == 0 and resume normal operation.
    std::atomic_thread_fence(std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);
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
void SchedulerContext::handle_drain_mode(int32_t thread_idx) {
    // Every spin in this function honors is_completed(): once the run latches
    // completed_ (all tasks done, or a fatal error raised elsewhere), peers leave
    // the dispatch loop and stop participating in the drain. A thread parked in a
    // drain spin would then wait forever for acks / a gate-open that can no longer
    // arrive -- the AICPU watchdog never fires here because these spins live
    // outside the dispatch loop's wall-clock budget, so the hang escalates straight
    // to the 3 s STARS op-exec timeout (507018) and poisons the device. Bailing on
    // completed_ is always safe: any pending sync_start task is either already
    // dispatched (a stale re-popped slot) or moot under teardown, and deinit()
    // resets drain_state_ before the next run, so leaving it dirty is harmless.
    // Spin until drain is fully initialized (sentinel -1 -> block_num > 0).
    int32_t block_num;
    do {
        if (is_completed()) return;
        block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
    } while (block_num < 0);
    if (block_num == 0) return;

    uint32_t all_acked = (1u << active_sched_threads_) - 1;

    // Ack barrier -- signal this thread has stopped dispatch.
    drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    // Spin until all threads have acked.
    // If our bit is cleared while waiting, elected reset due to insufficient resources.
    while (true) {
        if (is_completed()) return;
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
            if (is_completed()) return;
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        return;
    }

    // Elected: check if global resources are sufficient.
    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    if (slot_state == nullptr) {
        // pending_task is observed null only when a concurrent drain completion
        // already cleared it (drain_worker_dispatch nulls it before reopening the
        // gate). That drain is done and this is a stale-elected thread, so just
        // release the election lock and return. Do NOT clear drain_ack_mask or
        // sync_start_pending: a *new* drain run may already be active and
        // accumulating acks, and zeroing them would corrupt it into a hang.
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
