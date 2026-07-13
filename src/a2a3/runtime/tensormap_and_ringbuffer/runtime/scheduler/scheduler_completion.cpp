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
#include "aicpu/args_dump_aicpu.h"

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
                // Case 3.3: running FIN, pending is a EARLY-DISPATCH GATED task. The
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
    PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
#if SIMPLER_DFX
    ,
    uint64_t dispatch_ts, uint64_t finish_ts
#endif
) {
#if SIMPLER_DFX
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#else
    (void)hank;
#endif
    // MPSC fast-path is opt-in per task: only tasks with at least one subtask
    // that registered a deferred condition route through the mailbox. Pure
    // non-deferred tasks complete inline on this thread (matching pre-MPSC
    // behavior — keeps the common case parallelized across scheduler threads
    // instead of serializing through the single consumer). The
    // deferred-completion flag on slot_state is the discriminator; it's set
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
            // visible to whichever subtask sees task_complete=true (which may
            // be this thread or a later one).
            slot_state.mark_any_subtask_deferred();

            const PTO2TaskId token = slot_state.task->task_id;
            for (uint32_t i = 0; i < cond_count; ++i) {
                volatile DeferredCompletionEntry *e = &deferred_slab->entries[i];
                while (!mailbox->try_push_condition(token, e->addr, e->expected_value, e->engine, e->completion_type)) {
                    sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
                    SPIN_WAIT_HINT();
                }
            }
            // Re-clear for the next reuse of this (core, buf) slot. Done here — on
            // the hot cache line we just read — instead of on every dispatch, since
            // only a deferred task (count > 0) ever dirties it. error_code needs no
            // reset: a non-NONE code aborted the run above.
            deferred_slab->count = 0;
        }
    }

    bool task_complete = sched_->on_subtask_complete(slot_state);

#if SIMPLER_DFX
    // Sub-block retire that did not finish the slot: record it so the poll
    // iteration becomes visible on the scheduler lane (the SPMD harvest tail).
    if (!task_complete && l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        l2_swimlane.phase_subretire_count++;
    }
#endif

    if (task_complete && slot_state.payload != nullptr && slot_state.has_any_subtask_deferred()) {
        // Some subtask of this task registered conditions; finish the
        // registration by handing the slot_state off to the consumer.
        while (!mailbox->try_push_normal_done(slot_state.task->task_id, reinterpret_cast<uint64_t>(&slot_state))) {
            sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
            SPIN_WAIT_HINT();
        }
        defer_completion_to_consumer = true;
    }

    if (task_complete && !defer_completion_to_consumer) {
#if SIMPLER_DFX
        if (is_dump_args_enabled()) {
            dump_args_for_task<PTO2_SUBTASK_SLOT_COUNT>(
                thread_idx, slot_state, ArgsDumpStage::AFTER_COMPLETION,
                [](ActiveMask active_mask, int raw_subtask_id) {
                    return active_mask.subtask_active(static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                },
                [this](int32_t func_id) {
                    return get_function_bin_addr(func_id);
                }
            );
        }
#endif
#if SIMPLER_DFX
        // Time Resolve (walk the consumer list, decrement each consumer's
        // fanin, push the newly-ready ones, ring doorbells for early-dispatch
        // hits) so it renders as a child bar nested inside this iteration's
        // Complete bar. The 1 µs floor below filters out the ~88% of tasks
        // with 1-2 consumers (~500 ns Resolve) so only the long broadcast /
        // reduction walks stand out on the lane.
        uint64_t resolve_t0 = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
        // [[maybe_unused]] silences -Werror=unused-but-set-variable on the
        // profiling-flags-smoke build path where SIMPLER_DFX is OFF and
        // the Resolve emit below is excluded.
        [[maybe_unused]] uint32_t consumers_resolved = 0;
#if SIMPLER_SCHED_PROFILING
        // SCHED_PROFILING variant takes thread_idx for its per-thread atomic
        // counter side-effects (g_sched_*_atomic_count[thread_idx], consumed
        // by the otc_* log lines). It returns CompletionStats whose
        // `fanout_edges` is the consumer-walk count.
        consumers_resolved = sched_->on_task_complete(slot_state, thread_idx).fanout_edges;
#else
        consumers_resolved = sched_->on_task_complete(slot_state);
#endif
#if SIMPLER_DFX
        if (resolve_t0 != 0) {
            uint64_t resolve_t1 = get_sys_cnt_aicpu();
            // Filter: drop Resolve bars under 1 µs so the lane shows only
            // resolves that did meaningful work (high consumer counts or
            // doorbells). 50 cycles @ 50 MHz = 1 µs (PLATFORM_PROF_SYS_CNT_FREQ
            // is the device sys-cnt frequency).
            constexpr uint64_t RESOLVE_EMIT_MIN_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ / 1'000'000;  // 1 µs
            if (resolve_t1 - resolve_t0 >= RESOLVE_EMIT_MIN_CYCLES) {
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Resolve, resolve_t0, resolve_t1, l2_swimlane.sched_loop_count,
                    consumers_resolved
                );
            }
        }
        l2_swimlane.phase_complete_count++;
#endif
        if (deferred_release_count < PTO2_DEFERRED_RELEASE_CAP) {
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        } else {
            LOG_INFO_V9("Thread %d: release", thread_idx);
            while (deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
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

#if SIMPLER_DFX
    // Level gate: at AICORE_TIMING (level=1) the AICore record alone carries
    // {start, end, task_token_raw}, host resolves func_id/core_type from
    // dep_gen / per-core mapping, and AICPU has nothing to write. Only at
    // AICPU_TIMING (level=2) and above does AICPU contribute dispatch/finish
    // timestamps via complete_task. Bypassing here saves the per-completion
    // hot-path cost (counter inc + ring lookup + record store + wmb + buffer
    // rotation bookkeeping) for runs that only want AICore timing.
    if (l2_swimlane.l2_swimlane_enabled && l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
#if SIMPLER_SCHED_PROFILING
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
#if SIMPLER_SCHED_PROFILING
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
#if SIMPLER_DFX
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
    bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
) {
#if SIMPLER_SCHED_PROFILING
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#endif
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto running_core_states = tracker.get_all_running_cores();
    while (running_core_states.has_value()) {
        int32_t bit_pos = running_core_states.pop_first();
        int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
        CoreExecState &core = core_exec_states_[core_id];

        // Skip gated early-dispatch cores. A STAGED task is parked on this core
        // waiting for its doorbell — it physically cannot ACK/FIN yet, so
        // reading its COND (MMIO, and the core is hot-spinning on its own SPR)
        // every poll is pure waste that drags out the completion phase. The
        // doorbell (try_early_dispatch_release) flips early_dispatch_state to DISPATCHED, at
        // which point the core becomes pollable again and its FIN is caught.
        // Cheap cacheable load; no MMIO. Pending slot is empty while gated.
        {
            PTO2TaskSlotState *rs = core.running_slot_state;
            if (rs != nullptr && rs->payload != nullptr &&
                rs->payload->early_dispatch_state.load(std::memory_order_relaxed) == PTO2_EARLY_DISPATCH_STAGING) {
                continue;
            }
        }

        // --- Judgment phase: read register, derive transition ---
        // Use the precomputed cond_ptr (resolved once in handshake) to skip
        // the reg_offset switch and reg_addr addition on every poll.
        uint64_t reg_val = static_cast<uint64_t>(*core.cond_ptr);
        // ARM64 allows Device-nGnRnE -> Normal-cacheable load reorder; the
        // rmb() pins any AICore-published cacheable reads downstream of the
        // FIN observation. Replaces the post-`__sync_synchronize` that the
        // old read_reg() helper carried implicitly.
        rmb();
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

#if SIMPLER_SCHED_PROFILING
        if (l2_swimlane.l2_swimlane_enabled) {
            l2_swimlane.complete_probe_count++;
        }
#endif

        // A pending task is "gated" when it is an early-dispatch pre-stage still parked on
        // its doorbell: it will not ack on the producer's FIN, so the Case 3.1 wait-for-
        // pending-ack shortcut would deadlock. Detect it so decide_slot_transition completes
        // the running FIN and PROMOTES it (Case 3.3) instead.
        //
        // "Gated" is "not yet launched (rung)", which is NOT the same as
        // early_dispatch_state. STAGING covers the pre-release window. But a sync_start block
        // stays gated even AFTER its producer releases (early_dispatch_state STAGING ->
        // DISPATCHED): the cohort is not rung until the rendezvous has assembled EVERY core
        // into a running slot, and this pending block (being promoted now) is by definition
        // not yet counted, so the ring has not fired and it is still gated. Classifying it by
        // STAGING alone would, once the producer's release beats the last promotion, treat it
        // as a normal task and wait for an ack that never comes -> deadlock (the
        // nondeterministic sync_start stall).
        uint8_t pending_ss =
            (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr) ?
                core.pending_slot_state->payload->early_dispatch_state.load(std::memory_order_relaxed) :
                static_cast<uint8_t>(PTO2_EARLY_DISPATCH_NONE);
        bool pending_gated =
            (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr &&
             (pending_ss == PTO2_EARLY_DISPATCH_STAGING ||
              (pending_ss == PTO2_EARLY_DISPATCH_DISPATCHED &&
               core.pending_slot_state->active_mask.requires_sync_start())));
        SlotTransition t = decide_slot_transition(
            reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id, pending_gated
        );
        if (!t.matched) continue;

#if SIMPLER_DFX
        // Release an ACK-gated AICore swimlane buffer if this matched ACK/FIN is
        // the one a rotation is waiting on: it proves the core advanced past the
        // just-rotated buffer's tail record (FIN precedes the record write on
        // this runtime, so rotation could not release the buffer itself).
        if (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED) {
            l2_swimlane_aicpu_on_aicore_ack(core_id, thread_idx, static_cast<uint32_t>(reg_task_id));
        }
#endif

#if SIMPLER_SCHED_PROFILING
        if (l2_swimlane.l2_swimlane_enabled && (t.running_done || t.pending_done)) {
            l2_swimlane.complete_hit_count++;
        }
#endif

#if SIMPLER_DFX
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
                completed_this_turn, deferred_release_slot_states, deferred_release_count
#if SIMPLER_DFX
                ,
                core.pending_dispatch_timestamp, finish_ts
#endif
            );
            cur_thread_completed++;
        }
        if (t.running_done) {
            complete_slot_task(
                *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                completed_this_turn, deferred_release_slot_states, deferred_release_count
#if SIMPLER_DFX
                ,
                core.running_dispatch_timestamp, finish_ts
#endif
            );
            cur_thread_completed++;
        }

        // 2. Update slot data
        if (t.running_freed) {
            if (core.pending_slot_state != nullptr && !t.pending_done) {
                // A gated sync_start block promoting into the running slot advances the
                // rendezvous. Capture that BEFORE promote nulls the pending fields; after
                // it lands, bump running_slot_count and ring iff this was the block that
                // completed the cohort (and the producer already released).
                PTO2TaskSlotState *promoted = core.pending_slot_state;
                bool sync_start_promote = pending_gated && promoted->active_mask.requires_sync_start();
                promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
                if (sync_start_promote) {
                    promoted->payload->running_slot_count.fetch_add(1, std::memory_order_seq_cst);
                    if (sched_->maybe_rendezvous_ring(*promoted)) {
                        sched_->propagate_dispatch_fanin(*promoted);
                    }
                }
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
    // We own the drain slot.  Store the task and reset the coordination flags before making
    // it visible.
    drain_state_.pending_task.store(slot_state, std::memory_order_release);
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    drain_state_.drain_stage_go.store(0, std::memory_order_relaxed);
    drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_running_staged.store(0, std::memory_order_relaxed);
    // Release store: all stores above are now visible to any thread that
    // acquire-loads sync_start_pending and sees block_num > 0.
    drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
    return true;
}

// Count total available resources across all scheduler threads for a given shape.
// include_pending adds each thread's pending-capable cores/clusters — used by the
// gated (early) sync_start drain, which pre-stages onto idle running slots AND onto
// busy cores' pending slots. The ready drain (include_pending=false) counts idle only.
int32_t SchedulerContext::count_global_available(PTO2ResourceShape shape, uint8_t core_mask, bool include_pending) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        if (shape == PTO2ResourceShape::MIX) {
            // Gated MIX uses split placement (each core to running-if-idle / pending-if-busy),
            // so a cluster is available iff every used core has some free slot. The ready
            // path (include_pending=false) still needs whole-cluster idle placement.
            total += include_pending ? core_trackers_[t].count_mix_split_clusters(core_mask) :
                                       core_trackers_[t].count_mix_running_clusters(core_mask);
        } else {
            total += core_trackers_[t].get_idle_core_offset_states(shape).count();
            if (include_pending) {
                total += core_trackers_[t].get_pending_core_offset_states(shape).count();
            }
        }
    }
    return total;
}

// One thread's share of the drain staging: CAS-claim block indices and publish them onto
// THIS thread's own cores, concurrently with peers. Returns the number of cores placed on a
// RUNNING slot (the rendezvous seed contribution). Each thread touches only its own tracker
// and its own cores' doorbell-table entries; the CAS on next_block_idx and the fetch_or into
// staged_core_mask are the only cross-thread points.
//
// A gated (early) sync_start drain pre-stages every block behind its doorbell
// (prepare_block_for_dispatch is force-gated for the claimed drain range) and defers the launch
// to the rendezvous: idle cores take a gated RUNNING slot; busy cores take a gated PENDING
// slot (promoted by Case 3.3 as those cores' running tasks FIN). A non-gated (ready) drain
// leaves early_dispatch_state==NONE, so every block launches immediately on an idle running slot and
// the pending pass is skipped. For MIX, a gated block uses SPLIT placement (each core
// independently: idle->running, busy->pending) — safe only because gated.
int32_t
SchedulerContext::drain_stage_cores(PTO2TaskSlotState *slot_state, int32_t block_num, int32_t thread_idx, bool gated) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();
    uint8_t core_mask = slot_state->active_mask.core_mask();
    bool mix_split = gated && shape == PTO2ResourceShape::MIX;
    int32_t running_staged = 0;

    // Stage from this thread's `valid` cores/clusters: CAS-claim a block-index range sized to
    // what this thread can place (against peers claiming concurrently), then publish those
    // blocks onto valid cores. prepare_block_for_dispatch decides each MIX core's slot per-core
    // (idle -> running, busy -> pending when to_pending); a MIX cluster's idle cores are the
    // running-slot cores, counted BEFORE staging mutates the tracker (rendezvous seed).
    auto stage = [&](CoreTracker::BitStates valid, bool to_pending) {
        while (valid.has_value()) {
            int32_t avail = valid.count();
            int32_t start = 0;
            int32_t claim = slot_state->claim_block_range(block_num, avail, start);
            if (claim == 0) return;
#if SIMPLER_DFX
            bool sub_prof = l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES;
            uint64_t prep_t0 = sub_prof ? get_sys_cnt_aicpu() : 0;
#endif
            PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
            int handle_count = 0;
            int32_t claimed[CoreTracker::MAX_CLUSTERS * 3];
            for (int32_t b = 0; b < claim; b++)
                claimed[b] = valid.pop_first();
            bool is_mix = (shape == PTO2ResourceShape::MIX);
            if (claim > 0) prefetch_block_dst(thread_idx, claimed[0], is_mix);
            for (int32_t b = 0; b < claim; b++) {
                if (b + 1 < claim) prefetch_block_dst(thread_idx, claimed[b + 1], is_mix);
                auto core_offset = claimed[b];
                if (shape == PTO2ResourceShape::MIX) {
                    running_staged += tracker.mix_cluster_idle_core_count(core_offset, core_mask);
                }
                handle_count += prepare_block_for_dispatch(
                    thread_idx, core_offset, *slot_state, shape, to_pending, start + b, &handles[handle_count], gated
                );
            }
            wmb();
            uint64_t dispatch_ts = 0;
#if SIMPLER_DFX
            uint64_t pub_t0 = 0;
            if (sub_prof) {
                pub_t0 = get_sys_cnt_aicpu();
                // DrainPrepare bar: cluster scan happened before this lambda, so this covers the
                // build_payload work for `claim` blocks (handle_count subtasks).
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::DrainPrepare, prep_t0, pub_t0,
                    sched_l2_swimlane_[thread_idx].sched_loop_count, static_cast<uint32_t>(handle_count)
                );
            }
            if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
                dispatch_ts = pub_t0 != 0 ? pub_t0 : get_sys_cnt_aicpu();
            }
#endif
            // Accumulate this batch's gated cores into a LOCAL mask and OR it into the shared
            // staged_core_mask ONCE below, instead of a seq_cst fetch_or per subtask — that
            // per-write atomic contends across all drain threads on the same 2 words and was
            // ~half the publish cost. The doorbell-table writes stay per-core (unique cid, no
            // contention).
            uint64_t my_mask[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS] = {0};
            for (int i = 0; i < handle_count; i++) {
                publish_subtask_to_core(handles[i], dispatch_ts);
                if (gated) {
                    int32_t cid = tracker.get_core_id_by_offset(handles[i].core_offset);
                    sched_->early_dispatch_doorbell_table[cid].addr = handles[i].reg_addr;
                    sched_->early_dispatch_doorbell_table[cid].token = handles[i].reg_task_id;
                    my_mask[cid >> 6] |= 1ULL << (cid & 63);
                }
            }
            if (gated) {
                for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                    if (my_mask[w] != 0) {
                        slot_state->payload->staged_core_mask[w].fetch_or(my_mask[w], std::memory_order_seq_cst);
                    }
                }
            }
#if SIMPLER_DFX
            if (sub_prof) {
                // DrainPublish bar: the MMIO write_reg per subtask (+ gated doorbell/mask record).
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::DrainPublish, pub_t0, get_sys_cnt_aicpu(),
                    sched_l2_swimlane_[thread_idx].sched_loop_count, static_cast<uint32_t>(handle_count)
                );
            }
#endif
            sched_->record_published_blocks(*slot_state, claim);
            // AIC/AIV running placement (whole block on idle cores); MIX running cores are
            // counted per-cluster above (mix_cluster_idle_core_count).
            if (gated && shape != PTO2ResourceShape::MIX && !to_pending) running_staged += handle_count;
        }
    };

    if (mix_split) {
        // Gated MIX: to_pending=true opts every BUSY used core into its pending slot while idle
        // used cores take running slots (prepare_block_for_dispatch: to_pending && !is_core_idle).
        stage(tracker.get_mix_split_cluster_offset_states(core_mask), /*to_pending=*/true);
    } else {
        auto idle = (shape == PTO2ResourceShape::MIX) ? tracker.get_mix_running_cluster_offset_states(core_mask) :
                                                        tracker.get_idle_core_offset_states(shape);
        stage(idle, /*to_pending=*/false);  // idle -> running (ready launch + gated pre-stage)
        if (gated) {
            stage(tracker.get_pending_core_offset_states(shape), /*to_pending=*/true);
        }
    }
    return running_staged;
}

// Called by each scheduler thread when drain_state_.sync_start_pending != 0.
//
// Protocol:
//   1. Ack barrier: all threads signal they've stopped dispatch, spin until all acked.
//      If this thread's ack bit gets cleared while waiting, a reset occurred -- return.
//   2. Election + availability: one thread wins the CAS. It checks global resources; if
//      insufficient it resets ack/election so all threads resume completion polling to free
//      cores, then retry. If sufficient it releases parallel staging (stage_go).
//   3. Parallel stage: EVERY thread stages its OWN cores concurrently (CAS-claimed block
//      indices), accumulates its running-slot cores, and marks its stage_done bit.
//   4. Finalize: the elected thread waits for all stage_done bits, seeds the rendezvous
//      (running_slot_count) for a gated drain, and reopens the gate
//      (a release-store the non-elected threads acquire, so the seed is visible before any
//      completion promotes a pending block). Non-elected threads spin until the gate reopens.
void SchedulerContext::handle_drain_mode(int32_t thread_idx, [[maybe_unused]] uint64_t *out_stage_wall_cycles) {
#if SIMPLER_DFX
    bool drain_prof = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && out_stage_wall_cycles != nullptr);
    uint64_t drain_acked_ts = 0;  // set at ack-barrier end; used to measure the stage wall
#endif
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
    bool elected = drain_state_.drain_worker_elected.load(std::memory_order_relaxed) == thread_idx + 1;

    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    // OWNER is acquired before the drain is published and persists through
    // completion, so every staging thread makes the same gate decision even if
    // producer release changes early_dispatch_state during the barrier.
    bool gated = slot_state != nullptr && slot_state->payload != nullptr &&
                 PTO2SchedulerState::owns_early_sync_drain(*slot_state->payload);

    if (elected) {
        if (slot_state == nullptr) {
            // pending_task observed null only when a concurrent drain completion already cleared
            // it. Stale-elected: release the election lock and return. Do NOT clear drain_ack_mask
            // / sync_start_pending -- a *new* drain run may already be accumulating acks.
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        // A gated drain may pre-stage onto pending slots too (idle+pending); the ready drain
        // needs block_num idle cores/clusters.
        int32_t available =
            count_global_available(shape, slot_state->active_mask.core_mask(), /*include_pending=*/gated);
        if (available < block_num) {
            // Insufficient -- reset so all threads resume completion polling to free cores, then retry.
            drain_state_.drain_ack_mask.store(0, std::memory_order_release);
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }
        // Release parallel staging: every thread (this one included) now stages its own cores.
        drain_state_.drain_running_staged.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_go.store(1, std::memory_order_release);
    } else {
        // Non-elected: wait for the go signal, or bail if the elected thread reset (stale /
        // insufficient resources).
        while (drain_state_.drain_stage_go.load(std::memory_order_acquire) == 0) {
            if (is_completed()) return;
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
        if (slot_state == nullptr) return;
        gated = slot_state->payload != nullptr && PTO2SchedulerState::owns_early_sync_drain(*slot_state->payload);
    }

    // Parallel stage this thread's own cores (CAS-claimed block indices), then mark done.
#if SIMPLER_DFX
    if (drain_prof) drain_acked_ts = get_sys_cnt_aicpu();  // pre-stage
#endif
    int32_t my_running = drain_stage_cores(slot_state, block_num, thread_idx, gated);
#if SIMPLER_DFX
    // out param carries the PURE drain_stage_cores wall (build_payload + MMIO publish of
    // this thread's cores), isolating it from availability + stage_go handshake.
    if (drain_prof && drain_acked_ts != 0) *out_stage_wall_cycles = get_sys_cnt_aicpu() - drain_acked_ts;
#endif
    drain_state_.drain_running_staged.fetch_add(my_running, std::memory_order_acq_rel);
    drain_state_.drain_stage_done_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    if (!elected) {
        // Non-elected: staging done; wait for the elected thread to reopen the gate. Exiting via
        // sync_start_pending==0 (release/acquire) or drain_worker_elected==0 both synchronize
        // with the elected's finalize (its release fence sequences the seed before both stores),
        // so the running_slot_count seed is visible before this thread resumes completions.
        while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            if (is_completed()) return;
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        return;
    }

    // Elected: wait for all threads to finish staging, then seed the rendezvous and reopen.
    while ((drain_state_.drain_stage_done_mask.load(std::memory_order_acquire) & all_acked) != all_acked) {
        if (is_completed()) return;
        SPIN_WAIT_HINT();
    }
    if (gated) {
        // Seed the rendezvous with the running-slot cores staged across all threads; pending
        // cores advance it as they promote. maybe_rendezvous_ring (producer release) rings iff
        // this already equals popcount(staged_core_mask) — i.e. no pending spill.
        slot_state->payload->running_slot_count.store(
            static_cast<int16_t>(drain_state_.drain_running_staged.load(std::memory_order_acquire)),
            std::memory_order_seq_cst
        );
    }
    // Clear drain state and reopen the gate FIRST, so the other threads resume immediately.
    // Release fence sequences the seed + tracker mutations before every clear, so any thread
    // that acquire-observes one of them (sync_start_pending==0 / drain_worker_elected==0) sees
    // the seed. `slot_state` is a local holding the fa_fused slot (not drain_state_), so it stays
    // valid for the propagate below even if a new drain reuses pending_task after reopen.
    std::atomic_thread_fence(std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);
    drain_state_.drain_stage_go.store(0, std::memory_order_relaxed);
    drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    drain_state_.sync_start_pending.store(0, std::memory_order_release);

    // Recheck after publishing the drain seed. The producer-side rendezvous check can race
    // ahead of drain completion and fail while running_slot_count is still incomplete. When
    // every block landed directly in a running slot, no pending promotion remains to retry it.
    if (gated) {
        sched_->retry_sync_start_rendezvous_after_drain(*slot_state);
    } else {
        sched_->propagate_dispatch_fanin(*slot_state);
    }
    PTO2SchedulerState::finish_early_sync_drain(*slot_state->payload);
}
