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
#include "aicpu/device_phase_aicpu.h"
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
                // Case 3.3: running FIN, pending is early-dispatch gated. Complete
                // running now and promote; waiting for a gated ACK would deadlock.
                t.matched = true;
                t.running_done = true;
                t.running_freed = true;
            }
            // Case 3.1: running FIN, NON-gated pending exists -> skip (transient).
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
    // MPSC fast-path: see a2a3 mirror for the full design narrative. The
    // any_subtask_deferred flag on slot_state discriminates non-deferred
    // tasks (inline complete in parallel on FIN thread) from deferred ones
    // (route through the lock-free AICoreCompletionMailbox).
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
            slot_state.mark_any_subtask_deferred();

            const PTO2TaskId token = slot_state.task->task_id;
            for (uint32_t i = 0; i < cond_count; ++i) {
                volatile DeferredCompletionEntry *e = &deferred_slab->entries[i];
                while (!mailbox->try_push_condition(
                    token, e->addr, e->backend_cookie, e->expected_value, e->engine, e->completion_type
                )) {
                    sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
                    SPIN_WAIT_HINT();
                }
            }
        }
    }

    bool task_complete = sched_->on_subtask_complete(slot_state);

    if (task_complete && slot_state.payload != nullptr && slot_state.has_any_subtask_deferred()) {
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
#if SIMPLER_SCHED_PROFILING
        // SCHED_PROFILING variant takes thread_idx for its per-thread atomic
        // counter side-effects (g_sched_*_atomic_count[thread_idx], consumed
        // by the otc_* log lines). Its return value is unused.
        (void)sched_->on_task_complete(slot_state, thread_idx);
#else
        sched_->on_task_complete(slot_state);
#endif
#if SIMPLER_DFX
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
    // timestamps via complete_task.
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
#endif

#if SIMPLER_DFX
    if (is_pmu_enabled()) {
        // Slot key must be the 32-bit register token AICore wrote into
        // dual_issue_slots[task_id & 1].task_id (= DATA_MAIN_BASE value).
        // task_id.raw is the full PTO2 (ring_id<<32|local_id) encoding —
        // matching on that would never hit. Pass the PTO2 id separately
        // for the PmuRecord.
        pmu_aicpu_complete_record(
            core_id, thread_idx, static_cast<uint32_t>(expected_reg_task_id), slot_state.task->task_id.raw,
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

        // Skip gated early-dispatch cores still waiting for their doorbell.
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
        // reg_load_acquire makes this an atomic acquire under __CPU_SIM so it
        // pairs with the AICore's release store of the FIN (without it the sim
        // poll races the FIN publish and can miss it); on hardware it is the
        // same plain volatile load the bare deref used to be.
        uint64_t reg_val = static_cast<uint64_t>(reg_load_acquire(core.cond_ptr));
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

        // Phase 1+: gated == not yet launched (STAGING, or sync_start DISPATCHED rendezvous).
        uint8_t pending_ss =
            (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr) ?
                core.pending_slot_state->payload->early_dispatch_state.load(std::memory_order_relaxed) :
                static_cast<uint8_t>(PTO2_EARLY_DISPATCH_NONE);
        bool pending_gated =
            (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr &&
             (pending_ss == PTO2_EARLY_DISPATCH_STAGING ||
              (pending_ss == PTO2_EARLY_DISPATCH_DISPATCHED &&
               core.pending_slot_state->task_attrs.requires_sync_start())));
        SlotTransition t = decide_slot_transition(
            reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id, pending_gated
        );
        if (!t.matched) continue;

#if SIMPLER_SCHED_PROFILING
        if (l2_swimlane.l2_swimlane_enabled && (t.running_done || t.pending_done)) {
            l2_swimlane.complete_hit_count++;
        }
#endif

#if SIMPLER_DFX
        // Capture finish_ts at the FIN observation point — right after rmb()
        // pinned cacheable AICore reads downstream of the register load, and
        // BEFORE any fanin / deferred-release work. Anything later would
        // charge AICPU completion-processing cost to (end → finish).
        uint64_t finish_ts = 0;
        if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING && (t.pending_done || t.running_done)) {
            finish_ts = get_sys_cnt_aicpu();
        }
#endif

        // --- Apply phase: execute actions based on transition ---

        // 1. Complete finished tasks (capture pointers before modifying core state)
        if (t.pending_done) {
            // Task-timing finish: latest FIN observation for a tagged task, folded
            // as max. Sampled after the rmb above and before complete_slot_task runs
            // fanin / deferred-completion (which may also clear pending_slot_state),
            // matching L2's finish_time point. Independent of L2 swimlane level, so
            // it works in SIMPLER_DFX=0 builds; untagged tasks pay only the compare.
            if (core.pending_slot_state->task_attrs.is_timed()) {
                aicpu_task_timing_finish(core.pending_slot_state->task_attrs.timing_slot(), thread_idx);
            }
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
            if (core.running_slot_state->task_attrs.is_timed()) {
                aicpu_task_timing_finish(core.running_slot_state->task_attrs.timing_slot(), thread_idx);
            }
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
                PTO2TaskSlotState *promoted = core.pending_slot_state;
                bool sync_start_promote = pending_gated && promoted->task_attrs.requires_sync_start();
                promote_pending_to_running(core);
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
    // We own the drain slot.  Store the task and reset election flag before making it visible.
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
int32_t SchedulerContext::count_global_available(PTO2ResourceShape shape, uint8_t core_mask, bool include_pending) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        if (shape == PTO2ResourceShape::MIX) {
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

int32_t
SchedulerContext::drain_stage_cores(PTO2TaskSlotState *slot_state, int32_t block_num, int32_t thread_idx, bool gated) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    PTO2ResourceShape shape = slot_state->active_mask.to_shape();
    uint8_t core_mask = slot_state->active_mask.core_mask();
    bool mix_split = gated && shape == PTO2ResourceShape::MIX;
    int32_t running_staged = 0;

    auto stage = [&](CoreTracker::BitStates valid, bool to_pending) {
        while (valid.has_value()) {
            int32_t avail = valid.count();
            int32_t start = 0;
            int32_t claim = slot_state->claim_block_range(block_num, avail, start);
            if (claim == 0) return;
            PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
            int handle_count = 0;
            int32_t claimed[CoreTracker::MAX_CLUSTERS * 3];
            for (int32_t b = 0; b < claim; b++)
                claimed[b] = valid.pop_first();
            for (int32_t b = 0; b < claim; b++) {
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
            if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
                dispatch_ts = get_sys_cnt_aicpu();
            }
#endif
            uint64_t my_mask[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS] = {0};
            for (int i = 0; i < handle_count; i++) {
                publish_subtask_to_core(handles[i], dispatch_ts, thread_idx);
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
            sched_->record_published_blocks(*slot_state, claim);
            if (gated && shape != PTO2ResourceShape::MIX && !to_pending) running_staged += handle_count;
        }
    };

    if (mix_split) {
        stage(tracker.get_mix_split_cluster_offset_states(core_mask), /*to_pending=*/true);
    } else {
        auto idle = (shape == PTO2ResourceShape::MIX) ? tracker.get_mix_running_cluster_offset_states(core_mask) :
                                                        tracker.get_idle_core_offset_states(shape);
        stage(idle, /*to_pending=*/false);
        if (gated) {
            stage(tracker.get_pending_core_offset_states(shape), /*to_pending=*/true);
        }
    }
    return running_staged;
}

void SchedulerContext::handle_drain_mode(int32_t thread_idx, [[maybe_unused]] uint64_t *out_stage_wall_cycles) {
#if SIMPLER_DFX
    bool drain_prof = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && out_stage_wall_cycles != nullptr);
    uint64_t drain_acked_ts = 0;
#endif
    int32_t block_num;
    do {
        if (is_completed()) return;
        block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
    } while (block_num < 0);
    if (block_num == 0) return;

    uint32_t all_acked = (1u << active_sched_threads_) - 1;

    drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    while (true) {
        if (is_completed()) return;
        uint32_t ack = drain_state_.drain_ack_mask.load(std::memory_order_acquire);
        if ((ack & all_acked) == all_acked) break;
        if ((ack & (1u << thread_idx)) == 0) return;
        SPIN_WAIT_HINT();
    }

    int32_t expected = 0;
    drain_state_.drain_worker_elected.compare_exchange_strong(
        expected, thread_idx + 1, std::memory_order_acquire, std::memory_order_relaxed
    );
    bool elected = drain_state_.drain_worker_elected.load(std::memory_order_relaxed) == thread_idx + 1;

    PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    bool gated = slot_state != nullptr && slot_state->payload != nullptr &&
                 PTO2SchedulerState::owns_early_sync_drain(*slot_state->payload);

    if (elected) {
        if (slot_state == nullptr) {
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        int32_t available =
            count_global_available(shape, slot_state->active_mask.core_mask(), /*include_pending=*/gated);
        if (available < block_num) {
            drain_state_.drain_ack_mask.store(0, std::memory_order_release);
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }
        drain_state_.drain_running_staged.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_go.store(1, std::memory_order_release);
    } else {
        while (drain_state_.drain_stage_go.load(std::memory_order_acquire) == 0) {
            if (is_completed()) return;
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
        if (slot_state == nullptr) return;
        gated = slot_state->payload != nullptr && PTO2SchedulerState::owns_early_sync_drain(*slot_state->payload);
    }

#if SIMPLER_DFX
    if (drain_prof) drain_acked_ts = get_sys_cnt_aicpu();
#endif
    int32_t my_running = drain_stage_cores(slot_state, block_num, thread_idx, gated);
#if SIMPLER_DFX
    if (drain_prof && drain_acked_ts != 0) *out_stage_wall_cycles = get_sys_cnt_aicpu() - drain_acked_ts;
#endif
    drain_state_.drain_running_staged.fetch_add(my_running, std::memory_order_acq_rel);
    drain_state_.drain_stage_done_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    if (!elected) {
        while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            if (is_completed()) return;
            if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
            SPIN_WAIT_HINT();
        }
        return;
    }

    while ((drain_state_.drain_stage_done_mask.load(std::memory_order_acquire) & all_acked) != all_acked) {
        if (is_completed()) return;
        SPIN_WAIT_HINT();
    }
    if (gated) {
        slot_state->payload->running_slot_count.store(
            static_cast<int16_t>(drain_state_.drain_running_staged.load(std::memory_order_acquire)),
            std::memory_order_seq_cst
        );
    }
    std::atomic_thread_fence(std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);
    drain_state_.drain_stage_go.store(0, std::memory_order_relaxed);
    drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
    drain_state_.sync_start_pending.store(0, std::memory_order_release);

    if (gated) {
        sched_->retry_sync_start_rendezvous_after_drain(*slot_state);
    } else {
        sched_->propagate_dispatch_fanin(*slot_state);
    }
    PTO2SchedulerState::finish_early_sync_drain(*slot_state->payload);
}
