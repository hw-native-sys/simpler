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
#include "common/chip_swimlane_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "runtime_core.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"

// =============================================================================
// Dual-slot state machine helpers
// =============================================================================

namespace {
inline constexpr int32_t DEFERRED_RELEASE_CAP = 256;
inline constexpr int32_t COMPLETION_POLL_BATCH_CAP = 8;

struct CompletionObservation {
    int32_t bit_pos;
    uint32_t reg_value;
    SlotTransition transition;
    bool pending_gated;
};
}  // namespace

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
    ChipTaskSlotState &slot_state, int32_t expected_reg_task_id, [[maybe_unused]] SubtaskSlot subslot,
    [[maybe_unused]] int32_t thread_idx, int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
    ChipTaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
#if SIMPLER_DFX
    ,
    uint64_t dispatch_ts, uint64_t finish_ts
#endif
    ,
    int16_t *deferred_completion_count, bool flush_completion_count
) {
#if SIMPLER_DFX
    auto &chip_swimlane = sched_chip_swimlane_[thread_idx];
#else
    (void)hank;
#endif
    // MPSC fast-path: see a2a3 mirror for the full design narrative. The
    // any_subtask_deferred flag on slot_state discriminates non-deferred
    // tasks (inline complete in parallel on FIN thread) from deferred ones
    // (route through the lock-free AICoreCompletionMailbox).
    AICoreCompletionMailbox *mailbox = rt_ != nullptr ? rt_->aicore_mailbox : nullptr;
    bool defer_completion_to_consumer = false;

    auto flush_before_error_return = [&] {
        if (deferred_completion_count != nullptr && *deferred_completion_count != 0) {
            // The failed physical completion remains uncounted, so these valid
            // earlier completions cannot retire the task.
            bool task_complete = sched_->on_subtask_complete(slot_state, *deferred_completion_count);
            always_assert(!task_complete);
            *deferred_completion_count = 0;
        }
    };

    if (slot_state.payload != nullptr) {
        volatile DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][expected_reg_task_id & 1];
        // A5 AICPU reads are coherent with AICore-published slab writes.
        // The polling path's rmb already orders COND before these reads.
        int32_t slab_err = deferred_slab->error_code;
        if (slab_err != SIMPLER_ERROR_NONE) {
            int32_t expected = SIMPLER_ERROR_NONE;
            sched_->sm_header->sched_error_code.compare_exchange_strong(
                expected, slab_err, std::memory_order_acq_rel, std::memory_order_acquire
            );
            completed_.store(true, std::memory_order_release);
            flush_before_error_return();
            return;
        }

        uint32_t cond_count = deferred_slab->count;
        if (cond_count > MAX_COMPLETIONS_PER_TASK) {
            int32_t expected = SIMPLER_ERROR_NONE;
            sched_->sm_header->sched_error_code.compare_exchange_strong(
                expected, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED, std::memory_order_acq_rel, std::memory_order_acquire
            );
            completed_.store(true, std::memory_order_release);
            flush_before_error_return();
            return;
        }

        if (cond_count > 0) {
            cache_invalidate_range(
                reinterpret_cast<const void *>(const_cast<DeferredCompletionEntry *>(deferred_slab->entries)),
                cond_count * sizeof(DeferredCompletionEntry)
            );
            slot_state.mark_any_subtask_deferred();

            const TaskId token = slot_state.task->task_id;
            for (uint32_t i = 0; i < cond_count; ++i) {
                volatile DeferredCompletionEntry *e = &deferred_slab->entries[i];
                while (!mailbox->try_push_condition(
                    token, e->addr, e->backend_cookie, e->expected_value, e->engine, e->completion_type
                )) {
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

    int16_t completion_count = 1;
    if (deferred_completion_count != nullptr) {
        ++*deferred_completion_count;
        completion_count = flush_completion_count ? *deferred_completion_count : 0;
        if (flush_completion_count) *deferred_completion_count = 0;
    }
    bool task_complete = completion_count != 0 && sched_->on_subtask_complete(slot_state, completion_count);

#if SIMPLER_DFX
    // A multi-block task can retire several subtasks before its final block
    // arrives. Count those non-final FINs so the scheduler lane does not hide
    // the serial-harvest tail between two logical task completions.
    if (!task_complete && chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES) {
        chip_swimlane.phase_subretire_count++;
    }
#endif

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
            dump_args_for_task<SUBTASK_SLOT_COUNT>(
                thread_idx, *slot_state.task, *slot_state.payload, slot_state.active_mask,
                ArgsDumpStage::AFTER_COMPLETION, [this](int32_t func_id) {
                    return get_function_bin_addr(func_id);
                }
            );
        }
#endif
#if SIMPLER_DFX
        uint64_t resolve_t0 = (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
#if SIMPLER_SCHED_PROFILING
        // SCHED_PROFILING variant takes thread_idx for its per-thread atomic
        // counter side-effects (g_sched_*_atomic_count[thread_idx], consumed
        // by the otc_* log lines). The returned fanout_edges feeds Resolve.
        [[maybe_unused]] uint32_t consumers_resolved = sched_->on_task_complete(slot_state, thread_idx).fanout_edges;
#else
        [[maybe_unused]] uint32_t consumers_resolved = sched_->on_task_complete(slot_state);
#endif
#if SIMPLER_DFX
        if (resolve_t0 != 0) {
            // The completion call above is the Resolve work. Keep the bar
            // narrow and filter the short consumer walks that add no signal.
            uint64_t resolve_t1 = get_sys_cnt_aicpu();
            constexpr uint64_t RESOLVE_EMIT_MIN_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ / 1'000'000;
            if (resolve_t1 - resolve_t0 >= RESOLVE_EMIT_MIN_CYCLES) {
                chip_swimlane_aicpu_record_sched_phase(
                    thread_idx, SchedPhaseKind::Resolve, resolve_t0, resolve_t1, chip_swimlane.sched_loop_count,
                    consumers_resolved
                );
            }
        }
        chip_swimlane.phase_complete_count++;
#endif
        // At capacity, elide deferred releases (including the overflowing slot)
        // when orchestration is done; otherwise drain then push.
        bool release_elided = false;
        if (deferred_release_count >= DEFERRED_RELEASE_CAP) {
            release_elided = orchestrator_done_.load(std::memory_order_acquire);
            drain_or_elide_deferred_releases(
                sched_, deferred_release_slot_states, deferred_release_count, release_elided
#if SIMPLER_SCHED_PROFILING
                ,
                thread_idx
#endif
            );
        }
        if (!release_elided) {
            deferred_release_slot_states[deferred_release_count++] = &slot_state;
        }
        completed_this_turn++;
    }

#if SIMPLER_DFX
    // Level gate: at TASK_TIMING (level=1) the AICore record alone carries
    // {start, end, task_token}, host resolves func_id/core_type from
    // dep_gen / per-core mapping, and AICPU has nothing to write. Only at
    // SCHEDULE_TIMING (level=2) and above does AICPU contribute dispatch/finish
    // timestamps via complete_task.
    if (chip_swimlane.chip_swimlane_enabled && chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING) {
#if SIMPLER_SCHED_PROFILING
        uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif

        if (chip_swimlane_aicpu_complete_task(
                core_id, thread_idx, static_cast<uint32_t>(expected_reg_task_id), dispatch_ts, finish_ts
            ) != 0) {
            LOG_ERROR(
                "Core %d: chip_swimlane_aicpu_complete_task failed for task 0x%" PRIx64, core_id,
                TaskId::to_uint64(slot_state.task->task_id)
            );
        }
#if SIMPLER_SCHED_PROFILING
        chip_swimlane.sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
    }
#endif

#if SIMPLER_DFX
    if (is_pmu_enabled()) {
        // Two keys, two roles: the slot is found by the 32-bit register token
        // AICore staged its counters under, and the identity that lands in the
        // PmuRecord is the task's own. The token is not derivable from the
        // handle, which is why both are passed.
        pmu_aicpu_complete_record(
            core_id, thread_idx, static_cast<uint32_t>(expected_reg_task_id), slot_state.task->task_id,
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
    bool &made_progress, ChipTaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
) {
    if (core_trackers_[thread_idx].get_groupable_running_cores().has_value()) {
        poll_running_cores<true>(
            thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress, deferred_release_slot_states,
            deferred_release_count
        );
    } else {
        poll_running_cores<false>(
            thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress, deferred_release_slot_states,
            deferred_release_count
        );
    }
}

template <bool Batched>
void SchedulerContext::poll_running_cores(
    int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
    bool &made_progress, ChipTaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
) {
#if SIMPLER_SCHED_PROFILING
    auto &chip_swimlane = sched_chip_swimlane_[thread_idx];
#endif
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto running_core_states = tracker.get_all_running_cores();
    constexpr int32_t capacity = Batched ? COMPLETION_POLL_BATCH_CAP : 1;
    CompletionObservation observations[capacity];
    while (running_core_states.has_value()) {
        int32_t observation_count = 0;
        while (observation_count < capacity && running_core_states.has_value()) {
            int32_t bit_pos = running_core_states.pop_first();
            CoreExecState &core = core_exec_states_[tracker.get_core_id_by_offset(bit_pos)];
            ChipTaskSlotState *running = core.running_slot_state;
            if (running != nullptr && running->payload != nullptr &&
                running->payload->early_dispatch_state.load(std::memory_order_relaxed) == EARLY_DISPATCH_STAGING) {
                continue;
            }
            auto &observation = observations[observation_count++];
            observation.bit_pos = bit_pos;
            // Each simulator COND load is an acquire; hardware reads are volatile.
            // AICore-published memory is read only after the batch barrier.
            observation.reg_value = reg_load_acquire(core.cond_ptr);
        }
        if (observation_count == 0) continue;

        // One barrier acquires every captured COND, including ACK-only and
        // unmatched values, before any slab read or buffer reuse/publication.
        rmb();
#if SIMPLER_DFX
        // This is the batch-acquisition time, before classification or callbacks.
        // Later rows must not include earlier rows' completion work in end->finish.
        uint64_t finish_ts = chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING ? get_sys_cnt_aicpu() : 0;
#endif
        constexpr int32_t MAX_BATCH_COMPLETIONS = 2 * capacity;
        constexpr uint8_t NO_GROUP = UINT8_MAX;
        static_assert(MAX_BATCH_COMPLETIONS < NO_GROUP);
        [[maybe_unused]] int16_t completion_counts[MAX_BATCH_COMPLETIONS];
        [[maybe_unused]] uint8_t completion_groups[MAX_BATCH_COMPLETIONS];
        [[maybe_unused]] uint8_t last_group_event[MAX_BATCH_COMPLETIONS];
        [[maybe_unused]] uint8_t *touched[MAX_BATCH_COMPLETIONS];
        [[maybe_unused]] int32_t group_count = 0;
        [[maybe_unused]] int32_t completion_event_count = 0;
        for (int32_t i = 0; i < observation_count; ++i) {
            auto &observation = observations[i];
            CoreExecState &core = core_exec_states_[tracker.get_core_id_by_offset(observation.bit_pos)];
            int32_t reg_task_id = EXTRACT_TASK_ID(observation.reg_value);
            int32_t reg_state = EXTRACT_TASK_STATE(observation.reg_value);
#if SIMPLER_SCHED_PROFILING
            if (chip_swimlane.chip_swimlane_enabled) {
                chip_swimlane.complete_probe_count++;
            }
#endif
            uint8_t pending_ss =
                (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr) ?
                    core.pending_slot_state->payload->early_dispatch_state.load(std::memory_order_relaxed) :
                    static_cast<uint8_t>(EARLY_DISPATCH_NONE);
            observation.pending_gated =
                (core.pending_slot_state != nullptr && core.pending_slot_state->payload != nullptr &&
                 (pending_ss == EARLY_DISPATCH_STAGING || (pending_ss == EARLY_DISPATCH_DISPATCHED &&
                                                           core.pending_slot_state->task_attrs.requires_sync_start())));
            observation.transition = decide_slot_transition(
                reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id, observation.pending_gated
            );
            const SlotTransition &t = observation.transition;
            if (!t.matched) continue;
#if SIMPLER_SCHED_PROFILING
            if (chip_swimlane.chip_swimlane_enabled && (t.running_done || t.pending_done)) {
                chip_swimlane.complete_hit_count++;
            }
#endif
            // Outstanding physical completions keep these task pointers alive.
            // Fold every tagged finish before any callback can retire a task;
            // selective timing also runs with DFX=0 or chip swimlane disabled.
            if (t.pending_done && core.pending_slot_state->task_attrs.is_timed()) {
                aicpu_task_timing_finish(core.pending_slot_state->task_attrs.timing_slot(), thread_idx);
            }
            if (t.running_done && core.running_slot_state->task_attrs.is_timed()) {
                aicpu_task_timing_finish(core.running_slot_state->task_attrs.timing_slot(), thread_idx);
            }
            if constexpr (Batched) {
                auto append_completion = [&](ChipTaskSlotState *slot, bool groupable) {
                    int32_t event = completion_event_count++;
                    if (!groupable) {
                        completion_groups[event] = NO_GROUP;
                        return;
                    }
                    uint8_t &entry = completion_scratch_[thread_idx].cell(*slot);
                    if (entry == 0) {
                        int32_t group = group_count++;
                        entry = static_cast<uint8_t>(group + 1);
                        touched[group] = &entry;
                        completion_counts[group] = 0;
                    }
                    uint8_t group = entry - 1;
                    completion_groups[event] = group;
                    last_group_event[group] = static_cast<uint8_t>(event);
                };
                if (t.pending_done) {
                    append_completion(core.pending_slot_state, tracker.is_pending_groupable(observation.bit_pos));
                }
                if (t.running_done) {
                    append_completion(core.running_slot_state, tracker.is_running_groupable(observation.bit_pos));
                }
            }
        }

        // The lookup is needed only for classification. Clear only touched cells
        // before callbacks can retire/reuse a slot; application uses dense groups.
        if constexpr (Batched) {
            for (int32_t group = 0; group < group_count; ++group)
                *touched[group] = 0;
        }

        // A later local FIN pins every deferred count until its group's last
        // event. Before that event, no owner can retire the task.
        [[maybe_unused]] int32_t next_completion_event = 0;
        // Only this owner changes its core slots. Earlier callbacks may release
        // an installed gate, so retain the acquired transition and occupancy
        // decision instead of reclassifying against a partly applied batch.
        for (int32_t i = 0; i < observation_count; ++i) {
            const auto &observation = observations[i];
            const SlotTransition &t = observation.transition;
            if (!t.matched) continue;
            int32_t bit_pos = observation.bit_pos;
            int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
            CoreExecState &core = core_exec_states_[core_id];
#if SIMPLER_DFX
            int32_t reg_task_id = EXTRACT_TASK_ID(observation.reg_value);
            // All observations in this batch precede ACK-buffer publication and
            // completion work, including selective task timing with swimlane off.
            if (chip_swimlane_level_ != ChipSwimlaneLevel::DISABLED) {
                chip_swimlane_aicpu_on_aicore_ack(core_id, thread_idx, static_cast<uint32_t>(reg_task_id));
            }
#endif

            // --- Apply phase: execute actions based on transition ---

            // 1. Complete finished tasks (capture pointers before modifying core state)
            if (t.pending_done) {
                int16_t *count = nullptr;
                bool flush = true;
                if constexpr (Batched) {
                    int32_t event = next_completion_event++;
                    uint8_t group = completion_groups[event];
                    if (group != NO_GROUP) {
                        count = &completion_counts[group];
                        flush = last_group_event[group] == event;
                    }
                }
                complete_slot_task(
                    *core.pending_slot_state, core.pending_reg_task_id, core.pending_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count
#if SIMPLER_DFX
                    ,
                    core.pending_dispatch_timestamp, finish_ts
#endif
                    ,
                    count, flush
                );
                cur_thread_completed++;
            }
            if (t.running_done) {
                int16_t *count = nullptr;
                bool flush = true;
                if constexpr (Batched) {
                    int32_t event = next_completion_event++;
                    uint8_t group = completion_groups[event];
                    if (group != NO_GROUP) {
                        count = &completion_counts[group];
                        flush = last_group_event[group] == event;
                    }
                }
                complete_slot_task(
                    *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count
#if SIMPLER_DFX
                    ,
                    core.running_dispatch_timestamp, finish_ts
#endif
                    ,
                    count, flush
                );
                cur_thread_completed++;
            }

            // 2. Update slot data
            if (t.running_freed) {
                if (core.pending_slot_state != nullptr && !t.pending_done) {
                    ChipTaskSlotState *promoted = core.pending_slot_state;
                    bool sync_start_promote = observation.pending_gated && promoted->task_attrs.requires_sync_start();
                    promote_pending_to_running(core);
                    if constexpr (Batched) tracker.promote_pending_groupable(bit_pos);
                    if (sync_start_promote) {
                        promoted->payload->running_slot_count.fetch_add(1, std::memory_order_seq_cst);
                        if (sched_->try_launch_sync_start_cohort(*promoted)) {
                            sched_->propagate_dispatch_fanin(*promoted);
                        }
                    }
                } else {
                    clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                    if constexpr (Batched) tracker.clear_completion_slots(bit_pos);
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
}

// =============================================================================
// sync_start drain protocol
// =============================================================================

// Take ownership of slot_state and signal all threads to enter drain mode.
// Returns true if this thread won the CAS and owns the drain slot.
// Returns false if another thread already holds drain; caller must re-push slot_state.
//
// Two-phase protocol: CAS 0 -> -1 (sentinel) to claim ownership, store task and
// reset staging state, then release-store block_num. Other threads acquire-load
// sync_start_pending; seeing block_num > 0 ensures all relaxed stores are visible.
bool SchedulerContext::enter_drain_mode(ChipTaskSlotState *slot_state, int32_t block_num) {
    int32_t expected = 0;
    if (!drain_state_.sync_start_pending.compare_exchange_strong(
            expected, -1, std::memory_order_acquire, std::memory_order_relaxed
        )) {
        return false;  // Another thread already holds the drain slot.
    }
    // Advance the attempt before publishing the new task so delayed participants
    // from the previous round cannot satisfy this round's ack tree.
    uint64_t next_attempt = sync_start_drain_next_attempt(drain_state_.drain_attempt.load(std::memory_order_relaxed));
    drain_state_.drain_attempt.store(next_attempt, std::memory_order_release);
    drain_state_.pending_task.store(slot_state, std::memory_order_release);
    drain_state_.drain_stage_go.store(0, std::memory_order_relaxed);
    drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
    drain_state_.drain_running_staged.store(0, std::memory_order_relaxed);
    // Release store: all stores above are now visible to any thread that
    // acquire-loads sync_start_pending and sees block_num > 0.
    drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
    return true;
}

// Count total available resources across all scheduler threads for a given shape.
int32_t SchedulerContext::count_global_available(ResourceShape shape, LaneMask lanes, bool include_pending) {
    int32_t total = 0;
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        total += core_trackers_[t].count_available_blocks(shape, lanes, include_pending);
    }
    return total;
}

// Stage one scheduler thread's share of a sync_start cohort. The global drain
// calls this concurrently on every tracker; the Case-A fast path calls it once
// after proving this tracker can hold the whole cohort. staged_blocks is logical
// cohort progress, while running_cores seeds the core-count rendezvous (MIX may
// contribute multiple running cores per logical block).
SchedulerContext::SyncStartStageResult SchedulerContext::stage_sync_start_cores(
    ChipTaskSlotState *slot_state, int32_t block_num, int32_t thread_idx, bool gated,
    [[maybe_unused]] bool record_drain_phases
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    ResourceShape shape = slot_state->active_mask.to_shape();
    LaneMask lanes = slot_state->active_mask;
    bool mix_split = gated && shape == ResourceShape::MIX;
    SyncStartStageResult result;

    auto stage = [&](CoreTracker::BitStates valid, bool to_pending) {
        while (valid.has_value()) {
            int32_t avail = valid.count();
            int32_t start = 0;
            int32_t claim = slot_state->claim_block_range(block_num, avail, start);
            if (claim == 0) return;
            result.staged_blocks += claim;
#if SIMPLER_DFX
            bool sub_prof = record_drain_phases && chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES;
            uint64_t prep_t0 = sub_prof ? get_sys_cnt_aicpu() : 0;
#endif
            PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
            int handle_count = 0;
            int32_t claimed[CoreTracker::MAX_CLUSTERS * 3];
            for (int32_t b = 0; b < claim; b++)
                claimed[b] = valid.pop_first();
            for (int32_t b = 0; b < claim; b++) {
                auto core_offset = claimed[b];
                if (shape == ResourceShape::MIX) {
                    result.running_cores += tracker.mix_cluster_idle_core_count(core_offset, lanes);
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
                chip_swimlane_aicpu_record_sched_phase(
                    thread_idx, SchedPhaseKind::DrainPrepare, prep_t0, pub_t0,
                    sched_chip_swimlane_[thread_idx].sched_loop_count, static_cast<uint32_t>(handle_count)
                );
            }
            if (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHEDULE_TIMING) {
                dispatch_ts = pub_t0 != 0 ? pub_t0 : get_sys_cnt_aicpu();
            }
#endif
            uint64_t my_mask[EARLY_DISPATCH_CORE_MASK_WORDS] = {0};
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
                for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                    if (my_mask[w] != 0) {
                        slot_state->payload->staged_core_mask[w].fetch_or(my_mask[w], std::memory_order_seq_cst);
                    }
                }
            }
#if SIMPLER_DFX
            if (sub_prof) {
                chip_swimlane_aicpu_record_sched_phase(
                    thread_idx, SchedPhaseKind::DrainPublish, pub_t0, get_sys_cnt_aicpu(),
                    sched_chip_swimlane_[thread_idx].sched_loop_count, static_cast<uint32_t>(handle_count)
                );
            }
#endif
            sched_->record_published_blocks(*slot_state, claim);
            if (gated && shape != ResourceShape::MIX && !to_pending) result.running_cores += handle_count;
        }
    };

    if (mix_split) {
        stage(tracker.get_mix_split_cluster_offset_states(lanes), /*to_pending=*/true);
    } else {
        auto idle = (shape == ResourceShape::MIX) ? tracker.get_mix_running_cluster_offset_states(lanes) :
                                                    tracker.get_idle_core_offset_states(shape);
        stage(idle, /*to_pending=*/false);
        if (gated) {
            stage(tracker.get_pending_core_offset_states(shape), /*to_pending=*/true);
        }
    }
    return result;
}

void SchedulerContext::handle_drain_mode(
    int32_t thread_idx, [[maybe_unused]] uint64_t *out_stage_wall_cycles, [[maybe_unused]] int32_t *out_staged_blocks
) {
#if SIMPLER_DFX
    bool drain_prof = (chip_swimlane_level_ >= ChipSwimlaneLevel::SCHED_PHASES && out_stage_wall_cycles != nullptr);
    uint64_t drain_acked_ts = 0;
#endif
#if SIMPLER_DFX
    if (out_staged_blocks != nullptr) *out_staged_blocks = 0;
#endif
    int32_t block_num;
    do {
        if (is_completed()) return;
        block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
    } while (block_num < 0);
    if (block_num == 0) return;

    uint32_t all_acked = (1u << active_sched_threads_) - 1;
    uint64_t drain_attempt = drain_state_.drain_attempt.load(std::memory_order_acquire);
    uint64_t subtree_token = sync_start_drain_ack_subtree_token(drain_attempt);

    // O(log N) reduction with O(1) fan-out per thread and no shared RMW.
    for (int32_t child : {thread_idx * 2 + 1, thread_idx * 2 + 2}) {
        if (child >= active_sched_threads_) continue;
        while (drain_ack_tokens_[child].load(std::memory_order_acquire) != subtree_token) {
            if (is_completed()) return;
            if (drain_state_.drain_attempt.load(std::memory_order_acquire) != drain_attempt) return;
            SPIN_WAIT_HINT();
        }
    }
    drain_ack_tokens_[thread_idx].store(subtree_token, std::memory_order_release);
    if (thread_idx != 0) {
        while (drain_ack_tokens_[0].load(std::memory_order_acquire) != subtree_token) {
            if (is_completed()) return;
            if (drain_state_.drain_attempt.load(std::memory_order_acquire) != drain_attempt) return;
            SPIN_WAIT_HINT();
        }
    }
    bool coordinator = thread_idx == 0;

    ChipTaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
    bool gated = slot_state->payload != nullptr && SchedulerState::owns_early_sync_drain(*slot_state->payload);

    if (coordinator) {
        ResourceShape shape = slot_state->active_mask.to_shape();
        int32_t available = count_global_available(shape, slot_state->active_mask, /*include_pending=*/gated);
        if (available < block_num) {
            drain_state_.drain_attempt.store(sync_start_drain_next_attempt(drain_attempt), std::memory_order_release);
            return;
        }
        drain_state_.drain_running_staged.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_done_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_stage_go.store(1, std::memory_order_release);
    } else {
        while (drain_state_.drain_stage_go.load(std::memory_order_acquire) == 0) {
            if (is_completed()) return;
            if (drain_state_.drain_attempt.load(std::memory_order_acquire) != drain_attempt) return;
            SPIN_WAIT_HINT();
        }
    }

#if SIMPLER_DFX
    if (drain_prof) drain_acked_ts = get_sys_cnt_aicpu();
#endif
    SyncStartStageResult staged =
        stage_sync_start_cores(slot_state, block_num, thread_idx, gated, /*record_drain_phases=*/true);
#if SIMPLER_DFX
    if (drain_prof && drain_acked_ts != 0) *out_stage_wall_cycles = get_sys_cnt_aicpu() - drain_acked_ts;
    if (out_staged_blocks != nullptr) *out_staged_blocks = staged.staged_blocks;
#endif
    drain_state_.drain_running_staged.fetch_add(staged.running_cores, std::memory_order_acq_rel);
    drain_state_.drain_stage_done_mask.fetch_or(1u << thread_idx, std::memory_order_release);

    if (!coordinator) {
        while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            if (is_completed()) return;
            if (drain_state_.drain_attempt.load(std::memory_order_acquire) != drain_attempt) return;
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
    drain_state_.sync_start_pending.store(0, std::memory_order_release);

    if (gated) {
        sched_->retry_sync_start_rendezvous_after_staging(*slot_state);
    } else {
        sched_->propagate_dispatch_fanin(*slot_state);
    }
    SchedulerState::finish_early_sync_drain(*slot_state->payload);
}
