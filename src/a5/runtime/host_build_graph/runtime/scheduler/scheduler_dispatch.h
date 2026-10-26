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

#pragma once

// AICore-resident dispatch library; scheduler_dispatch.cpp is the independent AICPU scheduler implementation.

#include "scheduler_completion.h"

inline __aicore__ bool scheduler_service_gm_mix(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, SchedulerWakeStats *wake_stats, SchedulerReadyStats *ready_stats,
    SchedulerCompletionStats *completion_stats, bool *progress
) {
    *progress = false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    bool closed = false;
    if (!scheduler_gm_mix_close(graph, base, local, run, &closed)) return false;
    if (closed) {
        *progress = true;
        return true;
    }
    if (local->is_mix_coordinator()) {
        if (group.phase == SchedulerGmMixPhase::RELEASED) {
            const auto retired = scheduler_gm_mix_retire(graph, base, local, run);
            if (retired == SchedulerGmMixResult::ERROR) return false;
            if (retired == SchedulerGmMixResult::TASK_LAST &&
                !scheduler_complete_ordered_task(
                    graph, base, local, run, admission.current_task_ref, profiling_level, wake_stats, ready_stats,
                    completion_stats
                ))
                return false;
            if (retired == SchedulerGmMixResult::PROGRESS || retired == SchedulerGmMixResult::TASK_LAST) {
                *progress = true;
                return true;
            }
        }
        if (group.phase == SchedulerGmMixPhase::IDLE) {
            const auto claimed = scheduler_gm_mix_claim(graph, base, local, run);
            if (claimed == SchedulerBlockClaimResult::ERROR) return false;
            if (claimed != SchedulerBlockClaimResult::CLAIMED) return true;
            *progress = true;
        }
        if (group.phase == SchedulerGmMixPhase::CLAIMED) {
            if (!scheduler_gm_mix_publish(graph, base, local, run)) return false;
            *progress = true;
        }
    } else if (group.phase == SchedulerGmMixPhase::IDLE || group.phase == SchedulerGmMixPhase::RELEASED) {
        bool observed = false;
        if (!scheduler_gm_mix_observe(graph, base, local, run, &observed)) return false;
        *progress = observed;
        if (scheduler_task_ref_is_null(admission.current_task_ref)) return true;
    }
    bool arrived = false;
    bool prepared = false;
    // Phase is core-local. Arrival may enable preparation in this same pass.
    if (group.phase == SchedulerGmMixPhase::PUBLISHED && !scheduler_gm_mix_arrive(graph, base, local, run, &arrived))
        return false;
    if (group.phase == SchedulerGmMixPhase::ARRIVED &&
        !scheduler_gm_mix_prepare_lane(graph, base, local, run, profiling_level, &prepared))
        return false;
    *progress = *progress || arrived || prepared;
    return true;
}

inline __aicore__ bool scheduler_fill_ordered_slots(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t *victim_cursors, SchedulerReadyStats *ready_stats, uint64_t profiling_level, bool *failed,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerCompletionStats *completion_stats = nullptr
) {
    if (failed == nullptr) return false;
    *failed = false;
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) {
        *failed = true;
        return false;
    }
    const bool had_pending = !scheduler_task_ref_is_null(admission->pending_task_ref);
    auto published = scheduler_ordered_publish_pending(graph, base, local, run);
    if (published == SchedulerAdmissionResult::ERROR) {
        *failed = true;
        return false;
    }
    bool progress = had_pending && published == SchedulerAdmissionResult::PROGRESS;
    // Discovery is bounded by candidate credits, not by physical slot capacity.
    // A later cooperative task may be admitted while an earlier one is in flight.
    const uint32_t queues = local->has_mix && local->is_mix_coordinator() ? 2 : 1;
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT && published != SchedulerAdmissionResult::RETRY;
         ++index) {
        auto &candidate = admission->candidates[index];
        if (scheduler_task_ref_is_null(candidate.ready.task_ref)) {
            candidate = {};
            for (uint32_t attempt = 0; attempt < queues; ++attempt) {
                const uint32_t choice = (index + admission->next_candidate + attempt) % queues;
                const uint32_t type = choice == 0 ? local->config.self_core_type : SCHEDULER_MIX_QUEUE;
                const bool steal = scheduler_ready_directory_nonempty(base, local, local->config.scheduler_count, type);
                const uint64_t probe_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                if (!scheduler_claim_ready_for_slot(
                        graph, base, local, run, local->config.scheduler_count, type, &victim_cursors[type],
                        ready_stats, &candidate.ready, nullptr, steal
                    )) {
                    *failed = true;
                    return progress;
                }
                if (scheduler_task_ref_is_null(candidate.ready.task_ref)) continue;
                candidate.ready.state_probe_start_cycles = probe_start;
                candidate.ready.state_probe_end_cycles =
                    scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                break;
            }
            if (scheduler_task_ref_is_null(candidate.ready.task_ref)) continue;
            progress = true;
        }
        const auto metadata = scheduler_load_dispatch_metadata(base, local, candidate.ready.task_ref, profiling_level);
        if (scheduler_task_is_inline(metadata.flags)) {
            const int64_t task_ref = candidate.ready.task_ref;
            candidate = {};
            if (!scheduler_complete_inline_task(graph, base, local, run, task_ref, profiling_level)) {
                *failed = true;
                return progress;
            }
            progress = true;
            continue;
        }
        if (!scheduler_task_requires_admission(metadata)) continue;
        if (candidate.slot_index != UINT8_MAX ||
            scheduler_try_claim_admission(graph, base, local, run, admission, candidate.ready) !=
                SchedulerAdmissionResult::PROGRESS) {
            scheduler_admission_fail(
                graph, local, run, candidate.ready.task_ref, SchedulerErrorSite::ORDERED_CANDIDATE_INVALID
            );
            *failed = true;
            return progress;
        }
        candidate = {};
        published = scheduler_ordered_publish_pending(graph, base, local, run);
        if (published == SchedulerAdmissionResult::ERROR) {
            *failed = true;
            return progress;
        }
        progress = true;
    }
    if (!scheduler_refresh_local_obligation(graph, base, local, run, admission)) {
        *failed = true;
        return progress;
    }
    if (!scheduler_task_ref_is_null(admission->current_task_ref)) {
        if (!scheduler_ordered_park_ordinary(graph, local, run)) {
            *failed = true;
            return progress;
        }
        if (scheduler_task_is_mix(admission->current_flags)) {
            bool group_progress = false;
            if (!scheduler_service_gm_mix(
                    graph, base, local, run, profiling_level, wake_stats, ready_stats, completion_stats, &group_progress
                ))
                *failed = true;
            return progress || group_progress;
        }
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
            if (local->slots[index].state != SchedulerDispatchSlotState::FREE) continue;
            const auto claim_result = scheduler_claim_ordered_block(graph, base, local, run, index);
            if (claim_result == SchedulerBlockClaimResult::ERROR) {
                *failed = true;
                return progress;
            }
            if (claim_result != SchedulerBlockClaimResult::CLAIMED) break;
            auto &slot = local->slots[index];
            SchedulerReadyClaim ready{};
            ready.task_ref = slot.task_ref;
            ready.source = SchedulerReadySource::COOPERATIVE;
            const SchedulerFreeSlotClaim claim{local->worker_id(), index, slot.generation, local->config.self_lane};
            if (!scheduler_fill_dispatch_slot(graph, base, local, run, claim, ready, profiling_level, ssbuf_region)) {
                *failed = true;
                return progress;
            }
            progress = true;
        }
        bool closed = false;
        if (!scheduler_close_ordered_obligation(graph, base, local, run, &closed)) *failed = true;
        return progress || closed;
    }
    bool may_prepare = false;
    if (!scheduler_can_enter_ordinary(graph, base, local, run, admission, &may_prepare)) {
        *failed = true;
        return progress;
    }
    if (!may_prepare) return progress;
    for (uint32_t offset = 0; offset < SCHEDULER_PENDING_SLOT_COUNT; ++offset) {
        const uint32_t candidate_index = (admission->next_candidate + offset) % SCHEDULER_PENDING_SLOT_COUNT;
        auto &candidate = admission->candidates[candidate_index];
        if (scheduler_task_ref_is_null(candidate.ready.task_ref) || candidate.slot_index != UINT8_MAX) continue;
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
            auto &slot = local->slots[index];
            if (slot.state != SchedulerDispatchSlotState::FREE) continue;
            slot.candidate_index = static_cast<uint8_t>(candidate_index);
            slot.predicate_cached = candidate.predicate_cached;
            slot.predicate_pass = candidate.predicate_pass;
            slot.state = SchedulerDispatchSlotState::FILLING;
            candidate.ready.publication_mode = SchedulerPublicationMode::DISPATCH;
            const SchedulerFreeSlotClaim claim{local->worker_id(), index, slot.generation, local->config.self_lane};
            if (!scheduler_fill_dispatch_slot(
                    graph, base, local, run, claim, candidate.ready, profiling_level, ssbuf_region
                )) {
                *failed = true;
                return progress;
            }
            candidate.slot_index = static_cast<uint8_t>(index);
            candidate.predicate_cached = slot.predicate_cached;
            candidate.predicate_pass = slot.predicate_pass;
            progress = true;
            break;
        }
    }
    return progress;
}

inline __aicore__ bool scheduler_drain_ordered_exit(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *done
) {
    if (done == nullptr) return false;
    *done = false;
    if (local == nullptr) return false;
    auto *admission = local->admission;
    if (admission == nullptr) {
        *done = true;
        return true;
    }
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (!scheduler_task_ref_is_null(admission->pending_task_ref))
        return scheduler_admission_fail(
            graph, local, run, admission->pending_task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    for (const auto &candidate : admission->candidates)
        if (!scheduler_task_ref_is_null(candidate.ready.task_ref))
            return scheduler_admission_fail(
                graph, local, run, candidate.ready.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
            );
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
            );
    if (!scheduler_refresh_local_obligation(graph, base, local, run, admission)) return false;
    if (!scheduler_task_ref_is_null(admission->current_task_ref)) {
        bool closed = false;
        if (scheduler_task_is_mix(admission->current_flags)) {
            const auto phase = admission->gm_mix.phase;
            if ((local->is_mix_coordinator() && phase != SchedulerGmMixPhase::IDLE) ||
                (!local->is_mix_coordinator() && phase != SchedulerGmMixPhase::IDLE &&
                 phase != SchedulerGmMixPhase::RELEASED))
                return scheduler_admission_fail(
                    graph, local, run, admission->current_task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
                );
            if (!scheduler_gm_mix_close(graph, base, local, run, &closed)) return false;
        } else if (!scheduler_close_ordered_obligation(graph, base, local, run, &closed)) {
            return false;
        }
        return true;
    }
    if (admission->capacity == 0) {
        *done = true;
        return true;
    }
    auto *control = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t tail = scheduler_gm_fetch_add(control->commit_tail, UINT64_C(0));
    if (tail != admission->capacity || admission->scan_cursor > tail)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    *done = admission->scan_cursor == tail;
    return true;
}

// A directory load is useful only when the corresponding work can be claimed,
// so this must admit every slot scheduler_fill_own_slots would fill: rejecting
// a fillable slot suppresses the directory query and starves its work.
inline __aicore__ bool
scheduler_has_usable_slot(const SchedulerLocalState *local, uint32_t core_type, uint64_t skip_slot_mask) {
    if (local->config.self_core_type != core_type) return false;
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if ((skip_slot_mask & (UINT64_C(1) << slot)) == 0 &&
            local->own_slots()[slot].state == SchedulerDispatchSlotState::FREE)
            return true;
    }
    return false;
}

// Fill this core's own free dispatch slots. One core, one core type, two slots:
// a Scheduler schedules only itself, so there is no peer to prefer, no reserved
// capacity to hold back, and no reason to defer a slot it is about to execute
// from. `skip_slot_mask` names slots a completion pass already refilled.
//
// The return value reports whether this pass made progress; `failed` separately
// reports an aborted pass.
inline __aicore__ bool scheduler_fill_own_slots(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t *ready_victim_cursors, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, uint64_t skip_slot_mask, bool *failed, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region,
    SchedulerWakeStats *wake_stats = nullptr, SchedulerCompletionStats *completion_stats = nullptr
) {
    if (failed != nullptr) *failed = false;
    if (scheduler == nullptr) return false;
    if (scheduler->admission != nullptr)
        return scheduler_fill_ordered_slots(
            graph, scheduler_state_base, scheduler, run_control, ready_victim_cursors, ready_stats, profiling_level,
            failed, ssbuf_region, wake_stats, completion_stats
        );
    const uint32_t core_type = scheduler->config.self_core_type;
    if (!scheduler_has_usable_slot(scheduler, core_type, skip_slot_mask)) return false;
    // A core polls its own inbox on every pass. The directory gates only the
    // steal scan, which reads one peer endpoint per victim. Any publisher sets
    // an owner's bit and only that owner clears it, after observing its own
    // inbox empty -- so gating the owner's own pop on that bit would strand
    // whatever is queued when a clear and a push cross, until a later push
    // sets the bit again.
    const bool allow_steal = scheduler_ready_directory_nonempty(
        scheduler_state_base, scheduler, scheduler->config.scheduler_count, core_type
    );

    const uint64_t worker_id = scheduler->worker_id();
    const uint32_t self_lane = scheduler->config.self_lane;
    uint64_t state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    bool progress = false;

    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if ((skip_slot_mask & (UINT64_C(1) << slot)) != 0) continue;
        SchedulerLocalSlotState *local_slot = &scheduler->own_slots()[slot];
        if (local_slot->state != SchedulerDispatchSlotState::FREE) continue;

        SchedulerReadyClaim ready{};
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count, core_type,
                &ready_victim_cursors[core_type], ready_stats, &ready, nullptr, allow_steal
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        // An empty inbox ends the pass: the next slot would query the same one.
        if (scheduler_task_ref_is_null(ready.task_ref)) break;

        ready.state_probe_start_cycles = state_probe_start_cycles;
        ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        const SchedulerFreeSlotClaim claim{worker_id, slot, local_slot->generation, self_lane};
        local_slot->state = SchedulerDispatchSlotState::FILLING;
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        progress = true;
        state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    }
    return progress;
}
