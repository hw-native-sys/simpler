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

// Keep the default recording contract for existing diagnostic callers. Only
// the resident loop's no-profiling instantiation opts out; ownership and all
// publication/acknowledgement operations remain independent of this mode.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_service_gm_mix(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, SchedulerGmMixPhase ack_prefilter_phase = SchedulerGmMixPhase::IDLE
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarMix, -1);
    *progress = false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    bool closed = false;
    // Closing validates the full context. If it retains this obligation,
    // subsequent stages share its task/resource binding until this return.
    if (!scheduler_gm_mix_close<RecordProtocol>(graph, base, local, run, &closed)) return false;
    if (closed) {
        *progress = true;
        return true;
    }
    if (local->is_mix_coordinator()) {
        if (group.phase == SchedulerGmMixPhase::RELEASED) {
            const auto retired = scheduler_gm_mix_retire<RecordProtocol, true>(
                graph, base, local, run, ack_prefilter_phase == SchedulerGmMixPhase::RELEASED
            );
            if (retired == SchedulerGmMixResult::ERROR) return false;
            if (retired == SchedulerGmMixResult::TASK_LAST &&
                !scheduler_complete_ordered_task<RecordProtocol>(
                    graph, base, local, run, admission.current_task_ref, profiling_level
                ))
                return false;
            if (retired == SchedulerGmMixResult::PROGRESS || retired == SchedulerGmMixResult::TASK_LAST) {
                *progress = true;
                if (!scheduler_gm_mix_close<RecordProtocol, true>(graph, base, local, run, &closed)) return false;
                if (closed) return true;
            }
        }
        if (group.phase == SchedulerGmMixPhase::IDLE) {
            const auto claimed = scheduler_gm_mix_claim<RecordProtocol, true>(graph, base, local, run);
            if (claimed == SchedulerBlockClaimResult::ERROR) return false;
            if (claimed != SchedulerBlockClaimResult::CLAIMED) return true;
            *progress = true;
        }
        if (group.phase == SchedulerGmMixPhase::CLAIMED) {
            if (!scheduler_gm_mix_publish<RecordProtocol, true>(graph, base, local, run)) return false;
            *progress = true;
        }
    } else if (group.phase == SchedulerGmMixPhase::IDLE || group.phase == SchedulerGmMixPhase::RELEASED) {
        bool observed = false;
        if (!scheduler_gm_mix_observe<RecordProtocol, true>(graph, base, local, run, &observed)) return false;
        *progress = observed;
        if (scheduler_task_ref_is_null(admission.current_task_ref)) return true;
    }
    bool arrived = false;
    bool prepared = false;
    // Phase is core-local. Arrival may enable preparation in this same pass.
    if (group.phase == SchedulerGmMixPhase::PUBLISHED &&
        !scheduler_gm_mix_arrive<RecordProtocol, true>(graph, base, local, run, &arrived))
        return false;
    if (group.phase == SchedulerGmMixPhase::ARRIVED &&
        !scheduler_gm_mix_prepare_lane<RecordProtocol, true>(
            graph, base, local, run, profiling_level, &prepared, ack_prefilter_phase == SchedulerGmMixPhase::ARRIVED
        ))
        return false;
    *progress = *progress || arrived || prepared;
    return true;
}

// A false hint skips only this poll. It neither grants a kernel permit nor
// changes ownership, a frontier, an acknowledgement, or a completion count.
// Stale observations may postpone work; every positive hint takes the normal
// ownership validation path before doing that work.
inline __aicore__ bool scheduler_ordered_poll_may_progress(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, bool *ready_probe_empty = nullptr,
    SchedulerGmMixPhase *ack_prefilter_phase = nullptr
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarOrderedPoll, -1);
    SchedulerScalarScope scalar_local(local, SchedPhaseKind::ScalarPollLocalState);
    if (ready_probe_empty != nullptr) *ready_probe_empty = false;
    if (ack_prefilter_phase != nullptr) *ack_prefilter_phase = SchedulerGmMixPhase::IDLE;
    const auto &admission = *local->admission;
    if (admission.capacity == 0 || admission.scan_cursor > admission.capacity || local->local_ready_mask != 0 ||
        !scheduler_task_ref_is_null(admission.pending_task_ref))
        return true;
    for (const auto &candidate : admission.candidates)
        if (!scheduler_task_ref_is_null(candidate.ready.task_ref)) return true;
    scalar_local.close();

    const uint32_t type = local->config.self_core_type;
    if (type >= SCHEDULER_READY_QUEUE_COUNT || local->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        !scheduler_ready_queue_holder(type, local->config.self_core_type))
        return true;
    {
        SchedulerScalarScope scalar_queue(local, SchedPhaseKind::ScalarPollReadyQueue);
        if (scheduler_scalar_result(local, SchedPhaseKind::ScalarPollReadyPool, [&]() __aicore__ {
                return scheduler_ready_pool_may_have_work(base, local, type);
            }))
            return true;
    }
    // Reuse only this poll's completed negative Ready probes. Invalid discovery
    // configuration must still reach the original claim validation path.
    if (ready_probe_empty != nullptr && local->config.scheduler_count != 0 &&
        local->config.scheduler_count <= SCHEDULER_WORKER_CAPACITY &&
        local->config.scheduler_index < local->config.scheduler_count)
        *ready_probe_empty = true;
    if (scheduler_task_ref_is_null(admission.current_task_ref)) {
        auto *control = scheduler_state_at<SchedulerAdmissionControl>(base, admission.control_offset);
        return scheduler_scalar_result(local, SchedPhaseKind::ScalarPollAdmissionTail, [&]() __aicore__ {
                   return scheduler_gm_query(control->commit_tail);
               }) != admission.scan_cursor;
    }
    if (!scheduler_task_ref_valid(graph, admission.current_task_ref) || !scheduler_task_is_mix(admission.current_flags))
        return true;
    SchedulerScalarScope scalar_mix(local, SchedPhaseKind::ScalarPollMixState);
    const auto &group = admission.gm_mix;
    if (group.mailbox_offset == 0 || group.cluster_index >= local->config.cluster_count) return true;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    if (!local->is_mix_coordinator() &&
        (group.phase == SchedulerGmMixPhase::IDLE || group.phase == SchedulerGmMixPhase::RELEASED)) {
        const auto *record = scheduler_admission_record_at(base, &admission, admission.current_task_ref);
        auto *seals = scheduler_state_at<SchedulerMixCloseSeal>(base, record->mix_close_seals_offset);
        return scheduler_scalar_result(
                   local, SchedPhaseKind::ScalarPollMixCloseSeal,
                   [&]() __aicore__ {
                       return scheduler_gm_query(seals[group.cluster_index].value);
                   }
               ) != 0 ||
               scheduler_scalar_result(local, SchedPhaseKind::ScalarPollMixAssignment, [&]() __aicore__ {
                   return scheduler_gm_query(mailbox->lanes[local->config.self_lane].assigned_seq);
               }) > group.group_seq;
    }
    const bool releases = group.phase == SchedulerGmMixPhase::RELEASED && local->is_mix_coordinator();
    if (group.phase == SchedulerGmMixPhase::ARRIVED || releases) {
        if (group.group_seq == 0 || group.group_seq == UINT64_MAX) return true;
        const LaneMask resources = scheduler_mix_resource_mask(admission.current_active_mask);
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
            auto &ack = releases ? mailbox->lanes[lane].released_seq : mailbox->lanes[lane].arrived_seq;
            if (scheduler_scalar_result(
                    local, releases ? SchedPhaseKind::ScalarPollMixReleased : SchedPhaseKind::ScalarPollMixArrived,
                    [&]() __aicore__ {
                        return scheduler_gm_query(ack);
                    }
                ) < group.group_seq)
                return false;
        }
        if (ack_prefilter_phase != nullptr) *ack_prefilter_phase = group.phase;
    }
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_fill_ordered_slots(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *failed
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarOrderedFill, -1);
    if (failed == nullptr) return false;
    *failed = false;
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) {
        *failed = true;
        return false;
    }
    bool ready_probe_empty = false;
    auto ack_prefilter_phase = SchedulerGmMixPhase::IDLE;
    if (!scheduler_ordered_poll_may_progress(graph, base, local, &ready_probe_empty, &ack_prefilter_phase))
        return false;
    if (scheduler_read_error(run) != 0) {
        *failed = true;
        return false;
    }
    const bool had_pending = !scheduler_task_ref_is_null(admission->pending_task_ref);
    auto published = scheduler_ordered_publish_pending<RecordProtocol>(graph, base, local, run);
    if (published == SchedulerAdmissionResult::ERROR) {
        *failed = true;
        return false;
    }
    bool progress = had_pending && published == SchedulerAdmissionResult::PROGRESS;
    bool holds_ordinary = false;
    for (const auto &candidate : admission->candidates) {
        if (scheduler_task_ref_is_null(candidate.ready.task_ref)) continue;
        const auto metadata = scheduler_load_dispatch_metadata(base, local, candidate.ready.task_ref, profiling_level);
        if (!scheduler_task_is_inline(metadata.flags) && !scheduler_task_requires_admission(metadata)) {
            holds_ordinary = true;
            break;
        }
    }
    // Discovery is bounded by candidate credits, not by physical slot capacity.
    // A later cooperative task may be admitted while an earlier one is in flight.
    // A concurrent Ready publication may wait for the next finite poll. The
    // hint grants no ownership or entry permit and never survives this fill.
    for (uint32_t index = 0;
         !ready_probe_empty && index < SCHEDULER_PENDING_SLOT_COUNT && published != SchedulerAdmissionResult::RETRY;
         ++index) {
        auto &candidate = admission->candidates[index];
        if (scheduler_task_ref_is_null(candidate.ready.task_ref)) {
            // Existing cooperative candidates progress while ordinary work is held privately.
            if (holds_ordinary) continue;
            candidate = {};
            const uint64_t probe_start = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
            if (!scheduler_claim_ready_for_slot(
                    graph, base, local, run, local->config.scheduler_count, local->config.self_core_type,
                    &candidate.ready
                )) {
                *failed = true;
                return progress;
            }
            if (scheduler_task_ref_is_null(candidate.ready.task_ref)) continue;
            candidate.ready.state_probe_start_cycles = probe_start;
            candidate.ready.state_probe_end_cycles =
                scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
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
        if (!scheduler_task_requires_admission(metadata)) {
            holds_ordinary = true;
            continue;
        }
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
        published = scheduler_ordered_publish_pending<RecordProtocol>(graph, base, local, run);
        if (published == SchedulerAdmissionResult::ERROR) {
            *failed = true;
            return progress;
        }
        progress = true;
    }
    // Discovery and completion service regain control after at most two
    // closed obligations; no member waits for a peer inside this pass.
    for (uint32_t step = 0; step < SCHEDULER_PENDING_SLOT_COUNT; ++step) {
        const uint64_t scan_cursor_before = admission->scan_cursor;
        if (!scheduler_refresh_local_obligation<RecordProtocol>(graph, base, local, run, admission)) {
            *failed = true;
            return progress;
        }
        // A bounded scan can retire obligations without retaining a current task.
        progress = progress || admission->scan_cursor != scan_cursor_before;
        if (scheduler_task_ref_is_null(admission->current_task_ref)) break;
        if (!scheduler_ordered_park_ordinary(graph, local, run)) {
            *failed = true;
            return progress;
        }
        if (scheduler_task_is_mix(admission->current_flags)) {
            bool group_progress = false;
            if (!scheduler_service_gm_mix<RecordProtocol>(
                    graph, base, local, run, profiling_level, &group_progress, ack_prefilter_phase
                )) {
                *failed = true;
                return progress || group_progress;
            }
            progress = progress || group_progress;
            if (scheduler_task_ref_is_null(admission->current_task_ref)) continue;
            return progress;
        }
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT && local->local_ready_mask == 0; ++index) {
            if (local->slots[index].state != SchedulerDispatchSlotState::FREE) continue;
            const auto claim_result = scheduler_claim_ordered_block<RecordProtocol>(graph, base, local, run, index);
            if (claim_result == SchedulerBlockClaimResult::ERROR) {
                *failed = true;
                return progress;
            }
            if (claim_result != SchedulerBlockClaimResult::CLAIMED) break;
            auto &slot = local->slots[index];
            SchedulerReadyClaim ready{};
            ready.task_ref = slot.task_ref;
            ready.source = SchedulerReadySource::COOPERATIVE;
            const SchedulerFreeSlotClaim claim{index};
            if (!scheduler_fill_dispatch_slot(graph, base, local, run, claim, ready, profiling_level)) {
                *failed = true;
                return progress;
            }
            progress = true;
        }
        bool closed = false;
        if (!scheduler_close_ordered_obligation<RecordProtocol>(graph, base, local, run, &closed)) {
            *failed = true;
            return progress;
        }
        progress = progress || closed;
        if (!closed) return progress;
    }
    bool may_prepare = false;
    for (uint32_t offset = 0; offset < SCHEDULER_PENDING_SLOT_COUNT; ++offset) {
        const uint32_t candidate_index = (admission->next_candidate + offset) % SCHEDULER_PENDING_SLOT_COUNT;
        auto &candidate = admission->candidates[candidate_index];
        if (scheduler_task_ref_is_null(candidate.ready.task_ref) || candidate.slot_index != UINT8_MAX) continue;
        // Completion can retain a direct successor after discovery. Inline
        // services and cooperative successors still require that path.
        const auto metadata = scheduler_load_dispatch_metadata(base, local, candidate.ready.task_ref, profiling_level);
        if (scheduler_task_is_inline(metadata.flags) || scheduler_task_requires_admission(metadata)) continue;
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
            auto &slot = local->slots[index];
            if (slot.state != SchedulerDispatchSlotState::FREE) continue;
            // This decision covers preparation in this pass; each kernel entry rechecks admission.
            if (!may_prepare) {
                if (!scheduler_can_enter_ordinary(graph, base, local, run, admission, &may_prepare)) {
                    *failed = true;
                    return progress;
                }
                if (!may_prepare) return progress;
            }
            slot.candidate_index = static_cast<uint8_t>(candidate_index);
            slot.args_observed = candidate.args_observed;
            slot.predicate = candidate.predicate;
            slot.state = SchedulerDispatchSlotState::FILLING;
            candidate.ready.publication_mode = SchedulerPublicationMode::DISPATCH;
            const SchedulerFreeSlotClaim claim{index};
            if (!scheduler_fill_dispatch_slot(graph, base, local, run, claim, candidate.ready, profiling_level)) {
                *failed = true;
                return progress;
            }
            candidate.slot_index = static_cast<uint8_t>(index);
            candidate.args_observed = slot.args_observed;
            candidate.predicate = slot.predicate;
            progress = true;
            break;
        }
    }
    return progress;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_drain_ordered_exit(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
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
    if (!scheduler_refresh_local_obligation<RecordProtocol>(graph, base, local, run, admission)) return false;
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
            if (!scheduler_gm_mix_close<RecordProtocol>(graph, base, local, run, &closed)) return false;
        } else if (!scheduler_close_ordered_obligation<RecordProtocol>(graph, base, local, run, &closed)) {
            return false;
        }
        return true;
    }
    if (admission->capacity == 0) {
        *done = true;
        return true;
    }
    auto *control = scheduler_state_at<SchedulerAdmissionControl>(base, admission->control_offset);
    const uint64_t tail = scheduler_gm_query(control->commit_tail);
    if (tail != admission->capacity || admission->scan_cursor > tail)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    *done = admission->scan_cursor == tail;
    return true;
}

// A ready probe is useful only when the corresponding work can be claimed,
// so this must admit every slot scheduler_fill_own_slots would fill: rejecting
// a fillable slot suppresses discovery and starves its work.
inline __aicore__ bool scheduler_has_usable_slot(const SchedulerLocalState *local, uint32_t core_type) {
    if (local->config.self_core_type != core_type) return false;
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if (local->own_slots()[slot].state == SchedulerDispatchSlotState::FREE) return true;
    }
    return false;
}

// Claim only the next ordinary execution. Reserving another ready task behind
// a running kernel can hide work needed by that kernel from every idle core.
// Cooperative admission retains its separate slot/participant protocol.
//
// The return value reports whether this pass made progress; `failed` separately
// reports an aborted pass.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_fill_own_slots(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t profiling_level, bool *failed
) {
    SchedulerScalarScope scalar_scope(scheduler, SchedPhaseKind::ScalarFillSlots, -1);
    if (failed != nullptr) *failed = false;
    if (scheduler == nullptr) return false;
    if (scheduler->admission != nullptr)
        return scheduler_fill_ordered_slots<RecordProtocol>(
            graph, scheduler_state_base, scheduler, run_control, profiling_level, failed
        );
    if (scheduler->local_ready_mask != 0) return false;
    const uint32_t core_type = scheduler->config.self_core_type;
    if (!scheduler_has_usable_slot(scheduler, core_type)) return false;
    uint64_t state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    bool progress = false;

    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        SchedulerLocalSlotState *local_slot = &scheduler->own_slots()[slot];
        if (local_slot->state != SchedulerDispatchSlotState::FREE) continue;

        SchedulerReadyClaim ready{};
        if (!scheduler_claim_ready_for_slot(
                graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count, core_type,
                &ready
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        // An unpublished head ends this discovery pass; other scheduler duties still progress.
        if (scheduler_task_ref_is_null(ready.task_ref)) break;

        ready.state_probe_start_cycles = state_probe_start_cycles;
        ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        const SchedulerFreeSlotClaim claim{slot};
        local_slot->state = SchedulerDispatchSlotState::FILLING;
        if (!scheduler_fill_dispatch_slot(
                graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level
            )) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        progress = true;
        break;
    }
    return progress;
}
