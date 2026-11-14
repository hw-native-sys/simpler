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

#include "scheduler_completion.h"
#include "scheduler_execution_tokens.h"
#include "scheduler_cooperative_discovery.h"

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
            if (retired == SchedulerGmMixResult::PROGRESS) {
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
static inline __aicore__ bool scheduler_ordered_poll_may_progress(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerGmMixPhase *ack_prefilter_phase = nullptr
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarOrderedPoll, -1);
    SchedulerScalarScope scalar_local(local, SchedPhaseKind::ScalarPollLocalState);
    if (ack_prefilter_phase != nullptr) *ack_prefilter_phase = SchedulerGmMixPhase::IDLE;
    const auto &admission = *local->admission;
    if (admission.capacity == 0 || admission.scan_cursor > admission.capacity || local->local_ready_mask != 0 ||
        !scheduler_task_ref_is_null(admission.pending_task_ref))
        return true;
    scalar_local.close();

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
        auto &seal = scheduler_state_at<SchedulerMixCloseSeal>(base, admission.current_mix_close_seal_offset)->value;
        return scheduler_scalar_result(
                   local, SchedPhaseKind::ScalarPollMixCloseSeal,
                   [&]() __aicore__ {
                       return scheduler_gm_query(seal);
                   }
               ) != 0 ||
               scheduler_scalar_result(local, SchedPhaseKind::ScalarPollMixAssignment, [&]() __aicore__ {
                   return scheduler_gm_query(mailbox->lanes[local->config.self_lane].assigned_seq);
               }) > group.group_seq;
    }
    const bool releases = group.phase == SchedulerGmMixPhase::RELEASED && local->is_mix_coordinator();
    if (group.phase == SchedulerGmMixPhase::ARRIVED || releases) {
        if (group.group_seq == 0 || group.group_seq == UINT64_MAX) return true;
        const LaneMask resources = admission.current_resource_mask;
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
    auto *admission = local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) {
        *failed = true;
        return false;
    }
    auto ack_prefilter_phase = SchedulerGmMixPhase::IDLE;
    if (!scheduler_ordered_poll_may_progress(graph, base, local, &ack_prefilter_phase)) return false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) {
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
        for (uint32_t index = 0;
             index < SCHEDULER_PENDING_SLOT_COUNT && local->local_ready_mask == 0 && !scheduler_has_direct_slot(local);
             ++index) {
            auto &slot = local->slots[index];
            if (slot.state == SchedulerDispatchSlotState::FREE) {
                const auto claim_result = scheduler_claim_ordered_block<RecordProtocol>(graph, base, local, run, index);
                if (claim_result == SchedulerBlockClaimResult::ERROR) {
                    *failed = true;
                    return progress;
                }
                if (claim_result != SchedulerBlockClaimResult::CLAIMED) break;
                progress = true;
            } else if (slot.state != SchedulerDispatchSlotState::FILLING) {
                continue;
            }
            SchedulerReadyClaim ready{};
            ready.task_ref = slot.task_ref;
            ready.source = SchedulerReadySource::COOPERATIVE;
            const SchedulerFreeSlotClaim claim{index};
            if (!scheduler_fill_dispatch_slot(graph, base, local, run, claim, ready, profiling_level)) {
                *failed = true;
                return progress;
            }
            progress = progress || slot.state == SchedulerDispatchSlotState::READY;
            if (scheduler_has_direct_slot(local)) return progress;
            if (slot.state == SchedulerDispatchSlotState::FILLING) return progress;
        }
        bool closed = false;
        if (!scheduler_close_ordered_obligation<RecordProtocol>(graph, base, local, run, &closed)) {
            *failed = true;
            return progress;
        }
        progress = progress || closed;
        if (!closed) return progress;
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
    bool cluster_done = false;
    if (!scheduler_cluster_mix_drain(graph, base, local, run, &cluster_done)) return false;
    if (!cluster_done) return true;
    auto *admission = local->admission;
    if (admission == nullptr) {
        *done = true;
        return true;
    }
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (!scheduler_task_ref_is_null(admission->pending_task_ref)) {
        const int64_t task = admission->pending_task_ref;
        if (!scheduler_task_ref_valid(graph, task) ||
            scheduler_gm_fetch_add(scheduler_task_control_at(base, local, task)->admission_state, UINT64_C(0)) !=
                static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED))
            return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT);
        admission->pending_task_ref = SCHEDULER_TASK_REF_INVALID;
    }
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE)
            return scheduler_admission_fail(
                graph, local, run, slot.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
            );
    if (!scheduler_refresh_local_obligation<RecordProtocol>(graph, base, local, run, admission)) return false;
    if (!scheduler_task_ref_is_null(admission->current_task_ref)) {
        bool closed = false;
        if (scheduler_task_is_mix(admission->current_flags)) {
            // An inactive coordinator can still owe its resource release
            // after both executing AIV lanes have completed the logical task.
            if (local->is_mix_coordinator() && admission->gm_mix.phase == SchedulerGmMixPhase::ARRIVED &&
                !admission->current_active_mask.has_aic() && admission->gm_mix.subtask_slot == UINT8_MAX) {
                if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
                bool arrived = false;
                if (!scheduler_gm_mix_check_acks<RecordProtocol>(graph, base, local, run, false, &arrived))
                    return false;
                if (!arrived) return true;
                scheduler_gm_mix_publish_release<RecordProtocol>(base, local);
            }
            if (local->is_mix_coordinator() && admission->gm_mix.phase == SchedulerGmMixPhase::RELEASED) {
                const auto result = scheduler_gm_mix_retire<RecordProtocol>(graph, base, local, run);
                if (result == SchedulerGmMixResult::ERROR) return false;
                if (result == SchedulerGmMixResult::WAIT) return true;
            }
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

// Discovery is independent of physical slot capacity and formal group acknowledgements.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_fill_cooperative_slots(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *failed, uint32_t *direct_slot = nullptr
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarFillCooperativeSlots, -1);
    if (failed != nullptr) *failed = false;
    if (direct_slot != nullptr) *direct_slot = UINT32_MAX;
    if (local == nullptr) return false;
    SchedulerDirectSlotScope direct_scope(local, direct_slot);
    bool progress = false;
    bool advanced = false;
    if (!scheduler_cluster_mix_service(graph, base, local, run, profiling_level, &progress) ||
        !scheduler_discover_cooperative_task(graph, base, local, run, &advanced)) {
        if (failed != nullptr) *failed = true;
        return progress;
    }
    progress = progress || advanced;
    if (!scheduler_cluster_mix_discover_next(graph, base, local, run, &advanced) ||
        !scheduler_prepare_cluster_lane(base, local)) {
        if (failed != nullptr) *failed = true;
        return progress;
    }
    progress = progress || advanced;
    if (!scheduler_cluster_mix_progress_pending(graph, base, local, run, &advanced)) {
        if (failed != nullptr) *failed = true;
        return progress;
    }
    progress = progress || advanced;
    if (advanced) {
        if (!scheduler_cluster_mix_service(graph, base, local, run, profiling_level, &advanced)) {
            if (failed != nullptr) *failed = true;
            return progress;
        }
        progress = progress || advanced;
    }
    const auto phase = local->cluster_mix.phase;
    if (phase == SchedulerGmMixPhase::ARRIVED || phase == SchedulerGmMixPhase::RUNNING ||
        (phase == SchedulerGmMixPhase::RELEASED && local->is_mix_coordinator()))
        return progress;
    if (local->admission != nullptr)
        return scheduler_fill_ordered_slots<RecordProtocol>(graph, base, local, run, profiling_level, failed) ||
               progress;
    return progress;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_fill_own_slots(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *failed, uint32_t *direct_slot = nullptr
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarFillOwnSlots);
    bool local_failed = false;
    if (failed == nullptr) failed = &local_failed;
    bool progress =
        scheduler_fill_cooperative_slots<RecordProtocol>(graph, base, local, run, profiling_level, failed, direct_slot);
    if (*failed || local == nullptr || local->local_ready_mask != 0 ||
        (direct_slot != nullptr && *direct_slot != UINT32_MAX))
        return progress;
    uint32_t selected = UINT32_MAX;
    bool advanced = false;
    if (scheduler_cluster_mix_needs_service(local)) {
        SchedulerDirectSlotScope direct_scope(local, direct_slot);
        if (!scheduler_cluster_mix_service(graph, base, local, run, profiling_level, &advanced)) *failed = true;
        return progress || advanced;
    }
    if (!scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, &selected, &advanced)) {
        *failed = true;
        return progress;
    }
    progress = progress || advanced;
    if (selected == UINT32_MAX) {
        if (!scheduler_try_execution_continuation(graph, base, local, run, profiling_level, &advanced) ||
            (!advanced && !scheduler_refill_execution_tokens(graph, base, local, run, profiling_level, &advanced))) {
            *failed = true;
            return progress;
        }
        progress = progress || advanced;
        if (!scheduler_progress_execution_tokens(graph, base, local, run, profiling_level, &selected, &advanced)) {
            *failed = true;
            return progress;
        }
        progress = progress || advanced;
    }
    if (selected != UINT32_MAX) {
        if (direct_slot != nullptr) *direct_slot = selected;
        else scheduler_local_ready_publish(local, selected);
    }
    return progress;
}
