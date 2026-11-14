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

#include "scheduler_admission.h"
#include "scheduler_async_control.h"

enum class SchedulerGmMixResult : uint8_t { WAIT, PROGRESS, ERROR };

// StaticContextValidated is confined to one service call whose task, resource
// membership and mailbox binding cannot change before returning. Admission
// observes error and epoch; physical-return release retains epoch and identity
// without granting execution.
template <bool StaticContextValidated = false, bool ReturningLane = false>
static inline __aicore__ bool scheduler_gm_mix_context_valid(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if constexpr (StaticContextValidated && !ReturningLane)
        return scheduler_admission_detail::active_run_valid(graph, base, local, run, local->admission);
    auto *admission = local->admission;
    if constexpr (ReturningLane) {
        // A real return still releases its resource after a run error. Epoch
        // and group identity remain mandatory; this grants no new execution.
        if (admission == nullptr) return false;
        const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
        if (admission->epoch != summary->run_epoch)
            return scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
            );
    } else if (!scheduler_admission_context_valid(graph, base, local, run, admission)) {
        return false;
    }
    if (!scheduler_task_ref_valid(graph, admission->current_task_ref) ||
        !scheduler_task_is_mix(admission->current_flags) || admission->gm_mix.mailbox_offset == 0 ||
        admission->gm_mix.cluster_index >= local->config.cluster_count ||
        local->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        !admission->current_resource_mask.has(static_cast<SubtaskSlot>(local->config.self_lane)))
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    return true;
}

inline __aicore__ __gm__ SchedulerGroupMailbox *
scheduler_gm_mix_mailbox(__gm__ void *base, const SchedulerLocalState *local) {
    return scheduler_state_at<SchedulerGroupMailbox>(base, local->admission->gm_mix.mailbox_offset);
}

// Role and lane mapping were validated when the obligation was adopted. The
// reusable mailbox carries only this group's dynamic identity and lane history.
inline __aicore__ bool
scheduler_gm_mix_descriptor_sequence_valid(__gm__ const SchedulerGroupDescriptor *descriptor, uint64_t seq) {
    if (seq == 0 || seq == UINT64_MAX || descriptor->group_seq != seq || descriptor->previous_lane_seq[0] != seq - 1)
        return false;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        if (descriptor->previous_lane_seq[lane] >= seq) return false;
    return true;
}

inline __aicore__ bool scheduler_gm_mix_descriptor_valid(
    const SchedulerLocalState *local, __gm__ const SchedulerGroupDescriptor *descriptor, uint64_t seq
) {
    const auto &admission = *local->admission;
    return descriptor->run_epoch == admission.epoch && descriptor->task_ref == admission.current_task_ref &&
           descriptor->order_key == admission.scan_cursor && descriptor->block_idx < admission.current_block_num &&
           descriptor->cluster_index == admission.gm_mix.cluster_index &&
           scheduler_gm_mix_descriptor_sequence_valid(descriptor, seq);
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_close(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *closed, bool critical = false
) {
    if (closed == nullptr) return false;
    *closed = false;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    auto &seal = scheduler_state_at<SchedulerMixCloseSeal>(base, admission.current_mix_close_seal_offset)->value;
    uint64_t closed_cursor = UINT64_MAX;
    uint64_t close_seal = 0;
    bool used_claim_seal = false;
    if (local->is_mix_coordinator()) {
        if (group.phase != SchedulerGmMixPhase::IDLE) return true;
        auto *control = scheduler_task_control_at(base, local, admission.current_task_ref);
        if (scheduler_gm_query(control->next_block) < admission.current_block_num && !admission.sync_claimed)
            return true;
        // The query may only postpone closing. The error gate and the atomic
        // exhausted-cursor confirmation precede publishing a close seal.
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        uint64_t cursor = 0;
        if (!scheduler_read_claim_cursor(graph, base, local, run, &cursor, &used_claim_seal)) return false;
        if (cursor < admission.current_block_num && !admission.sync_claimed) return true;
        if (cursor > admission.current_block_num || (admission.sync_claimed && cursor == 0) ||
            group.task_last_seq == UINT64_MAX)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        for (const auto &slot : local->slots)
            if (slot.state != SchedulerDispatchSlotState::FREE) return true;
        if (scheduler_gm_fetch_add(seal, UINT64_C(0)) != 0)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        scheduler_cache_barrier();
        scheduler_gm_store(seal, group.task_last_seq + 1);
        closed_cursor = cursor;
        close_seal = group.task_last_seq + 1;
    } else {
        if (!critical && scheduler_gm_query(seal) == 0) return true;
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        const uint64_t published = scheduler_gm_fetch_add(seal, UINT64_C(0));
        if (published == 0) return true;
        if ((group.phase != SchedulerGmMixPhase::IDLE && group.phase != SchedulerGmMixPhase::RELEASED) ||
            group.task_last_seq == UINT64_MAX || published != group.task_last_seq + 1)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        for (const auto &slot : local->slots)
            if (slot.state != SchedulerDispatchSlotState::FREE)
                return scheduler_admission_fail(
                    graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
                );
        close_seal = published;
    }
    if (admission.scan_cursor >= admission.capacity)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
        );
    if constexpr (RecordProtocol)
        scheduler_protocol_member_closed(
            base, local, &admission, closed_cursor, close_seal, group.task_last_seq, used_claim_seal
        );
    scheduler_retire_local_obligation(local, admission);
    group.phase = SchedulerGmMixPhase::IDLE;
    group.task_last_seq = 0;
    group.slot_index = UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.subtask_slot = UINT8_MAX;
    *closed = true;
    return true;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ SchedulerBlockClaimResult scheduler_gm_mix_claim(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    // No shared ownership changes until the cursor helper's error gate.
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run))
        return SchedulerBlockClaimResult::ERROR;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator()) {
        scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
        return SchedulerBlockClaimResult::ERROR;
    }
    if (group.phase != SchedulerGmMixPhase::IDLE) return SchedulerBlockClaimResult::RETRY;
    if (admission.sync_claimed) return SchedulerBlockClaimResult::RETRY;
    if (group.group_seq >= UINT64_MAX - 1) {
        const uint64_t cursor = scheduler_gm_fetch_add(
            scheduler_task_control_at(base, local, admission.current_task_ref)->next_block, UINT64_C(0)
        );
        if (cursor == admission.current_block_num) return SchedulerBlockClaimResult::EXHAUSTED;
        scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_SEQUENCE_OVERFLOW
        );
        return SchedulerBlockClaimResult::ERROR;
    }
    uint32_t block = 0;
    const auto result = scheduler_claim_ordered_cursor<RecordProtocol>(graph, base, local, run, &block);
    if (result != SchedulerBlockClaimResult::CLAIMED) return result;
    group.block_idx = block;
    ++group.group_seq;
    group.phase = SchedulerGmMixPhase::CLAIMED;
    group.slot_index = UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.subtask_slot = UINT8_MAX;
    return SchedulerBlockClaimResult::CLAIMED;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_publish(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::CLAIMED || group.group_seq == 0 ||
        group.group_seq == UINT64_MAX || group.block_idx >= admission.current_block_num)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    auto *descriptor = &mailbox->descriptor;
    if constexpr (RecordProtocol) scheduler_protocol_group_publish(base, local, true);
    descriptor->run_epoch = admission.epoch;
    descriptor->task_ref = admission.current_task_ref;
    descriptor->order_key = admission.scan_cursor;
    descriptor->group_seq = group.group_seq;
    descriptor->block_idx = group.block_idx;
    descriptor->cluster_index = group.cluster_index;
    // Physical communication endpoints retain their logical lane identities.
    // Per-lane predecessors exclude groups that never used that lane.
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        const uint64_t previous = lane == 0 ? group.group_seq - 1 : group.vector_last_published_seq[lane - 1];
        if (previous >= group.group_seq)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
            );
        descriptor->previous_lane_seq[lane] = previous;
    }
    group.subtask_slot = admission.current_single_subtask;
    for (uint32_t offset = 0; offset < sizeof(*descriptor); offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptor) + offset);
    scheduler_cache_barrier();
    scheduler_gm_store(mailbox->published_seq, group.group_seq);
    for (uint8_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (admission.current_resource_mask.has(static_cast<SubtaskSlot>(lane))) {
            scheduler_gm_store(mailbox->lanes[lane].assigned_seq, group.group_seq);
            group.vector_last_published_seq[lane - 1] = group.group_seq;
        }
    }
    if constexpr (RecordProtocol) scheduler_protocol_group_publish(base, local, false);
    group.task_last_seq = group.group_seq;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    if constexpr (RecordProtocol) scheduler_protocol_group_descriptor(base, local, descriptor);
    return true;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_observe(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (local->is_mix_coordinator() ||
        (group.phase != SchedulerGmMixPhase::IDLE && group.phase != SchedulerGmMixPhase::RELEASED))
        return true;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    // Only an assigned lane holds the descriptor stable through its release.
    // Unassigned lanes must not read the mailbox while its owner can reuse it.
    auto &assigned = mailbox->lanes[local->config.self_lane].assigned_seq;
    if (scheduler_gm_query(assigned) <= group.group_seq) return true;
    const uint64_t seq = scheduler_gm_fetch_add(assigned, UINT64_C(0));
    if (seq <= group.group_seq) return true;
    auto *descriptor = &mailbox->descriptor;
    for (uint32_t offset = 0; offset < sizeof(*descriptor); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptor) + offset);
    scheduler_cache_barrier();
    asm volatile("" ::: "memory");
    if (descriptor->task_ref != admission.current_task_ref) {
        // A released lane may still be consuming A's retained seal while its
        // coordinator has already published B into the reusable mailbox.
        auto &seal = scheduler_state_at<SchedulerMixCloseSeal>(base, admission.current_mix_close_seal_offset)->value;
        if (scheduler_gm_fetch_add(seal, UINT64_C(0)) != 0) {
            bool closed = false;
            if (!scheduler_gm_mix_close<RecordProtocol, StaticContextValidated>(graph, base, local, run, &closed, true))
                return false;
            if (closed) {
                *progress = true;
                return true;
            }
        }
    }
    if (seq == UINT64_MAX || descriptor->previous_lane_seq[local->config.self_lane] != group.group_seq ||
        !scheduler_gm_mix_descriptor_valid(local, descriptor, seq))
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
        );
    group.group_seq = seq;
    group.task_last_seq = seq;
    group.block_idx = descriptor->block_idx;
    group.subtask_slot = admission.current_single_subtask;
    group.slot_index = UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    if constexpr (RecordProtocol) scheduler_protocol_group_descriptor(base, local, descriptor);
    *progress = true;
    return true;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_arrive(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &group = local->admission->gm_mix;
    if (group.phase != SchedulerGmMixPhase::PUBLISHED) return true;
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE) return true;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, false, true);
    scheduler_cache_barrier();
    scheduler_gm_store(mailbox->lanes[local->config.self_lane].arrived_seq, group.group_seq);
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, false, false);
    group.phase = SchedulerGmMixPhase::ARRIVED;
    *progress = true;
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_gm_mix_check_acks(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool releases, bool *all, bool query_prefiltered = false
) {
    *all = false;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    const uint64_t seq = local->admission->gm_mix.group_seq;
    const bool record = RecordProtocol && scheduler_protocol_enabled(local);
    uint64_t observations[PLATFORM_CORES_PER_BLOCKDIM]{};
    const LaneMask resources = local->admission->current_resource_mask;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
        auto &value = releases ? mailbox->lanes[lane].released_seq : mailbox->lanes[lane].arrived_seq;
        // A reused positive poll skips only duplicate weak reads. Even a stale
        // hint still reaches every authoritative atomic check in lane order.
        if (!query_prefiltered && scheduler_gm_query(value) < seq) return true;
        const uint64_t observed = scheduler_gm_fetch_add(value, UINT64_C(0));
        if (record) observations[lane] = observed;
        if (observed < seq) return true;
        if (observed != seq)
            return scheduler_admission_fail(
                graph, local, run, local->admission->current_task_ref, SchedulerErrorSite::GROUP_ACK_INVALID
            );
    }
    *all = true;
    if (record) scheduler_protocol_group_acks(base, local, releases, observations);
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ void scheduler_gm_mix_publish_release(__gm__ void *base, SchedulerLocalState *local) {
    auto &group = local->admission->gm_mix;
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, true, true);
    scheduler_cache_barrier();
    scheduler_gm_store(
        scheduler_gm_mix_mailbox(base, local)->lanes[local->config.self_lane].released_seq, group.group_seq
    );
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, true, false);
    group.phase = SchedulerGmMixPhase::RELEASED;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_prepare_lane(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, bool query_prefiltered = false
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (group.phase != SchedulerGmMixPhase::ARRIVED) return true;
    bool all = false;
    if (!scheduler_gm_mix_check_acks<RecordProtocol>(graph, base, local, run, false, &all, query_prefiltered))
        return false;
    if (group.subtask_slot == UINT8_MAX) {
        if (!all) return true;
        scheduler_gm_mix_publish_release<RecordProtocol>(base, local);
        *progress = true;
        return true;
    }
    if (group.group_seq == 0 || group.group_seq == UINT64_MAX || group.task_last_seq != group.group_seq ||
        group.block_idx >= admission.current_block_num || group.subtask_slot != admission.current_single_subtask)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    const bool retained = group.slot_index != UINT8_MAX;
    if (!retained) {
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
            if (local->slots[index].state != SchedulerDispatchSlotState::FREE) continue;
            group.slot_index = static_cast<uint8_t>(index);
            break;
        }
        if (group.slot_index == UINT8_MAX) return true;
    }
    if (group.slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    auto &slot = local->slots[group.slot_index];
    SchedulerReadyClaim ready{};
    ready.task_ref = admission.current_task_ref;
    ready.source = SchedulerReadySource::COOPERATIVE;
    const SchedulerFreeSlotClaim claim{group.slot_index};
    SchedulerTaskMetadata metadata{};
    if (!retained) {
        slot.task_ref = admission.current_task_ref;
        slot.block_idx = group.block_idx;
        slot.args_observed = true;
        slot.gm_mix = true;
        slot.subtask_slot = group.subtask_slot;
        slot.state = SchedulerDispatchSlotState::FILLING;
        *progress = true;
    } else {
        uint64_t execution_index = UINT64_MAX;
        if (slot.state != SchedulerDispatchSlotState::FILLING || !slot.gm_mix ||
            slot.task_ref != admission.current_task_ref || slot.block_idx != group.block_idx ||
            slot.subtask_slot != group.subtask_slot ||
            !scheduler_prepared_execution_index(
                base, local->config.prepared, slot.task_ref, slot.block_idx, slot.subtask_slot, &execution_index
            ) ||
            slot.prepared.execution_index != execution_index ||
            (slot.prepared.claimed_state != 0 &&
             (scheduler_prepared_phase(slot.prepared.claimed_state) != SchedulerPreparedPhase::CLAIMED ||
              scheduler_prepared_execute_owner(slot.prepared.claimed_state) != local->worker_id())))
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
            );
    }
    if (slot.prepared.claimed_state == 0) {
        metadata = scheduler_load_dispatch_metadata(base, local, ready.task_ref, profiling_level);
        const auto prepared = scheduler_prepare_dispatch_slot(
            graph, base, local, run, claim, ready, metadata, profiling_level, group.subtask_slot,
            &group.prepare_start_cycles
        );
        if (prepared == PreparedAccessResult::WAIT) return true;
        if (prepared != PreparedAccessResult::READY)
            return scheduler_admission_fail(
                graph, local, run, ready.task_ref, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
        *progress = true;
    }
    // This group owns the FILLING slot while peers arrive. A later service
    // resumes the handle; it never repeats Call acquire or Binding writes.
    if (!all) return true;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    // The lane has not released its group, so the observed descriptor remains
    // immutable. Match its dynamic identity before turning the retained slot Ready.
    const auto *descriptor = &scheduler_gm_mix_mailbox(base, local)->descriptor;
    if (!scheduler_gm_mix_descriptor_valid(local, descriptor, group.group_seq) ||
        descriptor->block_idx != group.block_idx)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
        );
    if (retained) metadata = scheduler_load_dispatch_metadata(base, local, ready.task_ref, profiling_level);
    scheduler_publish_dispatch_slot(
        base, local, claim, ready, metadata, profiling_level, group.subtask_slot, group.prepare_start_cycles
    );
    group.phase = SchedulerGmMixPhase::RUNNING;
    *progress = true;
    return true;
}

// The completion snapshot names the released execution. Slot retirement and
// trace publication precede this release operation.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_gm_mix_release_lane(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerLocalSlotState &completed, uint32_t slot_index
) {
    if (!scheduler_gm_mix_context_valid<false, true>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    const int64_t task_ref = completed.task_ref;
    if (group.phase != SchedulerGmMixPhase::RUNNING || task_ref != admission.current_task_ref ||
        slot_index >= SCHEDULER_PENDING_SLOT_COUNT || group.slot_index != slot_index ||
        completed.state != SchedulerDispatchSlotState::EXECUTING || !scheduler_slot_uses_admission(local, completed) ||
        !completed.gm_mix || group.block_idx != completed.block_idx || group.subtask_slot != completed.subtask_slot ||
        group.group_seq == 0 || local->slots[slot_index].state != SchedulerDispatchSlotState::FREE)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_gm_mix_publish_release<RecordProtocol>(base, local);
    return true;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ SchedulerGmMixResult scheduler_gm_mix_retire(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool query_prefiltered = false
) {
    // Lane completion owns the task counter. Retirement releases only the
    // retained group descriptor after every resource lane has returned it.
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run))
        return SchedulerGmMixResult::ERROR;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::RELEASED) return SchedulerGmMixResult::WAIT;
    bool all = false;
    if (!scheduler_gm_mix_check_acks<RecordProtocol>(graph, base, local, run, true, &all, query_prefiltered))
        return SchedulerGmMixResult::ERROR;
    if (!all) return SchedulerGmMixResult::WAIT;
    if constexpr (RecordProtocol) {
        scheduler_protocol_block_complete(base, local, group.block_idx, true, UINT64_MAX);
        scheduler_protocol_block_complete(base, local, group.block_idx, false, UINT64_MAX);
    }
    group.phase = SchedulerGmMixPhase::IDLE;
    group.slot_index = UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.subtask_slot = UINT8_MAX;
    return SchedulerGmMixResult::PROGRESS;
}
