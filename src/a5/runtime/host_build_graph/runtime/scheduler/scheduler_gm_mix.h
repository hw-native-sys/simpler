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
#include "scheduler_completion_credit.h"

inline constexpr uint64_t SCHEDULER_GROUP_SEQUENCE_LIMIT = UINT64_C(1) << 63;

inline __aicore__ uint64_t scheduler_group_release_sequence(uint64_t value) {
    return value & ~SCHEDULER_GROUP_SEQUENCE_LIMIT;
}

enum class SchedulerGmMixResult : uint8_t { WAIT, PROGRESS, ERROR };

// One fill invocation owns this observation; no value survives a group transition.
struct SchedulerGmMixAssignmentObservation {
    uint64_t epoch{0};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t order_key{0};
    uint64_t group_seq{0};
    uint64_t assigned_seq{0};
    SchedulerGmMixPhase phase{SchedulerGmMixPhase::IDLE};
};
static_assert(alignof(SchedulerGmMixAssignmentObservation) <= 8);

// StaticContextValidated is confined to one service call whose task, resource
// membership and mailbox binding cannot change before returning. Admission
// observes error and epoch; physical-return release retains epoch and identity
// without granting execution.
template <bool StaticContextValidated = false, bool ReturningLane = false>
static inline __aicore__ bool scheduler_gm_mix_context_valid(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if constexpr (StaticContextValidated && !ReturningLane)
        return scheduler_admission_context_valid(graph, base, local, run, local->admission);
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

enum class SchedulerGmMixSignal : uint8_t { ASSIGNED, ARRIVED, RELEASED };
inline constexpr uint64_t SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT = UINT64_C(1) << 31;

static inline __aicore__ __gm__ volatile uint64_t &
scheduler_gm_mix_signal_word(__gm__ SchedulerGroupMailbox *mailbox, uint32_t lane, SchedulerGmMixSignal signal) {
    if (signal == SchedulerGmMixSignal::ASSIGNED) return mailbox->lanes[lane].assigned_seq;
    if (signal == SchedulerGmMixSignal::ARRIVED) return mailbox->lanes[lane].arrived_seq;
    return mailbox->lanes[lane].released_seq;
}

static inline __aicore__ uint64_t
scheduler_gm_mix_signal_address(__gm__ SchedulerGroupMailbox *mailbox, uint32_t lane, SchedulerGmMixSignal signal) {
    return lane == 0 ? 0 : mailbox->notification_addresses[lane - 1][static_cast<uint8_t>(signal)];
}

// Host admits an entire cluster to this backend only when the run's maximum
// group count fits 31 bits. The release failure bit retains its 64-bit meaning
// to every protocol consumer; descriptor and private sequence identities stay wide.
static inline __aicore__ uint64_t
scheduler_gm_mix_query_signal(__gm__ SchedulerGroupMailbox *mailbox, uint32_t lane, SchedulerGmMixSignal signal) {
    const uint64_t address = scheduler_gm_mix_signal_address(mailbox, lane, signal);
    if (address == 0) return scheduler_gm_query(scheduler_gm_mix_signal_word(mailbox, lane, signal));
    const uint64_t value = scheduler_notify_query(*reinterpret_cast<__gm__ volatile uint32_t *>(address));
    if (signal != SchedulerGmMixSignal::RELEASED) return value;
    return (value & (SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT - 1)) |
           ((value & SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT) != 0 ? SCHEDULER_GROUP_SEQUENCE_LIMIT : 0);
}

static inline __aicore__ void scheduler_gm_mix_store_signal(
    __gm__ SchedulerGroupMailbox *mailbox, uint32_t lane, SchedulerGmMixSignal signal, uint64_t value
) {
    const uint64_t address = scheduler_gm_mix_signal_address(mailbox, lane, signal);
    if (address == 0) {
        scheduler_gm_store(scheduler_gm_mix_signal_word(mailbox, lane, signal), value);
        return;
    }
    const uint32_t encoded = static_cast<uint32_t>(value & (SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT - 1)) |
                             ((value & SCHEDULER_GROUP_SEQUENCE_LIMIT) != 0 ? UINT32_C(1) << 31 : 0);
    scheduler_notify_store(*reinterpret_cast<__gm__ volatile uint32_t *>(address), encoded);
}

// Role and lane mapping were validated when the obligation was adopted. The
// reusable mailbox carries only this group's dynamic identity and lane history.
inline __aicore__ bool
scheduler_gm_mix_descriptor_sequence_valid(__gm__ const SchedulerGroupDescriptor *descriptor, uint64_t seq) {
    if (seq == 0 || seq >= SCHEDULER_GROUP_SEQUENCE_LIMIT || descriptor->group_seq != seq ||
        descriptor->previous_lane_seq[0] != seq - 1)
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
    auto *signal = local->cluster_signal;
    if (signal == nullptr || scheduler_cluster_load_field(&signal->run_epoch) != admission.epoch)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
        );
    uint64_t closed_cursor = UINT64_MAX;
    uint64_t close_frontier = 0;
    bool used_claim_seal = false;
    bool from_claim = false;
    if (local->is_mix_coordinator()) {
        if (group.phase != SchedulerGmMixPhase::IDLE) return true;
        if (scheduler_cluster_load_field(&signal->sync_frontier[0]) != admission.scan_cursor)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        auto *control = scheduler_task_control_at(base, local, admission.current_task_ref);
        from_claim = admission.sync_claimed;
        if (!from_claim && scheduler_gm_query(control->next_block) < admission.current_block_num) return true;
        // The query only postpones closing. An idle claimed sync position has
        // retired its sole group; a zero-work position must confirm exhaustion.
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        uint64_t cursor = UINT64_MAX;
        if (from_claim) {
            if (!scheduler_task_requires_sync_start(admission.current_flags) || admission.current_block_num == 0)
                return scheduler_admission_fail(
                    graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
                );
        } else {
            if (!scheduler_read_claim_cursor(graph, base, local, run, &cursor, &used_claim_seal)) return false;
            if (cursor < admission.current_block_num) return true;
            if (cursor > admission.current_block_num)
                return scheduler_admission_fail(
                    graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
                );
        }
        if (group.task_last_seq >= SCHEDULER_GROUP_SEQUENCE_LIMIT)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        for (const auto &slot : local->slots)
            if (slot.state != SchedulerDispatchSlotState::FREE) return true;
        scheduler_cache_barrier();
        closed_cursor = cursor;
        close_frontier = admission.scan_cursor + 1;
    } else {
        if (!critical && scheduler_cluster_load_field(&signal->sync_frontier[0]) <= admission.scan_cursor) return true;
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        const uint32_t frontier = scheduler_cluster_load_acquire(&signal->sync_frontier[0]);
        if (frontier > admission.capacity)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        if (frontier <= admission.scan_cursor) return true;
        if ((group.phase != SchedulerGmMixPhase::IDLE && group.phase != SchedulerGmMixPhase::RELEASED) ||
            group.task_last_seq >= SCHEDULER_GROUP_SEQUENCE_LIMIT)
            return scheduler_admission_fail(
                graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
            );
        for (const auto &slot : local->slots)
            if (slot.state != SchedulerDispatchSlotState::FREE)
                return scheduler_admission_fail(
                    graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_CLOSE_INVALID
                );
        close_frontier = frontier;
    }
    if (admission.scan_cursor >= admission.capacity)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
        );
    if constexpr (RecordProtocol)
        scheduler_protocol_member_closed(
            base, local, &admission, closed_cursor, close_frontier, group.task_last_seq, used_claim_seal, from_claim
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
    const auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    const uint64_t sequence_limit = mailbox->notification_addresses[0][0] != 0 ? SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT :
                                                                                 SCHEDULER_GROUP_SEQUENCE_LIMIT;
    if (group.group_seq >= sequence_limit - 1) {
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
    group.completion_submitted = false;
    group.arrivals_confirmed = 0;
    group.releases_confirmed = 0;
    group.release_failed = false;
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
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    const uint64_t sequence_limit = mailbox->notification_addresses[0][0] != 0 ? SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT :
                                                                                 SCHEDULER_GROUP_SEQUENCE_LIMIT;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::CLAIMED || group.group_seq == 0 ||
        group.group_seq >= sequence_limit || group.block_idx >= admission.current_block_num)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
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
    for (uint8_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (admission.current_resource_mask.has(static_cast<SubtaskSlot>(lane))) {
            scheduler_gm_mix_store_signal(mailbox, lane, SchedulerGmMixSignal::ASSIGNED, group.group_seq);
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
    bool *progress, SchedulerGmMixAssignmentObservation *observation = nullptr
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
    const bool reusable =
        observation != nullptr && observation->assigned_seq != 0 && observation->epoch == admission.epoch &&
        observation->task_ref == admission.current_task_ref && observation->order_key == admission.scan_cursor &&
        observation->group_seq == group.group_seq && observation->phase == group.phase;
    const uint64_t seq =
        reusable ? observation->assigned_seq :
                   scheduler_gm_mix_query_signal(mailbox, local->config.self_lane, SchedulerGmMixSignal::ASSIGNED);
    if (observation != nullptr) observation->assigned_seq = 0;
    if (seq <= group.group_seq) return true;
    auto *descriptor = &mailbox->descriptor;
    for (uint32_t offset = 0; offset < sizeof(*descriptor); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptor) + offset);
    scheduler_cache_barrier();
    asm volatile("" ::: "memory");
    if (descriptor->task_ref != admission.current_task_ref) {
        // The coordinator's closed admission prefix survives mailbox reuse.
        bool closed = false;
        if (!scheduler_gm_mix_close<RecordProtocol, StaticContextValidated>(graph, base, local, run, &closed, true))
            return false;
        if (closed) {
            *progress = true;
            return true;
        }
    }
    if (seq >= SCHEDULER_GROUP_SEQUENCE_LIMIT ||
        descriptor->previous_lane_seq[local->config.self_lane] != group.group_seq ||
        !scheduler_gm_mix_descriptor_valid(local, descriptor, seq))
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
        );
    group.group_seq = seq;
    group.completion_submitted = false;
    group.arrivals_confirmed = 0;
    group.releases_confirmed = 0;
    group.release_failed = false;
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
    // Only remote lanes publish an ACK; the coordinator already owns its local proof.
    if (!local->is_mix_coordinator())
        scheduler_gm_mix_store_signal(mailbox, local->config.self_lane, SchedulerGmMixSignal::ARRIVED, group.group_seq);
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, false, false);
    group.arrivals_confirmed |= static_cast<uint8_t>(1U << local->config.self_lane);
    group.phase = SchedulerGmMixPhase::ARRIVED;
    *progress = true;
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_gm_mix_check_acks(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool releases, bool *all, bool *successful = nullptr
) {
    *all = false;
    auto &group = local->admission->gm_mix;
    if (successful != nullptr) *successful = !group.release_failed;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    const uint64_t seq = group.group_seq;
    const bool record = RecordProtocol && scheduler_protocol_enabled(local);
    uint64_t observations[PLATFORM_CORES_PER_BLOCKDIM]{};
    const LaneMask resources = local->admission->current_resource_mask;
    auto &confirmed = releases ? group.releases_confirmed : group.arrivals_confirmed;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
        const uint8_t bit = static_cast<uint8_t>(1U << lane);
        if ((confirmed & bit) != 0) {
            if (record) observations[lane] = seq;
            continue;
        }
        const uint64_t raw = scheduler_gm_mix_query_signal(
            mailbox, lane, releases ? SchedulerGmMixSignal::RELEASED : SchedulerGmMixSignal::ARRIVED
        );
        const uint64_t observed = releases ? scheduler_group_release_sequence(raw) : raw;
        if (observed < seq) return true;
        if (observed != seq)
            return scheduler_admission_fail(
                graph, local, run, local->admission->current_task_ref, SchedulerErrorSite::GROUP_ACK_INVALID
            );
        if (releases && (raw & SCHEDULER_GROUP_SEQUENCE_LIMIT) != 0 &&
            local->admission->current_active_mask.has(static_cast<SubtaskSlot>(lane))) {
            group.release_failed = true;
            if (successful != nullptr) *successful = false;
        }
        confirmed |= bit;
        if (record) observations[lane] = observed;
    }
    *all = true;
    if (record) scheduler_protocol_group_acks(base, local, releases, observations);
    return true;
}

// The coordinator certifies the complete arrival set before entering its Call.
// Its epoch-scoped SSBUF sequence stays fixed until every participant releases.
template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_gm_mix_arrivals_ready(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *all
) {
    auto *signal = local->cluster_signal;
    if (signal == nullptr)
        return scheduler_admission_fail(
            graph, local, run, local->admission->current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    const uint64_t seq = local->admission->gm_mix.group_seq;
    if (local->is_mix_coordinator()) {
        if (!scheduler_gm_mix_check_acks<RecordProtocol>(graph, base, local, run, false, all)) return false;
        if (*all) {
            scheduler_cluster_barrier();
            scheduler_cluster_store_field(&signal->gm_arrival_seq, seq);
        }
        return true;
    }
    *all = false;
    const uint64_t observed = scheduler_cluster_load_field(&signal->gm_arrival_seq);
    if (observed < seq) return true;
    scheduler_cluster_barrier();
    if (observed != seq)
        return scheduler_admission_fail(
            graph, local, run, local->admission->current_task_ref, SchedulerErrorSite::GROUP_ACK_INVALID
        );
    *all = true;
    if constexpr (RecordProtocol) {
        uint64_t observations[PLATFORM_CORES_PER_BLOCKDIM]{};
        const auto resources = local->admission->current_resource_mask;
        for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            if (resources.has(static_cast<SubtaskSlot>(lane))) observations[lane] = seq;
        scheduler_protocol_group_acks(base, local, false, observations);
    }
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ void
scheduler_gm_mix_publish_release(__gm__ void *base, SchedulerLocalState *local, bool successful = true) {
    auto &group = local->admission->gm_mix;
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, true, true);
    scheduler_cache_barrier();
    if (!local->is_mix_coordinator())
        scheduler_gm_mix_store_signal(
            scheduler_gm_mix_mailbox(base, local), local->config.self_lane, SchedulerGmMixSignal::RELEASED,
            group.group_seq | (successful ? 0 : SCHEDULER_GROUP_SEQUENCE_LIMIT)
        );
    if constexpr (RecordProtocol) scheduler_protocol_lane_publication(base, local, true, false);
    group.releases_confirmed |= static_cast<uint8_t>(1U << local->config.self_lane);
    if (!successful && local->admission->current_active_mask.has(static_cast<SubtaskSlot>(local->config.self_lane)))
        group.release_failed = true;
    group.phase = SchedulerGmMixPhase::RELEASED;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ bool scheduler_gm_mix_prepare_lane(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (group.phase != SchedulerGmMixPhase::ARRIVED) return true;
    bool all = false;
    if (group.subtask_slot == UINT8_MAX) {
        if (!scheduler_gm_mix_arrivals_ready<RecordProtocol>(graph, base, local, run, &all)) return false;
        if (!all) return true;
        scheduler_gm_mix_publish_release<RecordProtocol>(base, local);
        *progress = true;
        return true;
    }
    if (group.group_seq == 0 || group.group_seq >= SCHEDULER_GROUP_SEQUENCE_LIMIT ||
        group.task_last_seq != group.group_seq || group.block_idx >= admission.current_block_num ||
        group.subtask_slot != admission.current_single_subtask)
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
    SchedulerTaskMetadata metadata{};
    if (!retained) {
        slot.task_ref = admission.current_task_ref;
        slot.block_idx = group.block_idx;
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
        if (!scheduler_prepare_dispatch_slot(
                graph, base, local, run, group.slot_index, ready, metadata, profiling_level, group.subtask_slot,
                &group.prepare_start_cycles
            ))
            return scheduler_admission_fail(
                graph, local, run, ready.task_ref, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
        *progress = true;
    }
    // This group owns the FILLING slot while peers arrive. A later service
    // resumes the handle; it never repeats Call acquire or Binding writes.
    if (!scheduler_gm_mix_arrivals_ready<RecordProtocol>(graph, base, local, run, &all)) return false;
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
        base, local, group.slot_index, ready, metadata, profiling_level, group.subtask_slot, group.prepare_start_cycles
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
    const SchedulerLocalSlotState &completed, uint32_t slot_index, bool successful = true
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
    scheduler_gm_mix_publish_release<RecordProtocol>(base, local, successful);
    return true;
}

template <bool RecordProtocol = true, bool StaticContextValidated = false>
static inline __aicore__ SchedulerGmMixResult scheduler_gm_mix_retire(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    // Returned resources carry separate success evidence for computing lanes.
    if (!scheduler_gm_mix_context_valid<StaticContextValidated>(graph, base, local, run))
        return SchedulerGmMixResult::ERROR;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::RELEASED) return SchedulerGmMixResult::WAIT;
    bool all = false;
    bool successful = true;
    if (!scheduler_gm_mix_check_acks<RecordProtocol>(graph, base, local, run, true, &all, &successful))
        return SchedulerGmMixResult::ERROR;
    if (!all) return SchedulerGmMixResult::WAIT;
    if (!successful || group.completion_submitted) {
        scheduler_admission_fail(graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_ACK_INVALID);
        return SchedulerGmMixResult::ERROR;
    }
    group.completion_submitted = true;
    if (!scheduler_contribute_mix_group<RecordProtocol>(
            graph, base, local, run, admission.current_task_ref, group.block_idx, admission.current_active_mask
        ))
        return SchedulerGmMixResult::ERROR;
    group.phase = SchedulerGmMixPhase::IDLE;
    group.slot_index = UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.subtask_slot = UINT8_MAX;
    return SchedulerGmMixResult::PROGRESS;
}
