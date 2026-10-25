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

enum class SchedulerGmMixResult : uint8_t { WAIT, PROGRESS, TASK_LAST, ERROR };

inline __aicore__ bool scheduler_gm_mix_context_valid(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    auto *admission = local == nullptr ? nullptr : local->admission;
    if (!scheduler_admission_context_valid(graph, base, local, run, admission)) return false;
    if (!scheduler_task_ref_valid(graph, admission->current_task_ref) ||
        !scheduler_task_is_mix(admission->current_flags) ||
        scheduler_task_requires_sync_start(admission->current_flags) || admission->gm_mix.mailbox_offset == 0 ||
        admission->gm_mix.cluster_index >= local->config.cluster_count ||
        local->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return scheduler_admission_fail(
            graph, local, run, admission->current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    return true;
}

inline __aicore__ __gm__ SchedulerGroupMailbox *
scheduler_gm_mix_mailbox(__gm__ void *base, const SchedulerLocalState *local) {
    return scheduler_state_at<SchedulerGroupMailbox>(base, local->admission->gm_mix.mailbox_offset);
}

inline __aicore__ bool scheduler_gm_mix_descriptor_valid(
    const SchedulerLocalState *local, __gm__ const SchedulerGroupDescriptor *descriptor, uint64_t seq
) {
    const auto &admission = *local->admission;
    if (descriptor->run_epoch != admission.epoch || descriptor->task_ref != admission.current_task_ref ||
        descriptor->order_key != admission.scan_cursor || descriptor->group_seq != seq ||
        descriptor->block_idx >= admission.current_block_num ||
        descriptor->logical_block_num != admission.current_block_num ||
        descriptor->cluster_index != admission.gm_mix.cluster_index ||
        descriptor->active_mask != admission.current_active_mask || descriptor->resource_mask != 7 ||
        descriptor->flags != admission.current_flags)
        return false;
    uint8_t mapped = 0;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        const uint8_t subtask = descriptor->lane_subtasks[lane];
        if (subtask == UINT8_MAX) continue;
        if (subtask >= 3 || lane != subtask || (mapped & (1U << subtask)) != 0) return false;
        mapped |= 1U << subtask;
    }
    return mapped == descriptor->active_mask;
}

inline __aicore__ bool scheduler_gm_mix_close(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *closed, bool critical = false
) {
    if (closed == nullptr) return false;
    *closed = false;
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    const auto *record = scheduler_admission_record_at(base, &admission, admission.current_task_ref);
    auto *seals = scheduler_state_at<uint64_t>(base, record->mix_close_seals_offset);
    auto &seal = seals[group.cluster_index];
    uint64_t closed_cursor = UINT64_MAX;
    uint64_t close_seal = 0;
    if (local->is_mix_coordinator()) {
        if (group.phase != SchedulerGmMixPhase::IDLE) return true;
        const uint64_t cursor = scheduler_gm_fetch_add(
            scheduler_task_control_at(base, local, admission.current_task_ref)->next_block, UINT64_C(0)
        );
        if (cursor < admission.current_block_num) return true;
        if (cursor != admission.current_block_num || group.task_last_seq == UINT64_MAX)
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
    scheduler_protocol_member_closed(base, local, &admission, closed_cursor, close_seal, group.task_last_seq);
    ++admission.scan_cursor;
    admission.current_task_ref = SCHEDULER_TASK_REF_INVALID;
    admission.current_block_num = 0;
    admission.current_active_mask = 0;
    admission.current_flags = 0;
    group.phase = SchedulerGmMixPhase::IDLE;
    group.task_last_seq = 0;
    group.slot_index = UINT8_MAX;
    group.subtask_slot = UINT8_MAX;
    group.slot_generation = 0;
    *closed = true;
    return true;
}

inline __aicore__ SchedulerBlockClaimResult scheduler_gm_mix_claim(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return SchedulerBlockClaimResult::ERROR;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator()) {
        scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
        return SchedulerBlockClaimResult::ERROR;
    }
    if (group.phase != SchedulerGmMixPhase::IDLE) return SchedulerBlockClaimResult::RETRY;
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
    const auto result = scheduler_claim_ordered_cursor(graph, base, local, run, &block);
    if (result != SchedulerBlockClaimResult::CLAIMED) return result;
    group.block_idx = block;
    ++group.group_seq;
    group.phase = SchedulerGmMixPhase::CLAIMED;
    group.slot_index = UINT8_MAX;
    group.slot_generation = 0;
    group.subtask_slot = UINT8_MAX;
    return SchedulerBlockClaimResult::CLAIMED;
}

inline __aicore__ bool scheduler_gm_mix_publish(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::CLAIMED || group.group_seq == 0 ||
        group.group_seq == UINT64_MAX || group.block_idx >= admission.current_block_num)
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    auto *descriptor = &mailbox->descriptor;
    descriptor->run_epoch = admission.epoch;
    descriptor->task_ref = admission.current_task_ref;
    descriptor->order_key = admission.scan_cursor;
    descriptor->group_seq = group.group_seq;
    descriptor->block_idx = group.block_idx;
    descriptor->logical_block_num = admission.current_block_num;
    descriptor->cluster_index = group.cluster_index;
    descriptor->active_mask = admission.current_active_mask;
    descriptor->resource_mask = 7;
    descriptor->flags = admission.current_flags;
    // Mix lane identities also select physical communication endpoints.
    // Inactive peers still participate in the cluster resource handshake.
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        descriptor->lane_subtasks[lane] = (admission.current_active_mask & (1U << lane)) != 0 ? lane : UINT8_MAX;
    group.subtask_slot = descriptor->lane_subtasks[0];
    for (uint32_t offset = 0; offset < sizeof(*descriptor); offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptor) + offset);
    scheduler_cache_barrier();
    scheduler_gm_store(mailbox->published_seq, group.group_seq);
    group.task_last_seq = group.group_seq;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    return true;
}

inline __aicore__ bool scheduler_gm_mix_observe(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (local->is_mix_coordinator() ||
        (group.phase != SchedulerGmMixPhase::IDLE && group.phase != SchedulerGmMixPhase::RELEASED))
        return true;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    if (scheduler_gm_query(mailbox->published_seq) <= group.group_seq) return true;
    const uint64_t seq = scheduler_gm_fetch_add(mailbox->published_seq, UINT64_C(0));
    if (seq <= group.group_seq) return true;
    auto *descriptor = &mailbox->descriptor;
    for (uint32_t offset = 0; offset < sizeof(*descriptor); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptor) + offset);
    scheduler_cache_barrier();
    if (descriptor->task_ref != admission.current_task_ref) {
        // A released lane may still be consuming A's retained seal while its
        // coordinator has already published B into the reusable mailbox.
        const auto *record = scheduler_admission_record_at(base, &admission, admission.current_task_ref);
        auto *seals = scheduler_state_at<uint64_t>(base, record->mix_close_seals_offset);
        if (scheduler_gm_fetch_add(seals[group.cluster_index], UINT64_C(0)) != 0) {
            bool closed = false;
            if (!scheduler_gm_mix_close(graph, base, local, run, &closed, true)) return false;
            if (closed) {
                *progress = true;
                return true;
            }
        }
    }
    if (seq != group.group_seq + 1 || seq == UINT64_MAX || !scheduler_gm_mix_descriptor_valid(local, descriptor, seq))
        return scheduler_admission_fail(
            graph, local, run, admission.current_task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
        );
    group.group_seq = seq;
    group.task_last_seq = seq;
    group.block_idx = descriptor->block_idx;
    group.subtask_slot = descriptor->lane_subtasks[local->config.self_lane];
    group.slot_index = UINT8_MAX;
    group.slot_generation = 0;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    *progress = true;
    return true;
}

inline __aicore__ bool scheduler_gm_mix_arrive(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &group = local->admission->gm_mix;
    if (group.phase != SchedulerGmMixPhase::PUBLISHED) return true;
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE) return true;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    scheduler_cache_barrier();
    scheduler_gm_store(mailbox->lanes[local->config.self_lane].arrived_seq, group.group_seq);
    group.phase = SchedulerGmMixPhase::ARRIVED;
    *progress = true;
    return true;
}

inline __aicore__ bool scheduler_gm_mix_check_acks(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool releases, bool *all
) {
    *all = false;
    auto *mailbox = scheduler_gm_mix_mailbox(base, local);
    const uint64_t seq = local->admission->gm_mix.group_seq;
    for (uint8_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        auto &value = releases ? mailbox->lanes[lane].released_seq : mailbox->lanes[lane].arrived_seq;
        if (scheduler_gm_query(value) < seq) return true;
        const uint64_t observed = scheduler_gm_fetch_add(value, UINT64_C(0));
        if (observed < seq) return true;
        if (observed != seq)
            return scheduler_admission_fail(
                graph, local, run, local->admission->current_task_ref, SchedulerErrorSite::GROUP_ACK_INVALID
            );
    }
    *all = true;
    return true;
}

inline __aicore__ void scheduler_gm_mix_publish_release(__gm__ void *base, SchedulerLocalState *local) {
    auto &group = local->admission->gm_mix;
    scheduler_cache_barrier();
    scheduler_gm_store(
        scheduler_gm_mix_mailbox(base, local)->lanes[local->config.self_lane].released_seq, group.group_seq
    );
    group.phase = SchedulerGmMixPhase::RELEASED;
}

inline __aicore__ bool scheduler_gm_mix_prepare_lane(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    if (progress == nullptr) return false;
    *progress = false;
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (group.phase != SchedulerGmMixPhase::ARRIVED) return true;
    bool all = false;
    if (!scheduler_gm_mix_check_acks(graph, base, local, run, false, &all)) return false;
    if (!all) return true;
    if (group.subtask_slot == UINT8_MAX) {
        scheduler_gm_mix_publish_release(base, local);
        *progress = true;
        return true;
    }
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        auto &slot = local->slots[index];
        if (slot.state != SchedulerDispatchSlotState::FREE) continue;
        slot.task_ref = admission.current_task_ref;
        slot.block_idx = group.block_idx;
        slot.cooperative = true;
        slot.gm_mix = true;
        slot.subtask_slot = group.subtask_slot;
        slot.mix_tracker = UINT8_MAX;
        slot.candidate_index = UINT8_MAX;
        slot.state = SchedulerDispatchSlotState::FILLING;
        SchedulerReadyClaim ready{};
        ready.task_ref = admission.current_task_ref;
        ready.source = SchedulerReadySource::COOPERATIVE;
        const SchedulerFreeSlotClaim claim{local->worker_id(), index, slot.generation, local->config.self_lane};
        const auto metadata = scheduler_load_dispatch_metadata(base, local, ready.task_ref, profiling_level);
        uint64_t prepare_start = 0;
        if (!scheduler_prepare_dispatch_slot(
                graph, base, local, run, claim, ready, metadata, profiling_level, nullptr, group.subtask_slot,
                &prepare_start
            ))
            return scheduler_admission_fail(
                graph, local, run, ready.task_ref, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
        scheduler_publish_dispatch_slot(
            base, local, claim, ready, metadata, profiling_level, nullptr, group.subtask_slot, prepare_start
        );
        group.slot_index = static_cast<uint8_t>(index);
        group.slot_generation = slot.generation;
        group.phase = SchedulerGmMixPhase::RUNNING;
        *progress = true;
        return true;
    }
    return true;
}

// Completion owns the generation check. Both slot retirement and trace
// publication precede this release operation.
inline __aicore__ bool scheduler_gm_mix_release_lane(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint32_t slot_index, uint32_t generation
) {
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return false;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (group.phase != SchedulerGmMixPhase::RUNNING || task_ref != admission.current_task_ref ||
        slot_index >= SCHEDULER_PENDING_SLOT_COUNT || group.slot_index != slot_index ||
        group.slot_generation != generation || generation == 0 ||
        local->slots[slot_index].state != SchedulerDispatchSlotState::FREE)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_gm_mix_publish_release(base, local);
    return true;
}

inline __aicore__ SchedulerGmMixResult scheduler_gm_mix_retire(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    if (!scheduler_gm_mix_context_valid(graph, base, local, run)) return SchedulerGmMixResult::ERROR;
    auto &admission = *local->admission;
    auto &group = admission.gm_mix;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::RELEASED) return SchedulerGmMixResult::WAIT;
    bool all = false;
    if (!scheduler_gm_mix_check_acks(graph, base, local, run, true, &all)) return SchedulerGmMixResult::ERROR;
    if (!all) return SchedulerGmMixResult::WAIT;
    const auto finished = scheduler_contribute_ordered_block(graph, base, local, run);
    if (finished == SchedulerBlockFinishResult::ERROR) return SchedulerGmMixResult::ERROR;
    group.phase = SchedulerGmMixPhase::IDLE;
    group.slot_index = UINT8_MAX;
    group.slot_generation = 0;
    group.subtask_slot = UINT8_MAX;
    return finished == SchedulerBlockFinishResult::TASK_LAST ? SchedulerGmMixResult::TASK_LAST :
                                                               SchedulerGmMixResult::PROGRESS;
}
