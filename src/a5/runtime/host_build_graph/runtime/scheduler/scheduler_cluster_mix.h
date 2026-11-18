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

#include "scheduler_completion_credit.h"
#include "scheduler_order.h"

inline __aicore__ bool scheduler_cluster_mix_armed(uint32_t generation) { return (generation & 1U) != 0; }

inline __aicore__ bool
scheduler_cluster_mix_all(const SCHEDULER_CLUSTER_SSBUF volatile uint32_t *acks, LaneMask mask, uint32_t generation) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarClusterMixAll);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        if (mask.has(static_cast<SubtaskSlot>(lane)) && scheduler_cluster_load_field(&acks[lane]) != generation)
            return false;
    scheduler_cluster_barrier();
    return true;
}

static inline __aicore__ __gm__ SchedulerProtocolBlockTrace *
scheduler_protocol_cluster_block_at(__gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return nullptr;
    const auto *metadata = scheduler_task_metadata_at(base, local, task);
    const uint64_t index =
        metadata->trace_index_base + static_cast<uint64_t>(block) * scheduler_task_lanes(*metadata).count();
    if (index >= layout->trace_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolBlockTrace>(base, layout->blocks_offset) + index;
}

static inline __aicore__ void scheduler_protocol_cluster_arm(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, uint32_t generation,
    uint32_t rank, uint64_t publish_before, uint64_t publish_after
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_cluster_block_at(base, local, task, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    scheduler_protocol_observe_record(owner);
    if (owner->publish_before.sequence != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return;
    }
    if (owner->claim_after.sequence == 0 || owner->run_epoch != layout->run_epoch || owner->task_ref != task ||
        owner->block_idx != block || owner->order_key != rank || owner->worker_index != local->worker_id() ||
        owner->cluster_index != local->config.shared_context->cluster_index) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    owner->group_seq = generation;
    scheduler_protocol_stamp_at(base, local, layout, &owner->publish_before, publish_before);
    scheduler_protocol_stamp_at(base, local, layout, &owner->publish_after, publish_after);
    scheduler_protocol_publish_record(owner);
}

static inline __aicore__ void scheduler_protocol_cluster_observe(__gm__ void *base, const SchedulerLocalState *local) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->cluster_mix;
    auto *record = scheduler_protocol_cluster_block_at(base, local, group.task_ref, group.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    scheduler_protocol_observe_record(lane);
    if (lane->descriptor_ready.sequence != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return;
    }
    lane->run_epoch = layout->run_epoch;
    lane->task_ref = group.task_ref;
    lane->order_key = group.rank;
    lane->group_seq = group.generation;
    lane->worker_index = static_cast<uint32_t>(local->worker_id());
    lane->block_idx = group.block_idx;
    lane->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    lane->lane = local->config.self_lane;
    const auto &metadata = *scheduler_task_metadata_at(base, local, group.task_ref);
    lane->logical_block_num = metadata.logical_block_num;
    lane->flags = metadata.flags;
    scheduler_lane_mask_store(lane->active_mask, group.active_mask);
    scheduler_lane_mask_store(lane->resource_mask, group.active_mask);
    for (uint32_t peer = 0; peer < 3; ++peer) {
        lane->lane_subtasks[peer] = group.active_mask.has(static_cast<SubtaskSlot>(peer)) ? peer : UINT8_MAX;
    }
    scheduler_protocol_stamp(base, local, layout, &lane->descriptor_ready);
    scheduler_protocol_publish_record(lane);
}

static inline __aicore__ void
scheduler_protocol_cluster_ack(__gm__ void *base, const SchedulerLocalState *local, bool release, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->cluster_mix;
    auto *record = scheduler_protocol_cluster_block_at(base, local, group.task_ref, group.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    if (lane->descriptor_ready.sequence == 0 || lane->group_seq != group.generation ||
        (release && lane->arrivals_observed.sequence == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!release && before) {
        lane->observed_self_floor = local->observed_self_floor;
        scheduler_writeback_cache_line(&lane->observed_self_floor);
    }
    auto *stamp = release ? (before ? &lane->release_before : &lane->release_after) :
                            (before ? &lane->arrive_before : &lane->arrive_after);
    scheduler_protocol_stamp(base, local, layout, stamp);
    scheduler_publish_cache_line(lane);
}

static inline __aicore__ void
scheduler_protocol_cluster_all(__gm__ void *base, const SchedulerLocalState *local, bool releases) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->cluster_mix;
    auto *record = scheduler_protocol_cluster_block_at(base, local, group.task_ref, group.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    auto *stamp = releases ? &record->owner.releases_observed : &lane->arrivals_observed;
    if (stamp->sequence != 0) return;
    auto *values = releases ? record->owner.released_seq : lane->arrived_seq;
    const auto resources = group.active_mask;
    for (uint32_t peer = 0; peer < 3; ++peer)
        values[peer] = resources.has(static_cast<SubtaskSlot>(peer)) ? group.generation : 0;
    scheduler_protocol_stamp(base, local, layout, stamp);
    if (releases) {
        record->owner.completion_worker = local->worker_id();
        scheduler_protocol_publish_record(&record->owner);
    } else {
        scheduler_protocol_publish_record(lane);
    }
}

static inline __aicore__ bool scheduler_cluster_mix_claim_head(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixDiscoverTask, task_ref);
    if (local == nullptr || progress == nullptr || base == nullptr || run == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr ||
        !scheduler_task_ref_is_null(local->cluster_pending.task_ref))
        return true;
    if (!scheduler_task_ref_valid(graph, task_ref) || local->config.prepared == nullptr)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const uint32_t blocks = metadata.logical_block_num;
    const LaneMask active = scheduler_task_lanes(metadata);
    const bool sync = scheduler_task_requires_sync_start(metadata.flags);
    if (!scheduler_task_is_mix(metadata.flags) || scheduler_task_has_predicate(metadata.flags) || !active.is_mix() ||
        blocks == 0 || (sync && blocks > local->config.cluster_count))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if ((sync && local->last_sync_mix_ref == task_ref) ||
        (local->issue_task_ref == task_ref && local->issue_ticket_exhausted))
        return true;
    auto *signal = local->cluster_signal;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    if (scheduler_cluster_mix_armed(generation) && scheduler_cluster_load_field(&signal->task_ref) == task_ref)
        return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto &order = scheduler_task_catalog_order(base, pool, static_cast<uint32_t>(task_ref));
    if (order.rank < scheduler_cluster_load_field(&signal->mix_floor))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
    auto &pending = local->cluster_pending;
    const bool trace = scheduler_protocol_enabled(local);
    const uint64_t before = trace ? scheduler_protocol_cycles(local) : 0;
    uint32_t block = blocks;
    bool closes = false;
    if (!scheduler_claim_ordinary_block(graph, base, local, run, task_ref, blocks, &block, &closes)) return false;
    scheduler_invalidate_issue_snapshot(snapshot);
    if (block >= blocks) {
        if (local->issue_task_ref == task_ref) local->issue_ticket_exhausted = true;
        return true;
    }
    if (trace)
        scheduler_protocol_block_claim(
            base, local, task_ref, block, before, scheduler_protocol_cycles(local), closes ? blocks : 0
        );
    pending.task_ref = task_ref;
    pending.block_idx = block;
    pending.rank = order.rank;
    if (sync) local->last_sync_mix_ref = task_ref;
    if (!scheduler_reserve_group_execution(
            base, local->config.prepared, local->worker_id(), task_ref, block, &pending.anchor
        ))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if (!scheduler_close_ordinary_supply(graph, base, local, run, task_ref, closes, blocks, block)) return false;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_discover_next(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixDiscoverNext);
    if (local == nullptr || progress == nullptr || base == nullptr || run == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr ||
        !scheduler_task_ref_is_null(local->cluster_pending.task_ref))
        return true;
    int64_t ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &ref, progress, snapshot)) return false;
    if (scheduler_task_ref_is_null(ref) || !scheduler_task_is_mix(scheduler_task_metadata_at(base, local, ref)->flags))
        return true;
    bool acquired = false;
    if (!scheduler_cluster_mix_claim_head(graph, base, local, run, ref, &acquired, snapshot)) return false;
    *progress = *progress || acquired;
    return true;
}

// The accountant has received FIN for every active lane, completed its credit
// and stopped touching the descriptor when its receipt reaches the proxy.
inline __aicore__ bool scheduler_cluster_mix_recycle(SchedulerLocalState *local) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixRecycle);
    auto *signal = local->cluster_signal;
    const uint32_t generation = scheduler_cluster_load_field(&signal->armed_generation);
    if (!scheduler_cluster_mix_armed(generation) ||
        scheduler_cluster_load_field(&signal->accounted_generation) != generation)
        return false;
    scheduler_cluster_barrier();
    scheduler_cluster_store_release(&signal->armed_generation, generation + 1);
    // An unnamed observer may still be sampling the old odd generation.
    // Publish invalidation before any new payload field can be overwritten.
    scheduler_cluster_barrier();
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_progress_pending(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixPending);
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr) return true;
    *progress = scheduler_cluster_mix_recycle(local);
    auto &pending = local->cluster_pending;
    if (scheduler_task_ref_is_null(pending.task_ref)) return true;
    auto *signal = local->cluster_signal;
    const uint32_t previous = scheduler_cluster_load_field(&signal->armed_generation);
    if (scheduler_cluster_mix_armed(previous)) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (previous >= UINT32_MAX - 1 || pending.rank < scheduler_cluster_load_field(&signal->mix_floor) ||
        scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch ||
        !scheduler_task_ref_valid(graph, pending.task_ref) ||
        !scheduler_arm_group_execution(base, local->config.prepared, local->worker_id(), &pending.anchor))
        return scheduler_admission_fail(graph, local, run, pending.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const uint32_t generation = previous + 1;
    const auto active = scheduler_task_lanes(*scheduler_task_metadata_at(base, local, pending.task_ref));
    const bool trace = scheduler_protocol_enabled(local);
    const uint64_t before = trace ? scheduler_protocol_cycles(local) : 0;
    scheduler_cluster_store_field(&signal->task_ref, pending.task_ref);
    scheduler_cluster_store_field(&signal->rank, pending.rank);
    scheduler_cluster_store_field(&signal->block_idx, pending.block_idx);
    scheduler_cluster_store_field(&signal->active_mask, static_cast<uint32_t>(active.raw()));
    scheduler_cluster_store_field(&signal->anchor_state, pending.anchor.claimed_state);
    scheduler_cluster_store_release(&signal->armed_generation, generation);
    if (trace) {
        scheduler_protocol_cluster_arm(
            base, local, pending.task_ref, pending.block_idx, generation, pending.rank, before,
            scheduler_protocol_cycles(local)
        );
        scheduler_cluster_store_release(&signal->trace_generation, generation);
    }
    pending = {};
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_observe(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixObserve);
    auto &group = local->cluster_mix;
    const auto *signal = local->cluster_signal;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    if (!scheduler_cluster_mix_armed(generation) || generation == group.generation) return true;
    const auto lane = local->config.self_lane;
    const uint32_t bits = scheduler_cluster_load_field(&signal->active_mask);
    if ((bits & (1u << lane)) == 0) {
        if (scheduler_cluster_load_acquire(&signal->armed_generation) != generation) return true;
        if (bits > 7 || (bits & (bits - 1)) == 0)
            return scheduler_admission_fail(
                graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
            );
        group.generation = generation;
        return true;
    }
    if (scheduler_protocol_enabled(local) && scheduler_cluster_load_acquire(&signal->trace_generation) != generation)
        return true;
    const auto task = scheduler_cluster_load_field(&signal->task_ref);
    const uint32_t rank = scheduler_cluster_load_field(&signal->rank);
    const uint32_t block = scheduler_cluster_load_field(&signal->block_idx);
    const uint64_t anchor = scheduler_cluster_load_field(&signal->anchor_state);
    if (scheduler_cluster_load_acquire(&signal->armed_generation) != generation) return true;
    if (scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch ||
        !scheduler_task_ref_valid(graph, task) || group.phase != SchedulerGmMixPhase::IDLE ||
        anchor != scheduler_prepared_state(SchedulerPreparedPhase::GROUP_ARMED, local->config.worker_ids[1]))
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task);
    const auto active = scheduler_task_lanes(metadata);
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    if (!scheduler_task_is_mix(metadata.flags) || scheduler_task_has_predicate(metadata.flags) || !active.is_mix() ||
        active.raw() != bits || block >= metadata.logical_block_num ||
        scheduler_task_catalog_order(base, pool, static_cast<uint32_t>(task)).rank != rank)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    group = {};
    group.task_ref = task;
    group.rank = rank;
    group.anchor_state = anchor;
    group.block_idx = block;
    group.generation = generation;
    group.active_mask = active;
    group.subtask_slot = lane;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    if (lane == scheduler_mix_accountant_lane(active) &&
        scheduler_profiled_node_fanin_view(graph, task, &group.fanin) != SchedulerGraphResult::OK)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_protocol_cluster_observe(base, local);
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_dependencies(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixDependencies);
    auto &group = local->cluster_mix;
    if (local->config.self_lane != scheduler_mix_accountant_lane(group.active_mask) || group.dependencies_published)
        return true;
    bool ready = false;
    const auto previous = group.next_fanin;
    if (!scheduler_observe_fanin_prefix(
            graph, base, local, run, group.task_ref, group.fanin, group.next_fanin, &ready, &group.wait
        ))
        return false;
    *progress = *progress || previous != group.next_fanin;
    if (!ready) return true;
    scheduler_cluster_store_release(&local->cluster_signal->dependencies_generation, group.generation);
    group.dependencies_published = true;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_account(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixAccount);
    auto &group = local->cluster_mix;
    if (group.phase != SchedulerGmMixPhase::RELEASED ||
        local->config.self_lane != scheduler_mix_accountant_lane(group.active_mask))
        return true;
    auto *signal = local->cluster_signal;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (!group.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
        const uint32_t finished = scheduler_cluster_load_field(&signal->finished[lane]);
        if (finished < group.generation) return true;
        if (finished != group.generation)
            return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_ACK_INVALID);
    }
    scheduler_cluster_barrier();
    if (group.completion_submitted || scheduler_cluster_load_field(&signal->armed_generation) != group.generation)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_ACK_INVALID);
    scheduler_protocol_cluster_all(base, local, true);
    group.completion_submitted = true;
    if (!scheduler_contribute_mix_group(graph, base, local, run, group.task_ref, group.block_idx, group.active_mask))
        return false;
    group.phase = SchedulerGmMixPhase::IDLE;
    scheduler_cluster_store_release(&signal->accounted_generation, group.generation);
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_prepare_member(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixPrepareMember);
    auto &group = local->cluster_mix;
    auto &owner = local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION);
    const uint32_t lane = group.subtask_slot;
    if (owner.state == SchedulerDispatchSlotState::FREE) {
        const auto binding_error = scheduler_observe_graph_arguments(graph, group.task_ref);
        if (binding_error != SchedulerGraphBindingError::NONE)
            return scheduler_admission_fail(
                graph, local, run, group.task_ref, scheduler_graph_binding_error_site(binding_error)
            );
        owner.task_ref = group.task_ref;
        owner.block_idx = group.block_idx;
        owner.subtask_slot = static_cast<uint8_t>(lane);
        owner.cluster_mix = true;
        owner.rank = group.rank;
        owner.state = SchedulerDispatchSlotState::FILLING;
        group.prepare_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        if (!scheduler_acquire_group_lane(
                base, local->config.prepared, local->worker_id(), local->config.worker_ids[1], group.task_ref,
                group.block_idx, static_cast<uint8_t>(lane), group.anchor_state, &owner.prepared
            ))
            return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        const auto metadata = scheduler_load_dispatch_metadata(base, local, group.task_ref, profiling_level);
        owner.sync_start = scheduler_task_requires_sync_start(metadata.flags);
        if (!scheduler_task_is_executable(metadata.flags) || !scheduler_task_is_mix(metadata.flags) ||
            !scheduler_task_lanes(metadata).has(static_cast<SubtaskSlot>(lane)) ||
            scheduler_metadata_core_type_index(lane) != local->config.self_core_type)
            return scheduler_admission_fail(
                graph, local, run, group.task_ref, SchedulerErrorSite::DISPATCH_INVALID_SHAPE
            );
        *progress = true;
    }
    if (!owner.cluster_mix || owner.state != SchedulerDispatchSlotState::FILLING || owner.task_ref != group.task_ref ||
        owner.block_idx != group.block_idx || owner.subtask_slot != lane || owner.prepared.claimed_state == 0)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    return true;
}

static inline __aicore__ __attribute__((noinline)) bool scheduler_cluster_mix_service_active(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixServiceActive);
    *progress = false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    auto &group = local->cluster_mix;
    if (group.phase == SchedulerGmMixPhase::RELEASED) {
        if (!scheduler_cluster_mix_account(graph, base, local, run, progress)) return false;
        if (group.phase == SchedulerGmMixPhase::RELEASED) return true;
    }
    if (group.phase == SchedulerGmMixPhase::IDLE && !scheduler_cluster_mix_observe(graph, base, local, run, progress))
        return false;
    if (group.phase != SchedulerGmMixPhase::PUBLISHED && group.phase != SchedulerGmMixPhase::ARRIVED) return true;
    auto *signal = local->cluster_signal;
    if (group.phase == SchedulerGmMixPhase::PUBLISHED) {
        if (!scheduler_cluster_mix_prepare_member(graph, base, local, run, profiling_level, progress) ||
            !scheduler_cluster_mix_dependencies(graph, base, local, run, progress))
            return false;
        if (scheduler_cluster_load_field(&signal->dependencies_generation) != group.generation) return true;
        scheduler_cluster_barrier();
        bool permit = false;
        if (!scheduler_mix_lane_order_ready(graph, base, local, run, group.task_ref, &permit)) return false;
        if (!permit) return true;
        if (scheduler_task_requires_sync_start(scheduler_task_metadata_at(base, local, group.task_ref)->flags))
            scheduler_protocol_member(base, local, group.task_ref, false);
        scheduler_protocol_cluster_ack(base, local, false, true);
        scheduler_cluster_store_release(&signal->arrived[group.subtask_slot], group.generation);
        scheduler_protocol_cluster_ack(base, local, false, false);
        group.phase = SchedulerGmMixPhase::ARRIVED;
        *progress = true;
    }
    if (!scheduler_cluster_mix_all(signal->arrived, group.active_mask, group.generation)) return true;
    scheduler_protocol_cluster_all(base, local, false);
    auto &owner = local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION);
    owner.predicate = SchedulerPredicateState::PASS;
    SchedulerReadyClaim ready{};
    ready.task_ref = group.task_ref;
    ready.source = SchedulerReadySource::COOPERATIVE;
    const auto metadata = scheduler_load_dispatch_metadata(base, local, group.task_ref, profiling_level);
    scheduler_publish_dispatch_slot(
        base, local, SCHEDULER_CLUSTER_MIX_EXECUTION, ready, metadata, profiling_level, group.subtask_slot,
        group.prepare_start_cycles
    );
    group.phase = SchedulerGmMixPhase::RUNNING;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_service(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixService);
    *progress = false;
    if (local->cluster_signal == nullptr) return true;
    const auto &group = local->cluster_mix;
    if (group.phase == SchedulerGmMixPhase::RUNNING) return true;
    if (group.phase == SchedulerGmMixPhase::IDLE) {
        const uint32_t generation = scheduler_cluster_load_field(&local->cluster_signal->armed_generation);
        if (!scheduler_cluster_mix_armed(generation) || generation == group.generation) return true;
    }
    return scheduler_cluster_mix_service_active(graph, base, local, run, profiling_level, progress);
}

static inline __aicore__ bool scheduler_cluster_mix_can_enter(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, bool *permit
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixEnter);
    *permit = false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &group = local->cluster_mix;
    const auto &owned = local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION);
    const auto *signal = local->cluster_signal;
    if (signal == nullptr || group.phase != SchedulerGmMixPhase::RUNNING || !owned.cluster_mix ||
        owned.state != SchedulerDispatchSlotState::READY || owned.task_ref != group.task_ref ||
        owned.block_idx != group.block_idx || owned.subtask_slot != group.subtask_slot ||
        scheduler_cluster_load_acquire(&signal->armed_generation) != group.generation ||
        scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    *permit = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_release(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerLocalSlotState &completed, uint32_t slot_index, bool successful = true
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixRelease, completed.task_ref);
    auto &group = local->cluster_mix;
    auto *signal = local->cluster_signal;
    if (signal == nullptr || group.phase != SchedulerGmMixPhase::RUNNING ||
        slot_index != SCHEDULER_CLUSTER_MIX_EXECUTION ||
        &completed != &local->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION) || !completed.cluster_mix ||
        completed.task_ref != group.task_ref || completed.block_idx != group.block_idx ||
        completed.subtask_slot != group.subtask_slot || completed.state != SchedulerDispatchSlotState::EXECUTING ||
        completed.prepared.claimed_state != 0)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_protocol_cluster_ack(base, local, true, true);
    scheduler_cluster_store_release(&signal->finished[group.subtask_slot], group.generation + (successful ? 0 : 1));
    scheduler_protocol_cluster_ack(base, local, true, false);
    if (scheduler_task_requires_sync_start(scheduler_task_metadata_at(base, local, group.task_ref)->flags))
        scheduler_protocol_member(base, local, group.task_ref, true);
    if (group.subtask_slot != scheduler_mix_accountant_lane(group.active_mask)) {
        group.phase = SchedulerGmMixPhase::IDLE;
        return true;
    }
    group.phase = SchedulerGmMixPhase::RELEASED;
    if (!successful) return true;
    bool accounted = false;
    return scheduler_cluster_mix_account(graph, base, local, run, &accounted);
}

static inline __aicore__ bool scheduler_cluster_mix_drain(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *done
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixDrain);
    if (!scheduler_task_ref_is_null(local->cluster_pending.task_ref))
        return scheduler_admission_fail(
            graph, local, run, local->cluster_pending.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    *done = false;
    if (local->cluster_signal == nullptr) {
        *done = true;
        return true;
    }
    bool advanced = false;
    if (!scheduler_cluster_mix_account(graph, base, local, run, &advanced)) return false;
    const auto &group = local->cluster_mix;
    if (group.phase != SchedulerGmMixPhase::IDLE) {
        if (group.phase == SchedulerGmMixPhase::RELEASED) return true;
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT);
    }
    if (local->is_mix_coordinator()) scheduler_cluster_mix_recycle(local);
    *done = local->is_mix_coordinator() ?
                !scheduler_cluster_mix_armed(scheduler_cluster_load_field(&local->cluster_signal->armed_generation)) :
                !scheduler_cluster_owes_before(local, UINT32_MAX);
    return true;
}
