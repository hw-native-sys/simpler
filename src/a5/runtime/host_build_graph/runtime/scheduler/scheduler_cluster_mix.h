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

inline __aicore__ bool scheduler_cluster_mix_armed(uint32_t generation) { return (generation & 1U) != 0; }

// Priority only: service still validates the descriptor and all member ACKs.
inline __aicore__ bool scheduler_cluster_mix_needs_service(const SchedulerLocalState *local) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixNeedsService);
    const auto *signal = local->cluster_signal;
    if (signal == nullptr) return false;
    const auto &group = local->cluster_mix;
    if (group.phase == SchedulerGmMixPhase::PUBLISHED || group.phase == SchedulerGmMixPhase::ARRIVED ||
        group.phase == SchedulerGmMixPhase::RUNNING)
        return true;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    if (!scheduler_cluster_mix_armed(generation) || generation == group.generation) return false;
    const bool named = scheduler_cluster_names_lane(signal, local->config.self_lane);
    // An unstable snapshot requests another service pass, never execution permission.
    return scheduler_cluster_load_acquire(&signal->armed_generation) != generation || named;
}

inline __aicore__ bool scheduler_cluster_mix_slots_free(const SchedulerLocalState *local) {
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE) return false;
    return true;
}

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
    uint64_t tail, const uint32_t *frontiers, const uint32_t *holds, uint64_t staged, uint64_t claim_before,
    uint64_t claim_after, uint64_t publish_before, uint64_t publish_after
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_cluster_block_at(base, local, task, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    scheduler_protocol_observe_record(owner);
    if (owner->claim_before.sequence != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return;
    }
    const auto &metadata = *scheduler_task_metadata_at(base, local, task);
    owner->run_epoch = layout->run_epoch;
    owner->task_ref = task;
    owner->order_key = UINT64_MAX;
    owner->group_seq = generation;
    owner->completion_ticket = UINT64_MAX;
    owner->worker_index = static_cast<uint32_t>(local->worker_id());
    owner->block_idx = block;
    owner->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    owner->logical_block_num = metadata.logical_block_num;
    scheduler_lane_mask_store(owner->active_mask, scheduler_task_lanes(metadata));
    owner->flags = metadata.flags;
    owner->sync_tail = tail;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        owner->sync_frontier[lane] = frontiers[lane];
        owner->sync_hold[lane] = holds[lane];
    }
    scheduler_protocol_stamp_at(base, local, layout, &owner->staged, staged);
    scheduler_protocol_stamp_at(base, local, layout, &owner->claim_before, claim_before);
    scheduler_protocol_stamp_at(base, local, layout, &owner->claim_after, claim_after);
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
    lane->order_key = UINT64_MAX;
    lane->group_seq = group.generation;
    lane->worker_index = static_cast<uint32_t>(local->worker_id());
    lane->block_idx = group.block_idx;
    lane->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    lane->lane = local->config.self_lane;
    const auto &metadata = *scheduler_task_metadata_at(base, local, group.task_ref);
    lane->logical_block_num = metadata.logical_block_num;
    lane->flags = metadata.flags;
    scheduler_lane_mask_store(lane->active_mask, group.active_mask);
    scheduler_lane_mask_store(lane->resource_mask, scheduler_mix_resource_mask(group.active_mask));
    for (uint32_t peer = 0; peer < 3; ++peer) {
        lane->lane_subtasks[peer] = group.active_mask.has(static_cast<SubtaskSlot>(peer)) ? peer : UINT8_MAX;
        lane->previous_lane_seq[peer] = 0;
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
    const auto resources = scheduler_mix_resource_mask(group.active_mask);
    for (uint32_t peer = 0; peer < 3; ++peer)
        values[peer] = resources.has(static_cast<SubtaskSlot>(peer)) ? group.generation : 0;
    scheduler_protocol_stamp(base, local, layout, stamp);
    if (releases) scheduler_protocol_publish_record(&record->owner);
    else scheduler_protocol_publish_record(lane);
}

static inline __aicore__ void
scheduler_protocol_cluster_retire(__gm__ void *base, const SchedulerLocalState *local, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->cluster_mix;
    auto *record = scheduler_protocol_cluster_block_at(base, local, group.task_ref, group.block_idx);
    if (record == nullptr) return;
    scheduler_protocol_stamp(
        base, local, layout, before ? &record->owner.completion_before : &record->owner.completion_after
    );
    scheduler_protocol_publish_record(&record->owner);
}

inline __aicore__ bool scheduler_cluster_mix_conflicts(
    const SCHEDULER_CLUSTER_SSBUF SchedulerClusterSignal *signal, LaneMask resources, uint64_t tail,
    uint32_t *frontiers = nullptr, uint32_t *holds = nullptr
) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarClusterMixConflicts);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
        const uint32_t hold = scheduler_cluster_load_field(&signal->sync_hold[lane]);
        const uint32_t frontier = scheduler_cluster_load_field(&signal->sync_frontier[lane]);
        if (holds != nullptr) holds[lane] = hold;
        if (frontiers != nullptr) frontiers[lane] = frontier;
        if (hold != 0 || frontier < tail) return true;
    }
    return false;
}

inline constexpr uint32_t SCHEDULER_CLUSTER_PENDING_BUDGET = 4;

static inline __aicore__ bool scheduler_cluster_mix_publish_preparation(SchedulerLocalState *local, LaneMask mask) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixPublishPreparation);
    auto *signal = local->cluster_signal;
    if (signal == nullptr || !local->is_mix_coordinator()) return false;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->prepare_generation);
    if ((generation & 1U) != 0 || generation > UINT32_MAX - 2) return false;
    scheduler_cluster_store_release(&signal->prepare_generation, generation + 1);
    scheduler_cluster_store_field(
        &signal->prepare_task_ref, mask.is_empty() ? SCHEDULER_TASK_REF_INVALID : local->cluster_pending.task_ref
    );
    scheduler_cluster_store_field(&signal->prepare_block, local->cluster_pending.block_idx);
    scheduler_cluster_store_field(&signal->prepare_mask, static_cast<uint32_t>(mask.raw()));
    scheduler_cluster_store_release(&signal->prepare_generation, generation + 2);
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_discover_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint64_t directory_round, bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixDiscoverTask, task_ref);
    if (local == nullptr || progress == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr) return true;
    if (base == nullptr || local->config.prepared == nullptr)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    auto &pending = local->cluster_pending;
    if (!scheduler_task_ref_is_null(pending.task_ref)) return true;
    if (!scheduler_task_ref_valid(graph, task_ref))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const LaneMask active = scheduler_task_lanes(metadata);
    if (!scheduler_task_is_mix(metadata.flags) || scheduler_task_requires_sync_start(metadata.flags) ||
        !active.is_mix() || metadata.logical_block_num == 0)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if (local->cluster_mix.phase != SchedulerGmMixPhase::IDLE && local->cluster_mix.task_ref == task_ref) return true;
    auto *signal = local->cluster_signal;
    if (scheduler_cluster_mix_armed(scheduler_cluster_load_acquire(&signal->armed_generation)) &&
        scheduler_cluster_load_field(&signal->task_ref) == task_ref)
        return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    SchedulerFaninView fanin{};
    if (scheduler_node_fanin_view(graph, task_ref, &fanin) != SchedulerGraphResult::OK)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const uint32_t blocks = metadata.logical_block_num;
    auto *control = scheduler_task_control_at(base, local, task_ref);
    const auto probes = scheduler_parked_probe_plan(
        local->worker_id(), directory_round, blocks, scheduler_gm_query(control->parked_count) != 0
    );
    for (uint32_t probe = 0; probe < probes.count; ++probe) {
        const uint32_t block = probes.blocks[probe];
        uint64_t index = UINT64_MAX;
        uint8_t anchor_lane = UINT8_MAX;
        if (!scheduler_group_anchor_index(base, local->config.prepared, task_ref, block, &index, &anchor_lane))
            return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        const uint64_t observed = scheduler_gm_fetch_add(
            scheduler_prepared_control_at(base, local->config.prepared, index)->state, UINT64_C(0)
        );
        if (scheduler_prepared_phase(observed) == SchedulerPreparedPhase::PARKED) {
            const auto result = scheduler_resume_group_execution(
                base, local->config.prepared, local->worker_id(), task_ref, block, observed, &pending.anchor,
                control->parked_count
            );
            if (result == PreparedAccessResult::ERROR)
                return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
            if (result == PreparedAccessResult::EXHAUSTED) continue;
            pending.task_ref = task_ref;
            pending.block_idx = block;
            break;
        }
    }
    if (scheduler_task_ref_is_null(pending.task_ref)) {
        if (scheduler_gm_query(control->next_block) >= blocks &&
            scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) >= blocks)
            return true;
        const uint64_t block = scheduler_gm_fetch_add(control->next_block, UINT64_C(1));
        if (block >= blocks) {
            if (scheduler_gm_fetch_add(control->next_block, UINT64_MAX) <= blocks)
                return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::BLOCK_CURSOR_INVALID);
            return true;
        }
        pending.task_ref = task_ref;
        pending.block_idx = static_cast<uint32_t>(block);
        if (scheduler_reserve_group_execution(
                base, local->config.prepared, local->worker_id(), task_ref, pending.block_idx, &pending.anchor
            ) != PreparedAccessResult::READY)
            return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    }
    pending.fanin = fanin;
    pending.next_fanin = 0;
    pending.blocked_rounds = 0;
    if (!scheduler_cluster_mix_publish_preparation(local, active))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    *progress = true;
    return true;
}

// Availability is an observation; only the later reservation CAS owns a group.
static inline __aicore__ bool scheduler_cluster_mix_ready_replacement(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint64_t directory_round, bool *ready
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixReadyReplacement, task_ref);
    *ready = false;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const uint32_t blocks = metadata.logical_block_num;
    auto *control = scheduler_task_control_at(base, local, task_ref);
    bool available = scheduler_gm_fetch_add(control->next_block, UINT64_C(0)) < blocks;
    if (!available) {
        const auto probes = scheduler_parked_probe_plan(
            local->worker_id(), directory_round, blocks, scheduler_gm_query(control->parked_count) != 0
        );
        for (uint32_t probe = 0; probe < probes.count; ++probe) {
            uint64_t index = UINT64_MAX;
            uint8_t lane = UINT8_MAX;
            if (!scheduler_group_anchor_index(
                    base, local->config.prepared, task_ref, probes.blocks[probe], &index, &lane
                ))
                return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
            const uint64_t observed = scheduler_gm_fetch_add(
                scheduler_prepared_control_at(base, local->config.prepared, index)->state, UINT64_C(0)
            );
            if (scheduler_prepared_phase(observed) != SchedulerPreparedPhase::PARKED) continue;
            const uint32_t generation = scheduler_prepared_generation(observed);
            if (generation == UINT32_MAX ||
                observed != scheduler_prepared_state(SchedulerPreparedPhase::PARKED, UINT32_MAX, generation))
                return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
            available = true;
            break;
        }
    }
    if (!available) return true;
    SchedulerFaninView fanin{};
    if (scheduler_node_fanin_view(graph, task_ref, &fanin) != SchedulerGraphResult::OK)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    uint32_t next_fanin = 0;
    return scheduler_observe_fanin_prefix(graph, base, local, run, task_ref, fanin, next_fanin, ready);
}

static inline __aicore__ bool scheduler_cluster_mix_discover_next(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixDiscoverNext);
    if (local == nullptr || progress == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr) return true;
    if (base == nullptr || local->config.prepared == nullptr || local->config.task_catalog_offset == 0)
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto catalog = scheduler_task_catalog(base, pool, local->config.self_core_type);
    if (!scheduler_task_catalog_valid(catalog))
        return scheduler_admission_fail(
            graph, local, run, SCHEDULER_TASK_REF_INVALID, SchedulerErrorSite::GROUP_INVALID_STATE
        );
    if (catalog.count == 0) return true;
    auto &pending = local->cluster_pending;
    if (!scheduler_task_ref_is_null(pending.task_ref)) {
        if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
        if (!scheduler_group_handle_valid(
                base, local->config.prepared, local->worker_id(), &pending.anchor,
                SchedulerPreparedPhase::GROUP_RESERVED
            ))
            return scheduler_admission_fail(
                graph, local, run, pending.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE
            );
        if (pending.next_fanin == pending.fanin.count) return true;
    }
    // Discovery yields after one reservation or a bounded slice of the catalog.
    constexpr uint32_t discovery_budget = 8;
    const uint64_t limit = catalog.count < discovery_budget ? catalog.count : discovery_budget;
    int64_t just_parked = SCHEDULER_TASK_REF_INVALID;
    for (uint64_t checked = 0; checked < limit; ++checked) {
        if (local->cluster_scan_cursor >= catalog.count) local->cluster_scan_cursor %= catalog.count;
        const int64_t ref = catalog.refs[local->cluster_scan_cursor];
        const uint64_t round = local->cluster_scan_round;
        if (++local->cluster_scan_cursor == catalog.count) {
            local->cluster_scan_cursor = 0;
            ++local->cluster_scan_round;
        }
        if (!scheduler_task_ref_valid(graph, ref))
            return scheduler_admission_fail(graph, local, run, ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        if (ref == just_parked || ref == pending.task_ref) continue;
        const auto flags = scheduler_task_metadata_at(base, local, ref)->flags;
        if (!scheduler_task_is_mix(flags) || scheduler_task_requires_sync_start(flags)) continue;
        if (!scheduler_task_ref_is_null(pending.task_ref)) {
            if (local->cluster_mix.phase != SchedulerGmMixPhase::IDLE && local->cluster_mix.task_ref == ref) continue;
            if (scheduler_cluster_mix_armed(scheduler_cluster_load_acquire(&local->cluster_signal->armed_generation)) &&
                scheduler_cluster_load_field(&local->cluster_signal->task_ref) == ref)
                continue;
            bool ready = false;
            if (!scheduler_cluster_mix_ready_replacement(graph, base, local, run, ref, round, &ready)) return false;
            if (!ready) continue;
            const uint32_t previous = pending.next_fanin;
            if (!scheduler_observe_fanin_prefix(
                    graph, base, local, run, pending.task_ref, pending.fanin, pending.next_fanin, &ready
                ))
                return false;
            *progress = *progress || pending.next_fanin != previous;
            if (ready) return true;
            if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
            const int64_t old_task = pending.task_ref;
            if (!scheduler_cluster_mix_publish_preparation(local, LaneMask::none()))
                return scheduler_admission_fail(graph, local, run, old_task, SchedulerErrorSite::GROUP_INVALID_STATE);
            auto *control = scheduler_task_control_at(base, local, old_task);
            if (!scheduler_park_group_execution(
                    base, local->config.prepared, local->worker_id(), &pending.anchor, control->parked_count
                ))
                return scheduler_admission_fail(graph, local, run, old_task, SchedulerErrorSite::GROUP_INVALID_STATE);
            just_parked = old_task;
            pending = {};
            *progress = true;
        }
        bool acquired = false;
        if (!scheduler_cluster_mix_discover_task(graph, base, local, run, ref, round, &acquired)) return false;
        *progress = *progress || acquired;
        if (acquired) return true;
    }
    return true;
}

// A reserved anchor names one logical block; arming consumes that reservation.
static inline __aicore__ bool scheduler_cluster_mix_try_arm(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, bool *armed
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixArm, task_ref);
    *armed = false;
    auto *signal = local->cluster_signal;
    if (signal == nullptr || !local->is_mix_coordinator() || !scheduler_task_ref_valid(graph, task_ref))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const LaneMask active = scheduler_task_lanes(metadata);
    if (!scheduler_task_is_mix(metadata.flags) || scheduler_task_requires_sync_start(metadata.flags) ||
        !active.is_mix() || metadata.logical_block_num == 0 ||
        scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const auto &reserved = local->cluster_pending;
    if (reserved.task_ref != task_ref || reserved.block_idx >= metadata.logical_block_num ||
        reserved.next_fanin != reserved.fanin.count ||
        !scheduler_group_handle_valid(
            base, local->config.prepared, local->worker_id(), &reserved.anchor, SchedulerPreparedPhase::GROUP_RESERVED
        ))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    if (scheduler_cluster_mix_armed(generation) || scheduler_cluster_load_field(&signal->staged_mask) != 0 ||
        !scheduler_cluster_mix_slots_free(local))
        return true;
    if (generation >= UINT32_MAX - 1)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    uint64_t tail = 0;
    if (local->admission != nullptr) {
        // commit_tail is a bypass-published signal, matching ordinary entry;
        // atomic-only words, unlike this signal, require RMW confirmation.
        tail = scheduler_gm_query(
            scheduler_state_at<SchedulerAdmissionControl>(base, local->admission->control_offset)->commit_tail
        );
        if (tail > local->admission->capacity)
            return scheduler_admission_fail(
                graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
    }
    const LaneMask resources = scheduler_mix_resource_mask(active);
    const bool record = scheduler_protocol_enabled(local);
    uint32_t frontiers[3]{};
    uint32_t holds[3]{};
    if (scheduler_cluster_mix_conflicts(signal, resources, tail)) return true;
    scheduler_cluster_store_field(&signal->staged_mask, static_cast<uint32_t>(resources.raw()));
    scheduler_cluster_barrier();
    const uint64_t staged = record ? scheduler_protocol_cycles(local) : 0;
    if (scheduler_cluster_mix_conflicts(
            signal, resources, tail, record ? frontiers : nullptr, record ? holds : nullptr
        )) {
        scheduler_cluster_store_release(&signal->staged_mask, UINT32_C(0));
        return true;
    }
    auto &pending = local->cluster_pending;
    // The final ARM claim is recorded in this uninterrupted publication slice.
    // A reservation may span other protocol events or be parked and resumed.
    const uint64_t claim_before = record ? scheduler_protocol_cycles(local) : 0;
    if (!scheduler_arm_group_execution(base, local->config.prepared, local->worker_id(), &pending.anchor)) {
        scheduler_cluster_store_release(&signal->staged_mask, UINT32_C(0));
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    }
    const uint64_t claim_after = record ? scheduler_protocol_cycles(local) : 0;
    const uint32_t block = pending.block_idx;
    const uint64_t publish_before = record ? scheduler_protocol_cycles(local) : 0;
    scheduler_cluster_store_field(&signal->anchor_state, pending.anchor.claimed_state);
    scheduler_cluster_store_field(&signal->task_ref, task_ref);
    scheduler_cluster_store_field(&signal->sync_tail, static_cast<uint32_t>(tail));
    scheduler_cluster_store_field(&signal->block_idx, static_cast<uint32_t>(block));
    scheduler_cluster_store_field(&signal->active_mask, static_cast<uint32_t>(active.raw()));
    scheduler_cluster_store_release(&signal->armed_generation, generation + 1);
    if (record)
        scheduler_protocol_cluster_arm(
            base, local, task_ref, static_cast<uint32_t>(block), generation + 1, tail, frontiers, holds, staged,
            claim_before, claim_after, publish_before, scheduler_protocol_cycles(local)
        );
    pending = {};
    *armed = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_progress_pending(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixPending);
    if (local == nullptr || progress == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || scheduler_task_ref_is_null(local->cluster_pending.task_ref)) return true;
    auto &pending = local->cluster_pending;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (!scheduler_group_handle_valid(
            base, local->config.prepared, local->worker_id(), &pending.anchor, SchedulerPreparedPhase::GROUP_RESERVED
        ))
        return scheduler_admission_fail(graph, local, run, pending.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const uint32_t previous_fanin = pending.next_fanin;
    bool ready = false;
    if (!scheduler_observe_fanin_prefix(
            graph, base, local, run, pending.task_ref, pending.fanin, pending.next_fanin, &ready
        ))
        return false;
    *progress = pending.next_fanin != previous_fanin;
    if (!ready) return true;
    if (pending.next_fanin == pending.fanin.count) {
        bool armed = false;
        if (!scheduler_cluster_mix_try_arm(graph, base, local, run, pending.task_ref, &armed)) return false;
        if (armed) {
            *progress = true;
            return true;
        }
    }
    if (++pending.blocked_rounds < SCHEDULER_CLUSTER_PENDING_BUDGET) return true;
    const int64_t task_ref = pending.task_ref;
    if (!scheduler_cluster_mix_publish_preparation(local, LaneMask::none()))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    auto *control = scheduler_task_control_at(base, local, task_ref);
    if (!scheduler_park_group_execution(
            base, local->config.prepared, local->worker_id(), &pending.anchor, control->parked_count
        ))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    pending = {};
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_retire(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixRetire);
    *progress = false;
    auto &group = local->cluster_mix;
    auto *signal = local->cluster_signal;
    if (!local->is_mix_coordinator() || group.phase != SchedulerGmMixPhase::RELEASED) return true;
    if (signal == nullptr || !scheduler_cluster_mix_armed(group.generation) ||
        scheduler_cluster_load_acquire(&signal->armed_generation) != group.generation)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if (!scheduler_cluster_mix_all(signal->finished, scheduler_mix_resource_mask(group.active_mask), group.generation))
        return true;
    scheduler_protocol_cluster_all(base, local, true);
    scheduler_protocol_cluster_retire(base, local, true);
    scheduler_cluster_store_release(&signal->armed_generation, group.generation + 1);
    scheduler_cluster_store_release(&signal->staged_mask, UINT32_C(0));
    scheduler_protocol_cluster_retire(base, local, false);
    group.phase = SchedulerGmMixPhase::IDLE;
    group.task_ref = SCHEDULER_TASK_REF_INVALID;
    group.slot_index = UINT8_MAX;
    group.subtask_slot = UINT8_MAX;
    *progress = true;
    return true;
}

// A named lane keeps the formal payload immutable until its finished ACK.
// Unnamed readers validate the generation around the mask before using fields.
static inline __aicore__ bool scheduler_cluster_mix_observe(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixObserve);
    *progress = false;
    auto *signal = local->cluster_signal;
    auto &group = local->cluster_mix;
    if (signal == nullptr) return true;
    if (!local->is_mix_coordinator() && group.phase == SchedulerGmMixPhase::RELEASED) {
        group.phase = SchedulerGmMixPhase::IDLE;
        group.task_ref = SCHEDULER_TASK_REF_INVALID;
    }
    if (group.phase != SchedulerGmMixPhase::IDLE) return true;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    if (!scheduler_cluster_mix_armed(generation) || generation == group.generation) return true;
    const uint32_t active_bits = scheduler_cluster_load_field(&signal->active_mask);
    const uint32_t resources = scheduler_cluster_load_field(&signal->staged_mask);
    if ((resources & (1U << local->config.self_lane)) == 0) return true;
    const int64_t task_ref = scheduler_cluster_load_field(&signal->task_ref);
    const uint64_t anchor_state = scheduler_cluster_load_field(&signal->anchor_state);
    const uint32_t block = scheduler_cluster_load_field(&signal->block_idx);
    const uint64_t sync_tail = scheduler_cluster_load_field(&signal->sync_tail);
    scheduler_cluster_barrier();
    if (scheduler_cluster_load_acquire(&signal->armed_generation) != generation) return true;
    if (scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch ||
        !scheduler_task_ref_valid(graph, task_ref) || active_bits > 7 ||
        anchor_state != scheduler_prepared_state(
                            SchedulerPreparedPhase::GROUP_ARMED, local->config.worker_ids[0],
                            scheduler_prepared_generation(anchor_state)
                        ))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    const LaneMask active = scheduler_task_lanes(metadata);
    if (!scheduler_task_is_mix(metadata.flags) || scheduler_task_requires_sync_start(metadata.flags) ||
        active.raw() != active_bits || scheduler_mix_resource_mask(active).raw() != resources ||
        block >= metadata.logical_block_num)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    group.task_ref = task_ref;
    group.anchor_state = anchor_state;
    group.sync_tail = sync_tail;
    group.block_idx = block;
    group.active_mask = active;
    group.generation = generation;
    group.slot_index = UINT8_MAX;
    group.subtask_slot =
        active.has(static_cast<SubtaskSlot>(local->config.self_lane)) ? local->config.self_lane : UINT8_MAX;
    group.prepare_start_cycles = 0;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    scheduler_protocol_cluster_observe(base, local);
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_service(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixService);
    *progress = false;
    if (local->cluster_signal == nullptr) return true;
    if (local->cluster_mix.phase == SchedulerGmMixPhase::IDLE &&
        (!scheduler_cluster_mix_armed(scheduler_cluster_load_acquire(&local->cluster_signal->armed_generation)) ||
         !scheduler_cluster_names_lane(local->cluster_signal, local->config.self_lane)))
        return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    bool advanced = false;
    if (!scheduler_cluster_mix_retire(graph, base, local, run, &advanced)) return false;
    *progress = advanced;
    if (!scheduler_cluster_mix_observe(graph, base, local, run, &advanced)) return false;
    *progress = *progress || advanced;
    auto &group = local->cluster_mix;
    auto *signal = local->cluster_signal;
    if (group.phase == SchedulerGmMixPhase::PUBLISHED) {
        if (!scheduler_cluster_mix_slots_free(local)) return true;
        scheduler_protocol_cluster_ack(base, local, false, true);
        scheduler_cluster_store_release(&signal->arrived[local->config.self_lane], group.generation);
        scheduler_protocol_cluster_ack(base, local, false, false);
        group.phase = SchedulerGmMixPhase::ARRIVED;
        *progress = true;
    }
    if (group.phase != SchedulerGmMixPhase::ARRIVED) return true;
    const bool all =
        scheduler_cluster_mix_all(signal->arrived, scheduler_mix_resource_mask(group.active_mask), group.generation);
    if (all) scheduler_protocol_cluster_all(base, local, false);
    if (group.subtask_slot == UINT8_MAX) {
        if (all) {
            scheduler_protocol_cluster_ack(base, local, true, true);
            scheduler_cluster_store_release(&signal->finished[local->config.self_lane], group.generation);
            scheduler_protocol_cluster_ack(base, local, true, false);
            group.phase = SchedulerGmMixPhase::RELEASED;
            *progress = true;
        }
        return true;
    }
    const bool retained = group.slot_index != UINT8_MAX;
    if (!retained) {
        for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index)
            if (local->slots[index].state == SchedulerDispatchSlotState::FREE) {
                group.slot_index = static_cast<uint8_t>(index);
                break;
            }
        if (group.slot_index == UINT8_MAX) return true;
        auto &slot = local->slots[group.slot_index];
        slot.task_ref = group.task_ref;
        slot.block_idx = group.block_idx;
        slot.subtask_slot = group.subtask_slot;
        slot.cluster_mix = true;
        slot.state = SchedulerDispatchSlotState::FILLING;
    }
    auto &slot = local->slots[group.slot_index];
    if (!slot.cluster_mix || slot.state != SchedulerDispatchSlotState::FILLING || slot.task_ref != group.task_ref ||
        slot.block_idx != group.block_idx || slot.subtask_slot != group.subtask_slot)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const auto metadata = scheduler_load_dispatch_metadata(base, local, group.task_ref, profiling_level);
    SchedulerReadyClaim ready{};
    ready.task_ref = group.task_ref;
    ready.source = SchedulerReadySource::COOPERATIVE;
    const SchedulerFreeSlotClaim claim{group.slot_index};
    if (slot.prepared.claimed_state == 0) {
        group.prepare_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        if (scheduler_acquire_group_lane(
                base, local->config.prepared, local->worker_id(), local->config.worker_ids[0], group.task_ref,
                group.block_idx, group.subtask_slot, group.anchor_state, &slot.prepared
            ) != PreparedAccessResult::READY)
            return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
        const auto result = scheduler_prepare_dispatch_slot(
            graph, base, local, run, claim, ready, metadata, profiling_level, group.subtask_slot, nullptr
        );
        if (result == PreparedAccessResult::WAIT) return true;
        if (result != PreparedAccessResult::READY)
            return scheduler_admission_fail(
                graph, local, run, group.task_ref, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
            );
        auto *binding = scheduler_execution_binding_at(base, local->config.prepared, slot.prepared.execution_index);
        binding->global_context.sub_block_id = local->config.self_lane == 2 ? 1 : 0;
        *progress = true;
    }
    if (!all) return true;
    scheduler_publish_dispatch_slot(
        base, local, claim, ready, metadata, profiling_level, group.subtask_slot, group.prepare_start_cycles
    );
    group.phase = SchedulerGmMixPhase::RUNNING;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_can_enter(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t slot_index,
    bool *permit
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixEnter);
    *permit = false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &group = local->cluster_mix;
    const auto &slot = local->slots[slot_index];
    const auto *signal = local->cluster_signal;
    if (signal == nullptr || group.phase != SchedulerGmMixPhase::RUNNING || group.slot_index != slot_index ||
        slot.state != SchedulerDispatchSlotState::READY || !slot.cluster_mix || slot.task_ref != group.task_ref ||
        slot.block_idx != group.block_idx || slot.subtask_slot != group.subtask_slot ||
        scheduler_cluster_load_acquire(&signal->armed_generation) != group.generation ||
        scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch ||
        !scheduler_cluster_mix_all(signal->arrived, scheduler_mix_resource_mask(group.active_mask), group.generation))
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    *permit = true;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_release(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerLocalSlotState &completed, uint32_t slot_index
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixRelease, completed.task_ref);
    auto &group = local->cluster_mix;
    if (local->cluster_signal == nullptr || group.phase != SchedulerGmMixPhase::RUNNING ||
        group.slot_index != slot_index || !completed.cluster_mix || completed.task_ref != group.task_ref ||
        completed.block_idx != group.block_idx || completed.subtask_slot != group.subtask_slot ||
        local->slots[slot_index].state != SchedulerDispatchSlotState::FREE)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_protocol_cluster_ack(base, local, true, true);
    scheduler_cluster_store_release(&local->cluster_signal->finished[local->config.self_lane], group.generation);
    scheduler_protocol_cluster_ack(base, local, true, false);
    group.phase = SchedulerGmMixPhase::RELEASED;
    group.slot_index = UINT8_MAX;
    return true;
}

// A local return is not the coordinator's retirement proof. EXIT keeps servicing
// that proof, including a vector-only group whose AIC ran no business Call.
static inline __aicore__ bool scheduler_cluster_mix_drain(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *done
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarClusterMixDrain);
    if (!scheduler_task_ref_is_null(local->cluster_pending.task_ref))
        return scheduler_admission_fail(
            graph, local, run, local->cluster_pending.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    *done = false;
    if (local->cluster_signal == nullptr) {
        *done = true;
        return true;
    }
    auto &group = local->cluster_mix;
    auto *signal = local->cluster_signal;
    if (local->is_mix_coordinator() && group.phase == SchedulerGmMixPhase::ARRIVED && group.subtask_slot == UINT8_MAX) {
        if (!scheduler_cluster_mix_all(
                signal->arrived, scheduler_mix_resource_mask(group.active_mask), group.generation
            ))
            return true;
        scheduler_protocol_cluster_all(base, local, false);
        scheduler_protocol_cluster_ack(base, local, true, true);
        scheduler_cluster_store_release(&signal->finished[local->config.self_lane], group.generation);
        scheduler_protocol_cluster_ack(base, local, true, false);
        group.phase = SchedulerGmMixPhase::RELEASED;
    }
    bool retired = false;
    if (!scheduler_cluster_mix_retire(graph, base, local, run, &retired)) return false;
    if (group.phase != SchedulerGmMixPhase::IDLE && group.phase != SchedulerGmMixPhase::RELEASED)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT);
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->armed_generation);
    const bool named = scheduler_cluster_names_lane(signal, local->config.self_lane);
    if (local->is_mix_coordinator()) {
        *done = !named;
    } else {
        *done = !named || !scheduler_cluster_mix_armed(generation) ||
                scheduler_cluster_load_acquire(&signal->finished[local->config.self_lane]) == generation;
    }
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_can_enter_dispatch_slot(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t slot_index, bool *permit
) {
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT || permit == nullptr) return false;
    if (local->slots[slot_index].cluster_mix) {
        if (!scheduler_cluster_mix_can_enter(graph, local, run, slot_index, permit)) return false;
        if (*permit && RecordProtocol)
            scheduler_protocol_execution_permit(
                base, local, slot_index, local->cluster_mix.sync_tail, local->admission != nullptr
            );
        return true;
    }
    if (local->admission != nullptr)
        return scheduler_ordered_can_enter_slot<RecordProtocol>(graph, base, local, run, slot_index, permit);
    *permit = scheduler_read_worker_error(run, local->worker_id()) == 0;
    return *permit;
}
