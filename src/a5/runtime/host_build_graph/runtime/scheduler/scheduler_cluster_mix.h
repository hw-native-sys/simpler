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

static inline __aicore__ void
scheduler_protocol_cluster_observe(__gm__ void *base, const SchedulerLocalState *local, uint32_t bank) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->mix_groups[bank];
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

static inline __aicore__ void scheduler_protocol_cluster_ack(
    __gm__ void *base, const SchedulerLocalState *local, bool release, bool before, uint32_t bank
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->mix_groups[bank];
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
scheduler_protocol_cluster_all(__gm__ void *base, const SchedulerLocalState *local, bool releases, uint32_t bank) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->mix_groups[bank];
    auto *record = scheduler_protocol_cluster_block_at(base, local, group.task_ref, group.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    auto *stamp = releases ? &record->owner.releases_observed : &lane->arrivals_observed;
    if (stamp->sequence != 0) return;
    auto *values = releases ? record->owner.released_seq : lane->arrived_seq;
    const auto resources = group.active_mask;
    for (uint32_t peer = 0; peer < 3; ++peer) {
        const bool observed = releases || group.requires_peer_order || peer == local->config.self_lane;
        values[peer] = observed && resources.has(static_cast<SubtaskSlot>(peer)) ? group.generation : 0;
    }
    scheduler_protocol_stamp(base, local, layout, stamp);
    if (releases) {
        record->owner.completion_worker = local->worker_id();
        scheduler_protocol_publish_record(&record->owner);
    } else {
        scheduler_protocol_publish_record(lane);
    }
}

// Only the proxy reuses a bank. Every named lane published its execution credit
// before FIN, so the matching FINs suffice when no accountant receipt is owed.
static inline __aicore__ bool scheduler_cluster_mix_recycle(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, uint32_t bank,
    uint32_t generation, bool *recycled
) {
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixRecycle);
    auto *signal = &local->cluster_signal->groups[bank];
    if (scheduler_mix_needs_receipt(local, signal)) {
        if (scheduler_cluster_load_field(&signal->accounted_generation) != generation) return true;
    } else {
        const auto fin =
            scheduler_mix_fin_status(signal, scheduler_cluster_load_field(&signal->active_mask), generation);
        if (fin == SchedulerMixFin::PENDING) return true;
        if (fin == SchedulerMixFin::FAILED)
            return scheduler_admission_fail(
                graph, local, run, scheduler_cluster_load_field(&signal->task_ref),
                SchedulerErrorSite::GROUP_ACK_INVALID
            );
    }
    scheduler_cluster_store_release(&signal->armed_generation, generation + 1);
    // An unnamed observer may still be sampling the old odd generation.
    // Publish invalidation before any new payload field can be overwritten.
    scheduler_cluster_barrier();
    *recycled = true;
    return true;
}

static inline __aicore__ void scheduler_notify_mix_members(
    __gm__ void *base, SchedulerLocalState *local, int64_t ref, uint32_t block, uint32_t rank, LaneMask active,
    uint32_t bank
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarMixNotifyMembers, ref);
    const bool trace = scheduler_protocol_enabled(local);
    const uint64_t before = __builtin_expect(trace, false) ? scheduler_protocol_cycles(local) : 0;
    auto *descriptor = &local->cluster_signal->groups[bank];
    const uint32_t generation = local->last_mix_generation == 0 ? 1 : local->last_mix_generation + 2;
    local->last_mix_generation = generation;
    const bool l2 = local->profiling != nullptr && local->profiling->capture_l2;
    if (__builtin_expect(l2, false)) {
        // Consume the handoff even for fixed blocks, which do not acquire a GM ticket.
        local->profiling->last_complete_cycles = 0;
        local->profiling->last_retired_cycles = 0;
        scheduler_cluster_store_field(&local->cluster_signal->l2_proxy[bank].prepare_cycles, scheduler_cycles());
    }
    scheduler_cluster_store_field(&descriptor->task_ref, ref);
    scheduler_cluster_store_field(&descriptor->rank, rank);
    scheduler_cluster_store_field(&descriptor->block_idx, block);
    scheduler_cluster_store_field(&descriptor->active_mask, static_cast<uint32_t>(active.raw()));
    if (__builtin_expect(l2, false))
        scheduler_cluster_store_field(&local->cluster_signal->l2_proxy[bank].arm_cycles, scheduler_cycles());
    scheduler_cluster_store_release(&descriptor->armed_generation, generation);
    const uint64_t after = __builtin_expect(trace, false) ? scheduler_protocol_cycles(local) : 0;
    scheduler_store_signal(
        scheduler_worker_context_at(base, local, local->worker_id())->mix_proxy_published[bank],
        scheduler_mix_proxy_publication(static_cast<uint32_t>(ref), block)
    );
    if (__builtin_expect(trace, false)) {
        scheduler_protocol_cluster_arm(base, local, ref, block, generation, rank, before, after);
        scheduler_cluster_store_release(&descriptor->trace_generation, generation);
    }
}

static inline __aicore__ bool scheduler_claim_for_execute(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    if (local == nullptr || progress == nullptr || base == nullptr || run == nullptr) return false;
    *progress = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr) return true;
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
        (local->issue_binding.task_ref == task_ref && local->issue_binding.ticket_exhausted))
        return true;

    const auto *signal = local->cluster_signal;
    uint32_t bank = SCHEDULER_MIX_MEMBER_COUNT;
    for (uint32_t i = 0; i < SCHEDULER_MIX_MEMBER_COUNT; ++i) {
        const auto *descriptor = &signal->groups[i];
        const uint32_t generation = scheduler_cluster_load_field(&descriptor->armed_generation);
        if (scheduler_cluster_mix_armed(generation)) {
            bool recycled = false;
            if (!scheduler_cluster_mix_recycle(graph, local, run, i, generation, &recycled)) return false;
            if (!recycled) {
                if (scheduler_cluster_load_field(&descriptor->task_ref) == task_ref) return true;
                continue;
            }
        }
        if (bank == SCHEDULER_MIX_MEMBER_COUNT) bank = i;
    }
    if (bank == SCHEDULER_MIX_MEMBER_COUNT) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &order = scheduler_order_for(local, task_ref);
    if (order.rank < scheduler_cluster_load_field(&signal->mix_floor))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER);
    if (local->last_mix_generation >= UINT32_MAX - 2 ||
        scheduler_cluster_load_field(&signal->run_epoch) != local->config.prepared->run_epoch)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);

    const bool trace = scheduler_protocol_enabled(local);
    uint32_t block = blocks;
    bool closes = false;
    const bool fixed = scheduler_task_has_fixed_owners(metadata.flags);
    {
        SchedulerScalarScope scope(local, SchedPhaseKind::ScalarMixClaimForExecution, task_ref);
        const uint64_t before = __builtin_expect(trace, false) ? scheduler_protocol_cycles(local) : 0;
        if (fixed) {
            block = scheduler_fixed_mix_block(local->config.cluster_index, blocks, active);
        } else {
            const bool l2 = local->profiling != nullptr && local->profiling->capture_l2;
            const uint64_t begin = __builtin_expect(l2, false) ? scheduler_cycles() : 0;
            if (!scheduler_claim_ticket(graph, base, local, run, task_ref, blocks, &block, &closes)) return false;
            if (__builtin_expect(l2 && block < blocks, false)) {
                const uint64_t end = scheduler_cycles();
                auto &proxy = local->cluster_signal->l2_proxy[bank];
                scheduler_cluster_store_field(&proxy.claim_begin_cycles, begin);
                scheduler_cluster_store_field(&proxy.claim_end_cycles, end);
                scheduler_cluster_store_field(&proxy.previous_complete_cycles, local->profiling->last_complete_cycles);
                scheduler_cluster_store_field(&proxy.previous_retired_cycles, local->profiling->last_retired_cycles);
            }
        }
        scheduler_invalidate_issue_snapshot(snapshot);
        if (block >= blocks) {
            if (local->issue_binding.task_ref == task_ref) local->issue_binding.ticket_exhausted = true;
            return true;
        }
        if (__builtin_expect(trace, false))
            scheduler_protocol_block_claim(
                base, local, task_ref, block, before, scheduler_protocol_cycles(local), closes ? blocks : 0
            );
    }
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    scheduler_notify_mix_members(base, local, task_ref, block, order.rank, active, bank);
    if (fixed) ++local->issue_cursor;
    if (sync) local->last_sync_mix_ref = task_ref;
    *progress = true;
    return scheduler_close_ordinary_supply(graph, base, local, run, task_ref, closes, blocks, block);
}

static inline __aicore__ bool scheduler_cluster_mix_discover_next(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, SchedulerIssueSnapshot *snapshot = nullptr, bool *published = nullptr
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixDiscoverNext);
    if (local == nullptr || progress == nullptr || base == nullptr || run == nullptr) return false;
    *progress = false;
    if (published != nullptr) *published = false;
    if (!local->is_mix_coordinator() || local->cluster_signal == nullptr) return true;
    int64_t ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &ref, progress, snapshot)) return false;
    if (scheduler_task_ref_is_null(ref) || !scheduler_task_is_mix(scheduler_task_metadata_at(base, local, ref)->flags))
        return true;
    bool issued = false;
    if (!scheduler_claim_for_execute(graph, base, local, run, ref, &issued, snapshot)) return false;
    *progress = *progress || issued;
    if (published != nullptr) *published = issued;
    return true;
}

static inline __aicore__ bool scheduler_cluster_mix_observe(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, uint32_t bank
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixObserve);
    auto &group = local->mix_groups[bank];
    const auto *signal = &local->cluster_signal->groups[bank];
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
        group.settled_generation = generation;
        return true;
    }
    if (__builtin_expect(scheduler_protocol_enabled(local), false) &&
        scheduler_cluster_load_acquire(&signal->trace_generation) != generation)
        return true;
    const auto task = scheduler_cluster_load_field(&signal->task_ref);
    const uint32_t rank = scheduler_cluster_load_field(&signal->rank);
    const uint32_t block = scheduler_cluster_load_field(&signal->block_idx);
    if (scheduler_cluster_load_acquire(&signal->armed_generation) != generation) return true;
    if (scheduler_cluster_load_field(&local->cluster_signal->run_epoch) != local->run_epoch ||
        !scheduler_task_ref_valid(graph, task) || group.phase != SchedulerGmMixPhase::IDLE)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task);
    const auto flags = metadata.flags;
    const auto active = scheduler_task_lanes(metadata);
    const auto &order = scheduler_order_for(local, task);
    if (!scheduler_task_is_mix(flags) || scheduler_task_has_predicate(flags) || !active.is_mix() ||
        active.raw() != bits || block >= metadata.logical_block_num || order.rank != rank)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR);
    if (group.state != SchedulerDispatchSlotState::FREE)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_STATE);
    // The descriptor supplies the identity and the fan-in view for this generation.
    group.wait = {};
    group.next_fanin = 0;
    group.ordinary_consumer_waits = local->execution_tokens[0].wait.predecessor == task;
    group.completion_submitted = false;
    if (__builtin_expect(local->profiling != nullptr, false) && local->profiling->capture_l2)
        group.prepare_start_cycles = scheduler_cycles();
    group.task_ref = task;
    group.rank = rank;
    group.block_idx = block;
    group.generation = generation;
    group.active_mask = active;
    group.sync_start = scheduler_task_requires_sync_start(flags);
    group.may_observe_async_inputs = (flags & SCHEDULER_TASK_ASYNC_INPUT) != 0;
    group.required_floor = order.required_floor[lane];
    group.requires_peer_order = order.requires_peer_order != 0;
    group.subtask_slot = lane;
    group.phase = SchedulerGmMixPhase::PUBLISHED;
    if (!scheduler_task_is_executable(flags) ||
        scheduler_metadata_core_type_index(lane) != local->config.self_core_type)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::DISPATCH_INVALID_SHAPE);
    SchedulerNativeNode node{};
    if (scheduler_profiled_node_fanin_view(graph, task, &group.fanin, &node) != SchedulerGraphResult::OK)
        return scheduler_admission_fail(graph, local, run, task, SchedulerErrorSite::GROUP_INVALID_STATE);
    scheduler_protocol_cluster_observe(base, local, bank);
    if (scheduler_resident_graph_is_native(graph)) {
        const auto binding_error = scheduler_observe_graph_arguments(graph, node);
        if (binding_error != SchedulerGraphBindingError::NONE)
            return scheduler_admission_fail(graph, local, run, task, scheduler_graph_binding_error_site(binding_error));
    }
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_poll_mix_fanin(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, uint32_t bank, bool first_observation
) {
    auto &group = local->mix_groups[bank];
    if (group.next_fanin == group.fanin.count) return true;
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixDependencies, group.task_ref);
    if (first_observation) {
        // Both order_ready and successful FIN follow this member's complete data fanin.
        // A late joiner can use either existing publication for this descriptor generation.
        const auto &signal = local->cluster_signal->groups[bank];
        const auto *witness = group.requires_peer_order ? signal.order_ready : signal.finished;
        const uint32_t lane = local->config.self_lane;
        constexpr uint32_t lanes = PLATFORM_CORES_PER_BLOCKDIM;
        static_assert(lanes == 3);
        if (scheduler_cluster_load_field(&witness[(lane + 1) % lanes]) == group.generation ||
            scheduler_cluster_load_field(&witness[(lane + 2) % lanes]) == group.generation) {
            scheduler_cluster_barrier();
            group.next_fanin = group.fanin.count;
            *progress = true;
            return true;
        }
    }
    const auto previous = group.next_fanin;
    if (!scheduler_observe_fanin_prefix(
            graph, base, local, run, group.task_ref, group.fanin, group.next_fanin, &group.wait
        ))
        return false;
    *progress = *progress || previous != group.next_fanin;
    if (group.next_fanin == group.fanin.count) scheduler_cluster_barrier();
    return true;
}

static inline __aicore__ bool scheduler_account_mix_returns(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, uint32_t bank
) {
    auto &group = local->mix_groups[bank];
    if (group.phase != SchedulerGmMixPhase::RELEASED ||
        local->config.self_lane != scheduler_mix_accountant_lane(group.active_mask))
        return true;
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixAccount, group.task_ref);
    auto *signal = &local->cluster_signal->groups[bank];
    const auto fin = scheduler_mix_fin_status(signal, group.active_mask.raw(), group.generation);
    if (fin == SchedulerMixFin::PENDING) return true;
    if (fin == SchedulerMixFin::FAILED)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_ACK_INVALID);
    scheduler_cluster_barrier();
    if (group.completion_submitted || scheduler_cluster_load_field(&signal->armed_generation) != group.generation)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_ACK_INVALID);
    scheduler_protocol_cluster_all(base, local, true, bank);
    group.completion_submitted = true;
    if (!scheduler_publish_fanout(graph, base, local, run, group.task_ref, group.block_idx, group.active_mask))
        return false;
    group.phase = SchedulerGmMixPhase::IDLE;
    scheduler_cluster_store_release(&signal->accounted_generation, group.generation);
    group.settled_generation = group.generation;
    *progress = true;
    return true;
}

static inline __aicore__ bool scheduler_prepare_mix_lane(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, uint32_t bank
) {
    auto &group = local->mix_groups[bank];
    SchedulerScalarScope function_scope(local, SchedPhaseKind::ScalarClusterMixPrepareMember, group.task_ref);
    const uint32_t lane = group.subtask_slot;
    group.cluster_mix = true;
    group.state = SchedulerDispatchSlotState::FILLING;
    if (scheduler_phase_timing_enabled(profiling_level)) group.prepare_start_cycles = scheduler_cycles();
    // Receipt is reusable; the SSBUF generation, mask and real fanin still govern execution.
    const uint32_t ref = static_cast<uint32_t>(group.task_ref);
    if (ref == local->terminal_mix_refs[0] || ref == local->terminal_mix_refs[1]) {
        if (group.block_idx !=
            scheduler_fixed_mix_block(local->config.cluster_index, local->config.cluster_count, group.active_mask))
            return scheduler_admission_fail(
                graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_DESCRIPTOR
            );
        group.prepared = local->terminal_mix_calls[ref == local->terminal_mix_refs[1]];
    } else if (!scheduler_acquire_group_lane(
                   base, local->config.prepared, local->worker_id(), group.task_ref, group.block_idx,
                   static_cast<uint8_t>(lane), group.active_mask.raw(), &group.prepared
               ))
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    if (scheduler_protocol_enabled(local)) {
        auto *cell = scheduler_prepared_control_at(base, local->config.prepared, group.prepared.execution_index);
        scheduler_publish_signal(cell->state, group.prepared.claimed_state);
    }
    if (profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL)
        local->profiling->l2_mix[bank] = {scheduler_cycles(), 0};
    *progress = true;
    return true;
}

static inline __aicore__ __attribute__((noinline)) bool scheduler_cluster_mix_service_active(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool *progress, uint32_t bank
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixServiceActive);
    *progress = false;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    auto &group = local->mix_groups[bank];
    if (group.phase == SchedulerGmMixPhase::RELEASED) {
        if (!scheduler_account_mix_returns(graph, base, local, run, progress, bank)) return false;
        if (group.phase == SchedulerGmMixPhase::RELEASED) return true;
    }
    const bool first_observation = group.phase == SchedulerGmMixPhase::IDLE;
    if (first_observation) {
        if (!scheduler_cluster_mix_observe(graph, base, local, run, progress, bank)) return false;
        if (group.phase == SchedulerGmMixPhase::PUBLISHED &&
            !scheduler_prepare_mix_lane(graph, base, local, run, profiling_level, progress, bank))
            return false;
    }
    if (group.phase != SchedulerGmMixPhase::PUBLISHED && group.phase != SchedulerGmMixPhase::ORDER_READY &&
        group.phase != SchedulerGmMixPhase::ARRIVED)
        return true;
    auto *signal = &local->cluster_signal->groups[bank];
    if (group.phase == SchedulerGmMixPhase::PUBLISHED) {
        if (!scheduler_poll_mix_fanin(graph, base, local, run, progress, bank, first_observation)) return false;
        if (group.next_fanin != group.fanin.count) {
#if defined(__CCE_AICORE__)
            // The validated Call is immutable. Warm its first argument line once while fanin is pending.
            if (first_observation) {
                auto *call = scheduler_prepared_call_at(base, local->config.prepared, group.prepared.execution_index);
                dc_preload(call->args, 0);
            }
#endif
            return true;
        }
        if (profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL &&
            local->profiling->l2_mix[bank].fanin_seen_cycles == 0)
            local->profiling->l2_mix[bank].fanin_seen_cycles = scheduler_cycles();
        bool permit = false;
        if (!scheduler_mix_lane_order_ready(graph, local, run, group, &permit)) return false;
        if (!permit) return true;
        if (group.requires_peer_order) {
            // This core passed its order check and can run ordinary work while it waits for the other members.
            scheduler_cluster_store_release(&signal->order_ready[group.subtask_slot], group.generation);
            group.phase = SchedulerGmMixPhase::ORDER_READY;
            *progress = true;
        }
    }
    if (group.requires_peer_order || __builtin_expect(scheduler_protocol_enabled(local), false)) {
        if (group.phase != SchedulerGmMixPhase::ARRIVED) {
            if (group.phase == SchedulerGmMixPhase::ORDER_READY) {
                if (!scheduler_cluster_mix_all(signal->order_ready, group.active_mask, group.generation)) return true;
                // This lane may have claimed an earlier block while waiting for its peers.
                if (scheduler_owned_token_blocks_mix_arrival(local, group.rank)) return true;
            }
            if (group.sync_start) scheduler_protocol_member(base, local, group.task_ref, false);
            scheduler_protocol_cluster_ack(base, local, false, true, bank);
            if (group.requires_peer_order) {
                scheduler_cluster_store_release(&signal->arrived[group.subtask_slot], group.generation);
                group.phase = SchedulerGmMixPhase::ARRIVED;
            }
            scheduler_protocol_cluster_ack(base, local, false, false, bank);
            *progress = true;
        }
        // An earlier conflicting cohort can still require a peer to run older work.
        if (group.requires_peer_order &&
            !scheduler_cluster_mix_all(signal->arrived, group.active_mask, group.generation))
            return true;
        scheduler_protocol_cluster_all(base, local, false, bank);
    }
    if (__builtin_expect(local->profiling == nullptr, true)) {
        // Only the async-input flag is needed at entry, and this bank already owns it.
        group.state = SchedulerDispatchSlotState::READY;
    } else {
        // Sampled L0 also needs the timing slot carried by dispatch metadata.
        const auto metadata = scheduler_load_dispatch_metadata(base, local, group.task_ref);
        scheduler_publish_dispatch_slot(
            base, local, SCHEDULER_CLUSTER_MIX_EXECUTION + bank, metadata, profiling_level, group.prepare_start_cycles
        );
    }
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
    for (uint32_t bank = 0; bank < SCHEDULER_MIX_MEMBER_COUNT; ++bank) {
        const auto &group = local->mix_groups[bank];
        if (group.phase == SchedulerGmMixPhase::RUNNING) continue;
        if (group.phase == SchedulerGmMixPhase::IDLE) {
            const uint32_t generation =
                scheduler_cluster_load_field(&local->cluster_signal->groups[bank].armed_generation);
            if (!scheduler_cluster_mix_armed(generation) || generation == group.generation) continue;
        }
        bool advanced = false;
        if (!scheduler_cluster_mix_service_active(graph, base, local, run, profiling_level, &advanced, bank))
            return false;
        *progress = *progress || advanced;
    }
    return true;
}

static inline __aicore__ bool scheduler_validate_mix_entry(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixEnter);
    static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
    const auto &group = local->mix_groups[0];
    if (local->cluster_signal == nullptr)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    const auto *signal = &local->cluster_signal->groups[0];
    if (group.phase != SchedulerGmMixPhase::RUNNING ||
        scheduler_cluster_load_acquire(&signal->armed_generation) != group.generation ||
        scheduler_cluster_load_field(&local->cluster_signal->run_epoch) != local->run_epoch)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    return true;
}

template <bool RecordProtocol = true>
static inline __aicore__ bool scheduler_release_mix_lane(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool successful = true
) {
    static_assert(SCHEDULER_MIX_MEMBER_COUNT == 1);
    constexpr uint32_t bank = 0;
    auto &group = local->mix_groups[bank];
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixRelease, group.task_ref);
    if (local->cluster_signal == nullptr || group.phase != SchedulerGmMixPhase::RUNNING ||
        group.state != SchedulerDispatchSlotState::EXECUTING || group.prepared.claimed_state != 0)
        return scheduler_admission_fail(graph, local, run, group.task_ref, SchedulerErrorSite::GROUP_INVALID_STATE);
    auto *signal = &local->cluster_signal->groups[bank];
    const bool accountant = group.subtask_slot == scheduler_mix_accountant_lane(group.active_mask) &&
                            scheduler_mix_needs_receipt(local, signal);
    if constexpr (RecordProtocol) scheduler_protocol_cluster_ack(base, local, true, true, bank);
    scheduler_cluster_store_release(&signal->finished[group.subtask_slot], group.generation + (successful ? 0 : 1));
    if constexpr (RecordProtocol) {
        scheduler_protocol_cluster_ack(base, local, true, false, bank);
        if (group.sync_start) scheduler_protocol_member(base, local, group.task_ref, true);
    }
    if (!accountant) {
        if (successful) group.settled_generation = group.generation;
        group.phase = SchedulerGmMixPhase::IDLE;
        return true;
    }
    group.phase = SchedulerGmMixPhase::RELEASED;
    if (!successful) return true;
    bool accounted = false;
    return scheduler_account_mix_returns(graph, base, local, run, &accounted, bank);
}

static inline __aicore__ bool scheduler_cluster_mix_drain(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *done
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterMixDrain);
    *done = false;
    if (local->cluster_signal == nullptr) {
        *done = true;
        return true;
    }
    bool retired = true;
    for (uint32_t bank = 0; bank < SCHEDULER_MIX_MEMBER_COUNT; ++bank) {
        bool advanced = false;
        if (!scheduler_account_mix_returns(graph, base, local, run, &advanced, bank)) return false;
        const auto &group = local->mix_groups[bank];
        if (group.phase == SchedulerGmMixPhase::RELEASED) retired = false;
        else if (group.phase != SchedulerGmMixPhase::IDLE)
            return scheduler_admission_fail(
                graph, local, run, group.task_ref, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
            );
        if (local->is_mix_coordinator()) {
            const uint32_t generation =
                scheduler_cluster_load_field(&local->cluster_signal->groups[bank].armed_generation);
            if (scheduler_cluster_mix_armed(generation)) {
                bool recycled = false;
                if (!scheduler_cluster_mix_recycle(graph, local, run, bank, generation, &recycled)) return false;
                retired = retired && recycled;
            }
        }
    }
    *done = retired && (local->is_mix_coordinator() || !scheduler_cluster_owes_before(local, UINT32_MAX));
    return true;
}
