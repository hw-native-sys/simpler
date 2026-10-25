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

#include "scheduler_protocol.h"

inline __aicore__ __gm__ SchedulerProtocolBlockTrace *scheduler_protocol_block_at(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout, uint32_t block
) {
    const auto *admission = local->admission;
    if (admission == nullptr || admission->epoch != layout->run_epoch || admission->current_task_ref < 0 ||
        static_cast<uint64_t>(admission->current_task_ref) >= layout->task_count ||
        admission->scan_cursor >= layout->cooperative_task_count || block >= admission->current_block_num) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    const auto *metadata = scheduler_task_metadata_at(base, local, admission->current_task_ref);
    const uint8_t mask = metadata->active_mask;
    const uint64_t index =
        static_cast<uint64_t>(metadata->trace_index_base) + static_cast<uint64_t>(block) * __builtin_popcount(mask);
    if (mask == 0 || (mask & ~7U) != 0 || mask != admission->current_active_mask ||
        metadata->logical_block_num != admission->current_block_num || index >= layout->trace_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolBlockTrace>(base, layout->blocks_offset) + index;
}

inline __aicore__ bool scheduler_protocol_owner_matches(
    const SchedulerLocalState *local, __gm__ const SchedulerProtocolBlockOwner *owner, uint32_t block
) {
    const auto &admission = *local->admission;
    return owner->claim_after.sequence != 0 && owner->run_epoch == admission.epoch &&
           owner->task_ref == admission.current_task_ref && owner->order_key == admission.scan_cursor &&
           owner->block_idx == block && owner->worker_index == local->worker_id();
}

inline __aicore__ void scheduler_protocol_block_claim(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t block, uint64_t before, uint64_t after
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    scheduler_protocol_observe_record(owner);
    if (!scheduler_protocol_stamp_at(base, local, layout, &owner->claim_before, before)) return;
    const auto &admission = *local->admission;
    owner->run_epoch = admission.epoch;
    owner->task_ref = admission.current_task_ref;
    owner->order_key = admission.scan_cursor;
    owner->worker_index = static_cast<uint32_t>(local->worker_id());
    owner->block_idx = block;
    owner->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    owner->logical_block_num = admission.current_block_num;
    owner->active_mask = admission.current_active_mask;
    owner->flags = admission.current_flags;
    owner->completion_ticket = UINT64_MAX;
    scheduler_protocol_stamp_at(base, local, layout, &owner->claim_after, after);
    scheduler_protocol_publish_record(owner);
}

inline __aicore__ void scheduler_protocol_block_complete(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t block, bool before, uint64_t ticket
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    if (!scheduler_protocol_owner_matches(local, owner, block) || (!before && owner->completion_before.sequence == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &owner->completion_before : &owner->completion_after))
        return;
    if (!before) owner->completion_ticket = ticket;
    scheduler_protocol_publish_record(owner);
}

inline __aicore__ void
scheduler_protocol_group_publish(__gm__ void *base, const SchedulerLocalState *local, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->admission->gm_mix;
    auto *record = scheduler_protocol_block_at(base, local, layout, group.block_idx);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    if (!scheduler_protocol_owner_matches(local, owner, group.block_idx) || group.group_seq == 0 ||
        (!before && (owner->publish_before.sequence == 0 || owner->group_seq != group.group_seq))) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &owner->publish_before : &owner->publish_after)) return;
    if (before) owner->group_seq = group.group_seq;
    scheduler_protocol_publish_record(owner);
}

inline __aicore__ void scheduler_protocol_group_descriptor(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerGroupDescriptor *descriptor
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &admission = *local->admission;
    const auto &group = admission.gm_mix;
    auto *record = scheduler_protocol_block_at(base, local, layout, group.block_idx);
    if (record == nullptr) return;
    if (local->config.self_lane >= 3 || descriptor->run_epoch != admission.epoch ||
        descriptor->task_ref != admission.current_task_ref || descriptor->order_key != admission.scan_cursor ||
        descriptor->group_seq != group.group_seq || descriptor->block_idx != group.block_idx) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *lane = &record->lanes[local->config.self_lane];
    scheduler_protocol_observe_record(lane);
    if (!scheduler_protocol_stamp(base, local, layout, &lane->descriptor_ready)) return;
    lane->run_epoch = descriptor->run_epoch;
    lane->task_ref = descriptor->task_ref;
    lane->order_key = descriptor->order_key;
    lane->group_seq = descriptor->group_seq;
    lane->worker_index = static_cast<uint32_t>(local->worker_id());
    lane->block_idx = descriptor->block_idx;
    lane->cluster_index = descriptor->cluster_index;
    lane->lane = local->config.self_lane;
    lane->active_mask = descriptor->active_mask;
    lane->flags = descriptor->flags;
    lane->logical_block_num = descriptor->logical_block_num;
    lane->resource_mask = descriptor->resource_mask;
    for (uint32_t peer = 0; peer < 3; ++peer)
        lane->lane_subtasks[peer] = descriptor->lane_subtasks[peer];
    scheduler_protocol_publish_record(lane);
}

inline __aicore__ bool
scheduler_protocol_lane_matches(const SchedulerLocalState *local, __gm__ const SchedulerProtocolLaneTrace *lane) {
    const auto &admission = *local->admission;
    return lane->descriptor_ready.sequence != 0 && lane->run_epoch == admission.epoch &&
           lane->task_ref == admission.current_task_ref && lane->order_key == admission.scan_cursor &&
           lane->group_seq == admission.gm_mix.group_seq && lane->block_idx == admission.gm_mix.block_idx &&
           lane->worker_index == local->worker_id() && lane->lane == local->config.self_lane;
}

inline __aicore__ void
scheduler_protocol_lane_publication(__gm__ void *base, const SchedulerLocalState *local, bool release, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, local->admission->gm_mix.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    if (!scheduler_protocol_lane_matches(local, lane) || (release && lane->arrivals_observed.sequence == 0) ||
        (!before && (release ? lane->release_before.sequence : lane->arrive_before.sequence) == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *stamp = release ? (before ? &lane->release_before : &lane->release_after) :
                            (before ? &lane->arrive_before : &lane->arrive_after);
    if (!scheduler_protocol_stamp(base, local, layout, stamp)) return;
    scheduler_publish_cache_line(lane);
}

inline __aicore__ void scheduler_protocol_group_acks(
    __gm__ void *base, const SchedulerLocalState *local, bool releases, const uint64_t *observed
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    const auto &group = local->admission->gm_mix;
    auto *record = scheduler_protocol_block_at(base, local, layout, group.block_idx);
    if (record == nullptr) return;
    auto *lane = &record->lanes[local->config.self_lane];
    if ((releases && !scheduler_protocol_owner_matches(local, &record->owner, group.block_idx)) ||
        (!releases && !scheduler_protocol_lane_matches(local, lane))) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *stamp = releases ? &record->owner.releases_observed : &lane->arrivals_observed;
    auto *values = releases ? record->owner.released_seq : lane->arrived_seq;
    for (uint32_t peer = 0; peer < 3; ++peer) {
        if (observed[peer] != group.group_seq || (stamp->sequence != 0 && values[peer] != observed[peer])) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
    }
    if (stamp->sequence != 0) return;
    if (!scheduler_protocol_stamp(base, local, layout, stamp)) return;
    for (uint32_t peer = 0; peer < 3; ++peer)
        values[peer] = observed[peer];
    if (releases) scheduler_protocol_publish_record(&record->owner);
    else scheduler_protocol_publish_record(lane);
}
