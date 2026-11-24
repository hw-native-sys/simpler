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

#include "scheduler_ready.h"

static inline __aicore__ __gm__ SchedulerProtocolBlockTrace *scheduler_protocol_block_at(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout, int64_t task,
    uint32_t block
) {
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count || local->config.task_catalog_offset == 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    const auto *metadata = scheduler_task_metadata_at(base, local, task);
    const LaneMask mask = scheduler_task_lanes(*metadata);
    const uint64_t index =
        static_cast<uint64_t>(metadata->trace_index_base) + static_cast<uint64_t>(block) * mask.count();
    if (mask.is_empty() || !mask.is_valid() || block >= metadata->logical_block_num || index >= layout->trace_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolBlockTrace>(base, layout->blocks_offset) + index;
}

inline __aicore__ bool scheduler_protocol_owner_matches(
    const SchedulerLocalState *local, __gm__ const SchedulerProtocolBlockOwner *owner, uint64_t epoch, int64_t task,
    uint32_t block
) {
    return owner->claim_after.sequence != 0 && owner->run_epoch == epoch && owner->task_ref == task &&
           owner->block_idx == block && owner->worker_index == local->worker_id();
}

static inline __aicore__ void scheduler_protocol_block_claim(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, uint64_t before, uint64_t after,
    uint64_t confirmed_count = 0
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, task, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    scheduler_protocol_observe_record(owner);
    if (!scheduler_protocol_stamp_at(base, local, layout, &owner->claim_before, before)) return;
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto *metadata = scheduler_task_metadata_at(base, local, task);
    owner->run_epoch = layout->run_epoch;
    owner->task_ref = task;
    owner->order_key = scheduler_task_catalog_order(base, catalog, static_cast<uint32_t>(task)).rank;
    owner->worker_index = static_cast<uint32_t>(local->worker_id());
    owner->block_idx = block;
    owner->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    owner->logical_block_num = metadata->logical_block_num;
    scheduler_lane_mask_store(owner->active_mask, scheduler_task_lanes(*metadata));
    owner->flags = metadata->flags;
    owner->completion_ticket = UINT64_MAX;
    scheduler_protocol_stamp_at(base, local, layout, &owner->claim_after, after);
    scheduler_protocol_publish_record(owner);
    auto *claim = &record->claim;
    scheduler_protocol_observe_record(claim);
    // Bitmap reporters discover the unique publisher after the owner has bound its execution.
    claim->confirmed_count = scheduler_task_control_at(base, local, task)->supply_address == 0 ? confirmed_count : 0;
    claim->worker_index = owner->worker_index;
    claim->block_idx = block;
    claim->run_epoch = layout->run_epoch;
    scheduler_protocol_publish_record(claim);
}

static inline __aicore__ void scheduler_protocol_claim_seal(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, bool before, uint64_t value
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, task, block);
    if (record == nullptr) return;
    auto *claim = &record->claim;
    const auto *metadata = scheduler_task_metadata_at(base, local, task);
    if (value != metadata->logical_block_num || value == 0 ||
        (!before && (claim->published_before.sequence == 0 || claim->run_epoch != layout->run_epoch ||
                     claim->worker_index != local->worker_id() || claim->block_idx != block))) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &claim->published_before : &claim->published_after))
        return;
    claim->run_epoch = layout->run_epoch;
    claim->worker_index = static_cast<uint32_t>(local->worker_id());
    claim->block_idx = block;
    claim->confirmed_count = value;
    claim->seal_value = value;
    scheduler_protocol_publish_record(claim);
}

static inline __aicore__ void scheduler_protocol_block_complete(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, bool before, uint64_t ticket,
    uint32_t credits = 0
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_block_at(base, local, layout, task, block);
    if (record == nullptr) return;
    auto *owner = &record->owner;
    if (!scheduler_protocol_owner_matches(local, owner, layout->run_epoch, task, block) ||
        (!before && owner->completion_before.sequence == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &owner->completion_before : &owner->completion_after))
        return;
    if (!before) owner->completion_ticket = ticket;
    owner->completion_worker = static_cast<uint32_t>(local->worker_id());
    owner->completion_credits = credits;
    scheduler_protocol_publish_record(owner);
}
