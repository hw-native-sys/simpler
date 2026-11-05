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

inline __aicore__ bool scheduler_protocol_execution_matches(
    const SchedulerLocalState *local, __gm__ const SchedulerProtocolExecutionTrace *record,
    const SchedulerLocalSlotState &slot, uint32_t pending_slot
) {
    return record->worker_index == local->worker_id() && record->task_ref == slot.task_ref &&
           record->block_idx == slot.block_idx && record->subtask_slot == slot.subtask_slot &&
           record->pending_slot == pending_slot && record->generation == slot.generation;
}

inline __aicore__ void scheduler_protocol_execution_permit(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t pending_slot, uint64_t tail, bool has_log
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (pending_slot >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &slot = local->slots[pending_slot];
    if (slot.gm_mix && (!slot.cooperative || local->admission == nullptr)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    uint64_t order = UINT64_MAX;
    if (slot.cooperative) {
        if (local->admission == nullptr || local->admission->epoch != layout->run_epoch ||
            local->admission->current_task_ref != slot.task_ref) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        order = local->admission->scan_cursor;
    }
    scheduler_protocol_observe_record(record);
    if ((record->flags & SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED) == 0 ||
        !scheduler_protocol_execution_matches(local, record, slot, pending_slot)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->permit)) return;
    record->run_epoch = layout->run_epoch;
    record->task_ref = slot.task_ref;
    record->block_idx = slot.block_idx;
    record->worker_index = static_cast<uint32_t>(local->worker_id());
    record->pending_slot = static_cast<uint16_t>(pending_slot);
    record->subtask_slot = slot.subtask_slot;
    record->generation = slot.generation;
    record->order_key = order;
    record->group_seq = slot.gm_mix ? local->admission->gm_mix.group_seq : 0;
    record->flags = SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED |
                    (slot.cooperative ? SCHEDULER_PROTOCOL_COOPERATIVE :
                     has_log          ? SCHEDULER_PROTOCOL_PERMIT_TAIL :
                                        SCHEDULER_PROTOCOL_PERMIT_NO_LOG) |
                    (slot.gm_mix ? SCHEDULER_PROTOCOL_GM_GROUP : 0);
    record->permit_tail = slot.cooperative ? UINT64_MAX : tail;
    scheduler_protocol_publish_record(record);
}

inline __aicore__ void scheduler_protocol_execution_entry(
    __gm__ void *base, SchedulerLocalState *local, uint32_t pending_slot, __gm__ const DispatchPayload *payload
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (pending_slot >= SCHEDULER_PENDING_SLOT_COUNT || payload == nullptr) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &slot = local->slots[pending_slot];
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    if (record->permit.sequence == 0 && layout->cooperative_task_count == 0)
        scheduler_protocol_execution_permit(base, local, pending_slot, 0, false);
    if (record->permit.sequence == 0 || !scheduler_protocol_execution_matches(local, record, slot, pending_slot)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->entry)) return;
    record->flags |= SCHEDULER_PROTOCOL_ENTRY | (payload->function_bin_addr != 0 ? SCHEDULER_PROTOCOL_EXECUTED : 0);
    record->block_idx = static_cast<uint32_t>(payload->local_context.block_idx);
    record->logical_block_num = static_cast<uint32_t>(payload->local_context.block_num);
    record->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    record->cluster_lane = local->config.self_lane;
    scheduler_protocol_publish_record(record);
}

inline __aicore__ void scheduler_protocol_execution_returned(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t pending_slot, uint64_t returned, uint64_t visible
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (pending_slot >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &slot = local->slots[pending_slot];
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    if (record->entry.sequence == 0 || !scheduler_protocol_execution_matches(local, record, slot, pending_slot) ||
        (record->flags & SCHEDULER_PROTOCOL_EXECUTED) == 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp_at(base, local, layout, &record->returned, returned)) return;
    scheduler_protocol_stamp_at(base, local, layout, &record->visible, visible);
    scheduler_protocol_publish_record(record);
}

inline __aicore__ void scheduler_protocol_execution_consumed(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t pending_slot, const SchedulerLocalSlotState &slot
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    if (!scheduler_protocol_execution_matches(local, record, slot, pending_slot) || record->entry.sequence == 0 ||
        ((record->flags & SCHEDULER_PROTOCOL_EXECUTED) != 0 && record->visible.sequence == 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->completion_consumed)) return;
    scheduler_protocol_publish_record(record);
}

inline __aicore__ void scheduler_protocol_execution_trace_published(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, uint32_t subtask,
    uint32_t pending_slot, uint32_t generation
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_execution_at(base, local, layout, task, block, subtask);
    if (record == nullptr) return;
    if (record->completion_consumed.sequence == 0 || record->worker_index != local->worker_id() ||
        record->generation != generation || record->pending_slot != pending_slot) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->trace_published)) return;
    scheduler_protocol_publish_record(record);
}
