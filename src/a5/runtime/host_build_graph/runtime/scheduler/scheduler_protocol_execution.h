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
           record->pending_slot == pending_slot;
}

static inline __aicore__ void scheduler_protocol_subtask_complete(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerLocalSlotState &slot, uint32_t pending_slot,
    bool before, uint64_t ticket
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    scheduler_protocol_observe_record(record);
    if (!scheduler_protocol_execution_matches(local, record, slot, pending_slot) ||
        record->completion_consumed.sequence == 0 ||
        (before ? record->contribution_before.sequence != 0 :
                  record->contribution_before.sequence == 0 || record->contribution_after.sequence != 0)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(
            base, local, layout, before ? &record->contribution_before : &record->contribution_after
        ))
        return;
    record->contribution_worker = static_cast<uint32_t>(local->worker_id());
    if (!before) record->contribution_ticket = ticket;
    scheduler_protocol_publish_record(record);
}

// The final physical RMW already published logical completion. Reuse its
// actual interval rather than fabricating a later DONE-store event. The reader
// validates this explicit alias against the unique contribution ticket N-1.
static inline __aicore__ void scheduler_protocol_task_done_from_contribution(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerLocalSlotState &slot, uint32_t pending_slot,
    uint64_t ticket
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *execution =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (execution == nullptr) return;
    scheduler_protocol_observe_record(execution);
    const auto *metadata = scheduler_task_metadata_at(base, local, slot.task_ref);
    const uint64_t total = static_cast<uint64_t>(metadata->logical_block_num) * scheduler_task_lanes(*metadata).count();
    if (!scheduler_task_is_executable(metadata->flags) || scheduler_task_is_inline(metadata->flags) || total == 0 ||
        ticket != total - 1 || !scheduler_protocol_execution_matches(local, execution, slot, pending_slot) ||
        execution->run_epoch != layout->run_epoch || execution->contribution_worker != local->worker_id() ||
        execution->contribution_ticket != ticket || execution->contribution_before.sequence == 0 ||
        execution->contribution_after.sequence == 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *actor = &scheduler_state_at<SchedulerProtocolTaskTrace>(base, layout->tasks_offset)[slot.task_ref].completion;
    scheduler_protocol_observe_record(actor);
    if (actor->before.sequence != 0 || actor->after.sequence != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return;
    }
    actor->before.cycles = execution->contribution_before.cycles;
    actor->before.sequence = execution->contribution_before.sequence;
    actor->after.cycles = execution->contribution_after.cycles;
    actor->after.sequence = execution->contribution_after.sequence;
    actor->run_epoch = layout->run_epoch;
    actor->task_ref = slot.task_ref;
    actor->order_key = execution->order_key;
    actor->worker_index = static_cast<uint32_t>(local->worker_id());
    actor->flags = metadata->flags | SCHEDULER_PROTOCOL_COMPLETION_FROM_CONTRIBUTION;
    scheduler_protocol_publish_record(actor);
}

static inline __aicore__ void scheduler_protocol_execution_permit(
    __gm__ void *base, const SchedulerLocalState *local, uint32_t pending_slot, uint64_t tail, bool has_log
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (pending_slot >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &slot = local->slots[pending_slot];
    const bool ordered = scheduler_slot_uses_admission(local, slot);
    if (slot.gm_mix && (!ordered || local->admission == nullptr)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    uint64_t order = UINT64_MAX;
    if (ordered) {
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
    record->order_key = order;
    record->group_seq = slot.cluster_mix ? local->cluster_mix.generation :
                        slot.gm_mix      ? local->admission->gm_mix.group_seq :
                                           0;
    record->flags = SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED |
                    (ordered ? SCHEDULER_PROTOCOL_COOPERATIVE :
                     has_log ? SCHEDULER_PROTOCOL_PERMIT_TAIL :
                               SCHEDULER_PROTOCOL_PERMIT_NO_LOG) |
                    (slot.gm_mix ? SCHEDULER_PROTOCOL_GM_GROUP : 0) |
                    (slot.cluster_mix ? SCHEDULER_PROTOCOL_CLUSTER_GROUP : 0);
    record->permit_tail = ordered ? UINT64_MAX : tail;
    scheduler_protocol_publish_record(record);
}

static inline __aicore__ void scheduler_protocol_execution_entry(
    __gm__ void *base, SchedulerLocalState *local, uint32_t pending_slot, uint64_t function_address,
    const __gm__ LocalContext &local_context
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
    if (record->permit.sequence == 0 && layout->cooperative_task_count == 0)
        scheduler_protocol_execution_permit(base, local, pending_slot, 0, false);
    if (record->permit.sequence == 0 || !scheduler_protocol_execution_matches(local, record, slot, pending_slot)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->entry)) return;
    record->flags |= SCHEDULER_PROTOCOL_ENTRY | (function_address != 0 ? SCHEDULER_PROTOCOL_EXECUTED : 0);
    record->block_idx = static_cast<uint32_t>(local_context.block_idx);
    record->logical_block_num = static_cast<uint32_t>(local_context.block_num);
    record->cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    record->cluster_lane = local->config.self_lane;
    scheduler_protocol_publish_record(record);
}

static inline __aicore__ void scheduler_protocol_execution_returned(
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

static inline __aicore__ void scheduler_protocol_execution_consumed(
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

static inline __aicore__ void scheduler_protocol_execution_trace_published(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint32_t block, uint32_t subtask,
    uint32_t pending_slot
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *record = scheduler_protocol_execution_at(base, local, layout, task, block, subtask);
    if (record == nullptr) return;
    if (record->completion_consumed.sequence == 0 || record->worker_index != local->worker_id() ||
        record->pending_slot != pending_slot) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &record->trace_published)) return;
    scheduler_protocol_publish_record(record);
}
