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

inline __aicore__ bool
scheduler_protocol_check_region(uint64_t offset, uint64_t count, uint64_t stride, uint64_t limit, uint64_t *end) {
    if ((offset & 127) != 0 || offset < *end || offset > limit || count > (limit - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}

inline __aicore__ bool scheduler_protocol_bind(__gm__ void *base, const SchedulerLocalState *local) {
    if (base == nullptr || local == nullptr || local->config.graph == nullptr ||
        local->config.graph_summary_offset == 0)
        return false;
    if (local->profiling != nullptr) local->profiling->protocol_layout_offset = 0;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    scheduler_observe_cache_line(&summary->protocol_layout_offset);
    const uint64_t offset = summary->protocol_layout_offset;
    if (offset == 0) return true;
    if (local->profiling == nullptr || (offset & 127) != 0 ||
        offset > UINT64_MAX - reinterpret_cast<uintptr_t>(base) - sizeof(SchedulerProtocolLayout))
        return false;
    auto *layout = scheduler_state_at<SchedulerProtocolLayout>(base, offset);
    for (uint64_t line = 0; line < sizeof(*layout); line += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(layout) + line);
    if (layout->magic != SCHEDULER_PROTOCOL_MAGIC || layout->run_epoch != summary->run_epoch ||
        layout->task_count != local->config.graph->task_count ||
        layout->task_count > static_cast<uint64_t>(INT32_MAX) || layout->worker_count == 0 ||
        layout->worker_count > SCHEDULER_WORKER_CAPACITY ||
        layout->worker_count != local->config.runtime_worker_count || local->worker_id() >= layout->worker_count ||
        layout->cooperative_task_count != summary->cooperative_task_count ||
        layout->cooperative_task_count > layout->task_count ||
        layout->member_count != layout->cooperative_task_count * layout->worker_count ||
        layout->storage_end_offset > UINT64_MAX - reinterpret_cast<uintptr_t>(base))
        return false;
    uint64_t end = offset + sizeof(*layout);
    const uint64_t limit = layout->storage_end_offset;
    if (!scheduler_protocol_check_region(
            layout->tasks_offset, layout->task_count, sizeof(SchedulerProtocolTaskTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->members_offset, layout->member_count, sizeof(SchedulerProtocolMemberTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->blocks_offset, layout->trace_count, sizeof(SchedulerProtocolBlockTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->executions_offset, layout->trace_count, sizeof(SchedulerProtocolExecutionTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->workers_offset, layout->worker_count, sizeof(SchedulerProtocolWorkerTrace), limit, &end
        ) ||
        end != limit)
        return false;
    auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(base, layout->workers_offset) + local->worker_id();
    scheduler_observe_cache_line(worker);
    if (worker->run_epoch != layout->run_epoch || worker->worker_index != local->worker_id() ||
        worker->next_sequence != 0 || worker->error_flags != 0 || worker->finished != 0)
        return false;
    local->profiling->protocol_layout_offset = offset;
    return true;
}

inline __aicore__ __gm__ SchedulerProtocolLayout *
scheduler_protocol_layout_at(__gm__ void *base, const SchedulerLocalState *local) {
    if (local == nullptr || local->profiling == nullptr || local->profiling->protocol_layout_offset == 0)
        return nullptr;
    return scheduler_state_at<SchedulerProtocolLayout>(base, local->profiling->protocol_layout_offset);
}

inline __aicore__ __gm__ SchedulerProtocolWorkerTrace *scheduler_protocol_worker_at(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout
) {
    return scheduler_state_at<SchedulerProtocolWorkerTrace>(base, layout->workers_offset) + local->worker_id();
}

inline __aicore__ void scheduler_protocol_fail(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout, uint64_t error
) {
    auto *worker = scheduler_protocol_worker_at(base, local, layout);
    worker->error_flags = worker->error_flags | error;
    scheduler_publish_cache_line(worker);
}

inline __aicore__ bool scheduler_protocol_stamp(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout,
    __gm__ SchedulerProtocolStamp *stamp
) {
    auto *worker = scheduler_protocol_worker_at(base, local, layout);
    if (stamp->sequence != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return false;
    }
    if (worker->finished != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_AFTER_FINISH);
        return false;
    }
    const uint64_t sequence = worker->next_sequence;
    if (sequence == UINT64_MAX) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_SEQUENCE_OVERFLOW);
        return false;
    }
    stamp->cycles = scheduler_cycles();
    stamp->sequence = sequence + 1;
    worker->next_sequence = sequence + 1;
    return true;
}

inline __aicore__ void scheduler_protocol_admission(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint64_t order, uint32_t flags, bool before
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count || order >= layout->cooperative_task_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *actor = &scheduler_state_at<SchedulerProtocolTaskTrace>(base, layout->tasks_offset)[task].admission;
    if (before) {
        scheduler_observe_cache_line(actor);
        if (!scheduler_protocol_stamp(base, local, layout, &actor->before)) return;
        actor->run_epoch = layout->run_epoch;
        actor->task_ref = task;
        actor->order_key = order;
        actor->worker_index = static_cast<uint32_t>(local->worker_id());
        actor->flags = flags;
    } else {
        if (actor->before.sequence == 0 || actor->task_ref != task || actor->order_key != order ||
            actor->run_epoch != layout->run_epoch || actor->worker_index != local->worker_id()) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        if (!scheduler_protocol_stamp(base, local, layout, &actor->after)) return;
    }
    scheduler_publish_cache_line(actor);
}

inline __aicore__ __gm__ SchedulerProtocolMemberTrace *scheduler_protocol_member_at(
    __gm__ void *base, const SchedulerLocalState *local, __gm__ const SchedulerProtocolLayout *layout,
    const SchedulerAdmissionLocalState *admission
) {
    if (admission == nullptr || admission->epoch != layout->run_epoch ||
        admission->scan_cursor >= layout->cooperative_task_count || admission->current_task_ref < 0 ||
        static_cast<uint64_t>(admission->current_task_ref) >= layout->task_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolMemberTrace>(base, layout->members_offset) +
           admission->scan_cursor * layout->worker_count + local->worker_id();
}

inline __aicore__ void scheduler_protocol_member_observed(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerAdmissionLocalState *admission, uint64_t tail
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *member = scheduler_protocol_member_at(base, local, layout, admission);
    if (member == nullptr) return;
    scheduler_observe_cache_line(member);
    scheduler_observe_cache_line(&member->closed_cursor);
    if (!scheduler_protocol_stamp(base, local, layout, &member->observed)) return;
    member->run_epoch = admission->epoch;
    member->task_ref = admission->current_task_ref;
    member->order_key = admission->scan_cursor;
    member->worker_index = static_cast<uint32_t>(local->worker_id());
    member->flags = admission->current_flags;
    member->observed_tail = tail;
    scheduler_writeback_cache_line(member);
    scheduler_publish_cache_line(&member->closed_cursor);
}

inline __aicore__ void scheduler_protocol_member_closed(
    __gm__ void *base, const SchedulerLocalState *local, const SchedulerAdmissionLocalState *admission, uint64_t cursor,
    uint64_t seal, uint64_t last_group_seq
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *member = scheduler_protocol_member_at(base, local, layout, admission);
    if (member == nullptr) return;
    if (member->observed.sequence == 0 || member->run_epoch != admission->epoch ||
        member->task_ref != admission->current_task_ref || member->order_key != admission->scan_cursor ||
        member->worker_index != local->worker_id()) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, &member->closed)) return;
    member->closed_cursor = cursor;
    member->close_seal = seal;
    member->last_group_seq = last_group_seq;
    member->pending_mask = 0;
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        const auto &slot = local->slots[index];
        if (slot.cooperative && slot.task_ref == admission->current_task_ref &&
            slot.state != SchedulerDispatchSlotState::FREE)
            member->pending_mask |= UINT64_C(1) << index;
    }
    scheduler_writeback_cache_line(member);
    scheduler_publish_cache_line(&member->closed_cursor);
}

inline __aicore__ void scheduler_protocol_finish(__gm__ void *base, const SchedulerLocalState *local, bool success) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    auto *worker = scheduler_protocol_worker_at(base, local, layout);
    if (worker->finished != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
        return;
    }
    worker->finished = success ? UINT64_C(1) : UINT64_C(2);
    scheduler_publish_cache_line(worker);
}
