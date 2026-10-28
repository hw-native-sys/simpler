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

#include "scheduler_memory.h"

inline __aicore__ uint64_t scheduler_cycles() {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

inline __aicore__ bool
scheduler_protocol_check_region(uint64_t offset, uint64_t count, uint64_t stride, uint64_t limit, uint64_t *end) {
    if ((offset & 127) != 0 || offset < *end || offset > limit || count > (limit - offset) / stride) return false;
    *end = offset + count * stride;
    return true;
}

template <typename Local>
inline __aicore__ bool scheduler_protocol_bind(__gm__ void *base, const Local *local) {
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

template <typename Local>
inline __aicore__ __gm__ SchedulerProtocolLayout *scheduler_protocol_layout_at(__gm__ void *base, const Local *local) {
    if (local == nullptr || local->profiling == nullptr || local->profiling->protocol_layout_offset == 0)
        return nullptr;
    return scheduler_state_at<SchedulerProtocolLayout>(base, local->profiling->protocol_layout_offset);
}

template <typename Local>
inline __aicore__ __gm__ SchedulerProtocolWorkerTrace *
scheduler_protocol_worker_at(__gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout) {
    return scheduler_state_at<SchedulerProtocolWorkerTrace>(base, layout->workers_offset) + local->worker_id();
}

template <typename Local>
inline __aicore__ void scheduler_protocol_fail(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout, uint64_t error
) {
    auto *worker = scheduler_protocol_worker_at(base, local, layout);
    worker->error_flags = worker->error_flags | error;
    scheduler_publish_cache_line(worker);
}

template <typename Local>
inline __aicore__ bool scheduler_protocol_stamp_at(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout,
    __gm__ SchedulerProtocolStamp *stamp, uint64_t cycles
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
    stamp->cycles = cycles;
    stamp->sequence = sequence + 1;
    worker->next_sequence = sequence + 1;
    return true;
}

template <typename Local>
inline __aicore__ bool scheduler_protocol_stamp(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout,
    __gm__ SchedulerProtocolStamp *stamp
) {
    return scheduler_protocol_stamp_at(base, local, layout, stamp, scheduler_cycles());
}

template <typename Local>
inline __aicore__ bool scheduler_protocol_enabled(const Local *local) {
    return local != nullptr && local->profiling != nullptr && local->profiling->protocol_layout_offset != 0;
}

template <typename Local>
inline __aicore__ uint64_t scheduler_protocol_cycles(const Local *local) {
    return scheduler_protocol_enabled(local) ? scheduler_cycles() : 0;
}

template <typename T>
inline __aicore__ void scheduler_protocol_publish_record(__gm__ T *record) {
    for (uint64_t offset = 0; offset < sizeof(T); offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(record) + offset);
    scheduler_cache_barrier();
}

template <typename T>
inline __aicore__ void scheduler_protocol_observe_record(__gm__ T *record) {
    for (uint64_t offset = 0; offset < sizeof(T); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(record) + offset);
    scheduler_cache_barrier();
}

template <typename Local>
inline __aicore__ void scheduler_protocol_task_actor(
    __gm__ void *base, const Local *local, int64_t task, uint64_t order, uint32_t flags, bool completion, bool before
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count ||
        (order >= layout->cooperative_task_count && !(completion && order == UINT64_MAX))) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *record = &scheduler_state_at<SchedulerProtocolTaskTrace>(base, layout->tasks_offset)[task];
    auto *actor = completion ? &record->completion : &record->admission;
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

template <typename Local>
inline __aicore__ __gm__ SchedulerProtocolExecutionTrace *scheduler_protocol_execution_at(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout, int64_t task, uint32_t block,
    uint32_t subtask
) {
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count || subtask >= 3) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    const auto &metadata = scheduler_state_at<SchedulerTaskMetadata>(base, local->config.task_metadata_offset)[task];
    const LaneMask mask = scheduler_task_lanes(metadata);
    const auto lane = static_cast<SubtaskSlot>(subtask);
    // Widened rather than taken from `scheduler_task_trace_index`, whose uint32
    // result would wrap before the bound below could reject it.
    const uint64_t index = static_cast<uint64_t>(metadata.trace_index_base) +
                           static_cast<uint64_t>(block) * static_cast<uint64_t>(mask.count()) +
                           static_cast<uint64_t>(mask.rank_of(lane));
    if (!mask.has(lane) || block >= metadata.logical_block_num || index >= layout->trace_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return nullptr;
    }
    return scheduler_state_at<SchedulerProtocolExecutionTrace>(base, layout->executions_offset) + index;
}

template <typename Local, typename Claim, typename Ready>
inline __aicore__ void scheduler_protocol_payload_publication(
    __gm__ void *base, Local *local, const Claim &claim, const Ready &ready, bool before
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr || claim.mix || claim.worker_id != local->worker_id()) return;
    if (claim.slot_index >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (scheduler_protocol_worker_at(base, local, layout)->finished != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_AFTER_FINISH);
        return;
    }
    const auto &slot = *local->slot_state(claim);
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    if (before) {
        scheduler_protocol_observe_record(record);
        if (record->permit.sequence != 0 || record->entry.sequence != 0 ||
            (record->publication_count != 0 &&
             (slot.cooperative || (record->flags & SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED) == 0 ||
              record->worker_index != local->worker_id() || record->run_epoch != layout->run_epoch ||
              record->task_ref != slot.task_ref || record->block_idx != slot.block_idx ||
              record->subtask_slot != slot.subtask_slot ||
              (record->pending_slot == claim.slot_index && record->generation >= slot.generation)))) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
            return;
        }
        if (record->publication_count == UINT32_MAX) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_SEQUENCE_OVERFLOW);
            return;
        }
        ++record->publication_count;
        record->publication_before_cycles = scheduler_cycles();
        record->publication_after_cycles = 0;
        record->flags = 0;
        record->run_epoch = layout->run_epoch;
        record->task_ref = slot.task_ref;
        record->block_idx = slot.block_idx;
        record->worker_index = static_cast<uint32_t>(local->worker_id());
        record->pending_slot = static_cast<uint16_t>(claim.slot_index);
        record->subtask_slot = slot.subtask_slot;
        record->generation = slot.generation;
        record->ready_source = static_cast<uint32_t>(ready.source);
        record->publication_mode = static_cast<uint32_t>(ready.publication_mode);
    } else {
        if (record->publication_count == 0 || record->worker_index != local->worker_id() ||
            record->task_ref != slot.task_ref || record->generation != slot.generation ||
            record->pending_slot != claim.slot_index || record->subtask_slot != slot.subtask_slot ||
            record->block_idx != slot.block_idx || record->run_epoch != layout->run_epoch) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        if ((record->flags & SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED) != 0) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_DUPLICATE);
            return;
        }
        record->publication_after_cycles = scheduler_cycles();
        record->flags |= SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED;
    }
    scheduler_protocol_publish_record(record);
}

template <typename Local>
inline __aicore__ void scheduler_protocol_admission(
    __gm__ void *base, const Local *local, int64_t task, uint64_t order, uint32_t flags, bool before
) {
    scheduler_protocol_task_actor(base, local, task, order, flags, false, before);
}

template <typename Local>
inline __aicore__ void scheduler_protocol_task_done(__gm__ void *base, const Local *local, int64_t task, bool before) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto *source = scheduler_state_at<SchedulerTaskMetadata>(base, local->config.task_metadata_offset) + task;
    SchedulerTaskMetadata metadata{};
    metadata.flags = source->flags;
    metadata.logical_block_num = source->logical_block_num;
    uint64_t order = UINT64_MAX;
    if (scheduler_task_requires_admission(metadata)) {
        const auto *admission = local->admission;
        if (admission == nullptr || admission->current_task_ref != task || admission->epoch != layout->run_epoch) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        order = admission->scan_cursor;
    }
    scheduler_protocol_task_actor(base, local, task, order, metadata.flags, true, before);
}

template <typename Local, typename Admission>
inline __aicore__ __gm__ SchedulerProtocolMemberTrace *scheduler_protocol_member_at(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout, const Admission *admission
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

template <typename Local, typename Admission>
inline __aicore__ void
scheduler_protocol_member_observed(__gm__ void *base, const Local *local, const Admission *admission, uint64_t tail) {
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

template <typename Local, typename Admission>
inline __aicore__ void scheduler_protocol_member_closed(
    __gm__ void *base, const Local *local, const Admission *admission, uint64_t cursor, uint64_t seal,
    uint64_t last_group_seq
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

template <typename Local>
inline __aicore__ void scheduler_protocol_finish(__gm__ void *base, const Local *local, bool success) {
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
