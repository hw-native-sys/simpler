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

#include "scheduler_task_catalog.h"

#if defined(__CPU_SIM)
#include "aicpu/device_time.h"
#endif

inline __aicore__ uint64_t scheduler_cycles() {
#if defined(__CPU_SIM)
    return sys_cnt_now_ticks();
#elif defined(__CCE_AICORE__)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

inline __aicore__ bool
scheduler_protocol_check_region(uint64_t offset, uint64_t count, uint64_t stride, uint64_t limit, uint64_t *end) {
    if ((offset & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 || offset < *end || offset > limit ||
        count > (limit - offset) / stride)
        return false;
    *end = offset + count * stride;
    return true;
}

template <typename Local>
static inline __aicore__ bool scheduler_protocol_bind(__gm__ void *base, const Local *local) {
    if (base == nullptr || local == nullptr || local->config.graph == nullptr ||
        local->config.graph_summary_offset == 0)
        return false;
    if (local->profiling != nullptr) local->profiling->protocol_layout_offset = 0;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    scheduler_observe_cache_line(&summary->protocol_layout_offset);
    const uint64_t offset = summary->protocol_layout_offset;
    if (offset == 0) return true;
    if (local->profiling == nullptr || (offset & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 ||
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
        summary->sync_mix_task_count > summary->mix_task_count ||
        layout->cooperative_task_count !=
            summary->cooperative_task_count + (summary->mix_task_count - summary->sync_mix_task_count) ||
        layout->floor_count !=
            (summary->mix_task_count != 0 ? (summary->mix_task_count + 1) * summary->cluster_count : 0) ||
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
            layout->floors_offset, layout->floor_count, sizeof(SchedulerProtocolFloorTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->blocks_offset, layout->trace_count, sizeof(SchedulerProtocolBlockTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->executions_offset, layout->trace_count, sizeof(SchedulerProtocolExecutionTrace), limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->condition_contributions_offset, layout->trace_count, sizeof(SchedulerProtocolConditionContribution),
            limit, &end
        ) ||
        !scheduler_protocol_check_region(
            layout->counter_observations_offset, layout->task_count, sizeof(SchedulerProtocolCounterObservation), limit,
            &end
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
    local->profiling->published_mix_count = 0;
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
static inline __aicore__ void scheduler_protocol_fail(
    __gm__ void *base, const Local *local, __gm__ const SchedulerProtocolLayout *layout, uint64_t error
) {
    auto *worker = scheduler_protocol_worker_at(base, local, layout);
    worker->error_flags = worker->error_flags | error;
    scheduler_publish_cache_line(worker);
}

template <typename Local>
static inline __aicore__ bool scheduler_protocol_stamp_at(
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
static inline __aicore__ bool scheduler_protocol_stamp(
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
static inline __aicore__ void scheduler_protocol_publish_record(__gm__ T *record) {
    for (uint64_t offset = 0; offset < sizeof(T); offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ uint8_t *>(record) + offset);
    scheduler_cache_barrier();
}

template <typename T>
static inline __aicore__ void scheduler_protocol_observe_record(__gm__ T *record) {
    for (uint64_t offset = 0; offset < sizeof(T); offset += 64)
        scheduler_invalidate_cache_line(reinterpret_cast<__gm__ uint8_t *>(record) + offset);
    scheduler_cache_barrier();
}

template <typename Local>
static inline __aicore__ void scheduler_protocol_task_actor(
    __gm__ void *base, const Local *local, int64_t task, uint32_t flags, bool before, uint64_t cycles = UINT64_MAX
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count || local->config.task_catalog_offset == 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const uint32_t order = scheduler_task_catalog_order(base, catalog, static_cast<uint32_t>(task)).rank;
    auto *actor = &scheduler_state_at<SchedulerProtocolTaskTrace>(base, layout->tasks_offset)[task].completion;
    if (before) {
        scheduler_observe_cache_line(actor);
        if (!(cycles == UINT64_MAX ? scheduler_protocol_stamp(base, local, layout, &actor->before) :
                                     scheduler_protocol_stamp_at(base, local, layout, &actor->before, cycles)))
            return;
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
        if (!(cycles == UINT64_MAX ? scheduler_protocol_stamp(base, local, layout, &actor->after) :
                                     scheduler_protocol_stamp_at(base, local, layout, &actor->after, cycles)))
            return;
    }
    scheduler_publish_cache_line(actor);
}

template <typename Local>
static inline __aicore__ __gm__ SchedulerProtocolExecutionTrace *scheduler_protocol_execution_at(
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

template <typename Local, typename Ready>
static inline __aicore__ void scheduler_protocol_payload_publication(
    __gm__ void *base, Local *local, uint32_t slot_index, const Ready &ready, bool before
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (slot_index >= SCHEDULER_LOCAL_EXECUTION_COUNT) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (scheduler_protocol_worker_at(base, local, layout)->finished != 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_AFTER_FINISH);
        return;
    }
    const auto &slot = local->execution_at(slot_index);
    auto *record =
        scheduler_protocol_execution_at(base, local, layout, slot.task_ref, slot.block_idx, slot.subtask_slot);
    if (record == nullptr) return;
    if (before) {
        scheduler_protocol_observe_record(record);
        if (record->permit.sequence != 0 || record->entry.sequence != 0 || record->publication_count != 0) {
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
        record->pending_slot = scheduler_trace_source_index(slot_index);
        record->subtask_slot = slot.subtask_slot;
        record->ready_source = static_cast<uint32_t>(ready.source);
    } else {
        if (record->publication_count == 0 || record->worker_index != local->worker_id() ||
            record->task_ref != slot.task_ref || record->pending_slot != scheduler_trace_source_index(slot_index) ||
            record->subtask_slot != slot.subtask_slot || record->block_idx != slot.block_idx ||
            record->run_epoch != layout->run_epoch) {
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
static inline __aicore__ void
scheduler_protocol_task_done(__gm__ void *base, const Local *local, int64_t task, bool before, bool deferred = false) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, local->config.task_metadata_offset);
    const uint32_t flags = metadata[task].flags | (deferred ? SCHEDULER_PROTOCOL_DEFERRED_FINISH : 0);
    scheduler_protocol_task_actor(base, local, task, flags, before);
}

template <typename Local>
static inline __aicore__ void
scheduler_protocol_member(__gm__ void *base, const Local *local, int64_t task, bool retire) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (task < 0 || static_cast<uint64_t>(task) >= layout->task_count || local->config.task_catalog_offset == 0) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto &order = scheduler_task_catalog_order(base, catalog, static_cast<uint32_t>(task));
    const uint32_t worker = static_cast<uint32_t>(local->worker_id());
    if (order.cooperative_index >= layout->cooperative_task_count || worker >= layout->worker_count) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const uint64_t index = static_cast<uint64_t>(order.cooperative_index) * layout->worker_count + worker;
    auto *record = scheduler_state_at<SchedulerProtocolMemberTrace>(base, layout->members_offset) + index;
    scheduler_observe_cache_line(record);
    if (retire && (record->committed.sequence == 0 || record->run_epoch != layout->run_epoch ||
                   record->task_ref != task || record->worker_index != worker || record->order_key != order.rank)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, retire ? &record->retired : &record->committed)) return;
    record->run_epoch = layout->run_epoch;
    record->task_ref = task;
    record->order_key = order.rank;
    record->worker_index = worker;
    if (!retire) {
        const uint32_t required = order.required_floor[local->config.self_lane];
        const auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, local->config.task_metadata_offset);
        const bool mix = scheduler_task_is_mix(metadata[task].flags);
        record->observed_mix_floor = required != 0 && !mix ? local->observed_mix_floor : 0;
        record->observed_self_floor = required != 0 && mix ? local->observed_self_floor : 0;
    }
    scheduler_publish_cache_line(record);
}

// A cluster has at most one publication for each excluded MIX; slot zero
// of the per-cluster columns records the initial lower bound.
template <typename Local>
static inline __aicore__ void scheduler_protocol_mix_floor(
    __gm__ void *base, const Local *local, uint32_t cluster, uint32_t prior_mix_ref, uint32_t floor, bool before
) {
    auto *layout = scheduler_protocol_layout_at(base, local);
    if (layout == nullptr) return;
    if (local->config.self_lane != 1 || cluster >= local->config.cluster_count ||
        local->config.task_catalog_offset == 0 || local->config.shared_context == nullptr ||
        local->config.shared_context->cluster_index != cluster || floor > layout->task_count ||
        (prior_mix_ref != UINT32_MAX && prior_mix_ref >= layout->task_count)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    uint64_t row = 0;
    uint32_t next = catalog.first_mix_ref;
    if (prior_mix_ref != UINT32_MAX) {
        const auto &prior = scheduler_task_catalog_order(base, catalog, prior_mix_ref);
        if (prior.cooperative_index >= layout->cooperative_task_count || prior.rank >= floor) {
            scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
            return;
        }
        // Each coordinator excludes MIX tasks in catalog rank order. Pure sync
        // tasks have member records but occupy no rows in this table.
        row = static_cast<uint64_t>(local->profiling->published_mix_count) + 1;
        next = prior.next_mix_ref;
    }
    const uint64_t index = row * local->config.cluster_count + cluster;
    if (index >= layout->floor_count || (next != UINT32_MAX && next >= layout->task_count) ||
        floor != (next == UINT32_MAX ? layout->task_count : scheduler_task_catalog_order(base, catalog, next).rank)) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    auto *record = scheduler_state_at<SchedulerProtocolFloorTrace>(base, layout->floors_offset) + index;
    scheduler_observe_cache_line(record);
    if (!before && (record->before.sequence == 0 || record->run_epoch != layout->run_epoch || record->floor != floor ||
                    record->prior_mix_ref != prior_mix_ref || record->cluster_index != cluster ||
                    record->worker_index != local->worker_id())) {
        scheduler_protocol_fail(base, local, layout, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
        return;
    }
    if (!scheduler_protocol_stamp(base, local, layout, before ? &record->before : &record->after)) return;
    record->run_epoch = layout->run_epoch;
    record->cluster_index = cluster;
    record->worker_index = static_cast<uint32_t>(local->worker_id());
    record->prior_mix_ref = prior_mix_ref;
    record->floor = floor;
    scheduler_publish_cache_line(record);
    if (!before && prior_mix_ref != UINT32_MAX) ++local->profiling->published_mix_count;
}

template <typename Local>
static inline __aicore__ void scheduler_protocol_finish(__gm__ void *base, const Local *local, bool success) {
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
