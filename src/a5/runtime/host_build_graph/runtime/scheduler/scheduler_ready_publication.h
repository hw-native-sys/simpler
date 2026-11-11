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

inline __aicore__ SchedulerReadyAppendResult scheduler_ready_publication_append(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, int64_t ref,
    SchedulerReadyPublication &publication
) {
    if (local == nullptr || !scheduler_task_ref_valid(graph, ref) ||
        publication.phase != SchedulerReadyPublishPhase::COLLECTING ||
        publication.count > SCHEDULER_READY_PUBLICATION_CAPACITY)
        return SchedulerReadyAppendResult::ERROR;
    if (publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY) return SchedulerReadyAppendResult::FULL;
    bool control = false;
    if (graph.native_header_offset != 0) {
        SchedulerNodeKind kind{};
        if (scheduler_node_kind(graph, ref, &kind) != SchedulerGraphResult::OK)
            return SchedulerReadyAppendResult::ERROR;
        control = scheduler_node_is_control(kind);
    }
    uint32_t type = SCHEDULER_READY_QUEUE_COUNT;
    if (!control) {
        const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
        const LaneMask lanes = scheduler_task_lanes(metadata);
        type = scheduler_task_ready_queue(metadata.flags, lanes);
        if (!scheduler_task_is_executable(metadata.flags) || type >= SCHEDULER_READY_QUEUE_COUNT ||
            (scheduler_metadata_single_subtask_slot(lanes) == UINT8_MAX && !scheduler_task_is_mix(metadata.flags)))
            return SchedulerReadyAppendResult::ERROR;
    }
    const auto destination = control ? SchedulerReadyDestination::CONTROL : scheduler_ready_destination(type);
    if (publication.count != 0 && publication.destination != destination) return SchedulerReadyAppendResult::FULL;
    uint32_t target = 0;
    uint64_t inbox_offset = 0;
    if (control) {
        if (local->config.control_inboxes_offset == 0 || local->worker_id() >= local->config.runtime_worker_count)
            return SchedulerReadyAppendResult::ERROR;
        target = local->worker_id();
        inbox_offset =
            local->config.control_inboxes_offset + static_cast<uint64_t>(target) * sizeof(SchedulerReadyInbox);
    } else if (local->config.ready_pool_offset == 0) {
        return SchedulerReadyAppendResult::ERROR;
    }
    for (uint32_t index = 0; index < publication.count; ++index)
        if (publication.refs[index] == ref) return SchedulerReadyAppendResult::ERROR;
    if (publication.count == 0) {
        publication.destination = destination;
        publication.target = target;
        publication.inbox_offset = inbox_offset;
    }
    publication.refs[publication.count++] = ref;
    return SchedulerReadyAppendResult::ADDED;
}

// Executable tickets publish a bounded batch in one call. Control inboxes
// persist preparation before atomically appending their immutable chain.
static inline __aicore__ SchedulerReadyPublishResult scheduler_ready_publication_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerReadyPublication &publication
) {
    auto fail = [&]() __aicore__ {
        publication.phase = SchedulerReadyPublishPhase::FAILED;
        return SchedulerReadyPublishResult::ERROR;
    };
    if (publication.phase == SchedulerReadyPublishPhase::DONE) return SchedulerReadyPublishResult::DONE;
    if (publication.phase == SchedulerReadyPublishPhase::FAILED) return SchedulerReadyPublishResult::ERROR;
    const bool control = publication.destination == SchedulerReadyDestination::CONTROL;
    const auto type = static_cast<uint32_t>(publication.destination);
    if (local == nullptr || publication.count == 0 || publication.count > SCHEDULER_READY_PUBLICATION_CAPACITY ||
        (!control && type >= SCHEDULER_READY_QUEUE_COUNT))
        return fail();
    const __gm__ SchedulerReadyPoolHeader *pool = nullptr;
    if (control) {
        const uint64_t expected_offset = local->config.control_inboxes_offset +
                                         static_cast<uint64_t>(publication.target) * sizeof(SchedulerReadyInbox);
        if (local->config.control_inboxes_offset == 0 || publication.target >= local->config.runtime_worker_count ||
            publication.inbox_offset != expected_offset)
            return fail();
    } else {
        if (local->config.ready_pool_offset == 0 || publication.target != 0 || publication.inbox_offset != 0)
            return fail();
        pool = scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
    }

    if (publication.phase == SchedulerReadyPublishPhase::COLLECTING) {
        for (uint32_t index = 0; index < publication.count; ++index)
            if (!scheduler_task_ref_valid(graph, publication.refs[index])) return fail();
        if (!control) {
            const auto queue = scheduler_ready_pool_queue(base, *pool, type);
            SchedulerTicketRange range{};
            if (scheduler_ticket_reserve(queue, publication.count, &range) != SchedulerTicketResult::SUCCESS ||
                scheduler_ticket_publish(queue, range, publication.refs) != SchedulerTicketResult::SUCCESS)
                return fail();
            publication.phase = SchedulerReadyPublishPhase::DONE;

            return SchedulerReadyPublishResult::DONE;
        } else {
            for (uint32_t index = 0; index < publication.count; ++index) {
                auto *task = scheduler_task_control_at(base, local, publication.refs[index]);
                scheduler_gm_store(
                    task->next_waiter,
                    index + 1 == publication.count ? SCHEDULER_INBOX_EMPTY : publication.refs[index + 1]
                );
            }
            scheduler_cache_barrier();
        }
        publication.phase = SchedulerReadyPublishPhase::PREPARED;
        return SchedulerReadyPublishResult::YIELD;
    }
    if (publication.phase == SchedulerReadyPublishPhase::PREPARED) {
        if (!control) return fail();
        auto *inbox = scheduler_state_at<SchedulerReadyInbox>(base, publication.inbox_offset);
        const uint64_t packed = scheduler_gm_fetch_add(inbox->endpoints, UINT64_C(0));
        if (!scheduler_ready_endpoints_consistent(packed)) return fail();
        const bool empty = scheduler_ready_endpoints_empty(packed);
        const int64_t head = scheduler_ready_endpoints_head(packed);
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        if (!empty && (!scheduler_task_ref_valid(graph, head) || !scheduler_task_ref_valid(graph, tail))) return fail();
        const uint64_t next =
            scheduler_ready_endpoints_pack(empty ? publication.refs[0] : head, publication.refs[publication.count - 1]);
        if (scheduler_scalar_atomic_result(local, [&]() __aicore__ {
                return scheduler_gm_compare_exchange(inbox->endpoints, packed, next);
            }) != packed) {
            return SchedulerReadyPublishResult::YIELD;
        }
        if (!empty) scheduler_gm_store(scheduler_task_control_at(base, local, tail)->next_waiter, publication.refs[0]);
        scheduler_cache_barrier();
        publication.phase = SchedulerReadyPublishPhase::DONE;

        return SchedulerReadyPublishResult::DONE;
    }
    return fail();
}
