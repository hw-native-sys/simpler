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

#include "scheduler_ticket_queue.h"
#include "scheduler_dependency.h"

inline constexpr uint32_t SCHEDULER_READY_POOL_VERSION = 8;
inline constexpr uint32_t SCHEDULER_READY_POOL_REF_CHUNK = 4;

inline constexpr uint32_t SCHEDULER_READY_POOL_GROUP_WIDTH = 8;

// Immutable offsets into one run's scheduler allocation. No pointer fix-up
// or device-side allocation is needed; entries persist through the run fence.
struct alignas(64) SchedulerReadyPoolHeader {
    // Dispatch, publication and completion share this immutable first line.
    uint64_t task_count{0};
    uint64_t descriptors_offset{0};
    uint64_t controls_offset{0};
    uint64_t dependencies_offset{0};
    uint32_t group_counts[SCHEDULER_READY_QUEUE_COUNT]{};
    uint32_t queue_begins[SCHEDULER_READY_QUEUE_COUNT]{};
    uint32_t queue_count{0};

    uint32_t version{SCHEDULER_READY_POOL_VERSION};
    uint32_t group_width{SCHEDULER_READY_POOL_GROUP_WIDTH};
    uint32_t ref_chunk{SCHEDULER_READY_POOL_REF_CHUNK};
    uint64_t storage_end_offset{0};
    uint64_t entry_count{0};
};

struct SchedulerReadyPoolDescriptor {
    uint64_t entries_offset{0};
    uint64_t capacity{0};
};

static_assert(sizeof(SchedulerReadyPoolHeader) == 128);
static_assert(offsetof(SchedulerReadyPoolHeader, queue_count) < 64);
static_assert(sizeof(SchedulerReadyPoolDescriptor) == 16);
static_assert(
    std::is_standard_layout_v<SchedulerReadyPoolHeader> && std::is_trivially_copyable_v<SchedulerReadyPoolHeader>
);
static_assert(
    std::is_standard_layout_v<SchedulerReadyPoolDescriptor> &&
    std::is_trivially_copyable_v<SchedulerReadyPoolDescriptor>
);

inline __aicore__ uint32_t scheduler_ready_pool_groups(uint32_t clusters, uint32_t type) {
    const uint32_t workers = type == 1 ? clusters * (PLATFORM_CORES_PER_BLOCKDIM - 1) : clusters;
    return (workers + SCHEDULER_READY_POOL_GROUP_WIDTH - 1) / SCHEDULER_READY_POOL_GROUP_WIDTH;
}

// Chunking keeps short contiguous fanouts batchable. The integer avalanche
// avoids routing a strided task sequence solely by its low address bits.
inline __aicore__ uint32_t scheduler_ready_pool_ref_group(uint64_t ref, uint32_t groups) {
    if (groups <= 1) return 0;
    uint32_t mixed = static_cast<uint32_t>(ref / SCHEDULER_READY_POOL_REF_CHUNK);
    mixed ^= mixed >> 16;
    mixed *= UINT32_C(0x7feb352d);
    mixed ^= mixed >> 15;
    mixed *= UINT32_C(0x846ca68b);
    mixed ^= mixed >> 16;
    return mixed % groups;
}

inline __aicore__ SchedulerTicketQueueView
scheduler_ready_pool_queue(__gm__ void *base, const __gm__ SchedulerReadyPoolHeader &pool, uint32_t queue_index) {
    if (queue_index >= pool.queue_count) return {};
    const auto *descriptors = scheduler_state_at<SchedulerReadyPoolDescriptor>(base, pool.descriptors_offset);
    const auto &descriptor = descriptors[queue_index];
    return {
        scheduler_state_at<SchedulerTicketQueue>(base, pool.controls_offset) + queue_index,
        scheduler_state_at<int64_t>(base, descriptor.entries_offset), descriptor.capacity, pool.task_count
    };
}

// Called once after READY acquire. All subsequent descriptor reads are of
// immutable, already observed data; mutable queue words always bypass cache.
inline __aicore__ bool scheduler_ready_pool_validate(
    __gm__ void *base, uint64_t state_bytes, uint64_t pool_offset, uint64_t task_count, uint32_t clusters
) {
    if (base == nullptr || pool_offset == 0 || pool_offset % SCHEDULER_STATE_ALIGNMENT != 0 ||
        pool_offset > state_bytes || sizeof(SchedulerReadyPoolHeader) > state_bytes - pool_offset || clusters == 0 ||
        clusters > SCHEDULER_CLUSTER_CAPACITY || task_count > INT32_MAX)
        return false;
    auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(base, pool_offset);
    for (uint64_t offset = 0; offset < sizeof(*pool); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(pool) + offset);
    if (pool->version != SCHEDULER_READY_POOL_VERSION || pool->group_width != SCHEDULER_READY_POOL_GROUP_WIDTH ||
        pool->task_count != task_count || pool->entry_count > task_count ||
        pool->ref_chunk != SCHEDULER_READY_POOL_REF_CHUNK || pool->storage_end_offset > state_bytes)
        return false;
    uint32_t queue_count = 0;
    for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
        const uint32_t groups = scheduler_ready_pool_groups(clusters, type);
        if (pool->queue_begins[type] != queue_count || pool->group_counts[type] != groups) return false;
        queue_count += groups;
    }
    if (pool->queue_count != queue_count || pool->descriptors_offset < pool_offset + sizeof(*pool) ||
        pool->descriptors_offset % SCHEDULER_STATE_ALIGNMENT != 0 ||
        pool->controls_offset % SCHEDULER_STATE_ALIGNMENT != 0)
        return false;
    uint64_t cursor = pool->descriptors_offset;
    const uint64_t descriptor_bytes = queue_count * sizeof(SchedulerReadyPoolDescriptor);
    if (cursor > pool->storage_end_offset || descriptor_bytes > pool->storage_end_offset - cursor) return false;
    cursor += descriptor_bytes;
    if (pool->controls_offset < cursor || pool->controls_offset > pool->storage_end_offset ||
        queue_count * sizeof(SchedulerTicketQueue) > pool->storage_end_offset - pool->controls_offset)
        return false;
    cursor = pool->controls_offset + queue_count * sizeof(SchedulerTicketQueue);
    auto *descriptors = scheduler_state_at<SchedulerReadyPoolDescriptor>(base, pool->descriptors_offset);
    for (uint64_t offset = 0; offset < descriptor_bytes; offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptors) + offset);
    uint64_t entries = 0;
    for (uint32_t queue = 0; queue < queue_count; ++queue) {
        const auto &descriptor = descriptors[queue];
        if (descriptor.capacity > task_count - entries) return false;
        entries += descriptor.capacity;
        if (descriptor.entries_offset < cursor || descriptor.entries_offset % SCHEDULER_STATE_ALIGNMENT != 0 ||
            descriptor.entries_offset > pool->storage_end_offset ||
            descriptor.capacity > (pool->storage_end_offset - descriptor.entries_offset) / sizeof(int64_t))
            return false;
        cursor = descriptor.entries_offset + descriptor.capacity * sizeof(int64_t);
    }
    if (pool->dependencies_offset != 0 &&
        (pool->dependencies_offset < cursor ||
         !scheduler_dependency_validate(base, pool->storage_end_offset, pool->dependencies_offset, task_count)))
        return false;
    return entries == pool->entry_count;
}

inline __aicore__ const __gm__ SchedulerDependencyHeader *
scheduler_dependencies(__gm__ void *base, uint64_t pool_offset) {
    if (pool_offset == 0) return nullptr;
    const auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(base, pool_offset);
    return pool->dependencies_offset == 0 ?
               nullptr :
               scheduler_state_at<SchedulerDependencyHeader>(base, pool->dependencies_offset);
}
