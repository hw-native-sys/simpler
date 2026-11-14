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

// This mandatory run-entry ABI also covers TaskControl and its canonical completion gate.
inline constexpr uint32_t SCHEDULER_TASK_CATALOG_VERSION = 16;

// Immutable engine catalogs name physical tasks; control nodes are scanned separately.
struct alignas(64) SchedulerTaskCatalogHeader {
    uint64_t task_count{0};
    uint64_t descriptors_offset{0};
    uint64_t control_count{0};
    uint64_t storage_end_offset{0};
    uint64_t entry_count{0};
    uint32_t version{SCHEDULER_TASK_CATALOG_VERSION};
};

struct SchedulerTaskCatalogDescriptor {
    uint64_t entries_offset{0};
    uint64_t capacity{0};
};

static_assert(sizeof(SchedulerTaskCatalogHeader) == 64);
static_assert(sizeof(SchedulerTaskCatalogDescriptor) == 16);
static_assert(
    std::is_standard_layout_v<SchedulerTaskCatalogHeader> && std::is_trivially_copyable_v<SchedulerTaskCatalogHeader>
);
static_assert(
    std::is_standard_layout_v<SchedulerTaskCatalogDescriptor> &&
    std::is_trivially_copyable_v<SchedulerTaskCatalogDescriptor>
);

struct SchedulerTaskCatalog {
    __gm__ const int64_t *refs{nullptr};
    uint64_t count{0};
    uint64_t task_count{0};
};

inline __aicore__ SchedulerTaskCatalog
scheduler_task_catalog(__gm__ void *base, const __gm__ SchedulerTaskCatalogHeader &pool, uint32_t engine) {
    if (engine >= SCHEDULER_EXECUTION_ENGINE_COUNT) return {};
    const auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(base, pool.descriptors_offset);
    const auto &descriptor = descriptors[engine];
    return {scheduler_state_at<int64_t>(base, descriptor.entries_offset), descriptor.capacity, pool.task_count};
}

inline __aicore__ bool scheduler_task_catalog_valid(const SchedulerTaskCatalog &catalog) {
    return catalog.count <= catalog.task_count && catalog.task_count <= INT32_MAX &&
           (catalog.count == 0 || catalog.refs != nullptr);
}

// Called once after READY acquire. All subsequent descriptor reads are of
// immutable, already observed data; ownership uses returned atomic operations.
static inline __aicore__ bool scheduler_task_catalog_validate(
    __gm__ void *base, uint64_t state_bytes, uint64_t pool_offset, uint64_t task_count, uint32_t clusters
) {
    if (base == nullptr || reinterpret_cast<uintptr_t>(base) % SCHEDULER_STATE_BASE_ALIGNMENT != 0 ||
        pool_offset == 0 || pool_offset % SCHEDULER_STATE_ALIGNMENT != 0 || pool_offset > state_bytes ||
        sizeof(SchedulerTaskCatalogHeader) > state_bytes - pool_offset || clusters == 0 ||
        clusters > SCHEDULER_CLUSTER_CAPACITY || task_count > INT32_MAX)
        return false;
    auto *pool = scheduler_state_at<SchedulerTaskCatalogHeader>(base, pool_offset);
    for (uint64_t offset = 0; offset < sizeof(*pool); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(pool) + offset);
    if (pool->version != SCHEDULER_TASK_CATALOG_VERSION || pool->task_count != task_count ||
        pool->entry_count > task_count || pool->control_count > task_count - pool->entry_count ||
        pool->storage_end_offset > state_bytes || pool->descriptors_offset < pool_offset + sizeof(*pool) ||
        pool->descriptors_offset % SCHEDULER_STATE_ALIGNMENT != 0)
        return false;
    uint64_t cursor = pool->descriptors_offset;
    const uint64_t descriptor_bytes = SCHEDULER_EXECUTION_ENGINE_COUNT * sizeof(SchedulerTaskCatalogDescriptor);
    if (cursor > pool->storage_end_offset || descriptor_bytes > pool->storage_end_offset - cursor) return false;
    cursor += descriptor_bytes;
    auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(base, pool->descriptors_offset);
    for (uint64_t offset = 0; offset < descriptor_bytes; offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(descriptors) + offset);
    uint64_t entries = 0;
    for (uint32_t queue = 0; queue < SCHEDULER_EXECUTION_ENGINE_COUNT; ++queue) {
        const auto &descriptor = descriptors[queue];
        if (descriptor.capacity > task_count - entries) return false;
        entries += descriptor.capacity;
        if (descriptor.entries_offset < cursor || descriptor.entries_offset % SCHEDULER_STATE_ALIGNMENT != 0 ||
            descriptor.entries_offset > pool->storage_end_offset ||
            descriptor.capacity > (pool->storage_end_offset - descriptor.entries_offset) / sizeof(int64_t))
            return false;
        cursor = descriptor.entries_offset + descriptor.capacity * sizeof(int64_t);
    }
    return entries == pool->entry_count;
}
