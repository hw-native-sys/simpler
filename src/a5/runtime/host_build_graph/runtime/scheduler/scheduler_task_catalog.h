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
inline constexpr uint32_t SCHEDULER_TASK_CATALOG_VERSION = 19;

// Immutable engine catalogs name physical tasks; control nodes are scanned separately.
struct alignas(64) SchedulerTaskCatalogHeader {
    uint64_t task_count{0};
    uint64_t descriptors_offset{0};
    uint64_t control_count{0};
    uint64_t storage_end_offset{0};
    uint64_t entry_count{0};
    uint64_t issue_cursors_offset{0};
    uint32_t version{SCHEDULER_TASK_CATALOG_VERSION};
    uint32_t ordering_edge_count{0};
    uint64_t ordering_rows_offset{0};
};

struct alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) SchedulerIssueCursor {
    volatile uint64_t next_task{0};
};
static_assert(sizeof(SchedulerIssueCursor) == SCHEDULER_GM_ATOMIC_SECTOR_BYTES);

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
    __gm__ SchedulerIssueCursor *issue{nullptr};
};

struct SchedulerOrderingFanin {
    __gm__ const int64_t *refs{nullptr};
    uint32_t count{0};
};

inline __aicore__ uint64_t scheduler_ordering_refs_offset(const __gm__ SchedulerTaskCatalogHeader &pool) {
    const uint64_t rows_end = pool.ordering_rows_offset + (pool.task_count + 1) * sizeof(uint32_t);
    return (rows_end + alignof(int64_t) - 1) & ~(uint64_t{alignof(int64_t)} - 1);
}

inline __aicore__ bool scheduler_task_ordering_fanin(
    __gm__ void *base, const __gm__ SchedulerTaskCatalogHeader &pool, int64_t ref, SchedulerOrderingFanin *view
) {
    if (view == nullptr || ref < 0 || static_cast<uint64_t>(ref) >= pool.task_count) return false;
    *view = {};
    if (pool.ordering_edge_count == 0) return true;
    const auto *rows = scheduler_state_at<uint32_t>(base, pool.ordering_rows_offset);
    const uint32_t begin = rows[ref];
    const uint32_t end = rows[ref + 1];
    if (begin > end || end > pool.ordering_edge_count) return false;
    view->refs = scheduler_state_at<int64_t>(base, scheduler_ordering_refs_offset(pool)) + begin;
    view->count = end - begin;
    return true;
}

inline __aicore__ SchedulerTaskCatalog
scheduler_task_catalog(__gm__ void *base, const __gm__ SchedulerTaskCatalogHeader &pool, uint32_t engine) {
    if (engine >= SCHEDULER_EXECUTION_ENGINE_COUNT) return {};
    const auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(base, pool.descriptors_offset);
    const auto &descriptor = descriptors[engine];
    return {
        scheduler_state_at<int64_t>(base, descriptor.entries_offset), descriptor.capacity, pool.task_count,
        scheduler_state_at<SchedulerIssueCursor>(base, pool.issue_cursors_offset) + engine
    };
}

inline __aicore__ bool scheduler_task_catalog_valid(const SchedulerTaskCatalog &catalog) {
    return catalog.issue != nullptr && catalog.count <= catalog.task_count && catalog.task_count <= INT32_MAX &&
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
    const uint64_t issue_bytes = SCHEDULER_EXECUTION_ENGINE_COUNT * sizeof(SchedulerIssueCursor);
    if (pool->issue_cursors_offset < cursor || pool->issue_cursors_offset % SCHEDULER_GM_ATOMIC_SECTOR_BYTES != 0 ||
        pool->issue_cursors_offset > pool->storage_end_offset ||
        issue_bytes > pool->storage_end_offset - pool->issue_cursors_offset)
        return false;
    cursor = pool->issue_cursors_offset + issue_bytes;
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
    if (entries != pool->entry_count) return false;
    if (pool->ordering_edge_count == 0) return pool->ordering_rows_offset == 0;
    const uint64_t row_bytes = (task_count + 1) * sizeof(uint32_t);
    if (pool->ordering_rows_offset < cursor || pool->ordering_rows_offset % SCHEDULER_STATE_ALIGNMENT != 0 ||
        pool->ordering_rows_offset > pool->storage_end_offset ||
        row_bytes > pool->storage_end_offset - pool->ordering_rows_offset ||
        pool->ordering_rows_offset + row_bytes > UINT64_MAX - (alignof(int64_t) - 1))
        return false;
    const uint64_t refs_offset = scheduler_ordering_refs_offset(*pool);
    if (refs_offset > pool->storage_end_offset ||
        pool->ordering_edge_count > (pool->storage_end_offset - refs_offset) / sizeof(int64_t))
        return false;
    auto *rows = scheduler_state_at<uint32_t>(base, pool->ordering_rows_offset);
    scheduler_observe_cache_line(rows);
    if (task_count * sizeof(uint32_t) >= 64) scheduler_observe_cache_line(rows + task_count);
    return rows[0] == 0 && rows[task_count] == pool->ordering_edge_count;
}
