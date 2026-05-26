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
/**
 * PTO Runtime2 - TensorMap Implementation
 *
 * Implements TensorMap with ring buffer pool, lazy invalidation,
 * and chain truncation optimization.
 *
 * Key features:
 * 1. O(1) insert at bucket head
 * 2. O(valid_entries) lookup with chain truncation
 * 3. Automatic stale entry cleanup during lookup
 * 4. Periodic explicit cleanup for long chains
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_tensormap.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "common/unified_log.h"
#include "pto_orchestrator.h"

// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
#if PTO2_TENSORMAP_PROFILING
uint64_t g_lookup_chain_total = 0;
uint64_t g_lookup_count = 0;
int32_t g_lookup_chain_max = 0;
uint64_t g_lookup_overlap_checks = 0;
uint64_t g_lookup_overlap_hits = 0;
uint64_t g_insert_count = 0;
#endif

// =============================================================================
// Initialization and Destruction
// =============================================================================

PTO2TensorMapLayout PTO2TensorMap::reserve_layout(
    DeviceArena &arena, int32_t new_num_buckets, int32_t new_pool_size,
    const int32_t new_task_window_sizes[PTO2_MAX_RING_DEPTH]
) {
    // num_buckets must be a power of two for the hash truncation to work.
    always_assert((new_num_buckets & (new_num_buckets - 1)) == 0);

    PTO2TensorMapLayout layout{};
    layout.num_buckets = new_num_buckets;
    layout.pool_size = new_pool_size;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        layout.task_window_sizes[r] = new_task_window_sizes[r];
    }

    layout.off_buckets = arena.reserve(
        static_cast<size_t>(new_num_buckets) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
    );
    layout.off_entry_pool =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry), alignof(PTO2TensorMapEntry));
    layout.off_free_entry_list =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *));
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        layout.off_task_entry_heads[r] = arena.reserve(
            static_cast<size_t>(new_task_window_sizes[r]) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
        );
    }
    return layout;
}

PTO2TensorMapLayout
PTO2TensorMap::reserve_layout_default(DeviceArena &arena, const int32_t new_task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    return reserve_layout(arena, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE, new_task_window_sizes);
}

bool PTO2TensorMap::init_from_layout(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    num_buckets = layout.num_buckets;
    pool_size = layout.pool_size;

    buckets = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    entry_pool = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    free_entry_list = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));

    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }

    memset(entry_pool, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry));
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = PTO2TaskId{};
    }

    memset(free_entry_list, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry *));

    next_entry_idx = 0;
    free_num = 0;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_entry_heads[r] = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_task_entry_heads[r]));
        for (int32_t i = 0; i < layout.task_window_sizes[r]; i++) {
            task_entry_heads[r][i] = nullptr;
        }
        task_window_sizes[r] = layout.task_window_sizes[r];
        last_task_alives[r] = 0;
        last_cleanup[r] = 0;
    }

    return true;
}

void PTO2TensorMap::destroy() {
    // Arena owns the backing memory; here we only forget our pointers so any
    // stray post-destroy access trips a nullptr dereference instead of reading
    // a recycled allocation.
    buckets = nullptr;
    entry_pool = nullptr;
    free_entry_list = nullptr;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_entry_heads[r] = nullptr;
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void PTO2TensorMap::print_stats() {
    int32_t valid = 0;
    int32_t stale = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;

    // Count entries
    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1) {
            if (entry_valid(entry_pool[i])) {
                valid++;
            } else {
                stale++;
            }
        }
    }

    // Count bucket stats
    for (int32_t b = 0; b < num_buckets; b++) {
        int32_t chain_len = 0;
        auto cur_entry = buckets[b];

        while (cur_entry != nullptr) {
            chain_len++;
            cur_entry = cur_entry->next_in_bucket;
        }

        if (chain_len == 0) {
            empty_buckets++;
        } else {
            non_empty_buckets++;
            total_chain += chain_len;
            if (chain_len > max_chain) {
                max_chain = chain_len;
            }
        }
    }

    LOG_INFO_V0("=== TensorMap Statistics ===");
    LOG_INFO_V0("Pool size:           %d", pool_size);
    LOG_INFO_V0("Pool next entry idx: %d", next_entry_idx);
    LOG_INFO_V0("Pool free_num:       %d", free_num);
    LOG_INFO_V0("Num buckets:         %d", num_buckets);
    LOG_INFO_V0("Valid entries:       %d", valid);
    LOG_INFO_V0("Stale entries:       %d", stale);
    LOG_INFO_V0("Empty buckets:       %d", empty_buckets);
    LOG_INFO_V0("Max chain len:       %d", max_chain);
    LOG_INFO_V0("Avg chain len:       %.2f", non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        LOG_INFO_V0("Last task alive[%d]: %d", r, last_task_alives[r]);
    }
    LOG_INFO_V0("============================");
}

int32_t PTO2TensorMap::valid_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1 && entry_valid(entry_pool[i])) {
            count++;
        }
    }

    return count;
}

void PTO2TensorMap::sync_tensormap(PTO2TaskId task_id, int32_t sm_last_task_alive) {
    auto ring_id = task_id.ring();
    auto local_id = task_id.local();
    sync_validity(ring_id, sm_last_task_alive);

    // Only attempt cleanup when last_task_alive has actually advanced;
    // otherwise cleanup_retired would empty-loop and we'd spin forever.
    auto overlap = get_task_local_id_slot(ring_id, local_id) == get_task_local_id_slot(ring_id, last_cleanup[ring_id]);
    if (sm_last_task_alive - last_cleanup[ring_id] >= PTO2_TENSORMAP_CLEANUP_INTERVAL || overlap) {
        cleanup_retired(ring_id, last_cleanup[ring_id], sm_last_task_alive);
        last_cleanup[ring_id] = sm_last_task_alive;
    }
}

// =============================================================================
// TensorMap Lookup Profiling
// =============================================================================
#if PTO2_TENSORMAP_PROFILING
PTO2TensorMapProfilingData pto2_tensormap_get_profiling() {
    PTO2TensorMapProfilingData d;
    d.lookup_chain_total = g_lookup_chain_total;
    d.lookup_count = g_lookup_count;
    d.lookup_chain_max = g_lookup_chain_max;
    d.overlap_checks = g_lookup_overlap_checks;
    d.overlap_hits = g_lookup_overlap_hits;
    d.insert_count = g_insert_count;

    // Reset
    g_lookup_chain_total = 0;
    g_lookup_count = 0;
    g_lookup_chain_max = 0;
    g_lookup_overlap_checks = 0;
    g_lookup_overlap_hits = 0;
    g_insert_count = 0;
    return d;
}
#endif
