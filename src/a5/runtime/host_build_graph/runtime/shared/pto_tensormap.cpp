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
 * Host-build-graph TensorMap implementation.
 *
 * Implements TensorMap with a fixed-capacity arena pool. Task completion does
 * not invalidate entries; dependency computation explicitly removes only
 * producers made redundant by coverage.
 *
 * Key features:
 * 1. O(1) insert at bucket head
 * 2. O(live_entries) lookup
 * 3. O(1) unlink through bucket/task predecessor links
 * 4. Free-list reuse after explicit semantic removal
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_tensormap.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "common/unified_log.h"

// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
#if SIMPLER_TENSORMAP_PROFILING
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
    DeviceArena &arena, int32_t new_num_buckets, int32_t new_pool_size, int32_t new_task_capacity
) {
    // num_buckets must be a power of two for the hash truncation to work.
    always_assert((new_num_buckets & (new_num_buckets - 1)) == 0);

    PTO2TensorMapLayout layout{};
    layout.num_buckets = new_num_buckets;
    layout.pool_size = new_pool_size;
    layout.task_capacity = new_task_capacity;

    layout.off_buckets = arena.reserve(
        static_cast<size_t>(new_num_buckets) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
    );
    layout.off_entry_pool =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry), alignof(PTO2TensorMapEntry));
    layout.off_free_entry_list =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *));
    layout.off_task_entry_heads = arena.reserve(
        static_cast<size_t>(new_task_capacity) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
    );
    return layout;
}

PTO2TensorMapLayout PTO2TensorMap::reserve_layout_default(DeviceArena &arena, int32_t new_task_capacity) {
    return reserve_layout(arena, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE, new_task_capacity);
}

/**
 * Reset the map to empty for a fresh orchestration pass. Addresses the arena
 * regions directly through arena.region_ptr, so it runs before
 * wire_arena_pointers binds the struct's pointer fields (and before any
 * insert/lookup). Clears the bucket heads and the per-task entry heads and
 * resets the pool cursors (next_entry_idx / free_num); the entry pool itself is
 * left untouched and initialized on write by new_entry(), so this is
 * O(num_buckets + task_capacity), not O(pool_size).
 */
bool PTO2TensorMap::init_data_from_layout(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    num_buckets = layout.num_buckets;
    pool_size = layout.pool_size;

    // Address arena regions for data writes; do not store these in struct
    // fields (wire_arena_pointers does that).
    auto *buckets_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));

    // buckets[]: empty == nullptr.
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets_arena[i] = nullptr;
    }

    // Init-on-write: the entry pool is not pre-zeroed. new_entry() puts each
    // bump-allocated slot into the clean "unlinked" state (bucket_index == -1,
    // link pointers null) -- the same state free_entry() leaves a recycled slot
    // in -- so only current_used() live entries are ever touched. buckets[] start
    // empty and unallocated slots are never reached via a bucket chain, so
    // nothing reads an uninitialized slot; the two debug scans (print_stats /
    // valid_count) are bounded to next_entry_idx. free_entry_list is a stack
    // sized by free_num, meaningful only after frees, so it needs no init.

    next_entry_idx = 0;
    free_num = 0;

    auto *heads_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_task_entry_heads));
    for (int32_t i = 0; i < layout.task_capacity; i++) {
        heads_arena[i] = nullptr;
    }
    task_capacity = layout.task_capacity;

    return true;
}

void PTO2TensorMap::wire_arena_pointers(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    buckets = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    entry_pool = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    free_entry_list = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));
    task_entry_heads = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_task_entry_heads));
}

void PTO2TensorMap::destroy() {
    // Arena owns the backing memory; here we only forget our pointers so any
    // stray post-destroy access trips a nullptr dereference instead of reading
    // a recycled allocation.
    buckets = nullptr;
    entry_pool = nullptr;
    free_entry_list = nullptr;
    task_entry_heads = nullptr;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void PTO2TensorMap::print_stats() {
    int32_t valid = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;

    // Count entries
    // Init-on-write: only [0, next_entry_idx) slots have ever been allocated and
    // thus initialized; slots beyond that are untouched and must not be read.
    for (int32_t i = 0; i < next_entry_idx; i++) {
        if (entry_pool[i].bucket_index != -1) {
            valid++;
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

    LOG_DEBUG("=== TensorMap Statistics ===");
    LOG_DEBUG("Pool size:           %d", pool_size);
    LOG_DEBUG("Pool next entry idx: %d", next_entry_idx);
    LOG_DEBUG("Pool free_num:       %d", free_num);
    LOG_DEBUG("Num buckets:         %d", num_buckets);
    LOG_DEBUG("Valid entries:       %d", valid);
    LOG_DEBUG("Empty buckets:       %d", empty_buckets);
    LOG_DEBUG("Max chain len:       %d", max_chain);
    LOG_DEBUG("Avg chain len:       %.2f", non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    LOG_DEBUG("============================");
}

int32_t PTO2TensorMap::valid_count() {
    int32_t count = 0;

    // Init-on-write: only [0, next_entry_idx) slots have ever been allocated and
    // thus initialized; slots beyond that are untouched and must not be read.
    for (int32_t i = 0; i < next_entry_idx; i++) {
        if (entry_pool[i].bucket_index != -1) {
            count++;
        }
    }

    return count;
}

// =============================================================================
// TensorMap Lookup Profiling
// =============================================================================
#if SIMPLER_TENSORMAP_PROFILING
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
