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
 * host_build_graph TensorMap implementation
 *
 * Implements TensorMap with a fixed-capacity entry pool. Task completion does
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

#include "host_build_graph/host/tensormap.h"

#include <stdlib.h>

#include <new>

#include "assert_compat.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime_types.h"

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

/**
 * Allocate the four arrays and reset the map to empty. Clears the bucket heads
 * and the per-task entry heads and resets the pool cursors (next_entry_idx /
 * free_num); the entry pool is left uninitialized and initialized on write by
 * new_entry(), so this is O(num_buckets + max_tasks), not O(pool_size).
 *
 * `new (std::nothrow) T[n]` rather than a vector: a vector would value-initialize
 * the whole entry pool, which is megabytes of zeroing the init-on-write path
 * exists to avoid, and nothrow keeps the failure reportable to a caller that
 * still has an alternative path.
 */
bool ChipTensorMap::init(int32_t new_num_buckets, int32_t new_pool_size, int32_t new_max_tasks) {
    // num_buckets must be a power of two for the hash truncation to work. The
    // task-chain count needs no such property: a task id indexes its chain directly.
    always_assert(new_num_buckets > 0 && (new_num_buckets & (new_num_buckets - 1)) == 0);
    always_assert(new_max_tasks > 0);
    always_assert(new_pool_size > 0);

    buckets.reset(new (std::nothrow) ChipTensorMapEntry *[new_num_buckets]);
    entry_pool.reset(new (std::nothrow) ChipTensorMapEntry[new_pool_size]);
    free_entry_list.reset(new (std::nothrow) ChipTensorMapEntry *[new_pool_size]);
    task_entry_heads.reset(new (std::nothrow) ChipTensorMapEntry *[new_max_tasks]);
    if (buckets == nullptr || entry_pool == nullptr || free_entry_list == nullptr || task_entry_heads == nullptr) {
        LOG_ERROR(
            "TensorMap init failed (buckets=%d, pool=%d, max_tasks=%d)", new_num_buckets, new_pool_size, new_max_tasks
        );
        return false;
    }

    num_buckets = new_num_buckets;
    pool_size = new_pool_size;
    max_tasks = new_max_tasks;

    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }
    for (int32_t i = 0; i < max_tasks; i++) {
        task_entry_heads[i] = nullptr;
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
    return true;
}

void ChipTensorMap::reset() {
    always_assert(buckets != nullptr && entry_pool != nullptr && task_entry_heads != nullptr);

    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }
    for (int32_t i = 0; i < max_tasks; i++) {
        task_entry_heads[i] = nullptr;
    }
    // Entries are reached only through a bucket chain, and every chain is now empty, so
    // the pool needs nothing: the next new_entry() bump-allocates from index 0 and puts
    // the slot into the unlinked state, exactly as after init().
    next_entry_idx = 0;
    free_num = 0;
}

bool ChipTensorMap::init_default(int32_t new_max_tasks) {
    return init(CHIP_TENSORMAP_NUM_BUCKETS, CHIP_TENSORMAP_POOL_SIZE, new_max_tasks);
}

// =============================================================================
// Debug Utilities
// =============================================================================

void ChipTensorMap::print_stats() {
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

int32_t ChipTensorMap::valid_count() {
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
ChipTensorMapProfilingData chip_tensormap_get_profiling() {
    ChipTensorMapProfilingData d;
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
