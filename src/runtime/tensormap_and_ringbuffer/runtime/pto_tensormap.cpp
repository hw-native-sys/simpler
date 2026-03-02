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
 * Based on: docs/runtime_buffer_manager_methods.md
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
#if PTO2_ORCH_PROFILING
static uint64_t g_lookup_chain_total = 0;
static uint64_t g_lookup_count = 0;
static int32_t  g_lookup_chain_max = 0;
static uint64_t g_lookup_overlap_checks = 0;
static uint64_t g_lookup_overlap_hits = 0;
static uint64_t g_insert_count = 0;
#endif

// =============================================================================
// Initialization and Destruction
// =============================================================================

bool PTO2TensorMap::init(int32_t new_num_buckets, int32_t new_pool_size) {
    // Validate power of 2 for fast modulo
    if ((new_num_buckets & (new_num_buckets - 1)) != 0) {
        return false;  // num_buckets must be power of 2
    }

    // Allocate buckets
    buckets = (PTO2TensorMapEntry**)malloc(new_num_buckets * sizeof(PTO2TensorMapEntry*));
    if (!buckets) {
        return false;
    }

    // Initialize all buckets to empty (-1)
    for (int32_t i = 0; i < new_num_buckets; i++) {
        buckets[i] = nullptr;
    }

    num_buckets = new_num_buckets;

    // Allocate entry pool
    entry_pool = (PTO2TensorMapEntry*)calloc(new_pool_size, sizeof(PTO2TensorMapEntry));
    if (!entry_pool) {
        free(buckets);
        buckets = NULL;
        return false;
    }

    // Allocate free entry list
    free_entry_list = (PTO2TensorMapEntry**)calloc(new_pool_size, sizeof(PTO2TensorMapEntry*));
    if (!free_entry_list) {
        free(buckets);
        free(entry_pool);
        buckets = NULL;
        entry_pool = NULL;
        return false;
    }

    pool_size = new_pool_size;
    next_entry_idx = 0;
    free_num = 0;

    // Initialize all entries as not in bucket
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = -1;
    }

    // Allocate per-task entry tracking
    task_entry_head = (PTO2TensorMapEntry**)malloc(new_pool_size * sizeof(PTO2TensorMapEntry*));
    if (!task_entry_head) {
        free(entry_pool);
        free(buckets);
        free(free_entry_list);
        entry_pool = NULL;
        buckets = NULL;
        free_entry_list = NULL;
        return false;
    }

    // Initialize all task entry heads to -1 (no entries)
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        task_entry_head[i] = nullptr;
    }

    last_task_alive = 0;

    return true;
}

bool PTO2TensorMap::init_default() {
    return init(PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE);
}

void PTO2TensorMap::destroy() {
    if (buckets) {
        free(buckets);
        buckets = NULL;
    }

    if (entry_pool) {
        free(entry_pool);
        entry_pool = NULL;
    }

    if (task_entry_head) {
        free(task_entry_head);
        task_entry_head = NULL;
    }

    if (free_entry_list) {
        free(free_entry_list);
        free_entry_list = NULL;
    }
}

void PTO2TensorMap::reset() {
    // Reset all buckets to empty
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }

    // Reset all entries
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = -1;
    }

    // Reset per-task entry tracking
    for (int32_t i = 0; i < pool_size; i++) {
        task_entry_head[i] = nullptr;
    }

    next_entry_idx = 0;
    free_num = 0;
    last_task_alive = 0;
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

    LOG_INFO("=== TensorMap Statistics ===");
    LOG_INFO("Pool size:           %d", pool_size);
    LOG_INFO("Pool next entry idx: %d", next_entry_idx);
    LOG_INFO("Pool free_num:       %d", free_num);
    LOG_INFO("Num buckets:         %d", num_buckets);
    LOG_INFO("Valid entries:       %d", valid);
    LOG_INFO("Stale entries:       %d", stale);
    LOG_INFO("Empty buckets:       %d", empty_buckets);
    LOG_INFO("Max chain len:       %d", max_chain);
    LOG_INFO("Avg chain len:       %.2f", non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    LOG_INFO("Last task alive:     %d", last_task_alive);
    LOG_INFO("============================");
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

void PTO2TensorMap::sync_tensormap() {
    constexpr int MIN_FREE_NUM = 1024;
    always_assert(orch != nullptr);
    while(true) {
        // Read current last_task_alive from shared memory
        int32_t new_last_task_alive = PTO2_LOAD_ACQUIRE(&orch->sm_handle->header->last_task_alive);
        sync_validity(new_last_task_alive);
        if ((pool_size - next_entry_idx + free_num < MIN_FREE_NUM) || new_last_task_alive - orch->tensormap_last_cleanup >= PTO2_TENSORMAP_CLEANUP_INTERVAL) {
            cleanup_retired(orch->tensormap_last_cleanup, new_last_task_alive);
            orch->tensormap_last_cleanup = new_last_task_alive;
        } else {
            break;
        }
    }
}

// =============================================================================
// TensorMap Lookup Profiling
// =============================================================================
#if PTO2_ORCH_PROFILING
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
