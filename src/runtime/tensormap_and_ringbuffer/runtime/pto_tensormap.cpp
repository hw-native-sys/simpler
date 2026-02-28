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
#include "tensor.h"

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

bool pto2_tensormap_init(PTO2TensorMap* tm, int32_t num_buckets, int32_t pool_size) {
    // Validate power of 2 for fast modulo
    if ((num_buckets & (num_buckets - 1)) != 0) {
        return false;  // num_buckets must be power of 2
    }

    // Allocate buckets
    tm->buckets = (int32_t*)malloc(num_buckets * sizeof(int32_t));
    if (!tm->buckets) {
        return false;
    }

    // Initialize all buckets to empty (-1)
    for (int32_t i = 0; i < num_buckets; i++) {
        tm->buckets[i] = -1;
    }

    tm->num_buckets = num_buckets;

    // Allocate entry pool
    tm->entry_pool = (PTO2TensorMapEntry*)calloc(pool_size, sizeof(PTO2TensorMapEntry));
    if (!tm->entry_pool) {
        free(tm->buckets);
        tm->buckets = NULL;
        return false;
    }

    // Allocate free entry list
    tm->free_entry_list = (int32_t*)calloc(pool_size, sizeof(int32_t));
    if (!tm->free_entry_list) {
        free(tm->buckets);
        free(tm->entry_pool);
        tm->buckets = NULL;
        tm->entry_pool = NULL;
        return false;
    }

    tm->pool_size = pool_size;
    tm->next_entry_idx = 0;
    tm->free_num = 0;

    // Initialize all entries as not in bucket
    for (int32_t i = 0; i < pool_size; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].prev_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].prev_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
    }

    // Allocate per-task entry tracking
    tm->task_entry_head = (int32_t*)malloc(PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    if (!tm->task_entry_head) {
        free(tm->entry_pool);
        free(tm->buckets);
        free(tm->free_entry_list);
        tm->entry_pool = NULL;
        tm->buckets = NULL;
        tm->free_entry_list = NULL;
        return false;
    }

    // Initialize all task entry heads to -1 (no entries)
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }

    tm->last_task_alive = 0;

    return true;
}

bool pto2_tensormap_init_default(PTO2TensorMap* tm) {
    return pto2_tensormap_init(tm, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE);
}

void pto2_tensormap_destroy(PTO2TensorMap* tm) {
    if (tm->buckets) {
        free(tm->buckets);
        tm->buckets = NULL;
    }

    if (tm->entry_pool) {
        free(tm->entry_pool);
        tm->entry_pool = NULL;
    }

    if (tm->task_entry_head) {
        free(tm->task_entry_head);
        tm->task_entry_head = NULL;
    }
}

void pto2_tensormap_reset(PTO2TensorMap* tm) {
    // Reset all buckets to empty
    for (int32_t i = 0; i < tm->num_buckets; i++) {
        tm->buckets[i] = -1;
    }

    // Reset all entries
    for (int32_t i = 0; i < tm->pool_size; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].prev_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].prev_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
    }

    // Reset per-task entry tracking
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }

    tm->next_entry_idx = 0;
    tm->free_num = 0;
    tm->last_task_alive = 0;
}

// =============================================================================
// Hash Function
// =============================================================================

uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, const Tensor& tensor) {
    // ========================================================================
    // CRITICAL: Hash ONLY by base_ptr for correct overlap detection!
    // ========================================================================
    //
    // For overlap detection to work, ALL regions accessing the same base
    // tensor MUST be in the SAME hash bucket. This allows lookup to find
    // and check all potentially overlapping regions.
    //
    // If we included offset in the hash, overlapping regions with different
    // offsets would end up in different buckets and never be compared:
    //   Region A: base=X, offset=0   → bucket 5
    //   Region B: base=X, offset=128 → bucket 12  (WRONG! Can't detect overlap)
    //
    // With base_ptr-only hash:
    //   Region A: base=X, offset=0   → bucket 5
    //   Region B: base=X, offset=128 → bucket 5   (CORRECT! Same bucket)
    //
    uint64_t key = (uint64_t)(uintptr_t)tensor.data().buffer.addr;

    // Improve distribution by mixing bits (pointers often have aligned low bits)
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);

    // Use bitwise AND for power-of-2 modulo (faster than %)
    return (uint32_t)(key & (tm->num_buckets - 1));
}

// =============================================================================
// Validity and Cleanup
// =============================================================================

void pto2_tensormap_sync_validity(PTO2TensorMap* tm, int32_t last_task_alive) { tm->last_task_alive = last_task_alive; }

void pto2_tensormap_remove_entry(PTO2TensorMap& tm, int32_t entry_idx) {
    pto2_tensormap_remove_from_task(&tm, entry_idx);
    tm.free_entry(entry_idx);
}

void pto2_tensormap_remove_from_task(PTO2TensorMap* tm, int32_t entry_idx) {
    auto entry = &tm->entry_pool[entry_idx];
    // Update predecessor's next pointer (O(1) via prev_in_task)
    if (entry->prev_in_task == -1) {
        // Entry is the head of its task chain, update task_entry_head
        int32_t task_slot = entry->producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        tm->task_entry_head[task_slot] = entry->next_in_task;
    } else {
        tm->entry_pool[entry->prev_in_task].next_in_task = entry->next_in_task;
    }

    // Update successor's prev pointer
    if (entry->next_in_task >= 0) {
        tm->entry_pool[entry->next_in_task].prev_in_task = entry->prev_in_task;
    }

    entry->next_in_task = -1;
    entry->prev_in_task = -1;
}

void pto2_tensormap_cleanup_retired(PTO2TensorMap* tm, int32_t old_last_task_alive, int32_t new_last_task_alive) {
    // Iterate through retired tasks and remove their entries from bucket chains
    for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
        int32_t task_slot = task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        int32_t offset = tm->task_entry_head[task_slot];

        while (offset >= 0) {
            PTO2TensorMapEntry* entry = &tm->entry_pool[offset];
            int32_t next = entry->next_in_task;  // Save before clearing
            // Only remove if this entry belongs to the retiring task
            // (slot may have been reused by a newer task)
            if (entry->producer_task_id == task_id) {
                // Clear task chain pointers (entire chain is being destroyed)
                tm->free_entry(offset);
            }
            offset = next;
        }

        // Clear task's entry head (slot will be reused by task_id + TASK_WINDOW_SIZE)
        tm->task_entry_head[task_slot] = -1;
    }
}

// =============================================================================
// Lookup with Chain Truncation
// =============================================================================

void pto2_tensormap_lookup(PTO2TensorMap* tm, const Tensor &tensor, PTO2LookupResult* result) {
    uint32_t bucket = pto2_tensormap_hash(tm, tensor);
    int32_t* prev_ptr = &tm->buckets[bucket];  // For truncation
    int32_t offset = *prev_ptr;

    result->count = 0;
#if PTO2_ORCH_PROFILING
    int32_t chain_len = 0;
#endif

    while (offset >= 0) {
        PTO2TensorMapEntry* entry = &tm->entry_pool[offset];

        // Check validity first
        if (!pto2_tensormap_entry_valid(tm, entry)) {
            // ========== STALE ENTRY: Truncate chain here ==========
            // All subsequent entries are guaranteed to be stale too!
            // Truncate: unlink this and all following entries
            *prev_ptr = -1;  // Terminate chain at previous entry

            // Mark truncated entries as not in bucket (for correct reuse)
            while (offset >= 0) {
                PTO2TensorMapEntry* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                stale->prev_in_bucket = -1;
                offset = next;
            }

#if PTO2_ORCH_PROFILING
            g_lookup_chain_total += chain_len;
            g_lookup_count++;
            if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
            return;
        }

#if PTO2_ORCH_PROFILING
        chain_len++;
        g_lookup_overlap_checks++;
#endif
        // Entry is valid - check if regions OVERLAP (not just exact match)
        // Since we hash only by base_ptr, all entries in this bucket have
        // potential to overlap. We must check actual byte-range overlap.
        auto overlap_status = tensor.is_overlap(entry->tensor);
        if (overlap_status != OverlapStatus::NO_OVERLAP) {
            result->push(offset, overlap_status);
#if PTO2_ORCH_PROFILING
            g_lookup_overlap_hits++;
#endif
        }

        // Move to next entry
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }

#if PTO2_ORCH_PROFILING
    g_lookup_chain_total += chain_len;
    g_lookup_count++;
    if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
}

int32_t PTO2TensorMap::new_entry() {
    if (free_num > 0) {
        int32_t res = free_entry_list[--free_num];
        debug_assert(!entry_pool[res].in_bucket);
        return res;
    }
    if (next_entry_idx < pool_size) {
        int32_t res = next_entry_idx++;
        debug_assert(!entry_pool[res].in_bucket);
        return res;
    }

    size_t wait_count = 0;
    while (free_num == 0) {
        pto2_orchestrator_sync_tensormap(this, true);
        always_assert(wait_count++ <= 1000000000UL);
    }
    debug_assert(free_num > 0);
    int32_t res = free_entry_list[--free_num];
    debug_assert(!entry_pool[res].in_bucket);
    return res;
}

void PTO2TensorMap::free_entry(int32_t entry_idx) {
    auto entry = &entry_pool[entry_idx];
    if (!entry->in_bucket) {
        return;  // Already removed
    }

    // Update predecessor's next pointer (O(1) via prev_in_bucket)
    if (entry->prev_in_bucket == -1) {
        // Entry is the head of its bucket chain, update bucket head
        // Must compute hash BEFORE clearing tensor (tensor.data() needs valid tensor_pool)
        uint32_t bucket = pto2_tensormap_hash(this, entry->tensor);
        buckets[bucket] = entry->next_in_bucket;
    } else {
        entry_pool[entry->prev_in_bucket].next_in_bucket = entry->next_in_bucket;
    }

    // Update successor's prev pointer
    if (entry->next_in_bucket >= 0) {
        entry_pool[entry->next_in_bucket].prev_in_bucket = entry->prev_in_bucket;
    }

    // Clear tensor AFTER bucket chain manipulation (hash computation needs valid tensor)
    entry->tensor = Tensor();

    free_entry_list[free_num++] = entry_idx;
    entry->in_bucket = false;
    entry->next_in_bucket = -1;
    entry->prev_in_bucket = -1;
    entry->next_in_task = -1;
    entry->prev_in_task = -1;
}

// =============================================================================
// Insert
// =============================================================================

void pto2_tensormap_insert(PTO2TensorMap* tm, const Tensor& tensor, int32_t producer_task_id, bool with_alloc) {
#if PTO2_ORCH_PROFILING
    g_insert_count++;
#endif
    // Allocate entry from ring buffer pool
    int32_t entry_offset = tm->new_entry();
    PTO2TensorMapEntry* entry = &tm->entry_pool[entry_offset];

    // Initialize new entry
    entry->tensor = tensor;
    entry->producer_task_id = producer_task_id;
    entry->with_alloc = with_alloc;

    // Insert at head of hash bucket (maintains task_id descending order)
    uint32_t bucket = pto2_tensormap_hash(tm, tensor);
    entry->next_in_bucket = tm->buckets[bucket];
    entry->prev_in_bucket = -1;  // New head has no predecessor
    // Update old head's prev pointer
    if (entry->next_in_bucket >= 0) {
        tm->entry_pool[entry->next_in_bucket].prev_in_bucket = entry_offset;
    }
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;

    // Link to task's entry list (for cleanup)
    int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
    entry->next_in_task = tm->task_entry_head[task_slot];
    entry->prev_in_task = -1;  // New head has no predecessor
    // Update old head's prev pointer
    if (entry->next_in_task >= 0) {
        tm->entry_pool[entry->next_in_task].prev_in_task = entry_offset;
    }
    tm->task_entry_head[task_slot] = entry_offset;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_tensormap_print_stats(PTO2TensorMap* tm) {
    int32_t valid = 0;
    int32_t stale = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;

    // Count entries
    for (int32_t i = 0; i < tm->pool_size; i++) {
        if (tm->entry_pool[i].in_bucket) {
            if (pto2_tensormap_entry_valid(tm, &tm->entry_pool[i])) {
                valid++;
            } else {
                stale++;
            }
        }
    }

    // Count bucket stats
    for (int32_t b = 0; b < tm->num_buckets; b++) {
        int32_t chain_len = 0;
        int32_t offset = tm->buckets[b];

        while (offset >= 0) {
            chain_len++;
            offset = tm->entry_pool[offset].next_in_bucket;
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
    LOG_INFO("Pool size:           %d", tm->pool_size);
    LOG_INFO("Pool next entry idx: %d", tm->next_entry_idx);
    LOG_INFO("Pool free_num:       %d", tm->free_num);
    LOG_INFO("Num buckets:         %d", tm->num_buckets);
    LOG_INFO("Valid entries:       %d", valid);
    LOG_INFO("Stale entries:       %d", stale);
    LOG_INFO("Empty buckets:       %d", empty_buckets);
    LOG_INFO("Max chain len:       %d", max_chain);
    LOG_INFO("Avg chain len:       %.2f", non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    LOG_INFO("Last task alive:     %d", tm->last_task_alive);
    LOG_INFO("============================");
}

int32_t pto2_tensormap_valid_count(PTO2TensorMap* tm) {
    int32_t count = 0;

    for (int32_t i = 0; i < tm->pool_size; i++) {
        if (tm->entry_pool[i].in_bucket && pto2_tensormap_entry_valid(tm, &tm->entry_pool[i])) {
            count++;
        }
    }

    return count;
}

// =============================================================================
// TensorMap Synchronization
// =============================================================================

void pto2_orchestrator_sync_tensormap(PTO2TensorMap* tm, bool force) {
    always_assert(tm->orch != nullptr);
    // Read current last_task_alive from shared memory
    int32_t new_last_task_alive = PTO2_LOAD_ACQUIRE(&tm->orch->sm_handle->header->last_task_alive);

    // Update TensorMap validity threshold
    pto2_tensormap_sync_validity(tm, new_last_task_alive);

    // Periodically cleanup TensorMap to remove stale entries from bucket chains
    if (force || (new_last_task_alive - tm->orch->tensormap_last_cleanup >= PTO2_TENSORMAP_CLEANUP_INTERVAL)) {
        pto2_tensormap_cleanup_retired(tm, tm->orch->tensormap_last_cleanup, new_last_task_alive);
        tm->orch->tensormap_last_cleanup = new_last_task_alive;
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
