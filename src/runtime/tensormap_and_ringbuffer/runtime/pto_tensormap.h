/**
 * PTO Runtime2 - TensorMap Interface
 *
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps Tensor -> producer task ID
 * - Used by pto_submit_task() to find dependencies
 *
 * Key design features:
 * 1. Ring buffer pool for entries (no malloc/free)
 * 2. Lazy invalidation (entries become stale when producer retires)
 * 3. Chain truncation optimization (truncate on first stale entry)
 * 4. Per-task entry tracking for efficient cleanup
 * 5. OVERLAP DETECTION: Detects dependencies for overlapping sub-regions
 *
 * Hash table with chaining:
 * - buckets[] array of head offsets
 * - Entries linked via next_in_bucket
 * - Insert at head (newest first) for sorted chains
 *
 * CRITICAL: Hash only by base_ptr
 * ==============================
 * For overlap detection to work, ALL sub-regions of the same base tensor
 * MUST be in the SAME hash bucket. This allows lookup to compare all
 * potentially overlapping regions.
 *
 * Overlap detection: Two regions create a dependency if:
 *   1. Same base_ptr (raw tensor pointer)
 *   2. Byte ranges [offset, offset+size) intersect
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#pragma once

#include "common.h"
#include "pto_runtime2_types.h"
#include "tensor.h"

struct PTO2OrchestratorState;  // forward declare

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap entry structure
 * Maps tensor region -> producer task ID
 *
 * Stored in ring buffer pool with lazy invalidation:
 * - Entry is valid only if producer_task_id >= last_task_alive
 * - Stale entries ignored during lookup
 * - Pool wraps around, overwriting stale entries
 *
 * Chain truncation optimization:
 * - Entries in bucket chains sorted by task_id (newest first)
 * - When lookup hits stale entry, truncate rest of chain
 */
struct PTO2TensorMapEntry {
    bool with_alloc{true};     // True if entry is task output, False if entry is task inout
    int32_t producer_task_id;  // Task that produces this region
    PTO2TensorMapEntry* next_in_bucket;    // Offset to next entry in hash bucket (-1 = end)
    PTO2TensorMapEntry* prev_in_bucket;    // Offset to prev entry in hash bucket (-1 = head is buckets[bucket])
    PTO2TensorMapEntry* next_in_task;      // Offset to next entry for same task (-1 = end)
    PTO2TensorMapEntry* prev_in_task;      // Offset to prev entry for same task (-1 = head is task_entry_head[slot])
    int32_t bucket_index;      // != -1 if entry is linked in a bucket chain
                               // CRITICAL: Must be set -1 before overwriting!
    uint64_t addr;
    Tensor tensor;             // Tensor descriptor key
};

/**
 * Stack-allocated lookup result (avoids heap allocation per lookup)
 */
#define PTO2_LOOKUP_MAX_RESULTS 16
// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
struct PTO2LookupResult {
    struct Entry {
        PTO2TensorMapEntry* entry;
        OverlapStatus overlap_status;
    };
    Entry entries[PTO2_LOOKUP_MAX_RESULTS];
    int32_t count{0};

    void push(PTO2TensorMapEntry* entry, OverlapStatus s) {
        if (count < PTO2_LOOKUP_MAX_RESULTS) {
            entries[count++] = {entry, s};
        }
    }
};

/**
 * TensorMap structure
 *
 * Hash table with ring buffer entry pool and lazy invalidation.
 */
struct PTO2TensorMap {
    // Hash table buckets (fixed size, power of 2)
    PTO2TensorMapEntry** buckets;     // Array of offsets into entry_pool (-1 = empty)
    int32_t num_buckets;  // Must be power of 2 for fast modulo

    // Entry pool as ring buffer
    PTO2TensorMapEntry* entry_pool;  // Ring buffer of entries
    PTO2TensorMapEntry** free_entry_list;        // free entry ids
    int32_t pool_size;               // Total pool capacity
    int32_t next_entry_idx;          // id when next entry insert
    int32_t free_num;                // free entry number in entry pool

    // Per-task entry tracking (for efficient bucket cleanup)
    PTO2TensorMapEntry** task_entry_head;  // Per-task head offset (-1 = no entries)
                               // Indexed by task_id % TASK_WINDOW_SIZE

    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;  // Cached value from shared memory

    PTO2OrchestratorState* orch{nullptr};

    // new_entry目前不负责分配属性，仅分配内存
    PTO2TensorMapEntry* new_entry() {
        if (free_num > 0) {
            PTO2TensorMapEntry* res = free_entry_list[--free_num];
            debug_assert(res->bucket_index == -1);
            return res;
        }
        always_assert(next_entry_idx < pool_size);
        PTO2TensorMapEntry* res = &entry_pool[next_entry_idx++];
        debug_assert(res->bucket_index == -1);
        return res;
    }

    void free_entry(PTO2TensorMapEntry& entry) {
        always_assert(entry.bucket_index != -1); // 必须保证仍在桶中

        // Update predecessor's next pointer (O(1) via prev_in_bucket)
        if (entry.prev_in_bucket == nullptr) {
            // Entry is the head of its bucket chain, update bucket head
            // Must compute hash BEFORE clearing tensor (tensor.data() needs valid tensor_pool)
            buckets[entry.bucket_index] = entry.next_in_bucket;
        } else {
            entry.prev_in_bucket->next_in_bucket = entry.next_in_bucket;
        }

        // Update successor's prev pointer
        if (entry.next_in_bucket != nullptr) {
            entry.next_in_bucket->prev_in_bucket = entry.prev_in_bucket;
        }

        // Clear tensor AFTER bucket chain manipulation (hash computation needs valid tensor)
        entry.tensor = Tensor();

        free_entry_list[free_num++] = &entry;
        entry.bucket_index = -1;
        entry.next_in_bucket = nullptr;
        entry.prev_in_bucket = nullptr;
        entry.next_in_task = nullptr;
        entry.prev_in_task = nullptr;
    }

    // =============================================================================
    // TensorMap API
    // =============================================================================

    /**
     * Initialize TensorMap
     *
     * @param num_buckets Number of hash buckets (must be power of 2)
     * @param pool_size   Size of entry pool
     * @return true on success, false on allocation failure
     */
    bool init(int32_t num_buckets, int32_t pool_size);

    /**
     * Initialize TensorMap with default sizes
     */
    bool init_default();

    /**
     * Destroy TensorMap and free resources
     */
    void destroy();

    /**
     * Reset TensorMap to empty state
     */
    void reset();

    /**
     * Update validity threshold from shared memory
     * Called periodically to refresh the lazy invalidation threshold.
     *
     * @param last_task_alive  Current value from shared memory
     */
    void sync_validity(int32_t last_task_alive) { this->last_task_alive = last_task_alive; }

    /**
     * Lookup producer for a tensor region
     *
     * Searches the hash table for a matching region.
     * Returns producer entry if found and valid.
     *
     * Chain truncation: When first stale entry is found, truncates
     * the rest of the chain (all subsequent entries are also stale).
     *
     * @param tensor  Tensor to look up
     * @param result  Output: stack-allocated result buffer
     */
    void lookup(const Tensor& tensor, PTO2LookupResult& result) {
        auto& query_tensor_data = tensor.data();
        uint32_t bucket_index = hash(query_tensor_data.buffer.addr);
        PTO2TensorMapEntry** prev_ptr = &buckets[bucket_index];  // For truncation
        PTO2TensorMapEntry* cur_entry = *prev_ptr;

        result.count = 0;

        while (cur_entry != nullptr) {
            // Check validity first
            if (!entry_valid(*cur_entry)) {
                // ========== STALE ENTRY: Truncate chain here ==========
                // All subsequent entries are guaranteed to be stale too!
                // Truncate: unlink this and all following entries
                *prev_ptr = nullptr;  // Terminate chain at previous entry

                // Mark truncated entries as not in bucket (for correct reuse)
                while (cur_entry != nullptr) {
                    PTO2TensorMapEntry* next_entry = cur_entry->next_in_bucket;
                    remove_entry(*cur_entry);
                    cur_entry = next_entry;
                }
                return;
            }

            // Entry is valid - check if regions OVERLAP (not just exact match)
            // Since we hash only by base_ptr, all entries in this bucket have
            // potential to overlap. We must check actual byte-range overlap.
            if (query_tensor_data.buffer.addr == cur_entry->addr) {
                auto overlap_status = query_tensor_data.is_overlap(cur_entry->tensor.data());
                if (overlap_status != OverlapStatus::NO_OVERLAP) {
                    result.push(cur_entry, overlap_status);
                }
            }

            // Move to next entry
            prev_ptr = &cur_entry->next_in_bucket;
            cur_entry = *prev_ptr;
        }
    }

    /**
     * Insert a new entry (called when task produces output)
     *
     * Allocates from ring buffer pool, may overwrite stale entries.
     * Inserts at head of hash bucket chain (maintains task_id ordering).
     *
     * @param tensor            Tensor produced
     * @param producer_task_id  Task ID of producer
     */
    void insert(const Tensor& tensor, int32_t producer_task_id, bool with_alloc) {
        // Allocate entry from ring buffer pool
        PTO2TensorMapEntry* entry = new_entry();

        // Initialize new entry
        entry->tensor = tensor;
        entry->producer_task_id = producer_task_id;
        entry->with_alloc = with_alloc;

        // Insert at head of hash bucket (maintains task_id descending order)
        entry->addr = tensor.data().buffer.addr;
        entry->bucket_index = hash(entry->addr);
        entry->next_in_bucket = buckets[entry->bucket_index];
        // Update old head's prev pointer
        if (entry->next_in_bucket != nullptr) {
            entry->next_in_bucket->prev_in_bucket = entry;
        }
        buckets[entry->bucket_index] = entry;
        entry->prev_in_bucket = nullptr;  // New head has no predecessor

        // Link to task's entry list (for cleanup)
        int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        entry->next_in_task = task_entry_head[task_slot];
        entry->prev_in_task = nullptr;  // New head has no predecessor
        // Update old head's prev pointer
        if (entry->next_in_task != nullptr) {
            entry->next_in_task->prev_in_task = entry;
        }
        task_entry_head[task_slot] = entry;
    }

    /**
     * Cleanup stale entries for retired tasks
     *
     * Called periodically by Orchestrator when last_task_alive advances.
     * Removes entries from bucket chains for tasks in [old, new) range.
     *
     * @param old_last_task_alive  Previous threshold
     * @param new_last_task_alive  New threshold
     */
    void cleanup_retired(int32_t old_last_task_alive, int32_t new_last_task_alive) {
        // Iterate through retired tasks and remove their entries from bucket chains
        for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
            int32_t task_slot = task_id & (PTO2_TASK_WINDOW_SIZE - 1);
            PTO2TensorMapEntry* cur_entry = task_entry_head[task_slot];

            while (cur_entry != nullptr) {
                PTO2TensorMapEntry* next_entry = cur_entry->next_in_task;  // Save before clearing
                // Only remove if this entry belongs to the retiring task
                // (slot may have been reused by a newer task)
                debug_assert(cur_entry->producer_task_id == task_id);
                free_entry(*cur_entry);
                cur_entry = next_entry;
            }

            // Clear task's entry head (slot will be reused by task_id + TASK_WINDOW_SIZE)
            task_entry_head[task_slot] = nullptr;
        }
    }

    // =============================================================================
    // Internal Helpers (exposed for testing)
    // =============================================================================

    /**
     * Compute hash for tensor addr
     */
    uint32_t hash(uint64_t key) {
        // Improve distribution by mixing bits (pointers often have aligned low bits)
        key = key ^ (key >> 16);
        key = key ^ (key >> 32);

        // Use bitwise AND for power-of-2 modulo (faster than %)
        return (uint32_t)(key & (num_buckets - 1));
    }

    /**
     * Check if entry is valid (producer has not retired)
     */
    bool entry_valid(const PTO2TensorMapEntry& entry) const {
        return entry.producer_task_id >= last_task_alive;
    }

    void remove_entry(PTO2TensorMapEntry& entry) {
        remove_from_task(entry);
        free_entry(entry);
    }

    /**
     * Remove entry from its task chain (O(1) with prev pointer)
     * Called during pool wrap-around to unlink reused entries.
     */
    void remove_from_task(PTO2TensorMapEntry& entry) {
        always_assert(entry.bucket_index != -1); // 必须保证仍在桶中
        // Update predecessor's next pointer (O(1) via prev_in_task)
        if (entry.prev_in_task == nullptr) {
            // Entry is the head of its task chain, update task_entry_head
            int32_t task_slot = entry.producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
            task_entry_head[task_slot] = entry.next_in_task;
        } else {
            entry.prev_in_task->next_in_task = entry.next_in_task;
        }

        // Update successor's prev pointer
        if (entry.next_in_task != nullptr) {
            entry.next_in_task->prev_in_task = entry.prev_in_task;
        }

        entry.next_in_task = nullptr;
        entry.prev_in_task = nullptr;
    }

    // =============================================================================
    // Debug Utilities
    // =============================================================================

    /**
     * Print TensorMap statistics
     */
    void print_stats();

    /**
     * Get count of valid entries
     */
    int32_t valid_count();

    // =============================================================================
    // TensorMap Synchronization
    // =============================================================================

    /**
     * Sync TensorMap validity threshold from shared memory
     *
     * Called periodically to refresh the lazy invalidation threshold.
     * Also triggers cleanup if threshold has advanced significantly.
     */
    void sync_tensormap();
};

// =============================================================================
// TensorMap Lookup Profiling
// =============================================================================
#ifndef PTO2_ORCH_PROFILING
#define PTO2_ORCH_PROFILING 1
#endif

#if PTO2_ORCH_PROFILING
struct PTO2TensorMapProfilingData {
    uint64_t lookup_chain_total;
    uint64_t lookup_count;
    int32_t  lookup_chain_max;
    uint64_t overlap_checks;
    uint64_t overlap_hits;
    uint64_t insert_count;
};

PTO2TensorMapProfilingData pto2_tensormap_get_profiling();
#endif
