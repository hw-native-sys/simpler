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
 * 3. Per-task per-ring entry tracking for efficient cleanup
 * 4. OVERLAP DETECTION: Detects dependencies for overlapping sub-regions
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
// TensorMap Lookup Profiling (must precede inline lookup/insert methods)
// =============================================================================
#ifndef PTO2_TENSORMAP_PROFILING
#define PTO2_TENSORMAP_PROFILING 0
#endif

#if PTO2_TENSORMAP_PROFILING
extern uint64_t g_lookup_chain_total;
extern uint64_t g_lookup_count;
extern int32_t  g_lookup_chain_max;
extern uint64_t g_lookup_overlap_checks;
extern uint64_t g_lookup_overlap_hits;
extern uint64_t g_insert_count;
#endif

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap entry structure
 * Maps tensor region -> producer task ID
 *
 * Stored in ring buffer pool with lazy invalidation:
 * - Entry is valid only if local_id >= last_task_alives[ring_id]
 * - Stale entries skipped during lookup
 * - Pool wraps around, overwriting stale entries
 */
struct PTO2TensorMapEntry {
    bool with_alloc{true};     // True if entry is task output, False if entry is task inout
    PTO2TaskId producer_task_id;  // raw: (ring_id << 32) | local_id
    PTO2TensorMapEntry* next_in_bucket;    // Offset to next entry in hash bucket (-1 = end)
    PTO2TensorMapEntry* prev_in_bucket;    // Offset to prev entry in hash bucket (-1 = head is buckets[bucket])
    PTO2TensorMapEntry* next_in_task;      // Offset to next entry for same task (-1 = end)
    PTO2TensorMapEntry* prev_in_task;      // Offset to prev entry for same task (-1 = head is task_entry_head[slot])
    int32_t bucket_index;      // != -1 if entry is linked in a bucket chain
                               // CRITICAL: Must be set -1 before overwriting!
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

    // Per-ring per-task entry tracking (for efficient bucket cleanup)
    // Indexed by [ring_id][local_id & (task_window_sizes[ring_id] - 1)]
    PTO2TensorMapEntry** task_entry_heads[PTO2_MAX_RING_DEPTH];
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];  // Per-ring task window size (for slot masking)

    // Per-ring validity threshold (for lazy invalidation)
    int32_t last_task_alives[PTO2_MAX_RING_DEPTH];  // Cached from shared memory per ring

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
            // Must compute hash BEFORE clearing tensor
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
    bool init(int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    /**
     * Initialize TensorMap with default sizes
     */
    bool init_default(const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    /**
     * Destroy TensorMap and free resources
     */
    void destroy();

    /**
     * Update validity threshold from shared memory
     * Called periodically to refresh the lazy invalidation threshold.
     *
     * @param last_task_alive  Current value from shared memory
     */
    void sync_validity(int32_t ring_id, int32_t last_task_alive) {
        this->last_task_alives[ring_id] = last_task_alive;
    }

    /**
     * Lookup producer for a tensor region
     *
     * Searches the hash table for a matching region.
     * Returns producer entry if found and valid.
     * Stale entries from different rings are skipped (not truncated).
     *
     * @param tensor  Tensor to look up
     * @param result  Output: stack-allocated result buffer
     */
    void lookup(const Tensor& tensor, PTO2LookupResult& result) {
        uint32_t bucket_index = hash(tensor.buffer.addr);
        PTO2TensorMapEntry* cur_entry = buckets[bucket_index];

        result.count = 0;
#if PTO2_TENSORMAP_PROFILING
        g_lookup_count++;
        int32_t chain_len = 0;
#endif

        while (cur_entry != nullptr) {
            // Prefetch next entry to hide pointer-chasing latency.
            // entry_valid() + is_overlap() computation provides hide time.
            PTO2TensorMapEntry* next_entry = cur_entry->next_in_bucket;
            if (next_entry) __builtin_prefetch(next_entry, 0, 0);
#if PTO2_TENSORMAP_PROFILING
            chain_len++;
#endif
            // Skip stale entries (no chain truncation — entries from different
            // rings can be interleaved, so a stale entry from one ring does NOT
            // imply subsequent entries from other rings are also stale)
            if (!entry_valid(*cur_entry)) {
                cur_entry = next_entry;
                continue;
            }

            // Entry is valid - check if regions OVERLAP (not just exact match)
            // Since we hash only by base_ptr, all entries in this bucket have
            // potential to overlap. We must check actual byte-range overlap.
            if (tensor.buffer.addr == cur_entry->tensor.buffer.addr) {
#if PTO2_TENSORMAP_PROFILING
                g_lookup_overlap_checks++;
#endif
                auto overlap_status = tensor.is_overlap(cur_entry->tensor);
                if (overlap_status != OverlapStatus::NO_OVERLAP) {
                    result.push(cur_entry, overlap_status);
#if PTO2_TENSORMAP_PROFILING
                    g_lookup_overlap_hits++;
#endif
                }
            }

            // Move to next entry
            cur_entry = next_entry;
        }
#if PTO2_TENSORMAP_PROFILING
        g_lookup_chain_total += chain_len;
        if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
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
    void insert(const Tensor& tensor, PTO2TaskId producer_task_id, bool with_alloc) {
#if PTO2_TENSORMAP_PROFILING
        g_insert_count++;
#endif
        // Prefetch bucket head and task_entry_head early; new_entry() + field
        // initialization below provides hide time for these RFOs.
        uint32_t bucket_index = hash(tensor.buffer.addr);
        __builtin_prefetch(&buckets[bucket_index], 1, 0);
        auto ring_id = producer_task_id.ring();
        auto local_id = producer_task_id.local();
        int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);
        __builtin_prefetch(&task_entry_heads[ring_id][task_slot], 1, 0);

        // Allocate entry from ring buffer pool
        PTO2TensorMapEntry* entry = new_entry();

        // Initialize new entry
        entry->tensor.copy(tensor);
        entry->producer_task_id = producer_task_id;
        entry->with_alloc = with_alloc;

        // Insert at head of hash bucket (maintains task_id descending order)
        entry->bucket_index = bucket_index;
        entry->next_in_bucket = buckets[bucket_index];
        // Update old head's prev pointer
        if (entry->next_in_bucket != nullptr) {
            entry->next_in_bucket->prev_in_bucket = entry;
        }
        buckets[entry->bucket_index] = entry;
        entry->prev_in_bucket = nullptr;  // New head has no predecessor

        // Link to task's entry list (for cleanup), indexed by ring and local slot
        entry->next_in_task = task_entry_heads[ring_id][task_slot];
        entry->prev_in_task = nullptr;  // New head has no predecessor
        // Update old head's prev pointer
        if (entry->next_in_task != nullptr) {
            entry->next_in_task->prev_in_task = entry;
        }
        task_entry_heads[ring_id][task_slot] = entry;
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
    void cleanup_retired(int32_t ring_id, int32_t old_last_task_alive, int32_t new_last_task_alive) {
        // Iterate through retired tasks on this ring and remove their entries
        for (int32_t local_id = old_last_task_alive; local_id < new_last_task_alive; local_id++) {
            int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);
            PTO2TensorMapEntry* cur_entry = task_entry_heads[ring_id][task_slot];

            while (cur_entry != nullptr) {
                PTO2TensorMapEntry* next_entry = cur_entry->next_in_task;  // Save before clearing
                // Only remove if this entry belongs to the retiring task
                // (slot may have been reused by a newer task)
                debug_assert(cur_entry->producer_task_id ==
                             pto2_make_task_id(static_cast<uint8_t>(ring_id),
                                               static_cast<uint32_t>(local_id)));
                free_entry(*cur_entry);
                cur_entry = next_entry;
            }

            // Clear task's entry head (slot will be reused by local_id + task_window_sizes[ring_id])
            task_entry_heads[ring_id][task_slot] = nullptr;
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
        int32_t ring_id = pto2_task_id_ring(entry.producer_task_id);
        int32_t local_id = static_cast<int32_t>(pto2_task_id_local(entry.producer_task_id));
        return local_id >= last_task_alives[ring_id];
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
            // Entry is the head of its task chain, update task_entry_heads
            int32_t ring_id = pto2_task_id_ring(entry.producer_task_id);
            int32_t local_id = static_cast<int32_t>(pto2_task_id_local(entry.producer_task_id));
            int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);
            task_entry_heads[ring_id][task_slot] = entry.next_in_task;
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

#if PTO2_TENSORMAP_PROFILING
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
