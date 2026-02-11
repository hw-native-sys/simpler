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

#ifndef PTO_TENSORMAP_H
#define PTO_TENSORMAP_H

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
typedef struct {
    Tensor tensor;             // Tensor descriptor key
    int32_t producer_task_id;  // Task that produces this region
    int32_t next_in_bucket;    // Offset to next entry in hash bucket (-1 = end)
    int32_t prev_in_bucket;    // Offset to prev entry in hash bucket (-1 = head is buckets[bucket])
    int32_t next_in_task;      // Offset to next entry for same task (-1 = end)
    int32_t prev_in_task;      // Offset to prev entry for same task (-1 = head is task_entry_head[slot])
    bool in_bucket;            // True if entry is linked in a bucket chain
                               // CRITICAL: Must be set false before overwriting!
    bool with_alloc{true};     // True if entry is task output, False if entry is task inout
} PTO2TensorMapEntry;

/**
 * TensorMap structure
 *
 * Hash table with ring buffer entry pool and lazy invalidation.
 */
typedef struct {
    // Hash table buckets (fixed size, power of 2)
    int32_t* buckets;     // Array of offsets into entry_pool (-1 = empty)
    int32_t num_buckets;  // Must be power of 2 for fast modulo

    // Entry pool as ring buffer
    PTO2TensorMapEntry* entry_pool;  // Ring buffer of entries
    int32_t pool_size;               // Total pool capacity
    int32_t pool_head;               // Next allocation position (wraps around)

    // Per-task entry tracking (for efficient bucket cleanup)
    int32_t* task_entry_head;  // Per-task head offset (-1 = no entries)
                               // Indexed by task_id % TASK_WINDOW_SIZE

    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;  // Cached value from shared memory

    PTO2OrchestratorState* orch{nullptr};
} PTO2TensorMap;

// =============================================================================
// TensorMap API
// =============================================================================

/**
 * Initialize TensorMap
 *
 * @param tm          TensorMap to initialize
 * @param num_buckets Number of hash buckets (must be power of 2)
 * @param pool_size   Size of entry pool
 * @return true on success, false on allocation failure
 */
bool pto2_tensormap_init(PTO2TensorMap* tm, int32_t num_buckets, int32_t pool_size);

/**
 * Initialize TensorMap with default sizes
 */
bool pto2_tensormap_init_default(PTO2TensorMap* tm);

/**
 * Destroy TensorMap and free resources
 */
void pto2_tensormap_destroy(PTO2TensorMap* tm);

/**
 * Reset TensorMap to empty state
 */
void pto2_tensormap_reset(PTO2TensorMap* tm);

/**
 * Update validity threshold from shared memory
 * Called periodically to refresh the lazy invalidation threshold.
 *
 * @param tm               TensorMap
 * @param last_task_alive  Current value from shared memory
 */
void pto2_tensormap_sync_validity(PTO2TensorMap* tm, int32_t last_task_alive);

/**
 * Lookup producer for a tensor region
 *
 * Searches the hash table for a matching region.
 * Returns producer entry if found and valid.
 *
 * Chain truncation: When first stale entry is found, truncates
 * the rest of the chain (all subsequent entries are also stale).
 *
 * @param tm      TensorMap
 * @param tensor  Tensor to look up
 * @return Producer entry, and overlap status
 */
std::vector<std::pair<PTO2TensorMapEntry*, OverlapStatus>> pto2_tensormap_lookup(PTO2TensorMap* tm, Tensor* tensor);

/**
 * Insert a new entry (called when task produces output)
 *
 * Allocates from ring buffer pool, may overwrite stale entries.
 * Inserts at head of hash bucket chain (maintains task_id ordering).
 *
 * @param tm                TensorMap
 * @param tensor            Tensor produced
 * @param producer_task_id  Task ID of producer
 */
void pto2_tensormap_insert(PTO2TensorMap* tm, Tensor* tensor, int32_t producer_task_id, bool with_alloc);

/**
 * Cleanup stale entries for retired tasks
 *
 * Called periodically by Orchestrator when last_task_alive advances.
 * Removes entries from bucket chains for tasks in [old, new) range.
 *
 * @param tm                   TensorMap
 * @param old_last_task_alive  Previous threshold
 * @param new_last_task_alive  New threshold
 */
void pto2_tensormap_cleanup_retired(PTO2TensorMap* tm, int32_t old_last_task_alive, int32_t new_last_task_alive);

// =============================================================================
// Internal Helpers (exposed for testing)
// =============================================================================

/**
 * Compute hash for tensor region
 */
uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, Tensor* tensor);

/**
 * Check if entry is valid (producer has not retired)
 */
static inline bool pto2_tensormap_entry_valid(PTO2TensorMap* tm, PTO2TensorMapEntry* entry) {
    return entry->producer_task_id >= tm->last_task_alive;
}

void pto2_tensormap_remove_entry(PTO2TensorMap& tm, PTO2TensorMapEntry* entry);

/**
 * Remove entry from its bucket chain (O(1) with prev pointer)
 * Called during pool wrap-around or cleanup.
 */
void pto2_tensormap_remove_from_bucket(PTO2TensorMap* tm, PTO2TensorMapEntry* entry);

/**
 * Remove entry from its task chain (O(1) with prev pointer)
 * Called during pool wrap-around to unlink reused entries.
 */
void pto2_tensormap_remove_from_task(PTO2TensorMap* tm, PTO2TensorMapEntry* entry);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print TensorMap statistics
 */
void pto2_tensormap_print_stats(PTO2TensorMap* tm);

/**
 * Get count of valid entries
 */
int32_t pto2_tensormap_valid_count(PTO2TensorMap* tm);

// =============================================================================
// TensorMap Synchronization
// =============================================================================

/**
 * Sync TensorMap validity threshold from shared memory
 *
 * Called periodically to refresh the lazy invalidation threshold.
 * Also triggers cleanup if threshold has advanced significantly.
 */
void pto2_orchestrator_sync_tensormap(PTO2TensorMap* tm);

#endif  // PTO_TENSORMAP_H
