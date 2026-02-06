/**
 * PTO Runtime2 - TensorMap Interface
 * 
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps TensorRegion -> producer task ID
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
#include "pto_logical_tensor.h"

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap structure
 * 
 * Hash table with ring buffer entry pool and lazy invalidation.
 */
typedef struct {
    // Hash table buckets (fixed size, power of 2)
    int32_t* buckets;             // Array of offsets into entry_pool (-1 = empty)
    int32_t  num_buckets;         // Must be power of 2 for fast modulo
    
    // Entry pool as ring buffer
    PTO2TensorMapEntry* entry_pool;   // Ring buffer of entries
    int32_t pool_size;                // Total pool capacity
    int32_t pool_head;                // Next allocation position (wraps around)
    
    // Per-task entry tracking (for efficient bucket cleanup)
    int32_t* task_entry_head;     // Per-task head offset (-1 = no entries)
                                  // Indexed by task_id % TASK_WINDOW_SIZE
    
    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;      // Cached value from shared memory
    
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
 * Returns producer task ID if found and valid, -1 otherwise.
 * 
 * Chain truncation: When first stale entry is found, truncates
 * the rest of the chain (all subsequent entries are also stale).
 * 
 * @param tm      TensorMap
 * @param region  Tensor region to look up
 * @return Producer task ID, or -1 if not found
 */
int32_t pto2_tensormap_lookup(PTO2TensorMap* tm, PTO2TensorRegion* region);

/**
 * Insert a new entry (called when task produces output)
 * 
 * Allocates from ring buffer pool, may overwrite stale entries.
 * Inserts at head of hash bucket chain (maintains task_id ordering).
 * 
 * @param tm                TensorMap
 * @param region            Tensor region produced
 * @param producer_task_id  Task ID of producer
 */
void pto2_tensormap_insert(PTO2TensorMap* tm, PTO2TensorRegion* region, 
                            int32_t producer_task_id);

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
void pto2_tensormap_cleanup_retired(PTO2TensorMap* tm, 
                                     int32_t old_last_task_alive,
                                     int32_t new_last_task_alive);

// =============================================================================
// Internal Helpers (exposed for testing)
// =============================================================================

/**
 * Compute hash for tensor region
 */
uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, PTO2TensorRegion* region);

/**
 * Check if entry is valid (producer has not retired)
 */
static inline bool pto2_tensormap_entry_valid(PTO2TensorMap* tm, PTO2TensorMapEntry* entry) {
    return entry->producer_task_id >= tm->last_task_alive;
}

/**
 * Check if two regions OVERLAP (for dependency detection)
 * 
 * Returns true if regions have same base_ptr AND their byte ranges
 * [offset, offset+size) intersect.
 * 
 * Overlap condition: (a.start < b.end) AND (b.start < a.end)
 */
bool pto2_region_overlap(PTO2TensorRegion* a, PTO2TensorRegion* b);

/**
 * Check if two regions match exactly (legacy, for compatibility)
 */
bool pto2_region_match(PTO2TensorRegion* a, PTO2TensorRegion* b);

/**
 * Remove entry from its bucket chain
 * Called during pool wrap-around or cleanup.
 */
void pto2_tensormap_remove_from_bucket(PTO2TensorMap* tm, PTO2TensorMapEntry* entry);

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

/**
 * Get average bucket chain length
 */
float pto2_tensormap_avg_chain_length(PTO2TensorMap* tm);

// =============================================================================
// Extended TensorMap (for LogicalTensor support)
// =============================================================================

/**
 * Extended TensorMap structure
 * 
 * Supports multi-dimensional tensors with view/reshape/transpose operations.
 * Uses PTO2TensorMapEntryEx for bounding box based overlap detection.
 */
typedef struct {
    // Hash table buckets (fixed size, power of 2)
    int32_t* buckets;             // Array of offsets into entry_pool (-1 = empty)
    int32_t  num_buckets;         // Must be power of 2 for fast modulo
    
    // Entry pool as ring buffer
    PTO2TensorMapEntryEx* entry_pool;   // Ring buffer of extended entries
    int32_t pool_size;                   // Total pool capacity
    int32_t pool_head;                   // Next allocation position (wraps around)
    
    // Per-task entry tracking (for efficient bucket cleanup)
    int32_t* task_entry_head;     // Per-task head offset (-1 = no entries)
                                  // Indexed by task_id % TASK_WINDOW_SIZE
    
    // Validity threshold (for lazy invalidation)
    int32_t last_task_alive;      // Cached value from shared memory
    
} PTO2TensorMapEx;

// -----------------------------------------------------------------------------
// Extended TensorMap Initialization
// -----------------------------------------------------------------------------

/**
 * Initialize extended TensorMap
 */
bool pto2_tensormapex_init(PTO2TensorMapEx* tm, int32_t num_buckets, int32_t pool_size);

/**
 * Initialize extended TensorMap with default sizes
 */
bool pto2_tensormapex_init_default(PTO2TensorMapEx* tm);

/**
 * Destroy extended TensorMap
 */
void pto2_tensormapex_destroy(PTO2TensorMapEx* tm);

/**
 * Reset extended TensorMap to empty state
 */
void pto2_tensormapex_reset(PTO2TensorMapEx* tm);

// -----------------------------------------------------------------------------
// Extended TensorMap Operations
// -----------------------------------------------------------------------------

/**
 * Update validity threshold
 */
void pto2_tensormapex_sync_validity(PTO2TensorMapEx* tm, int32_t last_task_alive);

/**
 * Insert a LogicalTensor output into the extended TensorMap
 * 
 * Stores the tensor's bounding box and full layout info for overlap detection.
 * 
 * @param tm                Extended TensorMap
 * @param tensor            Logical tensor being produced
 * @param producer_task_id  Task ID of the producer
 */
void pto2_tensormapex_insert(PTO2TensorMapEx* tm, 
                              const PTO2LogicalTensor* tensor, 
                              int32_t producer_task_id);

/**
 * Lookup single producer for a LogicalTensor (first overlap found)
 * 
 * Returns the task ID of the most recent producer that overlaps with the input.
 * 
 * @param tm      Extended TensorMap
 * @param tensor  Logical tensor to look up
 * @return Producer task ID, or -1 if no overlapping producer
 */
int32_t pto2_tensormapex_lookup(PTO2TensorMapEx* tm, const PTO2LogicalTensor* tensor);

/**
 * Find ALL producers that overlap with a LogicalTensor
 * 
 * Returns all task IDs whose outputs overlap with the input tensor.
 * This is needed for correct dependency tracking when multiple tasks
 * write to overlapping regions.
 * 
 * @param tm            Extended TensorMap
 * @param tensor        Logical tensor to look up
 * @param producer_ids  Output array of producer task IDs
 * @param max_producers Maximum size of output array
 * @return Number of overlapping producers found
 */
int32_t pto2_tensormapex_lookup_all(PTO2TensorMapEx* tm, 
                                     const PTO2LogicalTensor* tensor,
                                     int32_t* producer_ids,
                                     int32_t max_producers);

/**
 * Cleanup retired entries from extended TensorMap
 */
void pto2_tensormapex_cleanup_retired(PTO2TensorMapEx* tm, 
                                       int32_t old_last_task_alive,
                                       int32_t new_last_task_alive);

// -----------------------------------------------------------------------------
// Extended TensorMap Internal Helpers
// -----------------------------------------------------------------------------

/**
 * Compute hash for a LogicalTensor (based on raw_base only)
 */
uint32_t pto2_tensormapex_hash(PTO2TensorMapEx* tm, const PTO2LogicalTensor* tensor);

/**
 * Check if extended entry is valid
 */
static inline bool pto2_tensormapex_entry_valid(PTO2TensorMapEx* tm, PTO2TensorMapEntryEx* entry) {
    return entry->producer_task_id >= tm->last_task_alive;
}

/**
 * Check if LogicalTensor overlaps with extended entry (bounding box check)
 */
bool pto2_tensormapex_overlap(const PTO2LogicalTensor* tensor, const PTO2TensorMapEntryEx* entry);

/**
 * Remove extended entry from its bucket chain
 */
void pto2_tensormapex_remove_from_bucket(PTO2TensorMapEx* tm, PTO2TensorMapEntryEx* entry);

// -----------------------------------------------------------------------------
// Extended TensorMap Debug
// -----------------------------------------------------------------------------

/**
 * Print extended TensorMap statistics
 */
void pto2_tensormapex_print_stats(PTO2TensorMapEx* tm);

/**
 * Get count of valid entries in extended TensorMap
 */
int32_t pto2_tensormapex_valid_count(PTO2TensorMapEx* tm);

#endif // PTO_TENSORMAP_H
