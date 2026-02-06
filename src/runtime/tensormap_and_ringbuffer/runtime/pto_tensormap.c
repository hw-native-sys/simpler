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
#include <stdio.h>

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
    
    tm->pool_size = pool_size;
    tm->pool_head = 0;
    
    // Initialize all entries as not in bucket
    for (int32_t i = 0; i < pool_size; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
    }
    
    // Allocate per-task entry tracking
    tm->task_entry_head = (int32_t*)malloc(PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    if (!tm->task_entry_head) {
        free(tm->entry_pool);
        free(tm->buckets);
        tm->entry_pool = NULL;
        tm->buckets = NULL;
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
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
    }
    
    // Reset per-task entry tracking
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }
    
    tm->pool_head = 0;
    tm->last_task_alive = 0;
}

// =============================================================================
// Hash Function
// =============================================================================

uint32_t pto2_tensormap_hash(PTO2TensorMap* tm, PTO2TensorRegion* region) {
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
    uint64_t key = (uint64_t)(uintptr_t)region->base_ptr;
    
    // Improve distribution by mixing bits (pointers often have aligned low bits)
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    // Use bitwise AND for power-of-2 modulo (faster than %)
    return (uint32_t)(key & (tm->num_buckets - 1));
}

// Check if two regions OVERLAP (not just exact match)
// 
// Overlap condition:
//   1. Same base_ptr (raw tensor pointer)
//   2. Same tile_index (different tiles are disjoint memory regions)
//   3. Byte ranges [offset, offset+size) intersect within the tile
//
// Note: tile_index represents a tile/block subdivision of the tensor.
// Different tile indices are treated as non-overlapping memory regions,
// even if their offsets are the same (offset is relative to tile start).
//
bool pto2_region_overlap(PTO2TensorRegion* a, PTO2TensorRegion* b) {
    // Must be same base tensor
    if (a->base_ptr != b->base_ptr) {
        return false;
    }
    
    // Must be same tile (different tiles don't overlap)
    if (a->tile_index != b->tile_index) {
        return false;
    }
    
    // Check 1D interval overlap within tile: [start_a, end_a) ∩ [start_b, end_b) ≠ ∅
    int32_t a_start = a->offset;
    int32_t a_end = a_start + a->size;
    int32_t b_start = b->offset;
    int32_t b_end = b_start + b->size;
    
    // Overlap exists if: (a_start < b_end) AND (b_start < a_end)
    return (a_start < b_end) && (b_start < a_end);
}

// Legacy exact match (kept for compatibility, not used in lookup)
bool pto2_region_match(PTO2TensorRegion* a, PTO2TensorRegion* b) {
    return a->base_ptr == b->base_ptr &&
           a->tile_index == b->tile_index &&
           a->offset == b->offset;
}

// =============================================================================
// Validity and Cleanup
// =============================================================================

void pto2_tensormap_sync_validity(PTO2TensorMap* tm, int32_t last_task_alive) {
    tm->last_task_alive = last_task_alive;
}

void pto2_tensormap_remove_from_bucket(PTO2TensorMap* tm, PTO2TensorMapEntry* entry) {
    if (!entry->in_bucket) {
        return;  // Already removed
    }
    
    uint32_t bucket = pto2_tensormap_hash(tm, &entry->region);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    int32_t target_offset = entry - tm->entry_pool;
    
    while (offset >= 0) {
        if (offset == target_offset) {
            *prev_ptr = entry->next_in_bucket;
            entry->in_bucket = false;
            entry->next_in_bucket = -1;
            return;
        }
        prev_ptr = &tm->entry_pool[offset].next_in_bucket;
        offset = *prev_ptr;
    }
}

void pto2_tensormap_cleanup_retired(PTO2TensorMap* tm, 
                                     int32_t old_last_task_alive,
                                     int32_t new_last_task_alive) {
    // Iterate through retired tasks and remove their entries from bucket chains
    for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
        int32_t task_slot = task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        int32_t offset = tm->task_entry_head[task_slot];
        
        while (offset >= 0) {
            PTO2TensorMapEntry* entry = &tm->entry_pool[offset];
            // Only remove if this entry belongs to the retiring task
            // (slot may have been reused by a newer task)
            if (entry->producer_task_id == task_id) {
                pto2_tensormap_remove_from_bucket(tm, entry);
            }
            offset = entry->next_in_task;
        }
        
        // Clear task's entry head (slot will be reused by task_id + TASK_WINDOW_SIZE)
        tm->task_entry_head[task_slot] = -1;
    }
}

// =============================================================================
// Lookup with Chain Truncation
// =============================================================================

int32_t pto2_tensormap_lookup(PTO2TensorMap* tm, PTO2TensorRegion* region) {
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    int32_t* prev_ptr = &tm->buckets[bucket];  // For truncation
    int32_t offset = *prev_ptr;
    
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
                offset = next;
            }
            
            return -1;  // Not found (and cleaned up stale tail)
        }
        
        // Entry is valid - check if regions OVERLAP (not just exact match)
        // Since we hash only by base_ptr, all entries in this bucket have
        // potential to overlap. We must check actual byte-range overlap.
        if (pto2_region_overlap(&entry->region, region)) {
            return entry->producer_task_id;  // FOUND (overlapping region)
        }
        
        // Move to next entry
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return -1;  // Not found
}

// =============================================================================
// Insert
// =============================================================================

void pto2_tensormap_insert(PTO2TensorMap* tm, PTO2TensorRegion* region, 
                            int32_t producer_task_id) {
    // Allocate entry from ring buffer pool
    int32_t entry_offset = tm->pool_head;
    PTO2TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    
    // Advance pool head (wrap around)
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;
    
    // ========== CRITICAL: MUST remove old entry from its bucket chain ==========
    // Even if entry is STALE (producer retired), it's still linked in its old 
    // bucket chain. If we overwrite without unlinking, the chain gets corrupted!
    if (entry->in_bucket) {
        pto2_tensormap_remove_from_bucket(tm, entry);
    }
    
    // Initialize new entry
    entry->region = *region;
    entry->producer_task_id = producer_task_id;
    
    // Insert at head of hash bucket (maintains task_id descending order)
    uint32_t bucket = pto2_tensormap_hash(tm, region);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;
    
    // Link to task's entry list (for cleanup)
    int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
    entry->next_in_task = tm->task_entry_head[task_slot];
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
    
    printf("=== TensorMap Statistics ===\n");
    printf("Pool size:       %d\n", tm->pool_size);
    printf("Pool head:       %d\n", tm->pool_head);
    printf("Num buckets:     %d\n", tm->num_buckets);
    printf("Valid entries:   %d\n", valid);
    printf("Stale entries:   %d\n", stale);
    printf("Empty buckets:   %d\n", empty_buckets);
    printf("Max chain len:   %d\n", max_chain);
    printf("Avg chain len:   %.2f\n", 
           non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    printf("Last task alive: %d\n", tm->last_task_alive);
    printf("============================\n");
}

int32_t pto2_tensormap_valid_count(PTO2TensorMap* tm) {
    int32_t count = 0;
    
    for (int32_t i = 0; i < tm->pool_size; i++) {
        if (tm->entry_pool[i].in_bucket && 
            pto2_tensormap_entry_valid(tm, &tm->entry_pool[i])) {
            count++;
        }
    }
    
    return count;
}

float pto2_tensormap_avg_chain_length(PTO2TensorMap* tm) {
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;
    
    for (int32_t b = 0; b < tm->num_buckets; b++) {
        int32_t chain_len = 0;
        int32_t offset = tm->buckets[b];
        
        while (offset >= 0) {
            chain_len++;
            offset = tm->entry_pool[offset].next_in_bucket;
        }
        
        if (chain_len > 0) {
            non_empty_buckets++;
            total_chain += chain_len;
        }
    }
    
    return non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0;
}

// =============================================================================
// Extended TensorMap Implementation (for LogicalTensor support)
// =============================================================================

// -----------------------------------------------------------------------------
// Initialization and Destruction
// -----------------------------------------------------------------------------

bool pto2_tensormapex_init(PTO2TensorMapEx* tm, int32_t num_buckets, int32_t pool_size) {
    // Validate power of 2 for fast modulo
    if ((num_buckets & (num_buckets - 1)) != 0) {
        return false;
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
    
    // Allocate extended entry pool
    tm->entry_pool = (PTO2TensorMapEntryEx*)calloc(pool_size, sizeof(PTO2TensorMapEntryEx));
    if (!tm->entry_pool) {
        free(tm->buckets);
        tm->buckets = NULL;
        return false;
    }
    
    tm->pool_size = pool_size;
    tm->pool_head = 0;
    
    // Initialize all entries
    for (int32_t i = 0; i < pool_size; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
        tm->entry_pool[i].is_deep_copy = false;
    }
    
    // Allocate per-task entry tracking
    tm->task_entry_head = (int32_t*)malloc(PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    if (!tm->task_entry_head) {
        free(tm->entry_pool);
        free(tm->buckets);
        tm->entry_pool = NULL;
        tm->buckets = NULL;
        return false;
    }
    
    // Initialize all task entry heads to -1
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }
    
    tm->last_task_alive = 0;
    
    return true;
}

bool pto2_tensormapex_init_default(PTO2TensorMapEx* tm) {
    return pto2_tensormapex_init(tm, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE);
}

void pto2_tensormapex_destroy(PTO2TensorMapEx* tm) {
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

void pto2_tensormapex_reset(PTO2TensorMapEx* tm) {
    // Reset all buckets to empty
    for (int32_t i = 0; i < tm->num_buckets; i++) {
        tm->buckets[i] = -1;
    }
    
    // Reset all entries
    for (int32_t i = 0; i < tm->pool_size; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
        tm->entry_pool[i].is_deep_copy = false;
    }
    
    // Reset per-task entry tracking
    for (int32_t i = 0; i < PTO2_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }
    
    tm->pool_head = 0;
    tm->last_task_alive = 0;
}

// -----------------------------------------------------------------------------
// Validity and Cleanup
// -----------------------------------------------------------------------------

void pto2_tensormapex_sync_validity(PTO2TensorMapEx* tm, int32_t last_task_alive) {
    tm->last_task_alive = last_task_alive;
}

void pto2_tensormapex_remove_from_bucket(PTO2TensorMapEx* tm, PTO2TensorMapEntryEx* entry) {
    if (!entry->in_bucket) {
        return;
    }
    
    // Compute hash for this entry
    uint64_t key = (uint64_t)(uintptr_t)entry->raw_base;
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    uint32_t bucket = (uint32_t)(key & (tm->num_buckets - 1));
    
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    int32_t target_offset = entry - tm->entry_pool;
    
    while (offset >= 0) {
        if (offset == target_offset) {
            *prev_ptr = entry->next_in_bucket;
            entry->in_bucket = false;
            entry->next_in_bucket = -1;
            return;
        }
        prev_ptr = &tm->entry_pool[offset].next_in_bucket;
        offset = *prev_ptr;
    }
}

void pto2_tensormapex_cleanup_retired(PTO2TensorMapEx* tm, 
                                       int32_t old_last_task_alive,
                                       int32_t new_last_task_alive) {
    for (int32_t task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
        int32_t task_slot = task_id & (PTO2_TASK_WINDOW_SIZE - 1);
        int32_t offset = tm->task_entry_head[task_slot];
        
        while (offset >= 0) {
            PTO2TensorMapEntryEx* entry = &tm->entry_pool[offset];
            if (entry->producer_task_id == task_id) {
                pto2_tensormapex_remove_from_bucket(tm, entry);
            }
            offset = entry->next_in_task;
        }
        
        tm->task_entry_head[task_slot] = -1;
    }
}

// -----------------------------------------------------------------------------
// Hash Function
// -----------------------------------------------------------------------------

uint32_t pto2_tensormapex_hash(PTO2TensorMapEx* tm, const PTO2LogicalTensor* tensor) {
    // Hash ONLY by raw_base for overlap detection
    // All tensors sharing the same storage must be in the same bucket
    uint64_t key = (uint64_t)(uintptr_t)tensor->raw_base;
    
    // Mix bits for better distribution
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    
    return (uint32_t)(key & (tm->num_buckets - 1));
}

// -----------------------------------------------------------------------------
// Overlap Detection
// -----------------------------------------------------------------------------

bool pto2_tensormapex_overlap(const PTO2LogicalTensor* tensor, const PTO2TensorMapEntryEx* entry) {
    // Use hybrid detection: fast bounding box for Simple tensors,
    // GCD-based exact check for Complex tensors
    return pto2_tensor_entry_overlap_hybrid(tensor, entry);
}

// -----------------------------------------------------------------------------
// Insert
// -----------------------------------------------------------------------------

void pto2_tensormapex_insert(PTO2TensorMapEx* tm, 
                              const PTO2LogicalTensor* tensor, 
                              int32_t producer_task_id) {
    // Allocate entry from ring buffer pool
    int32_t entry_offset = tm->pool_head;
    PTO2TensorMapEntryEx* entry = &tm->entry_pool[entry_offset];
    
    // Advance pool head (wrap around)
    tm->pool_head = (tm->pool_head + 1) % tm->pool_size;
    
    // Remove old entry from its bucket chain if needed
    if (entry->in_bucket) {
        pto2_tensormapex_remove_from_bucket(tm, entry);
    }
    
    // Initialize entry from LogicalTensor
    entry->raw_base = tensor->raw_base;
    entry->raw_total_size = tensor->raw_total_size;
    entry->min_byte_offset = tensor->min_byte_offset;
    entry->max_byte_offset = tensor->max_byte_offset;
    entry->storage_offset = tensor->storage_offset;
    
    // Copy shape and strides for GCD-based exact check (optional future use)
    for (int32_t d = 0; d < tensor->ndim && d < PTO2_MAX_TENSOR_DIM; d++) {
        entry->shape[d] = tensor->shape[d];
        entry->strides[d] = tensor->strides[d];
    }
    for (int32_t d = tensor->ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        entry->shape[d] = 0;
        entry->strides[d] = 0;
    }
    entry->ndim = tensor->ndim;
    
    entry->producer_task_id = producer_task_id;
    entry->is_deep_copy = (tensor->extraction_type >= PTO2_TENSOR_DEEP_VIEW);
    entry->is_simple = tensor->is_contiguous;  // Deprecated: use layout_depth == 1
    
    // Copy HBB layout history
    entry->layout_depth = tensor->layout_depth;
    for (int32_t i = 0; i < tensor->layout_depth && i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        entry->layout_ops[i] = tensor->layout_ops[i];
    }
    for (int32_t i = tensor->layout_depth; i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        entry->layout_ops[i].type = PTO2_LAYOUT_VIEW;
        entry->layout_ops[i].view.bbox_min = 0;
        entry->layout_ops[i].view.bbox_max = 0;
    }
    
    // Insert at head of hash bucket
    uint32_t bucket = pto2_tensormapex_hash(tm, tensor);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;
    
    // Link to task's entry list
    int32_t task_slot = producer_task_id & (PTO2_TASK_WINDOW_SIZE - 1);
    entry->next_in_task = tm->task_entry_head[task_slot];
    tm->task_entry_head[task_slot] = entry_offset;
}

// -----------------------------------------------------------------------------
// Lookup (single producer)
// -----------------------------------------------------------------------------

int32_t pto2_tensormapex_lookup(PTO2TensorMapEx* tm, const PTO2LogicalTensor* tensor) {
    uint32_t bucket = pto2_tensormapex_hash(tm, tensor);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    
    while (offset >= 0) {
        PTO2TensorMapEntryEx* entry = &tm->entry_pool[offset];
        
        // Check validity first
        if (!pto2_tensormapex_entry_valid(tm, entry)) {
            // Truncate stale tail
            *prev_ptr = -1;
            
            while (offset >= 0) {
                PTO2TensorMapEntryEx* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            
            return -1;  // Not found
        }
        
        // Check overlap using bounding box
        if (pto2_tensormapex_overlap(tensor, entry)) {
            return entry->producer_task_id;  // Found overlapping producer
        }
        
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return -1;  // Not found
}

// -----------------------------------------------------------------------------
// Lookup ALL overlapping producers
// -----------------------------------------------------------------------------

int32_t pto2_tensormapex_lookup_all(PTO2TensorMapEx* tm, 
                                     const PTO2LogicalTensor* tensor,
                                     int32_t* producer_ids,
                                     int32_t max_producers) {
    uint32_t bucket = pto2_tensormapex_hash(tm, tensor);
    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;
    int32_t count = 0;
    
    while (offset >= 0 && count < max_producers) {
        PTO2TensorMapEntryEx* entry = &tm->entry_pool[offset];
        
        // Check validity
        if (!pto2_tensormapex_entry_valid(tm, entry)) {
            // Truncate stale tail
            *prev_ptr = -1;
            
            while (offset >= 0) {
                PTO2TensorMapEntryEx* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            
            break;  // Done
        }
        
        // Check overlap
        if (pto2_tensormapex_overlap(tensor, entry)) {
            // Check for duplicates (same producer might have multiple outputs)
            bool duplicate = false;
            for (int32_t i = 0; i < count; i++) {
                if (producer_ids[i] == entry->producer_task_id) {
                    duplicate = true;
                    break;
                }
            }
            
            if (!duplicate) {
                producer_ids[count++] = entry->producer_task_id;
            }
        }
        
        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }
    
    return count;
}

// -----------------------------------------------------------------------------
// Debug Utilities
// -----------------------------------------------------------------------------

void pto2_tensormapex_print_stats(PTO2TensorMapEx* tm) {
    int32_t valid = 0;
    int32_t stale = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;
    
    // Count entries
    for (int32_t i = 0; i < tm->pool_size; i++) {
        if (tm->entry_pool[i].in_bucket) {
            if (pto2_tensormapex_entry_valid(tm, &tm->entry_pool[i])) {
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
    
    printf("=== Extended TensorMap Statistics ===\n");
    printf("Pool size:       %d\n", tm->pool_size);
    printf("Pool head:       %d\n", tm->pool_head);
    printf("Num buckets:     %d\n", tm->num_buckets);
    printf("Valid entries:   %d\n", valid);
    printf("Stale entries:   %d\n", stale);
    printf("Empty buckets:   %d\n", empty_buckets);
    printf("Max chain len:   %d\n", max_chain);
    printf("Avg chain len:   %.2f\n", 
           non_empty_buckets > 0 ? (float)total_chain / non_empty_buckets : 0);
    printf("Last task alive: %d\n", tm->last_task_alive);
    printf("=====================================\n");
}

int32_t pto2_tensormapex_valid_count(PTO2TensorMapEx* tm) {
    int32_t count = 0;
    
    for (int32_t i = 0; i < tm->pool_size; i++) {
        if (tm->entry_pool[i].in_bucket && 
            pto2_tensormapex_entry_valid(tm, &tm->entry_pool[i])) {
            count++;
        }
    }
    
    return count;
}
