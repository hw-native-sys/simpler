/**
 * PTO Runtime2 - Ring Buffer Implementation
 * 
 * Implements HeapRing, TaskRing, and DepListPool ring buffers
 * for zero-overhead memory management.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_ring_buffer.h"
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>  // for exit()
#include "common/unified_log.h"

// =============================================================================
// Heap Ring Buffer Implementation
// =============================================================================

void pto2_heap_ring_init(PTO2HeapRing* ring, void* base, uint64_t size,
                          std::atomic<uint64_t>* tail_ptr) {
    ring->base = base;
    ring->size = size;
    ring->top = 0;
    ring->tail_ptr = tail_ptr;
}

void pto2_heap_ring_reset(PTO2HeapRing* ring) {
    ring->top = 0;
}

// =============================================================================
// Task Ring Buffer Implementation
// =============================================================================

void pto2_task_ring_init(PTO2TaskRing* ring, PTO2TaskDescriptor* descriptors,
                          int32_t window_size, std::atomic<int32_t>* last_alive_ptr) {
    ring->descriptors = descriptors;
    ring->window_size = window_size;
    ring->current_index = 0;
    ring->last_alive_ptr = last_alive_ptr;
}

int32_t pto2_task_ring_active_count(PTO2TaskRing* ring) {
    int32_t last_alive = ring->last_alive_ptr->load(std::memory_order_acquire);
    return ring->current_index - last_alive;
}

bool pto2_task_ring_has_space(PTO2TaskRing* ring) {
    int32_t active = pto2_task_ring_active_count(ring);
    return active < ring->window_size - 1;
}

void pto2_task_ring_reset(PTO2TaskRing* ring) {
    ring->current_index = 0;
    
    // Clear all task descriptors
    memset(ring->descriptors, 0, ring->window_size * sizeof(PTO2TaskDescriptor));
}

// =============================================================================
// Dependency List Pool Implementation
// =============================================================================

void pto2_dep_pool_init(PTO2DepListPool* pool, PTO2DepListEntry* base, int32_t capacity) {
    pool->base = base;
    pool->capacity = capacity;
    pool->top = 1;  // Start from 1, 0 means NULL/empty
    
    // Initialize entry 0 as NULL marker
    pool->base[0].task_id = -1;
    pool->base[0].next_offset = 0;
}

int32_t pto2_dep_pool_alloc_one(PTO2DepListPool* pool) {
    if (pool->top >= pool->capacity) {
        // Wrap around to beginning (old entries reclaimed with task ring)
        pool->top = 1;  // Start from 1, 0 means NULL
    }
    return pool->top++;
}

int32_t pto2_dep_list_prepend(PTO2DepListPool* pool, int32_t current_head, int32_t task_id) {
    // Allocate new entry
    int32_t new_offset = pto2_dep_pool_alloc_one(pool);
    if (new_offset <= 0) {
        return current_head;  // Allocation failed, return unchanged
    }
    
    PTO2DepListEntry* new_entry = &pool->base[new_offset];
    
    // Fill in new entry: points to old head
    new_entry->task_id = task_id;
    new_entry->next_offset = current_head;  // Link to previous head
    
    return new_offset;  // New head
}

void pto2_dep_list_iterate(PTO2DepListPool* pool, int32_t head,
                            void (*callback)(int32_t task_id, void* ctx), void* ctx) {
    int32_t current = head;
    
    while (current > 0 && current < pool->capacity) {
        PTO2DepListEntry* entry = &pool->base[current];
        callback(entry->task_id, ctx);
        current = entry->next_offset;
    }
}

int32_t pto2_dep_list_count(PTO2DepListPool* pool, int32_t head) {
    int32_t count = 0;
    int32_t current = head;
    
    while (current > 0 && current < pool->capacity) {
        count++;
        current = pool->base[current].next_offset;
    }
    
    return count;
}

void pto2_dep_pool_reset(PTO2DepListPool* pool) {
    pool->top = 1;
    
    // Clear pool (optional, for debugging)
    memset(pool->base + 1, 0, (pool->capacity - 1) * sizeof(PTO2DepListEntry));
    
    // Re-initialize entry 0 as NULL marker
    pool->base[0].task_id = -1;
    pool->base[0].next_offset = 0;
}

int32_t pto2_dep_pool_used(PTO2DepListPool* pool) {
    return pool->top - 1;  // Exclude entry 0 (NULL marker)
}

int32_t pto2_dep_pool_available(PTO2DepListPool* pool) {
    return pool->capacity - pool->top;
}
