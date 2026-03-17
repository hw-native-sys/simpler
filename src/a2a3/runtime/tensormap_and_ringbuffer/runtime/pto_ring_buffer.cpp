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
#include "pto_scheduler.h"

// =============================================================================
// Heap Ring Buffer Implementation
// =============================================================================

void pto2_heap_ring_init(PTO2HeapRing* ring, void* base, uint64_t size,
                          std::atomic<uint64_t>* tail_ptr,
                          std::atomic<uint64_t>* top_ptr) {
    ring->base = base;
    ring->size = size;
    ring->top_ptr = top_ptr;
    ring->tail_ptr = tail_ptr;
}

// =============================================================================
// Task Ring Buffer Implementation
// =============================================================================

void pto2_task_ring_init(PTO2TaskRing* ring, PTO2TaskDescriptor* descriptors,
                          int32_t window_size, std::atomic<int32_t>* last_alive_ptr,
                          std::atomic<int32_t>* current_index_ptr) {
    ring->descriptors = descriptors;
    ring->window_size = window_size;
    ring->current_index_ptr = current_index_ptr;
    ring->last_alive_ptr = last_alive_ptr;
}

// =============================================================================
// Dependency List Pool Implementation
// =============================================================================

void pto2_dep_pool_init(PTO2DepListPool* pool, PTO2DepListEntry* base, int32_t capacity) {
    pool->base = base;
    pool->capacity = capacity;
    pool->top = 1;  // Start from 1, 0 means NULL/empty
    pool->tail = 1; // Match initial top (no reclaimable entries yet)
    pool->high_water = 0;
    pool->last_reclaimed = 0;

    // Initialize entry 0 as NULL marker
    pool->base[0].slot_state = nullptr;
    pool->base[0].next = nullptr;
}

void PTO2DepListPool::reclaim(PTO2SchedulerState* sched, uint8_t ring_id, int32_t sm_last_task_alive) {
    if (sm_last_task_alive >= last_reclaimed + PTO2_DEP_POOL_CLEANUP_INTERVAL && sm_last_task_alive > 0) {
        int32_t mark = sched->ring_sched_states[ring_id].get_slot_state_by_task_id(sm_last_task_alive - 1).dep_pool_mark;
        if (mark > 0) {
            advance_tail(mark);
        }
        last_reclaimed = sm_last_task_alive;
    }
}

int32_t pto2_dep_pool_used(PTO2DepListPool* pool) {
    return pool->top - pool->tail;
}

int32_t pto2_dep_pool_available(PTO2DepListPool* pool) {
    return pool->capacity - (pool->top - pool->tail);
}
