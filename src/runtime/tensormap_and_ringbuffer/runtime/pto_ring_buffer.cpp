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

// Set to 1 to enable periodic BLOCKED/Unblocked messages during spin-wait.
#ifndef PTO2_SPIN_VERBOSE_LOGGING
#define PTO2_SPIN_VERBOSE_LOGGING 1
#endif

// =============================================================================
// Heap Ring Buffer Implementation
// =============================================================================

void pto2_heap_ring_init(PTO2HeapRing* ring, void* base, uint64_t size,
                          volatile uint64_t* tail_ptr) {
    ring->base = base;
    ring->size = size;
    ring->top = 0;
    ring->tail_ptr = tail_ptr;
}

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL  10000
// Heap ring spin limit - after this, report deadlock and exit
#define PTO2_HEAP_SPIN_LIMIT        100000

void* pto2_heap_ring_alloc(PTO2HeapRing* ring, uint64_t size) {
    // Align size for DMA efficiency
    size = PTO2_ALIGN_UP(size, PTO2_ALIGN_SIZE);

    // Spin-wait if insufficient space (back-pressure from Scheduler)
    int spin_count = 0;
#if PTO2_SPIN_VERBOSE_LOGGING
    bool notified = false;
#endif

    while (1) {
        void* ptr = pto2_heap_ring_try_alloc(ring, size);
        if (ptr != NULL) {
#if PTO2_SPIN_VERBOSE_LOGGING
            if (notified) {
                LOG_INFO("[HeapRing] Unblocked after %d spins", spin_count);
            }
#endif
            return ptr;
        }

        // No space available, spin-wait
        spin_count++;

#if PTO2_SPIN_VERBOSE_LOGGING
        // Periodic block notification
        if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 &&
            spin_count < PTO2_HEAP_SPIN_LIMIT) {
            uint64_t tail = PTO2_LOAD_ACQUIRE(ring->tail_ptr);
            uint64_t available = pto2_heap_ring_available(ring);
            LOG_WARN("[HeapRing] BLOCKED: requesting %" PRIu64 " bytes, available=%" PRIu64 ", "
                     "top=%" PRIu64 ", tail=%" PRIu64 ", spins=%d",
                     size, available, ring->top, tail, spin_count);
            notified = true;
        }
#endif

        if (spin_count >= PTO2_HEAP_SPIN_LIMIT) {
            uint64_t tail = PTO2_LOAD_ACQUIRE(ring->tail_ptr);
            uint64_t available = pto2_heap_ring_available(ring);
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Heap Ring Deadlock Detected!");
            LOG_ERROR("========================================");
            LOG_ERROR("Orchestrator blocked waiting for heap space after %d spins.", spin_count);
            LOG_ERROR("  - Requested:     %" PRIu64 " bytes", size);
            LOG_ERROR("  - Available:     %" PRIu64 " bytes", available);
            LOG_ERROR("  - Heap top:      %" PRIu64, ring->top);
            LOG_ERROR("  - Heap tail:     %" PRIu64, tail);
            LOG_ERROR("  - Heap size:     %" PRIu64, ring->size);
            LOG_ERROR("Solution: Increase PTO2_HEAP_SIZE (e.g. 256*1024 for 4 x 64KB outputs).");
            LOG_ERROR("========================================");
            exit(1);
        }

        PTO2_SPIN_PAUSE();
    }
}

void* pto2_heap_ring_try_alloc(PTO2HeapRing* ring, uint64_t size) {
    // Align size for DMA efficiency
    size = PTO2_ALIGN_UP(size, PTO2_ALIGN_SIZE);

    // Read latest tail from shared memory (Scheduler updates this)
    uint64_t tail = PTO2_LOAD_ACQUIRE(ring->tail_ptr);
    uint64_t top = ring->top;

    if (top >= tail) {
        // Case 1: top is at or ahead of tail (normal case)
        //   [....tail====top......]
        //                   ^-- space_at_end = size - top

        uint64_t space_at_end = ring->size - top;

        if (space_at_end >= size) {
            // Enough space at end - allocate here
            void* ptr = (char*)ring->base + top;
            ring->top = top + size;
            return ptr;
        }

        // Not enough space at end - check if we can wrap to beginning
        // IMPORTANT: Don't split buffer, skip remaining space at end
        if (tail > size) {
            // Wrap to beginning (space available: [0, tail))
            ring->top = size;
            return ring->base;
        }

        // Not enough space anywhere - return NULL
        return NULL;

    } else {
        // Case 2: top has wrapped, tail is ahead
        //   [====top....tail=====]
        //         ^-- free space = tail - top

        uint64_t gap = tail - top;
        if (gap >= size) {
            void* ptr = (char*)ring->base + top;
            ring->top = top + size;
            return ptr;
        }

        // Not enough space - return NULL
        return NULL;
    }
}

uint64_t pto2_heap_ring_available(PTO2HeapRing* ring) {
    uint64_t tail = PTO2_LOAD_ACQUIRE(ring->tail_ptr);
    uint64_t top = ring->top;

    if (top >= tail) {
        // Space at end + space at beginning (if any)
        uint64_t at_end = ring->size - top;
        uint64_t at_begin = tail;
        return at_end > at_begin ? at_end : at_begin;  // Max usable
    } else {
        // Contiguous space between top and tail
        return tail - top;
    }
}

void pto2_heap_ring_reset(PTO2HeapRing* ring) {
    ring->top = 0;
}

// =============================================================================
// Task Ring Buffer Implementation
// =============================================================================

void pto2_task_ring_init(PTO2TaskRing* ring, PTO2TaskDescriptor* descriptors,
                          int32_t window_size, volatile int32_t* last_alive_ptr) {
    ring->descriptors = descriptors;
    ring->window_size = window_size;
    ring->current_index = 0;
    ring->last_alive_ptr = last_alive_ptr;
}

// Flow control spin limit - if exceeded, likely deadlock due to scope/fanout_count
#define PTO2_FLOW_CONTROL_SPIN_LIMIT  100000

int32_t pto2_task_ring_alloc(PTO2TaskRing* ring) {
    // Spin-wait if window is full (back-pressure from Scheduler)
    int spin_count = 0;
#if PTO2_SPIN_VERBOSE_LOGGING
    bool notified = false;
#endif
    
    while (1) {
        int32_t task_id = pto2_task_ring_try_alloc(ring);
        if (task_id >= 0) {
#if PTO2_SPIN_VERBOSE_LOGGING
            if (notified) {
                LOG_INFO("[TaskRing] Unblocked after %d spins, task_id=%d", spin_count, task_id);
            }
#endif
            return task_id;
        }
        
        // Window is full, spin-wait (with yield to prevent CPU starvation)
        spin_count++;
        
#if PTO2_SPIN_VERBOSE_LOGGING
        // Periodic block notification
        if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 && 
            spin_count < PTO2_FLOW_CONTROL_SPIN_LIMIT) {
            int32_t last_alive = PTO2_LOAD_ACQUIRE(ring->last_alive_ptr);
            int32_t active_count = ring->current_index - last_alive;
            LOG_WARN("[TaskRing] BLOCKED (Flow Control): current=%d, last_alive=%d, "
                     "active=%d/%d (%.1f%%), spins=%d",
                     ring->current_index, last_alive, active_count, ring->window_size,
                     100.0 * active_count / ring->window_size, spin_count);
            notified = true;
        }
#endif
        
        // Check for potential deadlock
        if (spin_count >= PTO2_FLOW_CONTROL_SPIN_LIMIT) {
            int32_t last_alive = PTO2_LOAD_ACQUIRE(ring->last_alive_ptr);
            int32_t active_count = ring->current_index - last_alive;
            
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Flow Control Deadlock Detected!");
            LOG_ERROR("========================================");
            LOG_ERROR("Task Ring is FULL and no progress after %d spins.", spin_count);
            LOG_ERROR("Flow Control Status:");
            LOG_ERROR("  - Current task index:  %d", ring->current_index);
            LOG_ERROR("  - Last task alive:     %d", last_alive);
            LOG_ERROR("  - Active tasks:        %d", active_count);
            LOG_ERROR("  - Window size:         %d", ring->window_size);
            LOG_ERROR("  - Window utilization:  %.1f%%", 100.0 * active_count / ring->window_size);
            LOG_ERROR("Root Cause:");
            LOG_ERROR("  Tasks cannot transition to CONSUMED state because:");
            LOG_ERROR("  - fanout_count includes 1 for the owning scope");
            LOG_ERROR("  - scope_end() requires orchestrator to continue");
            LOG_ERROR("  - But orchestrator is blocked waiting for task ring space");
            LOG_ERROR("  This creates a circular dependency (deadlock).");
            LOG_ERROR("Solution:");
            LOG_ERROR("  Current task_window_size: %d", ring->window_size);
            LOG_ERROR("  Default PTO2_TASK_WINDOW_SIZE: %d", PTO2_TASK_WINDOW_SIZE);
            LOG_ERROR("  Recommended: %d (at least 2x current active tasks)", active_count * 2);
            LOG_ERROR("  Option 1: Change PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Option 2: Use pto2_runtime_create_threaded_custom() with larger");
            LOG_ERROR("            task_window_size parameter.");
            LOG_ERROR("========================================");
            
            // Abort program
            exit(1);
        }
        
        PTO2_SPIN_PAUSE();
    }
}

int32_t pto2_task_ring_try_alloc(PTO2TaskRing* ring) {
    // Read latest last_task_alive from shared memory
    int32_t last_alive = PTO2_LOAD_ACQUIRE(ring->last_alive_ptr);
    int32_t current = ring->current_index;
    
    // Calculate number of active tasks (handles wrap-around)
    int32_t active_count = current - last_alive;
    
    // Check if there's room for one more task
    // Leave at least 1 slot empty to distinguish full from empty
    if (active_count < ring->window_size - 1) {
        int32_t task_id = current;
        int32_t slot = task_id & (ring->window_size - 1);
        
        // Mark slot as occupied (skip full memset — pto2_submit_task
        // explicitly initializes all fields it needs)
        PTO2TaskDescriptor* task = &ring->descriptors[slot];
        task->task_id = task_id;
        task->is_active = true;
        
        // Advance current index
        ring->current_index = current + 1;
        
        return task_id;
    }
    
    // Window is full
    return -1;
}

int32_t pto2_task_ring_active_count(PTO2TaskRing* ring) {
    int32_t last_alive = PTO2_LOAD_ACQUIRE(ring->last_alive_ptr);
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
