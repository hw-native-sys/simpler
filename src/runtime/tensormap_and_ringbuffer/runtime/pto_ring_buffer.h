/**
 * PTO Runtime2 - Ring Buffer Data Structures
 * 
 * Implements ring buffer designs for zero-overhead memory management:
 * 
 * 1. HeapRing - Output buffer allocation from GM Heap
 *    - O(1) bump allocation
 *    - Wrap-around at end, skip to beginning if buffer doesn't fit
 *    - Implicit reclamation via heap_tail advancement
 *    - Back-pressure: stalls when no space available
 * 
 * 2. TaskRing - Task slot allocation
 *    - Fixed window size (TASK_WINDOW_SIZE)
 *    - Wrap-around modulo window size
 *    - Implicit reclamation via last_task_alive advancement
 *    - Back-pressure: stalls when window is full
 * 
 * 3. DepListPool - Dependency list entry allocation
 *    - Ring buffer for linked list entries
 *    - O(1) prepend operation
 *    - Implicit reclamation with task ring
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// =============================================================================
// Heap Ring Buffer
// =============================================================================

/**
 * Heap ring buffer structure
 * 
 * Allocates output buffers from a contiguous GM Heap.
 * Wrap-around design with implicit reclamation.
 */
struct PTO2HeapRing {
    void*    base;        // GM_Heap_Base pointer
    int32_t  size;        // GM_Heap_Size (total heap size in bytes)
    int32_t  top;         // Allocation pointer (local copy)
    
    // Reference to shared memory tail (for back-pressure)
    volatile int32_t* tail_ptr;  // Points to header->heap_tail
    
};

/**
 * Initialize heap ring buffer
 * 
 * @param ring      Heap ring to initialize
 * @param base      Base address of heap memory
 * @param size      Total heap size in bytes
 * @param tail_ptr  Pointer to shared memory heap_tail
 */
void pto2_heap_ring_init(PTO2HeapRing* ring, void* base, int32_t size,
                          volatile int32_t* tail_ptr);

/**
 * Allocate memory from heap ring
 * 
 * O(1) bump allocation with wrap-around.
 * May STALL (spin-wait) if insufficient space (back-pressure).
 * Never splits a buffer across the wrap-around boundary.
 * 
 * @param ring  Heap ring
 * @param size  Requested size in bytes
 * @return Pointer to allocated memory, never NULL (stalls instead)
 */
void* pto2_heap_ring_alloc(PTO2HeapRing* ring, int32_t size);

/**
 * Try to allocate memory without stalling
 * 
 * @param ring  Heap ring
 * @param size  Requested size in bytes
 * @return Pointer to allocated memory, or NULL if no space
 */
void* pto2_heap_ring_try_alloc(PTO2HeapRing* ring, int32_t size);

/**
 * Get available space in heap ring
 */
int32_t pto2_heap_ring_available(PTO2HeapRing* ring);

/**
 * Reset heap ring to initial state
 */
void pto2_heap_ring_reset(PTO2HeapRing* ring);

// =============================================================================
// Task Ring Buffer
// =============================================================================

/**
 * Task ring buffer structure
 * 
 * Fixed-size sliding window for task management.
 * Provides back-pressure when window is full.
 */
struct PTO2TaskRing {
    PTO2TaskDescriptor* descriptors;  // Task descriptor array (from shared memory)
    int32_t window_size;              // Window size (power of 2)
    int32_t current_index;            // Next task to allocate (absolute ID)
    
    // Reference to shared memory last_task_alive (for back-pressure)
    volatile int32_t* last_alive_ptr;  // Points to header->last_task_alive
    
};

/**
 * Initialize task ring buffer
 * 
 * @param ring            Task ring to initialize
 * @param descriptors     Task descriptor array from shared memory
 * @param window_size     Window size (must be power of 2)
 * @param last_alive_ptr  Pointer to shared memory last_task_alive
 */
void pto2_task_ring_init(PTO2TaskRing* ring, PTO2TaskDescriptor* descriptors,
                          int32_t window_size, volatile int32_t* last_alive_ptr);

/**
 * Allocate a task slot from task ring
 * 
 * May STALL (spin-wait) if window is full (back-pressure).
 * Initializes the task descriptor to default values.
 * 
 * @param ring  Task ring
 * @return Allocated task ID (absolute, not wrapped)
 */
int32_t pto2_task_ring_alloc(PTO2TaskRing* ring);

/**
 * Try to allocate task slot without stalling
 * 
 * @param ring  Task ring
 * @return Task ID, or -1 if window is full
 */
int32_t pto2_task_ring_try_alloc(PTO2TaskRing* ring);

/**
 * Get number of active tasks in window
 */
int32_t pto2_task_ring_active_count(PTO2TaskRing* ring);

/**
 * Check if task ring has space for more tasks
 */
bool pto2_task_ring_has_space(PTO2TaskRing* ring);

/**
 * Get task descriptor by ID
 */
static inline PTO2TaskDescriptor* pto2_task_ring_get(PTO2TaskRing* ring, int32_t task_id) {
    return &ring->descriptors[task_id & (ring->window_size - 1)];
}

/**
 * Reset task ring to initial state
 */
void pto2_task_ring_reset(PTO2TaskRing* ring);

// =============================================================================
// Dependency List Pool
// =============================================================================

/**
 * Dependency list pool structure
 * 
 * Ring buffer for allocating linked list entries.
 * Supports O(1) prepend operation for fanin/fanout lists.
 */
typedef struct {
    PTO2DepListEntry* base;   // Pool base address (from shared memory)
    int32_t capacity;         // Total number of entries
    int32_t top;              // Next allocation position (starts from 1, 0=NULL)
    
} PTO2DepListPool;

/**
 * Initialize dependency list pool
 * 
 * @param pool      Pool to initialize
 * @param base      Pool base address from shared memory
 * @param capacity  Total number of entries
 */
void pto2_dep_pool_init(PTO2DepListPool* pool, PTO2DepListEntry* base, int32_t capacity);

/**
 * Allocate a single entry from the pool
 * 
 * @param pool  Dependency list pool
 * @return Offset to allocated entry (0 means allocation failed)
 */
int32_t pto2_dep_pool_alloc_one(PTO2DepListPool* pool);

/**
 * Prepend a task ID to a dependency list
 * 
 * O(1) operation: allocates new entry and links to current head.
 * 
 * @param pool          Dependency list pool
 * @param current_head  Current list head offset (0 = empty list)
 * @param task_id       Task ID to prepend
 * @return New head offset
 */
int32_t pto2_dep_list_prepend(PTO2DepListPool* pool, int32_t current_head, int32_t task_id);

/**
 * Get entry by offset
 */
static inline PTO2DepListEntry* pto2_dep_pool_get(PTO2DepListPool* pool, int32_t offset) {
    if (offset <= 0) return NULL;
    return &pool->base[offset];
}

/**
 * Iterate through a dependency list
 * Calls callback for each task ID in the list.
 * 
 * @param pool      Dependency list pool
 * @param head      List head offset
 * @param callback  Function to call for each entry
 * @param ctx       User context passed to callback
 */
void pto2_dep_list_iterate(PTO2DepListPool* pool, int32_t head,
                            void (*callback)(int32_t task_id, void* ctx), void* ctx);

/**
 * Count entries in a dependency list
 */
int32_t pto2_dep_list_count(PTO2DepListPool* pool, int32_t head);

/**
 * Reset dependency list pool
 */
void pto2_dep_pool_reset(PTO2DepListPool* pool);

/**
 * Get pool usage statistics
 */
int32_t pto2_dep_pool_used(PTO2DepListPool* pool);
int32_t pto2_dep_pool_available(PTO2DepListPool* pool);

#endif // PTO_RING_BUFFER_H
