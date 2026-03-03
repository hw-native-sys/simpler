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

#include <inttypes.h>
#include <stdlib.h>  // for exit()

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"

// Set to 1 to enable periodic BLOCKED/Unblocked messages during spin-wait.
#ifndef PTO2_SPIN_VERBOSE_LOGGING
#define PTO2_SPIN_VERBOSE_LOGGING 1
#endif

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL  10000
// Heap ring spin limit - after this, report deadlock and exit
#define PTO2_HEAP_SPIN_LIMIT        100000

// Flow control spin limit - if exceeded, likely deadlock due to scope/fanout_count
#define PTO2_FLOW_CONTROL_SPIN_LIMIT  100000

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
    uint64_t size;        // GM_Heap_Size (total heap size in bytes)
    uint64_t top;         // Allocation pointer (local copy)

    // Reference to shared memory tail (for back-pressure)
    std::atomic<uint64_t>* tail_ptr;  // Points to header->heap_tail

    /**
     * Allocate memory from heap ring
     *
     * O(1) bump allocation with wrap-around.
     * May STALL (spin-wait) if insufficient space (back-pressure).
     * Never splits a buffer across the wrap-around boundary.
     *
     * @param size  Requested size in bytes
     * @return Pointer to allocated memory, never NULL (stalls instead)
     */
    void* pto2_heap_ring_alloc(uint64_t size) {
        // Align size for DMA efficiency
        size = PTO2_ALIGN_UP(size, PTO2_ALIGN_SIZE);

        // Spin-wait if insufficient space (back-pressure from Scheduler)
        int spin_count = 0;
#if PTO2_SPIN_VERBOSE_LOGGING
        bool notified = false;
#endif

        while (1) {
            void* ptr = pto2_heap_ring_try_alloc(size);
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
            if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 && spin_count < PTO2_HEAP_SPIN_LIMIT) {
                uint64_t tail = tail_ptr->load(std::memory_order_acquire);
                uint64_t available = pto2_heap_ring_available();
                LOG_WARN("[HeapRing] BLOCKED: requesting %" PRIu64 " bytes, available=%" PRIu64
                     ", top=%" PRIu64 ", tail=%" PRIu64 ", spins=%d",
                     size, available, top, tail, spin_count);
                notified = true;
            }
#endif

            if (spin_count >= PTO2_HEAP_SPIN_LIMIT) {
                uint64_t tail = tail_ptr->load(std::memory_order_acquire);
                uint64_t available = pto2_heap_ring_available();
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: Heap Ring Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("Orchestrator blocked waiting for heap space after %d spins.", spin_count);
                LOG_ERROR("  - Requested:     %" PRIu64 " bytes", size);
                LOG_ERROR("  - Available:     %" PRIu64 " bytes", available);
                LOG_ERROR("  - Heap top:      %" PRIu64, top);
                LOG_ERROR("  - Heap tail:     %" PRIu64, tail);
                LOG_ERROR("  - Heap size:     %" PRIu64, this->size);
                LOG_ERROR("Solution: Increase PTO2_HEAP_SIZE (e.g. 256*1024 for 4 x 64KB outputs).");
                LOG_ERROR("========================================");
                exit(1);
            }

            PTO2_SPIN_PAUSE();
        }
    }

    /**
     * Try to allocate memory without stalling
     *
     * @param size  Requested size in bytes
     * @return Pointer to allocated memory, or NULL if no space
     */
    void* pto2_heap_ring_try_alloc(uint64_t alloc_size) {
        // Align size for DMA efficiency
        alloc_size = PTO2_ALIGN_UP(alloc_size, PTO2_ALIGN_SIZE);

        // Read latest tail from shared memory (Scheduler updates this)
        uint64_t tail = tail_ptr->load(std::memory_order_acquire);

        if (top >= tail) {
            // Case 1: top is at or ahead of tail (normal case)
            //   [....tail====top......]
            //                   ^-- space_at_end = size - top

            uint64_t space_at_end = size - top;

            if (space_at_end >= alloc_size) {
                // Enough space at end - allocate here
                void* ptr = (char*)base + top;
                top += alloc_size;
                return ptr;
            }

            // Not enough space at end - check if we can wrap to beginning
            // IMPORTANT: Don't split buffer, skip remaining space at end
            if (tail > alloc_size) {
                // Wrap to beginning (space available: [0, tail))
                top = alloc_size;
                return base;
            }

            // Not enough space anywhere - return NULL
            return NULL;

        } else {
            // Case 2: top has wrapped, tail is ahead
            //   [====top....tail=====]
            //         ^-- free space = tail - top

            uint64_t gap = tail - top;
            if (gap >= alloc_size) {
                void* ptr = (char*)base + top;
                top += alloc_size;
                return ptr;
            }

            // Not enough space - return NULL
            return NULL;
        }
    }

    /**
     * Get available space in heap ring
     */
    uint64_t pto2_heap_ring_available() {
        uint64_t tail = tail_ptr->load(std::memory_order_acquire);

        if (top >= tail) {
            // Space at end + space at beginning (if any)
            uint64_t at_end = size - top;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;  // Max usable
        } else {
            // Contiguous space between top and tail
            return tail - top;
        }
    }
};

/**
 * Initialize heap ring buffer
 * 
 * @param ring      Heap ring to initialize
 * @param base      Base address of heap memory
 * @param size      Total heap size in bytes
 * @param tail_ptr  Pointer to shared memory heap_tail
 */
void pto2_heap_ring_init(PTO2HeapRing* ring, void* base, uint64_t size,
                          std::atomic<uint64_t>* tail_ptr);

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
    std::atomic<int32_t>* last_alive_ptr;  // Points to header->last_task_alive

    /**
     * Allocate a task slot from task ring
     *
     * May STALL (spin-wait) if window is full (back-pressure).
     * Initializes the task descriptor to default values.
     *
     * @return Allocated task ID (absolute, not wrapped)
     */
    int32_t pto2_task_ring_alloc() {
        // Spin-wait if window is full (back-pressure from Scheduler)
        int spin_count = 0;
#if PTO2_SPIN_VERBOSE_LOGGING
        bool notified = false;
#endif

        while (1) {
            int32_t task_id = pto2_task_ring_try_alloc();
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
            if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 && spin_count < PTO2_FLOW_CONTROL_SPIN_LIMIT) {
                int32_t last_alive = last_alive_ptr->load(std::memory_order_acquire);
                int32_t active_count = current_index - last_alive;
                LOG_WARN("[TaskRing] BLOCKED (Flow Control): current=%d, last_alive=%d, "
                     "active=%d/%d (%.1f%%), spins=%d",
                     current_index, last_alive, active_count, window_size,
                     100.0 * active_count / window_size, spin_count);
                notified = true;
            }
#endif

            // Check for potential deadlock
            if (spin_count >= PTO2_FLOW_CONTROL_SPIN_LIMIT) {
                int32_t last_alive = last_alive_ptr->load(std::memory_order_acquire);
                int32_t active_count = current_index - last_alive;

                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: Flow Control Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("Task Ring is FULL and no progress after %d spins.", spin_count);
                LOG_ERROR("Flow Control Status:");
                LOG_ERROR("  - Current task index:  %d", current_index);
                LOG_ERROR("  - Last task alive:     %d", last_alive);
                LOG_ERROR("  - Active tasks:        %d", active_count);
                LOG_ERROR("  - Window size:         %d", window_size);
                LOG_ERROR("  - Window utilization:  %.1f%%", 100.0 * active_count / window_size);
                LOG_ERROR("Root Cause:");
                LOG_ERROR("  Tasks cannot transition to CONSUMED state because:");
                LOG_ERROR("  - fanout_count includes 1 for the owning scope");
                LOG_ERROR("  - scope_end() requires orchestrator to continue");
                LOG_ERROR("  - But orchestrator is blocked waiting for task ring space");
                LOG_ERROR("  This creates a circular dependency (deadlock).");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Current task_window_size: %d", window_size);
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

    /**
     * Try to allocate task slot without stalling
     *
     * @return Task ID, or -1 if window is full
     */
    int32_t pto2_task_ring_try_alloc() {
        // Read latest last_task_alive from shared memory
        int32_t last_alive = last_alive_ptr->load(std::memory_order_acquire);
        int32_t current = current_index;

        // Calculate number of active tasks (handles wrap-around)
        int32_t active_count = current - last_alive;

        // Check if there's room for one more task
        // Leave at least 1 slot empty to distinguish full from empty
        if (active_count < window_size - 1) {
            int32_t task_id = current;
            int32_t slot = task_id & (window_size - 1);

            // Mark slot as occupied (skip full memset — pto2_submit_task
            // explicitly initializes all fields it needs)
            PTO2TaskDescriptor* task = &descriptors[slot];
            task->task_id = task_id;
            task->is_active = true;

            // Advance current index
            current_index = current + 1;

            return task_id;
        }

        // Window is full
        return -1;
    }
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
                          int32_t window_size, std::atomic<int32_t>* last_alive_ptr);

/**
 * Get number of active tasks in window
 */
static inline int32_t pto2_task_ring_active_count(PTO2TaskRing* ring) {
    int32_t last_alive = ring->last_alive_ptr->load(std::memory_order_acquire);
    return ring->current_index - last_alive;
}

/**
 * Check if task ring has space for more tasks
 */
static inline bool pto2_task_ring_has_space(PTO2TaskRing* ring) {
    int32_t active = pto2_task_ring_active_count(ring);
    return active < ring->window_size - 1;
}

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
struct PTO2DepListPool {
    PTO2DepListEntry* base;   // Pool base address (from shared memory)
    int32_t capacity;         // Total number of entries
    int32_t top;              // Next allocation position (starts from 1, 0=NULL)

    /**
     * Allocate a single entry from the pool
     *
     * @return Offset to allocated entry (0 means allocation failed)
     */
    int32_t alloc() {
        if (top >= capacity) {
            top = 1;
        }
        return top++;
    }

    /**
     * Prepend a task ID to a dependency list
     *
     * O(1) operation: allocates new entry and links to current head.
     *
     * @param current_head  Current list head offset (0 = empty list)
     * @param task_id       Task ID to prepend
     * @return New head offset
     */
    PTO2DepListEntry* pto2_dep_list_prepend(PTO2DepListEntry* cur, int32_t task_id) {
        int32_t new_index = alloc();
        PTO2DepListEntry& new_entry = base[new_index];
        new_entry.task_id = task_id;
        new_entry.next = cur;
        return &new_entry;
    }

    /**
    * Get entry by offset
    */
    PTO2DepListEntry* pto2_dep_pool_get(int32_t offset) {
        if (offset <= 0) return NULL;
        return &base[offset];
    }
};

/**
 * Initialize dependency list pool
 * 
 * @param pool      Pool to initialize
 * @param base      Pool base address from shared memory
 * @param capacity  Total number of entries
 */
void pto2_dep_pool_init(PTO2DepListPool* pool, PTO2DepListEntry* base, int32_t capacity);

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
