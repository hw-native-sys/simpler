/**
 * PTO Runtime2 - Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout:
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (flow control + sync)
 *   +---------------------------+
 *   | TaskDescriptor[]          |  (ring buffer)
 *   +---------------------------+
 *   | TaskPayload[]             |  (cold task data)
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_SHARED_MEMORY_H
#define PTO_SHARED_MEMORY_H

#include "pto_runtime2_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Shared Memory Header
// =============================================================================

/**
 * Shared memory header structure
 * 
 * Contains flow control pointers and layout information.
 * Written/read by Orchestrator and Scheduler for synchronization.
 */
typedef struct {
    // === FLOW CONTROL POINTERS ===

    // Written by Orchestrator, Read by Scheduler
    std::atomic<uint64_t> heap_top;           // Heap ring allocation pointer
    std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)
    std::atomic<int32_t> orchestrator_done;   // Flag: orchestration complete
    
    // Written by Scheduler, Read by Orchestrator (for back-pressure)
    std::atomic<uint64_t> heap_tail;          // Heap ring free pointer (on-device, matches pto2_heap_ring_init)
    std::atomic<int32_t> last_task_alive;     // Task ring tail (oldest active task)
    std::atomic<int32_t> heap_tail_gen;       // Ticket counter for serialized heap_tail writes
                                              // (ensures concurrent threads write in task order)

    // === LAYOUT INFO (set once at init) ===
    uint64_t task_window_size;            // PTO2_TASK_WINDOW_SIZE
    uint64_t heap_size;                   // Total heap size

    // Offsets into shared memory (relative to SM_Base)
    uint64_t task_descriptors_offset;     // Offset to TaskDescriptor array

    // Total shared memory size (for validation)
    uint64_t total_size;

    // Graph output for copy-back (set by orchestrator when using packed buffer)
    // Host finalize copies from this address instead of dev_ptr when non-zero
    std::atomic<uint64_t> graph_output_ptr;   // Address where final output was written (packed buffer)
    std::atomic<uint64_t> graph_output_size;  // Size in bytes

    // Padding to 128-byte cache line
    uint64_t _padding[4];

} PTO2SharedMemoryHeader;

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory access
 * Provides both Orchestrator and Scheduler views of the same memory
 */
typedef struct {
    void*   sm_base;              // Base address of shared memory
    uint64_t sm_size;             // Total size of shared memory

    // Quick pointers into shared memory regions
    PTO2SharedMemoryHeader* header;
    PTO2TaskDescriptor*     task_descriptors;
    PTO2TaskPayload*        task_payloads;
    
    // Ownership flag
    bool    is_owner;             // True if this handle allocated the memory
    
} PTO2SharedMemoryHandle;

// =============================================================================
// Shared Memory API
// =============================================================================

/**
 * Calculate required shared memory size
 *
 * @param task_window_size  Number of task slots
 * @return Total bytes required
 */
uint64_t pto2_sm_calculate_size(uint64_t task_window_size);

/**
 * Create shared memory for Orchestrator and Scheduler
 *
 * @param task_window_size  Number of task slots
 * @param heap_size         Heap size for output buffers
 * @return Handle with both views, or NULL on failure
 */
PTO2SharedMemoryHandle* pto2_sm_create(uint64_t task_window_size,
                                        uint64_t heap_size);

/**
 * Create shared memory with default sizes
 */
PTO2SharedMemoryHandle* pto2_sm_create_default(void);

/**
 * Wrap an existing buffer as shared memory (e.g. device GM buffer).
 * Caller owns the buffer; handle will not free sm_base.
 *
 * @param sm_base            Base address of pre-allocated buffer
 * @param sm_size            Total size in bytes
 * @param task_window_size   Number of task slots (must match buffer layout)
 * @param heap_size          Heap size (for layout; buffer has no heap region)
 * @return Handle, or NULL on failure
 */
PTO2SharedMemoryHandle* pto2_sm_create_from_buffer(void* sm_base,
                                                    uint64_t sm_size,
                                                    uint64_t task_window_size,
                                                    uint64_t heap_size);

/**
 * Destroy shared memory and free resources
 */
void pto2_sm_destroy(PTO2SharedMemoryHandle* handle);

/**
 * Initialize shared memory header with layout information
 * Called after memory is allocated
 */
void pto2_sm_init_header(PTO2SharedMemoryHandle* handle,
                          uint64_t task_window_size,
                          uint64_t heap_size);

/**
 * Get task descriptor by task ID
 * Uses runtime window_size for ring buffer indexing (not compile-time constant)
 */
static inline PTO2TaskDescriptor* pto2_sm_get_task(PTO2SharedMemoryHandle* handle,
                                                    int32_t task_id) {
    uint64_t window_mask = handle->header->task_window_size - 1;
    return &handle->task_descriptors[task_id & window_mask];
}

/**
 * Get task descriptor by task slot
 * Uses runtime window_size for ring buffer indexing (not compile-time constant)
 */
static inline PTO2TaskDescriptor& pto2_sm_get_task_by_slot(PTO2SharedMemoryHandle* handle,
                                                    int32_t slot) {
    return handle->task_descriptors[slot];
}

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print shared memory layout info
 */
void pto2_sm_print_layout(PTO2SharedMemoryHandle* handle);

/**
 * Validate shared memory integrity
 * @return true if valid, false if corrupted
 */
bool pto2_sm_validate(PTO2SharedMemoryHandle* handle);

#ifdef __cplusplus
}
#endif

#endif // PTO_SHARED_MEMORY_H
