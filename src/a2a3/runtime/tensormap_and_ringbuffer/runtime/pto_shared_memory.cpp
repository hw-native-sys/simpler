/**
 * PTO Runtime2 - Shared Memory Implementation
 * 
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t pto2_sm_calculate_size(uint64_t task_window_size, uint64_t dep_list_pool_size) {
    uint64_t size = 0;
    
    // Header (aligned to cache line)
    size += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    
    // Task descriptors
    size += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    
    // Dependency list pool (entry 0 is reserved as NULL)
    size += PTO2_ALIGN_UP((dep_list_pool_size + 1) * sizeof(PTO2DepListEntry), PTO2_ALIGN_SIZE);
    
    return size;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

PTO2SharedMemoryHandle* pto2_sm_create(uint64_t task_window_size,
                                        uint64_t heap_size,
                                        uint64_t dep_list_pool_size) {
    // Allocate handle
    PTO2SharedMemoryHandle* handle = (PTO2SharedMemoryHandle*)calloc(1, sizeof(PTO2SharedMemoryHandle));
    if (!handle) {
        return NULL;
    }
    
    // Calculate total size
    uint64_t sm_size = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
    
    // Allocate shared memory (aligned for DMA efficiency)
    #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign(&handle->sm_base, PTO2_ALIGN_SIZE, static_cast<size_t>(sm_size)) != 0) {
            free(handle);
            return NULL;
        }
    #else
        handle->sm_base = aligned_alloc(PTO2_ALIGN_SIZE, static_cast<size_t>(sm_size));
        if (!handle->sm_base) {
            free(handle);
            return NULL;
        }
    #endif
    
    handle->sm_size = sm_size;
    handle->is_owner = true;
    
    // Initialize to zero
    memset(handle->sm_base, 0, static_cast<size_t>(sm_size));
    
    // Set up pointers
    char* ptr = (char*)handle->sm_base;
    
    // Header
    handle->header = (PTO2SharedMemoryHeader*)ptr;
    ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    
    // Task descriptors
    handle->task_descriptors = (PTO2TaskDescriptor*)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    
    // Dependency list pool
    handle->dep_list_pool = (PTO2DepListEntry*)ptr;
    
    // Initialize header
    pto2_sm_init_header(handle, task_window_size, heap_size, dep_list_pool_size);
    
    return handle;
}

PTO2SharedMemoryHandle* pto2_sm_create_default(void) {
    return pto2_sm_create(PTO2_TASK_WINDOW_SIZE,
                          PTO2_HEAP_SIZE,
                          PTO2_DEP_LIST_POOL_SIZE);
}

PTO2SharedMemoryHandle* pto2_sm_create_from_buffer(void* sm_base,
                                                    uint64_t sm_size,
                                                    uint64_t task_window_size,
                                                    uint64_t heap_size,
                                                    uint64_t dep_list_pool_size) {
    if (!sm_base || sm_size == 0) return NULL;

    uint64_t required = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
    if (sm_size < required) return NULL;

    PTO2SharedMemoryHandle* handle = (PTO2SharedMemoryHandle*)calloc(1, sizeof(PTO2SharedMemoryHandle));
    if (!handle) return NULL;

    handle->sm_base = sm_base;
    handle->sm_size = sm_size;
    handle->is_owner = false;

    char* ptr = (char*)sm_base;
    handle->header = (PTO2SharedMemoryHeader*)ptr;
    ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    handle->task_descriptors = (PTO2TaskDescriptor*)ptr;
    ptr += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    handle->dep_list_pool = (PTO2DepListEntry*)ptr;

    pto2_sm_init_header(handle, task_window_size, heap_size, dep_list_pool_size);

    // Zero task descriptors and dep list pool.
    // On multi-round execution the SM buffer may be reused at the same device
    // address.  Stale fanout_head/fanin_head pointers and fanout_lock atomics
    // from a previous round would corrupt the scheduler's dependency tracking.
    // Zeroing ensures clean state for every slot before new tasks are written.
    memset(handle->task_descriptors, 0,
           task_window_size * sizeof(PTO2TaskDescriptor));
    // Skip slot 0 (sentinel: task_id=-1, next=nullptr; deref(0) is a no-op)
    memset(handle->dep_list_pool + 1, 0,
           dep_list_pool_size * sizeof(PTO2DepListEntry));

    return handle;
}

void pto2_sm_destroy(PTO2SharedMemoryHandle* handle) {
    if (!handle) return;
    
    if (handle->is_owner && handle->sm_base) {
        free(handle->sm_base);
    }
    
    free(handle);
}

// =============================================================================
// Initialization
// =============================================================================

void pto2_sm_init_header(PTO2SharedMemoryHandle* handle,
                          uint64_t task_window_size,
                          uint64_t heap_size,
                          uint64_t dep_list_pool_size) {
    PTO2SharedMemoryHeader* header = handle->header;
    
    // Flow control pointers (start at 0)
    header->current_task_index.store(0, std::memory_order_relaxed);
    header->heap_top.store(0, std::memory_order_relaxed);
    header->orchestrator_done.store(0, std::memory_order_relaxed);
    header->last_task_alive.store(0, std::memory_order_relaxed);
    header->heap_tail.store(0, std::memory_order_relaxed);
    header->heap_tail_gen.store(0, std::memory_order_relaxed);

    // Layout info
    header->task_window_size = task_window_size;
    header->heap_size = heap_size;
    header->dep_list_pool_size = dep_list_pool_size;
    
    // Calculate offsets
    uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    header->task_descriptors_offset = offset;
    
    offset += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    header->dep_list_pool_offset = offset;
    
    header->total_size = handle->sm_size;
    header->graph_output_ptr.store(0, std::memory_order_relaxed);
    header->graph_output_size.store(0, std::memory_order_relaxed);
    
    // Initialize dep_list_pool entry 0 as NULL marker
    handle->dep_list_pool[0].task_id = -1;
    handle->dep_list_pool[0].next = nullptr;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_sm_print_layout(PTO2SharedMemoryHandle* handle) {
    if (!handle || !handle->header) return;
    
    PTO2SharedMemoryHeader* h = handle->header;
    
    LOG_INFO("=== PTO2 Shared Memory Layout ===");
    LOG_INFO("Base address:       %p", handle->sm_base);
    LOG_INFO("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_INFO("Task window size:   %" PRIu64, h->task_window_size);
    LOG_INFO("Heap size:          %" PRIu64 " bytes", h->heap_size);
    LOG_INFO("DepList pool size:  %" PRIu64 " entries", h->dep_list_pool_size);
    LOG_INFO("Offsets:");
    LOG_INFO("  TaskDescriptors:  %" PRIu64 " (0x%" PRIx64 ")", h->task_descriptors_offset, h->task_descriptors_offset);
    LOG_INFO("  DepListPool:      %" PRIu64 " (0x%" PRIx64 ")", h->dep_list_pool_offset, h->dep_list_pool_offset);
    LOG_INFO("Flow control:");
    LOG_INFO("  heap_top:           %" PRIu64, h->heap_top.load(std::memory_order_acquire));
    LOG_INFO("  heap_tail:          %" PRIu64, h->heap_tail.load(std::memory_order_acquire));
    LOG_INFO("  current_task_index: %d", h->current_task_index.load(std::memory_order_acquire));
    LOG_INFO("  orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_INFO("  last_task_alive:    %d", h->last_task_alive.load(std::memory_order_acquire));
    LOG_INFO("================================");
}

bool pto2_sm_validate(PTO2SharedMemoryHandle* handle) {
    if (!handle) return false;
    if (!handle->sm_base) return false;
    if (!handle->header) return false;
    
    PTO2SharedMemoryHeader* h = handle->header;
    
    // Check that offsets are within bounds
    if (h->task_descriptors_offset >= h->total_size) return false;
    if (h->dep_list_pool_offset >= h->total_size) return false;
    
    // Check pointer alignment
    if ((uintptr_t)handle->task_descriptors % PTO2_ALIGN_SIZE != 0) return false;
    if ((uintptr_t)handle->dep_list_pool % PTO2_ALIGN_SIZE != 0) return false;
    
    // Check flow control pointer sanity
    int32_t current_task_index = h->current_task_index.load(std::memory_order_acquire);
    int32_t last_task_alive = h->last_task_alive.load(std::memory_order_acquire);
    uint64_t heap_top = h->heap_top.load(std::memory_order_acquire);
    uint64_t heap_tail = h->heap_tail.load(std::memory_order_acquire);
    if (current_task_index < 0) return false;
    if (last_task_alive < 0) return false;
    if (heap_top > h->heap_size) return false;
    if (heap_tail > h->heap_size) return false;
    
    return true;
}
