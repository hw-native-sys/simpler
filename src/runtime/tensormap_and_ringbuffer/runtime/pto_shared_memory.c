/**
 * PTO Runtime2 - Shared Memory Implementation
 * 
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_shared_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// Size Calculation
// =============================================================================

int32_t pto2_sm_calculate_size(int32_t task_window_size, int32_t dep_list_pool_size) {
    int32_t size = 0;
    
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

PTO2SharedMemoryHandle* pto2_sm_create(int32_t task_window_size,
                                        int32_t heap_size,
                                        int32_t dep_list_pool_size) {
    // Allocate handle
    PTO2SharedMemoryHandle* handle = (PTO2SharedMemoryHandle*)calloc(1, sizeof(PTO2SharedMemoryHandle));
    if (!handle) {
        return NULL;
    }
    
    // Calculate total size
    int32_t sm_size = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
    
    // Allocate shared memory (aligned for DMA efficiency)
    #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign(&handle->sm_base, PTO2_ALIGN_SIZE, sm_size) != 0) {
            free(handle);
            return NULL;
        }
    #else
        handle->sm_base = aligned_alloc(PTO2_ALIGN_SIZE, sm_size);
        if (!handle->sm_base) {
            free(handle);
            return NULL;
        }
    #endif
    
    handle->sm_size = sm_size;
    handle->is_owner = true;
    
    // Initialize to zero
    memset(handle->sm_base, 0, sm_size);
    
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
                                                    int32_t sm_size,
                                                    int32_t task_window_size,
                                                    int32_t heap_size,
                                                    int32_t dep_list_pool_size) {
    if (!sm_base || sm_size <= 0) return NULL;

    int32_t required = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
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
                          int32_t task_window_size,
                          int32_t heap_size,
                          int32_t dep_list_pool_size) {
    PTO2SharedMemoryHeader* header = handle->header;
    
    // Flow control pointers (start at 0)
    header->current_task_index = 0;
    header->heap_top = 0;
    header->orchestrator_done = 0;
    header->last_task_alive = 0;
    header->heap_tail = 0;
    
    // Layout info
    header->task_window_size = task_window_size;
    header->heap_size = heap_size;
    header->dep_list_pool_size = dep_list_pool_size;
    
    // Calculate offsets
    int32_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    header->task_descriptors_offset = offset;
    
    offset += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    header->dep_list_pool_offset = offset;
    
    header->total_size = handle->sm_size;
    header->graph_output_ptr = 0;
    header->graph_output_size = 0;
    
    // Initialize dep_list_pool entry 0 as NULL marker
    handle->dep_list_pool[0].task_id = -1;
    handle->dep_list_pool[0].next_offset = 0;
}

void pto2_sm_reset(PTO2SharedMemoryHandle* handle) {
    if (!handle) return;
    
    PTO2SharedMemoryHeader* header = handle->header;
    
    // Reset flow control pointers
    header->current_task_index = 0;
    header->heap_top = 0;
    header->orchestrator_done = 0;
    header->last_task_alive = 0;
    header->heap_tail = 0;
    
    header->graph_output_ptr = 0;
    header->graph_output_size = 0;
    // Clear task descriptors
    memset(handle->task_descriptors, 0, 
           header->task_window_size * sizeof(PTO2TaskDescriptor));
    
    // Clear dependency list pool (keep entry 0 as NULL marker)
    memset(handle->dep_list_pool + 1, 0,
           header->dep_list_pool_size * sizeof(PTO2DepListEntry));
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_sm_print_layout(PTO2SharedMemoryHandle* handle) {
    if (!handle || !handle->header) return;
    
    PTO2SharedMemoryHeader* h = handle->header;
    
    printf("=== PTO2 Shared Memory Layout ===\n");
    printf("Base address:       %p\n", handle->sm_base);
    printf("Total size:         %d bytes\n", h->total_size);
    printf("\n");
    printf("Task window size:   %d\n", h->task_window_size);
    printf("Heap size:          %d bytes\n", h->heap_size);
    printf("DepList pool size:  %d entries\n", h->dep_list_pool_size);
    printf("\n");
    printf("Offsets:\n");
    printf("  TaskDescriptors:  %d (0x%x)\n", h->task_descriptors_offset, h->task_descriptors_offset);
    printf("  DepListPool:      %d (0x%x)\n", h->dep_list_pool_offset, h->dep_list_pool_offset);
    printf("\n");
    printf("Flow control:\n");
    printf("  current_task_index: %d\n", h->current_task_index);
    printf("  last_task_alive:    %d\n", h->last_task_alive);
    printf("  heap_top:           %d\n", h->heap_top);
    printf("  heap_tail:          %d\n", h->heap_tail);
    printf("  orchestrator_done:  %d\n", h->orchestrator_done);
    printf("================================\n");
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
    if (h->current_task_index < 0) return false;
    if (h->last_task_alive < 0) return false;
    if (h->heap_top < 0 || h->heap_top > h->heap_size) return false;
    if (h->heap_tail < 0 || h->heap_tail > h->heap_size) return false;
    
    return true;
}
