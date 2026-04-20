/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * PTO Runtime2 - Shared Memory Implementation
 *
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t pto2_sm_calculate_size(uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    return pto2_sm_calculate_size_per_ring(task_window_sizes);
}

uint64_t pto2_sm_calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    uint64_t size = 0;

    // Header (aligned to cache line)
    size += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Per-ring task descriptors and payloads
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    }

    return size;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

static void
pto2_sm_setup_pointers_per_ring(PTO2SharedMemoryHandle *handle, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    char *ptr = (char *)handle->sm_base;

    // Header
    handle->header = (PTO2SharedMemoryHeader *)ptr;
    ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Per-ring task descriptors and payloads
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        handle->task_descriptors[r] = (PTO2TaskDescriptor *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);

        handle->task_payloads[r] = (PTO2TaskPayload *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    }
}

static void pto2_sm_setup_pointers(PTO2SharedMemoryHandle *handle, uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    pto2_sm_setup_pointers_per_ring(handle, task_window_sizes);
}

PTO2SharedMemoryHandle *pto2_sm_create(uint64_t task_window_size, uint64_t heap_size) {
    // Allocate handle
    PTO2SharedMemoryHandle *handle = (PTO2SharedMemoryHandle *)calloc(1, sizeof(PTO2SharedMemoryHandle));
    if (!handle) {
        return NULL;
    }

    // Calculate total size
    uint64_t sm_size = pto2_sm_calculate_size(task_window_size);

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
    pto2_sm_setup_pointers(handle, task_window_size);

    // Initialize header
    pto2_sm_init_header(handle, task_window_size, heap_size);

    return handle;
}

PTO2SharedMemoryHandle *pto2_sm_create_default(void) { return pto2_sm_create(PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE); }

PTO2SharedMemoryHandle *
pto2_sm_create_from_buffer(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size) {
    if (!sm_base || sm_size == 0) return NULL;

    uint64_t required = pto2_sm_calculate_size(task_window_size);
    if (sm_size < required) return NULL;

    PTO2SharedMemoryHandle *handle = (PTO2SharedMemoryHandle *)calloc(1, sizeof(PTO2SharedMemoryHandle));
    if (!handle) return NULL;

    handle->sm_base = sm_base;
    handle->sm_size = sm_size;
    handle->is_owner = false;

    pto2_sm_setup_pointers(handle, task_window_size);
    pto2_sm_init_header(handle, task_window_size, heap_size);

    return handle;
}

void pto2_sm_destroy(PTO2SharedMemoryHandle *handle) {
    if (!handle) return;

    if (handle->is_owner && handle->sm_base) {
        free(handle->sm_base);
    }

    free(handle);
}

// =============================================================================
// Initialization
// =============================================================================
//
// no need init data in pool, init pool data when used
void pto2_sm_init_header(PTO2SharedMemoryHandle *handle, uint64_t task_window_size, uint64_t heap_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = heap_size;
    }
    pto2_sm_init_header_per_ring(handle, task_window_sizes, heap_sizes);
}

void pto2_sm_init_header_per_ring(
    PTO2SharedMemoryHandle *handle, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    PTO2SharedMemoryHeader *header = handle->header;

    // Per-ring flow control (start at 0)
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        header->rings[r].fc.init();
    }

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Per-ring layout info
    uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        header->rings[r].task_window_size = task_window_sizes[r];
        header->rings[r].heap_size = heap_sizes[r];
        header->rings[r].task_descriptors_offset = offset;
        offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    }

    header->total_size = handle->sm_size;
    header->graph_output_ptr.store(0, std::memory_order_relaxed);
    header->graph_output_size.store(0, std::memory_order_relaxed);

    // Error reporting
    header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_sm_print_layout(PTO2SharedMemoryHandle *handle) {
    if (!handle || !handle->header) return;

    PTO2SharedMemoryHeader *h = handle->header;

    LOG_INFO("=== PTO2 Shared Memory Layout ===");
    LOG_INFO("Base address:       %p", handle->sm_base);
    LOG_INFO("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_INFO("Ring depth:         %d", PTO2_MAX_RING_DEPTH);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        LOG_INFO("Ring %d:", r);
        LOG_INFO("  task_window_size: %" PRIu64, h->rings[r].task_window_size);
        LOG_INFO("  heap_size:        %" PRIu64 " bytes", h->rings[r].heap_size);
        LOG_INFO(
            "  descriptors_off:  %" PRIu64 " (0x%" PRIx64 ")", h->rings[r].task_descriptors_offset,
            h->rings[r].task_descriptors_offset
        );
        LOG_INFO("  current_task_idx: %d", h->rings[r].fc.current_task_index.load(std::memory_order_acquire));
        LOG_INFO("  last_task_alive:  %d", h->rings[r].fc.last_task_alive.load(std::memory_order_acquire));
    }
    LOG_INFO("orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_INFO("Error state:");
    LOG_INFO("  orch_error_code:    %d", h->orch_error_code.load(std::memory_order_relaxed));
    LOG_INFO("  sched_error_bitmap: 0x%x", h->sched_error_bitmap.load(std::memory_order_relaxed));
    LOG_INFO("  sched_error_code:   %d", h->sched_error_code.load(std::memory_order_relaxed));
    LOG_INFO("  sched_error_thread: %d", h->sched_error_thread.load(std::memory_order_relaxed));
    LOG_INFO("================================");
}

bool pto2_sm_validate(PTO2SharedMemoryHandle *handle) {
    if (!handle) return false;
    if (!handle->sm_base) return false;
    if (!handle->header) return false;

    PTO2SharedMemoryHeader *h = handle->header;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!h->rings[r].fc.validate(handle, r)) return false;
    }

    return true;
}

bool PTO2RingFlowControl::validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const {
    if (!handle) return false;
    if (!handle->header) return false;
    if (ring_id < 0 || ring_id >= PTO2_MAX_RING_DEPTH) return false;

    const PTO2SharedMemoryHeader *h = handle->header;

    // Check that offsets are within bounds
    if (h->rings[ring_id].task_descriptors_offset >= h->total_size) return false;

    // Check pointer alignment
    if ((uintptr_t)handle->task_descriptors[ring_id] % PTO2_ALIGN_SIZE != 0) return false;

    // Check flow control pointer sanity
    int32_t current = current_task_index.load(std::memory_order_acquire);
    int32_t last_alive = last_task_alive.load(std::memory_order_acquire);
    if (current < 0) return false;
    if (last_alive < 0) return false;

    return true;
}
