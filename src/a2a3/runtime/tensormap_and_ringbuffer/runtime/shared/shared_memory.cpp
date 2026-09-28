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
 * tensormap_and_ringbuffer shared-memory implementation
 *
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t SharedMemoryHandle::calculate_size(uint64_t task_window_size) {
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    return calculate_size_per_ring(task_window_sizes);
}

uint64_t SharedMemoryHandle::calculate_size_per_ring(const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]) {
    // Total SM size = offset just past the last ring, from the single source of
    // truth for the layout (sm_layout::ring_segment_offsets).
    return sm_layout::ring_segment_offsets(task_window_sizes, CHIP_MAX_RING_DEPTH - 1).end;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

void SharedMemoryHandle::setup_pointers_per_ring(const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]) {
    char *base = (char *)sm_base;
    header = (SharedMemoryHeader *)base;

    // Per-ring descriptors / payloads / slot_states — offsets from the single
    // source of truth (sm_layout::ring_segment_offsets), so this setup and
    // the device-address helpers cannot drift.
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        auto off = sm_layout::ring_segment_offsets(task_window_sizes, r);
        auto &ring = header->rings[r];
        ring.task_descriptors = (TaskDescriptor *)(base + off.descriptors);
        ring.task_payloads = (TaskPayload *)(base + off.payloads);
        ring.slot_states = (ChipTaskSlotState *)(base + off.slot_states);
    }
}

void SharedMemoryHandle::setup_pointers(uint64_t task_window_size) {
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    setup_pointers_per_ring(task_window_sizes);
}

bool SharedMemoryHandle::init(void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size) {
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = heap_size;
    }
    return init_per_ring(sm_base_arg, sm_size_arg, task_window_sizes, heap_sizes);
}

bool SharedMemoryHandle::init_per_ring(
    void *sm_base_arg, uint64_t sm_size_arg, const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH],
    const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH]
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    if (sm_size_arg < calculate_size_per_ring(task_window_sizes)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers_per_ring(task_window_sizes);
    init_header_per_ring(task_window_sizes, heap_sizes);
    return true;
}

SharedMemoryHandle *SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(CHIP_TASK_WINDOW_SIZE);
    const size_t off_handle = arena.reserve(sizeof(SharedMemoryHandle), alignof(SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), CHIP_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, CHIP_TASK_WINDOW_SIZE, CHIP_HEAP_SIZE)) return nullptr;
    return handle;
}

void SharedMemoryHandle::destroy() {
    // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
    // calling destroy on them is a no-op so existing callers stay safe.
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

// =============================================================================
// Initialization
// =============================================================================
//
// no need init data in pool, init pool data when used
void SharedMemoryHandle::init_header(uint64_t task_window_size, uint64_t heap_size) {
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = heap_size;
    }
    init_header_per_ring(task_window_sizes, heap_sizes);
}

void SharedMemoryHandle::init_header_per_ring(
    const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH], const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH]
) {
    // Per-ring flow control (start at 0)
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        header->rings[r].fc.init();
    }

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Per-ring layout info
    uint64_t offset = CHIP_ALIGN_UP(sizeof(SharedMemoryHeader), CHIP_ALIGN_SIZE);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        header->rings[r].task_window_size = task_window_sizes[r];
        header->rings[r].task_window_mask = static_cast<int32_t>(task_window_sizes[r] - 1);
        header->rings[r].heap_size = heap_sizes[r];
        header->rings[r].task_descriptors_offset = offset;
        offset += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(TaskDescriptor), CHIP_ALIGN_SIZE);
        offset += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(TaskPayload), CHIP_ALIGN_SIZE);
        offset += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(ChipTaskSlotState), CHIP_ALIGN_SIZE);
    }

    header->total_size = sm_size;

    // Error reporting
    header->orch_error_code.store(SIMPLER_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(SIMPLER_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);
    header->sched_stall_detail.store(SIMPLER_STALL_DETAIL_NONE, std::memory_order_relaxed);
    header->sched_stall_completed.store(0, std::memory_order_relaxed);
    header->sched_stall_total.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_running.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_ready.store(0, std::memory_order_relaxed);
    header->sched_stall_cnt_waiting.store(0, std::memory_order_relaxed);
    header->sched_stall_orch_done.store(0, std::memory_order_relaxed);
    header->sched_stall_task_id.store(TaskId::invalid(), std::memory_order_relaxed);
    header->sched_stall_core.store(-1, std::memory_order_relaxed);

    // No per-slot loop: prepare_task resets each slot when it allocates it, and
    // the scheduler only scans submitted task_ids [last_task_alive,
    // current_task_index), so unsubmitted slots are never read. Per-boot reset
    // is just the header fields above; per-slot state is set lazily at submit.
}

// =============================================================================
// Debug Utilities
// =============================================================================

void SharedMemoryHandle::print_layout() {
    if (!header) return;

    SharedMemoryHeader *h = header;

    LOG_DEBUG("=== Shared Memory Layout ===");
    LOG_DEBUG("Base address:       %p", sm_base);
    LOG_DEBUG("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_DEBUG("Ring depth:         %d", CHIP_MAX_RING_DEPTH);
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        LOG_DEBUG("Ring %d:", r);
        LOG_DEBUG("  task_window_size: %" PRIu64, h->rings[r].task_window_size);
        LOG_DEBUG("  heap_size:        %" PRIu64 " bytes", h->rings[r].heap_size);
        LOG_DEBUG(
            "  descriptors_off:  %" PRIu64 " (0x%" PRIx64 ")", h->rings[r].task_descriptors_offset,
            h->rings[r].task_descriptors_offset
        );
        LOG_DEBUG("  current_task_idx: %d", h->rings[r].fc.current_task_index.load(std::memory_order_acquire));
        LOG_DEBUG("  last_task_alive:  %d", h->rings[r].fc.last_task_alive.load(std::memory_order_acquire));
    }
    LOG_DEBUG("orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_DEBUG("Error state:");
    LOG_DEBUG("  orch_error_code:    %d", h->orch_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_bitmap: 0x%x", h->sched_error_bitmap.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_code:   %d", h->sched_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_thread: %d", h->sched_error_thread.load(std::memory_order_relaxed));
    LOG_DEBUG("================================");
}

bool SharedMemoryHandle::validate() {
    if (!sm_base) return false;
    if (!header) return false;

    SharedMemoryHeader *h = header;

    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        if (!h->rings[r].fc.validate(this, r)) return false;
    }

    return true;
}

bool ChipRingFlowControl::validate(SharedMemoryHandle *handle, int32_t ring_id) const {
    if (!handle) return false;
    if (!handle->header) return false;
    if (ring_id < 0 || ring_id >= CHIP_MAX_RING_DEPTH) return false;

    const SharedMemoryHeader *h = handle->header;

    // Check that offsets are within bounds
    if (h->rings[ring_id].task_descriptors_offset >= h->total_size) return false;

    // Check pointer alignment
    if ((uintptr_t)h->rings[ring_id].task_descriptors % CHIP_ALIGN_SIZE != 0) return false;

    // Check flow control pointer sanity
    int32_t current = current_task_index.load(std::memory_order_acquire);
    int32_t last_alive = last_task_alive.load(std::memory_order_acquire);
    if (current < 0) return false;
    if (last_alive < 0) return false;

    return true;
}
