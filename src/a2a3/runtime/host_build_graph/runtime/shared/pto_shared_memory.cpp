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

uint64_t PTO2SharedMemoryHandle::calculate_size(uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    return calculate_size_per_ring(task_window_sizes);
}

uint64_t PTO2SharedMemoryHandle::calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    // Total SM size = offset just past the ring's slot_states, from the single
    // source of truth for the layout (pto2_sm_layout::ring_segment_offsets).
    return pto2_sm_layout::ring_segment_offsets(task_window_sizes[0]).end;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

void PTO2SharedMemoryHandle::setup_pointers_per_ring(
    const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], uint64_t pitch, uint64_t payload_bytes
) {
    (void)task_window_sizes;
    char *base = (char *)sm_base;
    header = (PTO2SharedMemoryHeader *)base;

    // Per-ring descriptors / payloads / slot_states — offsets from the single
    // source of truth (pto2_sm_layout::ring_segment_offsets), so this setup and
    // the device-address helpers cannot drift.
    auto off = pto2_sm_layout::ring_segment_offsets_with_payload_bytes(pitch, payload_bytes);
    auto &ring = header->ring;
    ring.task_descriptors = (PTO2TaskDescriptor *)(base + off.descriptors);
    ring.task_payloads = (PTO2TaskPayload *)(base + off.payloads);
    ring.slot_states = (PTO2TaskSlotState *)(base + off.slot_states);
    ring.completion_flags = (std::atomic<uint8_t> *)(base + off.completion_flags);
}

void PTO2SharedMemoryHandle::setup_pointers(uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
    }
    setup_pointers_per_ring(task_window_sizes, task_window_size, task_window_size * sizeof(PTO2TaskPayload));
}

bool PTO2SharedMemoryHandle::init(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size
) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = heap_size;
    }
    return init_per_ring(sm_base_arg, sm_size_arg, task_window_sizes, heap_sizes);
}

bool PTO2SharedMemoryHandle::init_per_ring(
    void *sm_base_arg, uint64_t sm_size_arg, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    if (sm_size_arg < calculate_size_per_ring(task_window_sizes)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers_per_ring(task_window_sizes, task_window_sizes[0], task_window_sizes[0] * sizeof(PTO2TaskPayload));
    init_header_per_ring(task_window_sizes, heap_sizes);
    return true;
}

bool PTO2SharedMemoryHandle::attach_populated(
    void *sm_base_arg, uint64_t sm_size_arg, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], uint64_t live_slots,
    uint64_t payload_bytes
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    // A pitch above the capacity would name a slot the ring cannot address, and one
    // below what the host used would place every segment past the descriptors
    // short. This is the whole contract between the two sides, checked once at
    // attach rather than trusted.
    if (live_slots == 0 || live_slots > task_window_sizes[0]) return false;
    if (payload_bytes > task_window_sizes[0] * sizeof(PTO2TaskPayload)) return false;
    // The region holds the compacted image, so that is what has to fit — the ring
    // capacity is no longer its size.
    const uint64_t image_end = pto2_sm_layout::ring_segment_offsets_with_payload_bytes(live_slots, payload_bytes).end;
    if (sm_size_arg < image_end) return false;
    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers_per_ring(task_window_sizes, live_slots, payload_bytes);
    // Deliberately NO init_header_per_ring: the SM already holds the host
    // orchestrator's task graph (descriptors, slot states, ring counters).
    return true;
}

PTO2SharedMemoryHandle *PTO2SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(PTO2_TASK_WINDOW_SIZE);
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), PTO2_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, PTO2_TASK_WINDOW_SIZE, PTO2_HEAP_SIZE)) return nullptr;
    return handle;
}

void PTO2SharedMemoryHandle::destroy() {
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
void PTO2SharedMemoryHandle::init_header(uint64_t task_window_size, uint64_t heap_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = task_window_size;
        heap_sizes[r] = heap_size;
    }
    init_header_per_ring(task_window_sizes, heap_sizes);
}

void PTO2SharedMemoryHandle::init_header_per_ring(
    const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    // Flow control (starts at 0)
    header->ring.fc.init();

    // Polling completion: -1 = "no task completed yet"; the first task to
    // complete (local_id 0) advances the watermark to 0.
    header->ring.completed_watermark.store(-1, std::memory_order_relaxed);

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Ring layout info
    uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    header->ring.task_window_size = task_window_sizes[0];
    header->ring.task_window_mask = static_cast<int32_t>(task_window_sizes[0] - 1);
    header->ring.heap_size = heap_sizes[0];
    header->ring.task_descriptors_offset = offset;
    offset += PTO2_ALIGN_UP(task_window_sizes[0] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    offset += PTO2_ALIGN_UP(task_window_sizes[0] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    offset += PTO2_ALIGN_UP(task_window_sizes[0] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);

    header->total_size = sm_size;

    // Error reporting
    header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);

    // Per-slot init (slot_states.reset_for_reuse() + active_mask, and clearing the
    // completion flag) happens init-on-write in orch::prepare_task as each slot
    // [0, total_tasks) is claimed, so the SM init/upload cost tracks the task
    // count, not ring capacity. The device reads no slot past total_tasks, so the
    // unclaimed tail is left uninitialized.
}

// =============================================================================
// Debug Utilities
// =============================================================================

void PTO2SharedMemoryHandle::print_layout() {
    if (!header) return;

    PTO2SharedMemoryHeader *h = header;

    LOG_DEBUG("=== PTO2 Shared Memory Layout ===");
    LOG_DEBUG("Base address:       %p", sm_base);
    LOG_DEBUG("Total size:         %" PRIu64 " bytes", h->total_size);
    LOG_DEBUG("Ring depth:         %d", PTO2_MAX_RING_DEPTH);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        LOG_DEBUG("Ring %d:", r);
        LOG_DEBUG("  task_window_size: %" PRIu64, h->ring.task_window_size);
        LOG_DEBUG("  heap_size:        %" PRIu64 " bytes", h->ring.heap_size);
        LOG_DEBUG(
            "  descriptors_off:  %" PRIu64 " (0x%" PRIx64 ")", h->ring.task_descriptors_offset,
            h->ring.task_descriptors_offset
        );
        LOG_DEBUG("  current_task_idx: %d", h->ring.fc.current_task_index.load(std::memory_order_acquire));
    }
    LOG_DEBUG("orchestrator_done:  %d", h->orchestrator_done.load(std::memory_order_acquire));
    LOG_DEBUG("Error state:");
    LOG_DEBUG("  orch_error_code:    %d", h->orch_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_bitmap: 0x%x", h->sched_error_bitmap.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_code:   %d", h->sched_error_code.load(std::memory_order_relaxed));
    LOG_DEBUG("  sched_error_thread: %d", h->sched_error_thread.load(std::memory_order_relaxed));
    LOG_DEBUG("================================");
}

bool PTO2SharedMemoryHandle::validate() {
    if (!sm_base) return false;
    if (!header) return false;

    PTO2SharedMemoryHeader *h = header;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!h->ring.fc.validate(this, r)) return false;
    }

    return true;
}

bool PTO2RingFlowControl::validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const {
    if (!handle) return false;
    if (!handle->header) return false;
    if (ring_id < 0 || ring_id >= PTO2_MAX_RING_DEPTH) return false;

    const PTO2SharedMemoryHeader *h = handle->header;

    // Check that offsets are within bounds
    if (h->ring.task_descriptors_offset >= h->total_size) return false;

    // Check pointer alignment
    if ((uintptr_t)h->ring.task_descriptors % PTO2_ALIGN_SIZE != 0) return false;

    // Check flow control pointer sanity
    int32_t current = current_task_index.load(std::memory_order_acquire);
    if (current < 0) return false;

    return true;
}
