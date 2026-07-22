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

#include "pto_shared_memory.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"

uint64_t PTO2SharedMemoryHandle::calculate_size(uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        task_window_sizes[r] = task_window_size;
    return calculate_size_per_ring(task_window_sizes);
}

uint64_t PTO2SharedMemoryHandle::calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    uint64_t size = 0;

    // Header (aligned to cache line)
    size += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Per-ring task descriptors and payloads
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
        size += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(std::atomic<uint8_t>), PTO2_ALIGN_SIZE);
    }

    return size;
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

bool PTO2SharedMemoryHandle::init(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    if (sm_size_arg < calculate_size(task_window_size)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(task_window_size);
    init_header(task_window_size, heap_size);
    return true;
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
    setup_pointers_per_ring(task_window_sizes);
    init_header_per_ring(task_window_sizes, heap_sizes);
    return true;
}

void PTO2SharedMemoryHandle::destroy() {
    // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
    // calling destroy on them is a no-op so existing callers stay safe.
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

void PTO2SharedMemoryHandle::print_layout() {
    if (!header) return;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {}
}

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
    // Per-ring flow control (start at 0)
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        header->rings[r].fc.init();
        // -1 = "no task completed yet"; first task to complete (local_id 0)
        // will advance the watermark to 0.
        header->rings[r].completed_watermark.store(-1, std::memory_order_relaxed);
        // Shared memory is not guaranteed zero on device. The watermark-advance
        // loop reads completion_flags for not-yet-completed slots, so stale
        // non-zero bytes would prematurely advance completed_watermark and retire
        // live slots. Zero the whole per-ring flag block (1 byte/slot, cheap).
        __builtin_memset(
            (void *)header->rings[r].completion_flags, 0, task_window_sizes[r] * sizeof(std::atomic<uint8_t>)
        );
    }

    header->orchestrator_done.store(0, std::memory_order_relaxed);

    // Per-ring layout info
    uint64_t offset = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        header->rings[r].task_window_size = task_window_sizes[r];
        header->rings[r].task_window_mask = static_cast<int32_t>(task_window_sizes[r] - 1);
        header->rings[r].heap_size = heap_sizes[r];
        header->rings[r].task_descriptors_offset = offset;
        offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
        offset += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    }

    header->total_size = sm_size;
    header->graph_output_ptr.store(0, std::memory_order_relaxed);
    header->graph_output_size.store(0, std::memory_order_relaxed);

    // Error reporting
    header->orch_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_bitmap.store(0, std::memory_order_relaxed);
    header->sched_error_code.store(PTO2_ERROR_NONE, std::memory_order_relaxed);
    header->sched_error_thread.store(-1, std::memory_order_relaxed);

    // No per-slot loop: prepare_task() resets each slot when the allocator
    // hands it out (bind_ring + reset_for_reuse + per-submit fields). The
    // scheduler only scans submitted task_ids [last_task_alive,
    // current_task_index), so unsubmitted slots are never read. Cost moves
    // from O(sum(task_window_sizes)) every run to O(tasks actually
    // submitted) — and stays on the device. Mirrors upstream #1199.
}

void PTO2SharedMemoryHandle::setup_pointers(uint64_t task_window_size) {
    uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        task_window_sizes[r] = task_window_size;
    setup_pointers_per_ring(task_window_sizes);
}

void PTO2SharedMemoryHandle::setup_pointers_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    char *ptr = (char *)sm_base;

    // Header
    header = (PTO2SharedMemoryHeader *)ptr;
    ptr += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);

    // Per-ring task descriptors, payloads, and slot states
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto &ring = header->rings[r];
        ring.task_descriptors = (PTO2TaskDescriptor *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);

        ring.task_payloads = (PTO2TaskPayload *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);

        ring.slot_states = (PTO2TaskSlotState *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);

        ring.completion_flags = (std::atomic<uint8_t> *)ptr;
        ptr += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(std::atomic<uint8_t>), PTO2_ALIGN_SIZE);
    }
}

bool PTO2SharedMemoryHandle::validate() {
    if (!sm_base) return false;
    if (!header) return false;

    PTO2SharedMemoryHeader *h = header;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        if (!h->rings[r].fc.validate(this, r)) return false;

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
    if ((uintptr_t)h->rings[ring_id].task_descriptors % PTO2_ALIGN_SIZE != 0) return false;

    // Check flow control pointer sanity
    int32_t current = current_task_index.load(std::memory_order_acquire);
    int32_t last_alive = last_task_alive.load(std::memory_order_acquire);
    if (current < 0) return false;
    if (last_alive < 0) return false;

    return true;
}
