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
 * PTO Runtime2 - Shared Memory Layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout (per-ring sections repeat for each ring 0..PTO2_MAX_RING_DEPTH-1):
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (per-ring flow control + sync)
 *   +---------------------------+
 *   | Ring 0: TaskDescriptor[]  |
 *   | Ring 0: TaskPayload[]     |
 *   | Ring 0: TaskSlotState[]   |
 *   +---------------------------+
 *   | Ring 1: TaskDescriptor[]  |
 *   | Ring 1: TaskPayload[]     |
 *   | Ring 1: TaskSlotState[]   |
 *   +---------------------------+
 *   | ...                       |
 *   +---------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct PTO2SharedMemoryHandle;

/**
 * Per-ring flow control state in shared memory.
 * Written/read by Orchestrator and Scheduler for synchronization.
 */
struct alignas(64) PTO2RingFlowControl {
    // === Cache Line 0: Written by Orchestrator, Read by Scheduler ===
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // === Cache Line 1: Written by Scheduler, Read by Orchestrator (for back-pressure) ===
    alignas(64) std::atomic<int32_t> last_task_alive;  // Task ring tail (oldest active task)

    // Per-boot SM reset. PTO2TaskAllocator::init() seeds its private
    // local_task_id_ from initial_local_task_id (default 0 in production)
    // *without* dereferencing current_task_index — it relies on this reset
    // running on every AICPU boot so 0 stays in sync. If you ever change
    // the initial fc value or the boot ordering, update the default in
    // PTO2TaskAllocator::init (pto_ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() {
        current_task_index.store(0, std::memory_order_relaxed);
        last_task_alive.store(0, std::memory_order_relaxed);
    }

    bool validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 128, "PTO2RingFlowControl must be exactly 2 cache lines (128B)");

/**
 * Per-ring shared memory header section.
 *
 * Groups flow-control, layout info, and per-ring data pointers for a single ring.
 * Pointers are host-side only (set by setup_pointers, invalid on device).
 */
struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) { return task_descriptors[local_id & task_window_mask]; }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id) { return task_payloads[local_id & task_window_mask]; }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id) { return slot_states[local_id & task_window_mask]; }
};

/**
 * Shared memory header structure
 *
 * Contains per-ring flow control and global layout information.
 */
struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    // === PER-RING FLOW CONTROL + LAYOUT INFO (set once at init) ===
    PTO2SharedMemoryRingHeader rings[PTO2_MAX_RING_DEPTH];

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // Graph output for copy-back (set by orchestrator when using packed buffer)
    // Host finalize copies from this address instead of dev_ptr when non-zero
    std::atomic<uint64_t> graph_output_ptr;   // Address where final output was written (packed buffer)
    std::atomic<uint64_t> graph_output_size;  // Size in bytes

    // === ERROR REPORTING ===

    // Orchestrator fatal error code (Orchestrator → Scheduler, AICPU → Host)
    // Non-zero signals fatal error. Written by orchestrator, read by scheduler and host.
    std::atomic<int32_t> orch_error_code;

    // Scheduler error state (Scheduler → Host, independent of orchestrator)
    // Written by scheduler threads on timeout; read by orchestrator and host.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer
};

static_assert(
    (sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE == 0) && (sizeof(PTO2SharedMemoryHeader) < 4096),
    "PTO2SharedMemoryHeader should be reasonably sized"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use PTO2SharedMemoryHeader* directly.
 */
struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);
    static uint64_t calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init
    // using default PTO2_TASK_WINDOW_SIZE / PTO2_HEAP_SIZE. Only valid when the
    // arena is otherwise empty (the call performs the single commit). All
    // memory is owned by the arena — caller must not call destroy().
    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for the requested
    // `task_window_size`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);

    void destroy();
    void print_layout();
    bool validate();

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size);
    void init_header_per_ring(
        const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
    );
    void setup_pointers(uint64_t task_window_size);
    void setup_pointers_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (ring flow-control counters,
// task_descriptors arrays, orch_error_code) so it can wire them into the
// orchestrator / scheduler init_data path without dereferencing the SM —
// the SM lives in device memory and cannot be touched from host.
//
// These helpers compute those addresses by offset arithmetic on the SM
// device base. Pure pointer math, no loads/stores; safe to call from host.
// The same arithmetic happens on AICPU too (via PTO2SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace pto2_sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, orch_error_code)
    );
}

inline PTO2SharedMemoryRingHeader *ring_header_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<PTO2SharedMemoryRingHeader *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, rings) +
        static_cast<size_t>(ring_id) * sizeof(PTO2SharedMemoryRingHeader)
    );
}

inline std::atomic<int32_t> *ring_current_task_index_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(PTO2SharedMemoryRingHeader, fc) +
        offsetof(PTO2RingFlowControl, current_task_index)
    );
}

inline std::atomic<int32_t> *ring_last_task_alive_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(PTO2SharedMemoryRingHeader, fc) +
        offsetof(PTO2RingFlowControl, last_task_alive)
    );
}

// Walk the per-ring SM layout (same arithmetic as setup_pointers_per_ring)
// to compute ring `ring_id`'s task_descriptors device address. Accepts a
// per-ring window-size array so the helper's signature mirrors
// `PTO2SharedMemoryHandle::setup_pointers_per_ring` and cannot silently
// disagree with the SM layout when (hypothetically) ring sizes diverge.
inline PTO2TaskDescriptor *ring_task_descriptors_addr(
    void *sm_dev_base, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH], int ring_id
) noexcept {
    assert(ring_id >= 0 && ring_id < PTO2_MAX_RING_DEPTH && "pto2_sm_layout: ring_id out of range");
    char *p = static_cast<char *>(sm_dev_base);
    p += PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    for (int r = 0; r < ring_id; r++) {
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    }
    return reinterpret_cast<PTO2TaskDescriptor *>(p);
}

}  // namespace pto2_sm_layout
