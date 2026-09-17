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
 * tensormap_and_ringbuffer shared-memory layout
 *
 * Defines the shared memory structure for Orchestrator-Scheduler communication.
 *
 * Memory Layout (per-ring sections repeat for each ring 0..CHIP_MAX_RING_DEPTH-1):
 *   +-------------------------------+
 *   | SharedMemoryHeader            |  (per-ring flow control + sync)
 *   +-------------------------------+
 *   | Ring 0: TaskDescriptor[]      |
 *   | Ring 0: TaskPayload[]         |
 *   | Ring 0: ChipTaskSlotState[]   |
 *   +-------------------------------+
 *   | Ring 1: TaskDescriptor[]      |
 *   | Ring 1: TaskPayload[]         |
 *   | Ring 1: ChipTaskSlotState[]   |
 *   +-------------------------------+
 *   | ...                           |
 *   +-------------------------------+
 *
 * Design principles:
 * - Only data needed for Orchestrator<->Scheduler communication is here
 * - TensorMap, scope_stack, ready_queues, dep_pool are in private memory
 * - Flow control via atomic counters/flags (no locks needed for single-word R/W)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stddef.h>

#include "utils/device_arena.h"
#include "runtime_types.h"

// =============================================================================
// Shared Memory Header
// =============================================================================

struct SharedMemoryHandle;

/**
 * Per-ring flow control state in shared memory.
 * Written/read by Orchestrator and Scheduler for synchronization.
 */
struct alignas(64) ChipRingFlowControl {
    // === Cache Line 0: Written by Orchestrator, Read by Scheduler ===
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // === Cache Line 1: Written by Scheduler, Read by Orchestrator (for back-pressure) ===
    alignas(64) std::atomic<int32_t> last_task_alive;  // Task ring tail (oldest active task)

    // Per-boot SM reset. TaskAllocator::init() seeds its private
    // local_task_id_ from initial_local_task_id (default 0 in production)
    // *without* dereferencing current_task_index — it relies on this reset
    // running on every AICPU boot so 0 stays in sync. If you ever change
    // the initial fc value or the boot ordering, update the default in
    // TaskAllocator::init (ring_buffer.h) in the same change, or
    // submit IDs will be off by the divergence.
    void init() {
        current_task_index.store(0, std::memory_order_relaxed);
        last_task_alive.store(0, std::memory_order_relaxed);
    }

    bool validate(SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(ChipRingFlowControl) == 128, "ChipRingFlowControl must be exactly 2 cache lines (128B)");

/**
 * Per-ring shared memory header section.
 *
 * Groups flow-control, layout info, and per-ring data pointers for a single ring.
 * Pointers are host-side only (set by setup_pointers, invalid on device).
 */
struct alignas(64) SharedMemoryRingHeader {
    ChipRingFlowControl fc;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    TaskDescriptor *task_descriptors;
    TaskPayload *task_payloads;
    ChipTaskSlotState *slot_states;

    int32_t get_slot_by_task_id(int32_t local_task_id) { return local_task_id & task_window_mask; }

    TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    TaskDescriptor &get_task_by_task_id(int32_t local_id) { return task_descriptors[get_slot_by_task_id(local_id)]; }

    TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }

    TaskPayload &get_payload_by_task_id(int32_t local_id) { return task_payloads[get_slot_by_task_id(local_id)]; }

    ChipTaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    ChipTaskSlotState &get_slot_state_by_task_id(int32_t local_id) {
        return slot_states[get_slot_by_task_id(local_id)];
    }
};

static_assert(sizeof(SharedMemoryRingHeader) == 192, "SharedMemoryRingHeader layout drift");
static_assert(
    offsetof(SharedMemoryRingHeader, task_descriptors_offset) == 152,
    "SharedMemoryRingHeader task_descriptors_offset layout drift"
);

/**
 * Shared memory header structure
 *
 * Contains per-ring flow control and global layout information.
 */
struct alignas(CHIP_ALIGN_SIZE) SharedMemoryHeader {
    // === PER-RING FLOW CONTROL + LAYOUT INFO (set once at init) ===
    SharedMemoryRingHeader rings[CHIP_MAX_RING_DEPTH];

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // === ERROR REPORTING ===

    // Orchestrator fatal error code (Orchestrator → Scheduler, AICPU → Host)
    // Non-zero signals fatal error. Written by orchestrator, read by scheduler and host.
    std::atomic<int32_t> orch_error_code;

    // Scheduler error state (Scheduler → Host, independent of orchestrator)
    // Written by scheduler threads on timeout; read by orchestrator and host.
    std::atomic<uint32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;     // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;   // Thread index of last error writer

    // Sub-classification + locators for a sched_error_code==100 timeout. Written
    // by the scheduler thread that wins the code latch; read by host so it can
    // distinguish device error TYPES (SIMPLER_STALL_DETAIL_*) without reading the
    // device log. The full stall snapshot stays in the device log / plog — only
    // this one class + a few locator ints cross the boundary.
    std::atomic<int32_t> sched_stall_detail;       // SIMPLER_STALL_DETAIL_* (NONE when no timeout)
    std::atomic<int32_t> sched_stall_completed;    // completed_tasks_ at timeout
    std::atomic<int32_t> sched_stall_total;        // total_tasks_ at timeout
    std::atomic<int32_t> sched_stall_cnt_running;  // tasks observed RUNNING (on a core)
    std::atomic<int32_t> sched_stall_cnt_ready;    // tasks fanin-satisfied but not dispatched
    std::atomic<int32_t> sched_stall_cnt_waiting;  // tasks still waiting on fanin
    std::atomic<int32_t> sched_stall_orch_done;    // orchestrator_done flag at timeout (0/1)
    std::atomic<int64_t> sched_stall_task_id;      // S1: stuck task_id (-1 if N/A)
    std::atomic<int32_t> sched_stall_core;         // S1: stuck core id (-1 if N/A)
};

static_assert(sizeof(SharedMemoryHeader) == 896, "SharedMemoryHeader layout drift");
static_assert(offsetof(SharedMemoryHeader, total_size) == 776, "SharedMemoryHeader total_size layout drift");
static_assert(offsetof(SharedMemoryHeader, orch_error_code) == 784, "SharedMemoryHeader orch_error_code layout drift");
static_assert(
    offsetof(SharedMemoryHeader, sched_stall_task_id) == 832, "SharedMemoryHeader sched_stall_task_id layout drift"
);

// The error-reporting tail as one contiguous range: `orch_error_code` through
// the end of the header. A run publishes exactly this range into its own result
// region, and the host overlays it back at the same offset of a zeroed header,
// so neither side enumerates the fields and adding one needs no change on
// either. The static_assert is the guard that keeps the tail last.
constexpr size_t SHARED_MEMORY_ERROR_TAIL_OFFSET = offsetof(SharedMemoryHeader, orch_error_code);
constexpr size_t SHARED_MEMORY_ERROR_TAIL_BYTES = sizeof(SharedMemoryHeader) - SHARED_MEMORY_ERROR_TAIL_OFFSET;
static_assert(SHARED_MEMORY_ERROR_TAIL_BYTES == 112, "SharedMemoryHeader error tail is no longer the header's end");

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory lifecycle management (create/destroy).
 * Runtime components (orchestrator, scheduler) use SharedMemoryHeader* directly.
 */
struct SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);
    static uint64_t calculate_size_per_ring(const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]);

    // UT convenience: reserve wrapper + sm_base on `arena`, commit, and init
    // using default CHIP_TASK_WINDOW_SIZE / CHIP_HEAP_SIZE. Only valid when the
    // arena is otherwise empty (the call performs the single commit). All
    // memory is owned by the arena — caller must not call destroy().
    static SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    // In-place init for caller-provided wrapper storage (e.g. a region carved
    // out of a DeviceArena). Sets is_owner = false, calls setup_pointers and
    // init_header. Returns false when `sm_size` is too small for the requested
    // `task_window_size`.
    bool init(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);
    bool init_per_ring(
        void *sm_base, uint64_t sm_size, const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH],
        const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH]
    );

    void destroy();
    void print_layout();
    bool validate();

private:
    void init_header(uint64_t task_window_size, uint64_t heap_size);
    void init_header_per_ring(
        const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH], const uint64_t heap_sizes[CHIP_MAX_RING_DEPTH]
    );
    void setup_pointers(uint64_t task_window_size);
    void setup_pointers_per_ring(const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH]);
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
// The same arithmetic happens on AICPU too (via SharedMemoryHandle's
// own setup_pointers), so values are guaranteed consistent across sides.
namespace sm_layout {

inline std::atomic<int32_t> *orch_error_code_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(SharedMemoryHeader, orch_error_code)
    );
}

inline SharedMemoryRingHeader *ring_header_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<SharedMemoryRingHeader *>(
        static_cast<char *>(sm_dev_base) + offsetof(SharedMemoryHeader, rings) +
        static_cast<size_t>(ring_id) * sizeof(SharedMemoryRingHeader)
    );
}

inline std::atomic<int32_t> *ring_current_task_index_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(SharedMemoryRingHeader, fc) +
        offsetof(ChipRingFlowControl, current_task_index)
    );
}

inline std::atomic<int32_t> *ring_last_task_alive_addr(void *sm_dev_base, int ring_id) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        reinterpret_cast<char *>(ring_header_addr(sm_dev_base, ring_id)) + offsetof(SharedMemoryRingHeader, fc) +
        offsetof(ChipRingFlowControl, last_task_alive)
    );
}

// Byte offsets (from the SM base) of one ring's three segments. The per-ring
// layout is: header, then for each ring descriptors -> payloads -> slot_states,
// every segment CHIP_ALIGN_UP-padded.
struct ChipRingSegmentOffsets {
    uint64_t descriptors;
    uint64_t payloads;
    uint64_t slot_states;
    uint64_t end;  // offset just past this ring's slot_states (next ring's start; total SM size for the last ring)
};

// Single source of truth for the per-ring SM layout. Returns offsets (not
// pointers), so it serves BOTH the host-side pointer setup
// (`setup_pointers_per_ring`, which adds `sm_base`) and the device-address
// helpers below (which add `sm_dev_base`). Adding or reordering a per-ring
// segment is a one-line edit here; every consumer follows automatically, so the
// layout walk can never silently disagree across call sites.
inline ChipRingSegmentOffsets
ring_segment_offsets(const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH], int ring_id) noexcept {
    assert(ring_id >= 0 && ring_id < CHIP_MAX_RING_DEPTH && "sm_layout: ring_id out of range");
    uint64_t off = CHIP_ALIGN_UP(sizeof(SharedMemoryHeader), CHIP_ALIGN_SIZE);
    for (int r = 0; r < ring_id; r++) {
        off += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(TaskDescriptor), CHIP_ALIGN_SIZE);
        off += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(TaskPayload), CHIP_ALIGN_SIZE);
        off += CHIP_ALIGN_UP(task_window_sizes[r] * sizeof(ChipTaskSlotState), CHIP_ALIGN_SIZE);
    }
    ChipRingSegmentOffsets o{};
    o.descriptors = off;
    off += CHIP_ALIGN_UP(task_window_sizes[ring_id] * sizeof(TaskDescriptor), CHIP_ALIGN_SIZE);
    o.payloads = off;
    off += CHIP_ALIGN_UP(task_window_sizes[ring_id] * sizeof(TaskPayload), CHIP_ALIGN_SIZE);
    o.slot_states = off;
    off += CHIP_ALIGN_UP(task_window_sizes[ring_id] * sizeof(ChipTaskSlotState), CHIP_ALIGN_SIZE);
    o.end = off;
    return o;
}

// Device address of ring `ring_id`'s task_descriptors array.
inline TaskDescriptor *ring_task_descriptors_addr(
    void *sm_dev_base, const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH], int ring_id
) noexcept {
    return reinterpret_cast<TaskDescriptor *>(
        static_cast<char *>(sm_dev_base) + ring_segment_offsets(task_window_sizes, ring_id).descriptors
    );
}

// Device address of ring `ring_id`'s slot_states array (used by the allocator's
// deadlock detector to identify the head task's slot).
inline ChipTaskSlotState *
ring_slot_states_addr(void *sm_dev_base, const uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH], int ring_id) noexcept {
    return reinterpret_cast<ChipTaskSlotState *>(
        static_cast<char *>(sm_dev_base) + ring_segment_offsets(task_window_sizes, ring_id).slot_states
    );
}

}  // namespace sm_layout
