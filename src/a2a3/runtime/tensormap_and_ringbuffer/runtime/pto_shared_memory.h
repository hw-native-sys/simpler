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

#pragma once

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"

struct PTO2SharedMemoryHandle;

struct alignas(64) PTO2RingFlowControl {
    // === Cache Line 0: Written by Orchestrator, Read by Scheduler ===
    alignas(64) std::atomic<int32_t> current_task_index;  // Task ring head (next to allocate)

    // === Cache Line 1: Written by Scheduler, Read by Orchestrator (for back-pressure) ===
    alignas(64) std::atomic<int32_t> last_task_alive;  // Task ring tail (oldest active task)

    void init() {
        current_task_index.store(0, std::memory_order_relaxed);
        last_task_alive.store(0, std::memory_order_relaxed);
    }

    bool validate(PTO2SharedMemoryHandle *handle, int32_t ring_id) const;
};

static_assert(sizeof(PTO2RingFlowControl) == 128, "PTO2RingFlowControl must be exactly 2 cache lines (128B)");

struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Highest task_id such that every task with id in [0, completed_watermark]
    // has reached COMPLETED. Maintained at task-completion time. Used to gate
    // slot reclamation: a producer slot P is safe to retire when
    // completed_watermark >= P.last_consumer_local_id.
    alignas(64) std::atomic<int32_t> completed_watermark;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    // Compact contiguous array (one byte per slot) holding the polling-fast
    // "task X completed?" flag. 0 = pending, 1 = completed. Indexed by
    // local_id & task_window_mask. Writer: the task's completer at
    // on_mixed_task_complete; Resetter: orchestrator in prepare_task for the
    // newly-allocated slot. Reader: thread-0 fanin polling. Replaces a chain
    // of 128B-aligned slot_state pointer derefs with byte reads into a single
    // array — typically condenses 16 fanin checks into 1-2 cache lines.
    std::atomic<uint8_t> *completion_flags;

    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }

    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id) { return task_descriptors[local_id & task_window_mask]; }

    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }

    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id) { return task_payloads[local_id & task_window_mask]; }

    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id) { return slot_states[local_id & task_window_mask]; }
};

struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    // === PER-RING FLOW CONTROL + LAYOUT INFO (set once at init) ===
    PTO2SharedMemoryRingHeader rings[PTO2_MAX_RING_DEPTH];

    // === GLOBAL FIELDS ===
    std::atomic<int32_t> orchestrator_done;  // Flag: orchestration complete

    // Total shared memory size (for validation)
    uint64_t total_size;

    // Reserved legacy packed-output metadata. Keep these zero and retain them
    // only for shared-memory layout compatibility; host finalize ignores them.
    std::atomic<uint64_t> graph_output_ptr;
    std::atomic<uint64_t> graph_output_size;

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
    // distinguish device error TYPES (PTO2_STALL_DETAIL_*) without reading the
    // device log. The full stall snapshot stays in the device log / plog — only
    // this one class + a few locator ints cross the boundary.
    std::atomic<int32_t> sched_stall_detail;       // PTO2_STALL_DETAIL_* (NONE when no timeout)
    std::atomic<int32_t> sched_stall_completed;    // completed_tasks_ at timeout
    std::atomic<int32_t> sched_stall_total;        // total_tasks_ at timeout
    std::atomic<int32_t> sched_stall_cnt_running;  // tasks observed RUNNING (on a core)
    std::atomic<int32_t> sched_stall_cnt_ready;    // tasks fanin-satisfied but not dispatched
    std::atomic<int32_t> sched_stall_cnt_waiting;  // tasks still waiting on fanin
    std::atomic<int32_t> sched_stall_orch_done;    // orchestrator_done flag at timeout (0/1)
    std::atomic<int64_t> sched_stall_task_id;      // S1: stuck task_id (-1 if N/A)
    std::atomic<int32_t> sched_stall_core;         // S1: stuck core id (-1 if N/A)
};

static_assert(
    (sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE == 0) && (sizeof(PTO2SharedMemoryHeader) < 4096),
    "PTO2SharedMemoryHeader should be reasonably sized"
);

struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    PTO2SharedMemoryHeader *header;

    // Ownership flag
    bool is_owner;
    // True if this handle allocated the memory

    // === Static helpers ===

    static uint64_t calculate_size(uint64_t task_window_size);

    static uint64_t calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    static PTO2SharedMemoryHandle *create_and_init_default(DeviceArena &arena);

    // === Instance methods ===

    bool init(void *sm_base_arg, uint64_t sm_size_arg, uint64_t task_window_size, uint64_t heap_size);

    // Per-ring init adapter (upstream signature). Polling-side init treats
    // task_window_sizes[0] as canonical; rings 1..N inherit. heap_sizes[0] is
    // passed to the per-ring header init below.
    bool init_per_ring(
        void *sm_base_arg, uint64_t sm_size_arg, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
        const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
    );

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
        // Mirror setup_pointers_per_ring's per-ring stride exactly: completion_flags
        // follows slot_states for every ring. Omitting it made this address diverge
        // from where submit writes descriptors for rings >= 1, so the allocator read
        // unwritten memory and update_heap_tail underflowed the reclaim tail.
        p += PTO2_ALIGN_UP(task_window_sizes[r] * sizeof(std::atomic<uint8_t>), PTO2_ALIGN_SIZE);
    }
    return reinterpret_cast<PTO2TaskDescriptor *>(p);
}

}  // namespace pto2_sm_layout
