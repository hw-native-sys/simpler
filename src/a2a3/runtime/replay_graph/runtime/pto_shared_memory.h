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
 * Memory Layout:
 *   +---------------------------+
 *   | SharedMemoryHeader        |  (task publication + final completion)
 *   +---------------------------+
 *   | TaskDescriptor[]          |
 *   | TaskPayload[]             |
 *   | TaskSlotState[]           |
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

struct alignas(64) PTO2FlowControl {
    // Monotonic logical task count published as graph construction advances.
    alignas(64) std::atomic<int32_t> task_count;

    void init() { task_count.store(0, std::memory_order_relaxed); }
    bool validate(PTO2SharedMemoryHandle *handle) const;
};

static_assert(sizeof(PTO2FlowControl) == 64, "PTO2FlowControl must be exactly one cache line");

/**
 * Shared memory header structure
 *
 * Contains logical task publication state and orchestration/scheduler handoff.
 */
struct alignas(PTO2_ALIGN_SIZE) PTO2SharedMemoryHeader {
    PTO2FlowControl fc;

    uint64_t task_window_size;
    int32_t task_window_mask;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;

    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;
    int32_t *task_slot_map;

    int32_t get_slot_by_task_id(int32_t task_id) {
        int32_t map_slot = task_id & task_window_mask;
        return task_slot_map ? task_slot_map[map_slot] : map_slot;
    }
    PTO2TaskDescriptor &get_task_by_slot(int32_t slot) { return task_descriptors[slot]; }
    PTO2TaskDescriptor &get_task_by_task_id(int32_t task_id) { return task_descriptors[get_slot_by_task_id(task_id)]; }
    PTO2TaskPayload &get_payload_by_slot(int32_t slot) { return task_payloads[slot]; }
    PTO2TaskPayload &get_payload_by_task_id(int32_t task_id) { return task_payloads[get_slot_by_task_id(task_id)]; }
    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }
    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t task_id) { return slot_states[get_slot_by_task_id(task_id)]; }
    PTO2TaskSlotState *find_live_slot_state(PTO2TaskId task_id) {
        PTO2TaskSlotState &slot = get_slot_state_by_task_id(static_cast<int32_t>(task_id.local()));
        return slot.task != nullptr && slot.task->task_id == task_id ? &slot : nullptr;
    }

    // === GLOBAL FIELDS ===
    alignas(64) std::atomic<int32_t> orchestrator_done;  // final task-stream completion

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

static_assert(offsetof(PTO2SharedMemoryHeader, fc) == 0);
static_assert(offsetof(PTO2SharedMemoryHeader, task_descriptors_offset) == 88);
static_assert(offsetof(PTO2SharedMemoryHeader, orchestrator_done) == 128);
static_assert(offsetof(PTO2SharedMemoryHeader, orch_error_code) == 160);

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
    void setup_pointers(uint64_t task_window_size);
};

// =============================================================================
// SM Device Layout Helpers
// =============================================================================
//
// When the host pre-builds a runtime-arena image, it needs the device-side
// addresses of several SM sub-fields (flow-control counter,
// task_descriptors array, orch_error_code) so it can wire them into the
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

inline std::atomic<int32_t> *task_count_addr(void *sm_dev_base) noexcept {
    return reinterpret_cast<std::atomic<int32_t> *>(
        static_cast<char *>(sm_dev_base) + offsetof(PTO2SharedMemoryHeader, fc) + offsetof(PTO2FlowControl, task_count)
    );
}

struct PTO2SegmentOffsets {
    uint64_t descriptors;
    uint64_t payloads;
    uint64_t slot_states;
    uint64_t task_slot_map;
    uint64_t end;
};

inline PTO2SegmentOffsets segment_offsets(uint64_t task_window_size) noexcept {
    uint64_t off = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    PTO2SegmentOffsets result{};
    result.descriptors = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskDescriptor), PTO2_ALIGN_SIZE);
    result.payloads = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskPayload), PTO2_ALIGN_SIZE);
    result.slot_states = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(PTO2TaskSlotState), PTO2_ALIGN_SIZE);
    result.task_slot_map = off;
    off += PTO2_ALIGN_UP(task_window_size * sizeof(int32_t), PTO2_ALIGN_SIZE);
    result.end = off;
    return result;
}

inline PTO2TaskDescriptor *task_descriptors_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<PTO2TaskDescriptor *>(
        static_cast<char *>(sm_dev_base) + segment_offsets(task_window_size).descriptors
    );
}

inline PTO2TaskSlotState *slot_states_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<PTO2TaskSlotState *>(
        static_cast<char *>(sm_dev_base) + segment_offsets(task_window_size).slot_states
    );
}

inline int32_t *task_slot_map_addr(void *sm_dev_base, uint64_t task_window_size) noexcept {
    return reinterpret_cast<int32_t *>(
        static_cast<char *>(sm_dev_base) + segment_offsets(task_window_size).task_slot_map
    );
}

}  // namespace pto2_sm_layout
