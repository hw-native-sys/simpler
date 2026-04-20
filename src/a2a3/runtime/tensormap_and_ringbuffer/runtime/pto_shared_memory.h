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
 *   +---------------------------+
 *   | Ring 1: TaskDescriptor[]  |
 *   | Ring 1: TaskPayload[]     |
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

#ifndef PTO_SHARED_MEMORY_H
#define PTO_SHARED_MEMORY_H

#include "pto_runtime2_types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * Groups flow-control and layout info for a single ring to avoid parallel arrays.
 */
struct PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;
    uint64_t task_window_size;
    uint64_t heap_size;
    uint64_t task_descriptors_offset;  // Offset from SM base, in bytes
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
    std::atomic<int32_t> sched_error_bitmap;  // Bit X set = thread X had error
    std::atomic<int32_t> sched_error_code;    // Last scheduler error code (last-writer-wins)
    std::atomic<int32_t> sched_error_thread;  // Thread index of last error writer
};

static_assert(
    sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE == 0,
    "PTO2SharedMemoryHeader must be aligned to cache line (PTO2_ALIGN_SIZE)"
);

// =============================================================================
// Shared Memory Handle
// =============================================================================

/**
 * Handle for shared memory access
 * Provides both Orchestrator and Scheduler views of the same memory
 */
struct PTO2SharedMemoryHandle {
    void *sm_base;     // Base address of shared memory
    uint64_t sm_size;  // Total size of shared memory

    // Quick pointers into shared memory regions (per-ring)
    PTO2SharedMemoryHeader *header;
    PTO2TaskDescriptor *task_descriptors[PTO2_MAX_RING_DEPTH];
    PTO2TaskPayload *task_payloads[PTO2_MAX_RING_DEPTH];

    // Ownership flag
    bool is_owner;  // True if this handle allocated the memory
};

// =============================================================================
// Shared Memory API
// =============================================================================

/**
 * Calculate required shared memory size
 *
 * @param task_window_size  Number of task slots per ring
 * @return Total bytes required
 */
uint64_t pto2_sm_calculate_size(uint64_t task_window_size);

/**
 * Calculate required shared memory size for per-ring task windows.
 *
 * @param task_window_sizes  Array of window sizes per ring
 * @return Total bytes required
 */
uint64_t pto2_sm_calculate_size_per_ring(const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

/**
 * Create shared memory for Orchestrator and Scheduler
 *
 * @param task_window_size  Number of task slots per ring
 * @param heap_size         Heap size per ring for output buffers
 * @return Handle with both views, or NULL on failure
 */
PTO2SharedMemoryHandle *pto2_sm_create(uint64_t task_window_size, uint64_t heap_size);

/**
 * Create shared memory with default sizes
 */
PTO2SharedMemoryHandle *pto2_sm_create_default(void);

/**
 * Wrap an existing buffer as shared memory (e.g. device GM buffer).
 * Caller owns the buffer; handle will not free sm_base.
 *
 * @param sm_base            Base address of pre-allocated buffer
 * @param sm_size            Total size in bytes
 * @param task_window_size   Number of task slots per ring (must match buffer layout)
 * @param heap_size          Heap size per ring (for layout; buffer has no heap region)
 * @return Handle, or NULL on failure
 */
PTO2SharedMemoryHandle *
pto2_sm_create_from_buffer(void *sm_base, uint64_t sm_size, uint64_t task_window_size, uint64_t heap_size);

/**
 * Destroy shared memory and free resources
 */
void pto2_sm_destroy(PTO2SharedMemoryHandle *handle);

/**
 * Initialize shared memory header with layout information
 * Called after memory is allocated
 */
void pto2_sm_init_header(PTO2SharedMemoryHandle *handle, uint64_t task_window_size, uint64_t heap_size);

/**
 * Initialize shared memory header with per-ring layout information.
 */
void pto2_sm_init_header_per_ring(
    PTO2SharedMemoryHandle *handle, const uint64_t task_window_sizes[PTO2_MAX_RING_DEPTH],
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print shared memory layout info
 */
void pto2_sm_print_layout(PTO2SharedMemoryHandle *handle);

/**
 * Validate shared memory integrity
 * @return true if valid, false if corrupted
 */
bool pto2_sm_validate(PTO2SharedMemoryHandle *handle);

#ifdef __cplusplus
}
#endif

#endif  // PTO_SHARED_MEMORY_H
