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
 * PTO Runtime2 - Core Type Definitions
 *
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "pto_submit_types.h"  // NOLINT(build/include_subdir)
#include "pto_types.h"         // NOLINT(build/include_subdir)

// =============================================================================
// Profiling Configuration
// =============================================================================

#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

#ifndef PTO2_ORCH_PROFILING
#define PTO2_ORCH_PROFILING 0
#endif

#ifndef PTO2_SCHED_PROFILING
#define PTO2_SCHED_PROFILING 0
#endif

#if PTO2_ORCH_PROFILING && !PTO2_PROFILING
#error "PTO2_ORCH_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_SCHED_PROFILING && !PTO2_PROFILING
#error "PTO2_SCHED_PROFILING requires PTO2_PROFILING=1"
#endif

// =============================================================================
// AICPU Error Codes (written to shared memory for Host-side diagnosis)
// =============================================================================

// Orchestrator errors (1-99): detected in orchestrator thread
#define PTO2_ERROR_NONE 0
#define PTO2_ERROR_SCOPE_DEADLOCK 1
#define PTO2_ERROR_HEAP_RING_DEADLOCK 2
#define PTO2_ERROR_FLOW_CONTROL_DEADLOCK 3
#define PTO2_ERROR_DEP_POOL_OVERFLOW 4
#define PTO2_ERROR_INVALID_ARGS 5  // Arg construction error (invalid args)

// Scheduler errors (100+): detected in scheduler threads
#define PTO2_ERROR_SCHEDULER_TIMEOUT 100

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
// NOTE: PTO2_TASK_WINDOW_SIZE is now a per-ring default value.
// Actual window size is passed at runtime to pto2_runtime_create_threaded_custom().
// Use pto2_task_slot(sched, task_id) for slot calculation.
#define PTO2_TASK_WINDOW_SIZE 16384  // Default per-ring task window size (power of 2)

// Multi-ring: number of independent ring layers (HeapRing + TaskRing + DepPool per layer)
// Scope depth maps to ring index via: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
#define PTO2_MAX_RING_DEPTH 4

// Memory pools (per-ring defaults; total = value × PTO2_MAX_RING_DEPTH)
#define PTO2_HEAP_SIZE (256 * 1024 * 1024)  // 256MB per ring (1GB total)
#define PTO2_DEP_LIST_POOL_SIZE 16384       // Per-ring dependency list pool entries

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64          // Maximum nesting depth
#define PTO2_SCOPE_TASKS_INIT_CAP 65536  // Initial capacity for scope task buffer

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Memory alignment
#define PTO2_ALIGN_SIZE 64             // Cache line alignment
#define PTO2_PACKED_OUTPUT_ALIGN 1024  // Each output in packed buffer aligned to 1024B; gap is padding
#define PTO2_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Dep pool cleanup interval
#define PTO2_DEP_POOL_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks

// =============================================================================
// Multi-Ring task_id Encoding
// =============================================================================

/**
 * TaskId: 64-bit encoding used across Runtime2.
 *
 * raw encoding: (ring_id << 32) | local_id
 *
 * ring_id:  which ring layer (0..PTO2_MAX_RING_DEPTH-1)
 * local_id: per-ring monotonic counter
 */
struct PTO2TaskId {
    uint64_t raw;

    constexpr PTO2TaskId() : raw(0) {}
    constexpr explicit PTO2TaskId(uint64_t v) : raw(v) {}

    constexpr uint8_t ring() const { return static_cast<uint8_t>(raw >> 32); }
    constexpr uint32_t local() const { return static_cast<uint32_t>(raw & 0xFFFFFFFFu); }

    constexpr bool operator==(const PTO2TaskId& other) const { return raw == other.raw; }
    constexpr bool operator!=(const PTO2TaskId& other) const { return raw != other.raw; }
};

static_assert(sizeof(PTO2TaskId) == 8, "PTO2TaskId must stay 8 bytes (shared memory ABI)");

static inline PTO2TaskId pto2_make_task_id(uint8_t ring_id, uint32_t local_id) {
    return PTO2TaskId{(static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(local_id)};
}

static inline uint8_t pto2_task_id_ring(PTO2TaskId task_id) { return task_id.ring(); }

static inline uint32_t pto2_task_id_local(PTO2TaskId task_id) { return task_id.local(); }

static inline uint64_t pto2_task_id_raw(PTO2TaskId task_id) { return task_id.raw; }

/**
 * SubmitResult — return value from pto2_submit_mixed_task.
 * Bundles the task_id (for explicit dependencies) and the materialized
 * output tensors (for referencing runtime-allocated outputs).
 */
struct SubmitResult {
    PTO2TaskId task_id;
    TaskOutputTensors outputs;
};

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker type enumeration
 * Each worker type has its own ready queue for load balancing
 */
typedef enum {
    PTO2_WORKER_CUBE = 0,         // AICore CUBE unit (matrix ops)
    PTO2_WORKER_VECTOR = 1,       // AICore VECTOR unit (element-wise ops)
    PTO2_WORKER_AI_CPU = 2,       // AI_CPU (scalar ops, control flow)
    PTO2_WORKER_ACCELERATOR = 3,  // Fixed-function accelerators (DMA, etc.)
    PTO2_NUM_WORKER_TYPES = 4
} PTO2WorkerType;

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 *
 * State transitions:
 *   PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
 *
 * Conditions:
 *   PENDING->READY:     fanin_refcount == fanin_count
 *   COMPLETED->CONSUMED: fanout_refcount == fanout_count && state == COMPLETED
 */
typedef enum {
    PTO2_TASK_PENDING = 0,    // Waiting for dependencies (fanin_refcount < fanin_count)
    PTO2_TASK_READY = 1,      // All dependencies satisfied, waiting in ready queue
    PTO2_TASK_RUNNING = 2,    // Currently executing on a worker
    PTO2_TASK_COMPLETED = 3,  // Execution finished, output may still be in use
    PTO2_TASK_CONSUMED = 4    // Output fully consumed, buffers can be released
} PTO2TaskState;

// =============================================================================
// Dependency List Entry
// =============================================================================

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer
 *
 * Used for both fanin_list and fanout_list
 */
struct PTO2TaskSlotState;  // Forward declaration
struct PTO2DepListEntry {
    PTO2TaskSlotState* slot_state;  // Consumer slot state (direct pointer)
    PTO2DepListEntry* next;         // next entry
};

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in PTO2TaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct PTO2TaskDescriptor {
    // Mixed-task identification (encodes ring_id in upper 32 bits)
    PTO2TaskId task_id;  // raw: (ring_id << 32) | local_id

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void* packed_buffer_base;  // Start of packed buffer in GM Heap
    void* packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata (counts, fanin pointers) packed in the first 3 cache lines,
 * followed by bulk tensor and scalar data. This gives sequential write access
 * during orchestration and groups scheduler-hot fields (fanin_actual_count +
 * fanin_slot_states) together for on_task_release.
 */
struct PTO2TaskPayload {
    // === Cache line 0 (64B) — metadata ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_actual_count{0};  // Actual fanin count (without the +1 redundance)
    int32_t _reserved{0};           // Reserved (dep_pool_mark moved to SlotState for local access)
    PTO2TaskSlotState* fanin_slot_states[PTO2_MAX_INPUTS];  // Producer slot states (used by on_task_release)
    // === Cache lines 3-34 (2048B) — tensors (alignas(64) forces alignment) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 35-50 (1024B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    void init(const Arg& args, const TaskOutputTensors& materialized_outputs) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();
        int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            const Tensor* src;
            if (args.tag(i) == TensorArgType::OUTPUT) {
                src = materialized_outputs.output_ptr(out_idx++);
            } else {
                src = args.tensor(i).ptr;
            }
            tensors[i].copy(*src);
        }
        // Round up to cache line boundary. Both arrays are 1024B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalar_data(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));
    }
};

/**
 * Per-task slot scheduling state (scheduler-private, NOT in shared memory)
 *
 * Consolidates all hot-path scheduling fields into a single cache-friendly
 * structure (32 bytes = half a cache line). Accessing any field of a task's
 * slot state brings all related fields into the same cache line.
 *
 * Concurrency notes:
 * - fanout_head, fanout_count protected by fanout_lock (per-task spinlock)
 * - fanin_count set once at submission, read-only after (hot path for ready check)
 * - task_state, fanin_refcount, fanout_refcount updated atomically
 */
struct alignas(64) PTO2TaskSlotState {
    // Fanout lock + list (accessed together under lock in on_task_complete)
    std::atomic<int32_t> fanout_lock;  // Per-task spinlock (0=unlocked, 1=locked)
    int32_t fanout_count;              // 1 (owning scope) + number of consumers

    PTO2DepListEntry* fanout_head;  // Pointer to first fanout entry (nullptr = empty)

    // Task state (completion, consumed check, ready check)
    std::atomic<PTO2TaskState> task_state;  // PENDING/READY/RUNNING/COMPLETED/CONSUMED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;  // Dynamic: counts completed producers
    int32_t fanin_count;                  // Number of producer dependencies (set once)

    // Fanout refcount (accessed with fanout_count in check_and_handle_consumed)
    std::atomic<int32_t> fanout_refcount;  // Dynamic: counts released references

    PTO2TaskPayload* payload;

    PTO2TaskDescriptor* task;

    // Hot-path completion fields (moved from TaskDescriptor to avoid cross-struct access)
    uint8_t active_mask;                     // Bitmask of active subtask slots (set once)
    std::atomic<uint8_t> subtask_done_mask;  // Each subtask sets its done bit on completion
    uint8_t ring_id;                         // Ring layer this task belongs to (for per-ring reclamation)
    int32_t dep_pool_mark{0};  // Dep pool top after this task's submission (orchestrator-only, local memory)
};

static_assert(sizeof(PTO2TaskSlotState) == 64);

// =============================================================================
// Cycle Cost Function Type
// =============================================================================

/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*PTO2CycleCostFunc)(void** args, int32_t num_args);

// =============================================================================
// InCore Function Type
// =============================================================================

/**
 * InCore function signature
 * All InCore functions must match this signature
 */
typedef void (*PTO2InCoreFunc)(void** args, int32_t num_args);

// =============================================================================
// Utility Macros
// =============================================================================

/**
 * Memory barrier macros for different architectures
 */
#if defined(__aarch64__)
#define PTO2_MEMORY_BARRIER() __asm__ __volatile__("dmb sy" ::: "memory")
#elif defined(__x86_64__)
#define PTO2_MEMORY_BARRIER() __asm__ __volatile__("mfence" ::: "memory")
#else
#define PTO2_MEMORY_BARRIER() __sync_synchronize()
#endif

// Spin-wait hint for AICPU threads.  On real hardware the AICPU has dedicated
// ARM A55 cores — no OS yield is needed, so the hint is a no-op.  In simulation
// all threads share host CPU cores, so we yield to prevent starvation.
// This header is also compiled into the Host .so (for struct definitions only),
// where the hint is never called — the fallback no-op keeps Host builds clean.
#if __has_include("spin_hint.h")
#include "spin_hint.h"  // NOLINT(build/include_subdir)
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif

// =============================================================================
// Per-task fanout spinlock helpers
//
// Used by BOTH the orchestrator (pto_orchestrator.cpp) and the scheduler
// (aicpu_executor.cpp). Placing them here ensures both translation units use
// identical acquire/release semantics.
//
// The fanout_lock MUST be held whenever reading or writing fanout_head /
// fanout_count, because the orchestrator adds consumers concurrently with the
// scheduler traversing the list after task completion.
// =============================================================================

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
#include "aicpu/device_time.h"
#endif

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
static inline void pto2_fanout_lock(PTO2TaskSlotState& slot_state, uint64_t& atomic_count, uint64_t& wait_cycle) {
    uint64_t t0 = get_sys_cnt_aicpu();
    bool contended = false;
    uint32_t atomic_ops = 0;

    for (;;) {
        while (slot_state.fanout_lock.load(std::memory_order_acquire) != 0) {
            contended = true;
            atomic_ops++;  // each load = 1 atomic
            SPIN_WAIT_HINT();
        }
        int32_t expected = 0;
        if (slot_state.fanout_lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            atomic_ops++;  // successful CAS = 1 atomic
            atomic_count += atomic_ops;
            if (contended) {
                wait_cycle += (get_sys_cnt_aicpu() - t0);
            }
            return;
        }
        contended = true;
        atomic_ops++;  // failed CAS = 1 atomic
    }
}
#endif

static inline void pto2_fanout_lock(PTO2TaskSlotState& slot_state) {
    for (;;) {
        while (slot_state.fanout_lock.load(std::memory_order_acquire) != 0) {
            SPIN_WAIT_HINT();
        }
        int32_t expected = 0;
        if (slot_state.fanout_lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
    }
}

static inline void pto2_fanout_unlock(PTO2TaskSlotState& slot_state) {
    slot_state.fanout_lock.store(0, std::memory_order_release);
}

#endif  // SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
