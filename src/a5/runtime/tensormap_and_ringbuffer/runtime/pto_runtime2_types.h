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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "pto2_dispatch_payload.h"
#include "pto_submit_types.h"
#include "pto_task_id.h"
#include "pto_types.h"

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

#ifndef PTO2_TENSORMAP_PROFILING
#define PTO2_TENSORMAP_PROFILING 0
#endif

#if PTO2_ORCH_PROFILING && !PTO2_PROFILING
#error "PTO2_ORCH_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_SCHED_PROFILING && !PTO2_PROFILING
#error "PTO2_SCHED_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_TENSORMAP_PROFILING && !PTO2_ORCH_PROFILING
#error "PTO2_TENSORMAP_PROFILING requires PTO2_ORCH_PROFILING=1"
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
#define PTO2_ERROR_INVALID_ARGS 5         // Arg construction error (invalid args)
#define PTO2_ERROR_DEPENDENCY_OVERFLOW 6  // Too many unique fanin dependencies for one task

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
#define PTO2_TENSORMAP_POOL_SIZE (65536)    // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096     // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64          // Maximum nesting depth
#define PTO2_SCOPE_TASKS_INIT_CAP 65536  // Initial capacity for scope task buffer

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Memory alignment
#define PTO2_ALIGN_SIZE 64             // Cache line alignment
#define PTO2_PACKED_OUTPUT_ALIGN 1024  // Each output in packed buffer aligned to 1024B; gap is padding
#define PTO2_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Fanin storage
#define PTO2_FANIN_INLINE_CAP 16

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks
#define PTO2_DEP_POOL_CLEANUP_INTERVAL 64   // Cleanup every N retired tasks

// get_tensor_data/set_tensor_data spin wait timeout in cycles.
// ~10s on hardware (1.5 GHz counter), ~10s on simulation (chrono-based).
constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_CYCLES = 15 * 1000 * 1000 * 1000ULL;

// =============================================================================
// Multi-Ring task_id Encoding
// =============================================================================

/**
 * TaskId: defined in pto_task_id.h (included above).
 */

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
// Logical Tensor (for view/reshape/transpose operations)
// =============================================================================

/**
 * Maximum dimensions supported for logical tensors
 */
#define PTO2_MAX_TENSOR_DIM 8

/**
 * Maximum depth of layout history for HBB overlap detection
 * Simple (contiguous) tensor has depth=1, non-contiguous has depth>1
 */
#define PTO2_MAX_LAYOUT_DEPTH 8

/**
 * Layout operation type for HBB
 */
typedef enum {
    PTO2_LAYOUT_VIEW = 0,      // View/slice: records bounding box
    PTO2_LAYOUT_RESHAPE = 1,   // Reshape: records new shape
    PTO2_LAYOUT_TRANSPOSE = 2  // Transpose: records permutation
} PTO2LayoutOpType;

/**
 * Layout operation entry for HBB
 * Each entry records one derivation step from the parent tensor.
 */
typedef struct {
    PTO2LayoutOpType type;
    union {
        struct {               // PTO2_LAYOUT_VIEW
            int64_t bbox_min;  // First byte accessed
            int64_t bbox_max;  // Last byte accessed
        } view;
        struct {  // PTO2_LAYOUT_RESHAPE
            int32_t ndim;
            int64_t shape[PTO2_MAX_TENSOR_DIM];
        } reshape;
        struct {  // PTO2_LAYOUT_TRANSPOSE
            int32_t ndim;
            int32_t perm[PTO2_MAX_TENSOR_DIM];
        } transpose;
    };
} PTO2LayoutOp;

/**
 * Tensor extraction type (for tracking how tensor was created)
 */
typedef enum {
    PTO2_TENSOR_RAW = 0,            // Original raw tensor (owns storage)
    PTO2_TENSOR_VIEW = 1,           // view() - subset selection, shared storage
    PTO2_TENSOR_RESHAPE = 2,        // reshape() - shape change, shared storage
    PTO2_TENSOR_TRANSPOSE = 3,      // transpose() - dimension permute, shared storage
    PTO2_TENSOR_DEEP_VIEW = 4,      // deep_view() - copied subset, new storage
    PTO2_TENSOR_DEEP_RESHAPE = 5,   // deep_reshape() - copied reshape, new storage
    PTO2_TENSOR_DEEP_TRANSPOSE = 6  // deep_transpose() - copied transpose, new storage
} PTO2TensorExtractionType;

/**
 * Raw tensor (storage provider)
 *
 * The raw tensor owns the actual memory allocation.
 * Multiple logical tensors can share the same raw tensor (aliasing).
 */
typedef struct {
    void *base_ptr;      // Base pointer of allocated memory
    int64_t total_size;  // Total size in bytes
    int32_t refcount;    // Number of logical tensors referencing this storage
                         // (for memory management, 0 = can be freed)
} PTO2RawTensor;

/**
 * Logical tensor structure
 *
 * A "view" into raw tensor storage with specific layout.
 * Supports multi-dimensional tensors with strides (for view/reshape/transpose).
 *
 * Memory footprint is determined by:
 *   - storage_offset: byte offset from raw_base to first element
 *   - shape[d]: number of elements in dimension d
 *   - strides[d]: byte offset between consecutive elements in dimension d
 *
 * For element at indices [i0, i1, ..., i_{n-1}]:
 *   byte_offset = storage_offset + sum(i_d * strides[d])
 *
 * Examples:
 *   - Contiguous row-major (3,4): shape=[3,4], strides=[4*elem_size, elem_size]
 *   - Transposed (4,3): shape=[4,3], strides=[elem_size, 4*elem_size]
 *   - Sliced [1:3, 1:3]: offset adjusted, shape=[2,2], strides unchanged
 */
typedef struct {
    // === Raw tensor reference (shared storage) ===
    void *raw_base;          // Pointer to raw tensor's base (for aliasing check)
    int64_t raw_total_size;  // Total size of raw tensor in bytes

    // === Storage offset ===
    int64_t storage_offset;  // Byte offset from raw_base to first element

    // === Shape and strides ===
    int64_t shape[PTO2_MAX_TENSOR_DIM];    // Size in each dimension
    int64_t strides[PTO2_MAX_TENSOR_DIM];  // Byte stride in each dimension
    int32_t ndim;                          // Number of dimensions (0 = scalar)

    // === Precomputed bounding box (for fast overlap detection) ===
    int64_t min_byte_offset;  // First byte accessed (relative to raw_base)
    int64_t max_byte_offset;  // Last byte accessed (relative to raw_base)

    // === Element info ===
    int64_t elem_size;  // Size of each element in bytes
    int64_t numel;      // Total number of elements

    // === Extraction tracking ===
    PTO2TensorExtractionType extraction_type;  // How this tensor was created
    bool is_contiguous;                        // True if memory is contiguous (no gaps)
                                               // Equivalent to layout_depth == 1

    // === Layout history for HBB overlap detection ===
    int32_t layout_depth;                            // Number of layout ops (1=simple)
    PTO2LayoutOp layout_ops[PTO2_MAX_LAYOUT_DEPTH];  // Derivation history
} PTO2LogicalTensor;

// =============================================================================
// Dependency List Entry
// =============================================================================

struct PTO2TaskSlotState;  // Forward declaration
struct PTO2FaninPool;      // Forward declaration
struct PTO2FaninSpillEntry {
    PTO2TaskSlotState *slot_state;
};
static_assert(sizeof(PTO2FaninSpillEntry) == sizeof(PTO2TaskSlotState *));

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer.
 */
struct PTO2DepListEntry {
    PTO2TaskSlotState *slot_state;  // Consumer slot state (direct pointer)
    PTO2DepListEntry *next;         // next entry
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
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + inline fanin packed in the first 3 cache lines, followed
 * by bulk tensor and scalar data. Small fanins stay fully inline; larger
 * fanins spill into a per-ring ring buffer slice.
 */
struct PTO2TaskPayload {
    // === Cache lines 0-2 (192B) — metadata ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_actual_count{0};  // Actual fanin count (without the +1 redundance)
    int32_t fanin_spill_start{0};   // Linear start index in fanin spill pool (0 = no spill)
    PTO2FaninPool *fanin_spill_pool{nullptr};
    PTO2TaskSlotState *fanin_inline_slot_states[PTO2_FANIN_INLINE_CAP];
    // === Cache lines 3-34 (2048B) — tensors (alignas(64) forces alignment) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 35-50 (1024B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    // Layout verification (size checks that don't need offsetof).
    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == 1024, "scalar region must be 1024B (16 cache lines)");

    /**
     * Initialize payload: copy tensors, store scalars.
     *
     * For each param slot, the tensor source is determined by TensorArgType:
     * - OUTPUT -> use materialized_outputs.output_ptr(out_idx++)
     * - INPUT / INOUT -> use refs[i].tensor
     *
     * @param args                Task arguments (tensors + scalars)
     * @param materialized_outputs  Materialized output tensors (from TensorCreateInfo path)
     */
    void
    init(const Arg &args, TaskOutputTensors &result, void *base_addr, uint64_t offsets[], uint64_t buffer_sizes[]) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) {
                tensors[i].copy(*args.tensor(i).ptr);
            } else {
                tensors[i].init_from_create_info(
                    *args.tensor(i).create_info,
                    reinterpret_cast<void *>(reinterpret_cast<char *>(base_addr) + offsets[i]), buffer_sizes[i]
                );
                result.materialize_output(tensors[i]);
            }
            tensors[i].update_start_offset();
        }
        // Round up to cache line boundary. Both arrays are 1024B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalars(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
static_assert(
    offsetof(PTO2TaskPayload, fanin_inline_slot_states) == 24,
    "inline fanin array must follow spill metadata"
);
static_assert(offsetof(PTO2TaskPayload, tensors) == 192, "tensors must start at byte 192 (cache line 3)");
static_assert(
    offsetof(PTO2TaskPayload, scalars) == 192 + MAX_TENSOR_ARGS * sizeof(Tensor),
    "scalars must immediately follow tensors"
);

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

    PTO2DepListEntry *fanout_head;  // Pointer to first fanout entry (nullptr = empty)

    // Task state (completion, consumed check, ready check)
    std::atomic<PTO2TaskState> task_state;  // PENDING/READY/RUNNING/COMPLETED/CONSUMED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;  // Dynamic: counts completed producers
    int32_t fanin_count;                  // Number of producer dependencies (set once)

    // Fanout refcount (accessed with fanout_count in check_and_handle_consumed)
    std::atomic<int32_t> fanout_refcount;  // Dynamic: counts released references

    PTO2TaskPayload *payload;

    PTO2TaskDescriptor *task;

    // Hot-path completion fields (moved from TaskDescriptor to avoid cross-struct access)
    uint8_t active_mask;                     // Bitmask of active subtask slots (set once)
    std::atomic<uint8_t> subtask_done_mask;  // Deprecated: superseded by completed_subtasks
    uint8_t ring_id;                         // Ring layer this task belongs to (for per-ring reclamation)
    int32_t dep_pool_mark{0};  // Dep pool top after this task's submission (orchestrator-only, local memory)

    // SPMD multi-block (occupies the 8 tail bytes previously implicit padding)
    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = block_num * popcount(active_mask)
    int16_t block_num{1};                        // Total logical blocks (set by orchestrator)
    int16_t next_block_idx{0};                   // Next block to dispatch (scheduler state)
};

static_assert(sizeof(PTO2TaskSlotState) == 64);

// =============================================================================
// Cycle Cost Function Type
// =============================================================================

/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*PTO2CycleCostFunc)(void **args, int32_t num_args);

// =============================================================================
// InCore Function Type
// =============================================================================

/**
 * InCore function signature
 * All InCore functions must match this signature
 */
typedef void (*PTO2InCoreFunc)(void **args, int32_t num_args);

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
#include "spin_hint.h"
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
static inline void pto2_fanout_lock(PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
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
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
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

static inline void pto2_fanout_lock(PTO2TaskSlotState &slot_state) {
    for (;;) {
        while (slot_state.fanout_lock.load(std::memory_order_acquire) != 0) {
            SPIN_WAIT_HINT();
        }
        int32_t expected = 0;
        if (slot_state.fanout_lock.compare_exchange_weak(
                expected, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            return;
        }
    }
}

static inline void pto2_fanout_unlock(PTO2TaskSlotState &slot_state) {
    slot_state.fanout_lock.store(0, std::memory_order_release);
}

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
