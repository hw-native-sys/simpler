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
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_TYPES_H
#define PTO_RUNTIME2_TYPES_H

#include <atomic>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "pto_types.h"
#include "pto_submit_types.h"
#include "pto2_dispatch_payload.h"

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
// Configuration Constants
// =============================================================================

// Task management
// NOTE: PTO2_TASK_WINDOW_SIZE is now the DEFAULT value only.
// Actual window size is passed at runtime to pto2_runtime_create_threaded_custom().
// Use pto2_task_slot(sched, task_id) for slot calculation.
#define PTO2_TASK_WINDOW_SIZE     65536   // Default task window size (power of 2)

// Memory pools
#define PTO2_HEAP_SIZE            (1024 * 1024 * 1024)  // 1GB default heap
#define PTO2_DEP_LIST_POOL_SIZE    65536    // Dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE   (65536)   // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 65536    // Power of 2 for fast hash

// Task parameters
#define PTO2_MAX_OUTPUTS          16      // Maximum outputs per task
#define PTO2_MAX_INPUTS           16      // Maximum inputs per task
#define PTO2_MAX_INOUTS           8       // Maximum in-out params per task

// Scope management
#define PTO2_MAX_SCOPE_DEPTH      64      // Maximum nesting depth
#define PTO2_SCOPE_TASKS_INIT_CAP 65536     // Initial capacity for scope task buffer

// Ready queue
#define PTO2_READY_QUEUE_SIZE     65536   // Per-shape queue size

// Memory alignment
#define PTO2_ALIGN_SIZE           64      // Cache line alignment
#define PTO2_PACKED_OUTPUT_ALIGN  1024    // Each output in packed buffer aligned to 1024B; gap is padding
#define PTO2_ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker type enumeration
 * Each worker type has its own ready queue for load balancing
 */
typedef enum {
    PTO2_WORKER_CUBE = 0,       // AICore CUBE unit (matrix ops)
    PTO2_WORKER_VECTOR = 1,     // AICore VECTOR unit (element-wise ops)
    PTO2_WORKER_AI_CPU = 2,     // AI_CPU (scalar ops, control flow)
    PTO2_WORKER_ACCELERATOR = 3,// Fixed-function accelerators (DMA, etc.)
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
#define PTO2_MAX_TENSOR_DIM   8

/**
 * Maximum depth of layout history for HBB overlap detection
 * Simple (contiguous) tensor has depth=1, non-contiguous has depth>1
 */
#define PTO2_MAX_LAYOUT_DEPTH     8

/**
 * Layout operation type for HBB
 */
typedef enum {
    PTO2_LAYOUT_VIEW = 0,         // View/slice: records bounding box
    PTO2_LAYOUT_RESHAPE = 1,      // Reshape: records new shape
    PTO2_LAYOUT_TRANSPOSE = 2     // Transpose: records permutation
} PTO2LayoutOpType;

/**
 * Layout operation entry for HBB
 * Each entry records one derivation step from the parent tensor.
 */
typedef struct {
    PTO2LayoutOpType type;
    union {
        struct {  // PTO2_LAYOUT_VIEW
            int64_t bbox_min;     // First byte accessed
            int64_t bbox_max;     // Last byte accessed
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
    PTO2_TENSOR_RAW = 0,           // Original raw tensor (owns storage)
    PTO2_TENSOR_VIEW = 1,          // view() - subset selection, shared storage
    PTO2_TENSOR_RESHAPE = 2,       // reshape() - shape change, shared storage
    PTO2_TENSOR_TRANSPOSE = 3,     // transpose() - dimension permute, shared storage
    PTO2_TENSOR_DEEP_VIEW = 4,     // deep_view() - copied subset, new storage
    PTO2_TENSOR_DEEP_RESHAPE = 5,  // deep_reshape() - copied reshape, new storage
    PTO2_TENSOR_DEEP_TRANSPOSE = 6 // deep_transpose() - copied transpose, new storage
} PTO2TensorExtractionType;

/**
 * Raw tensor (storage provider)
 *
 * The raw tensor owns the actual memory allocation.
 * Multiple logical tensors can share the same raw tensor (aliasing).
 */
typedef struct {
    void*    base_ptr;        // Base pointer of allocated memory
    int64_t  total_size;      // Total size in bytes
    int32_t  refcount;        // Number of logical tensors referencing this storage
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
    void*    raw_base;            // Pointer to raw tensor's base (for aliasing check)
    int64_t  raw_total_size;      // Total size of raw tensor in bytes

    // === Storage offset ===
    int64_t  storage_offset;      // Byte offset from raw_base to first element

    // === Shape and strides ===
    int64_t  shape[PTO2_MAX_TENSOR_DIM];    // Size in each dimension
    int64_t  strides[PTO2_MAX_TENSOR_DIM];  // Byte stride in each dimension
    int32_t  ndim;                          // Number of dimensions (0 = scalar)

    // === Precomputed bounding box (for fast overlap detection) ===
    int64_t  min_byte_offset;     // First byte accessed (relative to raw_base)
    int64_t  max_byte_offset;     // Last byte accessed (relative to raw_base)

    // === Element info ===
    int64_t  elem_size;           // Size of each element in bytes
    int64_t  numel;               // Total number of elements

    // === Extraction tracking ===
    PTO2TensorExtractionType extraction_type;  // How this tensor was created
    bool     is_contiguous;       // True if memory is contiguous (no gaps)
                                  // Equivalent to layout_depth == 1

    // === Layout history for HBB overlap detection ===
    int32_t  layout_depth;                           // Number of layout ops (1=simple)
    PTO2LayoutOp layout_ops[PTO2_MAX_LAYOUT_DEPTH];  // Derivation history

} PTO2LogicalTensor;

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
    PTO2TaskSlotState* slot_state;    // Consumer slot state (direct pointer)
    PTO2DepListEntry* next;           // next entry
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
    // Mixed-task identification
    int32_t mixed_task_id;            // Canonical mixed-task ID

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void*    packed_buffer_base;  // Start of packed buffer in GM Heap
    void*    packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Separated from PTO2TaskDescriptor to keep the descriptor cache-friendly
 * for the scheduler's hot completion path (~80 bytes vs ~2912 bytes).
 */
struct PTO2TaskPayload {
    PTO2DispatchPayload dispatch;  // function_bin_addr + args[], built in-place at dispatch time
    Tensor tensors[16];
    uint64_t scalar_value[16];
    bool is_tensor[16];
    int param_count{0};
    PTO2TaskSlotState* fanin_slot_states[PTO2_MAX_INPUTS]; // Producer slot states (cold path, used by on_task_release)
    int32_t fanin_actual_count{0};             // Actual fanin count (without the +1 redundance)
    int32_t dep_pool_mark{0};                  // Dep pool top after this task's submission (for reclamation)
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
    std::atomic<int32_t> fanout_lock;       // Per-task spinlock (0=unlocked, 1=locked)
    int32_t fanout_count;                    // 1 (owning scope) + number of consumers

    PTO2DepListEntry* fanout_head;           // Pointer to first fanout entry (nullptr = empty)

    // Task state (completion, consumed check, ready check)
    std::atomic<PTO2TaskState> task_state;   // PENDING/READY/RUNNING/COMPLETED/CONSUMED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;     // Dynamic: counts completed producers
    int32_t fanin_count;                      // Number of producer dependencies (set once)

    // Fanout refcount (accessed with fanout_count in check_and_handle_consumed)
    std::atomic<int32_t> fanout_refcount;  // Dynamic: counts released references

    PTO2TaskPayload* payload;

    PTO2TaskDescriptor* task;

    // Hot-path completion fields (moved from TaskDescriptor to avoid cross-struct access)
    uint8_t active_mask;                         // Bitmask of active subtask slots (set once)
    std::atomic<uint8_t> subtask_done_mask;      // Each subtask sets its done bit on completion
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
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("dmb sy" ::: "memory")
#elif defined(__x86_64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("mfence" ::: "memory")
#else
    #define PTO2_MEMORY_BARRIER()     __sync_synchronize()
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
static inline void pto2_fanout_lock(PTO2TaskSlotState& slot_state,
                                     uint64_t& atomic_count, uint64_t& wait_cycle) {
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
        if (slot_state.fanout_lock.compare_exchange_weak(expected, 1,
                                        std::memory_order_acquire, std::memory_order_relaxed)) {
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
        if (slot_state.fanout_lock.compare_exchange_weak(expected, 1,
                                        std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
    }
}

static inline void pto2_fanout_unlock(PTO2TaskSlotState& slot_state) {
    slot_state.fanout_lock.store(0, std::memory_order_release);
}

#endif // PTO_RUNTIME2_TYPES_H
