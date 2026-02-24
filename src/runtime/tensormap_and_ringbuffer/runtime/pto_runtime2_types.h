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

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "pto_types.h"

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
#define PTO2_SCOPE_TASKS_INIT_CAP 256     // Initial capacity for scope task buffer

// Ready queue
#define PTO2_READY_QUEUE_SIZE     65536   // Per-worker-type queue size (16x larger to avoid queue full)

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
typedef struct {
    int32_t task_id;          // The dependent/dependency task ID
    int32_t next_offset;      // Offset to next entry (0 = end of list)
} PTO2DepListEntry;

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure
 * 
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains both static info (set at submission) and dynamic state.
 * 
 * Concurrency notes:
 * - fanout_head, fanout_count protected by fanout_lock (per-task spinlock)
 * - fanin_head, fanin_count set once at submission, read-only after
 * - Other fields set by Orchestrator, read by Scheduler
 */
struct PTO2TaskDescriptor {
    // Task identification
    int32_t task_id;              // Unique task identifier (absolute, not wrapped)
    int32_t kernel_id;            // InCore function to execute
    int32_t worker_type;          // Target: CUBE, VECTOR, AI_CPU, ACCELERATOR
    // Dependency lists (linked list heads - offsets into DepListPool)
    // Fanin: producers this task depends on (set once at submission)
    int32_t fanin_head;           // Offset to first fanin entry (0 = empty)
    int32_t fanin_count;          // Number of producer dependencies
    
    // Fanout: consumers that depend on this task (grows as consumers submit)
    // PROTECTED BY fanout_lock
    volatile int32_t fanout_lock; // Per-task spinlock (0=unlocked, 1=locked)
    volatile int32_t fanout_head; // Offset to first fanout entry (0 = empty)
    volatile int32_t fanout_count;// 1 (owning scope) + number of consumers
    
    // Packed output buffer (all outputs packed into single contiguous buffer)
    void*    packed_buffer_base;  // Start of packed buffer in GM Heap
    void*    packed_buffer_end;   // End of packed buffer (for heap reclamation)
    int32_t  output_index[PTO2_MAX_OUTPUTS]; // Offset of each output in params;
    int32_t  num_outputs;         // Number of output buffers

    // Status flags
    bool     is_active;           // Task slot is in use


    PTOParam params[16];
    Tensor tensor_copies[16];  // Owned tensor data (params[i].tensor points here)
    int param_count{0};
};

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
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#elif defined(__x86_64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("mfence" ::: "memory")
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#else
    #define PTO2_MEMORY_BARRIER()     __sync_synchronize()
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#endif

/**
 * Pause instruction for spin-wait loops
 * Include sched_yield() to prevent CPU starvation of other threads
 */
#include <sched.h>
#if defined(__aarch64__)
    #define PTO2_SPIN_PAUSE()         do { __asm__ __volatile__("yield" ::: "memory"); sched_yield(); } while(0)
    #define PTO2_SPIN_PAUSE_LIGHT()   __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__)
    #define PTO2_SPIN_PAUSE()         do { __builtin_ia32_pause(); sched_yield(); } while(0)
    #define PTO2_SPIN_PAUSE_LIGHT()   __builtin_ia32_pause()
#else
    #define PTO2_SPIN_PAUSE()         sched_yield()
    #define PTO2_SPIN_PAUSE_LIGHT()   ((void)0)
#endif

/**
 * Atomic compare-and-swap
 */
#define PTO2_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, expected, desired, false, \
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)

/**
 * Atomic fetch-and-add
 */
#define PTO2_FETCH_ADD(ptr, val) \
    __atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL)

/**
 * Atomic exchange
 */
#define PTO2_EXCHANGE(ptr, val) \
    __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)

#endif // PTO_RUNTIME2_TYPES_H
