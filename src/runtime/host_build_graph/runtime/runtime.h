/**
 * Runtime Class - Task Dependency Runtime Management
 *
 * This is a simplified, standalone runtime class for managing task
 * dependencies. Tasks are stored in a fixed-size array with compile-time
 * configurable bounds. Each task has:
 * - Unique ID (array index)
 * - Arguments (uint64_t array)
 * - Fanin (predecessor count)
 * - Fanout (array of successor task IDs)
 *
 * The runtime maintains a ready queue for tasks with fanin == 0.
 *
 * Based on patterns from pto_runtime.h/c but simplified for educational
 * and lightweight scheduling use cases.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <atomic>

// =============================================================================
// Configuration Macros
// =============================================================================

#ifndef RUNTIME_MAX_TASKS
#define RUNTIME_MAX_TASKS 1024
#endif

#ifndef RUNTIME_MAX_ARGS
#define RUNTIME_MAX_ARGS 16
#endif

#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 512
#endif

#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72  // 24 AIC + 48 AIV cores
#endif

#ifndef RUNTIME_MAX_TENSOR_PAIRS
#define RUNTIME_MAX_TENSOR_PAIRS 64
#endif

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Handshake Structure - Shared between Host, AICPU, and AICore
 *
 * This structure facilitates communication and synchronization between
 * AICPU and AICore during task execution.
 *
 * Protocol State Machine:
 * 1. Initialization: AICPU sets aicpu_ready=1
 * 2. Acknowledgment: AICore sets aicore_done=core_id+1
 * 3. Task Dispatch: AICPU assigns task pointer and sets task_status=1
 * 4. Task Execution: AICore reads task, executes, sets task_status=0
 * 5. Task Completion: AICPU reads task_status=0, clears task=0
 * 6. Shutdown: AICPU sets control=1, AICore exits
 *
 * Each AICore instance has its own handshake buffer to enable concurrent
 * task execution across multiple cores.
 */

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Field Access Patterns:
 * - aicpu_ready: Written by AICPU, read by AICore
 * - aicore_done: Written by AICore, read by AICPU
 * - task: Written by AICPU, read by AICore (0 = no task assigned)
 * - task_status: Written by both (AICPU=1 on dispatch, AICore=0 on completion)
 * - control: Written by AICPU, read by AICore (0 = continue, 1 = quit)
 * - core_type: Written by AICPU, read by AICore (0 = AIC, 1 = AIV)
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // Task pointer: 0=no task, non-zero=Task* address
    volatile int32_t task_status;   // Task execution status: 0=idle, 1=busy
    volatile int32_t control;       // Control signal: 0=execute, 1=quit
    volatile int32_t core_type;     // Core type: 0=AIC, 1=AIV
} __attribute__((aligned(64)));

/**
 * Core type enumeration
 *
 * Specifies which AICore type a task should run on.
 * AIC (AICore Compute) handles compute-intensive operations.
 * AIV (AICore Vector) handles vector/SIMD operations.
 */
enum class CoreType : int {
    AIC = 0,  // AICore Compute
    AIV = 1   // AICore Vector
};

/**
 * Core Information - Platform-agnostic core descriptor
 *
 * This structure describes a single core's properties without encoding
 * platform-specific mapping logic. Platform code is responsible for
 * populating these structures, while runtime code uses them.
 */
struct CoreInfo {
    int core_id;      // Logical core ID (index into Handshake buffer)
    int block_idx;    // Block index for this core (platform-specific mapping)
    int core_type;    // Core type: 0=AIC, 1=AIV
};

/**
 * Core Topology - Platform-provided core mapping
 *
 * This structure contains the complete core topology for the current
 * platform configuration. The platform must initialize this before
 * runtime execution begins. This allows the runtime to remain completely
 * platform-agnostic while still handling diverse core configurations.
 */
struct CoreTopology {
    int total_cores;                    // Total number of cores
    CoreInfo cores[RUNTIME_MAX_WORKER]; // Per-core information
    bool initialized;                   // Has topology been set?

    CoreTopology() : total_cores(0), initialized(false) {}
};

/**
 * Tensor pair for tracking host-device memory mappings.
 * Used for copy-back during finalize.
 */
struct TensorPair {
    void* host_ptr;
    void* dev_ptr;
    size_t size;
};

/**
 * Host API function pointers for device memory operations.
 * Allows runtime to use pluggable device memory backends.
 */
struct HostApi {
    void* (*device_malloc)(size_t size);
    void (*device_free)(void* dev_ptr);
    int (*copy_to_device)(void* dev_ptr, const void* host_ptr, size_t size);
    int (*copy_from_device)(void* host_ptr, const void* dev_ptr, size_t size);
};

/**
 * Task entry in the runtime
 *
 * Each task has a unique ID (its index in the task array), arguments,
 * and dependency information (fanin/fanout).
 */
typedef struct {
    int task_id;                      // Unique task identifier
    int func_id;                      // Function identifier
    uint64_t args[RUNTIME_MAX_ARGS];  // Task arguments
    int num_args;                     // Number of valid arguments

    // Runtime function pointer address (NEW)
    // This is the GM address where the kernel binary resides
    // It's cast to a function pointer at runtime: (KernelFunc)function_bin_addr
    uint64_t function_bin_addr;  // Address of kernel in device GM memory

    // Core type specification (NEW)
    // Specifies which core type this task should run on: 0=AIC, 1=AIV
    int core_type;  // 0=AIC, 1=AIV

    // Dependency tracking (using PTO runtime terminology)
    std::atomic<int> fanin;          // Number of predecessors (dependencies)
    int fanout[RUNTIME_MAX_FANOUT];  // Successor task IDs
    int fanout_count;                // Number of successors

    // DFX-specific fields
    uint64_t start_time;  // Start time of the task
    uint64_t end_time;    // End time of the task
} Task;

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for task dependency management
 *
 * Maintains a fixed-size array of tasks and uses a Queue for ready tasks.
 * Tasks are allocated monotonically and never reused within the same
 * runtime instance.
 *
 * Dependencies are managed manually via add_successor().
 */
class Runtime {
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    // Execution parameters for AICPU scheduling
    int sche_cpu_num;  // Number of AICPU threads for scheduling

    // Platform-provided core topology (NEW)
    // The platform must initialize this before runtime execution
    CoreTopology core_topology;

private:
    // Task storage
    Task tasks[RUNTIME_MAX_TASKS];  // Fixed-size task array
    int next_task_id;               // Next available task ID

    // Initial ready tasks (computed once, read-only after)
    int initial_ready_tasks[RUNTIME_MAX_TASKS];
    int initial_ready_count;

  // Tensor pairs for host-device memory tracking
  TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
  int tensor_pair_count;

public:
    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Task Management
    // =========================================================================

    /**
     * Allocate a new task with the given arguments
     *
     * @param args      Array of uint64_t arguments
     * @param num_args  Number of arguments (must be <= RUNTIME_MAX_ARGS)
     * @param func_id   Function identifier
     * @param core_type Core type for this task (0=AIC, 1=AIV)
     * @return Task ID (>= 0) on success, -1 on failure
     */
    int add_task(uint64_t *args, int num_args, int func_id, int core_type = 0);

    /**
     * Add a dependency edge: from_task -> to_task
     *
     * This adds to_task to from_task's fanout array and increments
     * to_task's fanin counter.
     *
     * @param from_task  Producer task ID
     * @param to_task    Consumer task ID (depends on from_task)
     */
    void add_successor(int from_task, int to_task);

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * Get a pointer to a task by ID
     *
     * @param task_id  Task ID to query
     * @return Pointer to task, or nullptr if invalid ID
     */
    Task *get_task(int task_id);

    /**
     * Get the total number of tasks in the runtime
     *
     * @return Total task count
     */
    int get_task_count() const;

    /**
     * Query core information by core ID (used by runtime executor)
     *
     * This method provides platform-agnostic access to core properties.
     * The runtime executor uses this to determine core assignments without
     * needing to know platform-specific mapping logic.
     *
     * @param core_id  Core ID to query (index into Handshake buffer)
     * @return Pointer to CoreInfo, or nullptr if invalid or not initialized
     */
    const CoreInfo* get_core_info(int core_id) const {
        if (!core_topology.initialized || core_id < 0 || core_id >= core_topology.total_cores) {
            return nullptr;
        }
        return &core_topology.cores[core_id];
    }

    /**
     * Get initially ready tasks (fanin == 0) as entry point for execution
     *
     * This scans all tasks and populates the provided array with task IDs
     * that have no dependencies (fanin == 0). The runtime can use this
     * as the starting point for task scheduling.
     *
     * @param ready_tasks  Array to populate with ready task IDs (can be
     * nullptr)
     * @return Number of initially ready tasks
     */
    int get_initial_ready_tasks(int *ready_tasks);

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /**
     * Print the runtime structure to stdout
     *
     * Shows task table with fanin/fanout information.
     */
    void print_runtime() const;

    // =========================================================================
    // Tensor Pair Management
    // =========================================================================

    /**
     * Record a host-device tensor pair for copy-back during finalize.
     *
     * @param host_ptr  Host memory pointer (destination for copy-back)
     * @param dev_ptr   Device memory pointer (source for copy-back)
     * @param size     Size of tensor in bytes
     */
    void record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size);

    /**
     * Get pointer to tensor pairs array.
     *
     * @return Pointer to tensor pairs array
     */
    TensorPair* get_tensor_pairs();

    /**
     * Get number of recorded tensor pairs.
     *
     * @return Number of tensor pairs
     */
    int get_tensor_pair_count() const;

    /**
     * Clear all recorded tensor pairs.
     */
    void clear_tensor_pairs();

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;
};

#endif  // RUNTIME_H
