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

#include "common/core_type.h"
#include "common/platform_config.h"

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
#define RUNTIME_MAX_WORKER PLATFORM_MAX_CORES_PER_THREAD
#endif

#ifndef RUNTIME_MAX_TENSOR_PAIRS
#define RUNTIME_MAX_TENSOR_PAIRS 64
#endif

// Max number of uint64_t arguments marshaled from host orchestration to AICPU builder.
#ifndef RUNTIME_MAX_ORCH_ARGS
#define RUNTIME_MAX_ORCH_ARGS 64
#endif

// Max func_id supported by kernel address table (func_id -> function_bin_addr).
// Keep this small and bump as needed.
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 64
#endif

// Max size of the AICPU orchestration plugin (.so) embedded in Runtime.
// This storage is read by AICPU and written to an executable temp file for dlopen().
#ifndef RUNTIME_MAX_AICPU_ORCH_SO_SIZE
#define RUNTIME_MAX_AICPU_ORCH_SO_SIZE (1024 * 1024)  // 1MB
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
 * - core_type: Written by AICPU, read by AICore (CoreType::AIC or CoreType::AIV)
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // Task pointer: 0=no task, non-zero=Task* address
    volatile int32_t task_status;   // Task execution status: 0=idle, 1=busy
    volatile int32_t control;       // Control signal: 0=execute, 1=quit
    volatile CoreType core_type;    // Core type: CoreType::AIC or CoreType::AIV
    volatile uint64_t perf_records_addr; // Performance records address
    volatile uint32_t perf_buffer_status; // 0 = not full, 1 == full
} __attribute__((aligned(64)));

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
 * Device allocations tracked for cleanup in finalize.
 *
 * This is distinct from TensorPair: not every device allocation needs copy-back.
 * Orchestration code should register any device buffers it allocates so the
 * runtime can free them in validate_runtime_impl().
 */
struct DeviceAlloc {
    void* dev_ptr;
};

class Runtime;

/**
 * AICPU graph-build API table (device-side).
 *
 * Motivation:
 * On some real AICPU deployments, dlopen'd orchestration plugins may not be able
 * to resolve undefined symbols from the main AICPU runtime binary at load time.
 * To keep the plugin small and avoid relinking/reuploading the runtime, we pass
 * graph-build entry points to the plugin via function pointers stored in
 * `Runtime`. The AICPU executor initializes these pointers on device before
 * calling into the plugin.
 *
 * Example usage (in plugin):
 *   auto& api = runtime->aicpu_build_api;
 *   int t = api.add_task(runtime, args, n, func_id, CoreType::AIV, 0);
 *   api.publish_task(runtime, t);
 */
struct AicpuBuildApi {
    int (*add_task)(
        Runtime* runtime, uint64_t* args, int num_args, int func_id, CoreType core_type, uint64_t function_bin_addr);
    void (*add_successor_conditional)(Runtime* runtime, int from_task, int to_task);
    void (*publish_task)(Runtime* runtime, int task_id);
    void* (*device_malloc)(size_t size);
    void (*device_free)(void* ptr);
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
    uint64_t (*upload_kernel_binary)(int func_id, const uint8_t* bin_data, size_t bin_size);
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

    // Core type specification
    // Specifies which core type this task should run on
    CoreType core_type;  // CoreType::AIC or CoreType::AIV

    // Dependency tracking (using PTO runtime terminology)
    std::atomic<int> fanin;          // Number of predecessors (dependencies)
    int fanout[RUNTIME_MAX_FANOUT];  // Successor task IDs
    int fanout_count;                // Number of successors

    // DFX-specific fields
    uint64_t start_time;  // Start time of the task
    uint64_t end_time;    // End time of the task

    /**
     * Scheduling state for concurrent build||schedule.
     *
     * `published`:
     * - Set by the builder when the task's fields (args/func_id/core_type, and any
     *   required dependency edges) are ready for scheduler consumption.
     * - Scheduler threads must treat unpublished tasks as non-existent.
     *
     * `completed`:
     * - Set by scheduler threads when the task finishes on AICore.
     * - Used to make `add_successor_conditional()` safe when edges are added late.
     */
    std::atomic<int> published;  // 0 = not visible to scheduler, 1 = published
    std::atomic<int> completed;  // 0 = not completed, 1 = completed
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

    /**
     * Orchestration payload (auto-populated by init_runtime_impl, consumed by AICPU orchestration).
     *
     * The framework iterates func_args using arg_types/arg_sizes:
     * - Pointer args (ARG_INPUT_PTR, ARG_OUTPUT_PTR, ARG_INOUT_PTR): device memory
     *   is allocated, input data is copied, and the device pointer is stored here.
     * - Scalar args (ARG_SCALAR): the value is stored directly.
     *
     * The AICPU orchestration plugin reads orch_args[] to obtain device pointers
     * and scalar values, then builds the task graph.
     */
    int orch_argc;
    uint64_t orch_args[RUNTIME_MAX_ORCH_ARGS];

    /**
     * Kernel address table (written on host before launch, read by AICPU builder).
     *
     * This enables AICPU-built tasks to bind `Task::function_bin_addr` without host
     * iterating the task table (tasks may not exist yet on host).
     *
     * Convention:
     * - `kernel_addrs[func_id]` holds the executable address for that `func_id`.
     * - Examples typically pass `function_bin_addr=0` to `aicpu_runtime_add_task()`
     *   to auto-bind via this table (the table is filled by the host runtime init,
     *   not by platform code).
     */
    uint64_t kernel_addrs[RUNTIME_MAX_FUNC_ID];

    /**
     * AICPU orchestration plugin (device-side dlopen builder).
     *
     * When set by host orchestration, the AICPU builder thread will:
     * - materialize the embedded `.so` bytes into a temp file
     * - `dlopen()` the temp file on AICPU
     * - `dlsym()` the entry function `aicpu_orch_func_name`
     * - call `int (*)(Runtime*)`
     *
     * This enables updating graph-building logic by uploading only a small
     * orchestration plugin `.so` (instead of relinking/reuploading the full runtime).
     */
    uint8_t aicpu_orch_so_storage[RUNTIME_MAX_AICPU_ORCH_SO_SIZE];
    uint32_t aicpu_orch_so_size;
    char aicpu_orch_func_name[64];

    // Attempt to embed AICPU orchestration plugin bytes into Runtime.
    // Returns false on invalid input or if the plugin is larger than the
    // built-in storage.
    bool try_set_aicpu_orch_so(const void* data, size_t size);
    void set_aicpu_orch_so(const void* data, size_t size);
    const void* get_aicpu_orch_so_data() const;
    size_t get_aicpu_orch_so_size() const;

    /**
     * Build mode:
     * - 0 = sequential build->schedule (scheduler threads wait for builder)
     * - 1 = concurrent build||schedule (builder publishes tasks while schedulers run)
     */
    int build_mode;

    /**
     * Device-side graph-build API table.
     *
     * This is initialized by the AICPU executor on device before any
     * orchestration plugin runs. Plugins should prefer this table over linking
     * against `aicpu_runtime_*` symbols directly.
     */
    AicpuBuildApi aicpu_build_api;

    // Profiling support
    bool enable_profiling;                  // Enable profiling flag
    uint64_t perf_data_base;                // Performance data shared memory base address (device-side)

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

    // Device allocations for cleanup (no copy-back implied).
    DeviceAlloc device_allocs[RUNTIME_MAX_TENSOR_PAIRS];
    int device_alloc_count;

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
     * @param core_type Core type for this task (CoreType::AIC or CoreType::AIV)
     * @return Task ID (>= 0) on success, -1 on failure
     */
    int add_task(uint64_t* args, int num_args, int func_id, CoreType core_type = CoreType::AIC);

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

    /**
     * Add a dependency edge conditionally for concurrent build.
     *
     * Always records the edge in from_task.fanout[]. If from_task is already
     * completed, the dependency is considered already satisfied and to_task.fanin
     * is NOT incremented.
     */
    void add_successor_conditional(int from_task, int to_task);

    // =========================================================================
    // Query Methods
    // =========================================================================

    /**
     * Get a pointer to a task by ID
     *
     * @param task_id  Task ID to query
     * @return Pointer to task, or nullptr if invalid ID
     */
    Task* get_task(int task_id);

    /**
     * Get the total number of tasks in the runtime
     *
     * @return Total task count
     */
    int get_task_count() const;

    /**
     * Resolve executable function address for a kernel func_id.
     *
     * Used by platform runners (e.g., `a2a3sim`) to populate `Task::function_bin_addr`
     * before dispatch. For `aicpu_build_graph`, the host runtime fills
     * `Runtime::kernel_addrs[]` during initialization.
     *
     * @return Executable address, or 0 if unknown/out-of-range.
     */
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
            return 0;
        }
        return kernel_addrs[func_id];
    }

    /**
     * Set PTO2 shared memory pointer (stub for API compatibility).
     *
     * Only used by the `tensormap_and_ringbuffer` runtime (rt2). This runtime
     * doesn't use PTO2 shared memory, so this is a no-op.
     */
    void set_pto2_gm_sm_ptr(void*) { /* no-op */ }

    /**
     * Set function binary address for a func_id.
     *
     * Called by the platform C API after kernel registration.
     */
    void set_function_bin_addr(int func_id, uint64_t addr) {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
            return;
        }
        kernel_addrs[func_id] = addr;
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
    int get_initial_ready_tasks(int* ready_tasks);

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
     * Record a device allocation for cleanup during finalize.
     *
     * This does not imply copy-back; it only affects `validate_runtime_impl()`.
     */
    void record_device_alloc(void* dev_ptr);

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
     * Get pointer to device allocations array.
     *
     * @return Pointer to device allocations array
     */
    DeviceAlloc* get_device_allocs();

    /**
     * Get number of recorded device allocations.
     *
     * @return Number of device allocations
     */
    int get_device_alloc_count() const;

    /**
     * Clear all recorded tensor pairs.
     */
    void clear_tensor_pairs();

    /**
     * Clear all recorded device allocations.
     */
    void clear_device_allocs();

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;
};

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// AICPU-side Graph Build API (for examples)
// =============================================================================
//
// These functions are implemented by the aicpu_build_graph AICPU executor and
// are intended to be called from an example-provided orchestration plugin.
//
// They provide:
// - Internal synchronization with the scheduler (graph mutex)
// - Published task counting and ready-queue insertion
//
// The builder program itself is compiled from the example (not hardcoded in the
// runtime).

/**
 * Create a task from AICPU during graph build.
 *
 * Thread-safety:
 * - Safe to call concurrently with scheduler threads in concurrent build||schedule mode.
 *
 * Kernel address binding:
 * - If `function_bin_addr != 0`, it is written into `Task::function_bin_addr` directly.
 * - If `function_bin_addr == 0`, the runtime will auto-fill it from `runtime->kernel_addrs[func_id]`.
 *   This is the intended path for most examples: pass 0 and rely on the host to populate
 *   `Runtime::kernel_addrs[]` before launching AICPU.
 */
int aicpu_runtime_add_task(
    Runtime* runtime, uint64_t* args, int num_args, int func_id, CoreType core_type, uint64_t function_bin_addr);

/**
 * Add an edge `from_task -> to_task` during AICPU-side graph build (concurrency-safe).
 *
 * This is the recommended edge API for concurrent build||schedule:
 * - It always appends `to_task` into `from_task.fanout[]`.
 * - It only increments `to_task.fanin` if `from_task` has not already completed.
 *
 * This avoids races where the scheduler completes `from_task` before the builder
 * adds the edge.
 */
void aicpu_runtime_add_successor_conditional(Runtime* runtime, int from_task, int to_task);

/**
 * Publish a task to the scheduler during AICPU-side graph build.
 *
 * Publishing makes the task visible to scheduler threads. If `task.fanin == 0` at
 * publish time, the task is pushed into the appropriate ready queue immediately.
 *
 * Typical builder order:
 * 1) Create task
 * 2) Add edges (successors)
 * 3) Publish the task
 */
void aicpu_runtime_publish_task(Runtime* runtime, int task_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RUNTIME_H
