/**
 * Runtime Class - Device Execution and Handshake Control
 *
 * This class manages device-side execution through AICPU-AICore handshake
 * protocol. Task graph construction is handled by PTO2Runtime; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, sche_cpu_num)
 * - Tensor pair management for host-device memory tracking
 * - Device orchestration state (pto2_gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses PTO2DispatchPayload from PTO2 shared memory.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include "common/core_type.h"
#include "pto2_dispatch_payload.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#ifndef RUNTIME_MAX_ARGS
#define RUNTIME_MAX_ARGS 16
#endif

#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72  // 24 AIC + 48 AIV cores
#endif

#ifndef RUNTIME_MAX_TENSOR_PAIRS
#define RUNTIME_MAX_TENSOR_PAIRS 64
#endif

#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 32
#endif

#ifndef RUNTIME_MAX_ORCH_SO_SIZE
#define RUNTIME_MAX_ORCH_SO_SIZE (1024 * 1024)  // 1MB max for orchestration SO
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
 * - task: Written by AICPU, read by AICore (0 = no task, non-zero = PTO2DispatchPayload*)
 * - task_status: Written by both (AICPU=1 on dispatch, AICore=0 on completion)
 * - control: Written by AICPU, read by AICore (0 = continue, 1 = quit)
 * - core_type: Written by AICPU, read by AICore (CoreType::AIC or CoreType::AIV)
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // Task pointer: 0=no task, non-zero=PTO2DispatchPayload*
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
 * Task structure - Compatibility stub for platform layer
 *
 * RT2 uses PTO2DispatchPayload instead of Task for task dispatch.
 * This stub exists only for API compatibility with device_runner.cpp.
 * Since get_task_count() returns 0, this struct is never actually used.
 */
struct Task {
    int func_id;
    uint64_t function_bin_addr;
};

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by PTO2Runtime; this class only handles
 * execution control and device orchestration state.
 */
class Runtime {
public:
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    // Execution parameters for AICPU scheduling
    int sche_cpu_num;  // Number of AICPU threads for scheduling

    // PTO2 integration: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Profiling support
    bool enable_profiling;                  // Enable profiling flag
    uint64_t perf_data_base;                // Performance data shared memory base address (device-side)

private:
    // Tensor pairs for host-device memory tracking
    TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
    int tensor_pair_count;

    // Device orchestration: when false, orchestration runs on device (thread 3)
    bool orch_built_on_host_;
    void* pto2_gm_sm_ptr_;   // GM pointer to PTO2 shared memory (device)
    uint64_t* orch_args_;    // Arguments for device orchestration
    int orch_arg_count_;
    uint64_t orch_args_storage_[RUNTIME_MAX_ARGS];  // Copy of args for device

    // Device orchestration SO binary (for dlopen on AICPU thread 3)
    // Stored as a copy to avoid lifetime issues with Python ctypes arrays
    uint8_t device_orch_so_storage_[RUNTIME_MAX_ORCH_SO_SIZE];
    size_t device_orch_so_size_;

public:
    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Tensor Pair Management
    // =========================================================================

    /**
     * Record a host-device tensor pair for copy-back during finalize.
     */
    void record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size);

    /**
     * Get pointer to tensor pairs array.
     */
    TensorPair* get_tensor_pairs();

    /**
     * Get number of recorded tensor pairs.
     */
    int get_tensor_pair_count() const;

    /**
     * Clear all recorded tensor pairs.
     */
    void clear_tensor_pairs();

    // =========================================================================
    // Device orchestration (for AICPU thread 3)
    // =========================================================================

    bool get_orch_built_on_host() const;
    void* get_pto2_gm_sm_ptr() const;
    uint64_t* get_orch_args() const;
    int get_orch_arg_count() const;
    void set_orch_built_on_host(bool v);
    void set_pto2_gm_sm_ptr(void* p);
    void set_orch_args(uint64_t* args, int count);

    // Device orchestration SO binary (for dlopen on AICPU thread 3)
    void set_device_orch_so(const void* data, size_t size);
    const void* get_device_orch_so_data() const;
    size_t get_device_orch_so_size() const;

    uint64_t get_function_bin_addr(int func_id) const;
    void set_function_bin_addr(int func_id, uint64_t addr);

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by PTO2Runtime, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in PTO2 shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses PTO2DispatchPayload, not Task. Always returns nullptr. */
    Task* get_task(int) { return nullptr; }

    /** @deprecated Use PTO2 dispatch mode */
    bool get_use_pto2_dispatch() const { return true; }

    /** @deprecated Use PTO2 dispatch mode */
    void set_use_pto2_dispatch(bool) { }

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;
};

#endif  // RUNTIME_H
