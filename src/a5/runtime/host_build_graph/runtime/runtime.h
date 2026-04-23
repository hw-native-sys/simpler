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

#ifndef SRC_A5_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_RUNTIME_H_
#define SRC_A5_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <atomic>

#include "common/core_type.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "tensor_info.h"

// Logging macros using unified logging interface
#include "common/unified_log.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#ifndef RUNTIME_MAX_TASKS
#define RUNTIME_MAX_TASKS 131072
#endif

#ifndef RUNTIME_MAX_ARGS
#define RUNTIME_MAX_ARGS 16
#endif

#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 128
#endif

#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER PLATFORM_MAX_CORES_PER_THREAD
#endif

#ifndef RUNTIME_MAX_TENSOR_PAIRS
#define RUNTIME_MAX_TENSOR_PAIRS 64
#endif

#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 32
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
 * enable_profiling_flag bit definitions (umbrella bitmask — "profiling"
 * is the umbrella, each bit is a parallel diagnostics sub-feature):
 * - bit0: tensor dump enabled
 * - bit1: L2 swimlane enabled
 * - bit2: PMU enabled
 *
 * Field Access Patterns:
 * - aicpu_ready: Written by AICPU, read by AICore
 * - aicore_done: Written by AICore, read by AICPU
 * - task: Written by AICPU, read by AICore (0 = no task assigned)
 * - task_status: Written by both (AICPU=1 on dispatch, AICore=0 on completion)
 * - control: Written by AICPU, read by AICore (0 = continue, 1 = quit)
 * - core_type: Written by AICPU, read by AICore (CoreType::AIC or CoreType::AIV)
 * - l2_perf_records_addr: Written by AICPU, read by AICore (performance records address)
 * - l2_perf_buffer_status: Written by both (AICPU=1 on buffer full, AICore=0 on buffer empty)
 * - physical_core_id: Written by AICPU, read by AICore (physical core ID)
 * - enable_profiling_flag: Written by host/AICPU init, read by AICore (bitmask)
 * - pmu_buffer_addr: Written by AICPU at PMU init (before aicpu_regs_ready=1), read by AICore
 * - pmu_reg_base: Written by AICPU at PMU init (before aicpu_regs_ready=1), read by AICore
 */
struct Handshake {
    volatile uint32_t aicpu_ready;            // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;            // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;                   // Task pointer: 0=no task, non-zero=Task* address
    volatile int32_t task_status;             // Task execution status: 0=idle, 1=busy
    volatile int32_t control;                 // Control signal: 0=execute, 1=quit
    volatile CoreType core_type;              // Core type: CoreType::AIC or CoreType::AIV
    volatile uint64_t l2_perf_records_addr;   // Performance records address
    volatile uint32_t l2_perf_buffer_status;  // 0 = not full, 1 = full
    volatile uint32_t physical_core_id;       // Physical core ID
    volatile uint32_t aicpu_regs_ready;       // AICPU register init done: 0=pending, 1=done
    volatile uint32_t aicore_regs_ready;      // AICore ID reported: 0=pending, 1=done
    volatile uint32_t
        enable_profiling_flag;          // Umbrella diagnostics bitmask; bit0=dump_tensor, bit1=l2_swimlane, bit2=pmu
    volatile uint64_t pmu_buffer_addr;  // Per-core PmuBuffer device address (for AICore-side PMU record)
    volatile uint64_t pmu_reg_base;     // Per-core PMU MMIO base (for AICore-side PMU register reads)
} __attribute__((aligned(64)));

/**
 * Tensor pair for tracking host-device memory mappings.
 * Used for copy-back during finalize.
 */
struct TensorPair {
    void *host_ptr;
    void *dev_ptr;
    size_t size;
};

/**
 * Host API function pointers for device memory operations.
 * Allows runtime to use pluggable device memory backends.
 */
struct HostApi {
    void *(*device_malloc)(size_t size);
    void (*device_free)(void *dev_ptr);
    int (*copy_to_device)(void *dev_ptr, const void *host_ptr, size_t size);
    int (*copy_from_device)(void *host_ptr, const void *dev_ptr, size_t size);
    uint64_t (*upload_kernel_binary)(int func_id, const uint8_t *bin_data, size_t bin_size);
    void (*remove_kernel_binary)(int func_id);
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

    // Perf swimlane collection
    bool enable_l2_swimlane;     // Enable perf swimlane collection
    uint64_t l2_perf_data_base;  // Performance data shared memory base address (device-side)

    // Task storage
    Task tasks[RUNTIME_MAX_TASKS];  // Fixed-size task array

private:
    int next_task_id;  // Next available task ID

    // Initial ready tasks (computed once, read-only after)
    int initial_ready_tasks[RUNTIME_MAX_TASKS];
    int initial_ready_count;

    // Tensor pairs for host-device memory tracking
    TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
    int tensor_pair_count;

    // Function address mapping (for API compatibility with rt2)
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Kernel binary tracking for cleanup
    int registered_kernel_func_ids_[RUNTIME_MAX_FUNC_ID];
    int registered_kernel_count_;

    // Tensor info metadata for tensor dump
    void *tensor_info_storage_;
    uint64_t tensor_info_storage_bytes_;
    uint32_t tensor_info_offsets_[RUNTIME_MAX_TASKS];
    uint16_t tensor_info_counts_[RUNTIME_MAX_TASKS];

    // Device allocation ranges used to recover tensor buffer addresses from task.args[]
    void *tensor_allocation_storage_;
    uint64_t tensor_allocation_storage_bytes_;
    uint32_t tensor_allocation_count_;

public:
    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // Orchestration is always built on the host for this runtime
    bool get_orch_built_on_host() const { return true; }

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
    int add_task(uint64_t *args, int num_args, int func_id, CoreType core_type = CoreType::AIC);

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
    void record_tensor_pair(void *host_ptr, void *dev_ptr, size_t size);

    /**
     * Get pointer to tensor pairs array.
     *
     * @return Pointer to tensor pairs array
     */
    TensorPair *get_tensor_pairs();

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
    // Tensor Info Metadata
    // =========================================================================

    void set_tensor_info_storage(void *ptr, uint64_t bytes) {
        tensor_info_storage_ = ptr;
        tensor_info_storage_bytes_ = bytes;
    }

    void clear_tensor_info_storage() {
        tensor_info_storage_ = nullptr;
        tensor_info_storage_bytes_ = 0;
    }

    void set_tensor_info_range(int task_id, uint32_t offset, uint16_t count) {
        if (task_id < 0 || task_id >= RUNTIME_MAX_TASKS) return;
        tensor_info_offsets_[task_id] = offset;
        tensor_info_counts_[task_id] = count;
    }

    const TensorInfo *get_tensor_info(int task_id, int *count) const {
        if (count != nullptr) {
            *count = 0;
        }
        if (task_id < 0 || task_id >= RUNTIME_MAX_TASKS || tensor_info_storage_ == nullptr) {
            return nullptr;
        }
        uint16_t tensor_info_count = tensor_info_counts_[task_id];
        if (tensor_info_count == 0) {
            return nullptr;
        }
        if (count != nullptr) {
            *count = static_cast<int>(tensor_info_count);
        }
        const TensorInfo *base = reinterpret_cast<const TensorInfo *>(tensor_info_storage_);
        return base + tensor_info_offsets_[task_id];
    }

    void *get_tensor_info_storage() const { return tensor_info_storage_; }

    uint64_t get_tensor_info_storage_bytes() const { return tensor_info_storage_bytes_; }

    void set_tensor_allocation_storage(void *ptr, uint32_t count, uint64_t bytes) {
        tensor_allocation_storage_ = ptr;
        tensor_allocation_count_ = count;
        tensor_allocation_storage_bytes_ = bytes;
    }

    void clear_tensor_allocation_storage() {
        tensor_allocation_storage_ = nullptr;
        tensor_allocation_count_ = 0;
        tensor_allocation_storage_bytes_ = 0;
    }

    bool is_tensor_buffer_addr(uint64_t addr) const {
        if (tensor_allocation_storage_ == nullptr || tensor_allocation_count_ == 0) {
            return false;
        }
        const TensorAllocationInfo *allocations =
            reinterpret_cast<const TensorAllocationInfo *>(tensor_allocation_storage_);
        for (uint32_t i = 0; i < tensor_allocation_count_; i++) {
            if (allocations[i].contains(addr)) {
                return true;
            }
        }
        return false;
    }

    void *get_tensor_allocation_storage() const { return tensor_allocation_storage_; }

    uint64_t get_tensor_allocation_storage_bytes() const { return tensor_allocation_storage_bytes_; }

    // =========================================================================
    // Device Orchestration (stub for API compatibility)
    // =========================================================================

    /**
     * Set PTO2 shared memory pointer (stub for host_build_graph).
     * This is a no-op for host orchestration; only used by rt2.
     */
    void set_pto2_gm_sm_ptr(void *) { /* no-op */ }

    /**
     * Get function binary address by func_id.
     * Used by platform layer to resolve kernel addresses.
     */
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

    /**
     * Set function binary address for a func_id.
     * Called by platform layer after kernel registration.
     */
    void set_function_bin_addr(int func_id, uint64_t addr) {
        if (func_id >= 0 && func_id < RUNTIME_MAX_FUNC_ID) {
            func_id_to_addr_[func_id] = addr;
            if (addr != 0 && registered_kernel_count_ < RUNTIME_MAX_FUNC_ID) {
                registered_kernel_func_ids_[registered_kernel_count_++] = func_id;
            }
        }
    }

    int get_registered_kernel_count() const { return registered_kernel_count_; }

    int get_registered_kernel_func_id(int index) const {
        if (index < 0 || index >= registered_kernel_count_) return -1;
        return registered_kernel_func_ids_[index];
    }

    void clear_registered_kernels() { registered_kernel_count_ = 0; }

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;
};

#endif  // SRC_A5_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_RUNTIME_H_
