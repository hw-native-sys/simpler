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
 * Runtime Class - Device Execution and Handshake Control
 *
 * This class manages device-side execution through AICPU-AICore handshake
 * protocol. Task graph construction is handled by PTO2Runtime; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, sche_cpu_num)
 * - Tensor pair management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses a per-core PTO2DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include "common/core_type.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "pto2_dispatch_payload.h"
#include "task_args.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_ARGS 128
#define RUNTIME_MAX_WORKER 72  // 24 AIC + 48 AIV cores
#define RUNTIME_MAX_TENSOR_PAIRS 64
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SO_SIZE (4 * 1024 * 1024)  // 4MB max for orchestration SO
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// Default ready queue shards: one shard per worker thread (total minus orchestrator)
constexpr int RUNTIME_DEFAULT_READY_QUEUE_SHARDS = PLATFORM_MAX_AICPU_THREADS - 1;

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
 * 3. Task Dispatch: AICPU writes DATA_MAIN_BASE after updating the per-core payload
 * 4. Task Execution: AICore reads the cached PTO2DispatchPayload and executes
 * 5. Task Completion: AICore writes FIN to COND; AICPU observes completion
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
 * - task: Written by AICPU, read by AICore (0 = not ready, non-zero = PTO2DispatchPayload*)
 * - core_type: Written by AICPU, read by AICore (CoreType::AIC or CoreType::AIV)
 */
struct Handshake {
    volatile uint32_t aicpu_ready;        // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;        // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;               // Init: PTO2DispatchPayload* (set before aicpu_ready); runtime: unused
    volatile CoreType core_type;          // Core type: CoreType::AIC or CoreType::AIV
    volatile uint32_t physical_core_id;   // Physical core ID
    volatile uint32_t aicpu_regs_ready;   // AICPU register init done: 0=pending, 1=done
    volatile uint32_t aicore_regs_ready;  // AICore ID reported: 0=pending, 1=done
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
    // Single-shot upload of the entire ChipCallable buffer. `callable` is a
    // `const ChipCallable *` (declared void* to avoid pulling task_interface
    // headers into runtime.h). DeviceRunner walks child_offsets_ to compute
    // total byte size, allocates device GM once, fixes up each child's
    // resolved_addr_ in an internal host scratch (onboard: device addr; sim:
    // dlopen function pointer), H2D's once, and returns the device-side
    // address of the ChipCallable header. Pool-managed: identical buffer
    // contents (FNV-1a 64-bit) hit the dedup cache; all chip buffers are
    // bulk-freed in DeviceRunner::finalize(). Returns 0 on error or when
    // child_count() == 0. Caller computes child addrs as
    //     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
    // and stores them via runtime->set_function_bin_addr(fid, child_dev).
    uint64_t (*upload_chip_callable_buffer)(const void *callable);
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
    int sche_cpu_num;        // Number of AICPU threads for scheduling
    int ready_queue_shards;  // Number of ready queue shards (1..MAX_AICPU_THREADS, default MAX-1)

    // Ring buffer size overrides (0 = use compile-time defaults)
    uint64_t task_window_size;
    uint64_t heap_size;
    uint64_t dep_pool_size;

    // PTO2 integration: kernel_id -> GM function_bin_addr mapping
    // NOTE: Made public for direct access from aicore code
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Orchestrator-to-scheduler transition control
    // When true, orchestrator threads convert to scheduler threads after orchestration completes.
    // When false (default), orchestrator threads exit after orchestration without dispatching tasks.
    // Controlled via PTO2_ORCH_TO_SCHED environment variable.
    bool orch_to_sched;

private:
    // Tensor pairs for host-device memory tracking
    TensorPair tensor_pairs[RUNTIME_MAX_TENSOR_PAIRS];
    int tensor_pair_count;

    // Kernel binary tracking for cleanup
    int registered_kernel_func_ids_[RUNTIME_MAX_FUNC_ID];
    int registered_kernel_count_;

    void *gm_sm_ptr_;                        // GM pointer to PTO2 shared memory (device)
    void *gm_heap_ptr_;                      // GM heap for orchestrator output buffers (device)
    void *slot_states_ptr_;                  // Pointer to PTO2TaskSlotState array (scheduler-private, for profiling)
    ChipStorageTaskArgs orch_args_storage_;  // Copy of args for device

    // Device orchestration SO (for dlopen on AICPU thread 3).
    // The SO bytes themselves live in a separately-allocated device buffer
    // owned by DeviceRunner; only the metadata below travels inside Runtime.
    uint64_t dev_orch_so_addr_;
    uint64_t dev_orch_so_size_;
    // Per-callable_id dispatch. AICPU dispatches via
    // `orch_so_table_[active_callable_id_]`; `register_new_callable_id_`
    // signals whether the host is delivering a freshly-registered
    // callable_id (write+dlopen) or reusing an already-loaded one.
    int32_t active_callable_id_;
    bool register_new_callable_id_;
    char device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];
    char device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME];

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
    void record_tensor_pair(void *host_ptr, void *dev_ptr, size_t size);

    /**
     * Get pointer to tensor pairs array.
     */
    TensorPair *get_tensor_pairs();

    /**
     * Get number of recorded tensor pairs.
     */
    int get_tensor_pair_count() const;

    /**
     * Clear all recorded tensor pairs.
     */
    void clear_tensor_pairs();

    // =========================================================================
    // Performance Profiling
    // =========================================================================

    // =========================================================================
    // Device orchestration (for AICPU thread 3)
    // =========================================================================

    void *get_gm_sm_ptr() const;
    void *get_gm_heap_ptr() const;
    const ChipStorageTaskArgs &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_gm_heap(void *p);
    void set_slot_states_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Device orchestration SO binary (for dlopen on AICPU thread 3)
    void set_dev_orch_so(uint64_t dev_addr, uint64_t size);
    uint64_t get_dev_orch_so_addr() const;
    uint64_t get_dev_orch_so_size() const;
    // Per-callable_id dispatch. callable_id must be in
    // [0, MAX_REGISTERED_CALLABLE_IDS); register_new_callable_id_ tells AICPU
    // whether to (re)load the orch SO into orch_so_table_[callable_id] or
    // reuse the cached entry.
    void set_active_callable_id(int32_t callable_id, bool is_new);
    int32_t get_active_callable_id() const;
    bool register_new_callable_id() const;
    void set_device_orch_func_name(const char *name);
    const char *get_device_orch_func_name() const;
    void set_device_orch_config_name(const char *name);
    const char *get_device_orch_config_name() const;

    uint64_t get_function_bin_addr(int func_id) const;
    void set_function_bin_addr(int func_id, uint64_t addr);
    /**
     * Replay a previously-uploaded kernel address onto a fresh Runtime
     * without recording it in registered_kernel_func_ids_. Used by
     * DeviceRunner::bind_prepared_callable_to_runtime so prepared kernel
     * binaries are not freed by validate_runtime_impl across runs.
     */
    void replay_function_bin_addr(int func_id, uint64_t addr);

    int get_registered_kernel_count() const;
    int get_registered_kernel_func_id(int index) const;
    void clear_registered_kernels();

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by PTO2Runtime, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in PTO2 shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses PTO2DispatchPayload, not Task. Always returns nullptr. */
    Task *get_task(int) { return nullptr; }

    // =========================================================================
    // Host API (host-only, not copied to device)
    // =========================================================================

    // Host API function pointers for device memory operations
    // NOTE: Placed at end of class to avoid affecting device memory layout
    HostApi host_api;
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
