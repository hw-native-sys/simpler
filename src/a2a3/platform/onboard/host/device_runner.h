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
 * Device Runner - Ascend Device Execution Utilities
 *
 * This module provides utilities for launching and managing AICPU and AICore
 * kernels on Ascend devices using CANN runtime APIs.
 *
 * Key Components:
 * - DeviceArgs: AICPU device argument structure
 * - KernelArgsHelper: Helper for managing kernel arguments with device memory
 * - AicpuSoInfo: AICPU shared object (.so) file management
 * - DeviceRunner: kernel launching and execution
 */

#ifndef RUNTIME_DEVICERUNNER_H
#define RUNTIME_DEVICERUNNER_H

#include <runtime/rt.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/l2_perf_collector.h"
#include "host/tensor_dump_collector.h"
#include "host/pmu_collector.h"
#include "runtime.h"

/**
 * DeviceArgs structure for AICPU device arguments
 *
 * This structure contains pointers to device memory for the AICPU shared
 * object. The layout is hardcoded in libaicpu_extend_kernels.so, which expects
 * specific offsets for aicpu_so_bin and aicpu_so_len fields.
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
};

/**
 * Helper class for managing KernelArgs with device memory
 *
 * This class wraps KernelArgs and provides host-side initialization methods
 * for allocating device memory and copying data to the device. It separates
 * the concerns of device memory management (host-only) from the structure
 * layout (shared with kernels).
 *
 * The helper provides implicit conversion to KernelArgs* for seamless use
 * with runtime APIs.
 */
struct KernelArgsHelper {
    KernelArgs args;
    MemoryAllocator *allocator_{nullptr};
    KernelArgs *device_k_args_{nullptr};  // Device copy of KernelArgs for AICore

    /**
     * Initialize device arguments by allocating device memory and copying data
     *
     * @param host_device_args  Host-side device arguments to copy
     * @param allocator       Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init_device_args(const DeviceArgs &host_device_args, MemoryAllocator &allocator);

    /**
     * Free device memory allocated for device arguments
     *
     * @return 0 on success, error code on failure
     */
    int finalize_device_args();

    /**
     * Initialize runtime arguments by allocating device memory and copying data
     *
     * @param host_runtime  Host-side runtime to copy to device
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator);

    /**
     * Free device memory allocated for runtime arguments
     *
     * @return 0 on success, error code on failure
     */
    int finalize_runtime_args();

    /**
     * Retrieve FFTS base address via rtGetC2cCtrlAddr and store in KernelArgs
     *
     * @return 0 on success, error code on failure
     */
    int init_ffts_base_addr();

    /**
     * Copy KernelArgs to device memory for AICore kernel parameter passing
     *
     * Must be called after init_runtime_args and init_ffts_base_addr.
     *
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init_device_kernel_args(MemoryAllocator &allocator);

    /**
     * Free device memory allocated for KernelArgs copy
     *
     * @return 0 on success, error code on failure
     */
    int finalize_device_kernel_args();

    /**
     * Implicit conversion operators for seamless use with runtime APIs
     *
     * These operators allow KernelArgsHelper to be used wherever KernelArgs*
     * is expected, enabling transparent device memory management while
     * maintaining API compatibility.
     */
    operator KernelArgs *() { return &args; }
    KernelArgs *operator&() { return &args; }
};

/**
 * AICPU shared object information and management
 *
 * This class manages loading and device memory allocation for AICPU
 * shared object (.so) files.
 */
struct AicpuSoInfo {
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
    MemoryAllocator *allocator_{nullptr};

    /**
     * Load shared object binary data and copy to device memory
     *
     * @param aicpu_so_binary  Binary data of the AICPU shared object
     * @param allocator      Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int init(const std::vector<uint8_t> &aicpu_so_binary, MemoryAllocator &allocator);

    /**
     * Free device memory allocated for shared object
     *
     * @return 0 on success, error code on failure
     */
    int finalize();
};

/**
 * Device runner for kernel execution
 *
 * This class provides a unified interface for launching AICPU and AICore
 * kernels on Ascend devices. It handles:
 * - Device initialization and resource management
 * - Tensor memory allocation and data transfer
 * - AICPU kernel launching with dynamic arguments
 * - AICore kernel registration and launching
 * - Coordinated execution of both kernel types
 * - Runtime execution workflow
 */
class DeviceRunner {
public:
    DeviceRunner() = default;
    ~DeviceRunner();

    /**
     * Create a thread bound to this device.
     * The thread calls rtSetDevice(device_id) on entry.
     */
    std::thread create_thread(std::function<void()> fn);

    /**
     * Allocate device tensor memory
     *
     * @param bytes  Size of tensor in bytes
     * @return Device pointer on success, nullptr on failure
     */
    void *allocate_tensor(size_t bytes);

    /**
     * Free device tensor memory
     *
     * @param dev_ptr  Device pointer to free
     */
    void free_tensor(void *dev_ptr);

    /**
     * Copy data from host to device
     *
     * @param dev_ptr   Device pointer
     * @param host_ptr  Host pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);

    /**
     * Copy data from device to host
     *
     * @param host_ptr  Host pointer
     * @param dev_ptr   Device pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);

    /**
     * Execute a runtime
     *
     * This method:
     * 1. Initializes device if not already done (lazy initialization)
     * 2. Initializes worker handshake buffers in the runtime based on block_dim
     * 3. Transfers runtime to device memory
     * 4. Launches AICPU init kernel
     * 5. Launches AICPU main kernel
     * 6. Launches AICore kernel
     * 7. Synchronizes streams
     * 8. Cleans up runtime memory
     *
     * @param runtime             Runtime to execute (will be modified to
     * initialize workers)
     * @param block_dim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param device_id            Device ID (0-15)
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @param launch_aicpu_num      Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int
    run(Runtime &runtime, int block_dim, int device_id, const std::vector<uint8_t> &aicpu_so_binary,
        const std::vector<uint8_t> &aicore_kernel_binary, int launch_aicpu_num = 1);

    /**
     * Enablement setters for the three diagnostics sub-features. Called by
     * the c_api entry point before run(); downstream run() paths read the
     * corresponding `enable_*_` members directly. Moved off the generic
     * Runtime struct / run() arg list so all three travel the same way.
     */
    void set_l2_swimlane_enabled(bool enable) { enable_l2_swimlane_ = enable; }
    void set_dump_tensor_enabled(bool enable) { enable_dump_tensor_ = enable; }
    void set_pmu_enabled(int enable_pmu) {
        enable_pmu_ = (enable_pmu > 0);
        pmu_event_type_ = resolve_pmu_event_type(enable_pmu);
    }
    // Severity floor (0=DEBUG..4=NUL) and INFO verbosity threshold (0..9).
    // Pushed in by the Python layer via run_runtime() and propagated to AICPU
    // through KernelArgs.
    void set_log_level(int log_level) { log_level_ = log_level; }
    void set_log_info_v(int log_info_v) { log_info_v_ = log_info_v; }
    // Directory under which all diagnostic artifacts (l2_perf_records.json /
    // tensor_dump/ / pmu.csv) land. Required (non-empty) when any diagnostic
    // is enabled; CallConfig::validate() enforces this contract upstream.
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

    /**
     * Print handshake results from device
     *
     * Copies handshake buffers from device and prints their status.
     * Must be called after run() and before finalize().
     */
    void print_handshake_results();

    /**
     * Cleanup all resources
     *
     * Frees all device memory, destroys streams, and resets state.
     * Use this for final cleanup when no more tests will run.
     *
     * @return 0 on success, error code on failure
     */
    int finalize();

    /**
     * Launch an AICPU kernel
     *
     * Internal method used by run(). Can be called directly for custom
     * workflows.
     *
     * @param stream      AICPU stream
     * @param k_args       Kernel arguments
     * @param kernel_name  Name of the kernel to launch
     * @param aicpu_num    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int launch_aicpu_kernel(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num);

    /**
     * Launch an AICore kernel
     *
     * Internal method used by run(). Can be called directly for custom
     * workflows.
     *
     * @param stream  AICore stream
     * @param k_args  Pointer to kernel arguments (includes runtime, ffts_base_addr, etc.)
     * @return 0 on success, error code on failure
     */
    int launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args);

    /**
     * Upload a kernel binary to device memory
     *
     * IMPORTANT: prepare_run_context() must be called before this function.
     * Kernels are immediately copied to device memory.
     *
     * Receives pre-extracted .text section binary data,
     * allocates device GM memory, copies the binary to device,
     * and returns the device GM address. The caller is responsible
     * for storing this address (typically in Runtime::func_id_to_addr_[]).
     *
     * If the kernel is already uploaded (same func_id), returns the
     * cached address without re-uploading.
     *
     * @param func_id   Function identifier (0, 1, 2, ...) for caching
     * @param bin_data  Kernel .text section binary data
     * @param bin_size  Size of binary data in bytes
     * @return Device GM address of kernel on success, 0 on error
     */
    uint64_t upload_kernel_binary(int func_id, const uint8_t *bin_data, size_t bin_size);

    /**
     * Remove a kernel binary from device memory
     *
     * Frees the device memory allocated for the kernel and removes the
     * cached entry. This should be called during per-case cleanup.
     *
     * @param func_id   Function identifier to remove
     */
    void remove_kernel_binary(int func_id);

    /**
     * Attach the current host thread to the target device.
     *
     * This is required before host-side runtime initialization may allocate or
     * free device memory on the current thread. No streams are created here.
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure
     */
    int attach_current_thread(int device_id);

    /**
     * Make the ACL context ready on the current thread.
     *
     * Calls aclInit() once per process (subsequent calls are idempotent and
     * tolerate the ACL_ERROR_REPEAT_INITIALIZE sentinel) and aclrtSetDevice()
     * on the current thread. This is the entry point for consumers that need
     * to call acl* / Hccl* APIs (for example the comm_hccl backend) but
     * intentionally do not want those modules to own ACL lifecycle themselves.
     *
     * Symmetric with finalize(): aclrtResetDevice + aclFinalize run there.
     *
     * @param device_id  Device ID to bind on the current thread.
     * @return 0 on success, error code on failure.
     */
    int ensure_acl_ready(int device_id);

    /**
     * Create a caller-owned aclrtStream for comm_* usage.
     *
     * Intended to back the ChipWorker Python wrapper's internal stream
     * ownership for distributed comm — callers pair it with
     * destroy_comm_stream() at teardown.  The ACL context must already be
     * ready on the calling thread (ensure_acl_ready()).
     *
     * @return aclrtStream pointer on success, NULL on failure.
     */
    void *create_comm_stream();

    /**
     * Destroy a stream previously returned by create_comm_stream().
     * Tolerates a nullptr stream (returns 0).
     *
     * @return 0 on success, error code on failure.
     */
    int destroy_comm_stream(void *stream);

    /**
     * Ensure the current thread has fresh run-scoped streams.
     *
     * This attaches the current thread to the target device and lazily creates
     * the AICPU/AICore streams used by a single run.
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure
     */
    int prepare_run_context(int device_id);

    /**
     * Release run-scoped resources owned by the current thread.
     *
     * This destroys AICPU/AICore streams but intentionally preserves device
     * allocations, uploaded binaries, and other session state so they can be
     * finalized later before rtDeviceReset().
     */
    void release_run_context();

private:
    // Internal state
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};  // Stored for print_handshake_results in destructor
    std::vector<uint8_t> aicore_kernel_binary_;

    // Memory management
    MemoryAllocator mem_alloc_;

    // Device resources
    rtStream_t stream_aicpu_{nullptr};
    rtStream_t stream_aicore_{nullptr};
    AicpuSoInfo so_info_;
    KernelArgsHelper kernel_args_;
    DeviceArgs device_args_;

    // Kernel binary management
    bool binaries_loaded_{false};              // true after AICPU SO loaded
    std::map<int, uint64_t> func_id_to_addr_;  // func_id -> function_bin_addr (device GM)

    // Orchestration SO cache. `cached_orch_so_hash_ == 0` means "no cache".
    // The device buffer grows monotonically — cache miss with a larger SO
    // reallocates, a smaller SO reuses the existing capacity. The host copy
    // insulates us from Python ctypes array lifetimes.
    uint64_t cached_orch_so_hash_{0};
    void *dev_orch_so_buffer_{nullptr};
    size_t dev_orch_so_capacity_{0};
    std::vector<uint8_t> host_orch_so_copy_;

    // ACL lifecycle (process-wide). aclInit must run exactly once; ensure_acl_ready
    // gates it behind this flag. finalize() drives aclFinalize only if we observed
    // acl_ready_, so runtimes that never ask for ACL (e.g. pure rt-layer) stay unaffected.
    bool acl_ready_{false};

    // Performance profiling
    L2PerfCollector l2_perf_collector_;

    // Tensor dump (independent shared memory + memory manager)
    TensorDumpCollector dump_collector_;
    // PMU collector (independent of profiling pipeline)
    PmuCollectorHost pmu_collector_;

    /**
     * Ensure device is initialized (lazy initialization)
     *
     * Checks if device is already initialized. If not, performs:
     * - Attach the current thread to the device
     * - Create AICPU and AICore streams
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param device_id            Device ID (0-15)
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int ensure_device_initialized(
        int device_id, const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
    );

    /**
     * Load AICPU SO and initialize device args
     *
     * Called by run() after prepare_run_context(). Performs:
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * @param aicpu_so_binary       Binary data of AICPU shared object
     * @param aicore_kernel_binary  Binary data of AICore kernel
     * @return 0 on success, error code on failure
     */
    int ensure_binaries_loaded(
        const std::vector<uint8_t> &aicpu_so_binary, const std::vector<uint8_t> &aicore_kernel_binary
    );

    /**
     * Populate runtime.{dev_orch_so_addr_, dev_orch_so_size_, has_new_orch_so_}
     * from `runtime.pending_orch_so_data_` / `_size_`.
     *
     * The host tracks the SO identity via a 64-bit hash derived from the ELF
     * GNU Build-ID. When the hash matches the previous run, the device-side
     * AICPU thread reuses its cached dlopen handle and we skip the rtMemcpy
     * entirely. On a miss we allocate (or grow) a device-resident buffer,
     * copy the SO bytes once, and update the cached hash.
     *
     * @param runtime  Runtime whose device-SO metadata will be rewritten.
     * @return 0 on success, non-zero on failure.
     */
    int prepare_orch_so(Runtime &runtime);

    /**
     * Initialize performance profiling shared memory
     *
     * Allocates device memory, maps to host for shared access, and initializes
     * performance data structures (header and double buffers).
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID for host registration
     * @return 0 on success, error code on failure
     */
    int init_l2_perf_collection(int num_aicore, int device_id);

    /**
     * Initialize tensor dump shared memory and collector.
     *
     * Allocates dump SHM + per-thread arenas, populates initial meta buffers,
     * and stores the dump base in AICPU launch arguments.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances (unused; dump is per-thread)
     * @param device_id Device ID for host registration
     * @return 0 on success, error code on failure
     */
    int init_tensor_dump(Runtime &runtime, int num_aicore, int device_id);

    /**
     * Initialize PMU streaming shared memory.
     *
     * Allocates PmuDataHeader + PmuBufferState array + pre-allocated PmuBuffers,
     * registers them via halHostRegister, and stores the header address in
     * kernel_args.pmu_data_base.
     *
     * @param num_cores  Number of AICore instances
     * @param num_threads Number of AICPU scheduling threads
     * @param csv_path   Output CSV file path
     * @param event_type PMU event type (written to CSV rows)
     * @param device_id  Device ID for host registration
     * @return 0 on success, error code on failure
     */
    int init_pmu_buffers(
        int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
    );
    // Enablement for the three diagnostics sub-features. Written by the c_api
    // entry point via set_enable_*() before run(), read inside run() and its
    // helpers. Moved off Runtime / run() args so all three sub-features use
    // the same plumbing shape.
    bool enable_l2_swimlane_{false};
    bool enable_dump_tensor_{false};
    bool enable_pmu_{false};
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};  // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                  // diagnostic artifact root directory
    int log_level_{1};                                             // 0=DEBUG..4=NUL; default INFO
    int log_info_v_{5};                                            // INFO verbosity threshold; default V5
};

#endif  // RUNTIME_DEVICERUNNER_H
