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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "callable.h"
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/l2_perf_collector.h"
#include "host/pmu_collector.h"
#include "host/tensor_dump_collector.h"
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
     * @param launch_aicpu_num     Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     *
     * The bound device id, AICPU/AICore executor binaries, and log filter
     * are captured once by simpler_init (binaries) / libsimpler_log.so (log)
     * and read off DeviceRunner state / HostLogger here — no per-run args.
     */
    int run(Runtime &runtime, int block_dim, int launch_aicpu_num = 1);

    /**
     * Take ownership of the AICPU + AICore executor binaries. Called once
     * by simpler_init at ChipWorker::init time; subsequent run() invocations
     * read from `aicpu_so_binary_` / `aicore_kernel_binary_`.
     */
    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }

    /** The device id captured by simpler_init's attach_current_thread call. */
    int device_id() const { return device_id_; }

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
     * Poll and collect performance data from the memory manager's queue
     *
     * Runs on a dedicated collector thread. Pulls ready buffers from
     * ProfMemoryManager, copies records to host vectors, and notifies
     * the manager to free old device buffers.
     *
     * @param expected_tasks Expected total number of tasks
     */
    void poll_and_collect_performance_data(int expected_tasks);

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
     * @param runtime   Pointer to device runtime
     * @return 0 on success, error code on failure
     */
    int launch_aicore_kernel(rtStream_t stream, Runtime *runtime);

    /**
     * Upload an entire ChipCallable buffer to device memory in one shot.
     *
     * Walks child_offsets_ to compute total byte size, allocates device GM
     * once, fixes up each child's resolved_addr_ in an internal host scratch
     * (= device-side address of that child's binary code), H2D's once, and
     * returns the device-side address of the ChipCallable header.
     *
     * Pool-managed: identical buffer bytes (FNV-1a 64-bit content hash) hit
     * the dedup cache and return the cached chip_dev without reallocating.
     * All chip buffers are bulk-freed in finalize() — there is no explicit
     * free API, mirroring the per-fid binary pool semantics.
     *
     * @return Device GM address of the ChipCallable header, or 0 on failure
     *         (also returns 0 when callable->child_count() == 0).
     */
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);

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

    /**
     * Stage a per-callable_id orchestration SO into device memory and remember
     * the supporting metadata (entry/config symbol names, kernel func_id ↔
     * dev_addr table). Identical SO bytes across two callable_ids share one
     * device buffer (refcounted by hash) so the worst case for an N-cid pool
     * is N distinct device buffers, not N copies of the same SO.
     *
     * @param callable_id   Caller-stable id, must be in [0, MAX_REGISTERED_CALLABLE_IDS).
     * @param orch_so_data  Host pointer to orchestration SO bytes (owned by caller).
     * @param orch_so_size  Size of orchestration SO in bytes.
     * @param func_name     Entry symbol name (copied).
     * @param config_name   Config symbol name (copied).
     * @param kernel_addrs  func_id ↔ dev_addr pairs already uploaded by the
     *                      caller. Stored verbatim so run_prepared can replay
     *                      them onto a fresh Runtime without re-uploading.
     * @return 0 on success, negative on failure.
     */
    int register_prepared_callable(
        int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name,
        const char *config_name, std::vector<std::pair<int, uint64_t>> kernel_addrs
    );

    /**
     * Host-orchestration sibling for hbg variants. See a2a3 onboard
     * device_runner.h for full contract. Mutually exclusive with the
     * trb-shaped overload.
     */
    int register_prepared_callable_host_orch(
        int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs
    );

    /**
     * Drop the prepared state for `callable_id`. trb path: decrement orch SO
     * refcount, free when zero. hbg path: dlclose the host handle. Kernel
     * binaries are shared and only released by finalize().
     */
    int unregister_prepared_callable(int32_t callable_id);

    /** True iff `callable_id` has prepared state staged. */
    bool has_prepared_callable(int32_t callable_id) const;

    /**
     * Replay the prepared state for `callable_id` onto a freshly-constructed
     * Runtime. See a2a3 onboard documentation for full contract.
     */
    int bind_prepared_callable_to_runtime(Runtime &runtime, int32_t callable_id, void **out_host_orch_func_ptr);

    /**
     * Number of distinct callable_ids the AICPU has been asked to dlopen for.
     * Monotonically increases on first-sighting bind; never decremented.
     */
    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }

    /**
     * Number of host-side dlopens triggered by
     * `register_prepared_callable_host_orch` (hbg variant). Mirrors
     * `aicpu_dlopen_count` for the host-orchestration path.
     */
    size_t host_dlopen_count() const { return host_dlopen_total_; }

private:
    // Internal state. device_id_ is set once in attach_current_thread() (called
    // from simpler_init during ChipWorker::init) and read on every subsequent
    // op. All ChipWorker callers run on the same thread that called init, so
    // plain int + the init→user happens-before edge is sufficient.
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};  // Stored for print_handshake_results in destructor
    // Executor binaries — populated once via set_executors() during
    // simpler_init, owned by this runner for the rest of its lifetime.
    std::vector<uint8_t> aicpu_so_binary_;
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
    bool binaries_loaded_{false};  // true after AICPU SO loaded

    // Chip-callable buffer pool. Keyed by FNV-1a 64-bit content hash of the
    // ChipCallable bytes. Each entry owns one device GM allocation holding
    // the entire ChipCallable buffer (header + storage_, with each child's
    // resolved_addr_ fixed up to its post-H2D device address). Pool-managed:
    // identical buffer bytes share one entry across cids; the map is bulk-
    // freed in finalize(). No explicit free API (mirrors per-fid binary pool
    // semantics today).
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};
        size_t total_size{0};
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Orchestration SO cache (host-tracked, device-resident).
    uint64_t cached_orch_so_hash_{0};
    void *dev_orch_so_buffer_{nullptr};
    size_t dev_orch_so_capacity_{0};
    std::vector<uint8_t> host_orch_so_copy_;

    // Per-callable_id prepared state. See a2a3 onboard device_runner.h for
    // the full design narrative; mirrored here so a5 shares the same
    // dispatch surface.
    struct PreparedCallableState {
        // trb path
        uint64_t hash{0};
        uint64_t dev_orch_so_addr{0};
        size_t dev_orch_so_size{0};
        std::string func_name;
        std::string config_name;
        // common
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        // hbg path
        void *host_dlopen_handle{nullptr};
        void *host_orch_func_ptr{nullptr};
    };
    struct OrchSoBuffer {
        void *dev_addr{nullptr};
        size_t capacity{0};
        int refcount{0};
    };
    std::unordered_map<int32_t, PreparedCallableState> prepared_callables_;
    std::unordered_map<uint64_t, OrchSoBuffer> orch_so_dedup_;
    std::unordered_set<int32_t> aicpu_seen_callable_ids_;
    // Monotonic AICPU dlopen counter (first-sighting bind only; never decremented).
    size_t aicpu_dlopen_total_{0};
    // Monotonic host-side dlopen counter for hbg variants.
    size_t host_dlopen_total_{0};
    // Performance profiling
    L2PerfCollector l2_perf_collector_;

    // Tensor dump (independent from profiling)
    TensorDumpCollector dump_collector_;

    // PMU profiling (per-task AICore hardware counters)
    PmuCollector pmu_collector_;

    /**
     * Ensure device is initialized (lazy initialization)
     *
     * Checks if device is already initialized. If not, performs:
     * - Attach the current thread to the device
     * - Create AICPU and AICore streams
     * - Load AICPU SO to device memory
     * - Initialize device args
     *
     * Reads the bound device id and executor binaries from runner state.
     * @return 0 on success, error code on failure
     */
    int ensure_device_initialized();

    /**
     * Validate block_dim against the stream's CUBE/VECTOR core limits
     * (aclrtGetStreamResLimit). When that query is unavailable or reports no
     * cores, falls back to the static PLATFORM_MAX_BLOCKDIM cap.
     * Returns 0 if block_dim fits, -1 otherwise (or if block_dim < 1).
     */
    int validate_block_dim(rtStream_t stream, int block_dim);

    /**
     * Load AICPU SO and initialize device args
     *
     * Called by run() after prepare_run_context(). Reads aicpu_so_binary_ /
     * aicore_kernel_binary_ off the runner.
     *
     * @return 0 on success, error code on failure
     */
    int ensure_binaries_loaded();

    /**
     * Stage the orchestration SO into a device-resident buffer (with hash
     * cache). See a2a3 onboard documentation for details.
     */
    int prepare_orch_so(Runtime &runtime);

    /**
     * Initialize performance profiling device buffers
     *
     * Allocates L2PerfSetupHeader and per-core/per-thread buffers on device;
     * caller publishes the device pointer via kernel_args.l2_perf_data_base
     * (AICPU reads it through get_platform_l2_perf_base()).
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances
     * @param device_id Device ID
     * @return 0 on success, error code on failure
     */
    int init_l2_perf_collection(int num_aicore, int device_id);

    /**
     * Initialize tensor dump device buffers.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of AICore instances (unused)
     * @param device_id Device ID for allocations
     * @return 0 on success, error code on failure
     */
    int init_tensor_dump(Runtime &runtime, int num_aicore, int device_id);

    /**
     * Initialize PMU profiling device buffers.
     *
     * Allocates a PmuDataHeader and one PmuBuffer per core on device, then
     * publishes the data-header pointer into kernel_args.pmu_data_base.
     * Signature matches a2a3 for cross-platform consistency.
     */
    // Enablement for the three diagnostics sub-features. Written by the c_api
    // entry point via set_enable_*() before run(), read inside run() and its
    // helpers. Moved off Runtime / run() args so all three sub-features use
    // the same plumbing shape.
    bool enable_l2_swimlane_{false};
    bool enable_dump_tensor_{false};
    bool enable_pmu_{false};
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};  // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                  // diagnostic artifact root directory

    int init_pmu_buffers(
        int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id
    );
};

#endif  // RUNTIME_DEVICERUNNER_H
