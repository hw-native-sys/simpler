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
 * Device Runner - Thread-Based Simulation
 *
 * This module simulates the Ascend AICPU/AICore execution model using threads.
 * It provides the SAME interface as the real a5 DeviceRunner, ensuring
 * API compatibility with Python bindings and examples.
 *
 * Key differences from real a5:
 * - Uses host memory instead of device memory
 * - Uses std::thread instead of CANN kernel launches
 * - Kernel .text binaries are loaded into executable memory (mmap)
 */

#ifndef SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_
#define SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "callable.h"
#include "common/core_type.h"
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
 * Device runner for simulated kernel execution
 *
 * This class provides the SAME interface as the real a5 DeviceRunner,
 * but implements execution using host threads instead of actual device
 * kernel launches.
 *
 * Key simulation features:
 * - Memory operations use host memory (malloc/free/memcpy)
 * - Kernel execution uses std::thread
 * - Kernel .text binaries are loaded into mmap'd executable memory
 */
class DeviceRunner {
public:
    DeviceRunner() = default;
    ~DeviceRunner();

    /**
     * Create a thread bound to this device.
     * The thread calls pto_cpu_sim_bind_device(device_id) on entry
     * and unbinds on exit.
     */
    std::thread create_thread(std::function<void()> fn);

    /**
     * Allocate tensor memory (host memory in simulation)
     *
     * @param bytes  Size of tensor in bytes
     * @return Pointer on success, nullptr on failure
     */
    void *allocate_tensor(size_t bytes);

    /**
     * Free tensor memory
     *
     * @param dev_ptr  Pointer to free
     */
    void free_tensor(void *dev_ptr);

    /**
     * Copy data (memcpy in simulation)
     *
     * @param dev_ptr   Destination pointer
     * @param host_ptr  Source pointer
     * @param bytes     Number of bytes to copy
     * @return 0 on success
     */
    int copy_to_device(void *dev_ptr, const void *host_ptr, size_t bytes);

    /**
     * Copy data (memcpy in simulation)
     *
     * @param host_ptr  Destination pointer
     * @param dev_ptr   Source pointer
     * @param bytes     Number of bytes to copy
     * @return 0 on success
     */
    int copy_from_device(void *host_ptr, const void *dev_ptr, size_t bytes);

    /**
     * Execute a runtime using threads
     *
     * This method simulates the complete execution:
     * 1. Initializes worker handshake buffers
     * 2. Sets function_bin_addr for all tasks
     * 3. Launches AICPU threads
     * 4. Launches AICore threads
     * 5. Waits for all threads to complete
     *
     * @param runtime              Runtime to execute
     * @param block_dim            Number of blocks (1 block = 1 AIC + 2 AIV)
     * @param launch_aicpu_num     Number of AICPU threads
     * @return 0 on success
     *
     * Bound device id, AICPU/AICore executor binaries, and log filter are
     * captured once by simpler_init / libsimpler_log.so and read off
     * DeviceRunner state / HostLogger here — no per-run args.
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
     * Attach the calling thread to the simulated device.
     *
     * Mirrors the onboard contract: binds the caller's TLS to `device_id`
     * and idempotently acquires the process-wide sim device registry entry.
     * Called from `simpler_init` and re-invoked at the top of every device-op
     * so any caller thread becomes the bound thread for the op without
     * requiring an explicit pre-attach step.
     *
     * @param device_id Device ID (>= 0).
     * @return 0 on success, negative on invalid id / device-id mismatch.
     */
    int attach_current_thread(int device_id);

    /**
     * Print handshake results
     */
    void print_handshake_results();

    /**
     * Cleanup all resources
     *
     * Use this for final cleanup when no more tests will run.
     *
     * @return 0 on success
     */
    int finalize();

    /**
     * Upload an entire ChipCallable buffer (sim path). See a2a3 sim
     * device_runner.h for the full contract. Returns the host scratch
     * address (== chip_dev in sim since host/device memory is unified);
     * caller computes per-child addrs via
     *     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i).
     */
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);

    /**
     * Stage a per-callable_id orchestration SO and its supporting metadata.
     * See a5 onboard or a2a3 device_runner.h for full contract.
     */
    int register_prepared_callable(
        int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name,
        const char *config_name, std::vector<std::pair<int, uint64_t>> kernel_addrs
    );

    /** Host-orchestration sibling for hbg variants. See a2a3 onboard. */
    int register_prepared_callable_host_orch(
        int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs
    );

    /** Drop prepared state for `callable_id`; trb refcounts SO, hbg dlcloses handle. */
    int unregister_prepared_callable(int32_t callable_id);

    /** True iff `callable_id` has prepared state staged. */
    bool has_prepared_callable(int32_t callable_id) const;

    /** Replay prepared state onto a freshly-constructed Runtime. */
    int bind_prepared_callable_to_runtime(Runtime &runtime, int32_t callable_id);

    /** Monotonic AICPU dlopen counter (first-sighting only; never decremented). */
    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }

    /** Monotonic host-side dlopen counter for hbg variants. */
    size_t host_dlopen_count() const { return host_dlopen_total_; }

private:
    // Configuration. device_id_ is set once in attach_current_thread() during
    // simpler_init and read by run() / create_thread() afterward — single-
    // threaded with respect to the user's call sequence, so plain int is
    // sufficient.
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};
    // Executor binaries — populated once via set_executors() during
    // simpler_init, owned by this runner for the rest of its lifetime.
    std::vector<uint8_t> aicpu_so_binary_;
    std::vector<uint8_t> aicore_kernel_binary_;

    // Memory management
    MemoryAllocator mem_alloc_;

    // Simulation state (no actual device resources)
    KernelArgs kernel_args_;

    // Chip-callable buffer pool (sim path). Keyed by FNV-1a 64-bit content
    // hash; each entry owns the host scratch + dlopen handles needed for the
    // children embedded in that buffer. Bulk-freed in finalize().
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // (uint64_t)host_scratch
        uint8_t *host_scratch{nullptr};
        size_t total_size{0};
        std::vector<void *> dlopen_handles;
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Orchestration SO cache (host-resident in sim).
    uint64_t cached_orch_so_hash_{0};
    void *dev_orch_so_buffer_{nullptr};
    size_t dev_orch_so_capacity_{0};
    std::vector<uint8_t> host_orch_so_copy_;

    // Per-callable_id prepared state. Mirrors onboard.
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
    size_t aicpu_dlopen_total_{0};
    size_t host_dlopen_total_{0};
    // Runtime pointer for print_handshake_results
    Runtime *last_runtime_{nullptr};

    // Dynamically loaded executor libraries and function pointers
    void *aicpu_so_handle_{nullptr};
    bool aicpu_so_loaded_{false};  // true after AICPU SO is dlopen'd; load-once across runs.
    void *aicore_so_handle_{nullptr};
    int (*aicpu_execute_func_)(Runtime *){nullptr};
    void (*aicore_execute_func_)(Runtime *, int, CoreType, uint32_t, uint64_t){nullptr};
    void (*set_platform_regs_func_)(uint64_t){nullptr};
    void (*set_platform_dump_base_func_)(uint64_t){nullptr};
    void (*set_platform_pmu_base_func_)(uint64_t){nullptr};
    void (*set_dump_tensor_enabled_func_)(bool){nullptr};
    void (*set_platform_l2_perf_base_func_)(uint64_t){nullptr};
    void (*set_l2_swimlane_enabled_func_)(bool){nullptr};
    void (*set_pmu_enabled_func_)(bool){nullptr};
    std::string aicpu_so_path_;
    std::string aicore_so_path_;

    // Performance profiling
    L2PerfCollector l2_perf_collector_;

    // Tensor dump (independent from profiling)
    TensorDumpCollector dump_collector_;

    // PMU profiling (per-task AICore hardware counters)
    PmuCollector pmu_collector_;

    // Private helper methods — read aicpu_so_binary_ / aicore_kernel_binary_
    // off the runner (populated by set_executors during simpler_init).
    int ensure_device_initialized();
    int ensure_binaries_loaded();
    void unload_executor_binaries();

    /**
     * Stage the orchestration SO bytes for aicpu_executor consumption, with
     * identity caching. See a2a3/sim/host/device_runner.h.
     */
    int prepare_orch_so(Runtime &runtime);

    /**
     * Initialize performance profiling shared memory
     *
     * Allocates and initializes host memory for performance profiling.
     *
     * @param runtime Runtime instance to configure
     * @param num_aicore Number of cores
     * @param device_id Device ID (ignored in simulation)
     * @return 0 on success, error code on failure
     */
    int init_l2_perf_collection(int num_aicore, int device_id);

    /**
     * Initialize tensor dump for simulation.
     */
    int init_tensor_dump(Runtime &runtime, int num_aicore, int device_id);

    /**
     * Initialize PMU profiling buffers for simulation.
     *
     * Allocates PmuDataHeader + per-core PmuBuffer on host memory, publishes
     * the header pointer into kernel_args.pmu_data_base. pmu_reg_addrs
     * stays 0 on sim (no hardware PMU model).
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

#endif  // SRC_A5_PLATFORM_SIM_HOST_DEVICE_RUNNER_H_
