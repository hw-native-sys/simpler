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
#include "prepare_callable_common.h"
#include "common/kernel_args.h"
#include "common/memory_barrier.h"
#include "common/l2_perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "device_arena.h"
#include "device_runner_base.h"     // common DeviceRunnerBase
#include "device_runner_helpers.h"  // common DeviceArgs + KernelArgsHelper
#include "host/function_cache.h"
#include "host/memory_allocator.h"
#include "host/l2_perf_collector.h"
#include "host/tensor_dump_collector.h"
#include "host/pmu_collector.h"
#include "host/dep_gen_collector.h"
#include "load_aicpu_op.h"
#include "host/scope_stats_collector.h"
#include "runtime.h"

/**
 * a2a3-only `KernelArgsHelper` extension: retrieve the FFTS base address via
 * `rtGetC2cCtrlAddr` and store it in the wrapped `KernelArgs`. a5's
 * `KernelArgs` has no `ffts_base_addr` field, so this helper lives in the
 * arch-specific header rather than on the common `KernelArgsHelper` struct.
 *
 * @return 0 on success, error code on failure.
 */
int kernel_args_init_ffts_base_addr(KernelArgsHelper &helper);

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
class DeviceRunner : public DeviceRunnerBase {
public:
    DeviceRunner() = default;
    ~DeviceRunner();

    /**
     * Commit the three per-Worker pooled regions (PTO2 GM heap, PTO2 shared
     * memory, trb prebuilt runtime arena) as three independent device
     * allocations. Must be called before any acquire_pooled_*. Idempotent
     * on identical sizes. `runtime_arena_size` is 0 for the hbg path (no
     * prebuilt runtime arena) — the corresponding arena stays uncommitted.
     * Returns 0 on success, -1 on failure.
     *
     * `allocate_tensor`, `free_tensor`, `copy_to_device`, `copy_from_device`,
     * `acquire_pooled_{gm_heap,gm_sm,runtime_arena}`, `create_thread`,
     * `attach_current_thread`, `ensure_device_initialized`,
     * `print_handshake_results`, `set_executors`, `set_dispatcher_binary`,
     * `device_id`, and `last_device_wall_ns` are inherited from
     * `DeviceRunnerBase`.
     */
    int setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

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
     * Enablement setters for the three diagnostics sub-features. Called by
     * the c_api entry point before run(); downstream run() paths read the
     * corresponding `enable_*_` members directly. Moved off the generic
     * Runtime struct / run() arg list so all three travel the same way.
     */
    void set_l2_swimlane_enabled(int level) {
        l2_perf_level_ = static_cast<L2PerfLevel>(level);
        enable_l2_swimlane_ = (l2_perf_level_ != L2PerfLevel::DISABLED);
    }
    void set_dump_tensor_enabled(bool enable) { enable_dump_tensor_ = enable; }
    void set_pmu_enabled(int enable_pmu) {
        enable_pmu_ = (enable_pmu > 0);
        pmu_event_type_ = resolve_pmu_event_type(enable_pmu);
    }
    void set_dep_gen_enabled(bool enable) { enable_dep_gen_ = enable; }
    void set_scope_stats_enabled(bool enable) { enable_scope_stats_ = enable; }
    // Directory under which all diagnostic artifacts (l2_perf_records.json /
    // tensor_dump/ / pmu.csv) land. Required (non-empty) when any diagnostic
    // is enabled; CallConfig::validate() enforces this contract upstream.
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

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

    // `upload_chip_callable_buffer` is inherited from `DeviceRunnerBase`.

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

    // `register_callable`, `register_callable_host_orch`,
    // `unregister_callable`, `has_callable`, `bind_callable_to_runtime`,
    // `aicpu_dlopen_count`, and `host_dlopen_count` are inherited from
    // `DeviceRunnerBase`.

private:
    // Most lifecycle state (device_id_, block_dim_, cores_per_blockdim_,
    // worker_count_, executor + dispatcher bytes, aicore_bin_handle_,
    // load_aicpu_op_, mem_alloc_, the three DeviceArenas, persistent
    // AICPU/AICore streams, kernel_args_, device_wall_*, device_args_,
    // binaries_loaded_) is inherited from `DeviceRunnerBase`.
    //
    // Arena cached sizes for setup_static_arena's "fits" check — avoids
    // re-allocating the same buffer when a later worker init asks for an
    // equal-or-smaller layout on an already-committed arena.
    size_t cached_gm_heap_size_{0};
    size_t cached_gm_sm_size_{0};
    size_t cached_runtime_arena_size_{0};

    // Group D state (`chip_callable_buffers_`, `callables_`,
    // `orch_so_dedup_`, `aicpu_seen_callable_ids_`, `aicpu_dlopen_total_`,
    // `host_dlopen_total_`) and inner struct types
    // (`ChipCallableBuffer`, `CallableState`, `OrchSoBuffer`) are
    // inherited from `DeviceRunnerBase`.

    // ACL lifecycle (process-wide). aclInit must run exactly once; ensure_acl_ready
    // gates it behind this flag. finalize() drives aclFinalize only if we observed
    // acl_ready_, so runtimes that never ask for ACL (e.g. pure rt-layer) stay unaffected.
    bool acl_ready_{false};

    // Performance profiling
    L2PerfCollector l2_perf_collector_;

    // Tensor dump (independent shared memory + memory manager)
    TensorDumpCollector dump_collector_;
    // PMU collector (independent of profiling pipeline)
    PmuCollector pmu_collector_;
    // dep_gen collector — captures orchestrator submit_task inputs for offline replay
    DepGenCollector dep_gen_collector_;

    // `query_max_block_dim`, `validate_block_dim`, `ensure_binaries_loaded`,
    // `configure_aicore_op_timeout`, and `prepare_orch_so` are inherited
    // (protected) from `DeviceRunnerBase`.

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
    int init_l2_perf(int num_aicore, int device_id);

    /**
     * Initialize tensor dump shared memory and collector.
     *
     * Allocates dump SHM + per-thread arenas, populates initial meta buffers,
     * and stores the dump base in AICPU launch arguments.
     *
     * @param runtime Runtime instance to configure
     * @param device_id Device ID for host registration
     * @return 0 on success, error code on failure
     */
    int init_tensor_dump(Runtime &runtime, int device_id);

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
    int init_pmu(int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type, int device_id);

    /**
     * Initialize dep_gen capture shared memory.
     *
     * Allocates DepGenDataHeader + 1 DepGenBufferState + N DepGenBuffers,
     * registers them via halHostRegister, and stores the header address in
     * kernel_args.dep_gen_data_base.
     *
     * @param num_threads        Number of AICPU scheduling threads
     * @param submit_trace_path  Output binary file path (.bin)
     * @param device_id          Device ID for host registration
     * @return 0 on success, error code on failure
     */
    int init_dep_gen(int num_threads, int device_id);
    int init_scope_stats(int num_threads, int device_id);

    /**
     * Finalize whichever diagnostics collectors are currently initialized,
     * releasing their device/host shared memory back to mem_alloc_.
     *
     * Idempotent and safe to call multiple times: each collector's finalize()
     * early-outs once its shm has been released. Invoked both at the end of
     * every run() (so a Worker reused across runs starts each run with the
     * collectors in a pristine, re-initializable state) and from finalize()
     * as a backstop before mem_alloc_.finalize().
     */
    void finalize_collectors();
    // Enablement for the three diagnostics sub-features. Written by the c_api
    // entry point via set_enable_*() before run(), read inside run() and its
    // helpers. Moved off Runtime / run() args so all three sub-features use
    // the same plumbing shape.
    bool enable_l2_swimlane_{false};
    bool enable_dump_tensor_{false};
    bool enable_pmu_{false};
    bool enable_dep_gen_{false};
    bool enable_scope_stats_{false};
    ScopeStatsCollector scope_stats_collector_;
    L2PerfLevel l2_perf_level_{L2PerfLevel::DISABLED};             // resolved from set_l2_swimlane_enabled()
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};  // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                  // diagnostic artifact root directory
};

#endif  // RUNTIME_DEVICERUNNER_H
