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

    /**
     * Upload an entire ChipCallable buffer to device memory in one shot.
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
     * Callers compute child addresses as
     *     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
     * and write them to Runtime::func_id_to_addr_[fid] via
     * Runtime::set_function_bin_addr().
     *
     * @param callable  Host-side ChipCallable pointer.
     * @return Device GM address of the ChipCallable header, or 0 on failure
     *         (also returns 0 when callable->child_count() == 0).
     */
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);

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
        const char *config_name, std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );

    /**
     * Host-orchestration variant of register_prepared_callable: stores a
     * dlopen handle + entry-symbol pointer that runtime_maker resolved on the
     * host (host_build_graph variant). Mutually exclusive with the trb-shaped
     * `register_prepared_callable` overload — exactly one is invoked for a
     * given callable_id, picked by the C ABI based on which staging fields the
     * runtime carries after prepare_callable_impl. dlopen handle is owned by
     * DeviceRunner from this call onward and dlclose'd by
     * unregister_prepared_callable. Increments `host_dlopen_count_`.
     */
    int register_prepared_callable_host_orch(
        int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );

    /**
     * Drop the prepared state for `callable_id`. trb path: decrement the orch
     * SO buffer's hash-keyed refcount and free when it hits zero. hbg path:
     * dlclose the host dlopen handle. Kernel binaries are shared across
     * callables and only released by finalize().
     *
     * @param callable_id  Id previously passed to one of the
     *                     register_prepared_callable* overloads.
     * @return 0 on success or if the id was not registered.
     */
    int unregister_prepared_callable(int32_t callable_id);

    /**
     * True iff `callable_id` has prepared state staged via
     * register_prepared_callable. Lets the c_api layer reject `run_prepared`
     * calls without a matching `prepare_callable`.
     */
    bool has_prepared_callable(int32_t callable_id) const;

    /**
     * Replay the prepared state for `callable_id` onto a freshly-constructed
     * Runtime: restores kernel func_id ↔ dev_addr table, the orch entry/config
     * symbol names, and stamps `runtime.set_active_callable_id` so the
     * subsequent `run` dispatches via the AICPU per-cid table. The kernel
     * addresses are written directly into func_id_to_addr_ (bypassing
     * registered_kernel_func_ids_) so validate_runtime_impl will not free them
     * — they survive until unregister_prepared_callable / finalize().
     *
     * Marks the cid as seen so the upcoming prepare_orch_so resolves
     * `register_new_callable_id_` correctly (true exactly on first sighting
     * after registration).
     *
     * @return 0 on success, -1 if the cid is not registered.
     */
    /**
     * Replay a previously-registered callable's state onto a fresh Runtime
     * for a per-run binding. Writes back kernel addrs, orch entry-symbol
     * names, and active_callable_id; returns the hbg `host_orch_func_ptr`
     * (or nullptr on trb / on error) inside a `BindPreparedCallableResult`
     * so the caller can destructure with structured bindings.
     */
    BindPreparedCallableResult bind_prepared_callable_to_runtime(Runtime &runtime, int32_t callable_id);

    /**
     * Number of distinct callable_ids the AICPU has been asked to dlopen for.
     * Monotonically increases on every first-sighting bind; `unregister_callable`
     * does NOT decrement it. So a `prepare → run → unregister → re-prepare → run`
     * sequence reports 2 (each AICPU dlopen counted once), even though only one
     * cid is currently registered. Tests assert this to verify per-cid
     * registration eliminates duplicate dlopens across repeated runs.
     */
    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }

    /**
     * Number of host-side dlopen() invocations triggered by
     * `register_prepared_callable_host_orch`. Mirrors `aicpu_dlopen_count` but
     * counts the host_build_graph variant's host-side dlopens; it never
     * decrements (re-prepare after unregister still counts). Tests assert
     * `host_dlopen_count == distinct_registered_cids` to verify the prepared
     * path doesn't dlopen on every run.
     */
    size_t host_dlopen_count() const { return host_dlopen_total_; }

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

    // Chip-callable buffer pool. Keyed by FNV-1a 64-bit content hash of the
    // ChipCallable bytes. Each entry owns one device GM allocation holding
    // the entire ChipCallable buffer (header + storage_, with each child's
    // resolved_addr_ fixed up to its post-H2D device address). Pool-managed:
    // identical buffer bytes share one entry across cids; the map is bulk-
    // freed in finalize(). No explicit free API (mirrors per-fid binary pool
    // semantics today).
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // device GM address of the ChipCallable header
        size_t total_size{0};  // byte size of the device allocation
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id prepared state.
    //
    // `prepared_callables_` maps the caller-stable callable_id to the orch
    // SO slice + symbol names needed to launch it. `orch_so_dedup_` shares
    // device buffers across callable_ids whose orch SO bytes have the same
    // ELF Build-ID hash (refcounted; freed when the count hits zero).
    // `aicpu_seen_callable_ids_` tracks which ids have already been delivered
    // to the AICPU at least once so prepare_orch_so can set
    // register_new_callable_id_ correctly on first sighting.
    struct PreparedCallableState {
        // trb path (AICPU dlopens orch SO from device buffer)
        uint64_t hash{0};
        uint64_t dev_orch_so_addr{0};
        size_t dev_orch_so_size{0};
        std::string func_name;
        std::string config_name;
        // common
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        std::vector<ArgDirection> signature;
        // hbg path (host already dlopen'd the orch SO)
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
    // Monotonic count of AICPU dlopens triggered (incremented on each
    // first-sighting bind; never decremented). Diverges from
    // aicpu_seen_callable_ids_.size() once any cid is unregistered and
    // re-prepared. Exposed via aicpu_dlopen_count() for tests.
    size_t aicpu_dlopen_total_{0};
    // Monotonic count of host-side dlopens triggered (incremented on every
    // register_prepared_callable_host_orch call; never decremented). Same
    // re-prepare semantics as aicpu_dlopen_total_, but for hbg variants.
    size_t host_dlopen_total_{0};
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
    // and `configure_aicore_op_timeout` are inherited (protected) from
    // `DeviceRunnerBase`.

    /**
     * Stamp `runtime.{dev_orch_so_addr_, dev_orch_so_size_}` from the
     * PreparedCallableState for `runtime.get_active_callable_id()`. The orch
     * SO bytes were already H2D'd at `register_prepared_callable` time and
     * are shared via `orch_so_dedup_` across cids; this method only refreshes
     * the device-SO metadata onto the per-run Runtime and bumps the AICPU
     * first-sighting counter when the cid is new since registration.
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
