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
 * Onboard host `DeviceRunnerBase` — common base class for a2a3 and a5
 * onboard `DeviceRunner`s.
 *
 * This module owns the host-side state and methods that are identical
 * between the two onboard arches today:
 *   - The `MemoryAllocator` and the three `DeviceArena`s (gm heap, PTO2
 *     SM, runtime arena) backing the per-Worker pooled regions.
 *   - The trivial tensor-memory wrappers (`allocate_tensor`,
 *     `free_tensor`, `copy_*_device`).
 *   - The arena-pool accessors (`acquire_pooled_gm_heap`, etc.).
 *   - Device lifecycle: `attach_current_thread`,
 *     `configure_aicore_op_timeout`, `ensure_device_initialized`,
 *     `ensure_binaries_loaded`, persistent AICPU/AICore streams,
 *     dispatcher/executor bytes, `LoadAicpuOp`, `KernelArgsHelper`.
 *   - block_dim resolution: `query_max_block_dim`, `validate_block_dim`.
 *   - Debug: `print_handshake_results`, `create_thread`.
 *
 * Subclasses (`{a2a3,a5}::DeviceRunner`) add arch-specific state
 * (callable registry, profiling collectors, ACL/HCCL plumbing on a2a3,
 * `enable_*` flags) and the divergent methods (`run`, `finalize`,
 * `setup_static_arena`, the kernel launch / chip-callable upload, the
 * per-callable registration helpers, and the per-diagnostic `init_*`).
 */

#ifndef SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H
#define SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H

#include <runtime/rt.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arg_direction.h"
#include "callable.h"
#include "common/l2_perf_profiling.h"
#include "device_arena.h"
#include "device_runner_helpers.h"
#include "host/load_aicpu_op.h"
#include "host/l2_perf_collector.h"
#include "host/memory_allocator.h"
#include "host/pmu_collector.h"
#include "host/scope_stats_collector.h"
#include "host/tensor_dump_collector.h"
#include "prepare_callable_common.h"

/**
 * Common base class for both a2a3 and a5 onboard `DeviceRunner`s.
 *
 * Ctor + dtor are `protected` so this class can only be used as a base;
 * direct instantiation and `delete` through a base pointer are both
 * compile errors. The arch subclass's `DeviceRunner` is what
 * `destroy_device_context` sees, so the non-virtual `~DeviceRunnerBase`
 * is safe — it never runs as a virtual base destructor.
 */
class DeviceRunnerBase {
public:
    DeviceRunnerBase(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase &operator=(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase(DeviceRunnerBase &&) = delete;
    DeviceRunnerBase &operator=(DeviceRunnerBase &&) = delete;

    /** Allocate / free / copy on the per-Worker `MemoryAllocator` + CANN runtime. */
    void *allocate_tensor(std::size_t bytes);
    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes);

    /**
     * Commit the three per-Worker pooled regions (PTO2 GM heap, PTO2
     * shared memory, trb prebuilt runtime arena) as three independent
     * device allocations. Must be called before any `acquire_pooled_*`.
     * Idempotent on identical (or smaller) sizes; an equal-or-smaller
     * follow-up request leaves the arena untouched. `runtime_arena_size`
     * is 0 for the hbg path (no prebuilt runtime arena) — the
     * corresponding arena stays uncommitted.
     *
     * On failure to commit a later region, earlier committed regions are
     * rolled back (a5's prior semantics). This is the safer default: a
     * partial commit otherwise leaves the caller with pooled pointers
     * that survive a "failure" return, masking the real error and risking
     * later mismatched-arena bugs. (The a2a3 implementation that
     * previously kept earlier committed peers alive on failure is
     * normalized away.)
     *
     * @return 0 on success, -1 on failure.
     */
    int setup_static_arena(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size);

    /**
     * Return the pooled GM heap / PTO2 SM / runtime arena base pointer.
     * `setup_static_arena` (arch subclass) must have already committed
     * the relevant region; otherwise returns nullptr. The runtime arena
     * accessor is trb-only — hbg's `setup_static_arena(...,0)` leaves
     * `runtime_arena_pool_` uncommitted and this returns nullptr.
     */
    void *acquire_pooled_gm_heap();
    void *acquire_pooled_gm_sm();
    void *acquire_pooled_runtime_arena();

    /**
     * Create a thread bound to this device. The thread calls
     * rtSetDevice(device_id) on entry.
     */
    std::thread create_thread(std::function<void()> fn);

    /**
     * Attach the current host thread to the target device.
     *
     * Required before host-side runtime initialization may allocate or
     * free device memory on the current thread. Idempotent for the same
     * id; errors if called with a different id after a prior attach.
     * No streams are created here.
     *
     * @param device_id  Device ID (0-15)
     * @return 0 on success, error code on failure.
     */
    int attach_current_thread(int device_id);

    /**
     * One-shot device initialization. Performs, in order:
     *   1. attach_current_thread on device_id_
     *   2. rtStreamCreate for AICPU + AICore streams (persistent, freed
     *      by the subclass `finalize()`).
     *   3. Bootstrap the dispatcher + register the inner AICPU SO via
     *      `ensure_binaries_loaded()`.
     *
     * Called from `simpler_init` after executor + dispatcher bytes have
     * been cached on the runner. Idempotent: subsequent calls
     * short-circuit on `binaries_loaded_`.
     *
     * @return 0 on success, error code on failure.
     */
    int ensure_device_initialized();

    /**
     * Print handshake results from device. Reads the per-core
     * `Handshake` array out of device memory and logs it at DEBUG. Must
     * be called after `run()` and before `finalize()`.
     */
    void print_handshake_results();

    /**
     * Take ownership of the AICPU + AICore executor binaries. Called
     * once by simpler_init at ChipWorker::init time; subsequent
     * `run()` invocations read from `aicpu_so_binary_` /
     * `aicore_kernel_binary_`.
     */
    void set_executors(std::vector<uint8_t> aicpu_so_binary, std::vector<uint8_t> aicore_kernel_binary) {
        aicpu_so_binary_ = std::move(aicpu_so_binary);
        aicore_kernel_binary_ = std::move(aicore_kernel_binary);
    }

    /**
     * Take ownership of the dispatcher SO bytes. Called by simpler_init
     * when the caller provided a dispatcher path; the eager
     * `ensure_device_initialized()` in simpler_init hands the buffer to
     * `LoadAicpuOp::BootstrapDispatcher` at init time. Leaving this
     * unset (empty buffer) makes `ensure_binaries_loaded()` fail with a
     * clear message — callers that drive `_ChipWorker.init` directly
     * without a dispatcher path get a deterministic error at
     * `simpler_init` time rather than a confusing dladdr-derived path.
     */
    void set_dispatcher_binary(std::vector<uint8_t> dispatcher_so_binary) {
        dispatcher_so_binary_ = std::move(dispatcher_so_binary);
    }

    /** The device id captured by simpler_init's `attach_current_thread` call. */
    int device_id() const { return device_id_; }

    /**
     * Device-side wall (ns) from the most recently completed run,
     * written by the platform AICPU entry. Returns 0 before any run
     * completes. Independent of any profiling / swimlane subsystem.
     */
    uint64_t last_device_wall_ns() const { return device_wall_ns_; }

    /**
     * Upload an entire ChipCallable buffer to device memory in one shot.
     * Walks child_offsets_ to compute total byte size, allocates device
     * GM once, fixes up each child's resolved_addr_ in an internal host
     * scratch (= device-side address of that child's binary code),
     * H2D's once, and returns the device-side address of the
     * ChipCallable header.
     *
     * Pool-managed: identical buffer bytes (FNV-1a 64-bit content hash)
     * hit the dedup cache and return the cached chip_dev without
     * reallocating. All chip buffers are bulk-freed by the subclass's
     * `finalize()` — there is no explicit free API, mirroring the
     * per-fid binary pool semantics.
     *
     * Callers compute child addresses as
     *     chip_dev + offsetof(ChipCallable, storage_) + child_offset(i)
     * and write them to Runtime::func_id_to_addr_[fid] via
     * Runtime::set_function_bin_addr().
     *
     * @param callable  Host-side ChipCallable pointer.
     * @return Device GM address of the ChipCallable header, or 0 on
     *         failure (also returns 0 when callable->child_count() == 0).
     */
    uint64_t upload_chip_callable_buffer(const ChipCallable *callable);

    /**
     * Stage a per-callable_id orchestration SO into device memory and
     * remember the supporting metadata (entry/config symbol names,
     * kernel func_id ↔ dev_addr table). Identical SO bytes across two
     * callable_ids share one device buffer (refcounted by hash) so the
     * worst case for an N-cid pool is N distinct device buffers, not
     * N copies of the same SO.
     *
     * @param callable_id   Caller-stable id, must be in [0, MAX_REGISTERED_CALLABLE_IDS).
     * @param orch_so_data  Host pointer to orchestration SO bytes (owned by caller).
     * @param orch_so_size  Size of orchestration SO in bytes.
     * @param func_name     Entry symbol name (copied).
     * @param config_name   Config symbol name (copied).
     * @param kernel_addrs  func_id ↔ dev_addr pairs already uploaded by
     *                      the caller. Stored verbatim so subsequent
     *                      runs can replay them onto a fresh Runtime
     *                      without re-uploading.
     * @return 0 on success, negative on failure.
     */
    int register_callable(
        int32_t callable_id, const void *orch_so_data, size_t orch_so_size, const char *func_name,
        const char *config_name, std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );

    /**
     * Host-orchestration variant of register_callable: stores a dlopen
     * handle + entry-symbol pointer that runtime_maker resolved on the
     * host (host_build_graph variant). Mutually exclusive with the
     * trb-shaped overload — exactly one is invoked for a given
     * callable_id, picked by the C ABI based on which staging fields
     * the runtime carries after prepare_callable_impl. dlopen handle
     * is owned by `DeviceRunnerBase` from this call onward and
     * dlclose'd by `unregister_callable`. Increments `host_dlopen_total_`.
     */
    int register_callable_host_orch(
        int32_t callable_id, void *host_dlopen_handle, void *host_orch_func_ptr,
        std::vector<std::pair<int, uint64_t>> kernel_addrs, std::vector<ArgDirection> signature
    );

    /**
     * Drop the registered state for `callable_id`. trb path: decrement
     * the orch SO buffer's hash-keyed refcount and free when it hits
     * zero. hbg path: dlclose the host dlopen handle. Kernel binaries
     * are shared across callables and only released by the subclass's
     * `finalize()`.
     *
     * @return 0 on success or if the id was not registered.
     */
    int unregister_callable(int32_t callable_id);

    /**
     * True iff `callable_id` has registered state staged via
     * `register_callable*`. Lets the c_api layer reject `run_prepared`
     * calls without a matching `prepare_callable`.
     */
    bool has_callable(int32_t callable_id) const;

    /**
     * Replay a previously-registered callable's state onto a fresh
     * Runtime for a per-run binding. Writes back kernel addrs, orch
     * entry-symbol names, and active_callable_id; returns the hbg
     * `host_orch_func_ptr` (or nullptr on trb / on error) inside a
     * `BindCallableResult` so the caller can destructure with
     * structured bindings.
     */
    BindCallableResult bind_callable_to_runtime(Runtime &runtime, int32_t callable_id);

    /**
     * Number of distinct callable_ids the AICPU has been asked to
     * dlopen for. Monotonically increases on every first-sighting
     * bind; `unregister_callable` does NOT decrement it. So a
     * `prepare → run → unregister → re-prepare → run` sequence reports
     * 2 (each AICPU dlopen counted once), even though only one cid is
     * currently registered.
     */
    size_t aicpu_dlopen_count() const { return aicpu_dlopen_total_; }

    /**
     * Number of host-side dlopen() invocations triggered by
     * `register_callable_host_orch`. Mirrors `aicpu_dlopen_count` but
     * counts the host_build_graph variant's host-side dlopens; it
     * never decrements.
     */
    size_t host_dlopen_count() const { return host_dlopen_total_; }

    /**
     * Launch an AICPU kernel. Internal helper used by the subclass's
     * `run()`; thin wrapper that dispatches through `load_aicpu_op_`'s
     * cached `rtFuncHandle` (resolved by `LoadAicpuOp::Init` at first
     * bootstrap).
     *
     * @param stream       AICPU stream
     * @param k_args       Kernel arguments
     * @param kernel_name  Name of the kernel to launch (e.g.
     *                     `host::KernelNames::InitName` / `RunName`)
     * @param aicpu_num    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int launch_aicpu_kernel(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num);

    /**
     * Launch an AICore kernel. Lazy-registers the kernel binary
     * (`aicore_kernel_binary_`) on first call via `rtRegisterAllKernel`
     * and caches the resulting `aicore_bin_handle_`; subsequent calls
     * reuse the cached handle. CANN has no public
     * `rtUnregisterAllKernel`, so re-registering on every run would pin
     * another device-side copy of the ELF and quickly exhaust HBM —
     * manifested in CI as 207001 at `rtKernelLaunchWithHandleV2` with a
     * 507899 cascade at `rtStreamCreate`.
     *
     * `k_args` reaches the AICore kernel through `rtArgsEx_t`; whether
     * it is a host-resident or device-resident `KernelArgs` pointer is
     * decided by the subclass's `run()` (a2a3 passes host; a5 passes
     * device).
     */
    int launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args);

    /**
     * Enablement setters for the four shared diagnostics sub-features.
     * Called by the c_api entry point before `run()`; downstream `run()`
     * paths read the corresponding `enable_*_` members directly.
     *
     * `set_dep_gen_enabled` is a2a3-only and lives on the subclass.
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
    void set_scope_stats_enabled(bool enable) { enable_scope_stats_ = enable; }

    /**
     * Directory under which all diagnostic artifacts
     * (l2_perf_records.json / tensor_dump/ / pmu.csv) land. Required
     * (non-empty) when any diagnostic is enabled; `CallConfig::validate()`
     * enforces this contract upstream.
     */
    void set_output_prefix(const char *prefix) { output_prefix_ = (prefix != nullptr) ? prefix : ""; }
    const std::string &output_prefix() const { return output_prefix_; }

protected:
    // Ctor / dtor are protected: this class is for inheritance only —
    // direct instantiation (`new DeviceRunnerBase()`) and polymorphic delete
    // (`delete (DeviceRunnerBase *)p`) are both compile errors.
    DeviceRunnerBase();
    ~DeviceRunnerBase() = default;

    /**
     * `DeviceArena` callback trampolines bridging from C-style
     * `void *(void *ctx, size_t)` / `void (void *ctx, void *)` to the
     * `MemoryAllocator` member function calls. The `ctx` opaque pointer
     * passed at arena construction time is `&mem_alloc_`.
     */
    static void *arena_alloc_trampoline(void *ctx, std::size_t size) {
        return static_cast<MemoryAllocator *>(ctx)->alloc(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) { static_cast<MemoryAllocator *>(ctx)->free(p); }

    /**
     * Configure STARS op execution timeout (once per DeviceRunner lifetime).
     *
     * Called on first device attach to set the hardware-level AICore op
     * execution timeout via `aclrtSetOpExecuteTimeOutV2`. The actual
     * timeout may differ from the requested value due to hardware timer
     * granularity.
     */
    void configure_aicore_op_timeout();

    /**
     * Load AICPU SO and initialize device args. Called from
     * `ensure_device_initialized()` after the persistent streams are
     * created. Reads `aicpu_so_binary_` / `dispatcher_so_binary_` off
     * the runner; releases both host buffers on success.
     *
     * @return 0 on success, error code on failure.
     */
    int ensure_binaries_loaded();

    /**
     * Query the maximum block_dim the stream can host.
     *
     * Uses `aclrtGetStreamResLimit(CUBE_CORE / VECTOR_CORE)` and
     * returns `min(cube / AIC_PER_BLOCKDIM, vector / AIV_PER_BLOCKDIM)`,
     * capped by `PLATFORM_MAX_BLOCKDIM`. Falls back to the static cap
     * when the query is unavailable or reports no cores.
     *
     * If non-null, `out_cube` / `out_vector` receive the raw ACL limits
     * when the query succeeded, or 0 when it failed. Callers use this
     * to distinguish the ACL-unavailable fallback path from the
     * success path in error logs.
     */
    int query_max_block_dim(rtStream_t stream, uint32_t *out_cube = nullptr, uint32_t *out_vector = nullptr);

    /**
     * Validate block_dim against the stream's CUBE/VECTOR core limits
     * (via `query_max_block_dim`). Returns 0 if block_dim fits, -1
     * otherwise (or if block_dim < 1).
     */
    int validate_block_dim(rtStream_t stream, int block_dim);

    // ---- run() sub-sequence helpers --------------------------------------
    //
    // Each arch's `run()` keeps the heavily-divergent middle (register
    // address setup, profiling flag building, init_*, collector start /
    // teardown, dep_gen, ffts setup, kernel launches). These helpers
    // cover the byte-identical sub-sequences at the head and tail.

    /**
     * Validate the caller's `launch_aicpu_num` against
     * `PLATFORM_MAX_AICPU_THREADS`. Returns 0 on success, -1 on
     * out-of-range with a logged error.
     */
    int validate_launch_aicpu_num(int launch_aicpu_num);

    /**
     * Lazy-allocate the 8-byte device-resident buffer that AICPU writes
     * the run wall (ns) into and that `read_device_wall_ns()` pulls
     * back after stream sync. Idempotent: a no-op once
     * `device_wall_dev_ptr_` is non-null. Routes the alloc through
     * `mem_alloc_`; the pointer is freed by `finalize_common()`.
     * Failure to alloc is non-fatal (`device_wall_data_base` stays 0,
     * subsequent `last_device_wall_ns()` reads 0).
     */
    void ensure_device_wall_buffer();

    /**
     * Resolve the caller's `requested_block_dim` into a concrete
     * block_dim:
     *  - `requested_block_dim == 0`: auto-resolve from
     *    `query_max_block_dim(stream_aicore_)`.
     *  - otherwise: pass through `validate_block_dim`.
     *
     * Returns the resolved block_dim on success, -1 on failure.
     * Updates `block_dim_` on success.
     */
    int resolve_block_dim(int requested_block_dim);

    /**
     * Per-run Runtime setup: derives `num_aicore = block_dim *
     * cores_per_blockdim_`, range-checks against `RUNTIME_MAX_WORKER`,
     * publishes `runtime.worker_count`, `worker_count_`,
     * `runtime.aicpu_thread_num`, zero-initializes the handshake
     * worker array with AIC/AIV core typing (first `block_dim` cores
     * are AIC, remaining are AIV), and rewrites each task's
     * `function_bin_addr` from `runtime.get_function_bin_addr(func_id)
     * + CoreCallable::binary_data_offset()`.
     *
     * Returns 0 on success, -1 on `block_dim`-too-large error.
     */
    int prepare_runtime_for_launch(Runtime &runtime, int block_dim, int launch_aicpu_num);

    /**
     * Wait for both per-Worker streams (AICPU first, then AICore) with
     * `PLATFORM_STREAM_SYNC_TIMEOUT_MS`. Distinguishes the timeout
     * sentinel `ACL_ERROR_RT_STREAM_SYNC_TIMEOUT` with a stream-id and
     * (device, block_dim) context in the log. Returns the first
     * non-zero rc encountered.
     */
    int sync_run_streams();

    /**
     * Pull the device wall (ns) back from `device_wall_dev_ptr_` and
     * cache it on `device_wall_ns_`. D2H copy failure is a soft warn —
     * `device_wall_ns_` stays at 0 so `last_device_wall_ns()` returns 0
     * to callers. No-op if `device_wall_dev_ptr_` is null (lazy alloc
     * may have failed silently).
     */
    void read_device_wall_ns();

    /**
     * Shared body of `finalize()`. Each arch subclass's `finalize()`
     * handles: (a) the early-return + thread attach prologue, (b) any
     * arch-specific collector teardown (e.g. a2a3's `dep_gen_collector_`),
     * and (c) the arch-specific device reset (a2a3's ACL/rt branch vs
     * a5's `rtDeviceReset`). Everything else lives here:
     *
     *   - rtStreamDestroy for both persistent streams
     *   - kernel_args_.finalize_device_args
     *   - aicore_bin_handle_ + binaries_loaded_ reset
     *   - chip_callable_buffers_ free + clear
     *   - orch_so_dedup_ free + clear
     *   - callables_ dlclose-on-hbg + clear + aicpu counter reset
     *   - 3 arenas release + cached size reset
     *   - device_wall_dev_ptr_ free (before mem_alloc_.finalize)
     *   - mem_alloc_.finalize
     *   - block_dim_, worker_count_, aicore_kernel_binary_ reset
     *
     * Device-wall free order is normalized to "before mem_alloc_.finalize"
     * (matching the prior a5 ordering). The prior a2a3 ordering freed it
     * AFTER `mem_alloc_.finalize` + `rtDeviceReset`, which routed through
     * an already-finalized allocator on a torn-down device context — a
     * latent UAF / no-op. This refactor fixes that.
     *
     * @return 0 on success, first nonzero rc encountered otherwise.
     */
    int finalize_common();

    /**
     * Stamp `runtime.{dev_orch_so_addr_, dev_orch_so_size_}` from the
     * registered CallableState for `runtime.get_active_callable_id()`.
     * The orch SO bytes were already H2D'd at `register_callable` time
     * and are shared via `orch_so_dedup_` across cids; this method only
     * refreshes the device-SO metadata onto the per-run Runtime and
     * bumps the AICPU first-sighting counter when the cid is new since
     * registration.
     *
     * @param runtime  Runtime whose device-SO metadata will be rewritten.
     * @return 0 on success, non-zero on failure.
     */
    int prepare_orch_so(Runtime &runtime);

    // ---- Group D state shared by both a2a3 and a5 -------------------------
    //
    // Chip-callable buffer pool. Keyed by FNV-1a 64-bit content hash of
    // the ChipCallable bytes. Each entry owns one device GM allocation
    // holding the entire ChipCallable buffer (header + storage_, with
    // each child's resolved_addr_ fixed up to its post-H2D device
    // address). Pool-managed: identical buffer bytes share one entry
    // across cids; the map is bulk-freed by the subclass's `finalize()`.
    struct ChipCallableBuffer {
        uint64_t chip_dev{0};  // device GM address of the ChipCallable header
        size_t total_size{0};  // byte size of the device allocation
    };
    std::unordered_map<uint64_t, ChipCallableBuffer> chip_callable_buffers_;

    // Per-callable_id registered state.
    //
    // `callables_` maps the caller-stable callable_id to the orch SO
    // slice + symbol names needed to launch it. `orch_so_dedup_` shares
    // device buffers across callable_ids whose orch SO bytes have the
    // same ELF Build-ID hash (refcounted; freed when the count hits
    // zero). `aicpu_seen_callable_ids_` tracks which ids have already
    // been delivered to the AICPU at least once so `prepare_orch_so`
    // can set `register_new_callable_id_` correctly on first sighting.
    struct CallableState {
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
    std::unordered_map<int32_t, CallableState> callables_;
    std::unordered_map<uint64_t, OrchSoBuffer> orch_so_dedup_;
    std::unordered_set<int32_t> aicpu_seen_callable_ids_;
    // Monotonic count of AICPU dlopens triggered (incremented on each
    // first-sighting bind; never decremented). Diverges from
    // aicpu_seen_callable_ids_.size() once any cid is unregistered and
    // re-registered. Exposed via `aicpu_dlopen_count()` for tests.
    size_t aicpu_dlopen_total_{0};
    // Monotonic count of host-side dlopens triggered (incremented on
    // every `register_callable_host_orch` call; never decremented).
    // Same re-register semantics as `aicpu_dlopen_total_`, but for hbg
    // variants.
    size_t host_dlopen_total_{0};

    // ---- State shared by both a2a3 and a5 ---------------------------------
    //
    // `device_id_` is set once in `attach_current_thread()` (called from
    // simpler_init during ChipWorker::init) and read on every subsequent
    // op. All ChipWorker callers run on the same thread that called
    // init, so plain int + the init→user happens-before edge is
    // sufficient.
    int device_id_{-1};
    int block_dim_{0};
    int cores_per_blockdim_{PLATFORM_CORES_PER_BLOCKDIM};
    int worker_count_{0};  // Stored for print_handshake_results

    // Executor binaries — populated once via `set_executors()` during
    // simpler_init. `aicore_kernel_binary_` is consumed once by
    // `launch_aicore_kernel()` (`rtRegisterAllKernel` returns
    // `aicore_bin_handle_`, cached and reused on every subsequent
    // launch). Caching is required: CANN has no public
    // `rtUnregisterAllKernel`, so re-registering on every run would pin
    // another device-side copy of the ELF and quickly exhaust HBM
    // (manifested in CI as 207001 at `rtKernelLaunchWithHandleV2` with
    // a 507899 cascade at `rtStreamCreate`). `aicpu_so_binary_` is
    // released by `ensure_binaries_loaded()` after bootstrap;
    // bootstrap is the only consumer and per-task launches go through
    // the cached `rtFuncHandle` on `LoadAicpuOp`, not the host bytes.
    std::vector<uint8_t> aicpu_so_binary_;
    std::vector<uint8_t> aicore_kernel_binary_;
    // AICore kernel handle from `rtRegisterAllKernel` — lazily
    // populated by the subclass's `launch_aicore_kernel()` and reused
    // across all runs. `nullptr` means not yet registered. Reset to
    // `nullptr` in `finalize()`; CANN releases the device-side state
    // implicitly when the device context tears down.
    void *aicore_bin_handle_{nullptr};
    // Dispatcher SO bytes — populated once via `set_dispatcher_binary()`
    // during simpler_init. Consumed exclusively by
    // `BootstrapDispatcher` on the first run and released by
    // `ensure_binaries_loaded()` right after. Empty buffer is permitted
    // at init time (callers that drive `ChipWorker.init` without a
    // dispatcher path); `ensure_binaries_loaded()` then fails fast
    // with a clear message if/when bootstrap is actually attempted.
    std::vector<uint8_t> dispatcher_so_binary_;

    // AICPU op loader — handles dispatcher bootstrap and per-task launches.
    host::LoadAicpuOp load_aicpu_op_;

    MemoryAllocator mem_alloc_;
    DeviceArena gm_heap_arena_;
    DeviceArena gm_sm_arena_;
    DeviceArena runtime_arena_pool_;

    // Cached arena sizes for `setup_static_arena`'s "fits" check — avoids
    // re-allocating the same buffer when a later worker init asks for an
    // equal-or-smaller layout on an already-committed arena. Reset by
    // the subclass's `finalize()` alongside the other identity state.
    size_t cached_gm_heap_size_{0};
    size_t cached_gm_sm_size_{0};
    size_t cached_runtime_arena_size_{0};

    // Persistent AICPU / AICore streams created in
    // `ensure_device_initialized()` and torn down in the subclass's
    // `finalize()`. `nullptr` before init.
    rtStream_t stream_aicpu_{nullptr};
    rtStream_t stream_aicore_{nullptr};
    KernelArgsHelper kernel_args_;

    // Platform-level device wall buffer: 8-byte device-resident slot
    // whose address rides on `KernelArgs.device_wall_data_base`. AICPU
    // writes the run wall (ns) through that pointer; subclass `run()`
    // pulls it back via `copy_from_device` after stream sync and
    // caches it for `last_device_wall_ns()`. Allocated once at
    // simpler_init, freed in the subclass `finalize()`.
    void *device_wall_dev_ptr_{nullptr};
    uint64_t device_wall_ns_{0};
    DeviceArgs device_args_;

    // True after AICPU SO loaded; reset by the subclass's `finalize()`.
    bool binaries_loaded_{false};

    // Shared diagnostics collectors. Each subclass initializes its own
    // (a2a3 wraps `halHostRegister`/`Unregister` callbacks, a5 uses
    // direct `rtMalloc`/`rtFree`), but the storage and lifetime live
    // on the base. `DepGenCollector` is a2a3-only and stays on the
    // a2a3 subclass.
    L2PerfCollector l2_perf_collector_;
    TensorDumpCollector dump_collector_;
    PmuCollector pmu_collector_;
    ScopeStatsCollector scope_stats_collector_;

    // Enablement for the four shared diagnostics sub-features.
    // Written by the c_api entry point via `set_*_enabled()` before
    // `run()`, read inside `run()` and its helpers.
    bool enable_l2_swimlane_{false};
    bool enable_dump_tensor_{false};
    bool enable_pmu_{false};
    bool enable_scope_stats_{false};
    L2PerfLevel l2_perf_level_{L2PerfLevel::DISABLED};             // resolved from set_l2_swimlane_enabled()
    PmuEventType pmu_event_type_{PmuEventType::PIPE_UTILIZATION};  // resolved from set_pmu_enabled()
    std::string output_prefix_{};                                  // diagnostic artifact root directory
};

#endif  // SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H
