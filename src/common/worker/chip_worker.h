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

#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../platform_comm/comm.h"
#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "host/chip_swimlane_runs.h"
#include "native_run_storage_tag.h"
#include "run_resource_arena.h"
#include "runtime_c_api.h"
#include "types.h"

/** Opaque identity for one prepared native run owned by a ChipWorker. */
struct ChipWorkerNativeRun {
    uint32_t slot_id{0};
    // Generation of the externally minted pipeline lease. One lease may
    // dispatch several runs, so this is not sufficient run identity alone.
    uint64_t generation{0};
    // Process-unique identity for exactly one prepare attempt.
    uint64_t run_epoch{0};
    // Optional L3 mailbox identity. Direct L2 callers leave these as zero.
    uint64_t run_id{0};
    uint64_t dispatch_id{0};
};

class ChipRun;
class ChipRunLane;
class RuntimeModule;
struct ChipRunLaneState;

// A current-context preflight refusal occurs before lane progress or native
// phase mutation; it supplies no execution or completion verdict.
class ExecutionContextRefused : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// This invocation enqueued no operator kernel, and its native input bindings and
// preparation resources have all been released. It supplies no verdict about
// any other run or about whether the execution environment accepts a retry.
class RunPreparationRefused : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class UnsupportedRuntimeOperation : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ChipWorker {
public:
    // Host-only identity of one native environment binding, including retained
    // rollback debt. It is independent of Buffer and pipeline lease generations.
    struct ExecutionEnvironment {
        uint64_t process_id;
        uint64_t incarnation;
        uint64_t context_generation;
        int device_id;
        bool borrowed;
    };

    ChipWorker() = default;
    ~ChipWorker();

    ChipWorker(const ChipWorker &) = delete;
    ChipWorker &operator=(const ChipWorker &) = delete;

    /// Bind the runtime library, cache platform binaries, and attach the
    /// calling thread to owned `device_id`, or validate its borrowed context.
    /// A successful init can only be called once per lifetime —
    /// the runtime and device cannot be changed after init.
    ///
    /// Every host module contains its own logger implementation and binds it to
    /// the process-owned state before use. On sim, libcpu_sim_context.so is
    /// retained in a process-wide registry because host_runtime.so resolves PTO
    /// simulator hooks from its RTLD_GLOBAL symbol scope.
    /// On an owned context, `prewarm_config`, when non-null, builds + caches the prebuilt
    /// runtime-arena for its ring sizing right after the device comes up (the
    /// sizing is fork-constant, delivered by COW into init). A no-op for
    /// runtimes without a prebuilt arena. Borrowed init reads it for admission
    /// without prewarming those arenas.
    /// `enable_sdma` opts this Worker into the async-DMA (SDMA) workspace, so
    /// kernels can use get_dma_workspace. Off by default; a Worker that does not
    /// opt in creates no SDMA streams and its kernels read a zero address for
    /// that engine. It names one engine rather than a set because SDMA is the
    /// only one a caller can decline — every other supported engine is
    /// provisioned unconditionally, and SDMA is conditional only because its
    /// workspace is inseparable from 48 CP-process STARS streams whose
    /// post-fault release CANN does not bound. Provisioning rides simpler_init,
    /// so a platform/runtime without SDMA support fails init (this throws) and
    /// no Worker reaches a run with a zero address it expected to be live.
    /// `sdma_warmup_path`, when non-empty, is the vector-only ELF that walks the
    /// SDMA control path once per channel once that workspace is live, moving
    /// the cold-start cost off the first TPREFETCH_ASYNC. Read only when
    /// `enable_sdma` is set; an empty path (or an arch that builds no such ELF)
    /// only costs that first-call latency.
    ///
    /// borrowed_context_generation selects the caller-owned, already-current RTS
    /// context. It must be nonzero; absent means the existing owned-device init.
    /// Both onboard routes manage internal workspace and accept an optional byte budget.
    /// Borrowed init supports no SDMA or simulator policy.
    /// The caller keeps its context current through successful finalize. Failed
    /// rollback requires explicit finalize if it could not close; that worker
    /// cannot be reinitialized.

    void init(
        const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
        const std::string &dispatcher_path, int device_id, const CallConfig *prewarm_config = nullptr,
        bool enable_sdma = false, const std::string &sim_context_path = "", const std::string &sdma_warmup_path = "",
        bool collect_across_runs = false, uint64_t workspace_budget_bytes = 0, bool manage_workspace = false,
        uint32_t requested_pipeline_depth = 0, std::optional<uint64_t> borrowed_context_generation = std::nullopt
    );

    /**
     * Publish every diagnostic run this chip has closed, then report.
     *
     * Only meaningful when the worker was initialized with `collect_across_runs`; a
     * chip without one has nothing deferred and returns success. Throws on a
     * run whose artifact is missing.
     */
    void flush_diagnostics(int timeout_ms);

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // Launch a cid previously staged via register_callable. `args` is the runtime.so-ABI POD, which
    // every caller already holds: the wire blob is materialized into one before it gets here.
    // Per-stage timing (host wall, on-NPU device wall + AICPU phase breakdown) is emitted by the
    // platform as `[STRACE]` log markers — see src/common/log/.../strace.h — not returned, so the
    // L3 dispatcher and L2 child are observed uniformly.
    void run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config);
    void
    run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config,
        volatile int32_t *accepted_state, int32_t accepted_value);
    void run_with_lease(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0, uint64_t run_id = 0,
        uint64_t dispatch_id = 0, NativeRunProvenance provenance = {}
    );

    /**
     * Progressable native-run lifecycle.
     *
     * prepare_native_run performs per-run Runtime construction and binding but
     * does not launch device work. launch_native_run returns only after the
     * backend crosses its launch fence (or terminates with an error), while
     * poll/wait observe completion and finalize owns validation, copy-back,
     * diagnostics, and Runtime destruction. The blocking run() overloads are
     * the compatibility composition of these phases. A successful prepare
     * transfers cleanup ownership to the caller: launch failure does not consume
     * the token, and the caller must still finalize it. The blocking composition
     * performs that cleanup internally on every exit.
     *
     * Onboard HBG may prepare one distinct-slot successor while another run
     * owns the execution claim. Diagnostics and backends without the explicit
     * capability remain depth-one. Lease generation gates admission; after a
     * successful prepare, the slot plus process-unique run epoch prevents a
     * delayed phase call from touching reused storage, including another run
     * under the same pipeline lease or on another ChipWorker.
     */
    ChipWorkerNativeRun prepare_native_run(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id = 0, uint64_t dispatch_id = 0, volatile int32_t *accepted_state = nullptr,
        int32_t accepted_value = 0, NativeRunProvenance provenance = {}
    );
    /**
     * What a launch attempt did, for the one outcome that is not a failure.
     *
     * `DiagnosticCreditFull` means every output slot or terminal bank some
     * enabled owner needs is still held by a run ahead: nothing was submitted,
     * nothing was admitted, and the run is unchanged and launchable again once
     * credit moves. Every other non-zero result still throws, because it is
     * this run's failure.
     */
    enum class NativeLaunch : int { Launched, DiagnosticCreditFull };
    NativeLaunch launch_native_run(const ChipWorkerNativeRun &run, void *caller_stream = nullptr);
    void validate_caller_stream(void *caller_stream);
    void validate_execution_context() const;
    /**
     * Launch `run` ordered behind `predecessor`, which must be a run this
     * ChipWorker launched and has not yet reaped.
     *
     * `run`'s native submission reaches the device while `predecessor` is still
     * executing; the device still executes one operator at a time, ordered by a
     * queued wait for the predecessor's whole-operator completion boundary.
     *
     * Returns false, having launched nothing and consumed nothing, when the
     * backend cannot order the two — an unsupported runtime, or a moment at
     * which it cannot. The token is still prepared and the caller is expected
     * to `launch_native_run` it later. Every other failure throws, exactly as
     * `launch_native_run` does.
     */
    bool launch_native_run_joined(
        const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor, void *caller_stream = nullptr
    );
    bool poll_native_run(const ChipWorkerNativeRun &run);
    void wait_native_run(const ChipWorkerNativeRun &run);
    void finalize_native_run(const ChipWorkerNativeRun &run);

    /**
     * Run #2267's late-read retention fixture over a launched predecessor and a
     * prepared successor on another slot. See `run_retention_probe.h`.
     *
     * Onboard only: it measures a stream-level property, and no simulated
     * backend has the streams. Throws when the loaded runtime does not export
     * it, which for an onboard module means the build is stale.
     *
     * Leaves both runs finalizable, and the caller still owes
     * `finalize_native_run` for each — the predecessor's drain, copy-back and
     * DFX teardown are the ordinary path, reached from the phase it is left in.
     */
    RunRetentionProbeReport probe_run_retention(
        const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &successor, const RunRetentionProbeConfig &config
    );

    /**
     * What the *first* captured `finalize()` observed about its device
     * teardown, or false when none was captured.
     *
     * First rather than latest: a teardown that failed keeps the context and
     * the module loaded so an explicit retry can finish the job, and the
     * attempt that failed is the one describing what happened to the device.
     * A retry therefore does not overwrite it.
     *
     * Empty on a module that exports no report accessor, on a capture that
     * itself failed, and on every `finalize()` reached with no context.
     */
    bool teardown_report(SimplerTeardownReport *out) const;

    /** Whether one workspace report could be produced, and why not. */
    enum class WorkspaceReportStatus : uint32_t {
        /** This context's workspace regions have no owner: a simulated
            backend, or any route that asked for neither management nor a
            budget — a chip child included, unless its parent opted in. */
        Disabled = 0,
        /** `out` carries this context's accounting. */
        Available = 1,
        /** This context is managed but its accounting could not be read. */
        Unavailable = 2,
    };

    /**
     * Read this context's workspace accounting.
     *
     * Three answers rather than two, because "not managed" and "managed but
     * unreadable" have opposite safety consequences: a caller that protects
     * teardown on this must refuse on `Unavailable` and must not mistake it
     * for `Disabled`. Whether this context is managed is this object's own
     * recorded fact, independent of any query succeeding — and it is a
     * different question from whether a finite budget is enforced, which the
     * report answers through `budget_enforced`.
     */
    WorkspaceReportStatus workspace_report(SimplerWorkspaceReport *out) const noexcept;

    ChipRun submit_chip_run(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0,
        bool activated = true, NativeRunProvenance provenance = {}
    );
    // Direct admission uses the lane's capacity and generations. With a caller
    // stream, success means the whole run and its exit wait have been enqueued;
    // a non-joinable predecessor is drained before returning. The caller owns
    // the stream through retirement. Host copy-back still requires wait/finalize.
    // run() is the blocking composition without caller stream ordering.
    ChipRun submit_chip_run(
        int32_t callable_id, const ChipStorageTaskArgs &args, const CallConfig &config,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0, void *caller_stream = nullptr
    );
    void close_chip_run_lane();
    /**
     * Stop the run lane admitting anything further, finishing every run that
     * has not launched without launching it. @see ChipRunLane::stop_admission.
     */
    void stop_chip_run_lane_admission() noexcept;

    // Per-callable_id preparation. Requires init() first and a callable_id
    // in [0, MAX_REGISTERED_CALLABLE_IDS) (cap 64). An ID that is already
    // registered is rejected; unregister it before reuse.
    void register_callable(int32_t callable_id, const void *callable);
    /// Register with a runtime-assigned ID in either initialization mode.
    int32_t register_callable(const void *callable, size_t callable_size);
    void unregister_callable(int32_t callable_id);

    /// Number of distinct callable_ids the AICPU has been asked to dlopen for
    /// on the bound device. Returns 0 when not initialized or the runtime
    /// variant has no per-cid registration support. Used by tests to assert
    /// that register_callable + repeated run do not trigger redundant
    /// AICPU dlopens.
    size_t aicpu_dlopen_count() const;

    /// Number of host-side dlopens (host_build_graph variant). Mirrors
    /// `aicpu_dlopen_count` for the trb path; returns 0 on device-orch variants.
    size_t host_dlopen_count() const;

    /// Number of AICore run streams the bound runner has created. One AICPU +
    /// AICore pair serves every run for the runner's lifetime. The AICPU stream
    /// persists; the AICore stream is recreated when a new code upload makes it
    /// stale, and destroyed when an unproven completion retires it. Platforms
    /// using the persistent bootstrap pair report 0.
    size_t run_stream_set_create_count() const;

    uint64_t malloc(size_t size);
    void free(uint64_t ptr);
    void copy_to(uint64_t dst, uint64_t src, size_t size);
    void copy_from(uint64_t dst, uint64_t src, size_t size);

    /// One span of a caller device allocation a run names. The lane's
    /// vocabulary for the borrow it takes on a run's behalf; the wire form it
    /// becomes is the platform's (`CallerBufferSpan`).
    struct CallerDeviceSpan {
        uint64_t addr{0};
        uint64_t bytes{0};
    };

    /// Take `borrow_id`'s reference on the caller allocations covering `spans`,
    /// retaining every known allocation even if another span is external.
    /// True proves every span's owner; false refuses joined admission but can
    /// still hold references until `release_caller_device_borrow`.
    ///
    /// The three `device_*_caller_buffer(s)_ctx` entries are mandatory C ABI,
    /// like the rest of the `device_*_ctx` family: `init` resolves them with
    /// `RuntimeModule::required`, which throws when one is missing, so a bound runtime
    /// always has them. (The two `HostApiOps` entries this capability adds are
    /// the optional half — those are null-guarded, and a platform publishing
    /// neither behaves as it did.) False therefore means what the paragraph
    /// above says — at least one span named no caller allocation of this context — or that
    /// this worker holds no device context at all, which is only so before
    /// `init` and after `finalize`.
    bool borrow_caller_device_spans(uint64_t borrow_id, const CallerDeviceSpan *spans, size_t count);

    /// Drop `borrow_id`'s reference. `keep` retains it for the process's
    /// remaining life, for a run whose last device consumer is unproven.
    void release_caller_device_borrow(uint64_t borrow_id, bool keep) noexcept;

    /// Distributed communication primitives (optional — only available when
    /// the bound runtime exports comm_*).  Wraps the backend-neutral C API
    /// defined in src/<arch>/platform/include/host/comm.h.
    ///
    /// Unlike the raw C API (which takes a caller-owned aclrtStream),
    /// ChipWorker's comm_init owns ACL + stream lifetime internally:
    ///   - On onboard, comm_init drives ensure_acl_ready_ctx + creates an
    ///     aclrtStream via the DeviceRunner, stashes the stream, and pairs
    ///     it with comm_destroy which destroys it.  This keeps ACL out of
    ///     the Python layer (matching the doc's L2-boundary contract:
    ///     device-side lifecycle stays in C++, not leaking up as
    ///     ensure_acl_ready / aclrtCreateStream surface area).
    ///   - On sim, ACL / stream are no-ops; the stashed stream is null.
    ///
    /// Multi-domain bootstrap allocates a hidden base communicator plus one
    /// symmetric pool, then derives per-domain views with comm_derive_context.
    uint64_t comm_init(int rank, int nranks, const std::string &rootinfo_path);
    uint64_t comm_alloc_windows(uint64_t comm_handle, size_t win_size);
    uint64_t comm_get_local_window_base(uint64_t comm_handle);
    size_t comm_get_window_size(uint64_t comm_handle);
    uint64_t comm_derive_context(
        uint64_t comm_handle, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank, size_t window_offset,
        size_t window_size
    );
    /// Collectively allocate a fresh per-rank symmetric pool for a subset of
    /// ranks.  Multiple concurrent allocations are disambiguated by
    /// `allocation_id`.  Returns (device_ctx, local_window_base).  Only
    /// participating ranks call this; non-members of the subset must not.
    std::pair<uint64_t, uint64_t> comm_alloc_domain_windows(
        uint64_t comm_handle, uint64_t allocation_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
        size_t window_size
    );
    /// Pair to `comm_alloc_domain_windows`: collectively free the per-rank
    /// pool and the device CommContext, then drop the allocation record.
    /// `rank_count` + `domain_rank` size the subset barrier; the rank list
    /// itself is not needed (the alloc-time identity is already cached
    /// inside the backend's per-allocation record).
    void
    comm_release_domain_windows(uint64_t comm_handle, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank);
    std::tuple<std::vector<uint8_t>, uint64_t, size_t> comm_global_domain_prepare(
        uint64_t domain_id, uint32_t domain_rank, uint32_t rank_count, size_t window_size, uint32_t profile
    );
    uint64_t comm_global_domain_import(uint64_t domain_id, const std::vector<uint8_t> &descriptors);
    void comm_global_domain_release(uint64_t domain_id);
    void comm_barrier(uint64_t comm_handle);
    void comm_destroy(uint64_t comm_handle);
    void comm_destroy_all();

    int device_id() const { return device_id_; }
    bool initialized() const { return initialized_; }
    std::optional<ExecutionEnvironment> execution_environment() const { return execution_environment_; }
    unsigned pipeline_depth() const { return pipeline_contract_.pipeline_depth; }
    size_t runtime_slot_count() const { return runtime_bufs_.size(); }
    bool supports_concurrent_native_prepare() const;

    /**
     * Let this worker's device context size its own run concurrency.
     *
     * @return false when the loaded module or the context cannot, in which case
     *         the worker keeps admitting exactly what it was granted
     */
    bool enable_automatic_admission();

    /** Whether this worker's context sizes its own concurrency. */
    bool automatic_admission() const { return automatic_admission_; }

    /**
     * Run-resource sets the context reports it can hold right now, or the
     * granted count when it does not size itself.
     */
    uint32_t admissible_run_resource_sets() const;

    /**
     * Give up every resource a waiting plan on this slot still holds.
     *
     * Called by the lane for a run it is abandoning that asked to wait, because
     * such a run holds things an unprepared run does not. Reports failure when
     * the run could still have reached the device, which the caller must treat
     * as a reason not to hand its borrows back.
     */
    int abort_run_plan(uint32_t pipeline_slot, uint64_t run_epoch);
    /**
     * Whether this worker may order one run's native submission behind
     * another's right now, which is both a runtime capability and a
     * moment-to-moment fact about the device streams.
     */
    bool supports_joined_native_launch() const;
    /**
     * Whether `run` may be ordered behind `predecessor` while both collect
     * swimlane diagnostics.
     *
     * Read-only: nothing is armed, reserved or submitted, so a false answer
     * costs the caller only the round it asked in, and a true answer is not a
     * reservation. The backend rechecks everything it would need when the
     * launch actually arms.
     */
    bool supports_joined_diagnostic_launch(const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor);

    /**
     * The device context and the three credit entries, for a caller that must
     * keep calling them after this worker may be gone.
     *
     * Handed out once, to be copied into storage that outlives the worker: a
     * parked waiter reaches the runtime through that copy, never through
     * `worker->`, because the lane holds the worker as a raw pointer and a
     * waiter is exactly the thing that can outlive it.
     */
    using CreditGenerationFn = decltype(&simpler_diagnostic_credit_generation_ctx);
    using CreditWaitFn = decltype(&simpler_wait_diagnostic_credit_ctx);
    using CreditCancelFn = decltype(&simpler_cancel_diagnostic_credit_waits_ctx);

    struct CreditEntries {
        void *ctx = nullptr;
        CreditGenerationFn generation = nullptr;
        CreditWaitFn wait = nullptr;
        CreditCancelFn cancel = nullptr;
    };
    CreditEntries credit_entries() const {
        return CreditEntries{device_ctx_, credit_generation_fn_, credit_wait_fn_, credit_cancel_fn_};
    }
    /**
     * Whether this worker holds a live communication session or global domain.
     *
     * A joined successor must not depend on either: both are released by the
     * child before its device reset, so a run ordered behind another one cannot
     * be allowed to still be naming them when that happens.
     */
    bool holds_live_comm_resources() const;
    /**
     * Declare whether this worker's host side still owns exported device
     * regions — resources this class does not itself create and so cannot see.
     *
     * While it does, no run is ordered behind another: those regions are
     * released before the child's device reset, so a run that could still be
     * naming them when a peer's failure ends the generation must not be queued
     * behind that peer.
     */
    void set_exported_device_regions_live(bool live) { exported_device_regions_live_ = live; }
    bool exported_device_regions_live() const { return exported_device_regions_live_; }
    /**
     * How many runs this worker may have launched at once. One unless the
     * caller configured a greater depth and the backend supports it, and never
     * more than the contract's pipeline depth.
     */
    unsigned launch_depth() const {
        if (!initialized_ || launch_depth_ <= pipeline_contract_.pipeline_depth) return launch_depth_;
        return pipeline_contract_.pipeline_depth;
    }
    /**
     * Ask for a launch depth, before init.
     *
     * Depth one is the serial path every run took before this existed: nothing
     * is ordered behind anything, and no run constructs the boundary that would
     * let it be. A greater depth is a request — `launch_depth()` reports what it
     * resolved to against the backend's contract.
     */
    void configure_launch_depth(unsigned depth);

    /// Opaque host native-run storage address for every slot the contract
    /// asked for. Two slots hold distinct storage; tests read this to prove
    /// per-run state is not one buffer under two slot ids.
    std::vector<uint64_t> runtime_buffer_addrs() const;

    /// Committed GM heap base of one arena bank on the bound runner, or 0 when
    /// that bank has never been committed or the platform shares one arena set.
    uint64_t arena_bank_gm_heap_base(uint32_t bank_id) const;

    /// Retained temporary-buffer address the bound runner holds for one
    /// pipeline slot, or 0 while that slot holds none.
    uint64_t retained_temp_addr(uint32_t slot_id) const;
    size_t committed_device_memory() const;
    DeviceMemoryInfo device_memory_info() const;

private:
    using CreateDeviceContextFn = void *(*)();
    using DestroyDeviceContextFn = void (*)(void *);
    using DeviceMallocCtxFn = void *(*)(void *, size_t);
    using DeviceFreeCtxFn = void (*)(void *, void *);
    using DeviceFreeCallerBufferCtxFn = decltype(&device_free_caller_buffer_ctx);
    using DeviceBorrowCallerBuffersCtxFn = decltype(&device_borrow_caller_buffers_ctx);
    using DeviceReleaseCallerBuffersCtxFn = decltype(&device_release_caller_buffers_ctx);
    using CopyToDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using CopyFromDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using GetRuntimeSizeFn = size_t (*)();
    using GetRuntimeAlignmentFn = size_t (*)();
    using GetCommittedDeviceMemoryFn = size_t (*)(void *);
    using GetDeviceMemoryInfoFn = decltype(&device_memory_info_ctx);
    // From host_runtime.so. Single platform-side init that does (a) thread
    // attach + device-id record, (b) executor binary takeover, (c) onboard
    // CANN dlog sync, (d) async-DMA workspace provisioning. Reads the current
    // log level off HostLogger itself.
    using SimplerInitFn = decltype(&simpler_init);
    using SimplerRegisterCallableFn = int (*)(void *, int32_t, const void *);
    using SimplerRunFn = decltype(&simpler_run);
    using SimplerPrepareRunFn = decltype(&simpler_prepare_run);
    using SimplerNativeRunFn = decltype(&simpler_launch_run);
    using SimplerJoinedLaunchFn = decltype(&simpler_launch_run_joined);
    using ValidateExecutionContextFn = decltype(&simpler_validate_execution_context);
    using ValidateCallerStreamFn = decltype(&simpler_validate_caller_stream);
    using CallerStreamLaunchFn = decltype(&simpler_launch_run_on_stream);
    // Resolved optionally rather than through `RuntimeModule::required`: only onboard
    // modules carry the retention fixture, so a hard requirement would oblige
    // every simulated and stand-in runtime to export a stub of it.
    using SimplerProbeRunRetentionFn = decltype(&simpler_probe_run_retention);
    using GetTeardownReportFn = decltype(&get_teardown_report);
    using SupportsConcurrentNativePrepareFn = int (*)(void *);
    using SupportsJoinedDiagnosticLaunchFn = int (*)(void *, void *, void *);
    using GetArenaBankGmHeapBaseFn = uint64_t (*)(void *, uint32_t);
    using GetRetainedTempAddrFn = uint64_t (*)(void *, uint32_t);
    using GetPipelineContractFn = const PipelineContract *(*)();
    using SimplerUnregisterCallableFn = int (*)(void *, int32_t);
    using GetAicpuDlopenCountFn = size_t (*)(void *);
    using FinalizeDeviceFn = int (*)(void *);
    using EnsureAclReadyFn = int (*)(void *, int);
    using CreateCommStreamFn = void *(*)(void *);
    using DestroyCommStreamFn = int (*)(void *, void *);
    using CommInitFn = void *(*)(int, int, void *, const char *);
    using CommAllocWindowsFn = int (*)(void *, size_t, uint64_t *);
    using CommGetLocalWindowBaseFn = int (*)(void *, uint64_t *);
    using CommGetWindowSizeFn = int (*)(void *, size_t *);
    using CommDeriveContextFn = int (*)(void *, const uint32_t *, size_t, uint32_t, size_t, size_t, uint64_t *);
    using CommAllocDomainWindowsFn =
        int (*)(void *, uint64_t, const uint32_t *, size_t, uint32_t, size_t, uint64_t *, uint64_t *);
    using CommReleaseDomainWindowsFn = int (*)(void *, uint64_t, size_t, uint32_t);
    using CommGlobalDomainPrepareFn =
        int (*)(uint64_t, uint32_t, uint32_t, size_t, uint32_t, CommGlobalDomainDescriptor *, uint64_t *);
    using CommGlobalDomainImportFn = int (*)(uint64_t, const CommGlobalDomainDescriptor *, size_t, uint64_t *);
    using CommGlobalDomainReleaseFn = int (*)(uint64_t);
    using CommBarrierFn = int (*)(void *);
    using CommDestroyFn = int (*)(void *);
    using KernelSupportedFn = decltype(&simpler_kernel_mode_supported);
    using KernelInitFn = decltype(&simpler_kernel_mode_init);
    using KernelPrepareCallableFn = decltype(&simpler_kernel_mode_prepare_callable);
    using KernelLaunchFn = decltype(&simpler_kernel_mode_launch);

    struct CommSession {
        void *handle = nullptr;
        void *stream = nullptr;
        bool is_base = false;
        uint64_t device_ctx = 0;
        uint64_t local_window_base = 0;
        size_t window_size = 0;
    };

    void *create_comm_stream_checked(const char *op_name);
    void destroy_comm_stream_best_effort(void *stream, int *rc);
    CommSession *find_comm_session(uint64_t comm_handle);
    CommSession *create_comm_session(void *handle, void *stream, bool is_base);
    int destroy_comm_session(CommSession &session);
    uint64_t create_base_comm(int rank, int nranks, const std::string &rootinfo_path);
    void clear_comm_sessions();

    std::shared_ptr<RuntimeModule> runtime_module_;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    DeviceMallocCtxFn device_malloc_ctx_fn_ = nullptr;
    DeviceFreeCtxFn device_free_ctx_fn_ = nullptr;
    DeviceFreeCallerBufferCtxFn device_free_caller_buffer_ctx_fn_ = nullptr;
    DeviceBorrowCallerBuffersCtxFn device_borrow_caller_buffers_ctx_fn_ = nullptr;
    DeviceReleaseCallerBuffersCtxFn device_release_caller_buffers_ctx_fn_ = nullptr;
    CopyToDeviceCtxFn copy_to_device_ctx_fn_ = nullptr;
    CopyFromDeviceCtxFn copy_from_device_ctx_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    GetRuntimeAlignmentFn get_runtime_alignment_fn_ = nullptr;
    GetCommittedDeviceMemoryFn device_committed_memory_fn_ = nullptr;
    GetDeviceMemoryInfoFn device_memory_info_fn_ = nullptr;
    SimplerInitFn simpler_init_fn_ = nullptr;
    SimplerRegisterCallableFn register_callable_fn_ = nullptr;
    using SimplerRegisterCallableV2Fn = decltype(&simpler_register_callable_v2);
    SimplerRegisterCallableV2Fn register_callable_v2_fn_ = nullptr;
    SimplerRunFn run_fn_ = nullptr;
    SimplerPrepareRunFn prepare_run_fn_ = nullptr;
    SimplerNativeRunFn launch_run_fn_ = nullptr;
    ValidateExecutionContextFn validate_execution_context_fn_ = nullptr;
    ValidateCallerStreamFn validate_caller_stream_fn_ = nullptr;
    CallerStreamLaunchFn launch_run_on_stream_fn_ = nullptr;
    SimplerJoinedLaunchFn launch_run_joined_fn_ = nullptr;
    SimplerNativeRunFn poll_run_fn_ = nullptr;
    SimplerNativeRunFn wait_run_fn_ = nullptr;
    SimplerNativeRunFn finalize_run_fn_ = nullptr;
    SimplerProbeRunRetentionFn probe_run_retention_fn_ = nullptr;
    GetTeardownReportFn get_teardown_report_fn_ = nullptr;
    // The first captured teardown observation, kept by value so it outlives
    // the context destruction and the dlclose that follow a successful
    // finalize. Write-once: a later finalize cannot replace a real result.
    SimplerTeardownReport teardown_report_{};
    bool teardown_report_captured_ = false;
    using SimplerSetRetainRunsFn = decltype(&simpler_set_retain_runs_ctx);
    using SimplerSetRetainedEpochLimitFn = decltype(&simpler_set_retained_epoch_limit_ctx);
    using SimplerSetRunResourceGrantFn = decltype(&simpler_set_run_resource_grant_ctx);
    using SimplerFlushDiagnosticsFn = decltype(&simpler_flush_diagnostics_ctx);
    SimplerSetRetainRunsFn set_retain_runs_fn_ = nullptr;
    // Optional: a module built before the capacity grant exports no such
    // symbol, and a null pointer leaves `retained_epoch_limit_granted_` at the
    // shipped figure.
    SimplerSetRetainedEpochLimitFn set_retained_epoch_limit_fn_ = nullptr;
    SimplerSetRunResourceGrantFn set_run_resource_grant_fn_ = nullptr;
    // What the backend actually granted, recorded only on a success return.
    // The joined-diagnostic gate reads it, so a refused request can never let
    // the lane order more runs than the collector can hold.
    size_t retained_epoch_limit_granted_ = simpler::dfx::runs::kMaxOpenEpochs;
    using SimplerSetWorkspaceBudgetFn = decltype(&simpler_set_workspace_budget_ctx);
    using SimplerEnableAutomaticAdmissionFn = decltype(&simpler_enable_automatic_admission_ctx);
    using SimplerAdmissibleRunResourceSetsFn = decltype(&simpler_admissible_run_resource_sets_ctx);
    using SimplerAbortRunPlanFn = decltype(&simpler_abort_run_plan_ctx);
    using SimplerEnableWorkspaceManagementFn = decltype(&simpler_enable_workspace_management_ctx);
    using SimplerGetWorkspaceReportFn = decltype(&simpler_get_workspace_report_ctx);
    SimplerSetWorkspaceBudgetFn set_workspace_budget_fn_ = nullptr;
    SimplerEnableAutomaticAdmissionFn enable_automatic_admission_fn_ = nullptr;
    SimplerAdmissibleRunResourceSetsFn admissible_run_resource_sets_fn_ = nullptr;
    SimplerAbortRunPlanFn abort_run_plan_fn_ = nullptr;
    bool automatic_admission_ = false;
    // Resolved for the retained compatibility entry, which a direct C caller
    // may still invoke before init. `init` no longer calls it: the program
    // init entry installs the manager for every caller.
    SimplerEnableWorkspaceManagementFn enable_workspace_management_fn_ = nullptr;
    SimplerGetWorkspaceReportFn get_workspace_report_fn_ = nullptr;
    // Latched when init accepted a budget. A caller protecting teardown reads
    // this, never "the last query worked", so a failed query cannot pass for a
    // context that never had a budget.
    // Management and a finite limit are separate predicates: a managed context
    // may enforce no limit, and the report must not confuse the two.
    bool workspace_managed_ = false;
    bool workspace_limit_latched_ = false;
    SimplerFlushDiagnosticsFn flush_diagnostics_fn_ = nullptr;
    SupportsConcurrentNativePrepareFn supports_concurrent_native_prepare_fn_ = nullptr;
    SupportsConcurrentNativePrepareFn supports_joined_native_launch_fn_ = nullptr;
    SupportsJoinedDiagnosticLaunchFn supports_joined_diagnostic_launch_fn_ = nullptr;
    // The credit bridge. A runtime that exports none of these leaves the
    // slots null, which degrades to "no credit wait": the caller re-asks on
    // its own next pass, exactly as it does today.
    CreditGenerationFn credit_generation_fn_ = nullptr;
    CreditWaitFn credit_wait_fn_ = nullptr;
    CreditCancelFn credit_cancel_fn_ = nullptr;
    GetArenaBankGmHeapBaseFn get_arena_bank_gm_heap_base_fn_ = nullptr;
    GetRetainedTempAddrFn get_retained_temp_addr_fn_ = nullptr;
    SimplerUnregisterCallableFn unregister_callable_fn_ = nullptr;
    GetAicpuDlopenCountFn get_aicpu_dlopen_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_host_dlopen_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_run_stream_set_create_count_fn_ = nullptr;
    FinalizeDeviceFn finalize_device_fn_ = nullptr;
    EnsureAclReadyFn ensure_acl_ready_fn_ = nullptr;
    CreateCommStreamFn create_comm_stream_fn_ = nullptr;
    DestroyCommStreamFn destroy_comm_stream_fn_ = nullptr;
    CommInitFn comm_init_fn_ = nullptr;
    CommAllocWindowsFn comm_alloc_windows_fn_ = nullptr;
    CommGetLocalWindowBaseFn comm_get_local_window_base_fn_ = nullptr;
    CommGetWindowSizeFn comm_get_window_size_fn_ = nullptr;
    CommDeriveContextFn comm_derive_context_fn_ = nullptr;
    CommAllocDomainWindowsFn comm_alloc_domain_windows_fn_ = nullptr;
    CommReleaseDomainWindowsFn comm_release_domain_windows_fn_ = nullptr;
    CommGlobalDomainPrepareFn comm_global_domain_prepare_fn_ = nullptr;
    CommGlobalDomainImportFn comm_global_domain_import_fn_ = nullptr;
    CommGlobalDomainReleaseFn comm_global_domain_release_fn_ = nullptr;
    CommBarrierFn comm_barrier_fn_ = nullptr;
    CommDestroyFn comm_destroy_fn_ = nullptr;
    KernelSupportedFn kernel_supported_fn_ = nullptr;
    KernelInitFn kernel_init_fn_ = nullptr;
    KernelPrepareCallableFn kernel_prepare_callable_fn_ = nullptr;
    KernelLaunchFn kernel_launch_fn_ = nullptr;
    void *device_ctx_ = nullptr;
    std::vector<CommSession> comm_sessions_;
    std::unordered_map<uint64_t, size_t> comm_session_index_;
    std::unordered_set<uint64_t> global_domain_ids_;
    uint64_t base_comm_handle_ = 0;

    uint32_t arena_bank_for_slot(uint32_t slot_id) const;

    enum class NativeRunPhase : uint8_t { EMPTY, PREPARING, PREPARED, LAUNCHED, REAPED, FINALIZING };
    class PreparedRunIncompatible : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };
    /**
     * This run's device resources are not available yet, and a consumer that
     * holds some of them has a finish this context will reach.
     *
     * Distinct from `PreparedRunIncompatible`, which is about a layout two runs
     * cannot share and clears when the run ahead retires, and from a failure,
     * which ends the run. The caller keeps the run queued with everything it
     * holds; the worker's saved plan is what the next attempt continues from,
     * so the caller's orchestration callback is not invoked again.
     */
    class PreparedRunResourceWait : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };
    /**
     * One live native run, owned by this worker and keyed by its own epoch.
     *
     * `phase` is what the run's execution established, `finalize_reserved` is which caller owns its
     * finalization, and `pending_rc` is the error a later finalize must report; they are separate
     * fields because one of them overwriting another is exactly what this must not permit.
     * `ops_in_flight` is zero or one, so at most one operation is inside the runtime for one run.
     * The entry outlives the carrier object the runtime places in that run's storage.
     */
    struct NativeRunEntry {
        uint32_t slot_id{0};
        uint64_t lease_generation{0};
        uint64_t run_epoch{0};
        NativeRunPhase phase{NativeRunPhase::EMPTY};
        unsigned ops_in_flight{0};
        bool finalize_reserved{false};
        /** Teardown progress is unknown, so no further call may reach this run. */
        bool destruction_unknown{false};
        /** First writer wins: an established error is never replaced by a later one. */
        int pending_rc{0};
        bool pending_rc_set{false};
        bool permits_prepared_successor{false};
    };
    /** What the storage tag says after a finalize that reached the C entry. */
    enum class RunStorageTag : uint8_t { Destroyed, Canonical, Noncanonical };
    /** Which operation a lease is being taken for; decides blocking and the admitted phases. */
    enum class NativeRunOp : uint8_t { Launch, Poll, Wait, Finalize, Probe, Predecessor, ProbeSuccessor };
    /** Why a lease was refused, or that the run had already completed. */
    enum class NativeRunLease : uint8_t { Acquired, AlreadyComplete, Absent, Stale, Reserved, Busy };
    ChipWorkerNativeRun prepare_native_run_on_slot(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, uint32_t slot_id,
        uint64_t generation, uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state,
        int32_t accepted_value, bool admit_pipeline_generation, NativeRunProvenance provenance
    );
    ChipWorkerNativeRun prepare_native_run_for_lane(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id, uint64_t dispatch_id, volatile int32_t *accepted_state, int32_t accepted_value,
        bool pipeline_leased, NativeRunProvenance provenance
    );
    void cleanup_native_runs_noexcept() noexcept;
    void clear_runtime_binding();
    void abandon_after_failed_teardown(std::exception_ptr error) noexcept;
    /** How many runs a teardown sweep left behind, each of them not proved destroyed. */
    size_t unproven_native_run_count() const noexcept;
    /**
     * Detach the run storage of every run not proved destroyed.
     *
     * Returns how many blocks were abandoned. Retaining the map entry cannot
     * preserve anything: the entry and the storage both belong to this object,
     * and once its destructor returns the storage is freed whatever the map
     * said. Abandoning the block is what actually outlives the owner.
     */
    size_t abandon_unproven_native_run_storage() noexcept;

    /** Fill the write-once teardown copy from the still-live context; never throws. */
    void capture_teardown_report_noexcept() noexcept;

    friend struct ChipRunLaneState;

    class RuntimeStorage {
    public:
        RuntimeStorage() = default;
        RuntimeStorage(size_t size, size_t alignment, std::shared_ptr<RuntimeModule> module);
        ~RuntimeStorage();
        RuntimeStorage(RuntimeStorage &&other) noexcept;
        RuntimeStorage &operator=(RuntimeStorage &&other) noexcept;
        RuntimeStorage(const RuntimeStorage &) = delete;
        RuntimeStorage &operator=(const RuntimeStorage &) = delete;

        // Unproven completion retains the allocation for the process lifetime.
        void *data() { return data_; }
        const void *data() const { return data_; }
        size_t size() const noexcept { return size_; }

        /**
         * Give up ownership of the block without freeing it.
         *
         * For a run whose carrier could not be proved destroyed. The block then
         * outlives this object, which is the whole point: a carrier the device
         * or the AICPU may still reach must not have its bytes returned to the
         * allocator, and no owner that dies can keep it. Deliberately a leak —
         * a bounded one, with the reason logged, which is what the alternative
         * of reusing those bytes cannot offer.
         */
        void abandon() noexcept;

    private:
        void *data_{nullptr};
        size_t size_{0};
        std::shared_ptr<RuntimeModule> module_;
    };

    // Allocated once during init and never resized. Each allocation honors the
    // runtime-reported alignment and keeps a stable C ABI address through
    // prepare/finalize even if the owning vector itself is moved.
    std::vector<RuntimeStorage> runtime_bufs_;
    /**
     * Every live native run, keyed by the epoch that names it.
     *
     * A run's identity is its epoch, which no later run repeats; a slot is only which resource set
     * it borrows. Values are held by pointer so an entry keeps its address across a rehash, though
     * correctness rests on every re-entry re-looking-up by epoch rather than on that.
     */
    std::unordered_map<uint64_t, std::unique_ptr<NativeRunEntry>> native_runs_;
    mutable std::mutex native_run_mu_;
    /** Signalled whenever an operation lease is released or a run entry is removed. */
    std::condition_variable native_run_ops_done_;

    /**
     * Take the exclusive operation lease for `run`, or report why not.
     *
     * Re-looks-up and fully revalidates after every wake: while the wait releases the mutex, a
     * finalizer may reserve the run and another operation may remove it, so a predicate proved
     * before the wait proves nothing after it. Never returns a pointer the caller holds across
     * the lock — `storage_out` is the runtime storage address, whose lifetime is the slot's.
     */
    /**
     * Holds one run's operation lease and releases it on every exit path.
     *
     * Scope-bound because the release has to survive a throw between the runtime call and the
     * caller's own tail: an exception constructed there would otherwise leave `ops_in_flight` set
     * and nothing could finalize that run again.
     */
    class NativeRunOpLease {
    public:
        NativeRunOpLease() = default;
        NativeRunOpLease(ChipWorker *owner, uint64_t run_epoch) noexcept :
            owner_(owner),
            run_epoch_(run_epoch) {}
        NativeRunOpLease(const NativeRunOpLease &) = delete;
        NativeRunOpLease &operator=(const NativeRunOpLease &) = delete;
        NativeRunOpLease(NativeRunOpLease &&other) noexcept :
            owner_(std::exchange(other.owner_, nullptr)),
            run_epoch_(other.run_epoch_) {}
        NativeRunOpLease &operator=(NativeRunOpLease &&other) noexcept {
            if (this != &other) {
                if (owner_ != nullptr) owner_->release_native_run_op(run_epoch_);
                owner_ = std::exchange(other.owner_, nullptr);
                run_epoch_ = other.run_epoch_;
            }
            return *this;
        }
        ~NativeRunOpLease() {
            if (owner_ != nullptr) owner_->release_native_run_op(run_epoch_);
        }

    private:
        ChipWorker *owner_{nullptr};
        uint64_t run_epoch_{0};
    };
    NativeRunLease acquire_native_run_op(
        const ChipWorkerNativeRun &run, NativeRunOp op, void **storage_out, NativeRunPhase *phase_out,
        NativeRunOpLease *lease_out
    );
    /** Whether a lease refused only because another operation holds it waits, or reports it now. */
    enum class LeaseWait : uint8_t { Block, Immediate };
    /**
     * Take both runs' leases at once, or neither.
     *
     * A joined launch dereferences the predecessor's carrier as well as its own, so holding only
     * the successor's lease leaves the other free to be finalized and its object destroyed under
     * the call. Both are taken under one hold of the mutex, lowest epoch first so two joins that
     * name the same pair in opposite orders cannot wedge, and the whole pair is retried from
     * scratch after any wake. `LeaseWait::Immediate` reports `Busy` instead of waiting, which is
     * what a scheduling probe needs: it asks again next round rather than holding a thread.
     */
    NativeRunLease acquire_native_run_pair(
        const ChipWorkerNativeRun &run, const ChipWorkerNativeRun &predecessor, NativeRunOp op, void **storage_out,
        void **predecessor_storage_out, NativeRunOpLease *lease_out, NativeRunOpLease *predecessor_lease_out,
        LeaseWait wait
    );
    /** Release the lease taken above and wake anything waiting for it. */
    void release_native_run_op(uint64_t run_epoch) noexcept;
    /** Classify the storage tag the runtime leaves behind, without dereferencing the object. */
    RunStorageTag read_run_storage_tag(const void *storage) const noexcept;
    /** Which run phases one operation may be taken for. */
    static bool native_run_op_admits_phase(NativeRunOp op, NativeRunPhase phase);
    /** Decide one run's lease eligibility; `native_run_mu_` is held by the caller. */
    NativeRunLease
    inspect_native_run_locked(const ChipWorkerNativeRun &run, NativeRunOp op, NativeRunEntry **entry_out);
    /** Report a refused lease with the reason the caller is owed. */
    [[noreturn]] static void
    throw_native_run_lease_error(const ChipWorkerNativeRun &run, NativeRunLease lease, const char *operation);
    /** Record that a run's work reached the device. */
    void note_native_run_launched(uint64_t run_epoch) noexcept;
    /** Record that a run's device work is over, and the error it carries, if any. */
    void note_native_run_completion(uint64_t run_epoch, int rc) noexcept;
    /** Remove an entry that never got past preparation. */
    void drop_preparing_native_run(uint64_t run_epoch) noexcept;
    /** Whether a finalization reports to a caller or is closing this worker down. */
    enum class FinalizeMode : uint8_t { Caller, Teardown };
    /** Why a finalization has something to report; the message belongs to whoever raises it. */
    enum class FinalizeFault : uint8_t {
        None,
        /** The token names no live run, or one already finalized. */
        Stale,
        /** Another caller owns this run's finalization. */
        Reserved,
        /** The teardown returned success with the carrier still canonical. */
        NonDestroying,
        /** The carrier's tag is neither destroyed nor canonical, or teardown progress is unknown. */
        UnknownDestruction,
        /** A nonzero code, carried in `rc`. */
        Failed,
    };
    /**
     * What one finalization established, in a form that costs no allocation to produce.
     *
     * `captured` is the original exception when one came out of the runtime or the bookkeeping;
     * everything else is a reason and a code. A message is built only by a caller that raises,
     * which is what lets the finalization itself be `noexcept` with nothing to construct —
     * teardown, which has no caller to report to, builds nothing at all.
     */
    struct NativeRunFinalizeOutcome {
        FinalizeFault fault{FinalizeFault::None};
        int rc{0};
        bool no_submission_released{false};
        std::exception_ptr captured;
    };
    // This out-fact survives a persisted launch error, never failed cleanup.
    void finalize_native_run_impl(const ChipWorkerNativeRun &run, bool *no_submission_released);
    /**
     * Finalize one run and report what it established.
     *
     * `noexcept` and returning the outcome rather than throwing it, so teardown can share this
     * exact ordering instead of carrying a second implementation of it. A null `token` identifies
     * the run by epoch alone, which is what teardown has.
     */
    NativeRunFinalizeOutcome
    finalize_native_run_epoch(uint64_t run_epoch, const ChipWorkerNativeRun *token, FinalizeMode mode) noexcept;
    /** Give back a finalization reservation that could not be used; never throws. */
    void clear_finalize_reservation_noexcept(uint64_t run_epoch) noexcept;
    PipelineSlotGenerationFilter pipeline_generations_;
    PipelineContract pipeline_contract_{PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};
    // How many runs may be launched at once. One is the serial path; a greater
    // value is clamped to the contract's pipeline depth at init, because a
    // launched run holds a slot for its whole lifetime.
    unsigned launch_depth_ = 1;
    // Set by the host side that owns exported device regions, which this class
    // neither creates nor tracks. Plain: written and read from the one thread
    // that drives this worker's control commands and its run lane.
    bool exported_device_regions_live_ = false;
    std::unique_ptr<ChipRunLane> run_lane_;
    // Names a successful binding or an init whose rollback still needs close.
    // Cleared only after successful teardown. Control calls run on the owning
    // thread, as enforced by Worker.close().
    std::optional<ExecutionEnvironment> execution_environment_;
    int device_id_ = -1;
    bool initialized_ = false;
    bool finalized_ = false;
};
