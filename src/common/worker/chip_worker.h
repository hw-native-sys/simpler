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

#ifndef SRC_COMMON_WORKER_CHIP_WORKER_H_
#define SRC_COMMON_WORKER_CHIP_WORKER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "pipeline_slot_pool.h"
#include "pto_runtime_c_api.h"
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

class ChipWorker {
public:
    ChipWorker() = default;
    ~ChipWorker();

    ChipWorker(const ChipWorker &) = delete;
    ChipWorker &operator=(const ChipWorker &) = delete;

    /// Bind the runtime library, cache platform binaries, and attach the
    /// calling thread to `device_id`. Can only be called once per lifetime —
    /// the runtime and device cannot be changed after init.
    ///
    /// The process-wide RTLD_GLOBAL bootstrap loads (libsimpler_log.so, and on
    /// sim libcpu_sim_context.so — plus seeding HostLogger via simpler_log_init)
    /// are the caller's responsibility and must already have happened before
    /// this call: host_runtime.so resolves its undefined HostLogger /
    /// unified_log_* (and, on sim, sim_context_*) symbols against those
    /// globals. The Python `ChipWorker` wrapper does this with `ctypes.CDLL(...,
    /// mode=RTLD_GLOBAL)`.
    /// `prewarm_config`, when non-null, builds + caches the prebuilt
    /// runtime-arena for its ring sizing right after the device comes up (the
    /// sizing is fork-constant, delivered by COW into init). A no-op for
    /// runtimes without a prebuilt arena.
    /// `dma_workspace_mask` (a bitmask of DmaWorkspaceKind bits, 0 = none)
    /// provisions those async-DMA workspaces once at init, so kernels can use
    /// get_dma_workspace. Empty by default; a Worker that does not opt in creates
    /// no SDMA streams. Provisioning fails fast (init throws) on a
    /// platform/runtime that does not support a requested engine. The mask stays
    /// a raw integer here so this platform-agnostic worker needs no platform
    /// headers; the binding derives it from the DmaWorkspaceKind enum.
    void init(
        const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
        const std::string &dispatcher_path, int device_id, const CallConfig *prewarm_config = nullptr,
        uint32_t dma_workspace_mask = 0
    );

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // Launch a cid previously staged via register_callable.
    // Materializes a ChipStorageTaskArgs from `args` (one memcpy of T*40B + S*8B
    // into a stack POD), then delegates to the overload below. Per-stage timing
    // (host wall, on-NPU device wall + AICPU phase breakdown) is emitted by the
    // platform as `[STRACE]` log markers — see src/common/log/.../strace.h — not
    // returned, so the L3 dispatcher and L2 child are observed uniformly.
    void run(int32_t callable_id, TaskArgsView args, const CallConfig &config);
    void
    run(int32_t callable_id, TaskArgsView args, const CallConfig &config, volatile int32_t *accepted_state,
        int32_t accepted_value);
    // Same launch, but the caller already holds the runtime.so-ABI POD —
    // skip the view→storage memcpy and hand the pointer straight to the C ABI.
    // Used by the ChipStorageTaskArgs path in the nanobind binding.
    void run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config);
    void
    run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config,
        volatile int32_t *accepted_state, int32_t accepted_value);
    void run_with_lease(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0
    );
    void run_with_lease(
        int32_t callable_id, TaskArgsView args, const CallConfig &config, const PipelineSlotLease &lease,
        volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0
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
     * A capable backend may prepare one distinct-slot successor while another
     * run owns the execution claim. Diagnostics and backends without that
     * explicit capability remain depth-one. The slot/lease-generation/process-
     * unique-run-epoch token prevents a delayed phase call from touching reused
     * storage, including another run under the same pipeline lease or on
     * another ChipWorker.
     */
    ChipWorkerNativeRun prepare_native_run(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id = 0, uint64_t dispatch_id = 0
    );
    ChipWorkerNativeRun prepare_native_run(
        int32_t callable_id, TaskArgsView args, const CallConfig &config, const PipelineSlotLease &lease,
        uint64_t run_id = 0, uint64_t dispatch_id = 0
    );
    void launch_native_run(
        const ChipWorkerNativeRun &run, volatile int32_t *accepted_state = nullptr, int32_t accepted_value = 0
    );
    bool poll_native_run(const ChipWorkerNativeRun &run);
    void wait_native_run(const ChipWorkerNativeRun &run);
    void finalize_native_run(const ChipWorkerNativeRun &run);

    // Per-callable_id preparation. Requires init() first and a callable_id
    // in [0, MAX_REGISTERED_CALLABLE_IDS) (cap 64).
    void register_callable(int32_t callable_id, const void *callable);
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

    /// Number of AICore run streams the bound runner has created. AICPU streams
    /// belong to slots for the worker's lifetime; each run gets a freshly
    /// created AICore stream, so this advances once per run. Platforms using
    /// the persistent bootstrap pair report 0.
    size_t run_stream_set_create_count() const;

    /// Number of persistent native execution threads created by the runner.
    size_t native_execution_thread_create_count() const;

    uint64_t malloc(size_t size);
    void free(uint64_t ptr);
    void copy_to(uint64_t dst, uint64_t src, size_t size);
    void copy_from(uint64_t dst, uint64_t src, size_t size);

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
    void comm_barrier(uint64_t comm_handle);
    void comm_destroy(uint64_t comm_handle);
    void comm_destroy_all();

    int device_id() const { return device_id_; }
    bool initialized() const { return initialized_; }
    unsigned pipeline_depth() const { return pipeline_contract_.pipeline_depth; }
    size_t runtime_slot_count() const { return runtime_bufs_.size(); }
    bool supports_concurrent_native_prepare() const;

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

private:
    using CreateDeviceContextFn = void *(*)();
    using DestroyDeviceContextFn = void (*)(void *);
    using DeviceMallocCtxFn = void *(*)(void *, size_t);
    using DeviceFreeCtxFn = void (*)(void *, void *);
    using CopyToDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using CopyFromDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using GetRuntimeSizeFn = size_t (*)();
    using GetRuntimeAlignmentFn = size_t (*)();
    using GetCommittedDeviceMemoryFn = size_t (*)(void *);
    // From host_runtime.so. Single platform-side init that does (a) thread
    // attach + device-id record, (b) executor binary takeover, (c) onboard
    // CANN dlog sync. Reads the current log level off HostLogger itself.
    using SimplerInitFn = int (*)(
        void *, int, const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, const CallConfig *
    );
    using SimplerRegisterCallableFn = int (*)(void *, int32_t, const void *);
    using SimplerRunFn = int (*)(void *, void *, int32_t, const void *, const CallConfig *);
    using SimplerPrepareRunFn = int (*)(void *, void *, int32_t, const void *, const CallConfig *);
    using SimplerNativeRunFn = int (*)(void *, void *);
    using SupportsConcurrentNativePrepareFn = int (*)(void *);
    using SetTaskAcceptedStateFn = int (*)(void *, volatile int32_t *, int32_t);
    using SelectPipelineSlotFn = int (*)(void *, uint32_t);
    using SelectArenaBankFn = int (*)(void *, uint32_t);
    using SetNativeRunIdentityFn = int (*)(void *, uint64_t, uint64_t, uint64_t, uint64_t);
    using GetArenaBankGmHeapBaseFn = uint64_t (*)(void *, uint32_t);
    using GetRetainedTempAddrFn = uint64_t (*)(void *, uint32_t);
    using GetPipelineContractFn = const PipelineContract *(*)();
    using SimplerUnregisterCallableFn = int (*)(void *, int32_t);
    using GetAicpuDlopenCountFn = size_t (*)(void *);
    using SimplerProvisionDmaWorkspaceFn = int (*)(void *, uint32_t);
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
    using CommBarrierFn = int (*)(void *);
    using CommDestroyFn = int (*)(void *);

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

    void *lib_handle_ = nullptr;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    DeviceMallocCtxFn device_malloc_ctx_fn_ = nullptr;
    DeviceFreeCtxFn device_free_ctx_fn_ = nullptr;
    CopyToDeviceCtxFn copy_to_device_ctx_fn_ = nullptr;
    CopyFromDeviceCtxFn copy_from_device_ctx_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    GetRuntimeAlignmentFn get_runtime_alignment_fn_ = nullptr;
    GetCommittedDeviceMemoryFn device_committed_memory_fn_ = nullptr;
    SimplerInitFn simpler_init_fn_ = nullptr;
    SimplerRegisterCallableFn register_callable_fn_ = nullptr;
    SimplerRunFn run_fn_ = nullptr;
    SimplerPrepareRunFn prepare_run_fn_ = nullptr;
    SimplerNativeRunFn launch_run_fn_ = nullptr;
    SimplerNativeRunFn poll_run_fn_ = nullptr;
    SimplerNativeRunFn wait_run_fn_ = nullptr;
    SimplerNativeRunFn finalize_run_fn_ = nullptr;
    SupportsConcurrentNativePrepareFn supports_concurrent_native_prepare_fn_ = nullptr;
    SetTaskAcceptedStateFn set_task_accepted_state_fn_ = nullptr;
    SelectPipelineSlotFn select_pipeline_slot_fn_ = nullptr;
    SelectArenaBankFn select_arena_bank_fn_ = nullptr;
    SetNativeRunIdentityFn set_native_run_identity_fn_ = nullptr;
    GetArenaBankGmHeapBaseFn get_arena_bank_gm_heap_base_fn_ = nullptr;
    GetRetainedTempAddrFn get_retained_temp_addr_fn_ = nullptr;
    SimplerUnregisterCallableFn unregister_callable_fn_ = nullptr;
    GetAicpuDlopenCountFn get_aicpu_dlopen_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_host_dlopen_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_run_stream_set_create_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_native_execution_thread_create_count_fn_ = nullptr;
    SimplerProvisionDmaWorkspaceFn simpler_provision_dma_workspace_fn_ = nullptr;
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
    CommBarrierFn comm_barrier_fn_ = nullptr;
    CommDestroyFn comm_destroy_fn_ = nullptr;
    void *device_ctx_ = nullptr;
    std::vector<CommSession> comm_sessions_;
    std::unordered_map<uint64_t, size_t> comm_session_index_;
    uint64_t base_comm_handle_ = 0;

    // Slot 0 with no generation bookkeeping: an unleased run is a caller that
    // is not using the pipeline, and minting a generation for it here would
    // advance the filter past the leases a real pool later presents.
    static constexpr uint32_t UNLEASED_SLOT = 0;
    void run_on_slot(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, uint32_t slot_id,
        volatile int32_t *accepted_state, int32_t accepted_value
    );
    uint32_t select_slot_resources(uint32_t slot_id);

    enum class NativeRunPhase : uint8_t { EMPTY, PREPARING, PREPARED, LAUNCHING, LAUNCHED, REAPED, FINALIZING };
    struct NativeRunSlotState {
        uint64_t lease_generation{0};
        uint64_t run_epoch{0};
        NativeRunPhase phase{NativeRunPhase::EMPTY};
        int wait_rc{0};
        bool permits_prepared_successor{false};
    };
    ChipWorkerNativeRun prepare_native_run_on_slot(
        int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config, uint32_t slot_id,
        uint64_t generation, uint64_t run_id, uint64_t dispatch_id
    );
    void cleanup_native_runs_noexcept() noexcept;

    class RuntimeStorage {
    public:
        RuntimeStorage() = default;
        RuntimeStorage(size_t size, size_t alignment);
        ~RuntimeStorage();
        RuntimeStorage(RuntimeStorage &&other) noexcept;
        RuntimeStorage &operator=(RuntimeStorage &&other) noexcept;
        RuntimeStorage(const RuntimeStorage &) = delete;
        RuntimeStorage &operator=(const RuntimeStorage &) = delete;

        void *data() { return data_; }
        const void *data() const { return data_; }

    private:
        void *data_{nullptr};
    };

    // Allocated once during init and never resized. Each allocation honors the
    // runtime-reported alignment and keeps a stable C ABI address through
    // prepare/finalize even if the owning vector itself is moved.
    std::vector<RuntimeStorage> runtime_bufs_;
    std::array<NativeRunSlotState, PTO_PIPELINE_MAX_DEPTH> native_run_states_{};
    mutable std::mutex native_run_mu_;
    PipelineSlotGenerationFilter pipeline_generations_;
    PipelineContract pipeline_contract_{PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};
    // device_id_ is set once in init() and never modified afterward. All
    // ChipWorker callers run on the thread that called init() (the same
    // thread is the only one that subsequently calls malloc / copy_to /
    // run / finalize), so plain `int` is sufficient — no cross-thread
    // synchronization required.
    int device_id_ = -1;
    bool initialized_ = false;
    bool finalized_ = false;
};

#endif  // SRC_COMMON_WORKER_CHIP_WORKER_H_
