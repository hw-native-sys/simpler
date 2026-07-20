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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "pto_runtime_c_api.h"
#include "types.h"

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
    void init(
        const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
        const std::string &dispatcher_path, int device_id, const CallConfig *prewarm_config = nullptr
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
    // Same launch, but the caller already holds the runtime.so-ABI POD —
    // skip the view→storage memcpy and hand the pointer straight to the C ABI.
    // Used by the ChipStorageTaskArgs path in the nanobind binding.
    void run(int32_t callable_id, const ChipStorageTaskArgs *args, const CallConfig &config);

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
    /// Zero this rank's local window for a live dynamic allocation. The caller
    /// must ensure all prior device work using the allocation has drained.
    void comm_reset_domain_windows(uint64_t comm_handle, uint64_t allocation_id);
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

private:
    using CreateDeviceContextFn = void *(*)();
    using DestroyDeviceContextFn = void (*)(void *);
    using DeviceMallocCtxFn = void *(*)(void *, size_t);
    using DeviceFreeCtxFn = void (*)(void *, void *);
    using CopyToDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using CopyFromDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using GetRuntimeSizeFn = size_t (*)();
    // From host_runtime.so. Single platform-side init that does (a) thread
    // attach + device-id record, (b) executor binary takeover, (c) onboard
    // CANN dlog sync. Reads the current log level off HostLogger itself.
    using SimplerInitFn = int (*)(
        void *, int, const uint8_t *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, const CallConfig *
    );
    using SimplerRegisterCallableFn = int (*)(void *, int32_t, const void *);
    using SimplerRunFn = int (*)(void *, void *, int32_t, const void *, const CallConfig *);
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
    using CommResetDomainWindowsFn = int (*)(void *, uint64_t);
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
    SimplerInitFn simpler_init_fn_ = nullptr;
    SimplerRegisterCallableFn register_callable_fn_ = nullptr;
    SimplerRunFn run_fn_ = nullptr;
    SimplerUnregisterCallableFn unregister_callable_fn_ = nullptr;
    GetAicpuDlopenCountFn get_aicpu_dlopen_count_fn_ = nullptr;
    GetAicpuDlopenCountFn get_host_dlopen_count_fn_ = nullptr;
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
    CommResetDomainWindowsFn comm_reset_domain_windows_fn_ = nullptr;
    CommReleaseDomainWindowsFn comm_release_domain_windows_fn_ = nullptr;
    CommBarrierFn comm_barrier_fn_ = nullptr;
    CommDestroyFn comm_destroy_fn_ = nullptr;
    void *device_ctx_ = nullptr;
    std::vector<CommSession> comm_sessions_;
    std::unordered_map<uint64_t, size_t> comm_session_index_;
    uint64_t base_comm_handle_ = 0;

    std::vector<uint8_t> runtime_buf_;
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
