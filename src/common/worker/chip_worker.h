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
#include <vector>

#include "../task_interface/call_config.h"
#include "../task_interface/task_args.h"
#include "types.h"

class ChipWorker : public IWorker {
public:
    ChipWorker() = default;
    ~ChipWorker();

    ChipWorker(const ChipWorker &) = delete;
    ChipWorker &operator=(const ChipWorker &) = delete;

    /// Bind the runtime library, cache platform binaries, and attach the
    /// calling thread to `device_id`. Can only be called once per lifetime —
    /// the runtime and device cannot be changed after init.
    ///
    /// `log_level` (0=DEBUG..4=NUL) and `log_info_v` (0..9) are pushed into
    /// HostLogger + runner state and (onboard) into CANN dlog at this point;
    /// they reflect the user's `simpler` Python logger at Worker.init() time
    /// and are then fixed for this ChipWorker's lifetime.
    void init(
        const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
        const std::string &simpler_log_lib_path, int device_id, const std::string &sim_context_lib_path = "",
        int log_level = 1, int log_info_v = 5
    );

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // IWorker: dispatch the prepared cid by delegating to run_prepared.
    // The cid must already have been prepared via prepare_callable.
    void run(int32_t callable_id, TaskArgsView args, const CallConfig &config) override;

    // Per-callable_id preparation. Requires init() first and a callable_id
    // in [0, MAX_REGISTERED_CALLABLE_IDS) (cap 64).
    void prepare_callable(int32_t callable_id, const void *callable);
    void run_prepared(int32_t callable_id, TaskArgsView args, const CallConfig &config);
    void run_prepared(int32_t callable_id, const void *args, const CallConfig &config);
    void unregister_callable(int32_t callable_id);

    /// Number of distinct callable_ids the AICPU has been asked to dlopen for
    /// on the bound device. Returns 0 when not initialized or the runtime
    /// variant has no per-cid registration support. Used by tests to assert
    /// that prepare_callable + repeated run_prepared do not trigger redundant
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
    /// One active comm session per ChipWorker is supported.  Users needing
    /// multiple concurrent comms should instantiate multiple ChipWorkers.
    uint64_t comm_init(int rank, int nranks, const std::string &rootinfo_path);
    uint64_t comm_alloc_windows(uint64_t comm_handle, size_t win_size);
    uint64_t comm_get_local_window_base(uint64_t comm_handle);
    size_t comm_get_window_size(uint64_t comm_handle);
    void comm_barrier(uint64_t comm_handle);
    void comm_destroy(uint64_t comm_handle);

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
    using SimplerInitFn = int (*)(void *, int, int, int);
    using PrepareCallableFn =
        int (*)(void *, int32_t, const void *, int, const uint8_t *, size_t, const uint8_t *, size_t);
    using RunPreparedFn = int (*)(
        void *, void *, int32_t, const void *, int, int, int, const uint8_t *, size_t, const uint8_t *, size_t, int,
        int, int, int, const char *
    );
    using UnregisterCallableFn = int (*)(void *, int32_t);
    using GetAicpuDlopenCountFn = size_t (*)(void *);
    using FinalizeDeviceFn = int (*)(void *);
    using EnsureAclReadyFn = int (*)(void *, int);
    using CreateCommStreamFn = void *(*)(void *);
    using DestroyCommStreamFn = int (*)(void *, void *);
    using CommInitFn = void *(*)(int, int, void *, const char *);
    using CommAllocWindowsFn = int (*)(void *, size_t, uint64_t *);
    using CommGetLocalWindowBaseFn = int (*)(void *, uint64_t *);
    using CommGetWindowSizeFn = int (*)(void *, size_t *);
    using CommBarrierFn = int (*)(void *);
    using CommDestroyFn = int (*)(void *);

    void *lib_handle_ = nullptr;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    DeviceMallocCtxFn device_malloc_ctx_fn_ = nullptr;
    DeviceFreeCtxFn device_free_ctx_fn_ = nullptr;
    CopyToDeviceCtxFn copy_to_device_ctx_fn_ = nullptr;
    CopyFromDeviceCtxFn copy_from_device_ctx_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    SimplerInitFn simpler_init_fn_ = nullptr;
    PrepareCallableFn prepare_callable_fn_ = nullptr;
    RunPreparedFn run_prepared_fn_ = nullptr;
    UnregisterCallableFn unregister_callable_fn_ = nullptr;
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
    CommBarrierFn comm_barrier_fn_ = nullptr;
    CommDestroyFn comm_destroy_fn_ = nullptr;
    void *device_ctx_ = nullptr;
    // aclrtStream owned by the currently-active comm session (created inside
    // comm_init on onboard via DeviceRunner::create_comm_stream, paired with
    // destroy_comm_stream in comm_destroy).  Null when no comm is active or
    // when running on a backend without ACL (sim).
    void *comm_stream_ = nullptr;

    std::vector<uint8_t> runtime_buf_;
    std::vector<uint8_t> aicpu_binary_;
    std::vector<uint8_t> aicore_binary_;
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
