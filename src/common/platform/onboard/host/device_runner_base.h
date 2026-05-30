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
#include <thread>
#include <vector>

#include "device_arena.h"
#include "device_runner_helpers.h"
#include "host/load_aicpu_op.h"
#include "host/memory_allocator.h"

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
};

#endif  // SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H
