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
 * PTO Runtime C API — canonical header
 *
 * Declares all C-linkage functions exported by the host runtime .so.
 * Both the ChipWorker (consumer, resolves public symbols via dlsym) and the
 * platform implementations (producers, define all symbols) include this file.
 *
 * Public API — resolved by ChipWorker via dlsym (every host_runtime.so must
 * export ALL of these; runtimes without a real backend ship not-supported
 * stubs rather than omitting symbols, so ChipWorker can dlsym the full set
 * unconditionally without per-symbol probing):
 *   - lifecycle:    create_device_context, destroy_device_context,
 *                   simpler_init, finalize_device
 *   - sizing:       get_runtime_size
 *   - device-mem:   device_malloc_ctx, device_free_ctx,
 *                   copy_to_device_ctx, copy_from_device_ctx
 *   - prepared run: prepare_callable, run_prepared, unregister_callable,
 *                   get_aicpu_dlopen_count, get_host_dlopen_count
 *   - ACL/stream:   ensure_acl_ready_ctx, create_comm_stream_ctx,
 *                   destroy_comm_stream_ctx
 *   - comm:         comm_init, comm_alloc_windows, comm_get_local_window_base,
 *                   comm_get_window_size, comm_barrier, comm_destroy
 *
 * Memory management: caller allocates a buffer of get_runtime_size() bytes
 * and passes it to run_prepared(). Error codes: 0 = success, negative = error.
 */

#ifndef SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
#define SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *RuntimeHandle;
typedef void *DeviceContextHandle;

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/**
 * Create a new device context (heap-allocated DeviceRunner).
 * Each ChipWorker should own one context for the lifetime of its init→finalize cycle.
 * @return Opaque handle on success, NULL on failure.
 */
DeviceContextHandle create_device_context(void);

/**
 * Destroy a device context created by create_device_context().
 * Calls finalize internally, then frees the underlying object.
 */
void destroy_device_context(DeviceContextHandle ctx);

/** Return sizeof(Runtime) for caller buffer allocation. */
size_t get_runtime_size(void);

/** Allocate device memory in the given device context. */
void *device_malloc_ctx(DeviceContextHandle ctx, size_t size);

/** Free device memory previously allocated in the given device context. */
void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr);

/** Copy host memory to a device pointer within the given device context. */
int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size);

/** Copy device memory to a host pointer within the given device context. */
int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size);

/**
 * One-shot platform-side init. Called once by ChipWorker::init() right
 * after dlopen, before any other entry. Three responsibilities:
 *
 *   1. Attach the calling thread to `device_id` (rtSetDevice on onboard,
 *      pto_cpu_sim_bind_device + pto_cpu_sim_acquire_device on sim) and
 *      record the device id on the DeviceRunner so subsequent device-ops
 *      can re-attach their own caller threads idempotently. Also creates
 *      the per-runner streams so binary load can proceed in the same call.
 *
 *   2. Copy the AICPU/AICore executor blobs into the DeviceRunner
 *      (one-shot H2D + AicpuSoInfo init). The blobs are raw byte buffers
 *      already loaded into host memory by the caller (typically the
 *      ChipWorker reading `aicpu.so` / `aicore_kernel.bin` from disk). The
 *      caller may release its own buffer immediately after `simpler_init`
 *      returns; every subsequent `run_prepared` reuses the runner-owned
 *      copies.
 *
 *   3. Push the user's chosen severity + INFO verbosity into HostLogger
 *      and into runner state (which run_prepared later forwards to AICPU
 *      via KernelArgs).
 *
 * On onboard, also calls dlog_setlevel(-1, log_level, 0) so CANN's runtime
 * filter matches the simpler logger — unless ASCEND_GLOBAL_LOG_LEVEL was
 * externally configured, in which case CANN keeps the user's explicit choice.
 *
 * On sim, no CANN dlog state to touch — only HostLogger + runner.
 *
 * `log_level` is CANN-aligned: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NUL.
 * `log_info_v` ∈ [0, 9]; only meaningful when severity is INFO.
 *
 * Returns 0 on success, negative on attach / stream / binary-load failure.
 */
int simpler_init(
    DeviceContextHandle ctx, int device_id, int log_level, int log_info_v, const uint8_t *aicpu_blob,
    size_t aicpu_blob_size, const uint8_t *aicore_blob, size_t aicore_blob_size
);

/**
 * Release all device resources held by the context.
 * Must be called before destroy_device_context() / dlclose().
 */
int finalize_device(DeviceContextHandle ctx);

/* ===========================================================================
 * Per-callable_id preparation
 *
 * The triplet below decouples the one-shot prep work (kernel upload + orch SO
 * H2D + caching keyed by `callable_id`) from each `run_prepared` invocation,
 * so the per-run cost shrinks to "rebuild Runtime args + launch". Callers
 * keep a stable small-int `callable_id` per ChipCallable; the platform side
 * caches the prepared state in a fixed-size table (cap 64, see
 * MAX_REGISTERED_CALLABLE_IDS in the AICPU executor) and rejects ids outside
 * `[0, 64)`. Lifetime: caller must `unregister_callable` before
 * `finalize_device` to release the device-side orch SO buffer; kernels stay
 * resident until finalize regardless.
 * =========================================================================== */

/**
 * Stage a callable for repeated cheap launches under the given `callable_id`.
 *
 * Uploads child kernels into the DeviceRunner's func_id-keyed cache and
 * copies the orchestration SO bytes into a device-resident buffer keyed by
 * the SO's ELF Build-ID hash (so two callable_ids with identical SO share
 * one buffer). Subsequent `run_prepared(callable_id, ...)` calls reuse this
 * state. The device and executor binaries were bound at `simpler_init`; no
 * per-call binary or device_id is needed.
 *
 * @return 0 on success, negative on error (NULL ctx, callable_id out of
 *         range, or upload/copy failure).
 */
int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable);

/**
 * Launch a callable previously staged via `prepare_callable`.
 *
 * Looks up the prepared state by `callable_id`, restores the kernel func_id ↔
 * dev_addr table onto a fresh Runtime, and dispatches without re-uploading
 * kernels or re-copying the orch SO. The AICPU side dispatches via
 * `orch_so_table_[callable_id]` (see runtime.h::set_active_callable_id). The
 * first run for a given callable_id sets `register_new_callable_id_` so the
 * AICPU does its one-time dlopen. The device and executor binaries were
 * bound at `simpler_init`; this entry only carries per-task state.
 *
 * @return 0 on success, negative on error (no prep state, NULL ctx, etc.).
 */
int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, const char *output_prefix
);

/**
 * Drop the prepared state for `callable_id` and release the per-id share of
 * the device orch SO buffer. The buffer itself is freed only when its
 * hash-keyed refcount drops to zero (different callable_ids with identical
 * SO share one allocation).
 *
 * Kernel binaries uploaded by `prepare_callable` remain resident — they are
 * shared across callables by func_id and only released by `finalize_device`.
 *
 * AICPU-side dlopen state in `orch_so_table_[callable_id]` is NOT released by
 * this call. It is reclaimed lazily when the cid is reused (the next
 * `register_new_callable_id()` triggers `dlclose` + reload), or at process
 * exit. Long-running processes that register / unregister cids without ever
 * reusing them will hold the AICPU SO handle until shutdown.
 *
 * @return 0 on success or if callable_id was not registered, negative on error.
 */
int unregister_callable(DeviceContextHandle ctx, int32_t callable_id);

/**
 * Number of distinct callable_ids the AICPU has been asked to dlopen for on
 * the device bound to `ctx`. Returns 0 on runtime variants without per-cid
 * registration support. Used by tests to assert that `prepare_callable` +
 * repeated `run_prepared` calls do not trigger redundant AICPU dlopens.
 */
size_t get_aicpu_dlopen_count(DeviceContextHandle ctx);

/**
 * Number of host-side dlopens triggered by `prepare_callable` on the host
 * orchestration variants (host_build_graph). Mirrors `get_aicpu_dlopen_count`
 * for the trb path. Returns 0 on runtime variants whose orchestration runs on
 * the device.
 */
size_t get_host_dlopen_count(DeviceContextHandle ctx);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
