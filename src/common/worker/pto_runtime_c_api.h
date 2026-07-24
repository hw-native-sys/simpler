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
 * Required API — resolved by ChipWorker via dlsym (every host_runtime.so must
 * export all of these; runtimes without a real backend ship not-supported
 * stubs rather than omitting symbols):
 *   - lifecycle:    create_device_context, destroy_device_context,
 *                   simpler_init, finalize_device
 *   - sizing:       get_runtime_size
 *   - device-mem:   device_malloc_ctx, device_free_ctx,
 *                   copy_to_device_ctx, copy_from_device_ctx
 *   - prepared run: simpler_register_callable, simpler_run, unregister_callable,
 *                   get_aicpu_dlopen_count, get_host_dlopen_count,
 *                   simpler_provision_dma_workspace
 *   - ACL/stream:   ensure_acl_ready_ctx, create_comm_stream_ctx,
 *                   destroy_comm_stream_ctx
 *   - comm:         comm_init, comm_alloc_windows, comm_get_local_window_base,
 *                   comm_get_window_size, comm_barrier, comm_destroy
 *
 * Optional metadata:
 *   - pipeline:     get_pipeline_contract
 *
 * Memory management: caller allocates a buffer of get_runtime_size() bytes
 * and passes it to simpler_run(). Error codes: 0 = success, negative = error.
 */

#ifndef SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
#define SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_

#include <stddef.h>
#include <stdint.h>

// simpler_run takes a pointer to the C++ CallConfig POD (task_interface/
// call_config.h). Forward-declared so this C-linkage header needn't pull the
// full C++ definition; both the ChipWorker consumer and the platform producers
// include call_config.h in their .cpp before calling / defining simpler_run.
#ifdef __cplusplus
struct CallConfig;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void *RuntimeHandle;
typedef void *DeviceContextHandle;

enum {
    PTO_RUNTIME_ERR_UNSUPPORTED = -2,
};

enum {
    PTO_PIPELINE_CONTRACT_ABI_VERSION = 1,
    PTO_PIPELINE_MAX_RESOURCES = 8,
    PTO_PIPELINE_MAX_SLOTS = 2,
};

typedef enum PipelineResourceClass {
    PTO_PIPELINE_FILL_MEM = 0,
    PTO_PIPELINE_REUSE_MEM = 1,
    PTO_PIPELINE_EXEC_HANDLE = 2,
} PipelineResourceClass;

typedef enum PipelineResourceKind {
    PTO_PIPELINE_GM_HEAP = 0,
    PTO_PIPELINE_GM_SM = 1,
    PTO_PIPELINE_RUNTIME_IMAGE = 2,
    PTO_PIPELINE_TASK_ARGS = 3,
    PTO_PIPELINE_AICPU_STREAM = 4,
    PTO_PIPELINE_AICORE_STREAM = 5,
} PipelineResourceKind;

typedef struct PipelineResource {
    uint32_t kind;
    uint32_t resource_class;
    size_t bytes_per_slot;
} PipelineResource;

/** Runtime-owned declaration of resources that cross KernelLaunch. */
typedef struct PipelineContract {
    uint32_t abi_version;
    uint32_t resource_count;
    uint32_t pipeline_slots;
    uint32_t arena_banks;
    PipelineResource resources[PTO_PIPELINE_MAX_RESOURCES];
} PipelineContract;

/* Per-stage run timing is no longer returned. The platform emits it as
 * `[STRACE]` log markers (host stages + the AICPU device-phase breakdown,
 * gated on SIMPLER_HOST_STRACE) — parse with simpler_setup.tools.strace_timing.
 * See docs/dfx/host-trace.md. */

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/** Return this runtime's immutable pipeline resource declaration. Optional. */
const PipelineContract *get_pipeline_contract(void);

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
 * after dlopen, before any other entry. Three responsibilities, in order:
 *
 *   1. (Onboard only) Sync CANN dlog with HostLogger::get_instance().level()
 *      via dlog_setlevel(-1, level, 0), unless ASCEND_GLOBAL_LOG_LEVEL was
 *      externally configured, in which case CANN keeps the user's choice.
 *      This must run before step 2 because CANN snapshots the device-side
 *      log session's level at context-open time (rtSetDevice); a later
 *      dlog_setlevel would not re-level the already-opened device session.
 *      The log level itself is owned by libsimpler_log.so (seeded earlier
 *      by simpler_log_init); it never travels through this ABI.
 *
 *   2. Attach the calling thread to `device_id` (rtSetDevice on onboard,
 *      pto_cpu_sim_bind_device + pto_cpu_sim_acquire_device on sim) and
 *      record the device id on the DeviceRunner so subsequent device-ops
 *      can re-attach their own caller threads idempotently.
 *
 *   3. Take ownership of the AICPU + AICore executor binaries (copied into
 *      DeviceRunner-owned vectors). All subsequent simpler_register_callable /
 *      simpler_run invocations reuse this resident pair — no binary bytes
 *      cross the C ABI on per-run paths.
 *
 *   4. When `prewarm_config` is non-null, build + upload + cache the prebuilt
 *      runtime-arena for its `runtime_env` ring sizing (tensormap_and_ringbuffer;
 *      a no-op for runtimes without a prebuilt arena). The device is up by this
 *      point, so the first simpler_run with matching sizing skips the (~800ms)
 *      cold build. The sizing is fork-constant, so it rides init rather than a
 *      separate call. Only `prewarm_config->runtime_env` is read.
 *
 * Returns 0 on success, negative on attach or prewarm-build failure.
 */
int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config
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
 * H2D + caching keyed by `callable_id`) from each `simpler_run` invocation,
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
 * Uploads child kernels into the DeviceRunner's func_id-keyed cache, copies
 * the orchestration SO bytes into a device-resident buffer keyed by the SO's
 * ELF Build-ID hash (so two callable_ids with identical SO share one buffer),
 * and prewarms device-orchestration callables by loading their AICPU-side SO
 * table entry before the first real task. Subsequent
 * `simpler_run(callable_id, ...)` calls reuse this state.
 *
 * `device_id` and the executor binaries are not threaded through this entry
 * — they were captured by `simpler_init` and live on the DeviceRunner.
 *
 * @return 0 on success, negative on error (NULL ctx, callable_id out of
 *         range, upload/copy failure, or AICPU prewarm failure).
 */
int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable);

/**
 * Launch a callable previously staged via `simpler_register_callable`.
 *
 * Looks up the prepared state by `callable_id`, restores the kernel func_id ↔
 * dev_addr table onto a fresh Runtime, and dispatches without re-uploading
 * kernels or re-copying the orch SO. The AICPU side dispatches via
 * `orch_so_table_[callable_id]` (see runtime.h::set_active_callable_id).
 * Successful TRB prepare has already populated that table; if a future
 * fallback leaves a callable prepared but not prewarmed, the first successful
 * run commits the AICPU seen state only after the device-side load succeeds.
 *
 * `device_id` and the executor binaries are not threaded through this entry
 * — they were captured by `simpler_init` and live on the DeviceRunner.
 *
 * Per-stage run timing is not returned — the platform emits it as `[STRACE]`
 * log markers (see docs/dfx/host-trace.md).
 *
 * `config` carries block_dim (0 = auto), aicpu_thread_num, the five diagnostic
 * enables + output_prefix, and the per-task ring sizing overrides
 * (`runtime_env.ring_task_window` / `.ring_heap` / `.ring_dep_pool`, each a
 * per-scope-depth-ring array of RUNTIME_ENV_RING_COUNT entries; 0 = unset,
 * precedence per ring: per-ring entry > PTO2_RING_* env var > compile-time
 * default). Ring overrides are consumed by tensormap_and_ringbuffer only; other
 * runtime variants accept and ignore them. Wire-compatible POD; the platform
 * reads it by pointer without copying.
 *
 * @return 0 on success, negative on error (no prep state, NULL ctx/config, etc.).
 */
int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
);

/** Bind a host mailbox state word for launch acceptance publication. Optional. */
int set_task_accepted_state_ctx(DeviceContextHandle ctx, volatile int32_t *state, int32_t accepted_value);

/**
 * Drop the prepared state for `callable_id` and release the per-id share of
 * the device orch SO buffer. The buffer itself is freed only when its
 * hash-keyed refcount drops to zero (different callable_ids with identical
 * SO share one allocation).
 *
 * Kernel binaries uploaded by `simpler_register_callable` remain resident — they are
 * shared across callables by func_id and only released by `finalize_device`.
 *
 * AICPU-side dlopen state in `orch_so_table_[callable_id]` is NOT released by
 * this call. It is reclaimed lazily when the cid is reused (the next
 * `launch_device_register` triggers `dlclose` + reload), or at process
 * exit. Long-running processes that register / unregister cids without ever
 * reusing them will hold the AICPU SO handle until shutdown.
 *
 * @return 0 on success or if callable_id was not registered, negative on error.
 */
int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id);

/**
 * Number of distinct callable_ids the AICPU has been asked to dlopen for on
 * the device bound to `ctx`. Returns 0 on runtime variants without per-cid
 * registration support. Used by tests to assert that `simpler_register_callable` +
 * repeated `simpler_run` calls do not trigger redundant AICPU dlopens.
 */
size_t get_aicpu_dlopen_count(DeviceContextHandle ctx);

/**
 * Number of host-side dlopens triggered by `simpler_register_callable` on the host
 * orchestration variants (host_build_graph). Mirrors `get_aicpu_dlopen_count`
 * for the trb path. Returns 0 on runtime variants whose orchestration runs on
 * the device.
 */
size_t get_host_dlopen_count(DeviceContextHandle ctx);

/**
 * Provision the async-DMA workspaces named in `required_mask` (a bitmask of
 * DmaWorkspaceKind bits) once at Worker init, latching their device addresses
 * into the resident KernelArgs so every subsequent run carries them. Called only
 * for a Worker created with SDMA enabled. Bits unsupported by this
 * platform/runtime are rejected, so a Worker opting into SDMA on sim / a5 / hbg
 * fails fast. Returns 0 on success, negative on unsupported/failed provisioning.
 */
int simpler_provision_dma_workspace(DeviceContextHandle ctx, uint32_t required_mask);

#ifdef __cplusplus
}
#endif

#endif  // SRC_COMMON_WORKER_PTO_RUNTIME_C_API_H_
