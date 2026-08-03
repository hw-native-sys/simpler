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
 * simpler host-runtime C API — canonical header
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
 *   - sizing:       get_runtime_size, get_runtime_alignment
 *   - device-mem:   device_malloc_ctx, device_free_ctx,
 *                   committed_device_memory_ctx,
 *                   copy_to_device_ctx, copy_from_device_ctx
 *   - prepared run: simpler_register_callable, simpler_prepare_run,
 *                   simpler_launch_run, simpler_poll_run, simpler_wait_run,
 *                   simpler_finalize_run, simpler_run,
 *                   simpler_unregister_callable,
 *                   get_aicpu_dlopen_count, get_host_dlopen_count,
 *                   get_run_stream_set_create_count,
 *                   simpler_provision_dma_workspace
 *   - pipeline:     get_pipeline_contract, select_pipeline_slot_ctx,
 *                   select_arena_bank_ctx,
 *                   supports_concurrent_native_prepare_ctx,
 *                   set_native_run_identity_ctx,
 *                   get_arena_bank_gm_heap_base_ctx,
 *                   get_retained_temp_addr_ctx,
 *                   set_task_accepted_state_ctx
 *   - ACL/stream:   ensure_acl_ready_ctx, create_comm_stream_ctx,
 *                   destroy_comm_stream_ctx
 *   - comm:         comm_init, comm_alloc_windows, comm_get_local_window_base,
 *                   comm_get_window_size, comm_barrier, comm_destroy
 *
 * Native-run storage: caller allocates at least get_runtime_size() bytes with
 * get_runtime_alignment() alignment, zero-initializes it before first use, and
 * keeps its address stable from prepare through finalize. After finalize the
 * same storage may be reused without re-zeroing. The storage must not be moved,
 * copied, or used for overlapping runs while it contains a prepared run. The
 * caller must serialize phase functions for a given context/storage pair;
 * poll/wait/finalize are not concurrent operations on the same run.
 * Error codes: 0 = success, negative = error.
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
    PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE = -3,
};

/** Return values from simpler_poll_run(). */
enum {
    SIMPLER_NATIVE_RUN_POLL_ERROR = -1,
    SIMPLER_NATIVE_RUN_POLL_NOT_READY = 0,
    SIMPLER_NATIVE_RUN_POLL_COMPLETE = 1,
};

enum {
    PTO_PIPELINE_CONTRACT_ABI_VERSION = 1,
    PTO_PIPELINE_MAX_RESOURCES = 8,
    /* Ceiling on pipeline_depth once a depth above 1 is enabled. */
    PTO_PIPELINE_MAX_DEPTH = 2,
};

/**
 * How a resource behaves across the KernelLaunch boundary, which is what
 * decides its copy count: HOST_PER_RUN and EXEC_HANDLE need one instance per
 * in-flight run (`pipeline_depth`), DEVICE_SCRATCH needs exactly one.
 */
typedef enum PipelineResourceClass {
    /* Carries this run's own content, so the device is still reading the
       previous run's content while the next run is prepared. */
    PTO_PIPELINE_HOST_PER_RUN = 0,
    /* Not rewritten per run: whoever populates it does so once, and device ops
       run one at a time, so a single instance is reused across runs. */
    PTO_PIPELINE_DEVICE_SCRATCH = 1,
    /* Execution context (stream) a run owns while its op runs and is reaped. */
    PTO_PIPELINE_EXEC_HANDLE = 2,
} PipelineResourceClass;

typedef enum PipelineResourceKind {
    /* Zero is not a resource: it is what an entry a runtime never filled in
       reads as, so a declaration that overstates resource_count is rejected
       instead of decaying into a valid-looking resource. */
    PTO_PIPELINE_KIND_UNSPECIFIED = 0,
    PTO_PIPELINE_GM_HEAP = 1,
    PTO_PIPELINE_GM_SM = 2,
    PTO_PIPELINE_RUNTIME_IMAGE = 3,
    PTO_PIPELINE_TASK_ARGS = 4,
    PTO_PIPELINE_AICPU_STREAM = 5,
    PTO_PIPELINE_AICORE_STREAM = 6,
} PipelineResourceKind;

typedef struct PipelineResource {
    uint32_t kind;
    uint32_t resource_class;
    /* Size of one copy. Reserved: currently declared as 0 and required to be 0. */
    uint64_t bytes_per_copy;
} PipelineResource;

/**
 * Runtime-owned declaration of resources that cross KernelLaunch.
 *
 * `pipeline_depth` is the only replication count: a resource needs
 * `pipeline_depth` copies unless its class is DEVICE_SCRATCH, which needs one.
 * Per-resource copy counts stay derivable from the class, so a runtime that
 * wants a cheap resource replicated and an expensive one shared says so by
 * classifying them, not by carrying a second global count.
 */
typedef struct PipelineContract {
    uint32_t abi_version;
    uint32_t resource_count;
    uint32_t pipeline_depth;
    PipelineResource resources[PTO_PIPELINE_MAX_RESOURCES];
} PipelineContract;

/**
 * Ownership token for one pipeline resource slot.
 *
 * `slot_id` selects the per-run/exec-handle copies. `generation` changes every
 * time that slot is leased, so delayed completion or cleanup from an older run
 * cannot mutate resources now owned by its successor. Generation zero is
 * reserved for an invalid/uninitialised lease.
 */
typedef struct PipelineSlotLease {
    uint32_t slot_id;
    uint32_t reserved;
    uint64_t generation;
} PipelineSlotLease;

/* Per-stage run timing is no longer returned. The platform emits it as
 * `[STRACE]` log markers (host stages + the AICPU device-phase breakdown,
 * gated on SIMPLER_HOST_STRACE) — parse with simpler_setup.tools.strace_timing.
 * See docs/dfx/host-trace.md. */

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/** Return this runtime's immutable pipeline resource declaration. */
const PipelineContract *get_pipeline_contract(void);

/**
 * Create a new device context (heap-allocated DeviceRunner).
 * Each ChipWorker should own one context for the lifetime of its init→finalize cycle.
 * @return Opaque handle on success, NULL on failure.
 */
DeviceContextHandle create_device_context(void);

/**
 * Destroy a device context created by create_device_context().
 * The caller must finalize every prepared native run and call
 * finalize_device() first. An active native run makes this operation log an
 * error and leave the context alive; otherwise it frees the underlying object.
 */
void destroy_device_context(DeviceContextHandle ctx);

/** Return the byte size of the opaque prepared-run storage. */
size_t get_runtime_size(void);

/** Return the required byte alignment of the opaque prepared-run storage. */
size_t get_runtime_alignment(void);

/** Allocate device memory in the given device context. */
void *device_malloc_ctx(DeviceContextHandle ctx, size_t size);

/** Free device memory previously allocated in the given device context. */
void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr);

/**
 * Total device HBM (bytes) currently committed by this device context's
 * MemoryAllocator (user tensors + pooled arenas + runtime buffers). Excludes
 * HCCL/VMM comm windows. Returns 0 on NULL ctx.
 */
size_t committed_device_memory_ctx(DeviceContextHandle ctx);

/** Copy host memory to a device pointer within the given device context. */
int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size);

/** Copy device memory to a host pointer within the given device context. */
int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size);

/**
 * One-shot platform-side init. Called once by ChipWorker::init() right
 * after dlopen, before any other entry. Three responsibilities, in order:
 *
 *   1. (Onboard only) Sync CANN dlog with
 *      HostLogger::get_instance().cann_level() via
 *      dlog_setlevel(-1, level, 0), unless ASCEND_GLOBAL_LOG_LEVEL was
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
 * Must be called before destroy_device_context() / dlclose(). Returns an error
 * without teardown while a prepared native run remains unfinalized.
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
 * resident until finalize regardless. Register and unregister mutate state
 * referenced by a prepared run, so both are rejected from successful prepare
 * until its matching finalize.
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
 * Callable-registry mutation is rejected while a native run is prepared or
 * executing on this context.
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
 * `config` carries aicpu_thread_num, the five diagnostic
 * enables + output_prefix, and the per-task ring sizing overrides
 * (`runtime_env.ring_task_window` / `.ring_heap` / `.ring_dep_pool`, each a
 * per-scope-depth-ring array of RUNTIME_ENV_RING_COUNT entries; 0 = unset,
 * precedence per ring: per-ring entry > PTO2_RING_* env var > compile-time
 * default). Ring overrides are consumed by tensormap_and_ringbuffer only; other
 * runtime variants accept and ignore them. Wire-compatible POD; prepare copies
 * it into the native-run state before returning.
 *
 * @return 0 on success, negative on error (no prep state, NULL ctx/config, etc.).
 */
int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
);

/**
 * Build and bind one run into caller-owned opaque storage without launching it.
 * A successful prepare must be paired with simpler_finalize_run(), even when
 * launch is abandoned or fails. The `args` container itself is consumed during
 * prepare and need not remain alive afterward, but every tensor backing buffer
 * referenced by it must remain valid through finalize (which may copy results
 * back to those addresses). See the storage size/alignment/lifetime contract at
 * the top of this header.
 */
int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
);

/**
 * Return nonzero when non-diagnostic preparation may overlap the execution
 * claim held by a run in another pipeline slot.
 */
int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx);

/**
 * Launch a prepared run. Returns only after the platform has published its
 * real kernel-launch marker, or after execution terminates before that marker.
 */
int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/**
 * Non-blocking completion query. Returns SIMPLER_NATIVE_RUN_POLL_NOT_READY,
 * SIMPLER_NATIVE_RUN_POLL_COMPLETE, or a negative validation/phase error. Call
 * only after simpler_launch_run() returns.
 */
int simpler_poll_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/** Wait for device execution to terminate. Does not release prepared resources. */
int simpler_wait_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/**
 * Wait if needed, validate/copy results, and release the opaque prepared run.
 * Also safely aborts a run that was prepared but never launched.
 */
int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/** Select the per-run/exec-handle slot used by the next synchronous run. */
int select_pipeline_slot_ctx(DeviceContextHandle ctx, uint32_t slot_id);

/** Select the HOST_PER_RUN arena bank used by the next synchronous run. */
int select_arena_bank_ctx(DeviceContextHandle ctx, uint32_t bank_id);

/** Attach optional L3 identity metadata to the current thread's next native prepare. */
int set_native_run_identity_ctx(
    DeviceContextHandle ctx, uint64_t run_id, uint64_t generation, uint64_t dispatch_id, uint64_t run_epoch
);

/**
 * Committed GM heap base of one arena bank, or 0 when that bank has never been
 * committed or the platform keeps a single shared arena set. Reports which
 * device allocation a bank actually owns; changes nothing.
 */
uint64_t get_arena_bank_gm_heap_base_ctx(DeviceContextHandle ctx, uint32_t bank_id);

/**
 * Retained temporary-buffer address held for one pipeline slot, or 0 while that
 * slot holds none. Reports which staging buffer a slot actually owns; changes
 * nothing.
 */
uint64_t get_retained_temp_addr_ctx(DeviceContextHandle ctx, uint32_t slot_id);

/**
 * Bind an optional host state word that the runner publishes after both device
 * kernels have been enqueued. Passing NULL clears the binding.
 */
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
 * Rejected while a native run is prepared or executing on this context.
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
 * Number of AICore run streams the runner bound to `ctx` has created. AICPU
 * streams belong to pipeline slots for the worker's lifetime; each run gets a
 * freshly created AICore stream, so this advances once per run. Returns 0 on
 * platforms whose runs use the persistent bootstrap pair.
 */
size_t get_run_stream_set_create_count(DeviceContextHandle ctx);

/** Number of persistent native execution threads created by this runner. */
size_t get_native_execution_thread_create_count(DeviceContextHandle ctx);

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
