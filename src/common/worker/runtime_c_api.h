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
 *                   committed_device_memory_ctx, device_memory_info_ctx,
 *                   copy_to_device_ctx, copy_from_device_ctx
 *   - prepared run: simpler_register_callable, simpler_prepare_run,
 *                   simpler_launch_run, simpler_poll_run, simpler_wait_run,
 *                   simpler_finalize_run, simpler_run,
 *                   simpler_unregister_callable,
 *                   get_aicpu_dlopen_count, get_host_dlopen_count,
 *                   get_run_stream_set_create_count
 *   - kernel probe: simpler_kernel_mode_supported
 *   - pipeline:     get_pipeline_contract,
 *                   supports_concurrent_native_prepare_ctx,
 *                   get_arena_bank_gm_heap_base_ctx,
 *                   get_retained_temp_addr_ctx
 *   - ACL/stream:   ensure_acl_ready_ctx, create_comm_stream_ctx,
 *                   destroy_comm_stream_ctx
 *   - comm:         comm_init, comm_alloc_windows, comm_get_local_window_base,
 *                   comm_get_window_size, comm_barrier, comm_destroy
 *
 * Capability-dependent API — every host_runtime.so exports
 * simpler_kernel_mode_init, simpler_kernel_mode_prepare_callable, and
 * simpler_kernel_mode_launch. ChipWorker resolves and requires this group
 * only when simpler_kernel_mode_supported returns nonzero.
 *
 * Native-run storage: caller allocates at least get_runtime_size() bytes with
 * get_runtime_alignment() alignment, zero-initializes it before first use, and
 * keeps its address stable from prepare through finalize. After finalize the
 * same storage may be reused without re-zeroing. The storage must not be moved,
 * copied, or used for overlapping runs while it contains a prepared run. The
 * caller must serialize phase functions for a given context/storage pair;
 * poll/wait/finalize are not concurrent operations on the same run.
 * Error codes: 0 = success, negative = error — see the two disjoint negative
 * bands documented on the PTO_RUNTIME_ERR_* enum below.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "device_memory_info.h"

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

/**
 * Host-side status codes, and the band they are confined to.
 *
 * A negative return from any entry point below names one of two mechanisms, and
 * a caller must be able to tell which from the number alone:
 *
 *   - a **device-latched** runtime code, reported as its own value negated.
 *     Orchestration codes occupy 1..99 and scheduler codes 100 upwards, so the
 *     latched band is -1..-PTO_RUNTIME_LATCHED_CODE_MAX (the codes themselves
 *     are SIMPLER_ERROR_* in src/{arch}/runtime/{runtime}/common/runtime_status.h;
 *     the per-code triage tables are in
 *     docs/troubleshooting/device-error-codes.md).
 *   - a **host-side** code from this enum, at or below PTO_RUNTIME_ERR_BASE.
 *
 * The two bands are disjoint, and a static_assert in each runtime's
 * runtime_maker.cpp holds them apart. So a host-side failure never reports -1
 * (which is SCOPE_DEADLOCK) or -3 (FLOW_CONTROL_DEADLOCK): it reports
 * PTO_RUNTIME_ERR_INTERNAL, or a more specific code from this band.
 */
enum {
    /* Ceiling on a device-latched code, and therefore the floor of the latched
       band once negated. */
    PTO_RUNTIME_LATCHED_CODE_MAX = 999,
    /* Highest host-side code; the band descends from here. */
    PTO_RUNTIME_ERR_BASE = -(PTO_RUNTIME_LATCHED_CODE_MAX + 1),
    /* A host-side failure whose diagnosis is the log line that preceded it. */
    PTO_RUNTIME_ERR_INTERNAL = PTO_RUNTIME_ERR_BASE,
    /* The request names a capability this platform/runtime does not implement. */
    PTO_RUNTIME_ERR_UNSUPPORTED = PTO_RUNTIME_ERR_BASE - 1,
    PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE = PTO_RUNTIME_ERR_BASE - 2,
    /* The call is structurally valid but conflicts with the context's
       execution mode or lifecycle (e.g. launch before kernel initialization). */
    PTO_RUNTIME_ERR_INVALID_STATE = PTO_RUNTIME_ERR_BASE - 3,
    /* A caller supplied an invalid pointer, id, size, or other argument. */
    PTO_RUNTIME_ERR_INVALID_ARGUMENT = PTO_RUNTIME_ERR_BASE - 4,
};

/** Return values from simpler_poll_run(). */
enum {
    SIMPLER_NATIVE_RUN_POLL_ERROR = PTO_RUNTIME_ERR_INTERNAL,
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
    /* Program: reserved, must be 0. Kernel: arena kinds declare nonzero
       required usable bytes per copy, not committed HBM or capacity budgets.
       Streams and TASK_ARGS must be 0; task-argument byte limits are not
       represented by this field. */
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

/**
 * Immutable resource and trace identity copied into one prepared run.
 * `pipeline_slot` and `arena_bank` must be smaller than
 * PTO_PIPELINE_MAX_DEPTH; they remain explicit because some runtimes map a
 * leased slot to a different arena bank. `run_epoch` is the process-unique
 * identity used by later phase calls; lease generation, run id, and dispatch
 * id remain diagnostic after admission. A non-null acceptance sink is written
 * only at the real kernel-launch marker.
 */
typedef struct NativeRunDescriptor {
    uint32_t pipeline_slot;
    uint32_t arena_bank;
    uint64_t run_id;
    uint64_t generation;
    uint64_t dispatch_id;
    uint64_t run_epoch;
    volatile int32_t *accepted_state;
    int32_t accepted_value;
} NativeRunDescriptor;

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
 * MemoryAllocator (user tensors + pooled arenas + Graph execution blocks +
 * runtime buffers). Excludes HCCL/VMM comm windows. Returns 0 on NULL ctx.
 */
size_t committed_device_memory_ctx(DeviceContextHandle ctx);

/**
 * Query the target device's ACL_HBM_MEM free/total byte snapshot. The caller
 * owns `info`; valid output is published only when the function returns 0.
 * Unsupported backends return PTO_RUNTIME_ERR_UNSUPPORTED.
 */
int device_memory_info_ctx(DeviceContextHandle ctx, DeviceMemoryInfo *info);

/** Copy host memory to a device pointer within the given device context. */
int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size);

/** Copy device memory to a host pointer within the given device context. */
int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size);

/**
 * One-shot platform-side init. Called once by ChipWorker::init() after the
 * runtime module's private HostLogger has been bound to process-owned state.
 * Its responsibilities, in order:
 *
 *   1. (Onboard only) Sync CANN dlog with
 *      HostLogger::get_instance().cann_level() via
 *      dlog_setlevel(-1, level, 0), unless ASCEND_GLOBAL_LOG_LEVEL was
 *      externally configured, in which case CANN keeps the user's choice.
 *      This must run before step 2 because CANN snapshots the device-side
 *      log session's level at context-open time (rtSetDevice); a later
 *      dlog_setlevel would not re-level the already-opened device session.
 *      The threshold is read from the state bound by ChipWorker immediately
 *      after dlopen.
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
 *   4. Provision this device's async-DMA workspaces and latch their device
 *      addresses into the resident AICPU globals, so every subsequent run
 *      carries them (the scheduler prefills each core's GlobalContext from
 *      there, and kernels read them through get_dma_workspace(args, kind); a
 *      kind this device did not provision reads back 0). The addresses are
 *      published by the same one-shot AICPU init launch that step 2's device
 *      bring-up performs, so provisioning is ordered before it rather than
 *      re-latching afterwards.
 *
 *      Every engine the platform supports is provisioned except SDMA, which is
 *      provisioned only when `enable_sdma` is non-zero. SDMA is the one engine
 *      a caller can decline, and the parameter is a flag rather than an engine
 *      set for that reason: its workspace cannot be obtained without also
 *      creating 48 CP-process STARS streams, and a Worker holding those gets a
 *      single device-reset attempt after an AICore fault instead of three, so
 *      defaulting it on would put every Worker in that population.
 *      `enable_sdma` on a platform/runtime without SDMA support fails init, so
 *      a Worker opting in on sim / a5 / hbg fails fast rather than reaching its
 *      first run reading a zero workspace address.
 *
 *      `sdma_warmup_binary` / `sdma_warmup_size` carry the vector-only ELF that
 *      walks the SDMA control path once per channel against the workspace just
 *      provisioned, so the first TPREFETCH_ASYNC does not pay that cold start.
 *      A null/empty buffer, or a warmup the platform cannot run, still
 *      initializes successfully — the only cost is first-call latency. A warmup
 *      whose device launch or sync fails does fail init, because that card is
 *      then poisoned.
 *
 *   5. When `prewarm_config` is non-null, build + upload + cache the prebuilt
 *      runtime-arena for its `runtime_env` ring sizing (tensormap_and_ringbuffer;
 *      a no-op for runtimes without a prebuilt arena). The device is up by this
 *      point, so the first simpler_run with matching sizing skips the (~800ms)
 *      cold build. The sizing is fork-constant, so it rides init rather than a
 *      separate call. Only `prewarm_config->runtime_env` is read.
 *
 * Returns 0 on success, negative on attach, provisioning, or prewarm-build
 * failure, and PTO_RUNTIME_ERR_INVALID_STATE when the context is already
 * latched to kernel mode.
 */
int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config, int enable_sdma, const void *sdma_warmup_binary, uint64_t sdma_warmup_size
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
 * precedence per ring: per-ring entry > compile-time
 * default). Ring overrides are consumed by tensormap_and_ringbuffer only; other
 * runtime variants accept and ignore them. Wire-compatible POD; prepare copies
 * it into the native-run context before returning.
 *
 * `descriptor` carries this run's immutable resource selection, trace identity,
 * and optional launch-acceptance sink. It is copied before prepare returns. The
 * platform release-stores the acceptance value only after the real
 * kernel-launch marker; the sink is not retained after this blocking call.
 *
 * @return 0 on success, negative on error (no prep state, NULL ctx/config, etc.).
 */
int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
);

/**
 * Build and bind one run into caller-owned opaque storage without launching it.
 * A successful prepare must be paired with simpler_finalize_run(), even when
 * launch is abandoned or fails. The `args` container itself is consumed during
 * prepare and need not remain alive afterward, but every tensor backing buffer
 * referenced by it must remain valid through finalize (which may copy results
 * back to those addresses). See the storage size/alignment/lifetime contract at
 * the top of this header. `descriptor` is required and copied before this
 * function returns. A non-null acceptance sink in the descriptor must remain
 * valid until launch returns or the prepared run is finalized without launch.
 */
int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
);

/**
 * Return nonzero when non-diagnostic preparation may overlap the execution
 * claim held by a run in another pipeline slot.
 */
int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx);

/**
 * Launch a prepared run. Returns only after the platform has published its
 * real kernel-launch marker, or after execution terminates before that marker.
 * The acceptance sink captured by prepare is published at that marker; an
 * execution failure before the marker leaves it unchanged.
 */
int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/**
 * Non-blocking device-completion query. Returns
 * SIMPLER_NATIVE_RUN_POLL_NOT_READY, SIMPLER_NATIVE_RUN_POLL_COMPLETE, or a
 * negative validation, phase, or device-query error. COMPLETE is published
 * only after the executor has also finished host-side drain work. Polling
 * never releases the prepared run's resources; call only after
 * simpler_launch_run() returns.
 */
int simpler_poll_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/** Wait for device execution to terminate. Does not release prepared resources. */
int simpler_wait_run(DeviceContextHandle ctx, RuntimeHandle runtime);

/**
 * Wait if needed, validate/copy results, and release the opaque prepared run.
 * Also safely aborts a run that was prepared but never launched.
 */
int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime);

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
 * Number of AICore run streams the runner bound to `ctx` has created. One
 * AICPU + AICore pair serves every run for the runner's lifetime. The AICPU
 * stream persists; the AICore stream is recreated when a new code upload makes
 * it stale, and destroyed when an unproven completion retires it, so this
 * advances per publication or unproven retirement rather than once per run or
 * per pipeline slot. Returns 0 on platforms whose runs use the persistent
 * bootstrap pair.
 */
size_t get_run_stream_set_create_count(DeviceContextHandle ctx);

/* ===========================================================================
 * Kernel-mode lifecycle (four entries + finalize_device)
 *
 * A context has one of two execution modes for its whole lifetime, decided by
 * which init entry runs first. simpler_init latches program mode — the
 * exclusive-device path driven through the prepared-run family
 * above. simpler_kernel_mode_init latches kernel mode, which borrows the
 * caller's already-current device and caller-owned stream to enqueue one
 * bounded asynchronous operator per launch: no device reset, no internal
 * stream/device synchronize on the prepare/launch/close paths, zero
 * allocation at launch, and no capture/model-state queries, so a launch is
 * capturable by ACLGraph as an ordinary node.
 *
 * Identity is a write-once property of the context, not a state that evolves:
 * ExecutionModeLatch on the platform runner holds it, the first init entry to
 * run latches it, and it never changes — not on finalize, not on error. There
 * is no separate declaration call to forget, and no unlatch, so a handle from
 * a failed kernel init can never be recycled into a program context.
 * simpler_init latches PROGRAM before touching any process or runner state, so
 * the mutual exclusion is enforced on every program init. Every kernel-mode
 * guard on the device/ACL lifecycle and arena paths keys on the latch reading
 * kernel. A call that conflicts with the context's execution mode returns
 * PTO_RUNTIME_ERR_INVALID_STATE.
 *
 * Kernel-mode capacity is a mode invariant, not a gated state: `config` is
 * context-static, so each pooled arena region is committed at most once and
 * never grown or released afterwards. The platform arena reports a growth or
 * release request under kernel mode as a capacity invariant break
 * (PTO_RUNTIME_ERR_INTERNAL), and capacity intent travels in
 * CallConfig.runtime_env like everywhere else.
 *
 * Every host_runtime.so exports all four entries below. Consumers require
 * supported for capability discovery and require the remaining three only
 * when supported returns nonzero. Variants without kernel-mode support
 * export stubs that run the same structural validation and then refuse —
 * supported returns 0, init reports PTO_RUNTIME_ERR_UNSUPPORTED, and
 * prepare_callable and launch report PTO_RUNTIME_ERR_INVALID_STATE because
 * no kernel context is live. The fifth lifecycle entry is the existing
 * finalize_device(): in kernel mode it releases only context-owned resources
 * and never resets the device or finalizes ACL. device_id_ records which
 * device a context is on rather than a claim on it, so a kernel context
 * reaches that teardown path; the platform finalize() skips the per-thread
 * device bind on a kernel latch, because the caller already holds its own
 * device current.
 *
 * Structural argument errors return PTO_RUNTIME_ERR_INVALID_ARGUMENT before
 * capability or lifecycle checks. These entries accept only POD structs,
 * serialized blobs, and device/stream pointers — never framework objects.
 * The caller stream is always an explicit parameter and is never stored
 * beyond the call or destroyed by simpler.
 * =========================================================================== */

/**
 * Return nonzero when this runtime/context can execute kernel-mode launches.
 *
 * Callable on a context that create_device_context() has returned but that no
 * init has touched: this is the probe a loader asks before deciding whether the
 * remaining kernel-mode entries are needed at all, so an implementation must
 * answer from the runtime build and the device the context names, never from
 * init state. It is the one kernel-mode entry every runtime must export.
 */
int simpler_kernel_mode_supported(DeviceContextHandle ctx);

/**
 * Initialize a kernel-mode context on the caller's already-current device.
 *
 * A successful call latches kernel mode on the context — mutually exclusive
 * with the program-mode simpler_init — and the latch controls the
 * kernel-mode guards on the platform's device/ACL lifecycle and arena paths.
 *
 * Takes no device ownership: no device reset, no ACL init/finalize, and no
 * stream or device synchronize on this path. Creates only context-owned
 * persistent handles used by asynchronous preparation and launch. `config`
 * is context-static; launches never mutate it. `context_generation` is a
 * nonzero host-process-unique identity minted by the caller for sequential
 * contexts; generation zero is invalid.
 */
int simpler_kernel_mode_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *config, uint64_t context_generation
);

/**
 * Stage one callable for kernel-mode launches, outside ACLGraph capture.
 *
 * `callable` points to a canonical ChipCallable image of exactly
 * `callable_size` bytes. Validating every flexible-array offset before the
 * image is hashed or uploaded is the implementation's obligation; the shared
 * entry validation checks only the image's alignment, its size floor, and the
 * callable id range. Preparation may allocate persistent state and
 * enqueue asynchronous device work on `caller_stream`, but never synchronizes
 * a stream or device — preparation errors surface through the caller's own
 * warmup + synchronize. The stream is borrowed for this call only.
 */
int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, int32_t callable_id, const void *callable, size_t callable_size, void *caller_stream
);

/**
 * Enqueue one bounded asynchronous kernel-mode operator invocation.
 *
 * `args` points to a ChipStorageTaskArgs POD whose tensor addresses are
 * caller-owned device addresses; they are passed through without ever being
 * dereferenced on the host. A launch performs no device malloc/free, no
 * tensor staging, no synchronize, no capture-state query, and no
 * stream-to-model attachment: KernelLaunchOps is the vocabulary that makes
 * those unrepresentable, and routing every runtime call through it is the
 * implementation's obligation. A return of 0 means the sequence was enqueued;
 * device execution may still be in flight and may still fail asynchronously.
 */
int simpler_kernel_mode_launch(DeviceContextHandle ctx, int32_t callable_id, const void *args, void *caller_stream);

#ifdef __cplusplus
}
#endif
