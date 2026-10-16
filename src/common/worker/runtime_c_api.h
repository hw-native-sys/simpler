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
 *                   simpler_launch_run, simpler_launch_run_joined,
 *                   simpler_poll_run, simpler_wait_run,
 *                   simpler_finalize_run, simpler_run, simpler_launch_run_on_stream,
 *                   simpler_validate_caller_stream, simpler_validate_execution_context,
 *                   simpler_unregister_callable,
 *                   get_aicpu_dlopen_count, get_host_dlopen_count,
 *                   get_run_stream_set_create_count
 *   - kernel mode:  simpler_kernel_mode_supported, simpler_kernel_mode_init,
 *                   simpler_kernel_mode_prepare_callable,
 *                   simpler_kernel_mode_launch
 *   - pipeline:     get_pipeline_contract,
 *                   supports_concurrent_native_prepare_ctx,
 *                   supports_joined_native_launch_ctx,
 *                   supports_joined_diagnostic_launch_ctx,
 *                   get_arena_bank_gm_heap_base_ctx,
 *                   get_retained_temp_addr_ctx
 *   - ACL/stream:   ensure_acl_ready_ctx, create_comm_stream_ctx,
 *                   destroy_comm_stream_ctx
 *   - comm:         comm_init, comm_alloc_windows, comm_get_local_window_base,
 *                   comm_get_window_size, comm_barrier, comm_destroy
 *
 * Optional API — resolved opportunistically; a runtime built before these
 * entries simply lacks the capability, and ChipWorker raises an explicit error
 * when a caller requests it:
 *   - registration: simpler_register_callable_v2
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
/** Opaque simpler runner handle, not an ACL/RTS current-context handle. */
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
    /* BASE - 5 is reserved for diagnostic collector capacity refusal. */
    /* A prepare could not take the device resources it needs now, and a
       consumer that still holds some of them has a retirement this context will
       reach. The plan is kept and retried once that consumer finishes, so this
       is a wait rather than a failure and the caller must not discard the run. */
    SIMPLER_RUNTIME_ERR_RESOURCE_WAIT = PTO_RUNTIME_ERR_BASE - 6,
    /* A prepare could not take the device resources it needs and no consumer
       remains whose retirement could supply them: either none ever held any, or
       this plan has already waited for every one that did. Terminal. */
    SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE = PTO_RUNTIME_ERR_BASE - 7,
};

/**
 * A launch result that is not a failure of the run.
 *
 * Every diagnostic output slot or terminal bank some enabled owner needs is
 * still held by a run ahead. Nothing was submitted and nothing was admitted:
 * the run keeps its prepared state and may be launched again, unchanged, once
 * `simpler_diagnostic_credit_generation_ctx` advances. A caller that treats it
 * as a failure turns a run that is merely waiting into one that is lost.
 *
 * It takes a value in the host-side band so it cannot collide with a latched
 * device code, and sits in its own enum because the band's other members are
 * externally-contracted legacy names this one does not join.
 */
enum {
    SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL = PTO_RUNTIME_ERR_BASE - 5,
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
};

/**
 * How a resource behaves across the KernelLaunch boundary, which is what
 * decides its copy count: HOST_PER_RUN needs one instance per in-flight run
 * (`pipeline_depth`); DEVICE_SCRATCH and EXEC_HANDLE need exactly one.
 */
typedef enum PipelineResourceClass {
    /* Carries this run's own content, so the device is still reading the
       previous run's content while the next run is prepared. */
    PTO_PIPELINE_HOST_PER_RUN = 0,
    /* Not rewritten per run: whoever populates it does so once, and device ops
       run one at a time, so a single instance is reused across runs. */
    PTO_PIPELINE_DEVICE_SCRATCH = 1,
    /* Execution context (stream) a run owns while its op runs and is reaped.
       One instance serves every run: a role is declared once and the platform
       holds that stream for the runner's lifetime. */
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
    /* Program: must be 0. Kernel: DEVICE_SCRATCH states nonzero required
       usable bytes per copy; HOST_PER_RUN may state 0 when host orchestration
       determines its size at bind time. EXEC_HANDLE states 0.
       The number is a per-copy requirement for the resource named by `kind`:
       it is neither committed HBM nor a capacity budget, and the declared
       resources are not a complete manifest of what a context commits. */
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
 * Where one dispatch came from in the submitting Worker's tree.
 *
 * Carried beside a run's resource identity so that a diagnostic artifact can be
 * attributed to the submission that produced it. **An all-zero
 * `owner_instance_id` means the whole structure is unstated**, which is what a
 * caller reaching the native lane without a Worker mailbox leaves behind: the
 * remaining fields are then not read, and no consumer may treat their zeros as
 * an identity. A stated nonce is a full-width random draw, so all-zero cannot
 * collide with a real one.
 */
typedef struct NativeRunProvenance {
    /** The submitting Worker's incarnation nonce; all-zero ⇔ nothing stated. */
    uint8_t owner_instance_id[8];
    /** Index of the chip child serving the dispatch. */
    uint32_t chip_rank;
    /** That child's own count of diagnostic dispatches before this one. */
    uint32_t capture_index;
    /** Parent task slot, and this member's place in a grouped dispatch. */
    uint32_t task_slot;
    uint32_t group_index;
    uint32_t group_size;
    uint32_t reserved;
} NativeRunProvenance;

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
 * One run's whole published identity, by value.
 *
 * Assembled from a prepared run's descriptor and owned by every stage that
 * keeps it, so no consumer dereferences the submitter's storage. Deliberately
 * excludes the descriptor's acceptance sink and its value — a caller pointer
 * and its sentinel, which must never reach output metadata — and excludes
 * `run_epoch`, which a retaining collector already holds and publishes from its
 * own bucket. An all-zero `owner_instance_id` carries the same meaning it has
 * on `NativeRunProvenance`: nothing was stated, and the rest is not an identity.
 */
typedef struct NativeRunArtifactIdentity {
    uint64_t run_id;
    uint64_t dispatch_id;
    uint64_t generation;
    uint8_t owner_instance_id[8];
    uint32_t pipeline_slot;
    uint32_t chip_rank;
    uint32_t capture_index;
    uint32_t task_slot;
    uint32_t group_index;
    uint32_t group_size;
} NativeRunArtifactIdentity;

/**
 * Immutable resource and trace identity copied into one prepared run.
 * `pipeline_slot` and `arena_bank` must be smaller than the run-resource grant
 * this context was given (`simpler_set_run_resource_grant_ctx`); they remain
 * explicit because some runtimes map a leased slot to a different arena bank.
 * `run_epoch` is the process-unique
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
    /**
     * Nonzero to have this run construct a whole-operator completion boundary,
     * so that another run may later be ordered behind it.
     *
     * A property of the launching Worker's configured launch depth rather than
     * of this run: a predecessor is launched before any successor can be
     * authorized to join it, so a boundary constructed only once a join were
     * known would never exist when it is needed. Zero keeps the boundary that
     * covers one kernel and makes the run unjoinable, which is the behaviour a
     * Worker at launch depth one gets on every run.
     */
    uint32_t joinable_boundary;
    /** Where this dispatch came from, for diagnostic attribution only. */
    NativeRunProvenance provenance;
} NativeRunDescriptor;

/**
 * What one late-read retention probe is asked to do.
 *
 * `launch_successor` 0 runs the same sequence with no successor at all, which
 * is the no-successor baseline the overlapped arms are compared against.
 *
 * `use_retained_sync` performs the whole-pair synchronize `wait_run_fence`
 * still appends, before reading the record. It is the control arm rather than
 * an option: with a successor in flight that call waits for it, so the
 * successor's completion boundary reads Complete afterwards where the candidate
 * leaves it Pending. Running both arms over one sequence is what makes the
 * Pending observation evidence instead of an assumption.
 */
typedef struct RunRetentionProbeConfig {
    uint32_t launch_successor;
    uint32_t boundary_timeout_ms;
    uint32_t successor_start_timeout_ms;
    uint32_t use_retained_sync;
} RunRetentionProbeConfig;

/**
 * What that probe observed. Every `_rc` is 0 on success; the first non-zero one
 * is the step the sequence stopped at, and the fields after it are unset.
 *
 * Durations are monotonic nanoseconds. `record_read_ns` is the measurement the
 * probe exists for: the time to read the predecessor's result while the
 * successor is still executing. It is reported beside
 * `successor_completion_before_read` / `_after_read` because a read that
 * silently waited for the successor and a read that did not are
 * indistinguishable from the duration alone — the successor having still been
 * Pending on both sides of the read is what makes the number mean anything.
 *
 * That reading is one-sided, and deliberately so. `Pending` after the read can
 * only mean the read did not wait. `Complete` after it has two causes — the read
 * waited, or the successor finished on its own — and **nothing here separates
 * them.** In particular `successor_drain_ns` does not: a drain costs time even
 * on an already-finished run, which the retained-sync arm measures directly, so
 * no threshold on it distinguishes "still had work" from "already done". It is
 * reported for comparison between arms, never as a per-sample verdict.
 */
typedef struct RunRetentionProbeReport {
    int32_t launch_rc;
    int32_t boundary_wait_rc;
    int32_t pair_retire_rc;
    int32_t successor_launch_rc;
    int32_t successor_start_rc;
    int32_t successor_drain_rc;
    int32_t retained_sync_rc;
    int32_t execution_state;
    int32_t execution_code;
    uint32_t execution_source;
    uint32_t successor_completion_before_read;
    uint32_t successor_completion_after_read;
    uint32_t successor_started;
    uint32_t reserved;
    uint64_t boundary_wait_ns;
    uint64_t successor_start_ns;
    uint64_t record_read_ns;
    uint64_t decision_ns;
    uint64_t candidate_drain_ns;
    uint64_t reference_sync_ns;
    uint64_t successor_drain_ns;
    char execution_reason[96];
} RunRetentionProbeReport;

/* Per-stage run timing is no longer returned. The platform emits it as
 * `[STRACE]` log markers (host stages + the AICPU device-phase breakdown,
 * gated on SIMPLER_HOST_STRACE) — parse with simpler_setup.tools.strace_timing.
 * See docs/dfx/host-trace.md. */

/**
 * Which teardown branch `finalize_device` took.
 *
 * NORMAL and FATAL are different code paths, not severities: the fatal branch
 * is the one `device_unusable_` selects, and it is the only one that runs a
 * recovery wrapper with a post-reset probe.
 */
typedef enum TeardownPath {
    TEARDOWN_PATH_UNKNOWN = 0,
    TEARDOWN_PATH_NORMAL = 1,
    TEARDOWN_PATH_FATAL = 2,
} TeardownPath;

/**
 * How far the *last* reset attempt got. Describes that attempt alone — the
 * cumulative counters beside it are what say whether any earlier attempt
 * reached a reset API.
 *
 * TEARDOWN_STAGE_API_OK_NO_PROBE and TEARDOWN_STAGE_API_OK_PROBE_RUN are
 * orthogonal facts rather than an ordering: the first is a reset call that
 * returned 0 with nothing checked afterwards, the second is a reset call that
 * returned 0 with a post-reset probe behind it.
 */
typedef enum TeardownResetStage {
    TEARDOWN_STAGE_UNKNOWN = 0,
    /* No reset arm was reached at all. */
    TEARDOWN_STAGE_NOT_ATTEMPTED = 1,
    /* The branch declined to attempt: a kernel context owns no device reset. */
    TEARDOWN_STAGE_REFUSED = 2,
    /* ACL init or the device bind failed, so this attempt called no reset API. */
    TEARDOWN_STAGE_PREAMBLE_FAILED = 3,
    /* A reset API ran in this attempt and returned non-zero. */
    TEARDOWN_STAGE_API_FAILED = 4,
    /* A reset API returned 0 and this path runs no probe behind it. */
    TEARDOWN_STAGE_API_OK_NO_PROBE = 5,
    /* A reset API returned 0 and the post-reset probe ran; the
       TEARDOWN_FLAG_PROBE_CONFIRMED bit says whether it passed. */
    TEARDOWN_STAGE_API_OK_PROBE_RUN = 6,
} TeardownResetStage;

/** Which reset entry the recorded invocations used. */
typedef enum TeardownResetApi {
    TEARDOWN_RESET_API_NONE = 0,
    TEARDOWN_RESET_API_RT_DEVICE_RESET = 1,
    TEARDOWN_RESET_API_ACL_RESET_DEVICE = 2,
    TEARDOWN_RESET_API_ACL_RESET_DEVICE_FORCE = 3,
} TeardownResetApi;

enum {
    /* `last_reset_api_rc` holds a real return value. Clear means no reset API
       was invoked, which is a different fact from one that returned 0. */
    TEARDOWN_FLAG_LAST_RESET_API_RC_VALID = 1u << 0,
    /* `recovery_sequence_rc` holds a real return value. Only the fatal branch
       runs a recovery wrapper, so the normal branch leaves this clear rather
       than reporting a zero it never obtained. */
    TEARDOWN_FLAG_RECOVERY_SEQUENCE_RC_VALID = 1u << 1,
    /* The post-reset probe ran and passed. Set only with
       TEARDOWN_STAGE_API_OK_PROBE_RUN and a valid zero recovery rc. */
    TEARDOWN_FLAG_PROBE_CONFIRMED = 1u << 2,
    TEARDOWN_FLAG_RESERVED_MASK = ~0x7u,
    /* Bumped when a field changes meaning; a reader that does not recognise
       the value treats the whole record as absent. */
    TEARDOWN_REPORT_SCHEMA = 1,
};

/**
 * What one `finalize_device` observed about the device teardown it performed.
 *
 * Observation only. No field asserts that device work has stopped, and a
 * confirmed reset invalidates that device generation's allocations rather than
 * making an old pointer reusable.
 *
 * Two scopes coexist and are named apart on purpose. `reset_stage` is the last
 * attempt's stage; `last_reset_api_rc`, its validity bit and both counters are
 * cumulative over the whole teardown, which is what keeps a final
 * PREAMBLE_FAILED from implying no reset API ever ran.
 *
 * `schema` is the commit marker on the wire: the producer fills every other
 * field first and releases `schema` last, so a reader that sees
 * TEARDOWN_REPORT_SCHEMA has the whole record.
 */
enum {
    /* Bumped when a field changes meaning; a reader that does not recognise
       the value treats the whole record as absent. */
    WORKSPACE_REPORT_SCHEMA = 1,
};

/**
 * What the workspace manager of one device context currently accounts for.
 *
 * Observation plus one safety input. `live_blocked` is the only field a caller
 * acts on: it counts runs that still reference workspace and whose completion
 * a caller can still prove by finalizing them, which is what makes refusing a
 * teardown on it a refusal the caller can resolve.
 *
 * `budget_enforced` is 0 for every context that never latched a budget, which
 * is the default. `coverage_is_partial` is always 1: external tensors,
 * run-result and diagnostics regions, registered code, device ELF, RTS
 * argument blocks and provider memory are outside this budget, so
 * `limit_bytes` is not a Worker-wide or device-wide memory ceiling.
 *
 * `blocks_published` never wraps: it saturates, because a caller uses "this
 * context never published a block" as the one fact that exempts it from the
 * teardown check, and a wrapped counter would fake that exemption.
 *
 * `schema` is the commit marker on the wire: the producer fills every other
 * field first and releases `schema` last, so a reader that sees
 * WORKSPACE_REPORT_SCHEMA has the whole record.
 */
typedef struct SimplerWorkspaceReport {
    uint32_t schema;
    uint32_t budget_enforced;
    uint32_t coverage_is_partial;
    /* Runs still holding workspace whose proof can still arrive. */
    uint32_t live_blocked;
    uint64_t limit_bytes;
    /* Backing bytes charged against the budget, retained generations included. */
    uint64_t reserved_bytes;
    /* Bytes forgotten without being freed, because their last consumer could
       not be proven finished. Device reclamation of these is unknown. */
    uint64_t relinquished_bytes;
    /* Host mappings deliberately left registered for the same reason. */
    uint64_t quarantined_mapped_bytes;
    uint32_t release_unconfirmed_blocks;
    uint32_t quarantined_blocks;
    /* 1 when any block's release or retirement could not be proven. */
    uint32_t proof_unavailable;
    /* Saturating count of blocks this context ever published. */
    uint32_t blocks_published;
    /* Terminal-sweep release failures for allocations outside this budget. */
    uint32_t foreign_release_failures;
    int32_t last_foreign_release_rc;
} SimplerWorkspaceReport;

typedef struct SimplerTeardownReport {
    uint32_t schema;
    int32_t child_pid;
    int32_t device_id;
    uint8_t path;
    uint8_t reset_stage;
    uint8_t reset_api;
    uint8_t flags;
    int32_t last_reset_api_rc;
    int32_t recovery_sequence_rc;
    int32_t teardown_rc;
    uint16_t reset_api_invocations_total;
    uint16_t recovery_attempts_total;
    uint64_t reserved;
} SimplerTeardownReport;

enum {
    SIMPLER_TEARDOWN_REPORT_BYTES = 40,
};

/* ===========================================================================
 * Public API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

/**
 * Return this runtime's immutable program-mode resource declaration.
 * Program resources have bytes_per_copy == 0. Both AICPU_STREAM and
 * AICORE_STREAM must appear exactly once, each with class EXEC_HANDLE.
 * ChipWorker rejects unserviceable declarations before creating a context.
 */
const PipelineContract *get_pipeline_contract(void);

/**
 * Create a new device context (heap-allocated DeviceRunner).
 * Each ChipWorker should own one context for the lifetime of its init→finalize cycle.
 * The caller serializes init/finalize/destroy against other operations on this
 * handle. The mode latch protects identity, not the runner's whole lifecycle.
 * finalize_device is additionally serialized against callable registration and
 * unregistration; initialization and destroy are not.
 * Finalize must establish resource retirement before destroy; the handle's
 * immutable mode alone is not evidence that its resources can be released.
 * @return Opaque handle on success, NULL on failure.
 */
DeviceContextHandle create_device_context(void);

/**
 * Destroy a device context created by create_device_context().
 * The caller must finalize every prepared native run and call
 * finalize_device() first. Two conditions make this operation log an error and
 * leave the context alive: an active native run, and a kernel context that
 * still owns a device resource because its close reported failure — that one is
 * deliberately leaked, since destroying it would free handles a captured
 * ACLGraph may still reference. This entry returns void, so the caller learns
 * of a refusal from the failing finalize_device(), not from here; otherwise it
 * frees the underlying object.
 */
void destroy_device_context(DeviceContextHandle ctx);

/** Return the byte size of the opaque prepared-run storage. */
size_t get_runtime_size(void);

/** Return the required byte alignment of the opaque prepared-run storage. */
size_t get_runtime_alignment(void);

/**
 * Allocate device memory in the given device context, on a caller's request.
 *
 * The allocation is recorded as the caller's, which is what lets a run's arguments name it and what
 * makes `device_free_caller_buffer_ctx` able to refuse a release a run still needs. Memory the
 * context allocates for itself does not come through here and is not nameable by a run.
 */
void *device_malloc_ctx(DeviceContextHandle ctx, size_t size);

/** Free device memory previously allocated in the given device context. */
void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr);

/**
 * Release a caller allocation, unless a run may still be using it.
 *
 * The caller keeps the right to release throughout; this answers only whether now is safe. The
 * check and the release are one step inside the context, so a borrow taken between a caller's
 * question and its release cannot be missed.
 *
 * @return 0 when the allocation is gone, PTO_RUNTIME_ERR_INVALID_STATE when a borrow still holds
 *         it — in which case nothing changed. Nothing else: the platform free underneath reports
 *         no status, so 0 means this path did not refuse rather than that the pages are provably
 *         returned.
 */
int device_free_caller_buffer_ctx(DeviceContextHandle ctx, void *dev_ptr);

/** One span of a caller device allocation a run names. */
typedef struct CallerBufferSpan {
    uint64_t addr;
    uint64_t bytes;
} CallerBufferSpan;

/**
 * Take `borrow_id`'s reference on the caller allocations covering `spans`.
 *
 * Every known allocation is retained, including when other spans are external. Re-borrowing
 * replaces this id's set. The caller must release the id even after incomplete coverage.
 *
 * @return 0 when every span resolved; PTO_RUNTIME_ERR_INVALID_STATE when any did not, so joined
 *         admission stays refused while references to known allocations remain held.
 */
int device_borrow_caller_buffers_ctx(
    DeviceContextHandle ctx, const CallerBufferSpan *spans, uint32_t count, uint64_t borrow_id
);

/**
 * Drop `borrow_id`'s reference, or keep it for the process's remaining life.
 *
 * Nonzero `keep` is for a run whose last device consumer could not be proven finished: the device
 * may still name those bytes and nothing later can establish otherwise, so the reference is moved
 * beyond the reach of any release rather than merely delayed.
 */
void device_release_caller_buffers_ctx(DeviceContextHandle ctx, uint64_t borrow_id, int keep);

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

/**
 * Let the swimlane collector hold a run past its own boundary. Latched once at
 * device init.
 *
 * Off by default: every diagnostic path keeps today's behaviour, including the
 * artifact's name and the guarantee that a run's file exists when its `run()`
 * returns. With it on, a run's records outlive its boundary, a predecessor's
 * host-side ingest and file write overlap the successor's device execution, and
 * the files appear under a reserved artifact directory.
 *
 * Separate from `simpler_init` rather than an argument to it: the init
 * signature is resolved by name across the runtime .so boundary, so extending
 * it would break every module that does not ship in lockstep.
 */
int simpler_set_retain_runs_ctx(DeviceContextHandle ctx, int32_t enabled);

/**
 * Raise how many retained runs this context's swimlane collector may hold open
 * at once, and report whether it was granted.
 *
 * Optional, and resolved by `dlsym` like the gate above: a module built before
 * this exports no such symbol, and the caller then keeps the shipped capacity.
 * A module that exports it still refuses whatever it cannot serve — a limit
 * outside its range, a runtime with no joined-diagnostic backend, or a
 * collector already in use — and a refusal changes nothing, so the caller may
 * carry on at the shipped capacity rather than treating it as an error.
 *
 * Returns 0 when the limit is in force. Only a 0 return licenses the caller to
 * record the larger grant.
 */
int simpler_set_retained_epoch_limit_ctx(DeviceContextHandle ctx, int32_t limit);

/**
 * Tell this context how many run-resource sets its worker granted it, so every
 * per-run resource is built to that count.
 *
 * The module declares in its `PipelineContract` what it can support; a worker
 * decides how much of that this context uses, and that decision is not derivable
 * here. So it is published once, after the context is created and before any run
 * names a set.
 *
 * Optional, and resolved by `dlsym` like the gate above: a module built before
 * this exports no such symbol, and its own default sizing then stands. A module
 * that exports it refuses a grant it cannot build or one that would change the
 * meaning of a set a run already holds, and a refusal leaves the previous sizing
 * in force.
 *
 * Returns 0 when the grant is in force. Only a 0 return licenses the caller to
 * hand this context a set index above its previous sizing.
 */
int simpler_set_run_resource_grant_ctx(DeviceContextHandle ctx, uint32_t granted);

/**
 * The name the gate above shipped under, forwarding to it.
 *
 * Both are exported by every runtime module built from this tree, and each is
 * resolved by `dlsym` rather than linked, so a caller and a module from
 * different builds still find one name in common.
 */
int simpler_set_dfx_session_ctx(DeviceContextHandle ctx, int32_t enabled);

/**
 * Latch a finite workspace budget on this context, once, at init.
 *
 * Both onboard init entries install ownership management without a request.
 * This adds an optional finite limit, applied before workspace allocation.
 * The first accepted request stands: a second is refused rather than replacing
 * it. A module without this symbol cannot honor a requested limit, and the
 * caller must fail instead of running without that limit.
 *
 * The budget covers the per-slot retained temporary buffer and the three pooled
 * arena regions. It is not a device-wide ceiling; see SimplerWorkspaceReport.
 *
 * @return 0 on success, PTO_RUNTIME_ERR_INVALID_ARGUMENT for a zero budget or a
 *         second call, PTO_RUNTIME_ERR_UNSUPPORTED on a backend that manages no
 *         workspace.
 */
int simpler_set_workspace_budget_ctx(DeviceContextHandle ctx, uint64_t limit_bytes);

/**
 * Ask for workspace ownership management with no byte limit.
 *
 * Compatibility entry: both owned and borrowed onboard initialization install
 * management by default. Calling this before init reserves the once-only
 * request slot also used by `simpler_set_workspace_budget_ctx`.
 *
 * @return 0 on success, PTO_RUNTIME_ERR_INVALID_ARGUMENT when this context has
 *         already made a request, PTO_RUNTIME_ERR_INVALID_STATE once
 *         management is installed
 */
int simpler_enable_workspace_management_ctx(DeviceContextHandle ctx);

/**
 * Read this context's workspace accounting into a caller-owned record.
 *
 * `out_bytes` is the caller's `sizeof`, so a module built against a shorter
 * record is refused rather than written past. Valid output is published only
 * when the call returns 0 and `schema` is recognised; a caller protecting a
 * teardown must treat any other outcome as "unknown", never as "not managed" —
 * the two have opposite safety consequences.
 *
 * @return 0 on success, PTO_RUNTIME_ERR_INVALID_ARGUMENT for a null or short
 *         record, PTO_RUNTIME_ERR_UNSUPPORTED on a context that manages no
 *         workspace.
 */
int simpler_get_workspace_report_ctx(DeviceContextHandle ctx, SimplerWorkspaceReport *out, size_t out_bytes);

/**
 * Let this context size its own run concurrency from the resources it proves.
 *
 * Asked once, before anything is prepared, by a caller that was given no
 * explicit depth and no explicit byte budget. A context that is never asked
 * admits exactly what it was granted, as it did before.
 *
 * @return 0 when the context will size itself, PTO_RUNTIME_ERR_UNSUPPORTED on a
 *         context that cannot (no managed workspace, or already prepared work)
 */
int simpler_enable_automatic_admission_ctx(DeviceContextHandle ctx);

/**
 * Run-resource sets this context reports it can hold.
 *
 * A count of identities, never a byte figure. One until a run has completed its
 * resource commit; it then grows with what the context has proved and shrinks
 * when those resources are reclaimed or become unprovable. Equal to the grant
 * on a context that does not size itself.
 */
uint32_t simpler_admissible_run_resource_sets_ctx(DeviceContextHandle ctx);

/**
 * Give up every resource a saved plan on `pipeline_slot` still holds, and
 * forget the plan.
 *
 * The one entry an abandonment uses for a run whose prepare asked to wait: such
 * a run holds its input staging reference, the context's growth permission and
 * whatever its in-flight transaction staged, none of which an unprepared run
 * holds. Idempotent — a slot with no plan for this epoch reports success.
 *
 * @return 0 when the slot holds no plan for `run_epoch` afterwards,
 *         PTO_RUNTIME_ERR_INVALID_STATE when the run could still have reached
 *         the device, so its borrows must not be handed back here
 */
int simpler_abort_run_plan_ctx(DeviceContextHandle ctx, uint32_t pipeline_slot, uint64_t run_epoch);

/**
 * Publish every diagnostic run this context has closed, then report.
 *
 * Returns 0 when each promised file exists — a published partial counts, an
 * absent file does not. `error` receives a NUL-terminated reason on failure
 * and is untouched on success. A zero or negative `timeout_ms` waits with the
 * collector's own budget.
 */
int simpler_flush_diagnostics_ctx(DeviceContextHandle ctx, int32_t timeout_ms, char *error, size_t error_capacity);

/**
 * A snapshot of this context's diagnostic credit progress.
 *
 * Monotonic. Every release of a diagnostic output slot or terminal bank
 * advances it, as does an owner becoming fatal, cancellation and teardown — so
 * "the value changed" is the whole of what a waiter needs, and no caller has to
 * name which owner it is waiting for.
 *
 * Non-blocking and safe to call while holding the caller's own lane mutex. A
 * runtime that does not export it leaves the caller with no progress
 * information, which the caller reads as "re-ask next round".
 */
uint64_t simpler_diagnostic_credit_generation_ctx(DeviceContextHandle ctx);

/**
 * Block until the credit generation differs from `seen`.
 *
 * Must NOT be called while holding the caller's lane mutex or any collector
 * lock: the threads that advance the generation are the collectors' own
 * writers, and a waiter holding either could stop the release it is waiting
 * for.
 *
 * @return 1 when the generation advanced, 0 when `timeout_ms` elapsed, and a
 *         negative value when waits were cancelled, the context is shutting
 *         down, or the handle is not one this runtime owns.
 */
int simpler_wait_diagnostic_credit_ctx(DeviceContextHandle ctx, uint64_t seen, int32_t timeout_ms);

/**
 * Make every present and future credit wait on this context return.
 *
 * Non-blocking and idempotent: it sets the cancelled flag, broadcasts, and
 * returns. It proves nothing about any waiter having left — that is the
 * caller's own to establish — which is what makes it safe to call from a
 * thread that is itself inside a guarded call.
 */
void simpler_cancel_diagnostic_credit_waits_ctx(DeviceContextHandle ctx);

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
 * Required runtime entry for initialization diagnostics; NULL means no detail.
 * The context owns the returned string; copy it before cleanup, destruction,
 * or another initialization attempt. Initialization and cleanup are serialized
 * by the caller. Backends with no detail still export this entry and return NULL.
 */
const char *simpler_get_init_error_ctx(DeviceContextHandle ctx);

/**
 * Release all device resources held by the context.
 * Must be called before destroy_device_context() / dlclose(). Returns an error
 * without teardown while a prepared native run remains unfinalized.
 */
int finalize_device(DeviceContextHandle ctx);

/* ===========================================================================
 * Per-callable_id preparation
 *
 * These entries decouple the one-shot prep work (kernel upload + orch SO
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
 *
 * The table is a slot array on both host and AICPU, and the slot number is
 * the callable_id itself. A registration occupies one slot until
 * `unregister_callable`; a later registration may reuse it. Capacity is a
 * slot count, not a memory budget: with all 64 slots occupied, explicit and
 * auto-ID registration both fail until an ID is unregistered, and a consumer
 * whose working set exceeds 64 resident callables must recycle IDs or raise
 * the table size on both sides in a protocol change.
 *
 * Registration and unregistration entry points serialize against each other
 * on one context, so an auto-ID scan and the slot it claims are atomic with
 * respect to an explicit-ID claim of the same slot. `finalize_device` takes
 * the same lock, so teardown cannot clear the registry under a registration.
 * Runs, initialization and destroy still rely on the caller contract.
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
 * — they were captured by context initialization and live on the DeviceRunner.
 * Owned contexts attach their device; borrowed contexts validate the caller
 * context and register on their context-owned control stream. Unknown device
 * registration completion retains the callable until context close.
 * Callable-registry mutation is rejected while a native run is prepared or
 * executing on this context.
 *
 * A callable_id that is already registered is rejected with
 * PTO_RUNTIME_ERR_INVALID_ARGUMENT; unregister the ID before reusing it.
 *
 * @return 0 on success, negative on error (NULL ctx, callable_id out of
 *         range or already registered, upload/copy failure, or AICPU
 *         prewarm failure).
 */
int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable);

/**
 * Register on an initialized PROGRAM or KERNEL context with an unused ID.
 * KERNEL contexts use the kernel preparation path internally. The entry does
 * not initialize the context; an uninitialized, failed-init or finalized
 * context returns INVALID_STATE. `callable` points to a canonical ChipCallable
 * image of exactly `callable_size` bytes, fully validated before it is hashed
 * or uploaded.
 *
 * The registry is shared with explicit-ID registration, so both draw from the
 * same MAX_REGISTERED_CALLABLE_IDS (64) slots. Each successful call owns a
 * distinct ID, even for equal images; upload sharing follows the runtime's
 * existing image cache. Exhaustion returns INVALID_STATE.
 *
 * Registration and unregistration serialize internally on the context, so
 * concurrent auto-ID and explicit-ID registration cannot claim the same slot.
 * Registration is refused with PTO_RUNTIME_ERR_INTERNAL while a native run is
 * prepared or executing on the context. finalize_device takes the same lock;
 * runs, initialization and destroy do not, and callers still order those.
 *
 * Explicit-ID registration rejects an already registered ID with
 * INVALID_ARGUMENT; auto-ID never returns an occupied slot. IDs may be reused
 * after unregister, so callers retire all users first and discard stale IDs.
 *
 * @return 0 on success, PTO_RUNTIME_ERR_INVALID_ARGUMENT for a null output
 *         pointer or malformed image, PTO_RUNTIME_ERR_INVALID_STATE for an
 *         uninitialized/wrong-mode context or exhausted slots, and
 *         PTO_RUNTIME_ERR_INTERNAL for an outstanding native run or an
 *         unexpected failure. On failure *out_callable_id is -1 and cleanup
 *         ownership stays with the runtime.
 */
int simpler_register_callable_v2(
    DeviceContextHandle ctx, const void *callable, size_t callable_size, int32_t *out_callable_id
);

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
 *
 * Onboard kernel contexts use this same per-invocation preparation and
 * finalize path. The caller must keep its borrowed device current. Arguments
 * must be DEVICE/NONE or HOST/NONE; host tensor consumption is supported only
 * by a host-orchestrating runtime. No implicit tensor transfer is performed.
 * This preparation can allocate and synchronously publish metadata and must
 * run outside capture. Borrowed execution requires simpler_launch_run_on_stream;
 * the legacy simpler_kernel_mode_launch entry remains unsupported.
 *
 * A clean failed preparation writes a host-only release marker and this
 * descriptor's run_epoch into its destroyed carrier storage. This marker is
 * consumed by ChipWorker; zero only means destruction, not input release.
 * Missing, stale or unrecognized markers do not permit releasing inputs.
 * Failed prepare storage can be reused directly; prepare clears an unconsumed
 * release marker before constructing its next carrier.
 *
 * Onboard prepare is non-reentrant on one thread within a loaded runtime
 * module, including calls for a different context or through simpler_run().
 * A nested call returns PTO_RUNTIME_ERR_INVALID_STATE before validating its
 * arguments or modifying runtime storage, reservations, plans, or acceptance
 * sinks. The outer prepare remains owned until it returns or unwinds; a later
 * sequential call is permitted. This guard does not serialize different
 * threads, which remain subject to the context's native reservation rules.
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
 * Check the execution environment and reject capture before per-call preparation.
 * No run is admitted or published. A successful check does not reserve the stream;
 * the caller serializes capture changes and keeps it live through finalization.
 */
int simpler_validate_caller_stream(DeviceContextHandle ctx, void *caller_stream);

/**
 * Validate the current borrowed RTS context before changing host run state.
 * Owned/unclaimed contexts impose no current-context requirement. This call
 * neither binds a device nor mutates resources; restoration permits retry.
 */
int simpler_validate_execution_context(DeviceContextHandle ctx);

/**
 * Submit a prepared native run between the caller stream's preceding and
 * following work. The caller retains its stream and external tensor storage
 * through run finalization. Prepare is a host operation performed outside
 * capture; this eager launch rejects an active/invalidated capture before
 * publication. `predecessor`, when non-null, has the joined-launch contract.
 *
 * Both owned and borrowed contexts use the same native run. A borrowed context
 * requires this explicit stream entry. Poll/wait/finalize retain the run and
 * its event references through the caller's exit wait. A partial enqueue
 * failure poisons the execution domain; it is not a rollback permission.
 */
int simpler_launch_run_on_stream(
    DeviceContextHandle ctx, RuntimeHandle runtime, void *caller_stream, RuntimeHandle predecessor
);

/**
 * Return nonzero when this context can order one prepared run's native
 * submission behind another run that is already executing.
 *
 * Time-varying, unlike the compile-time runtime capability it folds in: a code
 * publication can leave the AICore stream needing replacement, and replacing it
 * under a run that is still executing on it is not possible. A caller that gets
 * zero launches ordinarily.
 */
int supports_joined_native_launch_ctx(DeviceContextHandle ctx);

/**
 * Return nonzero when `successor` may be ordered behind `predecessor` while
 * both collect swimlane diagnostics.
 *
 * Separate from `supports_joined_native_launch_ctx` on purpose. That answer is
 * about the context's ability to order two runs at all and is shared by every
 * runtime that can; this one additionally requires the one runtime whose
 * diagnostics collector can hold two runs open, the same effective swimlane
 * level on both sides, no other diagnostic channel, a collector that is
 * retaining and ready and not fatal, and a free retained bucket. Folding the
 * two would make ordinary non-diagnostic early enqueue depend on swimlane
 * capacity.
 *
 * Read-only and allocation-free: it arms nothing, reserves nothing and takes no
 * device action. A nonzero answer is therefore not a reservation — admission
 * rechecks capacity and fatality under its own lock — and a zero answer is an
 * ordinary decline the caller absorbs by launching at the front later.
 *
 * Both handles must name live prepared storage owned by this context, as for
 * the other native-run entry points.
 */
int supports_joined_diagnostic_launch_ctx(DeviceContextHandle ctx, RuntimeHandle successor, RuntimeHandle predecessor);

/**
 * Launch a prepared run ordered behind `predecessor`, which must be a run
 * launched on the same context and still executing.
 *
 * The successor's native submission reaches the device while the predecessor is
 * still executing; the device still executes one at a time, ordered by a queued
 * wait for the predecessor's whole-operator completion boundary. Neither run's
 * results, outputs or errors are shared: each is decided from its own evidence.
 *
 * Returns PTO_RUNTIME_ERR_UNSUPPORTED, having changed nothing, when this
 * context, this runtime or this moment cannot order the two — the run is still
 * prepared and the caller is expected to launch it ordinarily once it reaches
 * the front. Every other non-zero return is an ordinary launch failure with the
 * same meaning it has for simpler_launch_run().
 */
int simpler_launch_run_joined(DeviceContextHandle ctx, RuntimeHandle runtime, RuntimeHandle predecessor);

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
 * A launch explicitly known not to have submitted execution-visible work
 * leaves its epoch release marker after successful cleanup. A returned
 * NotStarted transaction supplies that fact; a backend exception or an absent
 * ActiveExecution alone does not. The original launch error remains observable.
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
 * Run #2267's late-read retention probe over two already-prepared runs.
 *
 * This is a fixture entry, not a production one, and nothing inside the product
 * calls it: it deliberately creates a state production admission refuses — a
 * predecessor that has completed and been read but not finalized, while its
 * successor executes on the same streams. Reaching that state needs a launch
 * permit minted rather than claimed and a pair-owner retirement performed
 * without the drain that normally accompanies it, neither of which is
 * expressible through the ordinary phase entries.
 *
 * It exists because the property node #2267 turns on — that a completed run's
 * result survives its successor, and that reading it does not wait for that
 * successor — cannot be demonstrated by the production path, whose drain
 * synchronizes the whole stream pair and therefore waits for the successor by
 * construction.
 *
 * Both runtimes must hold runs prepared on *distinct* pipeline slots and not
 * yet launched. On return both runs are complete and awaiting
 * `simpler_finalize_run`, which the caller still owes for each, exactly as
 * after an ordinary launch/wait. `runtime_successor` may be NULL when
 * `config->launch_successor` is 0.
 *
 * @return 0 when the sequence ran to completion, negative on a setup refusal.
 *         A device error inside the sequence is reported through the report's
 *         per-step `_rc` fields and still returns 0 — the probe's job is to
 *         measure, not to decide.
 */
int simpler_probe_run_retention(
    DeviceContextHandle ctx, RuntimeHandle runtime, RuntimeHandle runtime_successor,
    const RunRetentionProbeConfig *config, RunRetentionProbeReport *report
);

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

/**
 * Which device-side scheduler the most recent bind on `ctx` selected, as a
 * SCHEDULER_RUNTIME_MODE_* value; 0 means no bind has published a selection.
 *
 * Optional: ChipWorker resolves it with a bare dlsym, so a module that does not
 * export it is a fact about the backend rather than a stale build. Only the a5
 * host_build_graph runtime publishes a selection — every other arch and runtime
 * reports 0 because nothing ever calls the recording hook.
 *
 * The value is latched by the runner, not read back from the runtime: the
 * launch descriptor's copy is cleared when a run releases its bindings.
 */
uint32_t get_last_scheduler_runtime_mode(DeviceContextHandle ctx);

/**
 * How many times a bind on `ctx` has published a scheduler selection. Separates
 * "this bind selected again" from "the latch still holds an older run's value",
 * which a mode alone cannot express. Optional, like the getter above.
 */
size_t get_scheduler_mode_publish_count(DeviceContextHandle ctx);

/**
 * Copy out what this context's `finalize_device` observed about its own device
 * teardown.
 *
 * Optional: ChipWorker resolves it with a bare dlsym, so a module that does
 * not export it is a fact about the backend rather than a stale build.
 *
 * Exporting it is not the same as publishing one. A platform runner is shared
 * by every runtime built on it, so a module answers
 * PTO_RUNTIME_ERR_UNSUPPORTED unless its own runtime opts in — the record's
 * fields describe that runtime's actual reset paths, and a runtime whose paths
 * were not traced against them publishes nothing rather than a report it
 * cannot stand behind.
 *
 * Call it after `finalize_device` returns and before the context is destroyed;
 * there is no other window in which the recording owner is still alive.
 *
 * @return 0 when `out` was filled, PTO_RUNTIME_ERR_UNSUPPORTED when this
 *         runtime publishes no teardown report or this context recorded none,
 *         PTO_RUNTIME_ERR_INVALID_ARGUMENT on a null or wrongly sized
 *         destination.
 */
int get_teardown_report(DeviceContextHandle ctx, void *out, size_t out_bytes);

/* ===========================================================================
 * Kernel-mode lifecycle (four entries + finalize_device)
 *
 * A context has one of two execution modes for its whole lifetime, decided by
 * which init entry runs first. simpler_init latches program mode — the
 * exclusive-device path driven through the prepared-run family
 * above. simpler_kernel_mode_init latches kernel mode, which borrows the
 * caller's already-current device. Native eager uses simpler_prepare_run,
 * simpler_launch_run_on_stream, and the shared poll/wait/finalize family.
 * Its caller edges order the whole operator on the explicitly supplied stream.
 * No borrowed path resets or synchronizes the whole device. Cold-path bring-up
 * and registration synchronize a context-owned stream; partial enqueue cleanup
 * may synchronize the bound caller and private streams to prove references are
 * consumed. Capture/replay and the legacy simpler_kernel_mode_launch entry
 * remain unsupported.
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
 * Each call supplies its own config. TMR init validates config-derived scratch
 * sizes; HBG declares per-run banks whose sizes are determined at bind time.
 * Neither declaration commits those arenas. Native eager calls reuse an arena
 * while its capacity suffices and transactionally replace it when more is
 * needed. Native-run admission protects a shared bank; HBG per-run banks may
 * prepare independently. Failed staging preserves the committed regions.
 * Capture/replay remains unsupported; graph retention of arena addresses is
 * not provided by this eager resource lifecycle.
 *
 * Every host_runtime.so exports all four entries below, so a consumer resolves
 * them unconditionally like the rest of the uniform ABI; supported answers the
 * capability question at call time rather than gating symbol resolution.
 * Variants without kernel-mode support
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
 * The caller stream is explicit and borrowed: a native execution retains its
 * handle through finalization, and simpler never destroys it.
 * =========================================================================== */

/**
 * Return nonzero when this runtime can execute kernel-mode launches.
 *
 * Launches, not resources: it stays zero while simpler_kernel_mode_launch is a
 * rejecting stub, including on a runtime whose simpler_kernel_mode_init already
 * establishes a kernel context. Both onboard runtimes support initialization
 * and the shared native eager path via simpler_launch_run_on_stream. This
 * legacy capability bit does not advertise that native entry. Simulated
 * components refuse kernel init
 * with PTO_RUNTIME_ERR_UNSUPPORTED and establish nothing.
 *
 * This is the capability question a caller asks before choosing between the
 * legacy kernel-mode entries and the shared native-run family. It is callable
 * on a context that create_device_context() has returned but that no init has
 * touched; at that point the context exists but names no device —
 * create_device_context() takes no device and device_id_ is still unset — so an
 * implementation must answer from the runtime build alone, never from init
 * state or device identity. `ctx` is accepted for signature symmetry with the
 * rest of the family and must not be dereferenced for a device.
 */
int simpler_kernel_mode_supported(DeviceContextHandle ctx);

/**
 * Initialize a kernel-mode context on the caller's already-current device.
 *
 * A successful call latches kernel mode on the context — mutually exclusive
 * with the program-mode simpler_init — and the latch controls the
 * kernel-mode guards on the platform's device/ACL lifecycle.
 *
 * Takes no device ownership: no device reset and no ACL init/finalize. Creates
 * only context-owned persistent handles used by asynchronous preparation and
 * launch. Cold-path bring-up may synchronize a context-owned stream — the AICPU
 * init handshake does — but never a caller stream and never the device; the
 * launch path synchronizes nothing at all. `config` is read here for the
 * runtime's admission check only. Each simpler_prepare_run() copies its own
 * config and resolves that invocation's shape, binding and diagnostics.
 * `context_generation` is a nonzero host-process-unique identity minted by the
 * caller for sequential contexts; generation zero is invalid.
 * The native RTS context current at init is borrowed as well as the device.
 * Callable prepare/unregister, per-run prepare/finalize and context close
 * require that same context current on the calling thread. A mismatch is
 * rejected before mutation; restoring it permits retry. Simpler never calls
 * SetCurrentContext on the caller's behalf. The caller keeps this context
 * alive through successful close; handle comparison cannot detect a destroyed
 * context whose address has been recycled.
 *
 * Structural argument errors and invalid TMR sizing configurations return
 * PTO_RUNTIME_ERR_INVALID_ARGUMENT. Invalid generated resource contracts or
 * C++ exceptions during admission return PTO_RUNTIME_ERR_INTERNAL.
 * Simulated variants return PTO_RUNTIME_ERR_UNSUPPORTED and establish nothing.
 * Both onboard runtimes admit initialization; HBG determines per-run image
 * sizes at bind time rather than claiming a config-only size here.
 *
 * Success establishes this context's cold-path resources and latches kernel
 * mode. It does not mean kernel-mode launches are available:
 * simpler_kernel_mode_supported() answers that question and stays zero while
 * simpler_kernel_mode_launch is a rejecting stub. Nor does it retain the
 * validated contract or resolve per-preparation configuration — the contract
 * is validated and discarded, and `context_generation` is checked for being
 * nonzero and not yet given a retained lifecycle.
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
 * callable id range. Preparation may allocate persistent state and enqueue
 * asynchronous device work on context-owned streams. Registration synchronizes
 * its internal AICPU control stream before committing the callable, but never
 * synchronizes a caller stream or the device. Preparation neither accepts nor
 * retains a caller stream; the current caller/capture stream is supplied
 * independently to each launch.
 *
 * A callable_id that is already registered is rejected with
 * PTO_RUNTIME_ERR_INVALID_ARGUMENT. Registration is refused with
 * PTO_RUNTIME_ERR_INTERNAL while a native run is prepared or executing on the
 * context; retire the run before reusing the registry.
 */
int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, int32_t callable_id, const void *callable, size_t callable_size
);

/**
 * Reserved legacy single-call entry; currently a rejecting stub.
 * Native eager callers use simpler_prepare_run and simpler_launch_run_on_stream.
 *
 * The intended single-call contract is:
 *
 * `args` points to a ChipStorageTaskArgs POD whose tensor addresses are
 * caller-owned device addresses; they are passed through without ever being
 * dereferenced on the host. A launch performs no device malloc/free, no
 * tensor staging, no synchronize, no capture-state query, and no
 * stream-to-model attachment — keeping those out of the launch path is the
 * implementation's obligation. A return of 0 means the sequence was enqueued;
 * device execution may still be in flight and may still fail asynchronously.
 */
int simpler_kernel_mode_launch(DeviceContextHandle ctx, int32_t callable_id, const void *args, void *caller_stream);

#ifdef __cplusplus
}
#endif
