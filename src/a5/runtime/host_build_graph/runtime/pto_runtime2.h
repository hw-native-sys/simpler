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
 * host_build_graph runtime interface.
 *
 * Defines the host-built graph image, runtime operations and lifecycle APIs
 * shared by the host orchestrator and the AICPU scheduler.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <type_traits>

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"
#include "graph_cache.h"
#include "pto_submit_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "scheduler/pto_scheduler.h"
#include "pto_orchestrator.h"
#include "aicore_completion_mailbox.h"

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum PTO2RuntimeMode {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

/**
 * Function-pointer ops table for runtime operations.
 *
 * The orchestration .so calls runtime functions through this table
 * (via pto_orchestration_api.h inline wrappers), so it has zero link
 * dependencies on runtime .cpp files.
 */
typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures
class HostTensorAccessor;

struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(
        PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const CoreTaskArgs &args);

    // This-run core geometry from runtime_finalize_after_wire: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(PTO2Runtime *rt);
    int32_t (*available_aiv_count)(PTO2Runtime *rt);
    GraphScopeResult (*graph_begin)(PTO2Runtime *rt, uint64_t graph_key, const GraphTaskArgs &args);
    bool (*graph_prepare)(PTO2Runtime *rt, void *recording_handle, const GraphTaskArgs &args);
    void (*graph_abort)(PTO2Runtime *rt, void *recording_handle);
    bool (*graph_end)(PTO2Runtime *rt);
    void (*graph_commit)(PTO2Runtime *rt);
    // Stash the call-site captured by PTO2ScopeGuard into the [ScopeStats]
    // collector. Always present in the struct to keep ops-table layout stable
    // across SIMPLER_DFX settings; set to nullptr at SIMPLER_DFX=0.
    void (*scope_set_site)(const char *file, int line);
};

/**
 * Layout descriptor for the prebuilt runtime arena. Holds all sub-region
 * offsets (orchestrator / scheduler / sm_handle wrapper / runtime header /
 * AICore mailbox) plus the layout-defining capacities. Produced once on the
 * host by runtime_reserve_layout(); consumed by runtime_init_data_from_layout
 * and runtime_wire_arena_pointers.
 */
struct PTO2RuntimeArenaLayout {
    size_t off_sm_handle{0};
    PTO2OrchestratorLayout orch;
    PTO2SchedulerLayout sched;
    size_t off_scheduler{0};
    size_t off_runtime{0};
    size_t off_mailbox{0};
    // The arena is reserved in zones, in this order, and the offsets above land in
    // exactly one of them:
    //
    //   device-only  sm_handle, the AICore mailbox, the scheduler state and its
    //                queue slot arrays. Reachable storage whose content is a
    //                function of the layout, not of the run, so the device writes
    //                it at boot instead of receiving an initialization pattern
    //                over PCIe.
    //   copied       [off_copied_begin, off_copied_end). Every byte carries this
    //                run's own content, so bind ships the whole zone as one
    //                contiguous range.
    //
    // off_copied_end is where the shared part of the layout stops. Past it the two
    // sides carry different tails, and neither reads the other's:
    //
    //   host tail    the orchestrator block. Dep-computation scratch no device code
    //                reads, so it is neither copied nor allocated on the device.
    //   device tail  the shared-memory image, whose size is the submitted task
    //                count and therefore not known when this layout is built. bind
    //                grows the device region to cover it once orchestration ends.
    //
    // The copied zone is padded to a PTO2_ALIGN_SIZE boundary, so the device tail
    // begins exactly at off_copied_end: the copied zone and the shared-memory image
    // are adjacent on the device and travel as one copy.
    size_t off_copied_begin{0};
    size_t off_copied_end{0};

    // Byte length of the device region before its shared-memory tail. The host
    // arena is arena_size, which instead covers the orchestrator block there.
    size_t device_bytes{0};

    // Cached parameter used to reserve host-only state and when the AICPU
    // attaches the populated SM image.
    uint64_t task_capacity{0};

    // Total arena byte size post-commit. Used by host to size the prebuilt
    // image buffer and as the rtMemcpy length.
    size_t arena_size{0};
};

static_assert(
    std::is_trivially_copyable_v<PTO2RuntimeArenaLayout> && std::is_standard_layout_v<PTO2RuntimeArenaLayout>,
    "PTO2RuntimeArenaLayout must remain a device-copyable POD layout"
);

/**
 * host_build_graph runtime context.
 *
 * Contains all state for orchestration and scheduling.
 * In simulated mode, runs in single process with shared address space.
 */
struct PTO2Runtime {
    // Ops table (first field — used by orchestration .so via function pointers)
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;

    // Components
    PTO2SharedMemoryHandle *sm_handle;
    PTO2OrchestratorState orchestrator;
    // Device-only zone: the scheduler state holds no per-run content, so it is
    // addressed through the arena rather than carried inside this header.
    PTO2SchedulerState *scheduler;
    AICoreCompletionMailbox *aicore_mailbox;

    // GM Heap for output buffers
    void *gm_heap;
    uint64_t gm_heap_size;
    bool gm_heap_owned;  // True if we allocated it

    // Mode
    PTO2RuntimeMode mode;

    // Statistics
    int64_t total_cycles;
    // Graph definitions are process-local host cache entries. The callable
    // identity prevents two orchestration DSOs from sharing the same key.
    uint64_t active_callable_hash;

    // Host views of the tensors this run staged, owned by the run that
    // registered them. Null on the AICPU path, which loads device addresses
    // directly; get_tensor_data / set_tensor_data then fail closed rather than
    // dereferencing one. Lives past the first two fields, so the orchestration
    // .so's partial PTO2Runtime definition neither sees nor needs it.
    HostTensorAccessor *tensor_access;

    // Prebuilt-arena fast path metadata. Carries every offset
    // wire_arena_pointers needs at AICPU boot so the AICPU can reconstruct
    // all arena-internal pointer fields without re-running init_data. The
    // device base of the runtime arena travels separately on the host-side
    // Runtime (Runtime::prebuilt_arena_base_), since the AICPU needs it
    // *before* dereferencing this image. Populated on host by
    // runtime_init_data_from_layout + runtime_wire_arena_pointers; read by
    // aicpu_executor.cpp.
    PTO2RuntimeArenaLayout prebuilt_layout;
};

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Phase 1 — declare every sub-region (sm_handle wrapper, orchestrator /
 * scheduler / tensor_map / mailbox / PTO2Runtime header) on the supplied
 * arena. Pure arithmetic; does not touch device memory and may run on host.
 * Returns the layout descriptor; caller commits/attaches the arena before
 * Phase 2/3.
 */
PTO2RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena, uint64_t task_capacity);

/**
 * Phase 2 — write the data half of the runtime arena: standalone fields,
 * memset'd arena regions, sub-structure initializers, and SM-side device
 * pointers. The arena must already be committed (or attached); writes go
 * into arena.base() + sub-region offsets.
 *
 * `sm_dev_base` / `gm_heap_dev_base` are device addresses; we only store
 * them (never dereference). Safe to run on a host arena that owns a host
 * mirror of the runtime image — the resulting buffer is rtMemcpy-ready.
 *
 * Returns the PTO2Runtime* that sits at layout.off_runtime within the arena.
 * Caller must follow up with runtime_wire_arena_pointers; rt->ops and the
 * AICore-side count fields are left untouched and must be filled by the
 * AICPU at boot. Initializes the scheduler only: the orchestrator is left
 * zeroed for the host-orch path (run_host_orchestration) to initialize
 * against the host SM, since initializing it here would be overwritten and
 * is never uploaded to the device.
 */
PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *sm_dev_base, uint64_t sm_size,
    void *gm_heap_dev_base, uint64_t heap_size
);

/**
 * Phase 3 — wire the arena-internal pointer fields that exist on both sides
 * (rt->sm_handle, rt->aicore_mailbox, rt->scheduler and
 * scheduler.{ready_queues, ready_sync_queues, early_dispatch_queues}) so each
 * holds arena.base() + offset. Idempotent — runs on both host (writing
 * host-mirror addresses) and AICPU (writing device addresses) sides.
 */
void runtime_wire_arena_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt);

/**
 * Phase 3b — wire the orchestrator's pointers into the host-only zone
 * (orchestrator.{scope_tasks, scope_begins, scheduler, tensor_map.*,
 * ring.fanin_pool.base}).
 *
 * Host-only by construction: that zone is past layout.device_bytes and so is not
 * allocated on the device, which makes the addresses this writes unrepresentable
 * there. Requires rt->scheduler, so it follows runtime_wire_arena_pointers.
 */
void runtime_wire_host_only_pointers(DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2Runtime *rt);

/**
 * Drop the pointers Phase 3b wrote, so the runtime header can be copied to the
 * device without carrying host addresses into it. The device reads one
 * orchestrator field, inline_completed_tasks, which this leaves alone.
 *
 * Call after orchestration finishes and before the copied zone is uploaded.
 */
void runtime_clear_host_only_pointers(PTO2Runtime *rt);

/**
 * AICPU-only Phase 4 — fill in the few fields the host could not know at
 * prebuilt-image build time: the ops table (s_runtime_ops is a device-side
 * file-local global, host cannot resolve its device address) and the
 * orchestrator's core counts (depend on the executor's scheduler context).
 * Call once per boot after runtime_wire_arena_pointers.
 */
void runtime_finalize_after_wire(PTO2Runtime *rt, int32_t aic_count, int32_t aiv_count);

/**
 * Destroy runtime. With the prebuilt-arena fast path the arena buffer is
 * pooled across runs by DeviceRunner, so we never call arena.release()
 * here — the destructor only forgets sub-structure pointers (idempotent
 * cleanup).
 */
void runtime_destroy(PTO2Runtime *rt, DeviceArena &arena);

/**
 * Set execution mode
 */
void runtime_set_mode(PTO2Runtime *rt, PTO2RuntimeMode mode);

// =============================================================================
// Orchestration API (called by orchestration function)
// =============================================================================

/**
 * Begin a new scope
 *
 * All tasks submitted within this scope will have their lifetime
 * bounded by the scope. When scope_end() is called, the scope
 * releases its reference to all enclosed tasks.
 */
void rt_scope_begin(PTO2Runtime *rt);

/**
 * End current scope
 *
 * Releases scope reference for all tasks submitted since scope_begin().
 * Tasks whose refcount reaches zero will have their buffers released.
 */
void rt_scope_end(PTO2Runtime *rt);

/**
 * Mark orchestration as complete
 *
 * Signals that no more tasks will be submitted.
 */
void rt_orchestration_done(PTO2Runtime *rt);

/**
 * Enter fatal state explicitly from orchestration.
 */
void rt_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

/**
 * Cross-layer data access: read a tensor value by waiting for its producer.
 */
uint64_t get_tensor_data(PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]);

/**
 * Cross-layer data access: write a value to a tensor at given indices.
 * Waits for producer completion (WAW) and all consumers (WAR) via TensorMap.
 * See set_tensor_data in pto_orchestration_api.h for full documentation.
 */
void set_tensor_data(
    PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
);

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with pto_orchestration_api.h (same layout, guarded).
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif
