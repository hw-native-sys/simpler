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
 * The `hbg` runtime's Scheduler phase vocabulary, and what a report may ask of it.
 *
 * The vocabulary is private to this runtime. `tmr` has its own SchedPhaseKind
 * (src/common/tensormap_and_ringbuffer/sched_phase_kind.h), and the two are distinct
 * types in distinct namespaces. A name appearing in both is not a shared phase: it is
 * two phases that happen to carry the same label, and each runtime's own entry is what
 * describes it. So the enumerators are numbered independently — the same value means
 * different things on the two sides, and nothing compares one against the other.
 *
 * Both of this runtime's producers are listed here. The AICPU scheduler writes through
 * chip_swimlane_aicpu_record_sched_phase(); the AICore scheduler writes through
 * append_scheduler_record() in a5/runtime/host_build_graph/host/runtime_maker.cpp. They
 * share one vocabulary because they are one runtime, and a reader of the capture asks
 * which producer a stream came from rather than which phase names it may hold.
 *
 * Nothing in the include path enforces that a translation unit sees only one runtime's
 * vocabulary — src/common is on every target's — so the trailing using-declaration
 * does: reaching both headers from one scope is a compile error, not a silent bind to
 * whichever came first.
 */

#pragma once

#include <cstdint>

namespace simpler::hbg {

/**
 * SchedPhaseKind: which phase of this runtime's scheduling loop a record times.
 *
 * How a report draws each one:
 *
 *   OUTER (mutually time-exclusive within an iteration; emit advances the phase
 *   anchor): Complete, Dispatch, Dummy, EarlyDispatch, AsyncPoll, Drain, GraphPrepare.
 *
 *   INNER (no anchor advance; Perfetto nests by containment): DrainPrepare and
 *   DrainPublish.
 *
 *   RESOLUTION-THREAD OUTER: ResolveStandalone, AsyncPoll and Dummy. This runtime hands
 *   completed slots from Scheduler threads to a dedicated resolution thread, so these
 *   are standalone bars rather than nested ones.
 *
 *   AICORE SCHEDULER: Bootstrap, StateProbe and Idle, plus Dispatch and Complete.
 *   These records come from the scheduler running on an AICore rather than the
 *   AICPU. Idle is a measured SchedulerIdleRecord per spin, rather than the
 *   host-side gap reconstruction used for an AICPU capture.
 */
enum class SchedPhaseKind : uint8_t {
    Complete = 0,            // Observe FINs and run completion work inline.
                             // tasks_processed = finished subtasks + sub-block retires.
    Dispatch = 1,            // Publish ready tasks to AICore.
                             // tasks_processed = subtasks published.
    Dummy = 2,               // Explicit-dummy and false-predicate drain.
                             // tasks_processed = dummy tasks consumed.
    EarlyDispatch = 3,       // Pre-stage a flagged producer's gated consumers.
                             // tasks_processed = blocks staged.
    Drain = 5,               // sync_start stop-the-world drain attempt.
    DrainPrepare = 6,        // Nested sync_start staging prepare pass.
                             // tasks_processed = subtasks prepared.
    DrainPublish = 7,        // Nested sync_start MMIO publication pass.
                             // tasks_processed = subtasks published.
    AsyncPoll = 8,           // Async-engine completion polling.
                             // tasks_processed = async subtasks completed.
    GraphPrepare = 9,        // Bounded Graph Definition materialization slice.
                             // tasks_processed = sub-tasks patched.
    ResolveStandalone = 10,  // Dedicated resolution-thread work.
                             // tasks_processed = completed SPSC slots.
    Bootstrap = 11,          // AICore scheduler bring-up.
                             // tasks_processed = tasks held at bootstrap.
    StateProbe = 12,         // Scheduler-local Dispatch Slot / Ready state read.
    Idle = 15,               // AICore scheduler spin with no progress.
    ScalarWindow = 16,
    ScalarCompletions = 18,
    ScalarGraphControl = 19,
    ScalarExecutorScan = 22,
    ScalarPrepare = 23,
    ScalarSeal = 24,
    ScalarAsyncService = 26,
    ScalarAsyncCheck = 33,
    ScalarAsyncFinish = 34,
    ScalarBootstrap = 35,
    ScalarLaunchWait = 36,
    ScalarExitDrain = 37,
    ScalarCacheObserve = 39,
    ScalarCachePublish = 40,
    ScalarAdvanceMixFloor = 42,
    ScalarFlushCompletions = 43,
    ScalarLoopControl = 44,
    ScalarMixLaneOrderReady = 46,
    ScalarClusterMixDependencies = 47,
    ScalarClusterMixAccount = 49,
    ScalarClusterOwesBefore = 50,
    ScalarClusterMixRecycle = 54,
    ScalarSelfOrderReady = 55,
    ScalarDispatchMetadata = 57,
    ScalarDispatchPublish = 59,
    ScalarClusterMixPrepareMember = 71,
    ScalarAtomicCas = 80,
    ScalarAtomicStoreIssue = 82,
    ScalarAtomicFetchAdd = 83,
    ScalarAtomicFetchAddIssue = 84,
    ScalarDcciOut = 85,
    ScalarDcciSingle = 86,
    ScalarDcciEntire = 87,
    ScalarDsbAll = 88,
    ScalarDsbDdr = 89,
    ScalarBypassLoad = 90,
    ScalarBypassStore = 91,
    ScalarTraceFinish = 92,
    ScalarTraceFinishClean = 93,
    ScalarTraceFinishBarrier = 94,
    ScalarExecutionTokenProgress = 97,
    ScalarExecutionTokenRefill = 98,
    ScalarExecutionTokenReceive = 99,
    ScalarExecutionTokenSelect = 100,
    ScalarFaninObserve = 101,
    ScalarFillOwnSlots = 103,
    ScalarClusterMixService = 105,
    ScalarClusterMixDiscoverTask = 106,
    ScalarClusterMixDiscoverNext = 107,
    ScalarClusterMixPending = 108,
    ScalarClusterMixObserve = 110,
    ScalarClusterMixEnter = 112,
    ScalarClusterMixRelease = 113,
    ScalarClusterMixDrain = 114,
    ScalarClusterMixAll = 117,
    ScalarPreparedReceive = 118,
    ScalarPreparedAcquire = 119,
    ScalarPreparedReturn = 123,
    ScalarPreparedObserve = 124,
    ScalarPreparedIndex = 125,
    ScalarPreparedValidate = 126,
    ScalarGroupReserve = 127,
    ScalarGroupArm = 130,
    ScalarGroupAcquire = 131,
    ScalarExecutionContinuation = 133,
    ScalarExecutionTokenFanin = 134,
    ScalarTaskIssueHead = 135,
    ScalarTaskIssueFanin = 136,
    ScalarTaskSupplyClosed = 137,
    ScalarServiceCooperativeSlots = 138,
    ScalarBootstrapInitWait = 139,
    ScalarBootstrapPeerWait = 140,
    ScalarBootstrapTopology = 141,
    ScalarPreparedClaim = 142,
    ScalarCounterAccounting = 143,
    ScalarOrdinaryBlockClaim = 144,
    ScalarTaskCompletionObserve = 147,
    ScalarCachedCompletionObserve = 148,
    ScalarCachedCompletionLookup = 149,
    ScalarSubtaskContribute = 150,
    ScalarMixGroupContribute = 151,
    ScalarReturnsContribute = 152,
    ScalarCounterConditionPublish = 153,
    ScalarExecutionIdentityCapture = 154,
    ScalarAsyncBeginPrebound = 155,
    ScalarAsyncSealExecution = 156,
    ScalarAsyncCloseComputation = 157,
    ScalarClusterMixServiceActive = 158,
    ScalarGraphFaninView = 169,
    ScalarGraphPayloadArguments = 170,
    ScalarPublishIdle = 171,
    ScalarTraceBeforeExecution = 172,
    ScalarTraceBeforeCompletion = 173,
};

// This phase's name as a report spells it. Returns a string literal, so it is safe on
// the AICPU. A value outside the enum matches no case and falls past the switch to
// "unknown": the caller holds a corrupt record rather than a phase this runtime forgot
// to name.
constexpr const char *sched_phase_kind_name(SchedPhaseKind kind) {
    switch (kind) {
    case SchedPhaseKind::Complete:
        return "complete";
    case SchedPhaseKind::Dispatch:
        return "dispatch";
    case SchedPhaseKind::Dummy:
        return "dummy";
    case SchedPhaseKind::EarlyDispatch:
        return "early_dispatch";
    case SchedPhaseKind::Drain:
        return "drain";
    case SchedPhaseKind::DrainPrepare:
        return "drain_prepare";
    case SchedPhaseKind::DrainPublish:
        return "drain_publish";
    case SchedPhaseKind::AsyncPoll:
        return "async_poll";
    case SchedPhaseKind::GraphPrepare:
        return "graph_prepare";
    case SchedPhaseKind::ResolveStandalone:
        return "resolve_standalone";
    case SchedPhaseKind::Bootstrap:
        return "bootstrap";
    case SchedPhaseKind::StateProbe:
        return "state_probe";
    case SchedPhaseKind::Idle:
        return "idle";
    case SchedPhaseKind::ScalarWindow:
        return "scalar_window";
    case SchedPhaseKind::ScalarCompletions:
        return "scalar_completions";
    case SchedPhaseKind::ScalarGraphControl:
        return "scalar_graph_control";
    case SchedPhaseKind::ScalarExecutorScan:
        return "scalar_executor_scan";
    case SchedPhaseKind::ScalarPrepare:
        return "scalar_prepare";
    case SchedPhaseKind::ScalarSeal:
        return "scalar_seal";
    case SchedPhaseKind::ScalarAsyncService:
        return "scalar_async_service";
    case SchedPhaseKind::ScalarAsyncCheck:
        return "scalar_async_check";
    case SchedPhaseKind::ScalarAsyncFinish:
        return "scalar_async_finish";
    case SchedPhaseKind::ScalarBootstrap:
        return "scalar_bootstrap";
    case SchedPhaseKind::ScalarBootstrapInitWait:
        return "scalar_bootstrap_init_wait";
    case SchedPhaseKind::ScalarBootstrapPeerWait:
        return "scalar_bootstrap_peer_wait";
    case SchedPhaseKind::ScalarBootstrapTopology:
        return "scalar_bootstrap_topology";
    case SchedPhaseKind::ScalarLaunchWait:
        return "scalar_launch_wait";
    case SchedPhaseKind::ScalarExitDrain:
        return "scalar_exit_drain";
    case SchedPhaseKind::ScalarCacheObserve:
        return "scalar_cache_observe";
    case SchedPhaseKind::ScalarCachePublish:
        return "scalar_cache_publish";
    case SchedPhaseKind::ScalarAdvanceMixFloor:
        return "scalar_advance_mix_floor";
    case SchedPhaseKind::ScalarFlushCompletions:
        return "scalar_flush_completions";
    case SchedPhaseKind::ScalarLoopControl:
        return "scalar_loop_control";
    case SchedPhaseKind::ScalarMixLaneOrderReady:
        return "scalar_mix_lane_order_ready";
    case SchedPhaseKind::ScalarClusterMixDependencies:
        return "scalar_cluster_mix_dependencies";
    case SchedPhaseKind::ScalarClusterMixAccount:
        return "scalar_cluster_mix_account";
    case SchedPhaseKind::ScalarClusterOwesBefore:
        return "scalar_cluster_owes_before";
    case SchedPhaseKind::ScalarClusterMixRecycle:
        return "scalar_cluster_mix_recycle";
    case SchedPhaseKind::ScalarSelfOrderReady:
        return "scalar_self_order_ready";
    case SchedPhaseKind::ScalarDispatchMetadata:
        return "scalar_dispatch_metadata";
    case SchedPhaseKind::ScalarDispatchPublish:
        return "scalar_dispatch_publish";
    case SchedPhaseKind::ScalarClusterMixPrepareMember:
        return "scalar_cluster_mix_prepare_member";
    case SchedPhaseKind::ScalarAtomicCas:
        return "scalar_atomic_cas";
    case SchedPhaseKind::ScalarAtomicStoreIssue:
        return "scalar_atomic_store_issue";
    case SchedPhaseKind::ScalarAtomicFetchAdd:
        return "scalar_atomic_fetch_add";
    case SchedPhaseKind::ScalarAtomicFetchAddIssue:
        return "scalar_atomic_fetch_add_issue";
    case SchedPhaseKind::ScalarDcciOut:
        return "scalar_dcci_out";
    case SchedPhaseKind::ScalarDcciSingle:
        return "scalar_dcci_single";
    case SchedPhaseKind::ScalarDcciEntire:
        return "scalar_dcci_entire";
    case SchedPhaseKind::ScalarDsbAll:
        return "scalar_dsb_all";
    case SchedPhaseKind::ScalarDsbDdr:
        return "scalar_dsb_ddr";
    case SchedPhaseKind::ScalarBypassLoad:
        return "scalar_bypass_load";
    case SchedPhaseKind::ScalarBypassStore:
        return "scalar_bypass_store";
    case SchedPhaseKind::ScalarTraceFinish:
        return "scalar_trace_finish";
    case SchedPhaseKind::ScalarTraceFinishClean:
        return "scalar_trace_finish_clean";
    case SchedPhaseKind::ScalarTraceFinishBarrier:
        return "scalar_trace_finish_barrier";
    case SchedPhaseKind::ScalarExecutionTokenProgress:
        return "scalar_execution_token_progress";
    case SchedPhaseKind::ScalarExecutionTokenRefill:
        return "scalar_execution_token_refill";
    case SchedPhaseKind::ScalarExecutionTokenReceive:
        return "scalar_execution_token_receive";
    case SchedPhaseKind::ScalarExecutionTokenSelect:
        return "scalar_execution_token_select";
    case SchedPhaseKind::ScalarFaninObserve:
        return "scalar_fanin_observe";
    case SchedPhaseKind::ScalarFillOwnSlots:
        return "scalar_fill_own_slots";
    case SchedPhaseKind::ScalarClusterMixService:
        return "scalar_cluster_mix_service";
    case SchedPhaseKind::ScalarClusterMixDiscoverTask:
        return "scalar_cluster_mix_discover_task";
    case SchedPhaseKind::ScalarClusterMixDiscoverNext:
        return "scalar_cluster_mix_discover_next";
    case SchedPhaseKind::ScalarClusterMixPending:
        return "scalar_cluster_mix_pending";
    case SchedPhaseKind::ScalarClusterMixObserve:
        return "scalar_cluster_mix_observe";
    case SchedPhaseKind::ScalarClusterMixEnter:
        return "scalar_cluster_mix_enter";
    case SchedPhaseKind::ScalarClusterMixRelease:
        return "scalar_cluster_mix_release";
    case SchedPhaseKind::ScalarClusterMixDrain:
        return "scalar_cluster_mix_drain";
    case SchedPhaseKind::ScalarClusterMixAll:
        return "scalar_cluster_mix_all";
    case SchedPhaseKind::ScalarPreparedReceive:
        return "scalar_prepared_receive";
    case SchedPhaseKind::ScalarPreparedAcquire:
        return "scalar_prepared_acquire";
    case SchedPhaseKind::ScalarPreparedClaim:
        return "scalar_prepared_claim";
    case SchedPhaseKind::ScalarCounterAccounting:
        return "scalar_counter_accounting";
    case SchedPhaseKind::ScalarOrdinaryBlockClaim:
        return "scalar_ordinary_block_claim";
    case SchedPhaseKind::ScalarTaskCompletionObserve:
        return "scalar_task_completion_observe";
    case SchedPhaseKind::ScalarCachedCompletionObserve:
        return "scalar_cached_completion_observe";
    case SchedPhaseKind::ScalarCachedCompletionLookup:
        return "scalar_cached_completion_lookup";
    case SchedPhaseKind::ScalarSubtaskContribute:
        return "scalar_subtask_contribute";
    case SchedPhaseKind::ScalarMixGroupContribute:
        return "scalar_mix_group_contribute";
    case SchedPhaseKind::ScalarReturnsContribute:
        return "scalar_returns_contribute";
    case SchedPhaseKind::ScalarCounterConditionPublish:
        return "scalar_counter_condition_publish";
    case SchedPhaseKind::ScalarExecutionIdentityCapture:
        return "scalar_execution_identity_capture";
    case SchedPhaseKind::ScalarAsyncBeginPrebound:
        return "scalar_async_begin_prebound";
    case SchedPhaseKind::ScalarAsyncSealExecution:
        return "scalar_async_seal_execution";
    case SchedPhaseKind::ScalarAsyncCloseComputation:
        return "scalar_async_close_computation";
    case SchedPhaseKind::ScalarClusterMixServiceActive:
        return "scalar_cluster_mix_service_active";
    case SchedPhaseKind::ScalarGraphFaninView:
        return "scalar_graph_fanin_view";
    case SchedPhaseKind::ScalarGraphPayloadArguments:
        return "scalar_graph_payload_arguments";
    case SchedPhaseKind::ScalarPublishIdle:
        return "scalar_publish_idle";
    case SchedPhaseKind::ScalarTraceBeforeExecution:
        return "scalar_trace_before_execution";
    case SchedPhaseKind::ScalarTraceBeforeCompletion:
        return "scalar_trace_before_completion";
    case SchedPhaseKind::ScalarPreparedReturn:
        return "scalar_prepared_return";
    case SchedPhaseKind::ScalarPreparedObserve:
        return "scalar_prepared_observe";
    case SchedPhaseKind::ScalarPreparedIndex:
        return "scalar_prepared_index";
    case SchedPhaseKind::ScalarPreparedValidate:
        return "scalar_prepared_validate";
    case SchedPhaseKind::ScalarGroupReserve:
        return "scalar_group_reserve";
    case SchedPhaseKind::ScalarGroupArm:
        return "scalar_group_arm";
    case SchedPhaseKind::ScalarGroupAcquire:
        return "scalar_group_acquire";
    case SchedPhaseKind::ScalarExecutionContinuation:
        return "scalar_execution_continuation";
    case SchedPhaseKind::ScalarExecutionTokenFanin:
        return "scalar_execution_token_fanin";
    case SchedPhaseKind::ScalarTaskIssueHead:
        return "scalar_task_issue_head";
    case SchedPhaseKind::ScalarTaskIssueFanin:
        return "scalar_task_issue_fanin";
    case SchedPhaseKind::ScalarTaskSupplyClosed:
        return "scalar_task_supply_closed";
    case SchedPhaseKind::ScalarServiceCooperativeSlots:
        return "scalar_service_cooperative_slots";
    }
    return "unknown";
}

// Whether a record of this phase names the task it acted on, which decides whether the
// exporter writes phase_data.task_id or null. GraphPrepare names the outer modular task
// whose Definition it expanded; every other phase of this runtime acts on a batch rather
// than one task, and its task_id field holds nothing.
constexpr bool sched_phase_carries_task_id(SchedPhaseKind kind) { return kind == SchedPhaseKind::GraphPrepare; }

// Whether a record of this phase carries the ready-queue hit/miss deltas, which decides
// whether the exporter writes them beside it. They are a property of a pop attempt, so
// only Dispatch has them.
constexpr bool sched_phase_carries_pop_counters(SchedPhaseKind kind) { return kind == SchedPhaseKind::Dispatch; }

}  // namespace simpler::hbg

// A translation unit includes only its own runtime's sched_phase_kind.h, so the
// unqualified name names this type. Two of these declarations in one scope are
// ill-formed, which is what makes a build that reaches both runtimes fail here rather
// than silently pick one.
//
// Bare rather than qualified for the same reason TaskId is: the on-device record struct
// in platform/include/common/scheduler_profiling.h holds this type as a field, and that
// header belongs to neither runtime, so it has no namespace to qualify it with.
using simpler::hbg::SchedPhaseKind;
