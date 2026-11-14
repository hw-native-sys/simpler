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
 *   are standalone bars rather than nested ones. Resolve is the AICore completion walk.
 *
 *   AICORE SCHEDULER: Bootstrap, StateProbe, Refill and Idle, plus Dispatch,
 *   Complete and Resolve. These come from the scheduler running on an AICore rather
 *   than the AICPU. Idle is a measured record here — that scheduler publishes a
 *   SchedulerIdleRecord per spin — not the host-side gap reconstruction a report
 *   derives for an AICPU capture.
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
    Resolve = 4,             // Completion work after FIN observation.
                             // tasks_processed = consumers visited.
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
    Refill = 14,             // Completed Slot reuse.
    Idle = 15,               // AICore scheduler spin with no progress.
    ScalarWindow = 16,
    ScalarCompletions = 18,
    ScalarGraphControl = 19,
    ScalarFillSlots = 20,
    ScalarAdmission = 21,
    ScalarExecutorScan = 22,
    ScalarPrepare = 23,
    ScalarSeal = 24,
    ScalarPublishCompletion = 25,
    ScalarAsyncService = 26,
    ScalarBackoff = 27,
    ScalarReadyClaim = 29,
    ScalarPayload = 30,
    ScalarResolve = 31,
    ScalarMix = 32,
    ScalarAsyncCheck = 33,
    ScalarAsyncFinish = 34,
    ScalarBootstrap = 35,
    ScalarLaunchWait = 36,
    ScalarExitDrain = 37,
    ScalarAtomicClaim = 38,
    ScalarCacheObserve = 39,
    ScalarCachePublish = 40,
    ScalarOrderedFill = 42,
    ScalarFlushCompletions = 43,
    ScalarLoopControl = 44,
    ScalarOrderedPoll = 45,
    ScalarAdmissionClaim = 46,
    ScalarAdmissionPublish = 47,
    ScalarAdmissionCommit = 49,
    ScalarAdmissionRefresh = 50,
    ScalarOrdinaryPark = 51,
    ScalarBlockClaim = 52,
    ScalarBlockCursorRead = 53,
    ScalarAdmissionClose = 54,
    ScalarOrdinaryPermit = 55,
    ScalarDispatchMetadata = 57,
    ScalarDispatchFill = 58,
    ScalarDispatchPublish = 59,
    ScalarInlineComplete = 60,
    ScalarCursorValidate = 61,
    ScalarClaimSealProbe = 62,
    ScalarClaimSealConfirm = 63,
    ScalarNextBlockProbe = 64,
    ScalarNextBlockConfirm = 65,
    ScalarPollLocalState = 66,
    ScalarPollReadyQueue = 67,
    ScalarPollReadyPool = 68,
    ScalarPollAdmissionTail = 70,
    ScalarPollMixState = 71,
    ScalarPollMixCloseSeal = 72,
    ScalarPollMixAssignment = 73,
    ScalarPollMixArrived = 74,
    ScalarPollMixReleased = 75,
    ScalarAtomicLoad = 79,
    ScalarAtomicCas = 80,
    ScalarAtomicExchange = 81,
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
    ScalarEarlyPrepare = 95,
    ScalarClusterPrepare = 96,
    ScalarExecutionTokenProgress = 97,
    ScalarExecutionTokenRefill = 98,
    ScalarExecutionTokenReceive = 99,
    ScalarExecutionTokenDispatch = 100,
    ScalarFaninObserve = 101,
    ScalarCooperativeDiscover = 102,
    ScalarFillOwnSlots = 103,
    ScalarFillCooperativeSlots = 104,
    ScalarClusterMixService = 105,
    ScalarClusterMixDiscoverTask = 106,
    ScalarClusterMixDiscoverNext = 107,
    ScalarClusterMixPending = 108,
    ScalarClusterMixArm = 109,
    ScalarClusterMixObserve = 110,
    ScalarClusterMixRetire = 111,
    ScalarClusterMixEnter = 112,
    ScalarClusterMixRelease = 113,
    ScalarClusterMixDrain = 114,
    ScalarClusterMixPublishPreparation = 115,
    ScalarClusterMixConflicts = 116,
    ScalarClusterMixAll = 117,
    ScalarPreparedReceive = 118,
    ScalarPreparedAcquire = 119,
    ScalarPreparedParkBegin = 120,
    ScalarPreparedParkPublish = 121,
    ScalarPreparedResume = 122,
    ScalarPreparedReturn = 123,
    ScalarPreparedObserve = 124,
    ScalarPreparedIndex = 125,
    ScalarPreparedValidate = 126,
    ScalarGroupReserve = 127,
    ScalarGroupPark = 128,
    ScalarGroupResume = 129,
    ScalarGroupArm = 130,
    ScalarGroupAcquire = 131,
    ScalarClusterMixNeedsService = 132,
    ScalarExecutionContinuation = 133,
    ScalarExecutionTokenFanin = 134,
    ScalarClusterMixReadyReplacement = 135,
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
    case SchedPhaseKind::Resolve:
        return "resolve";
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
    case SchedPhaseKind::Refill:
        return "refill";
    case SchedPhaseKind::Idle:
        return "idle";
    case SchedPhaseKind::ScalarWindow:
        return "scalar_window";
    case SchedPhaseKind::ScalarCompletions:
        return "scalar_completions";
    case SchedPhaseKind::ScalarGraphControl:
        return "scalar_graph_control";
    case SchedPhaseKind::ScalarFillSlots:
        return "scalar_fill_slots";
    case SchedPhaseKind::ScalarAdmission:
        return "scalar_admission";
    case SchedPhaseKind::ScalarExecutorScan:
        return "scalar_executor_scan";
    case SchedPhaseKind::ScalarPrepare:
        return "scalar_prepare";
    case SchedPhaseKind::ScalarSeal:
        return "scalar_seal";
    case SchedPhaseKind::ScalarPublishCompletion:
        return "scalar_publish_completion";
    case SchedPhaseKind::ScalarAsyncService:
        return "scalar_async_service";
    case SchedPhaseKind::ScalarBackoff:
        return "scalar_backoff";
    case SchedPhaseKind::ScalarReadyClaim:
        return "scalar_ready_claim";
    case SchedPhaseKind::ScalarPayload:
        return "scalar_payload";
    case SchedPhaseKind::ScalarResolve:
        return "scalar_resolve";
    case SchedPhaseKind::ScalarMix:
        return "scalar_mix";
    case SchedPhaseKind::ScalarAsyncCheck:
        return "scalar_async_check";
    case SchedPhaseKind::ScalarAsyncFinish:
        return "scalar_async_finish";
    case SchedPhaseKind::ScalarBootstrap:
        return "scalar_bootstrap";
    case SchedPhaseKind::ScalarLaunchWait:
        return "scalar_launch_wait";
    case SchedPhaseKind::ScalarExitDrain:
        return "scalar_exit_drain";
    case SchedPhaseKind::ScalarAtomicClaim:
        return "scalar_atomic_claim";
    case SchedPhaseKind::ScalarCacheObserve:
        return "scalar_cache_observe";
    case SchedPhaseKind::ScalarCachePublish:
        return "scalar_cache_publish";
    case SchedPhaseKind::ScalarOrderedFill:
        return "scalar_ordered_fill";
    case SchedPhaseKind::ScalarFlushCompletions:
        return "scalar_flush_completions";
    case SchedPhaseKind::ScalarLoopControl:
        return "scalar_loop_control";
    case SchedPhaseKind::ScalarOrderedPoll:
        return "scalar_ordered_poll";
    case SchedPhaseKind::ScalarAdmissionClaim:
        return "scalar_admission_claim";
    case SchedPhaseKind::ScalarAdmissionPublish:
        return "scalar_admission_publish";
    case SchedPhaseKind::ScalarAdmissionCommit:
        return "scalar_admission_commit";
    case SchedPhaseKind::ScalarAdmissionRefresh:
        return "scalar_admission_refresh";
    case SchedPhaseKind::ScalarOrdinaryPark:
        return "scalar_ordinary_park";
    case SchedPhaseKind::ScalarBlockClaim:
        return "scalar_block_claim";
    case SchedPhaseKind::ScalarBlockCursorRead:
        return "scalar_block_cursor_read";
    case SchedPhaseKind::ScalarAdmissionClose:
        return "scalar_admission_close";
    case SchedPhaseKind::ScalarOrdinaryPermit:
        return "scalar_ordinary_permit";
    case SchedPhaseKind::ScalarDispatchMetadata:
        return "scalar_dispatch_metadata";
    case SchedPhaseKind::ScalarDispatchFill:
        return "scalar_dispatch_fill";
    case SchedPhaseKind::ScalarDispatchPublish:
        return "scalar_dispatch_publish";
    case SchedPhaseKind::ScalarInlineComplete:
        return "scalar_inline_complete";
    case SchedPhaseKind::ScalarCursorValidate:
        return "scalar_cursor_validate";
    case SchedPhaseKind::ScalarClaimSealProbe:
        return "scalar_claim_seal_probe";
    case SchedPhaseKind::ScalarClaimSealConfirm:
        return "scalar_claim_seal_confirm";
    case SchedPhaseKind::ScalarNextBlockProbe:
        return "scalar_next_block_probe";
    case SchedPhaseKind::ScalarNextBlockConfirm:
        return "scalar_next_block_confirm";
    case SchedPhaseKind::ScalarPollLocalState:
        return "scalar_poll_local_state";
    case SchedPhaseKind::ScalarPollReadyQueue:
        return "scalar_poll_ready_queue";
    case SchedPhaseKind::ScalarPollReadyPool:
        return "scalar_poll_ready_pool";
    case SchedPhaseKind::ScalarPollAdmissionTail:
        return "scalar_poll_admission_tail";
    case SchedPhaseKind::ScalarPollMixState:
        return "scalar_poll_mix_state";
    case SchedPhaseKind::ScalarPollMixCloseSeal:
        return "scalar_poll_mix_close_seal";
    case SchedPhaseKind::ScalarPollMixAssignment:
        return "scalar_poll_mix_assignment";
    case SchedPhaseKind::ScalarPollMixArrived:
        return "scalar_poll_mix_arrived";
    case SchedPhaseKind::ScalarPollMixReleased:
        return "scalar_poll_mix_released";
    case SchedPhaseKind::ScalarAtomicLoad:
        return "scalar_atomic_load";
    case SchedPhaseKind::ScalarAtomicCas:
        return "scalar_atomic_cas";
    case SchedPhaseKind::ScalarAtomicExchange:
        return "scalar_atomic_exchange";
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
    case SchedPhaseKind::ScalarEarlyPrepare:
        return "scalar_early_prepare";
    case SchedPhaseKind::ScalarClusterPrepare:
        return "scalar_cluster_prepare";
    case SchedPhaseKind::ScalarExecutionTokenProgress:
        return "scalar_execution_token_progress";
    case SchedPhaseKind::ScalarExecutionTokenRefill:
        return "scalar_execution_token_refill";
    case SchedPhaseKind::ScalarExecutionTokenReceive:
        return "scalar_execution_token_receive";
    case SchedPhaseKind::ScalarExecutionTokenDispatch:
        return "scalar_execution_token_dispatch";
    case SchedPhaseKind::ScalarFaninObserve:
        return "scalar_fanin_observe";
    case SchedPhaseKind::ScalarCooperativeDiscover:
        return "scalar_cooperative_discover";
    case SchedPhaseKind::ScalarFillOwnSlots:
        return "scalar_fill_own_slots";
    case SchedPhaseKind::ScalarFillCooperativeSlots:
        return "scalar_fill_cooperative_slots";
    case SchedPhaseKind::ScalarClusterMixService:
        return "scalar_cluster_mix_service";
    case SchedPhaseKind::ScalarClusterMixDiscoverTask:
        return "scalar_cluster_mix_discover_task";
    case SchedPhaseKind::ScalarClusterMixDiscoverNext:
        return "scalar_cluster_mix_discover_next";
    case SchedPhaseKind::ScalarClusterMixPending:
        return "scalar_cluster_mix_pending";
    case SchedPhaseKind::ScalarClusterMixArm:
        return "scalar_cluster_mix_arm";
    case SchedPhaseKind::ScalarClusterMixObserve:
        return "scalar_cluster_mix_observe";
    case SchedPhaseKind::ScalarClusterMixRetire:
        return "scalar_cluster_mix_retire";
    case SchedPhaseKind::ScalarClusterMixEnter:
        return "scalar_cluster_mix_enter";
    case SchedPhaseKind::ScalarClusterMixRelease:
        return "scalar_cluster_mix_release";
    case SchedPhaseKind::ScalarClusterMixDrain:
        return "scalar_cluster_mix_drain";
    case SchedPhaseKind::ScalarClusterMixPublishPreparation:
        return "scalar_cluster_mix_publish_preparation";
    case SchedPhaseKind::ScalarClusterMixConflicts:
        return "scalar_cluster_mix_conflicts";
    case SchedPhaseKind::ScalarClusterMixAll:
        return "scalar_cluster_mix_all";
    case SchedPhaseKind::ScalarPreparedReceive:
        return "scalar_prepared_receive";
    case SchedPhaseKind::ScalarPreparedAcquire:
        return "scalar_prepared_acquire";
    case SchedPhaseKind::ScalarPreparedParkBegin:
        return "scalar_prepared_park_begin";
    case SchedPhaseKind::ScalarPreparedParkPublish:
        return "scalar_prepared_park_publish";
    case SchedPhaseKind::ScalarPreparedResume:
        return "scalar_prepared_resume";
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
    case SchedPhaseKind::ScalarGroupPark:
        return "scalar_group_park";
    case SchedPhaseKind::ScalarGroupResume:
        return "scalar_group_resume";
    case SchedPhaseKind::ScalarGroupArm:
        return "scalar_group_arm";
    case SchedPhaseKind::ScalarGroupAcquire:
        return "scalar_group_acquire";
    case SchedPhaseKind::ScalarClusterMixNeedsService:
        return "scalar_cluster_mix_needs_service";
    case SchedPhaseKind::ScalarExecutionContinuation:
        return "scalar_execution_continuation";
    case SchedPhaseKind::ScalarExecutionTokenFanin:
        return "scalar_execution_token_fanin";
    case SchedPhaseKind::ScalarClusterMixReadyReplacement:
        return "scalar_cluster_mix_ready_replacement";
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
