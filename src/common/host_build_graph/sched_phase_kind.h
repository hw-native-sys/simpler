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
 *   AICORE SCHEDULER: Bootstrap, StateProbe, Worksteal, Refill and Idle, plus Dispatch,
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
    Worksteal = 13,          // Remote Inbox claim and dispatch.
    Refill = 14,             // Completed Slot reuse.
    Idle = 15,               // AICore scheduler spin with no progress.
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
    case SchedPhaseKind::Worksteal:
        return "worksteal";
    case SchedPhaseKind::Refill:
        return "refill";
    case SchedPhaseKind::Idle:
        return "idle";
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
