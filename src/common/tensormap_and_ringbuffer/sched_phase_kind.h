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
 * The `tmr` runtime's Scheduler phase vocabulary, and what a report may ask of it.
 *
 * The vocabulary is private to this runtime. `hbg` has its own SchedPhaseKind
 * (src/common/host_build_graph/sched_phase_kind.h), and the two are distinct types in
 * distinct namespaces. A name appearing in both is not a shared phase: it is two phases
 * that happen to carry the same label, and each runtime's own entry is what describes
 * it. So the enumerators are numbered independently — the same value means different
 * things on the two sides, and nothing compares one against the other.
 *
 * One producer writes all of these: this runtime's AICPU scheduler, through
 * chip_swimlane_aicpu_record_sched_phase(). It has no AICore scheduler, so there is no
 * second vocabulary to reconcile.
 *
 * Nothing in the include path enforces that a translation unit sees only one runtime's
 * vocabulary — src/common is on every target's — so the trailing using-declaration
 * does: reaching both headers from one scope is a compile error, not a silent bind to
 * whichever came first.
 */

#pragma once

#include <cstdint>

namespace simpler::tmr {

/**
 * SchedPhaseKind: which phase of this runtime's scheduling loop a record times.
 *
 * How a report draws each one:
 *
 *   OUTER (mutually time-exclusive within an iteration; emit advances the phase
 *   anchor): Complete, Dispatch, Release, Dummy, EarlyDispatch, AsyncPoll, Drain.
 *
 *   INNER (no anchor advance; Perfetto nests by containment): Resolve, DrainPrepare and
 *   DrainPublish. Resolve is timed inside the phase that observed the FIN, so
 *   containment is the only thing that can tell it from standalone work.
 *
 *   SEPARATE-LANE (Worker View rather than the Scheduler lane): DummyTask and
 *   PredicatedSkip identity markers. Both name a task that never reached an AICore, so
 *   a report draws them as a DAG node briefly inhabiting the AICPU as a virtual worker.
 */
enum class SchedPhaseKind : uint8_t {
    Complete = 0,         // Observe FINs and run completion work inline.
                          // tasks_processed = finished subtasks + sub-block retires.
    Dispatch = 1,         // Publish ready tasks to AICore.
                          // tasks_processed = subtasks published.
    Dummy = 2,            // Explicit-dummy and false-predicate drain.
                          // tasks_processed = dummy tasks consumed.
    EarlyDispatch = 3,    // Pre-stage a flagged producer's gated consumers.
                          // tasks_processed = blocks staged.
    Resolve = 4,          // Nested completion work after FIN observation.
                          // tasks_processed = consumers visited.
    Drain = 5,            // sync_start stop-the-world drain attempt.
    DrainPrepare = 6,     // Nested sync_start staging prepare pass.
                          // tasks_processed = subtasks prepared.
    DrainPublish = 7,     // Nested sync_start MMIO publication pass.
                          // tasks_processed = subtasks published.
    AsyncPoll = 8,        // Async-engine completion polling.
                          // tasks_processed = async subtasks completed.
    Release = 9,          // Deferred-release drain.
                          // tasks_processed = slots released.
    DummyTask = 10,       // Zero-width dependency-only task identity marker.
    PredicatedSkip = 11,  // Zero-width identity marker for a task retired because its
                          // dispatch predicate was false.
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
    case SchedPhaseKind::Release:
        return "release";
    case SchedPhaseKind::DummyTask:
        return "dummy_task";
    case SchedPhaseKind::PredicatedSkip:
        return "predicated_skip";
    }
    return "unknown";
}

// Whether a record of this phase names the task it acted on, which decides whether the
// exporter writes phase_data.task_id or null. The two markers exist to name one task
// each; every other phase of this runtime acts on a batch rather than one task, and its
// task_id field holds nothing.
constexpr bool sched_phase_carries_task_id(SchedPhaseKind kind) {
    return kind == SchedPhaseKind::DummyTask || kind == SchedPhaseKind::PredicatedSkip;
}

// Whether a record of this phase carries the ready-queue hit/miss deltas, which decides
// whether the exporter writes them beside it. They are a property of a pop attempt, so
// only Dispatch has them.
constexpr bool sched_phase_carries_pop_counters(SchedPhaseKind kind) { return kind == SchedPhaseKind::Dispatch; }

}  // namespace simpler::tmr

// A translation unit includes only its own runtime's sched_phase_kind.h, so the
// unqualified name names this type. Two of these declarations in one scope are
// ill-formed, which is what makes a build that reaches both runtimes fail here rather
// than silently pick one.
//
// Bare rather than qualified for the same reason TaskId is: the on-device record struct
// in platform/include/common/scheduler_profiling.h holds this type as a field, and that
// header belongs to neither runtime, so it has no namespace to qualify it with.
using simpler::tmr::SchedPhaseKind;
