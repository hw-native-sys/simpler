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

#pragma once

#include <cstddef>
#include <cstdint>

inline constexpr const char *CHIP_SWIMLANE_ARCHITECTURE_NAME = "a5";

/** Discriminator for Scheduler phase records.
 *
 * The roles share one enum so the on-device record carries one discriminator:
 *
 *   OUTER (mutually time-exclusive within an iteration; emit advances the
 *   phase anchor): Complete, Dispatch, Release, Dummy, EarlyDispatch,
 *   AsyncPoll, Drain, and GraphPrepare.
 *
 *   INNER (no anchor advance; Perfetto nests by containment): Resolve in the
 *   tensormap_and_ringbuffer runtime, plus DrainPrepare and DrainPublish.
 *
 *   HBG RESOLUTION-THREAD OUTER: ResolveStandalone, AsyncPoll, and Dummy. The
 *   host_build_graph runtime hands completed slots from Scheduler threads to a
 *   dedicated resolution thread, so these are standalone bars.
 *
 *   SEPARATE-LANE (Worker View rather than the Scheduler lane): DummyTask and
 *   PredicatedSkip identity markers.
 */
enum class ChipSwimlaneSchedPhaseKind : uint32_t {
    Complete = 0,            // Observe FINs and run completion work inline.
                             // tasks_processed = finished subtasks + sub-block retires.
    Dispatch = 1,            // Publish ready tasks to AICore.
                             // tasks_processed = subtasks published.
    Release = 2,             // Deferred-release drain.
                             // tasks_processed = slots released.
    Dummy = 4,               // Explicit-dummy and false-predicate drain.
                             // tasks_processed = dummy tasks consumed.
    EarlyDispatch = 5,       // Pre-stage a flagged producer's gated consumers.
                             // tasks_processed = blocks staged.
    Resolve = 6,             // Nested completion work after FIN observation.
                             // tasks_processed = consumers visited.
    DummyTask = 7,           // Zero-width dependency-only task identity marker.
    Drain = 8,               // sync_start stop-the-world drain attempt.
    DrainPrepare = 9,        // Nested sync_start staging prepare pass.
                             // tasks_processed = subtasks prepared.
    DrainPublish = 10,       // Nested sync_start MMIO publication pass.
                             // tasks_processed = subtasks published.
    AsyncPoll = 11,          // Async-engine completion polling.
                             // tasks_processed = async subtasks completed.
    PredicatedSkip = 12,     // Zero-width identity marker for a task retired
                             // because its dispatch predicate was false.
    GraphPrepare = 13,       // Bounded Graph Definition materialization slice.
                             // tasks_processed = in-graph tasks patched.
    ResolveStandalone = 14,  // Dedicated HBG resolution-thread work.
                             // tasks_processed = completed SPSC slots.
};

/** Queue-depth array layout: AIC=0, AIV=1, MIX=2.
 *
 * Must match ResourceShape's first three values. Kept local so this
 * architecture-owned ABI header remains runtime-independent.
 */
constexpr int CHIP_SWIMLANE_NUM_QUEUE_SHAPES = 3;

/**
 * AICPU Scheduler phase record (64 bytes).
 *
 * Position in the per-thread buffer is the thread identity. All timestamps are
 * raw system-counter cycles.
 *
 * ``phase_data`` is tagged by ``kind``: Dispatch uses ``dispatch``;
 * DummyTask and PredicatedSkip use ``dummy_task``; GraphPrepare uses
 * ``graph_task``. Other kinds store zero in the union.
 *
 * Queue-depth snapshots use the [AIC, AIV, MIX] indexes above and capture
 * ready-queue occupancy at phase boundaries. They remain zero below
 * SCHED_PHASES.
 */
struct ChipSwimlaneAicpuSchedPhaseRecord {
    uint64_t start_time;              // Phase start, in system-counter cycles
    uint64_t end_time;                // Phase end, in system-counter cycles
    uint32_t loop_iter;               // Scheduler-loop iteration on this thread
    ChipSwimlaneSchedPhaseKind kind;  // Tagged-union discriminator
    uint32_t tasks_processed;         // Work items processed in this phase
    union {
        struct {
            uint32_t pop_hit;   // Ready-queue hit delta since the previous Dispatch
            uint32_t pop_miss;  // Ready-queue miss delta since the previous Dispatch
        } dispatch;
        struct {
            uint32_t local_id;  // task_id bits [31:0]
            uint32_t ring_id;   // task_id bits [63:32]
        } dummy_task;
        struct {
            uint32_t local_id;  // outer Graph task_id bits [31:0]
            uint32_t ring_id;   // outer Graph task_id bits [63:32]
        } graph_task;
    } phase_data;
    int16_t shared_depth_at_start[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];  // Ready depths at phase entry
    int16_t shared_depth_at_end[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];    // Ready depths at phase exit
    uint32_t _pad[4];                                               // Keep the wire record at 64 bytes
};

static_assert(
    sizeof(decltype(ChipSwimlaneAicpuSchedPhaseRecord::phase_data)) == 8,
    "ChipSwimlaneAicpuSchedPhaseRecord phase data must remain 8 bytes"
);
static_assert(
    offsetof(ChipSwimlaneAicpuSchedPhaseRecord, phase_data) == 28,
    "ChipSwimlaneAicpuSchedPhaseRecord phase data offset drift"
);
static_assert(sizeof(ChipSwimlaneAicpuSchedPhaseRecord) == 64, "ChipSwimlaneAicpuSchedPhaseRecord layout drift");
