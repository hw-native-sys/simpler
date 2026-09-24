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

// The owning runtime's task handle. Each runtime has its own TaskId in its own
// namespace, and the include path resolves this bare name to whichever runtime is
// being built: src/common/<runtime> is on that build's include path, and reaching
// both headers from one scope is a compile error rather than a silent pick.
#include "task_id.h"
// The owning runtime's Scheduler phase vocabulary, resolved the same way. Which phases
// exist, what each one means, and which of them name a task are that runtime's to
// state, so this header holds the discriminator without knowing its values.
#include "sched_phase_kind.h"

inline constexpr const char *CHIP_SWIMLANE_ARCHITECTURE_NAME = "a5";

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
 * ``phase_data`` is tagged by ``kind``, and which member a kind selects is the owning
 * runtime's to state rather than this header's:
 * ``sched_phase_carries_pop_counters()`` names the kinds holding ``dispatch``,
 * ``sched_phase_carries_task_id()`` those holding ``task_id``, and a kind answering
 * neither stores zero in the union. Both predicates live in that runtime's
 * sched_phase_kind.h, beside the phases they classify.
 *
 * Queue-depth snapshots use the [AIC, AIV, MIX] indexes above and capture
 * ready-queue occupancy at phase boundaries. They remain zero below
 * SCHED_PHASES.
 */
struct ChipSwimlaneAicpuSchedPhaseRecord {
    uint64_t start_time;  // Phase start, in system-counter cycles
    uint64_t end_time;    // Phase end, in system-counter cycles
    // Ahead of the 32-bit fields because TaskId is 8-byte aligned: after them this
    // union would land on offset 28 and the compiler would pad it to 32, growing the
    // record past its 64-byte line.
    union {
        struct {
            uint32_t pop_hit;   // Ready-queue hit delta since the previous Dispatch
            uint32_t pop_miss;  // Ready-queue miss delta since the previous Dispatch
        } dispatch;
        // The task this phase acted on, as the handle itself rather than a raw word:
        // its fields are the owning runtime's to name, and nothing here reads them.
        // Whole handle rather than two halves -- a 64-bit id is one value, and
        // splitting it made this header claim to know the layout inside.
        TaskId task_id;
    } phase_data;
    uint32_t loop_iter;                                             // Scheduler-loop iteration on this thread
    uint32_t tasks_processed;                                       // Work items processed in this phase
    int16_t shared_depth_at_start[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];  // Ready depths at phase entry
    int16_t shared_depth_at_end[CHIP_SWIMLANE_NUM_QUEUE_SHAPES];    // Ready depths at phase exit
    // Tagged-union discriminator for phase_data, and the owning runtime's phase
    // vocabulary. Last of the named fields despite discriminating a field near the
    // top: one byte placed ahead of tasks_processed would leave three bytes the
    // alignment forces and nothing can use, where here they fall into _pad and stay
    // available. See sched_phase_kind.h for what the values mean.
    SchedPhaseKind kind;
    uint8_t _pad[19];  // Keep the wire record at 64 bytes
};

static_assert(
    sizeof(decltype(ChipSwimlaneAicpuSchedPhaseRecord::phase_data)) == 8,
    "ChipSwimlaneAicpuSchedPhaseRecord phase data must remain 8 bytes"
);
static_assert(
    offsetof(ChipSwimlaneAicpuSchedPhaseRecord, phase_data) == 16,
    "ChipSwimlaneAicpuSchedPhaseRecord phase data offset drift"
);
static_assert(sizeof(ChipSwimlaneAicpuSchedPhaseRecord) == 64, "ChipSwimlaneAicpuSchedPhaseRecord layout drift");
