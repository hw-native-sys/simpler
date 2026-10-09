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

#include <stdint.h>

#include "host_build_graph/graph_recording.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/runtime_types.h"

// What the Graph submit path offers the ordinary one. The traffic in this direction is
// only the two questions the ordinary path has to ask on every submission -- "is this
// thread recording a body?" and, if so, "record this task instead of submitting it" --
// because a recording is a property of the calling thread rather than of the arguments.
//
// The Graph path's own entry points are OrchestratorState::graph_begin / graph_prepare /
// graph_abort / graph_end / graph_commit, declared with the rest of the class in
// orchestrator.h and defined in host/graph_submit.cpp.

// The body this thread is recording into for `orch`, or nullptr when it is recording none.
// The owner is part of the answer: a thread holding a recording for another orchestrator
// is not recording for this one.
GraphRecording *active_graph_recording(OrchestratorState *orch);

// Record one sub-task while recording, without consuming a task-table
// slot. Builds the task's metadata and materialized outputs exactly as
// submit_task_common would, but assigns output buffers from the recording's own
// address space and derives internal fanins from each argument's owner — so
// no task-table slot, tensormap entry, fanin-pool entry, or upload is produced for
// it. The resulting Definition is later attached to the outer GRAPH shells already
// submitted by the main thread. The returned TaskOutputTensors point into the
// recording's tensor pool, which is allocated at the cap and never grows, so they
// stay valid for the rest of the recording.
TaskOutputTensors graph_record_submit_sub_task(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
);
