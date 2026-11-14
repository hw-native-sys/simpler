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
 * Where a runtime that schedules AICore differently joins this one's bind.
 *
 * create_scheduler_state walks every task once: it classifies the shape,
 * validates it, checks each kernel id against the callable table, and fills the
 * SchedulerTaskMetadata the device reads. That walk is the same whoever
 * schedules the cores, so an alternative scheduler takes its result rather than
 * repeating it — this declaration is the whole of what it needs.
 *
 * It is defined weak and declines by default, so a build that links no
 * alternative behaves exactly as it did before it existed. A build that does
 * link one carries its strong definition instead.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "host_build_graph/shared_memory.h"
#include "scheduler/scheduler_types.h"

class Runtime;
struct HostApi;
namespace sm_layout {
struct SegmentOffsets;
}

enum class SchedulerExtensionResult : uint8_t {
    DECLINED,  // not this scheduler's graph; the default selection decides it
    TAKEN,     // the region is built and published, and the bind is done
    FAILED,    // this graph was accepted but building its region failed
};

/**
 * Offered the walk's result, after it has rejected any malformed task and
 * before this runtime expands its own native Graph nodes, so GRAPH tasks reach
 * an extension as the shells the task table holds. An extension that takes the
 * graph owns the rest of the bind, including reading those shells' Definitions.
 */
// Hidden, not weak: the weak marker belongs on this runtime's definition, so
// that an extension's strong one wins at static link time. Visibility keeps
// that resolution inside one shared object — two runtimes' host images can
// share a process, and a default-visibility weak symbol is preemptible, so
// this one would otherwise bind to whichever image defined it first and hand
// the graph to a scheduler the loaded AICore image cannot run.
__attribute__((visibility("hidden"))) SchedulerExtensionResult scheduler_extension_select(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    const std::vector<SchedulerTaskMetadata> &task_metadata, const std::vector<int64_t> &inline_completed_task_ids,
    const sm_layout::SegmentOffsets &device_segments
);

__attribute__((visibility("hidden"))) bool scheduler_extension_publish_profiling(Runtime *runtime);
__attribute__((visibility("hidden"))) void scheduler_extension_release(Runtime *runtime);
