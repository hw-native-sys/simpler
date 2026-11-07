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
 * Where this runtime joins host_build_graph's bind.
 *
 * Its create_scheduler_state walks every task once — classifying the shape,
 * validating it, checking each kernel id, filling the SchedulerTaskMetadata the
 * device reads — and that walk is the same whoever schedules the cores. So this
 * takes its result instead of repeating it, and the only thing left to do here
 * is build the percounter region from it.
 *
 * Both definitions are strong, which drops the declining ones that translation
 * unit carries.
 */

#include "state.h"
#include "scheduler_extension.h"

// Never DECLINED: percounter is the only scheduler this runtime carries, so a
// graph it cannot run has nowhere to go and fails the bind instead.
SchedulerExtensionResult scheduler_extension_select(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    const std::vector<SchedulerTaskMetadata> &task_metadata, const std::vector<int64_t> &inline_completed_task_ids,
    const sm_layout::SegmentOffsets &device_segments
) {
    const auto selection = simpler::percounter::select_percounter_state(
        runtime, api, host_sm_handle, total_tasks, task_metadata, inline_completed_task_ids, device_segments
    );
    return selection == simpler::percounter::StateSelection::PERCOUNTER ? SchedulerExtensionResult::TAKEN :
                                                                          SchedulerExtensionResult::FAILED;
}
