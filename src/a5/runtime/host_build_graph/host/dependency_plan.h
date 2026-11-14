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

#include <vector>

#include "scheduler/scheduler_graph.h"

inline bool scheduler_validate_dependencies(
    const SchedulerGraphAccess &graph, const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<int64_t> &completed
) {
    if (metadata.size() != static_cast<uint64_t>(graph.task_count)) return false;
    std::vector<uint8_t> active(graph.task_count, 1);
    for (int64_t ref : completed) {
        if (!scheduler_task_ref_valid(graph, ref)) return false;
        active[ref] = 0;
    }
    for (int64_t consumer = 0; consumer < graph.task_count; ++consumer) {
        if (!active[consumer]) continue;
        SchedulerFaninView view;
        if (scheduler_node_fanin_view(graph, consumer, &view) != SchedulerGraphResult::OK) return false;
        for (uint32_t edge = 0; edge < view.count; ++edge) {
            int64_t producer;
            if (scheduler_fanin_ref(view, consumer, edge, &producer) != SchedulerGraphResult::OK) return false;
        }
    }
    return true;
}
