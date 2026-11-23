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

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "scheduler/scheduler_task_catalog.h"

// At most one ordinary task moves behind each wide SPMD. All decisions use the
// initial topological ranks, including the check against every crossed consumer.
inline void scheduler_defer_side_tasks(
    const std::vector<std::vector<int64_t>> &successors, const std::vector<SchedulerTaskMetadata> &metadata,
    uint32_t clusters, const std::vector<uint32_t> &levels, const std::vector<uint32_t> &bottom_levels,
    std::vector<int64_t> &order, std::vector<uint32_t> &ranks
) {
    constexpr uint32_t aic = static_cast<uint32_t>(SchedulerExecutionEngine::AIC);
    constexpr uint32_t aiv = static_cast<uint32_t>(SchedulerExecutionEngine::AIV);
    const std::array<uint32_t, 2> capacity{clusters, clusters * PLATFORM_AIV_CORES_PER_BLOCKDIM};
    const auto count = static_cast<uint32_t>(order.size());
    struct Target {
        uint32_t rank;
        uint32_t width;
    };
    const Target none{count, 0};
    std::array<Target, 2> nearest{none, none};
    std::vector<int64_t> after;

    for (uint32_t rank = count; rank-- > 0;) {
        const int64_t ref = order[rank];
        const auto &task = metadata[ref];
        if (!scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags)) continue;
        const bool cooperative = scheduler_task_is_mix(task.flags) || scheduler_task_requires_sync_start(task.flags);
        if (cooperative) nearest.fill(none);
        const auto lanes = task.active_mask;
        const uint32_t blocks = task.logical_block_num;
        const uint32_t vector_lanes = static_cast<uint32_t>(lanes.count()) - lanes.has_aic();
        // A MIX block uses named lanes in one cluster. A pure AIV block can use either vector lane.
        const uint32_t active_blocks = scheduler_task_is_mix(task.flags) ? std::min(blocks, clusters) : blocks;
        const std::array<uint32_t, 2> demand{lanes.has_aic() ? active_blocks : 0, active_blocks * vector_lanes};
        if (!cooperative) {
            const uint32_t engine = lanes.has_aic() ? aic : aiv;
            const Target target = nearest[engine];
            const uint32_t side_width = demand[engine];
            if (target.rank < count && 2 * side_width < capacity[engine] &&
                side_width + target.width > capacity[engine]) {
                const int64_t wide = order[target.rank];
                const bool has_slack = levels[wide] - levels[ref] <= 1 && bottom_levels[ref] < bottom_levels[wide];
                if (has_slack && (after.empty() || after[target.rank] < 0) &&
                    std::none_of(successors[ref].begin(), successors[ref].end(), [&](int64_t child) {
                        return ranks[child] <= target.rank;
                    })) {
                    if (after.empty()) after.assign(count, -1);
                    after[target.rank] = ref;
                    order[rank] = -1;
                }
            }
        }
        if (!scheduler_task_is_spmd(task.flags)) continue;
        for (uint32_t engine : {aic, aiv}) {
            if (2 * demand[engine] >= capacity[engine])
                nearest[engine] = {rank, std::min(demand[engine], capacity[engine])};
        }
    }

    if (after.empty()) return;
    uint32_t write = 0;
    for (uint32_t rank = 0; rank < count; ++rank) {
        const int64_t ref = order[rank];
        if (ref >= 0) order[write++] = ref;
        if (after[rank] >= 0) order[write++] = after[rank];
    }
    for (uint32_t rank = 0; rank < count; ++rank)
        ranks[order[rank]] = rank;
}
