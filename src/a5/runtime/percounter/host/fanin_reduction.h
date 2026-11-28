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

#include "region_plan.h"

namespace simpler::percounter {

// An ordinary producer's completion already implies completion of its inputs.
// Remove a direct wait only after following a nonempty path from another input
// of the same row. The original DAG still determines placement and visible sinks.
inline size_t
reduce_table_waits(const std::vector<TaskInput> &tasks, const PercounterLayout &layout, void *host, uint64_t device) {
    if (layout.graph_invocation_count) return 0;
    const auto n = tasks.size();
    auto *rows = region_at<PercounterFanin>(host, layout.fanin_offset);
    auto *refs = region_at<uint64_t>(host, layout.fanin_addr_offset);
    const auto &terminal = region_at<PercounterControl>(host, layout.control_offset)->terminal;
    std::vector<bool> ordinary(n);
    for (size_t p = 0; p < n; ++p)
        ordinary[p] = !tasks[p].inline_completed && !tasks[p].published_off_lane && !tasks[p].has_predicate;
    for (uint32_t i = 0; i < terminal.step_count; ++i)
        ordinary[terminal.steps[i]] = false;
    std::vector<size_t> indirect(n, n), queued(n, n), pending;
    size_t removed = 0;
    for (size_t c = 0; c < n; ++c) {
        auto &row = rows[c];
        if (!ordinary[c] || row.count < 2) continue;
        auto *edges = refs + row.begin;
        size_t earliest = c;
        pending.clear();
        const auto enqueue = [&](size_t p) {
            if (queued[p] == c) return;
            queued[p] = c;
            pending.push_back(p);
        };
        for (int i = 0; i < row.count; ++i) {
            const auto p = table_fanin_producer_id(edges[i], device, layout);
            if (ordinary[p]) {
                earliest = std::min(earliest, p);
                enqueue(p);
            }
        }
        // Exhausting the traversal budget only leaves more original waits intact.
        size_t remaining_edge_visits = 256;
        while (!pending.empty() && remaining_edge_visits) {
            const size_t p = pending.back();
            pending.pop_back();
            if (p <= earliest) continue;
            for (int e = 0; e < tasks[p].fanin_count && remaining_edge_visits; ++e) {
                --remaining_edge_visits;
                const size_t a = tasks[p].fanin_ids[e];
                if (a < earliest || !ordinary[a]) continue;
                indirect[a] = c;
                enqueue(a);
            }
        }
        int kept = 0;
        for (int i = 0; i < row.count; ++i)
            if (indirect[table_fanin_producer_id(edges[i], device, layout)] != c) edges[kept++] = edges[i];
        removed += row.count - kept;
        row.count = kept;
    }
    return removed;
}

}  // namespace simpler::percounter
