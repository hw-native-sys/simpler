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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <set>
#include <utility>
#include <vector>

// Rows contain validated topological task IDs. Active control rows have no
// side effects and can be folded into a single live consumer. Folding a
// terminal control leaves its remaining live predecessors to reach completion.
template <typename TaskRef>
inline std::vector<int64_t> scheduler_fold_control_fanin(
    std::vector<std::vector<TaskRef>> &fanin, std::vector<bool> active, const std::vector<bool> &controls,
    size_t max_fanin
) {
    const size_t count = fanin.size();
    std::vector<bool> queued(count, false);
    std::deque<uint32_t> pending;
    for (size_t ref = 0; ref < count; ++ref) {
        if (!active[ref] || !controls[ref]) continue;
        queued[ref] = true;
        pending.push_back(static_cast<uint32_t>(ref));
    }
    if (pending.empty()) return {};
    std::vector<std::set<uint32_t>> consumers(count);
    for (size_t ref = 0; ref < count; ++ref) {
        if (!active[ref]) continue;
        for (const TaskRef producer : fanin[ref])
            consumers[producer].insert(static_cast<uint32_t>(ref));
    }
    std::vector<int64_t> folded;
    const auto enqueue = [&](uint32_t ref) {
        if (active[ref] && controls[ref] && !queued[ref]) {
            queued[ref] = true;
            pending.push_back(ref);
        }
    };
    while (!pending.empty()) {
        const uint32_t ref = pending.front();
        pending.pop_front();
        queued[ref] = false;
        if (!active[ref] || consumers[ref].size() > 1) continue;
        const uint32_t consumer = consumers[ref].empty() ? UINT32_MAX : *consumers[ref].begin();
        std::vector<TaskRef> merged;
        if (consumer != UINT32_MAX) {
            for (const TaskRef producer : fanin[consumer])
                if (producer != static_cast<TaskRef>(ref)) merged.push_back(producer);
            merged.insert(merged.end(), fanin[ref].begin(), fanin[ref].end());
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            if (merged.size() > max_fanin) continue;
        }
        std::vector<TaskRef> affected = fanin[ref];
        for (const TaskRef producer : fanin[ref])
            consumers[producer].erase(ref);
        if (consumer != UINT32_MAX) {
            affected.insert(affected.end(), fanin[consumer].begin(), fanin[consumer].end());
            for (const TaskRef producer : fanin[consumer])
                consumers[producer].erase(consumer);
            fanin[consumer] = std::move(merged);
            for (const TaskRef producer : fanin[consumer])
                consumers[producer].insert(consumer);
            enqueue(consumer);
        }
        active[ref] = false;
        fanin[ref].clear();
        folded.push_back(ref);
        for (const TaskRef producer : affected)
            enqueue(static_cast<uint32_t>(producer));
    }
    return folded;
}
