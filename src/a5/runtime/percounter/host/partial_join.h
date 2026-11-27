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

#include <map>

#include "region_plan.h"

namespace simpler::percounter {

struct PartialJoinStats {
    size_t groups{};
    size_t removed_refs{};
    size_t moved_from_gm{};
    size_t removed_terminal_refs{};
};

inline size_t partial_join_slot_capacity(const std::vector<TaskInput> &tasks) {
    const size_t wide_rows = std::count_if(tasks.begin(), tasks.end(), [](const TaskInput &task) {
        return task.fanin_count >= 2 && !task.inline_completed && !task.published_off_lane && !task.has_predicate;
    });
    // A multi-input terminal frontier joins paths in the ordinary suffix.
    // Reserve one extra slot for that frontier only when a wide row exists.
    return wide_rows == 0 ? 0 : wide_rows + 1;
}

// Group members have one common observer: an ordinary consumer or the terminal
// entry frontier. Its representative carries the sum of their completion credits.
inline PartialJoinStats build_partial_joins(
    const std::vector<TaskInput> &tasks, const PercounterLayout &layout, void *host, uint64_t device,
    const std::vector<uint64_t> &lease, size_t next_slot
) {
    PartialJoinStats stats;
    if (layout.graph_invocation_count || next_slot >= lease.size()) return stats;
    const size_t n = tasks.size();
    auto *rows = region_at<PercounterFanin>(host, layout.fanin_offset);
    auto *refs = region_at<uint64_t>(host, layout.fanin_addr_offset);
    auto *counters = region_at<PercounterCounter>(host, layout.counters_offset);
    auto *control = region_at<PercounterControl>(host, layout.control_offset);
    auto &terminal = control->terminal;
    std::vector<bool> terminal_step(n), boundary(n), joined(n);
    for (uint32_t i = 0; i < terminal.step_count; ++i)
        terminal_step[terminal.steps[i]] = true;
    const auto unseen = n + 1, shared = n + 2;
    std::vector<size_t> owner(n, unseen);
    const auto *sinks = region_at<uint64_t>(host, layout.sinks_offset);
    for (uint64_t i = 0; i < control->sink_count; ++i)
        owner[sinks[i]] = shared;
    // Row n is the single entry frontier of the proven terminal schedule. Inside the
    // suffix, lanes wait directly on the preceding step and never probe its CSR row.
    const auto count = [&](size_t c) {
        if (c == n) return static_cast<int>(terminal.prerequisite_count);
        return terminal_step[c] ? 0 : rows[c].count;
    };
    const auto producer = [&](size_t c, int i) -> size_t {
        if (c == n) return terminal.prerequisites[i];
        const auto ref = refs[rows[c].begin + i];
        return table_fanin_producer_id(ref, device, layout);
    };
    const auto credits = [&](size_t p) {
        return uint64_t(tasks[p].block_num) * tasks[p].subtask_count;
    };
    for (size_t c = 0; c <= n; ++c) {
        if (c < n)
            boundary[c] =
                terminal_step[c] || tasks[c].inline_completed || tasks[c].published_off_lane || tasks[c].has_predicate;
        for (int i = 0; i < count(c); ++i) {
            auto &consumer = owner[producer(c, i)];
            if (consumer != c) consumer = consumer == unseen ? c : shared;
        }
    }
    std::vector<size_t> members;
    for (size_t c = 0; c <= n; ++c) {
        const int width = count(c);
        if ((c < n && boundary[c]) || width < 2) continue;
        members.clear();
        uint64_t total = 0;
        int last = -1;
        for (int i = 0; i < width; ++i) {
            const auto p = producer(c, i);
            if (boundary[p] || owner[p] != c) continue;
            last = i;
            if (std::find(members.begin(), members.end(), p) != members.end()) continue;
            members.push_back(p);
            total += credits(p);
        }
        if (members.size() < 2 || total > UINT32_MAX) continue;
        while (next_slot < lease.size() && !a5_cnt_notify_increment_address(lease[next_slot]))
            ++next_slot;
        if (next_slot == lease.size()) break;
        const uint64_t slot = lease[next_slot++];
        uint32_t threshold = 0;
        for (auto p : members) {
            joined[p] = true;
            stats.moved_from_gm += counters[p].completion_notify_address == 0;
            counters[p].completion_notify_address = slot;
            threshold += static_cast<uint32_t>(credits(p));
            counters[p].completion_notify_target = threshold;
        }
        // The first member resets the slot through the original head rule. The last
        // distinct member carries the full threshold even if the CSR has duplicate edges.
        const auto representative = device + layout.counters_offset + members.back() * sizeof(PercounterCounter);
        int kept = 0;
        for (int i = 0; i < width; ++i) {
            if (i != last && joined[producer(c, i)]) continue;
            if (c == n) {
                terminal.prerequisites[kept++] = i == last ? members.back() : terminal.prerequisites[i];
            } else {
                auto *edges = refs + rows[c].begin;
                edges[kept++] = i == last ? representative | PERCOUNTER_COMPLETION_NOTIFY : edges[i];
            }
        }
        ++stats.groups;
        if (c == n) {
            stats.removed_terminal_refs = width - kept;
            terminal.prerequisite_count = kept;
        } else {
            stats.removed_refs += width - kept;
            rows[c].count = kept;
        }
    }
    if (stats.groups != 0) {
        std::map<uint64_t, uint32_t> thresholds;
        for (size_t p = 0; p < n; ++p) {
            auto &counter = counters[p];
            if (joined[p] || !counter.completion_notify_address) continue;
            auto &target = thresholds[counter.completion_notify_address];
            target += static_cast<uint32_t>(credits(p));
            counter.completion_notify_target = target;
        }
    }
    return stats;
}

}  // namespace simpler::percounter
