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
#include <numeric>
#include <vector>

#include "region_plan.h"
#include "scheduler/calendar_wire.h"

namespace simpler::percounter::range_calendar_detail {

struct CalendarFaninAssignment {
    uint16_t begin{0};
    uint16_t count{0};
};

struct CalendarWaitPlan {
    std::vector<std::vector<int32_t>> task_fanins;
    std::vector<bool> row_waits;
    std::vector<bool> summarized_fanins;
    std::vector<CalendarFaninAssignment> row_assignments;
};

inline bool calendar_event_order(
    const std::vector<std::vector<size_t>> &parents, std::vector<size_t> &order, std::vector<size_t> &ranks
) {
    std::vector<std::vector<size_t>> successors(parents.size());
    std::vector<size_t> pending(parents.size());
    order.reserve(parents.size());
    for (size_t node = 0; node < parents.size(); ++node) {
        pending[node] = parents[node].size();
        if (pending[node] == 0) order.push_back(node);
        for (const size_t parent : parents[node])
            successors[parent].push_back(node);
    }
    ranks.resize(parents.size());
    for (size_t i = 0; i < order.size(); ++i) {
        const size_t node = order[i];
        ranks[node] = i;
        for (const size_t successor : successors[node])
            if (--pending[successor] == 0) order.push_back(successor);
    }
    return order.size() == parents.size();
}

inline bool plan_calendar_waits(
    const std::vector<TaskInput> &tasks, const std::vector<CalendarRow> &rows, const std::vector<uint32_t> &indexes,
    size_t join_count, CalendarWaitPlan &result
) {
    const size_t row_count = rows.size();
    if (indexes.size() < 2 || indexes.front() != 0 || indexes.back() != row_count ||
        !std::is_sorted(indexes.begin(), indexes.end()))
        return false;
    const size_t gate_base = row_count;
    const size_t done_base = gate_base + row_count + join_count;
    const size_t task_count = tasks.size();
    const size_t cohort_base = done_base + task_count;
    const size_t event_count = cohort_base + task_count;
    const size_t worker_count = indexes.size() - 1;
    constexpr size_t max_prefix_bytes = 64 * 1024 * 1024;
    const bool use_prefixes = event_count <= max_prefix_bytes / sizeof(uint32_t) / worker_count;

    std::vector<std::vector<size_t>> parents(use_prefixes ? event_count : 0);
    std::vector<uint32_t> row_workers(use_prefixes ? row_count : 0);
    std::vector<std::vector<uint32_t>> last_rows(use_prefixes ? task_count : 0);
    std::vector<size_t> gate_tasks(row_count + join_count, task_count);
    const auto gate_of = [&](size_t row) -> size_t {
        return gate_base + (rows[row].local_join == CALENDAR_NO_JOIN ? row : row_count + rows[row].local_join);
    };
    for (size_t worker = 0; worker < worker_count; ++worker) {
        const size_t begin = indexes[worker];
        for (size_t row = begin; row < indexes[worker + 1]; ++row) {
            const auto id = percounter_order_task_id(rows[row].entry);
            if (id < 0 || static_cast<size_t>(id) >= task_count ||
                (rows[row].local_join != CALENDAR_NO_JOIN && rows[row].local_join >= join_count))
                return false;
            const auto task = static_cast<size_t>(id);
            const size_t gate = gate_of(row);
            auto &owner = gate_tasks[gate - gate_base];
            if (owner != task_count && owner != task) return false;
            owner = task;
            if (!use_prefixes) continue;

            row_workers[row] = static_cast<uint32_t>(worker);
            auto &last = last_rows[task];
            if (!last.empty() && row_workers[last.back()] == worker) {
                last.back() = static_cast<uint32_t>(row);
            } else {
                last.push_back(static_cast<uint32_t>(row));
            }
            parents[done_base + task].push_back(row);
            parents[row].push_back(gate);
            if (row != begin) {
                const size_t previous = row - 1;
                parents[gate].push_back(previous);
                if (tasks[task].sync_start) parents[cohort_base + task].push_back(previous);
            }
        }
    }

    result.task_fanins.resize(task_count);
    result.row_waits.resize(row_count);
    result.summarized_fanins.resize(task_count);
    result.row_assignments.resize(row_count);
    if (!use_prefixes) {
        for (size_t task = 0; task < task_count; ++task) {
            const auto &input = tasks[task];
            for (int32_t i = 0; i < input.fanin_count; ++i) {
                const auto &producer = tasks[input.fanin_ids[i]];
                if (!producer.inline_completed) result.task_fanins[task].push_back(input.fanin_ids[i]);
            }
        }
        for (size_t row = 0; row < row_count; ++row)
            result.row_waits[row] = !result.task_fanins[gate_tasks[gate_of(row) - gate_base]].empty();
        return true;
    }

    for (size_t gate = gate_base; gate < done_base; ++gate) {
        const size_t task = gate_tasks[gate - gate_base];
        if (task == task_count) continue;
        const auto &input = tasks[task];
        if (input.sync_start) parents[gate].push_back(cohort_base + task);
        for (int32_t edge = 0; edge < input.fanin_count; ++edge) {
            const auto pred = static_cast<size_t>(input.fanin_ids[edge]);
            if (!tasks[pred].inline_completed) parents[gate].push_back(done_base + pred);
        }
    }

    std::vector<size_t> event_order, ranks;
    if (!calendar_event_order(parents, event_order, ranks)) return false;
    std::vector<std::vector<size_t>> ordered_inputs(task_count);
    std::vector<std::vector<bool>> kept(task_count);
    for (size_t task = 0; task < task_count; ++task) {
        const auto &input = tasks[task];
        auto &order = ordered_inputs[task];
        order.resize(input.fanin_count);
        kept[task].resize(input.fanin_count);
        std::iota(order.begin(), order.end(), size_t{0});
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return ranks[done_base + input.fanin_ids[a]] > ranks[done_base + input.fanin_ids[b]];
        });
    }

    // An event's prefix contains the latest row that it orders on every
    // worker. Earlier rows on that worker are part of the same prefix.
    std::vector<uint32_t> prefixes(parents.size() * worker_count, 0);
    const auto prefix = [&](size_t node) {
        return prefixes.data() + node * worker_count;
    };
    const auto merge = [&](uint32_t *dst, size_t node) {
        const uint32_t *src = prefix(node);
        for (size_t worker = 0; worker < worker_count; ++worker)
            dst[worker] = std::max(dst[worker], src[worker]);
    };

    std::vector<bool> gate_waits(gate_tasks.size(), false);
    for (const size_t node : event_order) {
        auto *ordered = prefix(node);
        if (node < gate_base || node >= done_base) {
            for (const size_t parent : parents[node])
                merge(ordered, parent);
            if (node < gate_base) ordered[row_workers[node]] = static_cast<uint32_t>(node + 1);
            continue;
        }

        const size_t id = gate_tasks[node - gate_base];
        if (id == task_count) continue;
        // START requires every participating worker's previous row to
        // retire before any block enters its kernel. Its resource prefix
        // therefore already covers any input completed in that prefix.
        for (const size_t parent : parents[node]) {
            if (parent < gate_base || parent >= cohort_base) merge(ordered, parent);
        }
        for (const size_t edge : ordered_inputs[id]) {
            const auto pred = static_cast<size_t>(tasks[id].fanin_ids[edge]);
            if (tasks[pred].inline_completed) continue;
            const auto &required = last_rows[pred];
            // A deferred row can retire before its completion is published.
            if (tasks[pred].active_mask == 0 && !required.empty() &&
                std::all_of(required.begin(), required.end(), [&](uint32_t row) {
                    return ordered[row_workers[row]] >= row + 1;
                }))
                continue;
            kept[id][edge] = true;
            gate_waits[node - gate_base] = true;
            merge(ordered, done_base + pred);
        }
    }

    for (size_t task = 0; task < task_count; ++task) {
        for (size_t edge = 0; edge < kept[task].size(); ++edge) {
            if (kept[task][edge]) result.task_fanins[task].push_back(tasks[task].fanin_ids[edge]);
        }
    }
    std::vector<size_t> first_gate(task_count, parents.size());
    for (size_t row = 0; row < row_count; ++row) {
        const size_t gate = gate_of(row);
        result.row_waits[row] = gate_waits[gate - gate_base];
        if (!result.row_waits[row]) continue;
        const auto task = gate_tasks[gate - gate_base];
        first_gate[task] = std::min(first_gate[task], ranks[gate]);
    }

    std::vector<std::vector<uint32_t>> pollers(task_count);
    std::vector<size_t> last_worker(task_count, worker_count);
    for (size_t row = 0; row < row_count; ++row) {
        if (!result.row_waits[row]) continue;
        const auto task = static_cast<size_t>(percounter_order_task_id(rows[row].entry));
        const auto &shape = tasks[task];
        if (shape.is_mix || shape.sync_start || result.task_fanins[task].size() < 2) continue;
        const size_t worker = row_workers[row];
        if (last_worker[task] == worker) continue;
        last_worker[task] = worker;
        // The helper reaches its wait after its preceding row. Adding that
        // row as a prerequisite of every consumer gate retains event order.
        if (row == indexes[worker] || ranks[row - 1] < first_gate[task])
            pollers[task].push_back(static_cast<uint32_t>(row));
    }
    for (size_t task = 0; task < task_count; ++task) {
        const size_t fanin = result.task_fanins[task].size();
        const size_t count = std::min(fanin, pollers[task].size());
        if (count < 2 || fanin > count * CALENDAR_FANIN_TASKS_PER_LANE) continue;
        result.summarized_fanins[task] = true;
        for (size_t i = 0; i < count; ++i) {
            const size_t begin = i * fanin / count;
            const size_t end = (i + 1) * fanin / count;
            result.row_assignments[pollers[task][i]] = {
                static_cast<uint16_t>(begin), static_cast<uint16_t>(end - begin)
            };
        }
    }
    return true;
}

}  // namespace simpler::percounter::range_calendar_detail
