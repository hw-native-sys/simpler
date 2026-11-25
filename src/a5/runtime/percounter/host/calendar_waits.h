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
#include <cstddef>
#include <cstdint>
#include <limits>
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
    std::vector<uint32_t> row_input_owners;
    // Total credits for each eligible fused fan-in, or zero for ordinary waits.
    std::vector<uint32_t> fused_fanin_credits;
};

inline constexpr size_t calendar_wait_reach_width = std::numeric_limits<uint64_t>::digits;

// A bit at distance d - 1 denotes a DONE dependency already ordered by another direct predecessor.
inline bool calendar_transitive_waits(const std::vector<TaskInput> &tasks, std::vector<uint64_t> &covered) {
    std::array<uint64_t, calendar_wait_reach_width> reachable{};
    covered.resize(tasks.size(), 0);
    for (size_t task = 0; task < tasks.size(); ++task) {
        const auto &input = tasks[task];
        if (input.fanin_count < 0 || (input.fanin_count != 0 && input.fanin_ids == nullptr)) return false;
        uint64_t direct = 0;
        uint64_t via = 0;
        for (int32_t edge = 0; edge < input.fanin_count; ++edge) {
            const int32_t pred = input.fanin_ids[edge];
            if (pred < 0 || static_cast<size_t>(pred) >= task) return false;
            if (input.inline_completed || tasks[pred].inline_completed) continue;
            const size_t distance = task - static_cast<size_t>(pred);
            if (distance > calendar_wait_reach_width) continue;
            direct |= uint64_t{1} << (distance - 1);
            if (distance < calendar_wait_reach_width) {
                via |= reachable[pred % calendar_wait_reach_width] << distance;
            }
        }
        covered[task] = via;
        reachable[task % calendar_wait_reach_width] = direct | via;
    }
    return true;
}

// Each selected chain is a real directed path. Reaching a later node
// on a chain proves that all earlier nodes on that chain have completed.
inline constexpr size_t calendar_wait_chain_count = 8;
struct CalendarWaitChains {
    static constexpr uint32_t unassigned = UINT32_MAX;
    using Positions = std::array<uint32_t, calendar_wait_chain_count>;
    std::vector<uint32_t> chain_by_task;
    std::vector<Positions> via_parent;

    explicit CalendarWaitChains(const std::vector<TaskInput> &tasks) :
        chain_by_task(tasks.size(), unassigned),
        via_parent(tasks.size()) {
        std::vector<size_t> weight(tasks.size(), 1);
        for (const auto &task : tasks) {
            if (task.inline_completed || task.fanin_count < 2) continue;
            for (int32_t edge = 0; edge < task.fanin_count; ++edge)
                ++weight[task.fanin_ids[edge]];
        }

        // Greedily cover the paths with the most potential repeated waits.
        std::vector<size_t> length(tasks.size());
        std::vector<uint32_t> previous(tasks.size());
        for (uint32_t chain = 0; chain < calendar_wait_chain_count; ++chain) {
            size_t best = 0;
            uint32_t end = unassigned;
            for (uint32_t id = 0; id < tasks.size(); ++id) {
                if (tasks[id].inline_completed || chain_by_task[id] != unassigned) continue;
                size_t longest = 0;
                previous[id] = unassigned;
                for (int32_t edge = 0; edge < tasks[id].fanin_count; ++edge) {
                    const auto parent = static_cast<uint32_t>(tasks[id].fanin_ids[edge]);
                    if (!tasks[parent].inline_completed && chain_by_task[parent] == unassigned &&
                        length[parent] > longest) {
                        longest = length[parent];
                        previous[id] = parent;
                    }
                }
                length[id] = longest + weight[id];
                if (length[id] > best) {
                    best = length[id];
                    end = id;
                }
            }
            if (best < 2) break;
            for (uint32_t id = end; id != unassigned; id = previous[id])
                chain_by_task[id] = chain;
        }

        std::vector<Positions> ancestor(tasks.size());
        for (uint32_t id = 0; id < tasks.size(); ++id) {
            const auto &task = tasks[id];
            if (task.inline_completed) continue;
            // Merge strict ancestors before adding the direct producers.
            // A wait is removable only when reached through another input.
            auto &via = via_parent[id];
            for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
                const auto parent = static_cast<uint32_t>(task.fanin_ids[edge]);
                for (size_t chain = 0; chain < calendar_wait_chain_count; ++chain)
                    via[chain] = std::max(via[chain], ancestor[parent][chain]);
            }
            auto &strict = ancestor[id];
            strict = via;
            for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
                const auto parent = static_cast<uint32_t>(task.fanin_ids[edge]);
                const auto chain = chain_by_task[parent];
                if (chain != unassigned) strict[chain] = std::max(strict[chain], parent + 1U);
            }
        }
    }

    bool orders(size_t task, size_t predecessor) const {
        const auto chain = chain_by_task[predecessor];
        return chain != unassigned && via_parent[task][chain] >= predecessor + 1;
    }
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

inline void plan_fused_fanin_credits(
    const std::vector<TaskInput> &tasks, const std::vector<std::vector<uint32_t>> &pollers,
    const std::vector<size_t> &first_gate, size_t no_gate, CalendarWaitPlan &plan
) {
    const size_t count = tasks.size();
    std::vector<bool> consumed(count);
    std::vector<uint32_t> observers(count);
    uint64_t edges = 0;
    for (size_t task = 0; task < count; ++task) {
        const auto &input = tasks[task];
        edges += input.fanin_count;
        for (int32_t edge = 0; edge < input.fanin_count; ++edge)
            consumed[input.fanin_ids[edge]] = true;
        if (first_gate[task] != no_gate) {
            for (int32_t producer : plan.task_fanins[task])
                ++observers[producer];
        }
    }

    // Bound repeated walks over shared ancestors by the phase's edge count.
    uint64_t remaining = std::min(edges, UINT64_MAX / 16) * 16;
    std::vector<uint32_t> pending;
    std::vector<uint32_t> visited(count, UINT32_MAX);
    const auto independent = [&](const std::vector<int32_t> &producers) {
        const auto earliest = *std::min_element(producers.begin(), producers.end());
        for (int32_t producer : producers) {
            pending.assign(1, producer);
            while (!pending.empty()) {
                const auto &task = tasks[pending.back()];
                pending.pop_back();
                for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
                    if (remaining == 0) return false;
                    --remaining;
                    const int32_t pred = task.fanin_ids[edge];
                    if (pred < earliest || visited[pred] == static_cast<uint32_t>(producer)) continue;
                    if (std::find(producers.begin(), producers.end(), pred) != producers.end()) return false;
                    visited[pred] = producer;
                    pending.push_back(pred);
                }
            }
        }
        return true;
    };

    plan.fused_fanin_credits.assign(count, 0);
    for (size_t consumer = 0; consumer < count; ++consumer) {
        const auto &producers = plan.task_fanins[consumer];
        if (!consumed[consumer] || !plan.summarized_fanins[consumer] ||
            producers.size() > CALENDAR_FUSED_FANIN_MAX_PRODUCERS || pollers[consumer].size() < producers.size())
            continue;
        uint32_t credits = 0;
        for (int32_t producer : producers) {
            const auto &shape = tasks[producer];
            const uint64_t contribution = uint64_t{static_cast<uint32_t>(shape.block_num)} * shape.subtask_count;
            if (observers[producer] != 1 || !consumed[producer] || shape.active_mask == 0 || contribution == 0 ||
                contribution > CALENDAR_COUNT_MASK - credits) {
                credits = 0;
                break;
            }
            credits += static_cast<uint32_t>(contribution);
        }
        if (credits != 0 && independent(producers)) plan.fused_fanin_credits[consumer] = credits;
    }
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
    std::vector<uint64_t> covered;
    if (!calendar_transitive_waits(tasks, covered)) return false;
    const CalendarWaitChains chains(tasks);
    const auto needs_done = [&](size_t task, size_t pred) {
        if (tasks[pred].inline_completed || chains.orders(task, pred)) return false;
        const size_t distance = task - pred;
        return distance > calendar_wait_reach_width || (covered[task] & (uint64_t{1} << (distance - 1))) == 0;
    };

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
    result.row_waits.assign(row_count, false);
    result.summarized_fanins.resize(task_count);
    result.row_assignments.resize(row_count);
    result.row_input_owners.assign(row_count, CALENDAR_NO_INPUT_OWNER);
    result.fused_fanin_credits.clear();
    if (!use_prefixes) {
        for (size_t task = 0; task < task_count; ++task) {
            const auto &input = tasks[task];
            for (int32_t i = 0; i < input.fanin_count; ++i) {
                const auto pred = input.fanin_ids[i];
                if (needs_done(task, pred)) result.task_fanins[task].push_back(pred);
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

    // The resource proof can only discharge dependency-only producers.
    // With at most one machine word of such producers, propagate just their
    // retirements instead of maintaining a wide prefix in every lane.
    std::vector<uint64_t> control_bits(task_count);
    size_t control_count = 0;
    bool compact_proof = true;
    for (size_t task = 0; task < task_count; ++task) {
        if (tasks[task].active_mask != 0 || last_rows[task].empty()) continue;
        if (last_rows[task].size() != 1 || control_count == 64) {
            compact_proof = false;
            break;
        }
        control_bits[task] = uint64_t{1} << control_count++;
    }

    std::vector<uint64_t> known_controls(compact_proof ? parents.size() : 0, 0);
    std::vector<uint32_t> prefixes(compact_proof ? 0 : parents.size() * worker_count, 0);
    const auto prefix = [&](size_t node) {
        return prefixes.data() + node * worker_count;
    };
    const auto merge = [&](size_t node, size_t parent) {
        if (compact_proof) {
            known_controls[node] |= known_controls[parent];
            return;
        }
        auto *dst = prefix(node);
        const auto *src = prefix(parent);
        for (size_t worker = 0; worker < worker_count; ++worker)
            dst[worker] = std::max(dst[worker], src[worker]);
    };

    std::vector<bool> gate_waits(gate_tasks.size(), false);
    for (const size_t node : event_order) {
        if (node < gate_base || node >= done_base) {
            for (const size_t parent : parents[node])
                merge(node, parent);
            if (node < gate_base) {
                if (compact_proof) {
                    const auto task = static_cast<size_t>(percounter_order_task_id(rows[node].entry));
                    known_controls[node] |= control_bits[task];
                } else {
                    prefix(node)[row_workers[node]] = static_cast<uint32_t>(node + 1);
                }
            }
            continue;
        }

        const size_t id = gate_tasks[node - gate_base];
        if (id == task_count) continue;
        // The cohort edges validate progress if blocks wait inside their
        // kernels. Only the local gate's prefix is guaranteed before entry.
        for (const size_t parent : parents[node]) {
            if (parent < gate_base) merge(node, parent);
        }
        for (const size_t edge : ordered_inputs[id]) {
            const auto pred = static_cast<size_t>(tasks[id].fanin_ids[edge]);
            if (!needs_done(id, pred)) continue;
            const auto &required = last_rows[pred];
            // A deferred row can retire before its completion is published.
            if (tasks[pred].active_mask == 0 && !required.empty()) {
                const bool retired = compact_proof ? (known_controls[node] & control_bits[pred]) != 0 :
                                                     std::all_of(required.begin(), required.end(), [&](uint32_t row) {
                                                         return prefix(node)[row_workers[row]] >= row + 1;
                                                     });
                if (retired) continue;
            }
            kept[id][edge] = true;
            gate_waits[node - gate_base] = true;
            merge(node, done_base + pred);
        }
    }

    for (size_t task = 0; task < task_count; ++task) {
        for (size_t edge = 0; edge < kept[task].size(); ++edge) {
            if (kept[task][edge]) result.task_fanins[task].push_back(tasks[task].fanin_ids[edge]);
        }
    }
    std::vector<size_t> first_gate(task_count, parents.size());
    std::vector<size_t> observed(task_count, worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        for (size_t row = indexes[worker]; row < indexes[worker + 1]; ++row) {
            const size_t gate = gate_of(row);
            if (!gate_waits[gate - gate_base]) continue;
            const auto task = gate_tasks[gate - gate_base];
            const auto &shape = tasks[task];
            const auto &inputs = result.task_fanins[task];
            // A prior ordinary row only retires after observing its inputs.
            // Reuse that proof on this same lane. A predicated row need not wait.
            if (!shape.is_mix && !shape.sync_start && !shape.has_predicate) {
                const bool already_observed = std::all_of(inputs.begin(), inputs.end(), [&](int32_t parent) {
                    return observed[parent] == worker;
                });
                if (already_observed) continue;
                for (const auto parent : inputs)
                    observed[parent] = worker;
            }
            result.row_waits[row] = true;
            first_gate[task] = std::min(first_gate[task], ranks[gate]);
        }
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
    plan_fused_fanin_credits(tasks, pollers, first_gate, parents.size(), result);
    // Calendar workers are the AIC lanes followed by adjacent pairs of AIV
    // lanes. A second block of the same ordinary task can acquire the first
    // block's DONE or fan-in observation through the cluster's release token.
    if (worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0) return true;
    const size_t cluster_count = worker_count / PLATFORM_CORES_PER_BLOCKDIM;
    std::vector<uint32_t> first_lane_row(task_count, CALENDAR_NO_INPUT_OWNER);
    const auto candidate = [&](size_t row) {
        const auto task = static_cast<size_t>(percounter_order_task_id(rows[row].entry));
        const auto &shape = tasks[task];
        return result.row_waits[row] && shape.core_type_index == 1 && !shape.is_mix && !shape.sync_start &&
               !shape.has_predicate && shape.active_mask != 0 && rows[row].local_join == CALENDAR_NO_JOIN &&
               (result.task_fanins[task].size() == 1 || result.summarized_fanins[task]);
    };
    for (size_t worker = cluster_count; worker < worker_count; worker += 2) {
        for (uint32_t row = indexes[worker]; row < indexes[worker + 1]; ++row) {
            if (!candidate(row)) continue;
            const auto task = percounter_order_task_id(rows[row].entry);
            if (first_lane_row[task] == CALENDAR_NO_INPUT_OWNER) first_lane_row[task] = row;
        }
        for (uint32_t row = indexes[worker + 1]; row < indexes[worker + 2]; ++row) {
            if (!candidate(row)) continue;
            const auto task = percounter_order_task_id(rows[row].entry);
            auto &other = first_lane_row[task];
            if (other == CALENDAR_NO_INPUT_OWNER) continue;
            // This extra edge follows the existing topological order, so
            // it also composes with the edges from summarized fan-in helpers.
            const uint32_t owner = ranks[gate_of(other)] < ranks[gate_of(row)] ? other : row;
            const uint32_t follower = owner == row ? other : row;
            // A helper must keep running its original resolver loop.
            if (result.row_assignments[follower].count == 0) {
                result.row_input_owners[owner] = owner;
                result.row_input_owners[follower] = owner;
            }
            other = CALENDAR_NO_INPUT_OWNER;
        }
        for (uint32_t row = indexes[worker]; row < indexes[worker + 1]; ++row) {
            const auto task = percounter_order_task_id(rows[row].entry);
            first_lane_row[task] = CALENDAR_NO_INPUT_OWNER;
        }
    }
    return true;
}

}  // namespace simpler::percounter::range_calendar_detail
