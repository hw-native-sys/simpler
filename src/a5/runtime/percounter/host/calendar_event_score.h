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

#include "calendar_plan.h"

namespace simpler::percounter {

// All intervals are in cycles. The row sample supplies occupancy and the observed defer tail.
struct CalendarEventCosts {
    // Previous row retirement to reaching the wait for a new row.
    uint64_t row_prepare_cycles{0};
    uint64_t direct_observe_cycles{0};
    uint64_t fanin_credit_cycles{0};
    uint64_t fanin_observe_cycles{0};
    uint64_t mix_arrival_cycles{0};
    uint64_t mix_release_cycles{0};
    uint64_t mix_follower_observe_cycles{0};
    uint64_t entry_cycles{0};
    // Logical output readiness to publishing DONE, including on a deferred path.
    uint64_t done_publish_cycles{0};
    // End of occupancy to lane retirement, including any on-lane publication.
    uint64_t retire_after_occupancy_cycles{0};
    // Estimated cost of an SSBUF polling iteration between fallback probes.
    uint64_t input_follower_poll_cycles{1};
};

namespace calendar_detail {

struct FaninHelper {
    size_t arrival{};
    std::array<size_t, CALENDAR_FANIN_TASKS_PER_LANE> producers{};
    size_t count{};
};

struct InputFollower {
    size_t arrival;
    size_t owner_release;
    size_t done;
    bool summarized;
};

class CalendarEventGraph {
public:
    explicit CalendarEventGraph(uint32_t rows) {
        events_.reserve(static_cast<size_t>(rows) * 7);
        edges_.reserve(static_cast<size_t>(rows) * 12);
    }

    size_t add_event() {
        const size_t index = events_.size();
        events_.emplace_back();
        return index;
    }

    void link(size_t from, size_t to, uint64_t cost = 0) {
        edges_.push_back({to, cost, events_[from].first});
        events_[from].first = edges_.size() - 1;
        ++events_[to].pending;
    }

    void add_helper(const FaninHelper &helper, size_t summary) {
        const size_t index = add_event();
        events_[index].helper = helpers_.size();
        helpers_.push_back(helper);
        link(helper.arrival, index);
        for (size_t i = 0; i < helper.count; ++i) {
            link(helper.producers[i], index);
        }
        link(index, summary);
    }

    void add_follower(const InputFollower &follower, size_t ready) {
        events_[ready].follower = followers_.size();
        followers_.push_back(follower);
        link(follower.arrival, ready);
        link(follower.owner_release, ready);
        link(follower.done, ready);
    }

    bool replay(const CalendarEventCosts &costs, uint64_t &makespan) {
        std::vector<size_t> ready;
        ready.reserve(events_.size());
        for (size_t i = 0; i < events_.size(); ++i) {
            if (events_[i].pending == 0) ready.push_back(i);
        }
        uint64_t last = 0;
        for (size_t cursor = 0; cursor < ready.size(); ++cursor) {
            auto &event = events_[ready[cursor]];
            if (event.helper != no_event &&
                !finish_helper(helpers_[event.helper], costs.fanin_credit_cycles, event.cycles))
                return false;
            if (event.follower != no_event && !finish_follower(followers_[event.follower], costs, event.cycles))
                return false;
            last = std::max(last, event.cycles);
            for (size_t index = event.first; index != no_edge; index = edges_[index].next) {
                const auto &edge = edges_[index];
                if (edge.lag > UINT64_MAX - event.cycles) return false;
                auto &next = events_[edge.successor];
                next.cycles = std::max(next.cycles, event.cycles + edge.lag);
                if (--next.pending == 0) ready.push_back(edge.successor);
            }
        }
        if (ready.size() != events_.size()) return false;
        makespan = last;
        return true;
    }

private:
    static constexpr size_t no_event = SIZE_MAX;
    static constexpr size_t no_edge = SIZE_MAX;
    struct CalendarEventEdge {
        size_t successor;
        uint64_t lag;
        size_t next;
    };
    struct Event {
        size_t first{no_edge};
        size_t pending{0};
        uint64_t cycles{0};
        size_t helper{no_event};
        size_t follower{no_event};
    };
    std::vector<Event> events_;
    std::vector<CalendarEventEdge> edges_;
    std::vector<FaninHelper> helpers_;
    std::vector<InputFollower> followers_;

    bool finish_follower(const InputFollower &follower, const CalendarEventCosts &costs, uint64_t &cycles) const {
        const uint64_t arrival = events_[follower.arrival].cycles;
        const uint64_t owner = std::max(arrival, events_[follower.owner_release].cycles);
        if (costs.mix_follower_observe_cycles > UINT64_MAX - owner) return false;
        cycles = owner + costs.mix_follower_observe_cycles;

        const uint64_t interval =
            follower.summarized ? CALENDAR_FOLLOWER_FANIN_INTERVAL : CALENDAR_FOLLOWER_DONE_INTERVAL;
        const uint64_t probe = follower.summarized ? costs.fanin_observe_cycles : costs.direct_observe_cycles;
        if (costs.input_follower_poll_cycles > (UINT64_MAX - probe) / interval) return false;
        const uint64_t delay = interval * costs.input_follower_poll_cycles;
        if (delay > UINT64_MAX - arrival) return true;
        uint64_t next_probe = arrival + delay;
        const uint64_t done = events_[follower.done].cycles;
        const uint64_t period = delay + probe;
        if (done > next_probe) {
            const uint64_t remainder = period == 0 ? 0 : (done - next_probe) % period;
            const uint64_t wait = remainder == 0 ? 0 : period - remainder;
            if (wait > UINT64_MAX - done) return true;
            next_probe = done + wait;
        }
        if (probe <= UINT64_MAX - next_probe) cycles = std::min(cycles, next_probe + probe);
        return true;
    }

    bool finish_helper(const FaninHelper &helper, uint64_t cost, uint64_t &cycles) const {
        std::array<uint64_t, CALENDAR_FANIN_TASKS_PER_LANE> ready{};
        for (size_t i = 0; i < helper.count; ++i) {
            ready[i] = events_[helper.producers[i]].cycles;
        }
        // Ready credits use work-conserving serial service on this helper.
        std::sort(ready.begin(), ready.begin() + helper.count);
        cycles = events_[helper.arrival].cycles;
        for (size_t i = 0; i < helper.count; ++i) {
            cycles = std::max(cycles, ready[i]);
            if (cost > UINT64_MAX - cycles) return false;
            cycles += cost;
        }
        return true;
    }
};

struct CalendarTaskEvents {
    size_t done;
    size_t all_dependencies;
    size_t wait_gate;
    size_t cohort_start;
    uint64_t direct_observe_cycles{0};
    std::vector<bool> assigned{};
    size_t helper_count{0};
};

struct CalendarMixEvents {
    size_t arrivals;
    size_t release;
};

// An ordered direct wait folds into one shared maximum of DONE times plus suffix costs.
inline bool wire_calendar_task(
    const TaskInput &task, const std::vector<int32_t> &waits, const std::vector<TaskInput> &tasks,
    const std::vector<CalendarTaskEvents> &task_events, bool summarized, uint64_t observe_cycles,
    CalendarTaskEvents &events, CalendarEventGraph &graph
) {
    size_t kept = 0;
    for (int32_t i = 0; i < task.fanin_count; ++i) {
        const auto pred = static_cast<size_t>(task.fanin_ids[i]);
        if (tasks[pred].inline_completed) continue;
        graph.link(task_events[pred].done, events.all_dependencies);
        if (kept < waits.size() && waits[kept] == task.fanin_ids[i]) ++kept;
    }
    if (kept != waits.size()) return false;
    if (summarized) {
        if (waits.size() < 2 || task.is_mix || task.sync_start) return false;
        events.assigned.resize(waits.size(), false);
        return true;
    }
    uint64_t lag = 0;
    for (auto it = waits.rbegin(); it != waits.rend(); ++it) {
        if (observe_cycles > UINT64_MAX - lag) return false;
        lag += observe_cycles;
        graph.link(task_events[*it].done, events.wait_gate, lag);
    }
    events.direct_observe_cycles = lag;
    return true;
}

inline bool wire_calendar_helper(
    const range_calendar_detail::CalendarFaninAssignment &assignment, const std::vector<int32_t> &waits,
    const std::vector<CalendarTaskEvents> &task_events, size_t arrival, CalendarTaskEvents &events,
    CalendarEventGraph &graph, bool fused, const std::vector<bool> &has_defer
) {
    if (assignment.count == 0) return true;
    const size_t begin = assignment.begin;
    const size_t count = assignment.count;
    if (count > CALENDAR_FANIN_TASKS_PER_LANE || begin + count > waits.size()) return false;
    FaninHelper helper;
    helper.arrival = arrival;
    helper.count = count;
    for (size_t i = 0; i < count; ++i) {
        if (events.assigned[begin + i]) return false;
        events.assigned[begin + i] = true;
        const auto producer = waits[begin + i];
        helper.producers[i] = task_events[producer].done;
        if (fused) {
            graph.link(helper.producers[i], events.wait_gate);
            if (has_defer[producer]) graph.link(arrival, events.wait_gate);
        }
    }
    ++events.helper_count;
    if (!fused) graph.add_helper(helper, events.wait_gate);
    return true;
}

}  // namespace calendar_detail

// Replays a fixed placement through its last task DONE or lane retirement.
inline bool evaluate_calendar_events(
    const std::vector<TaskInput> &tasks, uint32_t cluster_count, const CalendarPlan &plan,
    const std::vector<CalendarRowCost> &row_costs, const CalendarEventCosts &costs, uint64_t *predicted_cycles,
    bool allow_fused_fanins = false
) {
    using namespace calendar_detail;
    Shape shape;
    CalendarLayout layout;
    if (predicted_cycles == nullptr || !describe(tasks, cluster_count, row_costs, &shape) ||
        !read_calendar_layout(tasks, cluster_count, shape, plan, layout))
        return false;
    const auto &waits = plan.waits;
    if (waits.task_fanins.size() != tasks.size() || waits.summarized_fanins.size() != tasks.size() ||
        waits.row_waits.size() != layout.rows.size() || waits.row_assignments.size() != layout.rows.size() ||
        waits.row_input_owners.size() != layout.rows.size())
        return false;

    CalendarEventGraph graph(shape.row_count);
    const size_t initial = graph.add_event();
    std::vector<CalendarTaskEvents> task_events;
    task_events.reserve(tasks.size());
    for (const auto &task : tasks) {
        task_events.push_back(
            {graph.add_event(), graph.add_event(), graph.add_event(), task.sync_start ? graph.add_event() : initial}
        );
    }
    for (size_t id = 0; id < tasks.size(); ++id) {
        if (!wire_calendar_task(
                tasks[id], waits.task_fanins[id], tasks, task_events, waits.summarized_fanins[id],
                costs.direct_observe_cycles, task_events[id], graph
            ))
            return false;
    }
    std::vector<CalendarMixEvents> mix;
    mix.reserve(layout.joins.size());
    for (size_t i = 0; i < layout.joins.size(); ++i) {
        mix.push_back({graph.add_event(), graph.add_event()});
    }

    std::vector<size_t> input_releases(layout.rows.size(), initial);
    const bool use_fusion = allow_fused_fanins && !waits.fused_fanin_credits.empty();
    std::vector<bool> has_defer(use_fusion ? tasks.size() : 0);
    for (size_t i = 0; i < layout.rows.size(); ++i) {
        if (waits.row_input_owners[i] == i) input_releases[i] = graph.add_event();
        if (use_fusion && row_cost(row_costs, plan.logical_row_by_packed[i]).defer_tail_cycles != 0)
            has_defer[percounter_order_task_id(layout.rows[i].entry)] = true;
    }

    for (uint32_t worker = 0; worker < layout.header.worker_count; ++worker) {
        size_t previous_retire = initial;
        const uint32_t lane = worker_lane(cluster_count, worker);
        for (uint32_t i = layout.indices[worker]; i < layout.indices[worker + 1]; ++i) {
            const auto &row = layout.rows[i];
            const auto id = static_cast<uint32_t>(percounter_order_task_id(row.entry));
            const auto &task = tasks[id];
            auto &events = task_events[id];
            const size_t arrival = graph.add_event();
            const size_t ready = graph.add_event();
            const size_t entry = graph.add_event();
            const size_t occupancy_end = graph.add_event();
            const size_t retire = graph.add_event();
            graph.link(previous_retire, arrival, costs.row_prepare_cycles);
            const auto row_duration = row_cost(row_costs, plan.logical_row_by_packed[i]);
            graph.link(ready, entry, costs.entry_cycles);
            graph.link(entry, occupancy_end, row_duration.occupancy_cycles);
            graph.link(occupancy_end, retire, costs.retire_after_occupancy_cycles);
            if (costs.done_publish_cycles > UINT64_MAX - row_duration.defer_tail_cycles) return false;
            graph.link(occupancy_end, events.done, row_duration.defer_tail_cycles + costs.done_publish_cycles);
            previous_retire = retire;

            const bool is_mix = row.local_join != CALENDAR_NO_JOIN;
            const bool leader = !is_mix || layout.joins[row.local_join].leader_lane == lane;
            const auto assignment = waits.row_assignments[i];
            if (assignment.count != 0 && (!waits.row_waits[i] || !waits.summarized_fanins[id])) return false;

            const uint32_t input_owner = waits.row_input_owners[i];
            if (input_owner != CALENDAR_NO_INPUT_OWNER) {
                if (input_owner >= input_releases.size() || input_releases[input_owner] == initial || is_mix ||
                    task.sync_start || !waits.row_waits[i] ||
                    (!waits.summarized_fanins[id] && waits.task_fanins[id].size() != 1))
                    return false;
                if (input_owner != i) {
                    if (assignment.count != 0) return false;
                    const bool summarized = waits.summarized_fanins[id];
                    const size_t done = summarized ? events.wait_gate : task_events[waits.task_fanins[id][0]].done;
                    graph.add_follower({arrival, input_releases[input_owner], done, summarized}, ready);
                    continue;
                }
            }

            size_t arrive_at_inputs = arrival;
            if (is_mix) {
                const auto &join = mix[row.local_join];
                graph.link(arrival, join.arrivals, costs.mix_arrival_cycles);
                graph.link(join.release, ready, leader ? 0 : costs.mix_follower_observe_cycles);
                if (!leader) continue;
                arrive_at_inputs = join.arrivals;
            }
            const size_t inputs_ready = graph.add_event();
            graph.link(events.all_dependencies, inputs_ready);
            if (!waits.row_waits[i]) {
                graph.link(arrive_at_inputs, inputs_ready);
            } else if (waits.summarized_fanins[id]) {
                if (!wire_calendar_helper(
                        assignment, waits.task_fanins[id], task_events, arrive_at_inputs, events, graph,
                        use_fusion && waits.fused_fanin_credits[id] != 0, has_defer
                    ))
                    return false;
                graph.link(arrive_at_inputs, inputs_ready, costs.fanin_observe_cycles);
                graph.link(events.wait_gate, inputs_ready, costs.fanin_observe_cycles);
            } else {
                if (waits.task_fanins[id].empty()) return false;
                graph.link(arrive_at_inputs, inputs_ready, events.direct_observe_cycles);
                graph.link(events.wait_gate, inputs_ready);
            }
            size_t start_permission = inputs_ready;
            if (task.sync_start) {
                graph.link(inputs_ready, events.cohort_start);
                start_permission = events.cohort_start;
            }
            if (is_mix) {
                graph.link(start_permission, mix[row.local_join].release, costs.mix_release_cycles);
            } else if (input_owner == i) {
                graph.link(start_permission, input_releases[i], costs.mix_release_cycles);
                graph.link(input_releases[i], ready);
            } else {
                graph.link(start_permission, ready);
            }
        }
    }
    for (size_t id = 0; id < tasks.size(); ++id) {
        const auto &events = task_events[id];
        if (waits.summarized_fanins[id] &&
            (events.helper_count < 2 || std::any_of(
                                            events.assigned.begin(), events.assigned.end(),
                                            [](bool assigned) {
                                                return !assigned;
                                            }
                                        )))
            return false;
    }
    return graph.replay(costs, *predicted_cycles);
}

}  // namespace simpler::percounter
