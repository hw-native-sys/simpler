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
 * Host-side selection of terminal suffixes for percounter. The shape planner
 * identifies closed suffixes. Dispatch requires the proven planner's additional
 * check that the prefix can drain even with deferred completions.
 *
 * Host-only (it uses the STL). Everything here is `inline` and free-standing so
 * it can be unit-tested without a Runtime.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

#include <cstdint>
#include <vector>

namespace simpler::percounter {

struct TerminalPipeline {
    std::vector<uint32_t> steps;
    std::vector<uint32_t> prerequisites;
};

template <typename Task>
inline void terminal_pipeline_collect_prerequisites(const Task *tasks, size_t task_count, TerminalPipeline &plan) {
    plan.prerequisites.clear();
    const auto in_pipeline = [&](uint32_t id) {
        return std::find(plan.steps.begin(), plan.steps.end(), id) != plan.steps.end();
    };
    const auto live = [&](uint32_t id) {
        return !tasks[id].inline_completed;
    };
    // Outside tasks that have no outside consumer form the entry frontier.
    // Any other outside task reaches one of those in this acyclic graph.
    std::vector<bool> outside_consumed(task_count, false);
    for (uint32_t id = 0; id < task_count; ++id) {
        if (!live(id) || in_pipeline(id)) continue;
        for (int32_t i = 0; i < tasks[id].fanin_count; ++i)
            outside_consumed[static_cast<uint32_t>(tasks[id].fanin_ids[i])] = true;
    }
    for (uint32_t id = 0; id < task_count; ++id)
        if (live(id) && !in_pipeline(id) && !outside_consumed[id]) plan.prerequisites.push_back(id);
}

// Identify a closed full-width V -> MIX -> MIX suffix after control folding.
// Waiting for every frontier task allows the suffix to use a deterministic
// placement without taking any resource from the general scheduler.
template <typename Task>
inline std::optional<TerminalPipeline> plan_terminal_pipeline(const Task *tasks, size_t task_count, uint32_t clusters) {
    constexpr uint32_t absent = std::numeric_limits<uint32_t>::max();
    if (clusters == 0 || task_count >= absent) return std::nullopt;
    const auto live = [&](uint32_t id) {
        return !tasks[id].inline_completed;
    };
    const auto full_wave = [&](uint32_t id, bool mix) {
        const auto &t = tasks[id];
        return live(id) && t.is_mix == mix && !t.sync_start && !t.has_predicate && t.active_mask == (mix ? 7U : 2U) &&
               t.block_num == static_cast<int64_t>(clusters) * (mix ? 1 : 2);
    };
    // A DAG with a unique sink must end with its live sink in submission order.
    size_t live_count = task_count;
    while (live_count != 0 && !live(static_cast<uint32_t>(live_count - 1)))
        --live_count;
    if (live_count == 0 || !full_wave(static_cast<uint32_t>(live_count - 1), true)) return std::nullopt;
    std::vector<bool> forked(task_count, false);
    std::vector<uint32_t> consumer(task_count, absent);
    std::vector<std::vector<uint32_t>> consumers(task_count);
    for (uint32_t id = 0; id < task_count; ++id) {
        const Task &task = tasks[id];
        if (task.graph_invocation >= 0 || task.published_off_lane) return std::nullopt;
        if (!live(id)) continue;
        if (task.fanin_count < 0 || (task.fanin_count > 0 && !task.fanin_ids)) return std::nullopt;
        for (int32_t i = 0; i < task.fanin_count; ++i) {
            const auto pred = static_cast<uint32_t>(task.fanin_ids[i]);
            if (pred >= id) return std::nullopt;
            if (!live(pred)) continue;
            if (consumer[pred] != absent && consumer[pred] != id) forked[pred] = true;
            consumer[pred] = id;
            consumers[pred].push_back(id);
        }
    }
    uint32_t sink = absent;
    for (uint32_t id = 0; id < task_count; ++id) {
        if (!live(id) || consumer[id] != absent) continue;
        if (sink != absent) return std::nullopt;
        sink = id;
    }
    if (sink == absent) return std::nullopt;
    const auto one_parent = [&](uint32_t id, bool mix) {
        uint32_t parent = absent;
        for (int32_t i = 0; i < tasks[id].fanin_count; ++i) {
            const auto pred = static_cast<uint32_t>(tasks[id].fanin_ids[i]);
            if (forked[pred] || consumer[pred] != id || !full_wave(pred, mix)) continue;
            if (parent != absent && parent != pred) return absent;
            parent = pred;
        }
        return parent;
    };
    const auto mix = one_parent(sink, true);
    if (mix == absent) return std::nullopt;
    const auto first = one_parent(mix, false);
    if (first == absent) return std::nullopt;
    TerminalPipeline plan{{first, mix, sink}, {}};
    const auto in_pipeline = [&](uint32_t id) {
        return std::find(plan.steps.begin(), plan.steps.end(), id) != plan.steps.end();
    };
    const auto collect_parent = [&](uint32_t id, bool mix, bool require_full_sync = true) {
        uint32_t candidate = absent;
        for (int32_t j = 0; j < tasks[id].fanin_count; ++j) {
            const uint32_t prior = static_cast<uint32_t>(tasks[id].fanin_ids[j]);
            const auto &t = tasks[prior];
            const bool matches = mix ? (t.is_mix && t.active_mask == 7 && t.block_num >= 2 && t.block_num <= clusters &&
                                        (!require_full_sync || (t.sync_start && t.block_num == clusters))) :
                                       (!t.is_mix && !t.sync_start && t.active_mask == 2 && t.block_num == 1);
            if (!live(prior) || !matches || t.has_predicate ||
                !std::all_of(consumers[prior].begin(), consumers[prior].end(), in_pipeline))
                continue;
            if (candidate != absent && candidate != prior) return absent;
            candidate = prior;
        }
        return candidate;
    };
    // A narrow wave can overlap with work still running in the generic prefix.
    if (const auto gather = collect_parent(first, true, false);
        gather != absent && tasks[gather].block_num == clusters) {
        plan.steps.insert(plan.steps.begin(), gather);
        if (const auto bridge = collect_parent(gather, false); bridge != absent) {
            plan.steps.insert(plan.steps.begin(), bridge);
            if (const auto begin = collect_parent(bridge, true); begin != absent) {
                plan.steps.insert(plan.steps.begin(), begin);
            } else {
                // A ready, independent bridge can execute in the generic
                // prefix. Keep its completion in the entry frontier instead
                // of serializing it after all other prefix work.
                bool root_bridge = true;
                for (int32_t i = 0; i < tasks[bridge].fanin_count; ++i)
                    root_bridge &= !live(static_cast<uint32_t>(tasks[bridge].fanin_ids[i]));
                if (root_bridge) {
                    const uint32_t begin = collect_parent(gather, true);
                    if (begin != absent) plan.steps.front() = begin;
                }
            }
        }
    }
    terminal_pipeline_collect_prerequisites(tasks, task_count, plan);
    return plan;
}

// A prefix task can register a deferred condition backed by work outside
// its explicit fan-in. Draining it before entry may add a new dependency
// unless entry already waits for that task through the DAG.
template <typename Task, typename MayDefer>
inline bool
terminal_prefix_has_progress(const Task *tasks, size_t task_count, const TerminalPipeline &plan, MayDefer may_defer) {
    const uint32_t entry = plan.steps.front();
    std::vector<bool> required(task_count, false);
    required[entry] = true;
    for (size_t i = static_cast<size_t>(entry) + 1; i > 0; --i) {
        const size_t id = i - 1;
        if (!required[id]) continue;
        for (int32_t j = 0; j < tasks[id].fanin_count; ++j) {
            const auto producer = static_cast<uint32_t>(tasks[id].fanin_ids[j]);
            if (!tasks[producer].inline_completed) required[producer] = true;
        }
    }
    for (uint32_t id = 0; id < task_count; ++id) {
        if (required[id] || tasks[id].inline_completed || tasks[id].active_mask == 0 ||
            std::find(plan.steps.begin(), plan.steps.end(), id) != plan.steps.end())
            continue;
        if (may_defer(id)) return false;
    }
    return true;
}

template <typename Task, typename MayDefer>
inline std::optional<TerminalPipeline>
plan_proven_terminal_pipeline(const Task *tasks, size_t task_count, uint32_t clusters, MayDefer may_defer) {
    auto plan = plan_terminal_pipeline(tasks, task_count, clusters);
    if (!plan) return std::nullopt;
    const auto original_steps = plan->steps.size();
    while (!terminal_prefix_has_progress(tasks, task_count, *plan, may_defer)) {
        // The three full-width output stages form the base terminal shape.
        if (plan->steps.size() == 3) return std::nullopt;
        plan->steps.erase(plan->steps.begin());
    }
    if (plan->steps.size() != original_steps) terminal_pipeline_collect_prerequisites(tasks, task_count, *plan);
    return plan;
}

}  // namespace simpler::percounter
