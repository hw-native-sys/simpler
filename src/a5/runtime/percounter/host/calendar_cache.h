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
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include "calendar_event_score.h"
#include "common/host_api.h"
#include "common/unified_log.h"
#include "host_build_graph/host/graph_cache.h"
#include "host_build_graph/runtime.h"
#include "scheduler/calendar_wire.h"

namespace simpler::percounter {

using CalendarKey = std::vector<uint64_t>;

inline uint64_t calendar_fingerprint(const void *data, size_t bytes) {
    return graph_hash_bytes(1469598103934665603ULL, data, bytes);
}

struct CalendarKeyHash {
    size_t operator()(const CalendarKey &key) const {
        return static_cast<size_t>(calendar_fingerprint(key.data(), key.size() * sizeof(key[0])));
    }
};

struct CalendarCache {
    std::mutex mutex;
    std::unordered_map<CalendarKey, uint64_t, CalendarKeyHash> outer_signatures;
    std::map<CalendarKey, std::weak_ptr<const CalendarPlan>> training;
    std::unordered_map<CalendarKey, std::shared_ptr<const CalendarPlan>, CalendarKeyHash> plans;
};

inline std::shared_ptr<void> make_calendar_cache() { return std::make_shared<CalendarCache>(); }

template <class Scalar, class Tensor>
inline void
append_calendar_argument_key(CalendarKey &key, int32_t scalars, int32_t tensors, Scalar scalar, Tensor tensor_arg) {
    for (int32_t index = 0; index < scalars; ++index)
        key.push_back(scalar(index));
    std::unordered_map<uint64_t, uint32_t> buffer_representatives;
    for (int32_t index = 0; index < tensors; ++index) {
        const auto &tensor = tensor_arg(index);
        const auto representative =
            buffer_representatives.try_emplace(tensor.buffer.addr, static_cast<uint32_t>(index)).first->second;
        key.insert(
            key.end(),
            {representative, tensor.buffer.size, tensor.start_offset - tensor_arg(representative).start_offset,
             static_cast<uint64_t>(tensor.dtype), tensor.is_contiguous, tensor.ndims}
        );
        for (uint32_t dim = 0; dim < tensor.ndims; ++dim) {
            key.push_back(tensor.shapes[dim]);
            key.push_back(tensor.strides[dim]);
        }
    }
}

inline CalendarKey make_calendar_key(
    const Runtime &runtime, const std::vector<TaskInput> &tasks, const std::vector<SchedulerTaskMetadata> &metadata
) {
    const auto &args = runtime.get_orch_args();
    CalendarKey key{
        static_cast<uint64_t>(runtime.get_worker_count()), static_cast<uint64_t>(args.scalar_count()),
        static_cast<uint64_t>(args.tensor_count()), static_cast<uint64_t>(tasks.size())
    };
    append_calendar_argument_key(
        key, args.scalar_count(), args.tensor_count(),
        [&](int32_t i) {
            return args.scalar(i);
        },
        [&](int32_t i) -> const auto & {
            return args.tensor(i);
        }
    );
    for (size_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        key.insert(
            key.end(),
            {task.active_mask, static_cast<uint64_t>(task.block_num), static_cast<uint64_t>(task.subtask_count),
             task.sync_start, task.inline_completed, static_cast<uint64_t>(task.fanin_count),
             static_cast<uint64_t>(task.graph_invocation), task.has_predicate}
        );
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if ((task.active_mask & (1U << lane)) != 0) key.push_back(metadata[id].kernel_ids[lane]);
        }
        for (int32_t edge = 0; edge < task.fanin_count; ++edge)
            key.push_back(task.fanin_ids[edge]);
    }
    return key;
}

inline uint64_t calendar_plan_id(const CalendarPlan &plan) {
    return calendar_fingerprint(plan.bytes.data(), plan.bytes.size());
}

inline std::optional<CalendarRowCost> calendar_sample_cost(const CalendarTiming &sample, uint64_t cohort_entry) {
    if (sample.start == 0 || sample.end < sample.start ||
        (sample.defer_register == 0) != (sample.defer_completion == 0))
        return std::nullopt;
    const uint64_t release = std::max(sample.end, sample.defer_register);
    if (sample.defer_register != 0 && (sample.defer_register < sample.start || sample.defer_completion < release))
        return std::nullopt;
    const uint64_t start = std::max(sample.start, cohort_entry);
    return CalendarRowCost{
        release > start ? release - start : 1,
        sample.defer_completion - std::min(sample.defer_completion, std::max(release, start))
    };
}

struct CalendarTraining {
    std::shared_ptr<CalendarCache> cache;
    CalendarKey key;
    std::shared_ptr<const CalendarPlan> baseline;
    uint32_t cluster_count{0};
    std::vector<TaskInput> tasks;
    std::vector<int32_t> fanin;
    bool owns_training{false};
    bool allow_fused_fanins{false};

    ~CalendarTraining() {
        if (!owns_training) return;
        const std::lock_guard<std::mutex> lock(cache->mutex);
        cache->training.erase(key);
    }

    void copy_tasks(const std::vector<TaskInput> &source) {
        tasks = source;
        size_t edge_count = 0;
        for (const auto &task : tasks)
            edge_count += static_cast<size_t>(task.fanin_count);
        fanin.reserve(edge_count);
        for (auto &task : tasks) {
            if (task.fanin_count == 0) {
                task.fanin_ids = nullptr;
                continue;
            }
            const size_t begin = fanin.size();
            fanin.insert(fanin.end(), task.fanin_ids, task.fanin_ids + task.fanin_count);
            task.fanin_ids = fanin.data() + begin;
        }
    }

    void finish(const HostApi &api, const void *device_timings) {
        const auto &order = baseline->logical_row_by_packed;
        std::vector<CalendarTiming> records(order.size());
        if (api.copy_from_device(records.data(), device_timings, records.size() * sizeof(records[0])) != 0) {
            LOG_WARN("A5 calendar: sample readback failed");
            return;
        }
        calendar_detail::CalendarLayout layout;
        std::vector<uint64_t> cohort_entries;
        if (std::any_of(tasks.begin(), tasks.end(), [](const auto &task) {
                return task.sync_start;
            })) {
            calendar_detail::Shape shape;
            if (!calendar_detail::describe(tasks, cluster_count, {}, &shape) ||
                !calendar_detail::read_calendar_layout(tasks, cluster_count, shape, *baseline, layout) ||
                layout.rows.size() != records.size()) {
                LOG_WARN("A5 calendar: could not identify sampled cohort rows");
                return;
            }
            cohort_entries.resize(tasks.size());
            // Replay a cohort from its latest entry. Early blocks can do work
            // and wait for their peers before that point.
            for (size_t i = 0; i < records.size(); ++i) {
                const auto task = percounter_order_task_id(layout.rows[i].entry);
                if (tasks[task].sync_start) cohort_entries[task] = std::max(cohort_entries[task], records[i].start);
            }
        }
        std::vector<CalendarRowCost> costs(order.size(), CalendarRowCost{0, 0});
        size_t deferred_rows = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            const uint64_t cohort_entry =
                cohort_entries.empty() ? 0 : cohort_entries[percounter_order_task_id(layout.rows[i].entry)];
            const auto cost = calendar_sample_cost(records[i], cohort_entry);
            if (!cost || order[i] >= costs.size() || costs[order[i]].occupancy_cycles != 0) {
                LOG_WARN("A5 calendar: incomplete or invalid sample at row=%zu", i);
                return;
            }
            costs[order[i]] = *cost;
            deferred_rows += records[i].defer_register != 0;
        }
        constexpr CalendarEventCosts event_costs{
            .row_prepare_cycles = 55,
            .direct_observe_cycles = 1010,
            .fanin_credit_cycles = 1750,
            .fanin_observe_cycles = 1550,
            .entry_cycles = 249,
            .done_publish_cycles = 46,
            .retire_after_occupancy_cycles = 59,
        };
        uint64_t baseline_cycles = 0;
        const auto training_start = std::chrono::steady_clock::now();
        if (!evaluate_calendar_events(
                tasks, cluster_count, *baseline, costs, event_costs, &baseline_cycles, allow_fused_fanins
            )) {
            LOG_WARN("A5 calendar: could not evaluate the sampled placement");
            return;
        }

        struct Variant {
            const char *name;
            CalendarBlockOrder order;
            uint64_t priority_window_cycles;
            bool charge_waits;
            bool backfill;
        };
        constexpr Variant variants[] = {
            {"submission", CalendarBlockOrder::SUBMISSION, 0, false, false},
            {"wait_aware", CalendarBlockOrder::LONGEST_OUTPUT_FIRST, 0, true, false},
            {"backfill", CalendarBlockOrder::LONGEST_OUTPUT_FIRST, 20000, true, true},
        };
        std::vector<CalendarRowCost> planning_costs;
        constexpr CalendarPriorityCosts placement_costs{
            .row_cycles =
                event_costs.row_prepare_cycles + event_costs.entry_cycles + event_costs.retire_after_occupancy_cycles,
            .observation_cycles = event_costs.fanin_observe_cycles * 3 / 2,
            .credit_cycles = event_costs.direct_observe_cycles * 3 / 2,
        };
        if (!calendar_placement_costs(tasks, cluster_count, costs, placement_costs, &planning_costs)) return;
        auto selected = baseline;
        uint64_t selected_cycles = baseline_cycles;
        const char *selected_variant = "cold";
        for (const auto &variant : variants) {
            auto candidate = std::make_shared<CalendarPlan>();
            uint64_t predicted = 0;
            const auto &weights = variant.charge_waits ? planning_costs : costs;
            const auto rank = variant.charge_waits ? CalendarPriority::WORK_AND_WAIT : CalendarPriority::LONGEST_ROW;
            if (!build_static_calendar(
                    tasks, cluster_count, weights, candidate.get(), variant.order, variant.priority_window_cycles, rank,
                    variant.backfill
                ) ||
                !evaluate_calendar_events(
                    tasks, cluster_count, *candidate, costs, event_costs, &predicted, allow_fused_fanins
                )) {
                continue;
            }
            LOG_INFO("A5 calendar: TRIAL variant=%s cycles=%" PRIu64 " pid=%d", variant.name, predicted, getpid());
            if (predicted >= selected_cycles) continue;
            selected = std::move(candidate);
            selected_cycles = predicted;
            selected_variant = variant.name;
        }
        const auto training_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - training_start)
                .count()
        );
        const auto id = calendar_plan_id(*selected);
        const auto rows = static_cast<uint32_t>(selected->logical_row_by_packed.size());
        const auto changed = static_cast<unsigned>(selected->bytes != baseline->bytes);
        const std::lock_guard<std::mutex> lock(cache->mutex);
        if (cache->plans.try_emplace(key, std::move(selected)).second) {
            LOG_INFO(
                "A5 calendar: STORE plan=%016" PRIx64 " rows=%u changed=%u pid=%d base_cycles=%" PRIu64
                " selected_cycles=%" PRIu64 " deferred_rows=%zu variant=%s training_us=%" PRIu64,
                id, rows, changed, getpid(), baseline_cycles, selected_cycles, deferred_rows, selected_variant,
                training_us
            );
        }
    }
};

}  // namespace simpler::percounter
