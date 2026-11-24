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

#include "calendar_plan.h"
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
            {task.active_mask, static_cast<uint64_t>(task.block_num), task.sync_start, task.inline_completed,
             static_cast<uint64_t>(task.fanin_count), static_cast<uint64_t>(task.graph_invocation), task.has_predicate}
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

inline std::optional<CalendarRowCost> calendar_sample_cost(const CalendarTiming &sample) {
    if (sample.start == 0 || sample.end < sample.start ||
        (sample.defer_register == 0) != (sample.defer_completion == 0))
        return std::nullopt;
    const uint64_t release = std::max(sample.end, sample.defer_register);
    if (sample.defer_register != 0 && (sample.defer_register < sample.start || sample.defer_completion < release))
        return std::nullopt;
    return CalendarRowCost{
        std::max(uint64_t{1}, release - sample.start),
        sample.defer_completion == 0 ? 0 : sample.defer_completion - release
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
        std::vector<CalendarRowCost> costs(order.size(), CalendarRowCost{0, 0});
        size_t deferred_rows = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            const auto cost = calendar_sample_cost(records[i]);
            if (!cost || order[i] >= costs.size() || costs[order[i]].occupancy_cycles != 0) {
                LOG_WARN("A5 calendar: incomplete or invalid sample at row=%zu", i);
                return;
            }
            costs[order[i]] = *cost;
            deferred_rows += records[i].defer_register != 0;
        }
        uint64_t baseline_cycles = 0;
        if (!evaluate_static_calendar(tasks, cluster_count, *baseline, costs, &baseline_cycles)) {
            LOG_WARN("A5 calendar: could not evaluate the sampled placement");
            return;
        }

        auto selected = baseline;
        auto candidate = std::make_shared<CalendarPlan>();
        uint64_t candidate_cycles = baseline_cycles;
        if (build_static_calendar(tasks, cluster_count, costs, candidate.get())) {
            candidate_cycles = candidate->predicted_cycles;
            if (candidate_cycles < baseline_cycles) selected = std::move(candidate);
        }
        const auto id = calendar_plan_id(*selected);
        const auto rows = static_cast<uint32_t>(selected->logical_row_by_packed.size());
        const auto changed = static_cast<unsigned>(selected->bytes != baseline->bytes);
        const std::lock_guard<std::mutex> lock(cache->mutex);
        if (cache->plans.try_emplace(key, std::move(selected)).second) {
            LOG_INFO(
                "A5 calendar: STORE plan=%016" PRIx64 " rows=%u changed=%u pid=%d base_cycles=%" PRIu64
                " selected_cycles=%" PRIu64 " deferred_rows=%zu",
                id, rows, changed, getpid(), baseline_cycles, changed ? candidate_cycles : baseline_cycles,
                deferred_rows
            );
        }
    }
};

}  // namespace simpler::percounter
