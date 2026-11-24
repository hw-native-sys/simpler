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
 * Host-side planning for the percounter scheduler: the layout of its GM region
 * and the tables inside it.
 *
 * Host-only (it uses the STL). Everything here is `inline` and free-standing so
 * it can be unit-tested without a Runtime.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dispatch_payload.h"
#include "host/scheduler_notification_address.h"
#include "host_build_graph/host/control_folding.h"
#include "host_build_graph/runtime_types.h"
#include "scheduler/wire_types.h"
#include "scheduler/profiling_types.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace simpler::percounter {

/** Host-side plan of the uploaded metadata and the device scratch. */
struct PercounterLayout {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t table_task_count;
    uint64_t control_offset;
    uint64_t sinks_offset;
    uint64_t counters_offset;
    uint64_t async_groups_offset;
    uint64_t async_entries_offset;
    uint64_t async_completions_offset;
    uint64_t async_completion_capacity;
    uint64_t mix_task_count;
    uint64_t mix_signal_ready_offset;
    uint64_t sync_cohort_count;
    uint64_t sync_notify_offset;
    uint64_t graph_invocation_count;
    uint64_t graph_invocations_offset;
    uint64_t graph_prepare_offset;
    uint64_t graph_templates_offset;
    uint64_t graph_templates_bytes;
    uint64_t upload_size;
    uint64_t profile_buffers_offset;
    uint64_t aicpu_profile_offset;
    uint64_t graph_ready_timing_offset;
    uint64_t task_records_offset;
    uint64_t task_record_stride;
    uint64_t scalar_records_offset;
    uint64_t event_records_offset;
    uint32_t task_record_capacity;
    uint32_t event_record_capacity;
    uint32_t profiling_level;
};

struct GraphLayoutInput {
    uint64_t body_task_count{0};
    uint64_t invocation_count{0};
    uint64_t templates_bytes{0};
};

struct HostBodyTask {
    uint32_t subtask_total;
    uint32_t active_mask;
    uint32_t block_num;
    bool sync_start;
    bool sink;
    bool has_predicate;
};

struct HostGraphTemplate {
    std::vector<HostBodyTask> tasks;
    std::vector<int32_t> fanin_offsets;
    std::vector<int32_t> fanin_indices;
    uint64_t sink_subtask_total{0};
    uint64_t executable_blocks{0};
    uint64_t mix_task_count{0};
    uint64_t sync_cohort_count{0};
    uint64_t defer_rows{0};
};

struct GraphInvocationInput {
    int64_t shell_task_id;
    uint32_t template_index;
};

struct GraphPlanInput {
    const HostGraphTemplate *templates{nullptr};
    size_t template_count{0};
    const GraphInvocationInput *invocations{nullptr};
    size_t invocation_count{0};
};

struct ExternalAddresses {
    uint64_t graph_storage{0};
    uint64_t graph_storage_count{0};
    uint64_t callable_table{0};
    uint64_t callable_count{0};
    const uint64_t *sync_notify{nullptr};
    uint64_t sync_notify_count{0};
};

inline bool plan_layout(
    uint64_t table_task_count, uint64_t block_total, PercounterLayout *layout, uint64_t mix_task_count = 0,
    uint64_t sync_cohort_count = 0, const GraphLayoutInput &graph = {}, uint64_t defer_rows = 0,
    uint32_t profiling_level = 0
) {
    if (layout == nullptr || graph.body_task_count > UINT32_MAX ||
        table_task_count > UINT32_MAX - graph.body_task_count)
        return false;
    const uint64_t task_count = table_task_count + graph.body_task_count;
    if (graph.invocation_count > table_task_count || mix_task_count > task_count || sync_cohort_count > task_count)
        return false;

    PercounterLayout next{};
    next.profiling_level = profiling_level;
    next.task_record_capacity = static_cast<uint32_t>(std::min(block_total, uint64_t{65536}));
    if (profiling_level >= 2) {
        constexpr uint32_t events_per_block = 12;
        next.event_record_capacity = std::min(next.task_record_capacity, 2048U) * events_per_block;
    }
    next.task_count = task_count;
    next.table_task_count = table_task_count;
    next.mix_task_count = mix_task_count;
    next.sync_cohort_count = sync_cohort_count;
    next.graph_invocation_count = graph.invocation_count;
    next.graph_templates_bytes = graph.templates_bytes;
    next.async_completion_capacity = defer_rows;

    const uint64_t ready_records = graph.body_task_count / PERCOUNTER_GRAPH_SLICE_TASKS + graph.invocation_count;
    uint64_t cursor = 0;
    const auto reserve = [&](uint64_t count, uint64_t size, uint64_t alignment, uint64_t &offset) {
        uint64_t bytes;
        return scheduler_layout_checked_mul(count, size, &bytes) &&
               scheduler_layout_reserve(&cursor, bytes, alignment, &offset);
    };
#define PERCOUNTER_RESERVE(count, type, field) reserve((count), sizeof(type), alignof(type), next.field)
    if (!PERCOUNTER_RESERVE(1, PercounterControl, control_offset) ||
        !PERCOUNTER_RESERVE(table_task_count, uint64_t, sinks_offset) ||
        !PERCOUNTER_RESERVE(task_count, PercounterCounter, counters_offset) ||
        (mix_task_count != 0 && !PERCOUNTER_RESERVE(SCHEDULER_CLUSTER_CAPACITY, uint64_t, mix_signal_ready_offset)) ||
        (sync_cohort_count != 0 && !PERCOUNTER_RESERVE(sync_cohort_count, uint64_t, sync_notify_offset)) ||
        (graph.invocation_count != 0 &&
         (!PERCOUNTER_RESERVE(graph.invocation_count, PercounterGraphInvocation, graph_invocations_offset) ||
          !PERCOUNTER_RESERVE(1, PercounterGraphPrepare, graph_prepare_offset) ||
          !scheduler_layout_reserve(
              &cursor, graph.templates_bytes, alignof(PercounterGraphTemplate), &next.graph_templates_offset
          ))) ||
        (profiling_level >= 2 &&
         (!PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, PercounterProfileBuffer, profile_buffers_offset) ||
          !PERCOUNTER_RESERVE(PLATFORM_MAX_AICPU_THREADS, PercounterAicpuTiming, aicpu_profile_offset) ||
          (ready_records != 0 &&
           !reserve(ready_records, sizeof(PercounterGraphReadyTiming), 64, next.graph_ready_timing_offset)))))
        return false;
    next.upload_size = cursor;

    if (!PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, pto2::resident::AsyncGroupData, async_groups_offset) ||
        !PERCOUNTER_RESERVE(
            SCHEDULER_WORKER_CAPACITY * PERCOUNTER_ASYNC_CAPACITY, DeferredCompletionEntry, async_entries_offset
        ) ||
        !PERCOUNTER_RESERVE(defer_rows, PercounterAsyncCompletion, async_completions_offset))
        return false;
#undef PERCOUNTER_RESERVE
    if (profiling_level >= 2 &&
        (!scheduler_layout_checked_align(
             uint64_t{next.task_record_capacity} * sizeof(PercounterTaskTiming), 64, &next.task_record_stride
         ) ||
         !reserve(SCHEDULER_WORKER_CAPACITY, next.task_record_stride, 64, next.task_records_offset) ||
         !reserve(
             SCHEDULER_WORKER_CAPACITY * uint64_t{next.event_record_capacity}, sizeof(PercounterTaskEvent), 64,
             next.event_records_offset
         )))
        return false;
    if (profiling_level >= 3 && !reserve(
                                    SCHEDULER_WORKER_CAPACITY * uint64_t{SCHEDULER_SCALAR_RECORD_CAPACITY},
                                    sizeof(SchedulerScalarRecord), 64, next.scalar_records_offset
                                ))
        return false;
    if (!scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) return false;
    *layout = next;
    return true;
}

/** Where a task goes, decided by which subtask slots it occupies. */
enum class TaskPlacement : uint8_t {
    AIC,          // mask 1: the cube core alone
    AIV,          // mask 2: the first vector core alone
    MIX,          // two or three slots of one cluster, together
    UNSUPPORTED,  // no slot at all, or AIV1 alone
};

// Classify the execution roles requested by the task.
inline TaskPlacement task_placement_from_active_mask(uint8_t active_mask, uint8_t *core_type_index) {
    if (core_type_index == nullptr) return TaskPlacement::UNSUPPORTED;
    if (active_mask == 1U) {
        *core_type_index = 0;
        return TaskPlacement::AIC;
    }
    if (active_mask == 2U) {
        *core_type_index = 1;
        return TaskPlacement::AIV;
    }
    // Two or three of the three slots, which is every legal MIX shape: AIC+AIV0
    // (0b011), AIC+AIV1 (0b101), both AIV (0b110), all three (0b111).
    if (active_mask <= 7U && (active_mask & (active_mask - 1U)) != 0) {
        *core_type_index = 0;
        return TaskPlacement::MIX;
    }
    return TaskPlacement::UNSUPPORTED;
}

/** The same mask as the lane bits upstream's shape checks take. */
inline LaneMask lane_mask_from_active_mask(uint8_t active_mask) {
    return {(active_mask & 1U) != 0, (active_mask & 2U) != 0, (active_mask & 4U) != 0};
}

/** One task, as the table builder needs to see it. */
struct TaskInput {
    // 0 for AIC and 1 for AIV. Ignored when `is_mix` is set: a MIX task occupies
    // two or three lanes of one cluster and belongs to no single core type.
    uint8_t core_type_index;
    bool is_mix;
    bool inline_completed;  // completed during bind and omitted from the calendar
    int32_t fanin_count;
    const int32_t *fanin_ids;  // producer task ids; may be null when count is 0
    int32_t block_num{1};
    // Each block contributes one completion credit per participating role.
    int32_t subtask_count{1};
    // All blocks must commit to entry before any may launch.
    bool sync_start{false};
    // Carries a dispatch predicate, evaluated on the lane before the kernel runs.
    bool has_predicate{false};
    // Index of the Graph invocation, or -1 for an ordinary task.
    int32_t graph_invocation{-1};
    // Actual kernels; a dependency-only block uses no kernel but still targets AIV.
    uint32_t active_mask{0};
};

// Fold only top-level joins with no payload, predicate or sync-start.
inline bool percounter_fold_dependency_tasks(
    std::vector<TaskInput> &tasks, std::vector<std::vector<int32_t>> &rows, std::vector<int64_t> &folded
) {
    const size_t count = tasks.size();
    const auto is_control = [](const TaskInput &task) {
        return !task.inline_completed && task.graph_invocation < 0 && task.active_mask == 0 && task.block_num == 1 &&
               task.subtask_count == 1 && !task.has_predicate && !task.sync_start && !task.is_mix;
    };
    if (std::none_of(tasks.begin(), tasks.end(), is_control)) return true;

    rows.resize(count);
    std::vector<bool> active(count), controls(count);
    for (size_t id = 0; id < count; ++id) {
        const auto &task = tasks[id];
        active[id] = !task.inline_completed;
        controls[id] = is_control(task);
        if (!active[id]) continue;
        if (task.fanin_count < 0 || (task.fanin_count > 0 && task.fanin_ids == nullptr)) return false;
        for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
            const int32_t producer = task.fanin_ids[edge];
            if (producer < 0 || static_cast<size_t>(producer) >= id) return false;
            rows[id].push_back(producer);
        }
    }
    folded = scheduler_fold_control_fanin(rows, active, controls, CHIP_MAX_FANIN);
    if (folded.empty()) return true;
    for (size_t id = 0; id < count; ++id) {
        if (!active[id]) continue;
        tasks[id].fanin_count = static_cast<int32_t>(rows[id].size());
        tasks[id].fanin_ids = rows[id].data();
    }
    for (const int64_t id : folded)
        tasks[id].inline_completed = true;
    return true;
}

inline uint64_t next_async_run_epoch() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

template <typename T>
inline T *region_at(void *base, uint64_t offset) {
    return reinterpret_cast<T *>(static_cast<uint8_t *>(base) + offset);
}

inline bool build_tables(
    const std::vector<TaskInput> &tasks, const PercounterLayout &layout, void *base, const ExternalAddresses &external,
    const GraphPlanInput &graph
) {
    if (base == nullptr || tasks.size() != layout.table_task_count ||
        graph.invocation_count != layout.graph_invocation_count ||
        graph.template_count * sizeof(PercounterGraphTemplate) != layout.graph_templates_bytes ||
        external.sync_notify_count != layout.sync_cohort_count ||
        (graph.template_count != 0 && graph.templates == nullptr) ||
        (graph.invocation_count != 0 && graph.invocations == nullptr) ||
        (external.sync_notify_count != 0 && external.sync_notify == nullptr))
        return false;
    std::memset(base, 0, static_cast<size_t>(layout.upload_size));
    auto *control = region_at<PercounterControl>(base, layout.control_offset);
    auto *sinks = region_at<uint64_t>(base, layout.sinks_offset);
    std::vector<bool> consumed(tasks.size(), false);
    for (size_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        if (task.inline_completed) continue;
        if (task.fanin_count < 0 || (task.fanin_count != 0 && task.fanin_ids == nullptr)) return false;
        for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
            const int32_t producer = task.fanin_ids[edge];
            if (producer < 0 || static_cast<size_t>(producer) >= id) return false;
            consumed[producer] = true;
        }
    }
    for (size_t id = 0; id < tasks.size(); ++id)
        if (!tasks[id].inline_completed && !consumed[id]) sinks[control->sink_count++] = id;

    auto *templates = region_at<PercounterGraphTemplate>(base, layout.graph_templates_offset);
    for (size_t i = 0; i < graph.template_count; ++i)
        templates[i].task_count = graph.templates[i].tasks.size();
    auto *invocations = region_at<PercounterGraphInvocation>(base, layout.graph_invocations_offset);
    uint64_t body_id = tasks.size();
    uint64_t ready_timing_offset = layout.graph_ready_timing_offset;
    for (size_t rank = 0; rank < graph.invocation_count; ++rank) {
        const auto &invocation = graph.invocations[rank];
        if (invocation.template_index >= graph.template_count || invocation.shell_task_id < 0 ||
            static_cast<uint64_t>(invocation.shell_task_id) >= tasks.size() ||
            tasks[invocation.shell_task_id].graph_invocation != static_cast<int32_t>(rank))
            return false;
        const auto &body = graph.templates[invocation.template_index];
        if (body.tasks.size() > layout.task_count - body_id) return false;
        invocations[rank].template_offset =
            layout.graph_templates_offset + invocation.template_index * sizeof(PercounterGraphTemplate);
        invocations[rank].shell_task_id = static_cast<uint64_t>(invocation.shell_task_id);
        invocations[rank].body_base = body_id;
        if (ready_timing_offset != 0) {
            invocations[rank].ready_timing_offset = ready_timing_offset;
            ready_timing_offset +=
                ((body.tasks.size() + PERCOUNTER_GRAPH_SLICE_TASKS - 1) / PERCOUNTER_GRAPH_SLICE_TASKS) *
                sizeof(PercounterGraphReadyTiming);
        }
        body_id += body.tasks.size();
    }
    if (body_id != layout.task_count) return false;
    auto *sync_notify = region_at<uint64_t>(base, layout.sync_notify_offset);
    for (uint32_t i = 0; i < external.sync_notify_count; ++i) {
        const uint64_t set = external.sync_notify[i];
        if (a5_cnt_notify_increment_address(set) == 0) return false;
        sync_notify[i] = set;
    }
    if (layout.profile_buffers_offset != 0) {
        auto *buffers = region_at<PercounterProfileBuffer>(base, layout.profile_buffers_offset);
        for (uint32_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
            auto &buffer = buffers[worker];
            buffer.task_capacity = layout.task_record_capacity;
            buffer.event_capacity = layout.event_record_capacity;
            buffer.event_records_offset = layout.event_records_offset +
                                          uint64_t{worker} * layout.event_record_capacity * sizeof(PercounterTaskEvent);
            buffer.task_records_offset = layout.task_records_offset + uint64_t{worker} * layout.task_record_stride;
            if (layout.scalar_records_offset != 0)
                buffer.scalar_records_offset = layout.scalar_records_offset + uint64_t{worker} *
                                                                                  SCHEDULER_SCALAR_RECORD_CAPACITY *
                                                                                  sizeof(SchedulerScalarRecord);
            buffer.physical_core_id = -1;
        }
    }
    control->task_count = layout.task_count;
    control->table_task_count = layout.table_task_count;
    control->sinks_offset = layout.sinks_offset;
    control->counters_offset = layout.counters_offset;
    control->mix_task_count = layout.mix_task_count;
    control->mix_signal_ready_offset = layout.mix_signal_ready_offset;
    control->sync_cohort_count = layout.sync_cohort_count;
    control->sync_notify_offset = layout.sync_notify_offset;
    control->graph_invocation_count = layout.graph_invocation_count;
    control->graph_invocations_offset = layout.graph_invocations_offset;
    control->graph_prepare_offset = layout.graph_prepare_offset;
    control->graph_storage_address = external.graph_storage;
    control->graph_storage_count = external.graph_storage_count;
    control->callable_addresses_address = external.callable_table;
    control->callable_addresses_count = external.callable_count;
    control->async_groups_offset = layout.async_groups_offset;
    control->async_entries_offset = layout.async_entries_offset;
    control->async_completions_offset = layout.async_completions_offset;
    control->async_completion_capacity = layout.async_completion_capacity;
    control->async_run_epoch = next_async_run_epoch();
    control->scheduler_timeout_cycles = SCHEDULER_TIMEOUT_CYCLES;
    control->profiling_level = layout.profiling_level;
    control->profile_buffers_offset = layout.profile_buffers_offset;
    control->aicpu_profile_offset = layout.aicpu_profile_offset;
    return true;
}

}  // namespace simpler::percounter
