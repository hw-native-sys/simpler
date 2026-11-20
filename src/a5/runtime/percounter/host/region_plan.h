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
#include <numeric>
#include <string>
#include <vector>

#include "dispatch_payload.h"
#include "host/scheduler_notification_address.h"
#include "host_build_graph/host/completion_counter_plan.h"
#include "host_build_graph/host/control_folding.h"
#include "host_build_graph/runtime_types.h"
#include "resource_order.h"
#include "scheduler/wire_types.h"
#include "scheduler/profiling_types.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace simpler::percounter {

/** Host-side plan of the percounter region. */
struct PercounterLayout {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t edge_count;
    // One order entry per task, including SPMD tasks.
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT];
    // Every entry of both lists, including MIX tasks in the vector list.
    uint64_t entry_count;
    uint64_t control_offset;
    uint64_t tickets_offset;
    uint64_t orders_offset[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t fanin_offset;
    uint64_t fanin_addr_offset;
    // One task index per top-level sink, in the
    // uploaded prefix because only the AICPU reads them. Reserved for every
    // table task, which is the bound a graph with no edges reaches.
    uint64_t sinks_offset;
    uint64_t counters_offset;
    // One deferred-completion registration scratch per physical lane.
    // Reserved for every graph, because whether a kernel
    // registers a condition is not visible at bind -- the record is what lets
    // the runtime see that one did. Placed past `upload_size`, so the bytes
    // cost no DMA and a lane writes its own record's header before its loop.
    uint64_t async_groups_offset;
    uint64_t async_entries_offset;
    uint64_t async_completions_offset;
    uint64_t async_completion_capacity;
    // MIX tasks share the vector list and publish records through cluster SSBUF.
    uint64_t mix_task_count;
    uint64_t mix_signal_ready_offset;
    uint64_t boundary_task_count;
    uint64_t task_claims_offset;
    uint64_t claim_summary_offset;
    uint64_t ticket_interval_count;
    uint64_t sync_cohort_count;
    uint64_t sync_notify_offset;
    uint64_t metadata_offset;
    uint64_t metadata_ready_offset;
    uint64_t segments_offset;
    // Tasks in the task table. `task_count` adds every Graph body task of every
    // invocation, numbered after them.
    uint64_t table_task_count;
    // Graph templates, one blob, and one invocation record per GRAPH task;
    // reserved only when the graph has one.
    uint64_t graph_invocation_count;
    uint64_t graph_invocations_offset;
    uint64_t graph_prepare_offset;
    uint64_t graph_templates_offset;
    uint64_t graph_templates_bytes;
    // The uploaded prefix includes zeroed body metadata-ready tokens; AICPU
    // materialization publishes those tokens after the body descriptors.
    uint64_t upload_size;
    uint64_t profile_buffers_offset;
    uint64_t task_records_offset;
    uint64_t task_record_stride;
    uint64_t scalar_records_offset;
    uint64_t event_records_offset;
    uint32_t task_record_capacity;
    uint32_t event_record_capacity;
    uint32_t profiling_level;
};

/** The Graph part of the region, sized by the caller from its templates. */
struct GraphLayoutInput {
    uint64_t body_task_count{0};
    uint64_t invocation_count{0};
    uint64_t templates_bytes{0};
};

struct HostBodyTask {
    uint32_t subtask_total;
    uint32_t flags;
    PercounterTaskMetadata metadata;
};

/** A template still in host memory, before it is copied into the region. */
struct HostGraphTemplate {
    std::vector<HostBodyTask> tasks;
    std::vector<int32_t> fanin_offsets;  // task_count + 1
    std::vector<int32_t> fanin_indices;
    // Order entries by list, carrying body indices as task ids.
    std::vector<uint64_t> orders[PERCOUNTER_LIST_COUNT];
    uint64_t sink_subtask_total{0};
    uint64_t executable_blocks{0};
    uint64_t boundary_task_count{0};
    uint64_t mix_task_count{0};
    uint64_t sync_cohort_count{0};
    bool has_predicates{false};
    uint64_t depth{0};
    uint64_t virtual_duration{0};
    uint8_t active_mask{0};
};

/** Bytes one template takes in the region, its header included. */
inline uint64_t graph_template_bytes(const HostGraphTemplate &tmpl) {
    uint64_t bytes = sizeof(PercounterGraphTemplate);
    bytes += tmpl.tasks.size() * sizeof(PercounterBodyTask);
    bytes += tmpl.fanin_offsets.size() * sizeof(int32_t);
    bytes = (bytes + alignof(uint64_t) - 1) & ~uint64_t{alignof(uint64_t) - 1};
    bytes += tmpl.fanin_indices.size() * sizeof(uint64_t);
    return (bytes + alignof(PercounterGraphTemplate) - 1) & ~uint64_t{alignof(PercounterGraphTemplate) - 1};
}

/** One GRAPH task of the task table and the template its body follows. */
struct GraphInvocationInput {
    int64_t shell_task_id;
    uint32_t template_index;
};

/** The Graph half of build_tables' input. */
struct GraphPlanInput {
    const HostGraphTemplate *templates{nullptr};
    size_t template_count{0};
    const GraphInvocationInput *invocations{nullptr};
    size_t invocation_count{0};
};

/** What a lane needs that lives outside this region, resolved by the host. */
struct ExternalAddresses {
    uint64_t graph_storage = 0;
    // Entries in the array `graph_storage` points at -- the task table's size,
    // not percounter's task count, which also counts Graph body tasks.
    uint64_t graph_storage_count = 0;
    uint64_t callable_table = 0;
    uint64_t callable_count = 0;
    // One CntNotify SET/read address per sync cohort, in cohort order.
    const uint64_t *sync_notify = nullptr;
    uint64_t sync_notify_count = 0;
    const SchedulerCompletionCounterPlan *completion_plan = nullptr;
    const uint64_t *completion_notify = nullptr;
    uint64_t completion_notify_count = 0;
    // One slot per Graph body task, in invocation and body order.
    const uint64_t *body_completion_notify = nullptr;
    uint64_t body_completion_notify_count = 0;
};

/**
 * Plan the percounter region.
 *
 * `order_count[t]` is how many *entries* of core type `t` will be placed -- one
 * per executable task, including one boundary entry for each SPMD task. It is a
 * parameter because the caller has already walked the graph to classify shapes,
 * and two walks could disagree. `block_total` is the same walk's count of every
 * (task, block) pair, which bounds what the lists may sum to.
 *
 * There are no per-cluster lists: the host does not know the cluster count at
 * bind time -- the AICPU discovers it after the handshake -- so one list per core
 * type, drawn by every lane of that type, is what needs no topology.
 *
 * `task_count` is the task table's. With Graph tasks, the list counts and
 * `block_total` include every invocation's body, and `graph` sizes the rest.
 */
inline bool plan_layout(
    uint64_t task_count, uint64_t block_total, uint64_t edge_count,
    const uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT], PercounterLayout *layout, uint64_t mix_task_count = 0,
    uint64_t boundary_task_count = 0, uint64_t sync_cohort_count = 0, const GraphLayoutInput &graph = {},
    uint64_t execution_count = 0, uint32_t profiling_level = 0
) {
    if (layout == nullptr || order_count == nullptr) return false;
    const uint64_t table_task_count = task_count;
    if (graph.body_task_count > UINT32_MAX || table_task_count > UINT32_MAX - graph.body_task_count) return false;
    task_count = table_task_count + graph.body_task_count;
    if (block_total < task_count) return false;
    if (mix_task_count > order_count[1]) return false;
    if (sync_cohort_count > boundary_task_count || boundary_task_count > task_count ||
        boundary_task_count > UINT32_MAX - PERCOUNTER_LIST_COUNT)
        return false;
    uint64_t placed = 0;
    for (uint32_t t = 0; t < PERCOUNTER_CORE_TYPE_COUNT; ++t) {
        if (order_count[t] > block_total) return false;
        placed += order_count[t];
    }
    // Each executable task has one entry; inline-completed tasks have none.
    if (placed > task_count || boundary_task_count > placed) return false;

    PercounterLayout next{};
    next.profiling_level = profiling_level;
    next.task_record_capacity = static_cast<uint32_t>(std::min(block_total, uint64_t{65536}));
    if (profiling_level >= 2) next.event_record_capacity = std::min(next.task_record_capacity * 8, 16384U);
    next.task_count = task_count;
    next.edge_count = edge_count;
    for (uint32_t t = 0; t < PERCOUNTER_CORE_TYPE_COUNT; ++t)
        next.order_count[t] = order_count[t];
    next.mix_task_count = mix_task_count;
    next.boundary_task_count = boundary_task_count;
    next.ticket_interval_count = std::min(placed - boundary_task_count, boundary_task_count + PERCOUNTER_LIST_COUNT);
    next.sync_cohort_count = sync_cohort_count;
    next.table_task_count = table_task_count;
    next.graph_invocation_count = graph.invocation_count;
    next.graph_templates_bytes = graph.templates_bytes;
    next.entry_count = placed;
    next.async_completion_capacity = execution_count;

    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define PERCOUNTER_RESERVE(count, type, field)                      \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(&cursor, bytes, alignof(type), &next.field))

    if (!scheduler_layout_reserve(
            &cursor, sizeof(PercounterControl), alignof(PercounterControl), &next.control_offset
        ) ||
        !PERCOUNTER_RESERVE(next.ticket_interval_count, PercounterTicket, tickets_offset) ||
        !PERCOUNTER_RESERVE(order_count[0], uint64_t, orders_offset[0]) ||
        !PERCOUNTER_RESERVE(order_count[1], uint64_t, orders_offset[1]) ||
        (mix_task_count != 0 && !PERCOUNTER_RESERVE(SCHEDULER_CLUSTER_CAPACITY, uint64_t, mix_signal_ready_offset)) ||
        !PERCOUNTER_RESERVE(
            (boundary_task_count + PERCOUNTER_LIST_COUNT + 63) / 64, PercounterClaimSummary, claim_summary_offset
        ) ||
        !PERCOUNTER_RESERVE(sync_cohort_count, uint64_t, sync_notify_offset) ||
        !PERCOUNTER_RESERVE(task_count, PercounterTaskMetadata, metadata_offset) ||
        !PERCOUNTER_RESERVE(task_count, uint64_t, metadata_ready_offset) ||
        !PERCOUNTER_RESERVE(boundary_task_count + PERCOUNTER_LIST_COUNT, PercounterSegment, segments_offset) ||
        (graph.invocation_count != 0 &&
         (!PERCOUNTER_RESERVE(graph.invocation_count, PercounterGraphInvocation, graph_invocations_offset) ||
          !PERCOUNTER_RESERVE(1, PercounterGraphPrepare, graph_prepare_offset) ||
          !scheduler_layout_reserve(
              &cursor, graph.templates_bytes, alignof(PercounterGraphTemplate), &next.graph_templates_offset
          ))) ||
        !PERCOUNTER_RESERVE(table_task_count, PercounterFanin, fanin_offset) ||
        !PERCOUNTER_RESERVE(edge_count, uint64_t, fanin_addr_offset) ||
        !PERCOUNTER_RESERVE(table_task_count, uint64_t, sinks_offset) ||
        (profiling_level >= 2 &&
         !PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, PercounterProfileBuffer, profile_buffers_offset)) ||
        !PERCOUNTER_RESERVE(task_count, PercounterCounter, counters_offset) ||
        !PERCOUNTER_RESERVE(boundary_task_count, PercounterTaskClaim, task_claims_offset) ||
        // The async records and scratch remain outside `upload_size`.
        // The entries follow the records for the reason `layout_valid` checks:
        // a record's entry span has to start after the record itself.
        !PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, pto2::resident::AsyncGroupData, async_groups_offset) ||
        !PERCOUNTER_RESERVE(
            SCHEDULER_WORKER_CAPACITY * PERCOUNTER_ASYNC_CAPACITY, DeferredCompletionEntry, async_entries_offset
        ) ||
        !PERCOUNTER_RESERVE(execution_count, PercounterAsyncCompletion, async_completions_offset) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) {
#undef PERCOUNTER_RESERVE
        return false;
    }
#undef PERCOUNTER_RESERVE
    next.upload_size = next.task_claims_offset + boundary_task_count * sizeof(PercounterTaskClaim);

    cursor = next.total_size;
    if (profiling_level >= 2 &&
        (!scheduler_layout_checked_align(
             uint64_t{next.task_record_capacity} * sizeof(PercounterTaskTiming), 64, &next.task_record_stride
         ) ||
         !scheduler_layout_checked_mul(SCHEDULER_WORKER_CAPACITY, next.task_record_stride, &bytes) ||
         !scheduler_layout_reserve(&cursor, bytes, 64, &next.task_records_offset)))
        return false;
    if (profiling_level >= 3 && (!scheduler_layout_checked_mul(
                                     uint64_t{SCHEDULER_WORKER_CAPACITY} * SCHEDULER_SCALAR_RECORD_CAPACITY,
                                     sizeof(SchedulerScalarRecord), &bytes
                                 ) ||
                                 !scheduler_layout_reserve(&cursor, bytes, 64, &next.scalar_records_offset)))
        return false;
    if (profiling_level >= 2 &&
        (!scheduler_layout_checked_mul(
             uint64_t{SCHEDULER_WORKER_CAPACITY} * next.event_record_capacity, sizeof(PercounterTaskEvent), &bytes
         ) ||
         !scheduler_layout_reserve(&cursor, bytes, 64, &next.event_records_offset)))
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

/**
 * Which list a task belongs in, from its active-subtask mask.
 *
 * A single-slot mask names a core type. A multi-slot mask is a MIX task: its
 * subtasks run on two or three lanes of one cluster at once, so it goes in the
 * MIX list, which a cluster claims as a whole.
 *
 * Two masks are refused. The empty one names no work. **AIV1 alone** is refused
 * because the AIV list is drawn by both vector lanes of every cluster, so such a
 * task would run on whichever lane drew it, giving `sub_block_id` a value its
 * kernel did not ask for.
 */
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
    bool inline_completed;  // already done at bind: counter preset, placed in no list
    int32_t fanin_count;
    const int32_t *fanin_ids;  // producer task ids; may be null when count is 0
    // The private SPMD counter hands out indices 0..block_num-1.
    int32_t block_num{1};
    // Subtask slots the task occupies at once: 1 for an AIC or AIV task, 2 or 3
    // for a MIX one. With `block_num` it gives the subtask total the counter is
    // preset against.
    int32_t subtask_count{1};
    // All blocks must commit to entry before any may launch.
    bool sync_start{false};
    // Carries a dispatch predicate, evaluated on the lane before the kernel runs.
    bool has_predicate{false};
    // No lane claims this task and it is not finished either: it takes no entry,
    // its counter is preset like any other task's, and something outside the
    // lanes publishes it. Mutually exclusive with `inline_completed`, which is
    // already published at bind.
    bool published_off_lane{false};
    // A GRAPH task: the index of its invocation in GraphPlanInput, or -1. It
    // runs nothing itself. It takes one range per list for its body instead of
    // entries, and its counter is preset against its body's sink subtasks, which
    // publish it.
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
        return !task.inline_completed && !task.published_off_lane && task.graph_invocation < 0 &&
               task.active_mask == 0 && task.block_num == 1 && task.subtask_count == 1 && !task.has_predicate &&
               !task.sync_start && !task.is_mix;
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

inline bool plan_percounter_completion_counters(
    const std::vector<TaskInput> &tasks, SchedulerCompletionCounterPlan *plan, std::string *error,
    const GraphPlanInput &graph = {}
) {
    std::vector<int64_t> order(tasks.size());
    std::iota(order.begin(), order.end(), int64_t{0});
    std::vector<std::vector<int64_t>> fanin(tasks.size());
    std::vector<uint32_t> credits(tasks.size(), 0);
    for (size_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        // A published node can satisfy its consumers without a device fanin probe.
        if (task.inline_completed || task.published_off_lane) continue;
        if (task.fanin_count < 0 || (task.fanin_count != 0 && task.fanin_ids == nullptr)) {
            *error = "invalid completion counter fanin";
            return false;
        }
        if (task.fanin_count != 0) fanin[id].assign(task.fanin_ids, task.fanin_ids + task.fanin_count);
        int64_t contribution = static_cast<int64_t>(task.block_num) * task.subtask_count;
        if (task.graph_invocation >= 0) {
            const auto invocation = static_cast<size_t>(task.graph_invocation);
            if (invocation >= graph.invocation_count ||
                graph.invocations[invocation].template_index >= graph.template_count) {
                *error = "invalid Graph for completion counter";
                return false;
            }
            contribution = graph.templates[graph.invocations[invocation].template_index].sink_subtask_total;
        }
        if (contribution <= 0 || contribution > UINT32_MAX) {
            *error = "invalid completion credit count";
            return false;
        }
        credits[id] = static_cast<uint32_t>(contribution);
    }
    return scheduler_plan_completion_counters(fanin, order, credits, plan, error);
}

/** Why build_tables refused. Every one of these fails the bind. */
enum class BuildStatus : uint8_t {
    OK = 0,
    INVALID_ARGUMENT,
    BAD_CORE_TYPE,
    ORDER_COUNT_MISMATCH,  // the plan's list lengths disagree with the tasks
    EDGE_COUNT_MISMATCH,   // the plan's edge total disagrees with the tasks
    PRODUCER_NOT_BEFORE_CONSUMER,
    PRODUCER_OUT_OF_RANGE,
};

inline const char *build_status_name(BuildStatus status) {
    switch (status) {
    case BuildStatus::OK:
        return "ok";
    case BuildStatus::INVALID_ARGUMENT:
        return "invalid argument";
    case BuildStatus::BAD_CORE_TYPE:
        return "core type out of range";
    case BuildStatus::ORDER_COUNT_MISMATCH:
        return "order count disagrees with the task list";
    case BuildStatus::EDGE_COUNT_MISMATCH:
        return "edge count disagrees with the task list";
    case BuildStatus::PRODUCER_NOT_BEFORE_CONSUMER:
        return "a fanin edge does not point backwards";
    case BuildStatus::PRODUCER_OUT_OF_RANGE:
        return "a fanin edge names no task";
    }
    return "?";
}

/**
 * A value no two binds of this process share, and never zero.
 *
 * The deferred-completion records sit past `upload_size`, in storage a pipeline
 * slot retains between runs, so a record can still hold the bytes an earlier
 * bind left there. The epoch is what a kernel's writer compares before it
 * treats a record as this run's, and zero is the writer's own "no record".
 */
inline uint64_t next_async_run_epoch() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

/** The task a failure is about, for the log line. -1 when it is not about one. */
struct BuildResult {
    BuildStatus status = BuildStatus::OK;
    int64_t task_id = -1;
    int32_t fanin_index = -1;
};

template <typename T>
inline T *region_at(void *base, uint64_t offset) {
    return reinterpret_cast<T *>(static_cast<uint8_t *>(base) + offset);
}

/**
 * Fill the percounter region.
 *
 * `base` is the host staging mirror; `device_base` is where that image will
 * land, and is what the stored fanin addresses are computed against. Only the
 * first `layout.upload_size` bytes are initialized, including the body
 * metadata-ready tokens that materialization will publish.
 *
 * Ascending task id is a topological order in hbg: a producer is always minted
 * before its consumer. The ordering below rests on that, so it is checked per
 * edge rather than trusted -- O(edges), and a wrong assumption becomes a bind
 * failure rather than a device spinning on a counter nobody will set.
 */
inline bool build_tables(
    const TaskInput *tasks, uint64_t task_count, const PercounterLayout &layout, void *base, uint64_t device_base,
    uint32_t clusters, const ExternalAddresses &external = {}, BuildResult *result = nullptr,
    const GraphPlanInput &graph = {}
) {
    const auto fail = [&](BuildStatus status, int64_t task_id, int32_t fanin_index) {
        if (result != nullptr) *result = BuildResult{status, task_id, fanin_index};
        return false;
    };
    if (result != nullptr) *result = BuildResult{};
    if (base == nullptr || (tasks == nullptr && task_count != 0) || task_count != layout.table_task_count ||
        graph.invocation_count != layout.graph_invocation_count ||
        (graph.invocation_count != 0 && (graph.invocations == nullptr || graph.templates == nullptr)))
        return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
    if (layout.sync_cohort_count != 0 &&
        (external.sync_notify == nullptr || external.sync_notify_count != layout.sync_cohort_count))
        return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
    for (uint64_t i = 0; i < layout.sync_cohort_count; ++i)
        if (a5_cnt_notify_increment_address(external.sync_notify[i]) == 0)
            return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
    if (external.body_completion_notify_count != layout.task_count - task_count ||
        (external.body_completion_notify_count != 0 && external.body_completion_notify == nullptr))
        return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);

    std::memset(base, 0, static_cast<size_t>(layout.upload_size));

    auto *control = region_at<PercounterControl>(base, layout.control_offset);
    for (auto &head : control->first_boundary_segment)
        head = PERCOUNTER_NO_BOUNDARY;
    auto *fanin = region_at<PercounterFanin>(base, layout.fanin_offset);
    auto *fanin_addr = region_at<uint64_t>(base, layout.fanin_addr_offset);
    auto *counters = region_at<PercounterCounter>(base, layout.counters_offset);
    if (external.completion_plan != nullptr) {
        const auto &plan = *external.completion_plan;
        if (external.completion_notify == nullptr || plan.chains.size() != external.completion_notify_count ||
            plan.task_chains.size() != task_count || plan.thresholds.size() != task_count)
            return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
        for (uint64_t id = 0; id < task_count; ++id) {
            const uint32_t chain = plan.task_chains[id];
            if (tasks[id].inline_completed || tasks[id].published_off_lane) continue;
            if (chain >= external.completion_notify_count || plan.thresholds[id] == 0)
                return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
            const uint64_t set = external.completion_notify[chain];
            if (a5_cnt_notify_increment_address(set) == 0)
                return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
            counters[id].completion_notify_address = set;
            counters[id].completion_notify_target = plan.thresholds[id];
        }
    }
    auto *metadata = region_at<PercounterTaskMetadata>(base, layout.metadata_offset);
    for (uint64_t id = 0; id < layout.task_count; ++id)
        metadata[id].sync_index = PERCOUNTER_NO_SYNC;
    auto *metadata_ready = region_at<uint64_t>(base, layout.metadata_ready_offset);
    for (uint64_t id = 0; id < task_count; ++id)
        metadata_ready[id] = 1;
    uint64_t *orders[PERCOUNTER_LIST_COUNT] = {
        region_at<uint64_t>(base, layout.orders_offset[0]),
        region_at<uint64_t>(base, layout.orders_offset[1]),
    };
    const uint64_t list_count[PERCOUNTER_LIST_COUNT] = {layout.order_count[0], layout.order_count[1]};
    // The tasks each list will hold, collected in id order and ordered by
    // the shared resource schedule. A GRAPH task stands for its
    // whole body in every list the body uses.
    std::vector<int64_t> eligible[PERCOUNTER_LIST_COUNT];
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l)
        eligible[l].reserve(static_cast<size_t>(list_count[l]));
    uint64_t edge_cursor = 0;
    // Set by the fanin walk below. What it leaves clear is a top-level sink:
    // no task in the table waits on it, so nothing but its own completion
    // reports that its branch of the graph is finished.
    std::vector<bool> consumed(static_cast<size_t>(task_count), false);
    // Entries, checked against the plan as the walk goes, so an overlong list
    // fails on the task that overran it.
    uint64_t entries[PERCOUNTER_LIST_COUNT] = {};
    uint64_t boundary_tasks = 0;
    uint64_t mix_tasks = 0;
    uint64_t sync_cohorts = 0;
    bool any_predicate = false;
    // Body task ids follow the table's, invocation by invocation.
    std::vector<uint64_t> body_base(graph.invocation_count, 0);
    std::vector<bool> invocation_seen(graph.invocation_count, false);
    uint64_t next_body_id = task_count;
    for (size_t k = 0; k < graph.invocation_count; ++k) {
        const GraphInvocationInput &invocation = graph.invocations[k];
        if (invocation.template_index >= graph.template_count || invocation.shell_task_id < 0 ||
            static_cast<uint64_t>(invocation.shell_task_id) >= task_count ||
            tasks[invocation.shell_task_id].graph_invocation != static_cast<int32_t>(k))
            return fail(BuildStatus::INVALID_ARGUMENT, invocation.shell_task_id, -1);
        body_base[k] = next_body_id;
        next_body_id += graph.templates[invocation.template_index].tasks.size();
    }
    if (next_body_id != layout.task_count) return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
    const auto template_of = [&graph](const TaskInput &task) -> const HostGraphTemplate & {
        return graph.templates[graph.invocations[task.graph_invocation].template_index];
    };
    const auto accept = [&](uint32_t list, uint64_t placed, uint64_t id) {
        entries[list] += placed;
        eligible[list].push_back(static_cast<int64_t>(id));
        return entries[list] <= list_count[list];
    };

    for (uint64_t id = 0; id < task_count; ++id) {
        const TaskInput &task = tasks[id];
        if (task.core_type_index >= PERCOUNTER_CORE_TYPE_COUNT)
            return fail(BuildStatus::BAD_CORE_TYPE, static_cast<int64_t>(id), -1);
        if (task.fanin_count < 0 || (task.fanin_count > 0 && task.fanin_ids == nullptr))
            return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);

        const uint64_t row_begin = edge_cursor;
        auto &row = fanin[id];
        row.count = 0;
        row.begin = static_cast<int32_t>(row_begin);
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            if (edge_cursor >= layout.edge_count)
                return fail(BuildStatus::EDGE_COUNT_MISMATCH, static_cast<int64_t>(id), e);
            const int32_t producer = task.fanin_ids[e];
            if (producer < 0 || static_cast<uint64_t>(producer) >= task_count)
                return fail(BuildStatus::PRODUCER_OUT_OF_RANGE, static_cast<int64_t>(id), e);
            if (static_cast<uint64_t>(producer) >= id)
                return fail(BuildStatus::PRODUCER_NOT_BEFORE_CONSUMER, static_cast<int64_t>(id), e);
            const auto &source = tasks[producer];
            // The device row contains only dependencies that can block at run time.
            if (!source.inline_completed) {
                auto &ref = fanin_addr[row_begin + row.count];
                ref =
                    device_base + layout.counters_offset + static_cast<uint64_t>(producer) * sizeof(PercounterCounter);
                if (counters[producer].completion_notify_address != 0) ref |= PERCOUNTER_COMPLETION_NOTIFY;
                ++row.count;
            }
            // Edge validation and sink discovery use the input DAG.
            consumed[static_cast<size_t>(producer)] = true;
            ++edge_cursor;
        }

        if (task.block_num < 1 || static_cast<uint64_t>(task.block_num) > PERCOUNTER_ORDER_BLOCK_MASK)
            return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
        if (task.subtask_count < 1) return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);

        if (task.has_predicate) any_predicate = true;
        if (task.inline_completed && (task.published_off_lane || task.graph_invocation >= 0))
            return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
        if (task.inline_completed) {
            // Done before the run starts: in no list, and its consumers do not wait.
            counters[id].value = PERCOUNTER_COUNTER_PUBLISHED;
            continue;
        }
        if (task.graph_invocation >= 0) {
            if (static_cast<size_t>(task.graph_invocation) >= graph.invocation_count ||
                invocation_seen[static_cast<size_t>(task.graph_invocation)] || task.sync_start)
                return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
            invocation_seen[static_cast<size_t>(task.graph_invocation)] = true;
            const HostGraphTemplate &tmpl = template_of(task);
            // Published by its body's sinks, one add per sink subtask.
            counters[id].value = percounter_counter_preset(static_cast<int64_t>(tmpl.sink_subtask_total));
            boundary_tasks += tmpl.boundary_task_count;
            mix_tasks += tmpl.mix_task_count;
            sync_cohorts += tmpl.sync_cohort_count;
            if (tmpl.has_predicates) any_predicate = true;
            for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
                if (!tmpl.orders[l].empty() && !accept(l, tmpl.orders[l].size(), id))
                    return fail(BuildStatus::ORDER_COUNT_MISMATCH, static_cast<int64_t>(id), -1);
            }
            continue;
        }
        if (task.published_off_lane) {
            // Preset and left unpublished: its consumers wait, and whatever runs
            // it off-lane walks the counter up. No lane carries it, so it is in
            // no order list.
            counters[id].value = percounter_counter_preset(static_cast<int64_t>(task.block_num) * task.subtask_count);
            continue;
        }
        // Every subtask adds one, so the counter starts this far below the value
        // a consumer reads as published. A task that never runs stays below it.
        counters[id].value = percounter_counter_preset(static_cast<int64_t>(task.block_num) * task.subtask_count);
        // A cohort of one would be released by its own claim, and the
        // orchestrator sets the attribute only above one block. The upper bound
        // keeps the sync sentinel a value no real block index takes.
        if (task.sync_start && (task.block_num < 2 || task.block_num >= PERCOUNTER_ORDER_BLOCK_IDX_SPMD))
            return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
        const uint64_t placed = 1;
        if (task.block_num > 1 || task.is_mix) ++boundary_tasks;
        if (task.is_mix) ++mix_tasks;
        if (task.sync_start) {
            ++sync_cohorts;
        }
        if (!accept(task.is_mix ? 1 : task.core_type_index, placed, id))
            return fail(BuildStatus::ORDER_COUNT_MISMATCH, static_cast<int64_t>(id), -1);
    }
    if (boundary_tasks != layout.boundary_task_count || sync_cohorts != layout.sync_cohort_count ||
        mix_tasks != layout.mix_task_count)
        return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
    for (size_t k = 0; k < graph.invocation_count; ++k) {
        if (!invocation_seen[k]) return fail(BuildStatus::INVALID_ARGUMENT, graph.invocations[k].shell_task_id, -1);
    }

    // A Graph invocation is represented by its shell, whose counter is preset
    // against every sink subtask of the body, so body tasks are not sinks of
    // their own. An inline-completed sink is preset published and reads so at
    // once.
    auto *sinks = region_at<uint64_t>(base, layout.sinks_offset);
    uint64_t sink_count = 0;
    for (uint64_t id = 0; id < task_count; ++id) {
        if (consumed[static_cast<size_t>(id)]) continue;
        sinks[sink_count++] = id;
    }

    std::vector<ResourceOrderTask> order_tasks(task_count);
    for (uint64_t id = 0; id < task_count; ++id) {
        const auto &task = tasks[id];
        auto &node = order_tasks[id];
        for (int32_t e = 0; e < task.fanin_count; ++e)
            node.predecessors.push_back(static_cast<uint32_t>(task.fanin_ids[e]));
        if (task.graph_invocation >= 0) {
            const auto &body = template_of(task);
            node.active_mask = body.active_mask;
            node.reserve_types = true;
            node.duration = body.virtual_duration;
            node.depth = body.depth;
        } else if (task.inline_completed || task.published_off_lane) {
            node.duration = 0;
        } else {
            node.active_mask = task.active_mask;
            node.blocks = static_cast<uint32_t>(task.block_num);
            node.sync = task.sync_start;
            if (task.active_mask == 0) node.duration = 0;
        }
    }
    const auto schedule = make_resource_order(order_tasks, clusters);
    if (!schedule) return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
    const auto &global_order = schedule->tasks;
    std::vector<uint32_t> position(task_count);
    for (uint32_t rank = 0; rank < global_order.size(); ++rank)
        position[global_order[rank]] = rank;
    const auto by_schedule = [&position](int64_t a, int64_t b) {
        return position[static_cast<size_t>(a)] < position[static_cast<size_t>(b)];
    };
    uint32_t global_rank = 0;
    uint32_t sync_index = 0;
    for (int64_t id : global_order) {
        const TaskInput &task = tasks[id];
        if (task.graph_invocation >= 0) {
            const auto &tmpl = template_of(task);
            const uint64_t body = body_base[task.graph_invocation];
            for (size_t b = 0; b < tmpl.tasks.size(); ++b) {
                metadata[body + b] = tmpl.tasks[b].metadata;
                counters[body + b].value = percounter_counter_preset(tmpl.tasks[b].subtask_total);
                const uint64_t set = external.body_completion_notify[body + b - task_count];
                if (a5_cnt_notify_increment_address(set) == 0)
                    return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(body + b), -1);
                counters[body + b].completion_notify_address = set;
                counters[body + b].completion_notify_target = tmpl.tasks[b].subtask_total;
                metadata[body + b].global_order_id += global_rank;
                if (metadata[body + b].sync_index != PERCOUNTER_NO_SYNC) metadata[body + b].sync_index = sync_index++;
            }
            global_rank += static_cast<uint32_t>(tmpl.tasks.size());
        } else {
            metadata[id] = {
                global_rank++, static_cast<uint16_t>(task.active_mask), static_cast<uint16_t>(task.block_num),
                task.sync_start ? sync_index++ : PERCOUNTER_NO_SYNC, PERCOUNTER_NO_BOUNDARY
            };
        }
    }
    if (sync_index != layout.sync_cohort_count) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
        std::stable_sort(eligible[l].begin(), eligible[l].end(), by_schedule);
        if (entries[l] != list_count[l]) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
        uint64_t cursor = 0;
        for (int64_t id : eligible[l]) {
            const TaskInput &placed_task = tasks[static_cast<size_t>(id)];
            if (placed_task.graph_invocation >= 0) {
                const size_t invocation = static_cast<size_t>(placed_task.graph_invocation);
                for (uint64_t entry : template_of(placed_task).orders[l]) {
                    orders[l][cursor++] = percounter_order_entry(
                        body_base[invocation] + percounter_order_task_id(entry), percounter_order_block_idx(entry),
                        percounter_order_block_num(entry)
                    );
                }
                continue;
            }
            const int32_t blocks = placed_task.block_num;
            if (blocks > 1) {
                orders[l][cursor++] = percounter_order_entry(id, PERCOUNTER_ORDER_BLOCK_IDX_SPMD, blocks);
                continue;
            }
            orders[l][cursor++] = percounter_order_entry(id, 0, 1);
        }
        if (cursor != list_count[l]) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
    }
    if (edge_cursor != layout.edge_count) return fail(BuildStatus::EDGE_COUNT_MISMATCH, -1, -1);

    auto *segments = region_at<PercounterSegment>(base, layout.segments_offset);
    auto *summaries = region_at<PercounterClaimSummary>(base, layout.claim_summary_offset);
    uint64_t segment_cursor = 0;
    uint64_t ticket_interval = 0;
    uint32_t previous_boundary_task[PERCOUNTER_SYNC_DOMAIN_COUNT];
    for (auto &previous : previous_boundary_task)
        previous = PERCOUNTER_NO_BOUNDARY;
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
        control->segment_begin[l] = segment_cursor;
        uint64_t begin = 0;
        for (uint64_t position = 0; position < list_count[l]; ++position) {
            const uint64_t entry = orders[l][position];
            const int64_t id = percounter_order_task_id(entry);
            const bool mix = __builtin_popcount(metadata[id].active_mask) > 1;
            if (!percounter_order_is_spmd(entry) && !mix) continue;
            if (mix || metadata[id].sync_index != PERCOUNTER_NO_SYNC) {
                const uint32_t domain = mix ? 2 : l;
                if (previous_boundary_task[domain] == PERCOUNTER_NO_BOUNDARY)
                    control->first_boundary_segment[domain] = static_cast<uint32_t>(segment_cursor);
                else
                    metadata[previous_boundary_task[domain]].next_boundary_segment =
                        static_cast<uint32_t>(segment_cursor);
                metadata[id].next_boundary_segment = PERCOUNTER_NO_BOUNDARY;
                previous_boundary_task[domain] = static_cast<uint32_t>(id);
            }
            segments[segment_cursor++] = {
                percounter_claim_shape(
                    id, metadata[id].block_num, metadata[id].active_mask, metadata[id].sync_index != PERCOUNTER_NO_SYNC
                ),
                static_cast<uint32_t>(begin), static_cast<uint32_t>(position), ticket_interval
            };
            if (begin < position) ++ticket_interval;
            if (metadata[id].sync_index != PERCOUNTER_NO_SYNC) {
                const uint32_t domain = mix ? 2 : l;
                control->max_sync_blocks[domain] =
                    std::max(control->max_sync_blocks[domain], uint64_t(metadata[id].block_num));
            }
            begin = position + 1;
        }
        if (begin == list_count[l]) summaries[segment_cursor / 64].closed |= uint64_t{1} << (segment_cursor % 64);
        segments[segment_cursor++] = {
            0, static_cast<uint32_t>(begin), static_cast<uint32_t>(list_count[l]), ticket_interval
        };
        if (begin < list_count[l]) ++ticket_interval;
        control->segment_count[l] = segment_cursor - control->segment_begin[l];
    }
    if (segment_cursor != layout.boundary_task_count + PERCOUNTER_LIST_COUNT ||
        ticket_interval > layout.ticket_interval_count)
        return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);

    // Templates, back to back, then one record per invocation pointing at its own.
    std::vector<uint64_t> template_offset(graph.template_count, 0);
    uint64_t template_cursor = 0;
    for (size_t t = 0; t < graph.template_count; ++t) {
        const HostGraphTemplate &tmpl = graph.templates[t];
        const uint64_t bytes = graph_template_bytes(tmpl);
        if (template_cursor + bytes > layout.graph_templates_bytes) return fail(BuildStatus::INVALID_ARGUMENT, -1, -1);
        const uint64_t at = layout.graph_templates_offset + template_cursor;
        template_offset[t] = at;
        auto *header = region_at<PercounterGraphTemplate>(base, at);
        *header = PercounterGraphTemplate{};
        header->task_count = tmpl.tasks.size();
        header->edge_count = tmpl.fanin_indices.size();
        uint64_t field = at + sizeof(PercounterGraphTemplate);
        const auto place = [&](const void *src, uint64_t len, uint64_t align) {
            field = (field + align - 1) & ~(align - 1);
            const uint64_t where = field;
            if (len != 0) std::memcpy(region_at<uint8_t>(base, where), src, static_cast<size_t>(len));
            field += len;
            return where;
        };
        std::vector<PercounterBodyTask> body;
        body.reserve(tmpl.tasks.size());
        for (const auto &task : tmpl.tasks)
            body.push_back({task.subtask_total, task.flags});
        header->tasks_offset =
            place(body.data(), body.size() * sizeof(PercounterBodyTask), alignof(PercounterBodyTask));
        header->fanin_offsets_offset =
            place(tmpl.fanin_offsets.data(), tmpl.fanin_offsets.size() * sizeof(int32_t), alignof(int32_t));
        const std::vector<uint64_t> producers(tmpl.fanin_indices.begin(), tmpl.fanin_indices.end());
        header->fanin_indices_offset = place(producers.data(), producers.size() * sizeof(uint64_t), alignof(uint64_t));
        template_cursor += bytes;
    }
    auto *invocations = layout.graph_invocation_count != 0 ?
                            region_at<PercounterGraphInvocation>(base, layout.graph_invocations_offset) :
                            nullptr;
    // Records go in the order the AICPU should materialize them: by their GRAPH
    // task's key in the lists, so the body the lanes reach first exists first.
    std::vector<size_t> rank(graph.invocation_count);
    for (size_t k = 0; k < graph.invocation_count; ++k)
        rank[k] = k;
    std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) {
        return by_schedule(graph.invocations[a].shell_task_id, graph.invocations[b].shell_task_id);
    });
    for (size_t r = 0; r < graph.invocation_count; ++r) {
        const size_t k = rank[r];
        PercounterGraphInvocation &record = invocations[r];
        record = PercounterGraphInvocation{};
        record.template_offset = template_offset[graph.invocations[k].template_index];
        record.shell_task_id = static_cast<uint64_t>(graph.invocations[k].shell_task_id);
        record.body_base = body_base[k];
    }

    control->profiling_level = layout.profiling_level;
    control->profile_buffers_offset = layout.profile_buffers_offset;
    if (layout.profile_buffers_offset != 0) {
        auto *buffers = region_at<PercounterProfileBuffer>(base, layout.profile_buffers_offset);
        for (uint32_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
            buffers[worker].task_capacity = layout.task_record_capacity;
            buffers[worker].event_capacity = layout.event_record_capacity;
            buffers[worker].event_records_offset = layout.event_records_offset + uint64_t{worker} *
                                                                                     layout.event_record_capacity *
                                                                                     sizeof(PercounterTaskEvent);
            buffers[worker].task_records_offset =
                layout.task_records_offset + uint64_t{worker} * layout.task_record_stride;
            buffers[worker].scalar_records_offset =
                layout.scalar_records_offset == 0 ?
                    0 :
                    layout.scalar_records_offset +
                        uint64_t{worker} * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord);
            buffers[worker].physical_core_id = -1;
        }
    }
    control->task_count = layout.task_count;
    control->table_task_count = task_count;
    control->sink_count = sink_count;
    control->sinks_offset = layout.sinks_offset;
    control->edge_count = layout.edge_count;
    control->tickets_offset = layout.tickets_offset;
    control->mix_task_count = layout.mix_task_count;
    control->boundary_task_count = layout.boundary_task_count;
    control->task_claims_offset = layout.task_claims_offset;
    control->claim_summary_offset = layout.claim_summary_offset;
    control->sync_cohort_count = layout.sync_cohort_count;
    control->sync_notify_offset = layout.sync_notify_offset;
    auto *sync_notify = region_at<uint64_t>(base, layout.sync_notify_offset);
    for (uint64_t cohort = 0; cohort < layout.sync_cohort_count; ++cohort)
        sync_notify[cohort] = external.sync_notify[cohort];
    control->metadata_offset = layout.metadata_offset;
    control->metadata_ready_offset = layout.metadata_ready_offset;
    control->segments_offset = layout.segments_offset;
    control->graph_invocation_count = layout.graph_invocation_count;
    control->graph_invocations_offset = layout.graph_invocations_offset;
    control->graph_prepare_offset = layout.graph_prepare_offset;
    control->has_predicates = any_predicate ? 1 : 0;
    control->mix_signal_ready_offset = layout.mix_signal_ready_offset;
    control->fanin_offset = layout.fanin_offset;
    control->fanin_addr_offset = layout.fanin_addr_offset;
    control->counters_offset = layout.counters_offset;
    control->image_bytes = layout.total_size;
    control->async_groups_offset = layout.async_groups_offset;
    control->async_entries_offset = layout.async_entries_offset;
    control->async_capacity = PERCOUNTER_ASYNC_CAPACITY;
    control->async_completions_offset = layout.async_completions_offset;
    control->async_completion_capacity = layout.async_completion_capacity;
    control->async_run_epoch = next_async_run_epoch();
    auto *claims = region_at<PercounterTaskClaim>(base, layout.task_claims_offset);
    const uint64_t aic_boundaries = control->segment_count[0] - 1;
    for (uint64_t index = 0; index < layout.boundary_task_count; ++index) {
        const uint64_t segment = index + (index >= aic_boundaries ? 1 : 0);
        const uint32_t blocks = static_cast<uint32_t>(segments[segment].task_shape >> 48);
        // The upload prefix was zeroed above, including every claim sector. An
        // empty sector has no owner to publish its bit, so it starts published;
        // the device decides emptiness by this same count.
        uint64_t closed = 0;
        for (uint32_t part = 0; part < PERCOUNTER_CLAIM_SECTOR_COUNT; ++part)
            if ((blocks + PERCOUNTER_CLAIM_SECTOR_COUNT - 1 - part) / PERCOUNTER_CLAIM_SECTOR_COUNT == 0)
                closed |= uint64_t{1} << part;
        claims[index].closed_sectors = closed;
    }
    control->scheduler_timeout_cycles = SCHEDULER_TIMEOUT_CYCLES;
    control->graph_storage_address = external.graph_storage;
    control->graph_storage_count = external.graph_storage_count;
    control->callable_addresses_address = external.callable_table;
    control->callable_addresses_count = external.callable_count;
    for (uint32_t t = 0; t < PERCOUNTER_CORE_TYPE_COUNT; ++t) {
        control->order_count[t] = layout.order_count[t];
        control->orders_offset[t] = layout.orders_offset[t];
    }
    return true;
}

}  // namespace simpler::percounter
