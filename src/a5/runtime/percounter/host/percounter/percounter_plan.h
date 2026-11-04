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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "scheduler/percounter/percounter_types.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace simpler::hbg::percounter {

/** Host-side plan of the percounter region. */
struct PercounterLayout {
    uint64_t total_size;
    uint64_t task_count;
    uint64_t edge_count;
    // Entries, not tasks: a task with `block_num == N` holds N of them.
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT];
    // Every entry of every list, the MIX list included.
    uint64_t entry_count;
    uint64_t control_offset;
    uint64_t tickets_offset;
    uint64_t orders_offset[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t fanin_offset;
    uint64_t fanin_addr_offset;
    uint64_t counters_offset;
    uint64_t dispatch_payloads_offset;
    // The MIX list, its tickets, and the per-cluster word saying the cluster's
    // SSBUF line has been cleared. MIX is not a third core type -- a lane is AIC
    // or AIV, never MIX -- so it keeps its own offsets. All four are zero on a
    // graph with no MIX task, which a lane reads as "there is no MIX here".
    uint64_t mix_order_count;
    uint64_t mix_order_offset;
    uint64_t mix_ticket_offset;
    uint64_t mix_signal_ready_offset;
    // The sync-start queue and its ring (descriptors, then per-cohort claim
    // counts), reserved only when the graph has a sync-start task. The count
    // is the number of such tasks.
    uint64_t sync_cohort_count;
    uint64_t sync_cohort_offset;
    uint64_t sync_pending_offset;
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
    // The prefix of the region the host fills and uploads. The counters are the
    // last segment, and a body task's counter is written by the AICPU before
    // any lane can reach it, so the body counters are left out.
    uint64_t upload_size;
};

/** The Graph part of the region, sized by the caller from its templates. */
struct GraphLayoutInput {
    uint64_t body_task_count{0};
    uint64_t invocation_count{0};
    uint64_t templates_bytes{0};
};

/** A template still in host memory, before it is copied into the region. */
struct HostGraphTemplate {
    std::vector<PercounterBodyTask> tasks;
    std::vector<int32_t> fanin_offsets;  // task_count + 1
    std::vector<int32_t> fanin_indices;
    // Order entries by list, carrying body indices as task ids.
    std::vector<uint64_t> orders[PERCOUNTER_LIST_COUNT];
    uint64_t sink_subtask_total{0};
    uint64_t executable_blocks{0};
    uint64_t mix_rendezvous_count{0};
    // Sync-start body tasks, and their blocks, per invocation.
    uint64_t sync_cohort_count{0};
    uint64_t sync_block_total{0};
    bool has_predicates{false};
    uint64_t depth{0};
};

/** Bytes one template takes in the region, its header included. */
inline uint64_t graph_template_bytes(const HostGraphTemplate &tmpl) {
    uint64_t bytes = sizeof(PercounterGraphTemplate);
    bytes += tmpl.tasks.size() * sizeof(PercounterBodyTask);
    bytes += tmpl.fanin_offsets.size() * sizeof(int32_t);
    bytes = (bytes + alignof(uint64_t) - 1) & ~uint64_t{alignof(uint64_t) - 1};
    bytes += tmpl.fanin_indices.size() * sizeof(uint64_t);
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l)
        bytes += tmpl.orders[l].size() * sizeof(uint64_t);
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
};

/**
 * Plan the percounter region.
 *
 * `order_count[t]` is how many *entries* of core type `t` will be placed -- one
 * per (task, block) pair, or one arming entry for a sync-start task. It is a
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
    const uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT], PercounterLayout *layout, uint64_t mix_order_count = 0,
    uint64_t sync_cohort_count = 0, const GraphLayoutInput &graph = {}
) {
    if (layout == nullptr || order_count == nullptr) return false;
    const uint64_t table_task_count = task_count;
    if (graph.body_task_count > UINT32_MAX || table_task_count > UINT32_MAX - graph.body_task_count) return false;
    task_count = table_task_count + graph.body_task_count;
    if (block_total < task_count) return false;
    if (mix_order_count > block_total) return false;
    if (sync_cohort_count > task_count) return false;
    uint64_t placed = mix_order_count;
    for (uint32_t t = 0; t < PERCOUNTER_CORE_TYPE_COUNT; ++t) {
        if (order_count[t] > block_total) return false;
        placed += order_count[t];
    }
    // Every (task, block) pair is placed at most once, so the lists cannot
    // together exceed the block total. They can be shorter: inline-completed
    // tasks belong in none of them.
    if (placed > block_total) return false;

    PercounterLayout next{};
    next.task_count = task_count;
    next.edge_count = edge_count;
    for (uint32_t t = 0; t < PERCOUNTER_CORE_TYPE_COUNT; ++t)
        next.order_count[t] = order_count[t];
    next.mix_order_count = mix_order_count;
    next.sync_cohort_count = sync_cohort_count;
    next.table_task_count = table_task_count;
    next.graph_invocation_count = graph.invocation_count;
    next.graph_templates_bytes = graph.templates_bytes;
    next.entry_count = placed;

    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define PERCOUNTER_RESERVE(count, type, field)                      \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(&cursor, bytes, alignof(type), &next.field))

    if (!scheduler_layout_reserve(
            &cursor, sizeof(PercounterControl), alignof(PercounterControl), &next.control_offset
        ) ||
        !PERCOUNTER_RESERVE(
            static_cast<uint64_t>(PERCOUNTER_CORE_TYPE_COUNT) * PERCOUNTER_TICKET_CAPACITY, PercounterTicket,
            tickets_offset
        ) ||
        !PERCOUNTER_RESERVE(order_count[0], uint64_t, orders_offset[0]) ||
        !PERCOUNTER_RESERVE(order_count[1], uint64_t, orders_offset[1]) ||
        (mix_order_count != 0 &&
         (!PERCOUNTER_RESERVE(mix_order_count, int64_t, mix_order_offset) ||
          !PERCOUNTER_RESERVE(PERCOUNTER_MIX_TICKET_COUNT, PercounterTicket, mix_ticket_offset) ||
          !PERCOUNTER_RESERVE(SCHEDULER_CLUSTER_CAPACITY, uint64_t, mix_signal_ready_offset))) ||
        (sync_cohort_count != 0 && (!PERCOUNTER_RESERVE(1, PercounterSyncQueue, sync_cohort_offset) ||
                                    !PERCOUNTER_RESERVE(2 * sync_cohort_count, uint64_t, sync_pending_offset))) ||
        (graph.invocation_count != 0 &&
         (!PERCOUNTER_RESERVE(graph.invocation_count, PercounterGraphInvocation, graph_invocations_offset) ||
          !PERCOUNTER_RESERVE(1, PercounterGraphPrepare, graph_prepare_offset) ||
          !scheduler_layout_reserve(
              &cursor, graph.templates_bytes, alignof(PercounterGraphTemplate), &next.graph_templates_offset
          ))) ||
        !PERCOUNTER_RESERVE(table_task_count, PercounterFanin, fanin_offset) ||
        !PERCOUNTER_RESERVE(edge_count, uint64_t, fanin_addr_offset) ||
        !PERCOUNTER_RESERVE(SCHEDULER_WORKER_CAPACITY, DispatchPayload, dispatch_payloads_offset) ||
        !PERCOUNTER_RESERVE(task_count, PercounterCounter, counters_offset) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) {
#undef PERCOUNTER_RESERVE
        return false;
    }
#undef PERCOUNTER_RESERVE
    next.upload_size = next.counters_offset + table_task_count * sizeof(PercounterCounter);

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

/** One task, as the table builder needs to see it. */
struct TaskInput {
    // 0 for AIC and 1 for AIV. Ignored when `is_mix` is set: a MIX task occupies
    // two or three lanes of one cluster and belongs to no single core type.
    uint8_t core_type_index;
    bool is_mix;
    bool inline_completed;  // already done at bind: counter preset, placed in no list
    int32_t fanin_count;
    const int32_t *fanin_ids;  // producer task ids; may be null when count is 0
    // SPMD blocks: the task takes that many consecutive entries in its list,
    // carrying indices 0..block_num-1.
    int32_t block_num{1};
    // Subtask slots the task occupies at once: 1 for an AIC or AIV task, 2 or 3
    // for a MIX one. With `block_num` it gives the subtask total the counter is
    // preset against.
    int32_t subtask_count{1};
    // The task's blocks must be co-resident. It takes a single arming entry
    // instead of one per block, and the blocks are handed out from the sync
    // queue, because a list allocates work that may run in turn.
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
};

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
 * first `layout.upload_size` bytes are written: the body counters past them are
 * the AICPU's.
 *
 * Ascending task id is a topological order in hbg: a producer is always minted
 * before its consumer. The ordering below rests on that, so it is checked per
 * edge rather than trusted -- O(edges), and a wrong assumption becomes a bind
 * failure rather than a device spinning on a counter nobody will set.
 */
inline bool build_tables(
    const TaskInput *tasks, uint64_t task_count, const PercounterLayout &layout, void *base, uint64_t device_base,
    const ExternalAddresses &external = {}, BuildResult *result = nullptr, const GraphPlanInput &graph = {}
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

    std::memset(base, 0, static_cast<size_t>(layout.upload_size));

    auto *control = region_at<PercounterControl>(base, layout.control_offset);
    auto *fanin = region_at<PercounterFanin>(base, layout.fanin_offset);
    auto *fanin_addr = region_at<uint64_t>(base, layout.fanin_addr_offset);
    auto *counters = region_at<PercounterCounter>(base, layout.counters_offset);
    uint64_t *orders[PERCOUNTER_LIST_COUNT] = {
        region_at<uint64_t>(base, layout.orders_offset[0]),
        region_at<uint64_t>(base, layout.orders_offset[1]),
        layout.mix_order_count != 0 ? region_at<uint64_t>(base, layout.mix_order_offset) : nullptr,
    };
    const uint64_t list_count[PERCOUNTER_LIST_COUNT] = {
        layout.order_count[0], layout.order_count[1], layout.mix_order_count
    };
    // The tasks each list will hold, collected in id order and ordered by
    // b-level once the whole graph has been walked. A GRAPH task stands for its
    // whole body in every list the body uses.
    std::vector<int64_t> eligible[PERCOUNTER_LIST_COUNT];
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l)
        eligible[l].reserve(static_cast<size_t>(list_count[l]));
    uint64_t edge_cursor = 0;
    // Entries, checked against the plan as the walk goes, so an overlong list
    // fails on the task that overran it.
    uint64_t entries[PERCOUNTER_LIST_COUNT] = {0, 0, 0};
    // Rendezvous the MIX done ticket will count. A sync-start MIX task's
    // rendezvous report to the sync queue, so it adds nothing here while adding
    // one arming entry to the MIX list.
    uint64_t mix_rendezvous = 0;
    // Blocks a lane will run and fold. Differs from the entry count on exactly
    // the sync-start tasks, whose single entry runs nothing.
    uint64_t executable_blocks = 0;
    uint64_t sync_cohorts = 0;
    uint64_t sync_blocks = 0;
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

        fanin[id].count = task.fanin_count;
        fanin[id].begin = static_cast<int32_t>(edge_cursor);
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            if (edge_cursor >= layout.edge_count)
                return fail(BuildStatus::EDGE_COUNT_MISMATCH, static_cast<int64_t>(id), e);
            const int32_t producer = task.fanin_ids[e];
            if (producer < 0 || static_cast<uint64_t>(producer) >= task_count)
                return fail(BuildStatus::PRODUCER_OUT_OF_RANGE, static_cast<int64_t>(id), e);
            if (static_cast<uint64_t>(producer) >= id)
                return fail(BuildStatus::PRODUCER_NOT_BEFORE_CONSUMER, static_cast<int64_t>(id), e);
            fanin_addr[edge_cursor] =
                device_base + layout.counters_offset + static_cast<uint64_t>(producer) * sizeof(PercounterCounter);
            ++edge_cursor;
        }

        if (task.block_num < 1) return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
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
            executable_blocks += tmpl.executable_blocks;
            mix_rendezvous += tmpl.mix_rendezvous_count;
            sync_cohorts += tmpl.sync_cohort_count;
            sync_blocks += tmpl.sync_block_total;
            if (tmpl.has_predicates) any_predicate = true;
            for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
                if (!tmpl.orders[l].empty() && !accept(l, tmpl.orders[l].size(), id))
                    return fail(BuildStatus::ORDER_COUNT_MISMATCH, static_cast<int64_t>(id), -1);
            }
            continue;
        }
        if (task.published_off_lane) {
            // Preset and left unpublished: its consumers wait, and whatever runs
            // it off-lane walks the counter up. It contributes no executable
            // block, which is what the AICPU's progress watchdog counts.
            counters[id].value = percounter_counter_preset(static_cast<int64_t>(task.block_num) * task.subtask_count);
            continue;
        }
        // Every subtask adds one, so the counter starts this far below the value
        // a consumer reads as published. A task that never runs stays below it.
        counters[id].value = percounter_counter_preset(static_cast<int64_t>(task.block_num) * task.subtask_count);
        // A cohort of one would be released by its own claim, and the
        // orchestrator sets the attribute only above one block. The upper bound
        // keeps the arming sentinel a value no real block index takes.
        if (task.sync_start && (task.block_num < 2 || task.block_num >= PERCOUNTER_ORDER_BLOCK_IDX_ARM))
            return fail(BuildStatus::INVALID_ARGUMENT, static_cast<int64_t>(id), -1);
        const uint64_t placed = task.sync_start ? 1 : static_cast<uint64_t>(task.block_num);
        executable_blocks += static_cast<uint64_t>(task.block_num);
        if (task.sync_start) {
            ++sync_cohorts;
            sync_blocks += static_cast<uint64_t>(task.block_num);
        }
        if (task.is_mix && !task.sync_start) mix_rendezvous += static_cast<uint64_t>(task.block_num);
        if (!accept(task.is_mix ? PERCOUNTER_LIST_MIX : task.core_type_index, placed, id))
            return fail(BuildStatus::ORDER_COUNT_MISMATCH, static_cast<int64_t>(id), -1);
    }
    if (sync_cohorts != layout.sync_cohort_count) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
    for (size_t k = 0; k < graph.invocation_count; ++k) {
        if (!invocation_seen[k]) return fail(BuildStatus::INVALID_ARGUMENT, graph.invocations[k].shell_task_id, -1);
    }

    // Longest path to a sink, per task. The walk above rejects any edge that
    // does not point backwards, so descending ids reach already-final values
    // and one pass suffices. Inline-completed tasks carry a level too: they are
    // in no list but still lengthen the paths that cross them.
    //
    // A GRAPH task weighs its body's depth rather than one: by the time its
    // successors can start, its longest body chain has run.
    std::vector<int32_t> blevel(static_cast<size_t>(task_count), 0);
    for (uint64_t id = task_count; id-- > 0;) {
        const TaskInput &task = tasks[id];
        if (task.graph_invocation >= 0) blevel[id] += static_cast<int32_t>(template_of(task).depth) - 1;
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const size_t producer = static_cast<size_t>(task.fanin_ids[e]);
            blevel[producer] = std::max(blevel[producer], blevel[id] + 1);
        }
    }

    // Order each list by descending b-level, ties by id: `eligible` is already
    // in id order, so a *stable* sort on the level yields (-b-level, id).
    //
    // The level is how much work still hangs off a task, which decides whether
    // running it now unblocks anybody. Depth from a root answers the opposite
    // question. On paged_attention Case1 the two disagree sharply: a chain's
    // last QK has depth 0 and b-level 4, its first PV depth 2 and b-level 65, so
    // depth order puts all 16384 QK tasks ahead of the first PV and the AIV pool
    // drains and spins (55 of 56 AIV lanes blocked for 2.4 ms in one trace).
    // Descending b-level hands each level's work to both pools in step.
    //
    // No deadlock: A -> B implies blevel(A) >= blevel(B) + 1, so a producer is
    // strictly earlier than its consumer under the global key, and tasks at
    // equal level have no edge between them. See percounter_claim_range.
    //
    // A body sits in one range, keyed as its GRAPH task, with its own tasks by
    // level inside it. The argument carries over: a root's producers precede
    // the GRAPH task, the GRAPH task's level is at least one above its
    // successors' however deep the body, and every edge inside the body points
    // forward within the range.
    //
    // Expansion into blocks happens after the sort, which keeps each task's
    // block indices ascending.
    const auto by_blevel = [&blevel](int64_t a, int64_t b) {
        return blevel[static_cast<size_t>(a)] > blevel[static_cast<size_t>(b)];
    };
    std::vector<uint64_t> order_base(graph.invocation_count * PERCOUNTER_LIST_COUNT, 0);
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
        std::stable_sort(eligible[l].begin(), eligible[l].end(), by_blevel);
        if (entries[l] != list_count[l]) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
        uint64_t cursor = 0;
        for (int64_t id : eligible[l]) {
            const TaskInput &placed_task = tasks[static_cast<size_t>(id)];
            if (placed_task.graph_invocation >= 0) {
                // Left zero: an entry of block count zero is one the AICPU has
                // not published yet.
                order_base[static_cast<size_t>(placed_task.graph_invocation) * PERCOUNTER_LIST_COUNT + l] = cursor;
                cursor += template_of(placed_task).orders[l].size();
                continue;
            }
            const int32_t blocks = placed_task.block_num;
            if (placed_task.sync_start) {
                orders[l][cursor++] = percounter_order_entry(id, PERCOUNTER_ORDER_BLOCK_IDX_ARM, blocks);
                continue;
            }
            for (int32_t b = 0; b < blocks; ++b)
                orders[l][cursor++] = percounter_order_entry(id, b, blocks);
        }
        if (cursor != list_count[l]) return fail(BuildStatus::ORDER_COUNT_MISMATCH, -1, -1);
    }
    if (edge_cursor != layout.edge_count) return fail(BuildStatus::EDGE_COUNT_MISMATCH, -1, -1);

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
        header->sink_subtask_total = tmpl.sink_subtask_total;
        header->executable_blocks = tmpl.executable_blocks;
        header->mix_rendezvous_count = tmpl.mix_rendezvous_count;
        header->has_predicates = tmpl.has_predicates ? 1 : 0;
        header->depth = tmpl.depth;
        uint64_t field = at + sizeof(PercounterGraphTemplate);
        const auto place = [&](const void *src, uint64_t len, uint64_t align) {
            field = (field + align - 1) & ~(align - 1);
            const uint64_t where = field;
            if (len != 0) std::memcpy(region_at<uint8_t>(base, where), src, static_cast<size_t>(len));
            field += len;
            return where;
        };
        header->tasks_offset =
            place(tmpl.tasks.data(), tmpl.tasks.size() * sizeof(PercounterBodyTask), alignof(PercounterBodyTask));
        header->fanin_offsets_offset =
            place(tmpl.fanin_offsets.data(), tmpl.fanin_offsets.size() * sizeof(int32_t), alignof(int32_t));
        const std::vector<uint64_t> producers(tmpl.fanin_indices.begin(), tmpl.fanin_indices.end());
        header->fanin_indices_offset = place(producers.data(), producers.size() * sizeof(uint64_t), alignof(uint64_t));
        for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
            header->order_count[l] = tmpl.orders[l].size();
            header->orders_offset[l] =
                place(tmpl.orders[l].data(), tmpl.orders[l].size() * sizeof(uint64_t), alignof(uint64_t));
        }
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
        return by_blevel(graph.invocations[a].shell_task_id, graph.invocations[b].shell_task_id);
    });
    for (size_t r = 0; r < graph.invocation_count; ++r) {
        const size_t k = rank[r];
        PercounterGraphInvocation &record = invocations[r];
        record = PercounterGraphInvocation{};
        record.template_offset = template_offset[graph.invocations[k].template_index];
        record.shell_task_id = static_cast<uint64_t>(graph.invocations[k].shell_task_id);
        record.body_base = body_base[k];
        for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l)
            record.order_base[l] = order_base[k * PERCOUNTER_LIST_COUNT + l];
    }

    control->task_count = layout.task_count;
    control->table_task_count = task_count;
    control->executable_task_count = executable_blocks;
    control->edge_count = layout.edge_count;
    control->tickets_offset = layout.tickets_offset;
    control->mix_order_count = layout.mix_order_count;
    control->mix_rendezvous_count = mix_rendezvous;
    control->sync_cohort_count = layout.sync_cohort_count;
    control->sync_cohort_offset = layout.sync_cohort_offset;
    control->sync_pending_offset = layout.sync_pending_offset;
    control->graph_invocation_count = layout.graph_invocation_count;
    control->graph_invocations_offset = layout.graph_invocations_offset;
    control->graph_prepare_offset = layout.graph_prepare_offset;
    control->sync_block_total = sync_blocks;
    control->has_predicates = any_predicate ? 1 : 0;
    control->mix_order_offset = layout.mix_order_offset;
    control->mix_ticket_offset = layout.mix_ticket_offset;
    control->mix_signal_ready_offset = layout.mix_signal_ready_offset;
    control->fanin_offset = layout.fanin_offset;
    control->fanin_addr_offset = layout.fanin_addr_offset;
    control->counters_offset = layout.counters_offset;
    control->dispatch_payloads_offset = layout.dispatch_payloads_offset;
    // The memset leaves this at 0, which is a real task id. A cohort takes the
    // open slot by flipping its generation and publishes by writing this field
    // last, so a lane gates on a non-negative id to know the shape behind an
    // armed generation is this cohort's. At 0 the very first cohort would read
    // as published between the claim and the payload, and a lane deciding there
    // would read a width of zero and exclude itself from that cohort for good.
    if (layout.sync_cohort_count != 0) {
        region_at<PercounterSyncQueue>(base, layout.sync_cohort_offset)->open_task_id = -1;
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

}  // namespace simpler::hbg::percounter
