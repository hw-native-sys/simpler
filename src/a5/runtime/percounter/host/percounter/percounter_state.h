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
 * The bind's percounter decision and, when percounter takes the graph, its
 * scheduler state.
 *
 * Two entry points, both called from create_scheduler_state:
 *
 *  - check_scheduler_request, before anything else: rejects an unrecognised
 *    SIMPLER_HBG_SCHEDULER_TYPE.
 *  - select_percounter_state, after the default selection's per-task walk:
 *    reuses that walk's validation and metadata, and either builds the
 *    percounter region or logs why the graph falls back. When percounter is
 *    requested the walk passes over GRAPH tasks, which are validated here
 *    through their Definitions.
 *
 * Header-only because the UT targets that compile runtime_maker.cpp list their
 * sources by name.
 */

#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include "common/host_api.h"
#include "common/host_phase_kind.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime.h"
#include "host_build_graph/shared_memory.h"
#include "intrinsic.h"
#include "percounter_graph.h"
#include "percounter_plan.h"
#include "percounter_selection.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_types.h"

namespace simpler::hbg::percounter {

inline bool percounter_requested() {
    SchedulerRequest request = SchedulerRequest::DEFAULT;
    return parse_scheduler_request(std::getenv(SCHEDULER_TYPE_ENV), &request) &&
           request == SchedulerRequest::PERCOUNTER;
}

inline void log_fallback(int32_t total_tasks, const char *reason, int64_t task_id) {
    LOG_INFO(
        "A5 HBG: %s=percounter for %d tasks: task id=%" PRId64 " is %s, which percounter does not run; keeping the "
        "default scheduler selection",
        SCHEDULER_TYPE_ENV, total_tasks, task_id, reason
    );
}

/**
 * Settle this bind's scheduler request before the default selection runs.
 * Returns false, failing the bind, only for an unrecognised value.
 */
inline bool check_scheduler_request() {
    SchedulerRequest request = SchedulerRequest::DEFAULT;
    return resolve_scheduler_request(&request);
}

/** Whether the default selection's walk leaves GRAPH tasks to percounter. */
inline bool percounter_takes_graph_tasks() { return percounter_requested(); }

/** The bind's Graph templates and invocations, gathered from the task table. */
struct GraphInputs {
    std::vector<HostGraphTemplate> templates;
    std::vector<GraphInvocationInput> invocations;
    GraphLayoutInput layout{};
    uint64_t body_blocks{0};
};

/**
 * Build one template per Definition the GRAPH tasks name, reading the bind's
 * Definition staging block, and one invocation per GRAPH task.
 *
 * Returns the status of the first template that failed, which is MALFORMED for
 * a Definition upstream would refuse and UNSUPPORTED for a body percounter does
 * not run. `failed_task_id` is then the GRAPH task that named it.
 */
inline TemplateResult gather_graph_inputs(
    const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks, std::vector<TaskInput> *inputs,
    GraphInputs *out, int64_t *failed_task_id
) {
    // Read on the first GRAPH task: a graph without one has no staging block.
    GraphImageView view{};
    std::map<uint32_t, uint32_t> template_of_definition;
    for (int64_t task_id = 0; task_id < total_tasks; ++task_id) {
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        if (slot.task_kind != TaskKind::GRAPH) continue;
        if (!view.valid()) {
            void *staging = nullptr;
            size_t staging_bytes = 0;
            api->get_graph_definition_staging(&staging, &staging_bytes);
            if (staging == nullptr || staging_bytes > UINT32_MAX) {
                *failed_task_id = task_id;
                return TemplateResult{TemplateStatus::MALFORMED, "the Definition staging block is unavailable", -1};
            }
            view = GraphImageView{static_cast<const std::byte *>(staging), static_cast<uint32_t>(staging_bytes)};
        }
        const uint32_t definition_offset = slot.to_descriptor().graph_definition_offset;
        auto found = template_of_definition.find(definition_offset);
        if (found == template_of_definition.end()) {
            HostGraphTemplate tmpl{};
            const TemplateResult built = read_graph_template(view, definition_offset, &tmpl);
            if (built.status != TemplateStatus::OK) {
                *failed_task_id = task_id;
                return built;
            }
            out->layout.templates_bytes += graph_template_bytes(tmpl);
            out->templates.push_back(std::move(tmpl));
            found = template_of_definition.emplace(definition_offset, out->templates.size() - 1).first;
        }
        const HostGraphTemplate &tmpl = out->templates[found->second];
        TaskInput &input = (*inputs)[static_cast<size_t>(task_id)];
        input.graph_invocation = static_cast<int32_t>(out->invocations.size());
        out->invocations.push_back(GraphInvocationInput{task_id, found->second});
        out->layout.body_task_count += tmpl.tasks.size();
        out->body_blocks += tmpl.executable_blocks;
    }
    out->layout.invocation_count = out->invocations.size();
    return TemplateResult{};
}

enum class StateSelection : uint8_t {
    DEFAULT,     // percounter was not requested, or does not run this graph
    PERCOUNTER,  // the percounter region is built and published; the bind is done
    FAILED,      // percounter was requested and accepted the graph, but building it failed
};

/**
 * Take the graph for percounter when it was requested and every task has a
 * shape percounter runs: one active subtask (AIC, or AIV0) or a MIX task of
 * two or three, any block count, with or without sync start.
 *
 * `task_metadata` and `inline_completed_task_ids` are the default selection's
 * per-task walk, which has already rejected any malformed task, checked every
 * kernel id against the callable table, and refused a dispatch predicate on any
 * task other than a single-subtask, single-block one.
 */
inline StateSelection select_percounter_state(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    const std::vector<SchedulerTaskMetadata> &task_metadata, const std::vector<int64_t> &inline_completed_task_ids,
    const sm_layout::SegmentOffsets &device_segments
) {
    if (!percounter_requested()) return StateSelection::DEFAULT;

    const size_t task_count = static_cast<size_t>(total_tasks);
    std::vector<bool> inline_completed(task_count, false);
    for (int64_t id : inline_completed_task_ids)
        inline_completed[static_cast<size_t>(id)] = true;

    std::vector<TaskInput> inputs(task_count);
    GraphInputs graph{};
    int64_t graph_failed_task = -1;
    const TemplateResult graph_status =
        gather_graph_inputs(api, host_sm_handle, total_tasks, &inputs, &graph, &graph_failed_task);
    if (graph_status.status == TemplateStatus::MALFORMED) {
        LOG_ERROR(
            "A5 HBG percounter: Graph task id=%" PRId64 ": %s (body task %d)", graph_failed_task, graph_status.reason,
            graph_status.body_index
        );
        return StateSelection::FAILED;
    }
    if (graph_status.status == TemplateStatus::UNSUPPORTED) {
        log_fallback(total_tasks, graph_status.reason, graph_failed_task);
        return StateSelection::DEFAULT;
    }
    std::vector<int32_t> edges;
    std::vector<size_t> edge_begin(task_count, 0);
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT] = {0, 0};
    uint64_t mix_order_count = 0;
    uint64_t sync_cohort_count = 0;
    // Each GRAPH task places its body's blocks, and holds one of its own that
    // no list carries.
    uint64_t block_total = graph.body_blocks;
    for (size_t t = 0; t < task_count; ++t) {
        const int64_t task_id = static_cast<int64_t>(t);
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        TaskInput &input = inputs[t];
        input.inline_completed = inline_completed[t];
        if (input.graph_invocation >= 0) {
            const HostGraphTemplate &tmpl =
                graph.templates[graph.invocations[static_cast<size_t>(input.graph_invocation)].template_index];
            order_count[0] += tmpl.orders[0].size();
            order_count[1] += tmpl.orders[1].size();
            mix_order_count += tmpl.orders[PERCOUNTER_LIST_MIX].size();
            sync_cohort_count += tmpl.sync_cohort_count;
            ++block_total;
        } else if (!input.inline_completed) {
            input.sync_start = slot.task_attrs.requires_sync_start();
            input.has_predicate = slot.task_attrs.has_predicate();
            const uint8_t active_mask = task_metadata[t].active_mask;
            const TaskPlacement placement = task_placement_from_active_mask(active_mask, &input.core_type_index);
            if (placement == TaskPlacement::UNSUPPORTED) {
                log_fallback(total_tasks, active_mask == 4U ? "an AIV1-only task" : "a task with no subtask", task_id);
                return StateSelection::DEFAULT;
            }
            input.block_num = slot.logical_block_num;
            input.is_mix = placement == TaskPlacement::MIX;
            input.subtask_count = __builtin_popcount(active_mask);
            // A sync-start task holds one arming entry; its blocks are handed
            // out from the cohort queue instead of from a list.
            const uint64_t entries = input.sync_start ? 1 : static_cast<uint64_t>(input.block_num);
            if (input.sync_start) ++sync_cohort_count;
            if (input.is_mix) mix_order_count += entries;
            else order_count[input.core_type_index] += entries;
            block_total += static_cast<uint64_t>(input.block_num);
        } else {
            ++block_total;
        }
        const TaskPayload &payload = slot.to_payload();
        input.fanin_count = payload.fanin_count;
        edge_begin[t] = edges.size();
        for (int32_t e = 0; e < payload.fanin_count; ++e)
            edges.push_back(payload.fanin_data()[e]);
    }
    for (size_t t = 0; t < task_count; ++t)
        inputs[t].fanin_ids = inputs[t].fanin_count > 0 ? edges.data() + edge_begin[t] : nullptr;

    PercounterLayout layout{};
    if (!plan_layout(
            task_count, block_total, edges.size(), order_count, &layout, mix_order_count, sync_cohort_count,
            graph.layout
        )) {
        LOG_ERROR("A5 HBG percounter: state layout overflow");
        return StateSelection::FAILED;
    }
    void *device_state = nullptr;
    void *host_base = nullptr;
    if (api->acquire_scheduler_state_storage(
            static_cast<size_t>(layout.total_size), static_cast<size_t>(SCHEDULER_STATE_ALIGNMENT), &device_state,
            &host_base
        ) != 0 ||
        device_state == nullptr || host_base == nullptr) {
        LOG_ERROR("A5 HBG percounter: failed to obtain %" PRIu64 " scheduler state bytes", layout.total_size);
        return StateSelection::FAILED;
    }
    const uint64_t device_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_state));

    ExternalAddresses external{};
    external.graph_storage = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr()) + device_segments.storage;
    external.graph_storage_count = task_count;
    external.callable_table = runtime->callable_entry_table_addr();
    external.callable_count = static_cast<uint64_t>(runtime->callable_table_len());
    BuildResult build{};
    const GraphPlanInput graph_plan{
        graph.templates.data(), graph.templates.size(), graph.invocations.data(), graph.invocations.size()
    };
    if (!build_tables(inputs.data(), task_count, layout, host_base, device_base, external, &build, graph_plan)) {
        LOG_ERROR(
            "A5 HBG percounter: %s (task id=%" PRId64 ", fanin index=%d)", build_status_name(build.status),
            build.task_id, build.fanin_index
        );
        return StateSelection::FAILED;
    }

    // Each lane's payload keeps these fields for the whole run: its argument
    // vector points at its own contexts. A lane rewrites the block fields for
    // every block it runs.
    for (uint64_t lane = 0; lane < SCHEDULER_WORKER_CAPACITY; ++lane) {
        const uint64_t offset = layout.dispatch_payloads_offset + lane * sizeof(DispatchPayload);
        auto *payload = region_at<DispatchPayload>(host_base, offset);
        const uint64_t device_payload = device_base + offset;
        payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, local_context);
        payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, global_context);
        payload->local_context.block_idx = 0;
        payload->local_context.block_num = 1;
        payload->local_context.async_ctx.task_token = TaskId::invalid();
        payload->src_payload = 0;
    }

    runtime->publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_PERCOUNTER, device_base + layout.control_offset);
    runtime->add_pending_metadata(
        device_state, host_base, static_cast<size_t>(layout.upload_size), HostPhaseKind::Count, {}
    );
    LOG_INFO(
        "A5 HBG: selected percounter for %d tasks (%" PRIu64 " AIC, %" PRIu64 " AIV and %" PRIu64
        " MIX list entries, %" PRIu64 " sync cohorts, %zu edges, %zu Graph invocations of %zu Definitions with %" PRIu64
        " body tasks; %" PRIu64 " of %" PRIu64 " region bytes uploaded)",
        total_tasks, order_count[0], order_count[1], mix_order_count, sync_cohort_count, edges.size(),
        graph.invocations.size(), graph.templates.size(), graph.layout.body_task_count, layout.upload_size,
        layout.total_size
    );
    return StateSelection::PERCOUNTER;
}

}  // namespace simpler::hbg::percounter
