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
 * This runtime's scheduler state, built from the default selection's per-task
 * walk.
 *
 * select_percounter_state runs after that walk, reuses its validation and
 * metadata, and builds the percounter region. GRAPH tasks pass the walk
 * untouched and are validated here through their Definitions.
 *
 * A graph percounter cannot run fails the bind. This runtime's AICore object
 * carries no legacy or resident executor, so there is no scheduler to fall back
 * to: a lane would run the percounter loop against a region that was never
 * published.
 *
 * Header-only because the UT targets that compile runtime_maker.cpp list their
 * sources by name.
 */

#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "common/host_api.h"
#include "common/host_phase_kind.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime.h"
#include "host_build_graph/shared_memory.h"
#include "intrinsic.h"
#include "host/platform_compile_info.h"
#include "callable_entry.h"
#include "image/image_pack.h"
#include "scheduler/argument_metadata.h"
#include "graph_template.h"
#include "region_plan.h"
#include "scheduler/scheduler_layout.h"
#include "profiling.h"
#include "scheduler/scheduler_types.h"

namespace simpler::percounter {

inline void log_unsupported(int32_t total_tasks, const char *reason, int64_t task_id) {
    LOG_ERROR(
        "A5 percounter: graph of %d tasks: task id=%" PRId64 " is %s, which percounter does not run", total_tasks,
        task_id, reason
    );
}

/** The bind's Graph templates and invocations, gathered from the task table. */
struct GraphInputs {
    std::vector<HostGraphTemplate> templates;
    std::vector<GraphInvocationInput> invocations;
    GraphLayoutInput layout{};
    uint64_t body_blocks{0};
    const std::byte *section{nullptr};
    uint32_t section_bytes{0};
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
    const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks, uint64_t worker_count,
    std::vector<TaskInput> *inputs, GraphInputs *out, int64_t *failed_task_id
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
            const TemplateResult built = read_graph_template(view, definition_offset, worker_count, &tmpl);
            if (built.status != TemplateStatus::OK) {
                *failed_task_id = task_id;
                return built;
            }
            GraphImageValue definition{};
            if (!graph_definition_decode_framed(view, definition_offset, &definition)) {
                *failed_task_id = task_id;
                return TemplateResult{TemplateStatus::MALFORMED, "the Definition framing is invalid", -1};
            }
            out->section = view.section;
            out->section_bytes = std::max(out->section_bytes, definition_offset + definition.definition.total_bytes);
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
    PERCOUNTER,  // the percounter region is built and published; the bind is done
    FAILED,      // this graph does not run on percounter, or building its region failed
};

/**
 * Build this run's percounter region. Every task must have a shape percounter
 * runs: one active subtask (AIC, or AIV0) or a MIX task of two or three, any
 * block count, with or without sync start.
 *
 * `task_metadata` and `inline_completed_task_ids` are the default selection's
 * per-task walk, which has already rejected any malformed task, checked every
 * kernel id against the callable table, and refused a dispatch predicate on any
 * task other than a single-subtask, single-block one.
 */
inline StateSelection select_percounter_state(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    const std::vector<SchedulerTaskMetadata> &task_metadata, const std::vector<int64_t> &inline_completed_task_ids,
    const sm_layout::SegmentOffsets &device_segments, const void *uploaded_sm
) {
    const size_t task_count = static_cast<size_t>(total_tasks);
    std::vector<bool> inline_completed(task_count, false);
    for (int64_t id : inline_completed_task_ids)
        inline_completed[static_cast<size_t>(id)] = true;

    std::vector<TaskInput> inputs(task_count);
    GraphInputs graph{};
    int64_t graph_failed_task = -1;
    const TemplateResult graph_status = gather_graph_inputs(
        api, host_sm_handle, total_tasks, runtime->get_worker_count(), &inputs, &graph, &graph_failed_task
    );
    if (graph_status.status == TemplateStatus::MALFORMED) {
        LOG_ERROR(
            "A5 percounter: Graph task id=%" PRId64 ": %s (body task %d)", graph_failed_task, graph_status.reason,
            graph_status.body_index
        );
        return StateSelection::FAILED;
    }
    if (graph_status.status == TemplateStatus::UNSUPPORTED) {
        // Named with its body index, as the MALFORMED case above is: the
        // reason belongs to one task of the body, and the GRAPH task id alone
        // does not say which.
        LOG_ERROR(
            "A5 percounter: graph of %d tasks: Graph task id=%" PRId64 " has %s (body task %d), which percounter does "
            "not run",
            total_tasks, graph_failed_task, graph_status.reason, graph_status.body_index
        );
        return StateSelection::FAILED;
    }
    std::vector<int32_t> edges;
    std::vector<size_t> edge_begin(task_count, 0);
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT] = {0, 0};
    uint64_t mix_task_count = 0;
    uint64_t boundary_task_count = 0;
    uint64_t sync_cohort_count = 0;
    uint64_t completion_notify_count = 0;
    // Each GRAPH task places its body's blocks, and holds one of its own that
    // no list carries.
    uint64_t block_total = graph.body_blocks;
    uint64_t execution_count = 0;
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
            mix_task_count += tmpl.mix_task_count;
            boundary_task_count += tmpl.boundary_task_count;
            sync_cohort_count += tmpl.sync_cohort_count;
            ++block_total;
            for (const auto &body_task : tmpl.tasks)
                execution_count += body_task.subtask_total;
        } else if (!input.inline_completed) {
            input.sync_start = slot.task_attrs.requires_sync_start();
            input.has_predicate = slot.task_attrs.has_predicate();
            const uint8_t active_mask = task_metadata[t].active_mask.raw();
            const TaskPlacement placement = task_placement_from_active_mask(active_mask, &input.core_type_index);
            if (placement == TaskPlacement::UNSUPPORTED) {
                log_unsupported(
                    total_tasks, active_mask == 4U ? "an AIV1-only task" : "a task with no subtask", task_id
                );
                return StateSelection::FAILED;
            }
            input.active_mask = 0;
            const auto &descriptor = slot.to_descriptor();
            for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
                if (descriptor.kernel_id[lane] >= 0) input.active_mask |= 1U << lane;
            input.block_num = slot.logical_block_num;
            input.is_mix = placement == TaskPlacement::MIX;
            input.subtask_count = __builtin_popcount(active_mask);
            execution_count += static_cast<uint64_t>(input.block_num) * input.subtask_count;
            // SPMD tasks allocate blocks through a task-private counter.
            const uint64_t entries = 1;
            if (input.block_num > 1 || input.is_mix) ++boundary_task_count;
            if (input.sync_start) ++sync_cohort_count;
            if (percounter_completion_notify_eligible(input)) ++completion_notify_count;
            if (input.is_mix) mix_task_count += entries;
            order_count[input.is_mix ? 1 : input.core_type_index] += entries;
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

    std::vector<std::vector<int32_t>> folded_rows;
    std::vector<int64_t> folded;
    if (!percounter_fold_dependency_tasks(inputs, folded_rows, folded)) {
        LOG_ERROR("A5 percounter: invalid dependency-only task fanin");
        return StateSelection::FAILED;
    }
    uint64_t edge_count = edges.size();
    if (!folded.empty()) {
        edge_count = 0;
        for (const auto &input : inputs)
            edge_count += static_cast<uint64_t>(input.fanin_count);
        for (const int64_t id : folded)
            --order_count[inputs[id].core_type_index];
        execution_count -= folded.size();
        LOG_TIMING("A5 percounter: folded %zu dependency-only tasks", folded.size());
    }

    PercounterLayout layout{};
    if (!plan_layout(
            task_count, block_total, edge_count, order_count, &layout, mix_task_count, boundary_task_count,
            sync_cohort_count, graph.layout, execution_count, runtime->dev.chip_swimlane_level
        )) {
        LOG_ERROR("A5 percounter: state layout overflow");
        return StateSelection::FAILED;
    }
    void *device_state = nullptr;
    void *host_base = nullptr;
    if (api->acquire_scheduler_state_storage(
            static_cast<size_t>(layout.total_size), alignof(PercounterControl), &device_state, &host_base
        ) != 0 ||
        device_state == nullptr || host_base == nullptr) {
        LOG_ERROR("A5 percounter: failed to obtain %" PRIu64 " scheduler state bytes", layout.total_size);
        return StateSelection::FAILED;
    }
    const uint64_t device_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_state));

    SchedulerCompletionCounterPlan completion_plan;
    if (completion_notify_count != 0) {
        std::string error;
        if (!plan_percounter_completion_counters(inputs, &completion_plan, &error)) {
            LOG_ERROR("A5 percounter: completion counter plan failed: %s", error.c_str());
            return StateSelection::FAILED;
        }
    }

    // Keep sync cohorts first in the lease so allocation pressure does not
    // displace their existing mappings. The AICPU clears both groups before READY.
    std::vector<uint64_t> notify_addresses;
    const uint64_t notify_count = sync_cohort_count + completion_plan.chains.size();
    if (notify_count != 0) {
        SchedulerDoneNotifications notifications{};
        if (api->acquire_scheduler_notifications(
                SchedulerNotificationPurpose::PERCOUNTER_SYNC_COHORT, static_cast<size_t>(notify_count), &notifications
            ) == 0) {
            notify_addresses = std::move(notifications.addresses);
            if (notifications.owner != nullptr) runtime->retain_pending_resource(std::move(notifications.owner));
        }
        const auto mapped = [&](size_t index) {
            return index < notify_addresses.size() && a5_cnt_notify_increment_address(notify_addresses[index]) != 0;
        };
        uint64_t mapped_sync = 0;
        for (size_t index = 0; index < sync_cohort_count; ++index)
            mapped_sync += mapped(index);
        size_t mapped_chains = 0;
        for (size_t chain = 0; chain < completion_plan.chains.size(); ++chain)
            mapped_chains += mapped(sync_cohort_count + chain);
        const auto mapped_tasks =
            std::count_if(completion_plan.task_chains.begin(), completion_plan.task_chains.end(), [&](uint32_t chain) {
                return chain != UINT32_MAX && mapped(sync_cohort_count + chain);
            });
        LOG_TIMING(
            "A5 percounter: completion counters planned_chains=%zu mapped_chains=%zu eligible_tasks=%" PRIu64
            " mapped_tasks=%zu mapped_sync=%" PRIu64 " sync_cohorts=%" PRIu64,
            completion_plan.chains.size(), mapped_chains, completion_notify_count, static_cast<size_t>(mapped_tasks),
            mapped_sync, sync_cohort_count
        );
    }

    ExternalAddresses external{};
    external.graph_storage = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr()) + device_segments.storage;
    external.graph_storage_count = task_count;
    external.callable_table = runtime->callable_entry_table_addr();
    external.callable_count = static_cast<uint64_t>(runtime->callable_table_len());
    external.sync_notify = notify_addresses.data();
    external.sync_notify_count = std::min<uint64_t>(notify_addresses.size(), sync_cohort_count);
    if (notify_addresses.size() > sync_cohort_count) {
        external.completion_plan = &completion_plan;
        external.completion_notify = notify_addresses.data() + sync_cohort_count;
        external.completion_notify_count = notify_addresses.size() - sync_cohort_count;
    }
    BuildResult build{};
    const GraphPlanInput graph_plan{
        graph.templates.data(), graph.templates.size(), graph.invocations.data(), graph.invocations.size()
    };
    const auto may_defer = [&](uint32_t id) {
        const auto &input = inputs[id];
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if ((input.active_mask & (1U << lane)) == 0) continue;
            try {
                const auto capability = runtime->get_function_async_capability(task_metadata[id].kernel_ids[lane]);
                if (capability.mode != CoreAsyncMode::SYNCHRONOUS) return true;
            } catch (const std::exception &) {
                return true;
            }
        }
        return false;
    };
    if (!build_tables(
            inputs.data(), task_count, layout, host_base, device_base,
            static_cast<uint32_t>(runtime->get_worker_count() / PLATFORM_CORES_PER_BLOCKDIM), external, &build,
            graph_plan, true, may_defer
        )) {
        LOG_ERROR(
            "A5 percounter: %s (task id=%" PRId64 ", fanin index=%d)", build_status_name(build.status), build.task_id,
            build.fanin_index
        );
        return StateSelection::FAILED;
    }

    const auto *terminal_control = region_at<PercounterControl>(host_base, layout.control_offset);
    if (terminal_control->terminal.step_count != 0)
        LOG_TIMING(
            "A5 percounter: terminal pipeline %u steps, first=%u, last=%u, prerequisites=%u",
            terminal_control->terminal.step_count, terminal_control->terminal.steps[0],
            terminal_control->terminal.steps[terminal_control->terminal.step_count - 1],
            terminal_control->terminal.prerequisite_count
        );
    if (task_count != 0 && uploaded_sm == nullptr) return StateSelection::FAILED;
    const auto address_space = std::strcmp(get_platform(), "a5sim") == 0 ? SchedulerCallableAddressSpace::HOST :
                                                                           SchedulerCallableAddressSpace::DEVICE;
    const uint64_t host_image = reinterpret_cast<uint64_t>(uploaded_sm);
    const uint64_t device_image = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr());
    const auto image_address = [&](const void *address, uint64_t bytes, uint64_t &device_address) {
        device_address = 0;
        if (bytes == 0) return true;
        const uint64_t source = reinterpret_cast<uint64_t>(address);
        if (source < host_image || source - host_image > runtime->dev.sm_image_bytes ||
            bytes > runtime->dev.sm_image_bytes - (source - host_image) ||
            source - host_image > UINT64_MAX - device_image)
            return false;
        device_address = device_image + (source - host_image);
        return true;
    };
    auto *counters = region_at<PercounterCounter>(host_base, layout.counters_offset);
    region_at<PercounterControl>(host_base, layout.control_offset)->arguments_address =
        device_image + device_segments.scalar_pool;
    for (size_t id = 0; id < task_count; ++id) {
        const auto &input = inputs[id];
        if (input.graph_invocation >= 0 || input.inline_completed || input.published_off_lane || input.active_mask == 0)
            continue;
        const auto &payload = task_image::payload(uploaded_sm, device_segments, id);
        if (payload.tensor_count < 0 || payload.scalar_count < 0) return StateSelection::FAILED;
        const uint32_t tensors = static_cast<uint32_t>(payload.tensor_count);
        const uint32_t scalars = static_cast<uint32_t>(payload.scalar_count);
        uint64_t arguments_address = 0;
        const void *arguments =
            reinterpret_cast<const uint8_t *>(uploaded_sm) + device_segments.scalar_pool + id * sizeof(DispatchPayload);
        if (!image_address(arguments, sizeof(DispatchPayload), arguments_address)) return StateSelection::FAILED;
        uint64_t functions[PLATFORM_CORES_PER_BLOCKDIM]{};
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            if (input.active_mask & (1U << lane))
                functions[lane] = scheduler_resolve_callable_entry(
                    runtime->get_function_bin_addr(task_metadata[id].kernel_ids[lane]), address_space
                );
        const auto prepared = percounter_prepare_argument_metadata(
            counters[id], functions, input.active_mask, arguments_address, tensors, scalars
        );
        if (prepared != SchedulerGraphResult::OK) {
            LOG_ERROR(
                "A5 percounter: invalid argument metadata task=%zu status=%u", id, static_cast<unsigned>(prepared)
            );
            return StateSelection::FAILED;
        }
    }

    if (graph.section_bytes != 0) {
        uint32_t source = 0;
        uint64_t base = 0;
        if (api->publish_graph_section(graph.section, graph.section_bytes, &source, &base) != 0) {
            LOG_ERROR("A5 percounter: failed to publish %u Graph Definition bytes", graph.section_bytes);
            return StateSelection::FAILED;
        }
        runtime->publish_graph_section(static_cast<GraphSectionSource>(source), base, graph.section_bytes);
    }
    retain_profile_state(runtime, api, device_state, layout, folded, inline_completed_task_ids);
    runtime->publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_PERCOUNTER, device_base + layout.control_offset);
    if (api != nullptr) api->record_scheduler_runtime_mode(SCHEDULER_RUNTIME_MODE_PERCOUNTER);
    runtime->add_pending_metadata(
        device_state, host_base, static_cast<size_t>(layout.upload_size), HostPhaseKind::Count, {}
    );
    LOG_INFO(
        "A5 percounter: selected for %d tasks (%" PRIu64 " AIC, %" PRIu64 " AIV and %" PRIu64
        " MIX list entries, %" PRIu64 " sync cohorts, %zu edges, %zu Graph invocations of %zu Definitions with %" PRIu64
        " body tasks; %" PRIu64 " of %" PRIu64 " region bytes uploaded)",
        total_tasks, terminal_control->order_count[0], terminal_control->order_count[1],
        terminal_control->mix_task_count, sync_cohort_count, edges.size(), graph.invocations.size(),
        graph.templates.size(), graph.layout.body_task_count, layout.upload_size, layout.total_size
    );
    return StateSelection::PERCOUNTER;
}

}  // namespace simpler::percounter
