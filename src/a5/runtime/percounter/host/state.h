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
#include <mutex>
#include <optional>
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
#include "calendar_cache.h"
#include "calendar_image.h"
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
    std::vector<CalendarKey> definitions;
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
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    uint64_t worker_count, std::vector<TaskInput> *inputs, GraphInputs *out, int64_t *failed_task_id
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
            for (int32_t i = 0; i < definition.definition.task_count; ++i) {
                SubTaskDefinition task{};
                *failed_task_id = task_id;
                if (!graph_definition_load_element(
                        view, definition, definition.definition.off_sub_tasks, definition.definition.task_count, i,
                        &task
                    ))
                    return TemplateResult{TemplateStatus::MALFORMED, "the body task does not decode", i};
                for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
                    if ((task.active_mask.raw() & (1U << lane)) != 0 &&
                        runtime->get_function_bin_addr(task.kernel_id[lane]) == 0)
                        return TemplateResult{TemplateStatus::MALFORMED, "a body kernel is unregistered", i};
                }
            }
            // The recorded definition contains relative tensor and scalar
            // records. Its bind-local graph_key is not a replay identity.
            const auto bytes = definition.definition.total_bytes;
            CalendarKey structure((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
            std::memcpy(structure.data(), view.section + definition_offset, bytes);
            structure.front() = 0;  // graph_key is the first uint64_t in Graph.
            out->definitions.push_back(std::move(structure));
            out->section = view.section;
            out->section_bytes = std::max(out->section_bytes, definition_offset + definition.definition.total_bytes);
            out->layout.templates_bytes += sizeof(PercounterGraphTemplate);
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
    if (total_tasks < 0) return StateSelection::FAILED;
    const size_t task_count = static_cast<size_t>(total_tasks);
    std::vector<bool> inline_completed(task_count, false);
    for (int64_t id : inline_completed_task_ids)
        inline_completed[static_cast<size_t>(id)] = true;

    std::vector<TaskInput> inputs(task_count);
    GraphInputs graph{};
    int64_t graph_failed_task = -1;
    const TemplateResult graph_status = gather_graph_inputs(
        runtime, api, host_sm_handle, total_tasks, runtime->get_worker_count(), &inputs, &graph, &graph_failed_task
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
    uint64_t mix_task_count = 0;
    // Each GRAPH task places its body's blocks, and holds one of its own that
    // no list carries.
    uint64_t block_total = graph.body_blocks;
    uint64_t defer_rows = 0;
    for (size_t t = 0; t < task_count; ++t) {
        const int64_t task_id = static_cast<int64_t>(t);
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        TaskInput &input = inputs[t];
        input.inline_completed = inline_completed[t];
        if (input.graph_invocation >= 0) {
            const HostGraphTemplate &tmpl =
                graph.templates[graph.invocations[static_cast<size_t>(input.graph_invocation)].template_index];
            mix_task_count += tmpl.mix_task_count;
            ++block_total;
            defer_rows += tmpl.defer_rows;
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
            input.active_mask = slot.active_mask.raw();
            input.block_num = slot.logical_block_num;
            input.is_mix = placement == TaskPlacement::MIX;
            input.subtask_count = __builtin_popcount(active_mask);
            if (input.active_mask != 0)
                defer_rows += uint64_t{static_cast<uint32_t>(input.block_num)} * input.subtask_count;
            mix_task_count += input.is_mix;
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
    if (!folded.empty()) LOG_TIMING("A5 percounter: folded %zu dependency-only tasks", folded.size());

    const int32_t worker_count = runtime->get_worker_count();
    const auto cluster_count = static_cast<uint32_t>(worker_count / PLATFORM_CORES_PER_BLOCKDIM);
    const GraphPlanInput graph_plan{
        graph.templates.data(), graph.templates.size(), graph.invocations.data(), graph.invocations.size()
    };
    std::vector<CalendarPhase> phases;
    std::vector<RangeCalendarInput> range_inputs;
    std::vector<std::shared_ptr<CalendarTraining>> training;
    std::vector<const char *> calendar_events;
    RangeCalendarImage calendar;
    if (worker_count < static_cast<int32_t>(PLATFORM_CORES_PER_BLOCKDIM) ||
        worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0 || !plan_calendar_phases(inputs, graph_plan, &phases)) {
        LOG_ERROR("A5 calendar: unsupported task graph or worker topology");
        return StateSelection::FAILED;
    }
    {
        auto cache = std::static_pointer_cast<CalendarCache>(
            api->acquire_callable_calendar_cache(runtime->get_active_callable_id(), make_calendar_cache)
        );
        range_inputs.resize(phases.size());
        training.resize(phases.size());
        calendar_events.resize(phases.size());
        std::optional<uint64_t> outer_signature_id;
        // The last equivalent Graph invocation supplies the timing sample.
        // Earlier invocations in the bind share its immutable seed plan.
        for (size_t p = phases.size(); p != 0; --p) {
            const auto n = p - 1;
            const auto &phase = phases[n];
            auto &entry = range_inputs[n];
            entry.phase = &phase;
            if (cache == nullptr) {
                auto initial = std::make_shared<CalendarPlan>();
                if (!build_static_calendar(phase.tasks, cluster_count, {}, initial.get())) break;
                entry.plan = std::move(initial);
                continue;
            }
            CalendarKey key;
            if (phase.invocation < 0) {
                if (!outer_signature_id) {
                    auto signature = make_calendar_key(*runtime, inputs, task_metadata);
                    const std::lock_guard<std::mutex> lock(cache->mutex);
                    auto &signatures = cache->outer_signatures;
                    outer_signature_id = signatures.try_emplace(std::move(signature), signatures.size()).first->second;
                }
                key = {*outer_signature_id, phase.task_ids.size()};
                key.insert(key.end(), phase.task_ids.begin(), phase.task_ids.end());
                for (size_t id = 0; id < phase.tasks.size(); ++id) {
                    key.push_back(phase.fanins[id].size());
                    key.insert(key.end(), phase.fanins[id].begin(), phase.fanins[id].end());
                }
            } else {
                const auto &invocation = graph.invocations[phase.invocation];
                const auto &definition = graph.definitions[invocation.template_index];
                const auto &payload =
                    host_sm_handle.header->tasks.get_slot_state_by_task_id(invocation.shell_task_id).to_payload();
                key = {
                    UINT64_MAX, static_cast<uint64_t>(worker_count), definition.size(),
                    static_cast<uint64_t>(payload.scalar_count), static_cast<uint64_t>(payload.tensor_count)
                };
                key.insert(key.end(), definition.begin(), definition.end());
                append_calendar_argument_key(
                    key, payload.scalar_count, payload.tensor_count,
                    [&](int32_t i) {
                        return payload.scalar_data()[i];
                    },
                    [&](int32_t i) -> const auto & {
                        return payload.tensor_data()[i];
                    }
                );
            }
            std::unique_lock<std::mutex> lock(cache->mutex);
            if (const auto found = cache->plans.find(key); found != cache->plans.end()) {
                entry.plan = found->second;
                calendar_events[n] = "HIT";
                continue;
            }
            if (const auto pending = cache->training.find(key); pending != cache->training.end()) {
                if ((entry.plan = pending->second.lock())) {
                    calendar_events[n] = "REUSE";
                    continue;
                }
            }
            auto initial = std::make_shared<CalendarPlan>();
            if (!build_static_calendar(phase.tasks, cluster_count, {}, initial.get())) break;
            const auto sample = std::make_shared<CalendarTraining>();
            sample->cache = cache;
            sample->baseline = initial;
            sample->cluster_count = cluster_count;
            sample->allow_fused_fanins = range_inputs.size() == 1 && graph_plan.invocation_count == 0;
            sample->key = std::move(key);
            sample->copy_tasks(phase.tasks);
            cache->training.insert_or_assign(sample->key, initial);
            sample->owns_training = true;
            entry.plan = std::move(initial);
            entry.sample = true;
            training[n] = sample;
            calendar_events[n] = "MISS";
        }
        if (!build_range_calendar(
                range_inputs, graph_plan, static_cast<uint32_t>(task_count), cluster_count, &calendar
            )) {
            LOG_ERROR("A5 calendar: could not plan execution ranges or completion counters");
            return StateSelection::FAILED;
        }
    }

    PercounterLayout layout{};
    if (!plan_layout(
            task_count, block_total, &layout, mix_task_count, graph.layout, defer_rows,
            runtime->dev.chip_swimlane_level, calendar.uses_cluster_input_share
        )) {
        LOG_ERROR("A5 percounter: state layout overflow");
        return StateSelection::FAILED;
    }
    constexpr uint64_t alignment = 128;
    if (layout.total_size > SIZE_MAX - alignment - calendar.bytes.size()) return StateSelection::FAILED;
    const uint64_t calendar_offset = (layout.total_size + alignment - 1) & ~(alignment - 1);
    const uint64_t allocation_size = calendar_offset + calendar.bytes.size();
    const auto calendar_header = calendar.load<CalendarRangeHeader>(0);
    void *device_state = nullptr;
    void *host_base = nullptr;
    if (api->acquire_scheduler_state_storage(
            static_cast<size_t>(allocation_size), alignof(PercounterControl), &device_state, &host_base
        ) != 0 ||
        device_state == nullptr || host_base == nullptr) {
        LOG_ERROR("A5 percounter: failed to obtain %" PRIu64 " scheduler state bytes", layout.total_size);
        return StateSelection::FAILED;
    }
    const uint64_t device_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_state));

    SchedulerDoneNotifications notifications{};
    // The last SET slot is a run-wide lifecycle word. Keep it out of the
    // calendar slot table: lanes sample the previous generation before discovery.
    const size_t requested = static_cast<size_t>(calendar_header.notify_slots_count) + 1;
    const int lease_rc =
        api->acquire_scheduler_notifications(SchedulerNotificationPurpose::CALENDAR_COUNTER, requested, &notifications);
    if (lease_rc != 0 || notifications.addresses.size() != requested || notifications.owner == nullptr ||
        a5_cnt_notify_increment_address(notifications.addresses.back()) == 0) {
        LOG_ERROR(
            "A5 calendar: complete CntNotify lease required (requested=%zu, received=%zu, status=%d)", requested,
            notifications.addresses.size(), lease_rc
        );
        return StateSelection::FAILED;
    }
    const uint64_t lifecycle_notify = notifications.addresses.back();
    notifications.addresses.pop_back();

    ExternalAddresses external{};
    external.graph_storage = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr()) + device_segments.storage;
    external.graph_storage_count = task_count;
    external.callable_table = runtime->callable_entry_table_addr();
    external.callable_count = static_cast<uint64_t>(runtime->callable_table_len());
    if (!build_tables(inputs, layout, host_base, external, graph_plan)) {
        LOG_ERROR("A5 calendar: failed to build scheduler metadata");
        return StateSelection::FAILED;
    }

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
    for (size_t id = 0; id < task_count; ++id) {
        const auto &input = inputs[id];
        if (input.graph_invocation >= 0 || input.inline_completed || input.active_mask == 0) continue;
        const auto &payload = task_image::payload(uploaded_sm, device_segments, id);
        if (payload.tensor_count < 0 || payload.scalar_count < 0) return StateSelection::FAILED;
        const uint32_t tensors = static_cast<uint32_t>(payload.tensor_count);
        const uint32_t scalars = static_cast<uint32_t>(payload.scalar_count);
        uint64_t arguments_address = 0;
        const void *arguments =
            reinterpret_cast<const uint8_t *>(uploaded_sm) + device_segments.scalar_pool + id * sizeof(DispatchPayload);
        if (!image_address(arguments, sizeof(DispatchPayload), arguments_address)) return StateSelection::FAILED;
        if (input.has_predicate &&
            !image_address(&payload.predicate, sizeof(payload.predicate), counters[id].predicate_address))
            return StateSelection::FAILED;
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
    {
        auto *control = region_at<PercounterControl>(host_base, layout.control_offset);
        control->calendar_offset = calendar_offset - layout.control_offset;
        control->image_bytes = allocation_size;
        if (!bind_range_calendar(calendar, device_base, layout, host_base, notifications.addresses)) {
            LOG_ERROR("A5 calendar: failed to bind leased CntNotify counters");
            return StateSelection::FAILED;
        }
        if (notifications.owner != nullptr) runtime->retain_pending_resource(std::move(notifications.owner));
        auto *host_calendar = region_at<std::byte>(host_base, calendar_offset);
        std::memcpy(host_calendar, calendar.bytes.data(), calendar.bytes.size());
        runtime->add_pending_metadata(
            static_cast<std::byte *>(device_state) + calendar_offset, host_calendar, calendar.bytes.size(),
            HostPhaseKind::Count, {}
        );
        for (size_t p = 0; p < range_inputs.size(); ++p) {
            if (calendar_events[p] == nullptr) continue;
            const auto &plan = *range_inputs[p].plan;
            LOG_INFO(
                "A5 calendar: %s plan=%016" PRIx64 " rows=%zu pid=%d", calendar_events[p], calendar_plan_id(plan),
                plan.logical_row_by_packed.size(), getpid()
            );
        }
        std::array<size_t, CALENDAR_FUSED_FANIN_MAX_PRODUCERS + 1> fusion_counts{};
        for (const auto &group : calendar.fused_fanins)
            ++fusion_counts[group.producer_count];
        LOG_INFO(
            "A5 calendar: ranges=%u phases=%zu graph_invocations=%zu counters=%u phase_slots=%u fanin_slots=%u "
            "fused_fanins=%zu fused_2=%zu fused_3=%zu fused_4=%zu",
            calendar_header.range_count, phases.size(), graph.invocations.size(), calendar_header.notify_slots_count,
            calendar.phase_slot_count, calendar.fanin_slot_count, calendar.fused_fanins.size(), fusion_counts[2],
            fusion_counts[3], fusion_counts[4]
        );
        if (calendar_header.sample_index_offset != 0) {
            runtime->on_pending_execution_complete(
                [training = std::move(training), offsets = std::move(calendar.sample_offsets),
                 timings_base = device_base + calendar_offset](const HostApi &run_api) {
                    for (size_t p = 0; p < training.size(); ++p) {
                        if (training[p] == nullptr) continue;
                        try {
                            training[p]->finish(run_api, reinterpret_cast<const void *>(timings_base + offsets[p]));
                        } catch (const std::exception &error) {
                            LOG_WARN("A5 calendar: could not retain schedule: %s", error.what());
                        }
                    }
                }
            );
        }
    }
    retain_profile_state(runtime, api, device_state, layout, folded, inline_completed_task_ids);
    runtime->publish_scheduler_bootstrap(
        SCHEDULER_RUNTIME_MODE_PERCOUNTER, device_base + layout.control_offset, 0, lifecycle_notify,
        device_base + layout.control_offset + offsetof(PercounterControl, rejected_exit)
    );
    if (api != nullptr) api->record_scheduler_runtime_mode(SCHEDULER_RUNTIME_MODE_PERCOUNTER);
    runtime->add_pending_metadata(
        device_state, host_base, static_cast<size_t>(layout.upload_size), HostPhaseKind::Count, {}
    );
    return StateSelection::PERCOUNTER;
}

}  // namespace simpler::percounter
