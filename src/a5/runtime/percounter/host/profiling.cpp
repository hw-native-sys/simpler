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

#include "profiling.h"
#include "scheduler_extension.h"
#include "common/host_api.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace simpler::percounter {
namespace {
struct ProfileOwner {
    const HostApi *api;
    void *device;
    PercounterLayout layout;
    std::vector<int64_t> folded_task_indices;
    std::vector<int64_t> inline_completed_task_indices;
};
std::mutex owners_mutex;
std::unordered_map<Runtime *, ProfileOwner> owners;

template <typename T>
std::vector<T> read_records(const ProfileOwner &owner, uint64_t offset, uint64_t count) {
    if (offset > owner.layout.total_size || count > (owner.layout.total_size - offset) / sizeof(T))
        throw std::runtime_error("percounter profiling range exceeds storage");
    std::vector<T> records(count);
    if (count != 0 && owner.api->copy_from_device(
                          records.data(), static_cast<const uint8_t *>(owner.device) + offset, count * sizeof(T)
                      ) != 0)
        throw std::runtime_error("percounter profiling download failed");
    return records;
}

bool publish(const ProfileOwner &owner, ChipSwimlaneExtensionSection section, const std::ostringstream &json) {
    const auto payload = json.str();
    return owner.api->publish_chip_swimlane_extension(section, payload.data(), payload.size());
}

const char *task_event_name(PercounterTaskEventKind kind) {
    switch (kind) {
    case PercounterTaskEventKind::RowBegin:
        return "row_begin";
    case PercounterTaskEventKind::InputsReady:
        return "inputs_ready";
    case PercounterTaskEventKind::MixReleaseObserved:
        return "mix_release_observed";
    case PercounterTaskEventKind::Ready:
        return "ready";
    case PercounterTaskEventKind::CompletionIssue:
        return "completion_issue";
    case PercounterTaskEventKind::DeferRegistered:
        return "defer_registered";
    case PercounterTaskEventKind::DeferredCompletionIssue:
        return "deferred_completion_issue";
    case PercounterTaskEventKind::LocalRetired:
        return "local_retired";
    case PercounterTaskEventKind::DeferPollBegin:
        return "defer_poll_begin";
    case PercounterTaskEventKind::DeferSignalReady:
        return "defer_signal_ready";
    case PercounterTaskEventKind::KernelEnd:
        return "kernel_end";
    case PercounterTaskEventKind::RangeEnterBegin:
        return "range_enter_begin";
    case PercounterTaskEventKind::GraphReadyObserved:
        return "graph_ready_observed";
    case PercounterTaskEventKind::RangeEntered:
        return "range_entered";
    }
    throw std::runtime_error("invalid percounter task event kind");
}

bool export_profile(const ProfileOwner &owner) {
    const auto &layout = owner.layout;
    const auto headers =
        read_records<PercounterProfileBuffer>(owner, layout.profile_buffers_offset, SCHEDULER_WORKER_CAPACITY);
    const bool capture_scalar = layout.profiling_level >= 3;
    auto invocations =
        read_records<PercounterGraphInvocation>(owner, layout.graph_invocations_offset, layout.graph_invocation_count);
    std::sort(invocations.begin(), invocations.end(), [](const auto &a, const auto &b) {
        return a.body_base < b.body_base;
    });
    std::ostringstream tasks, kernels, streams, task_events;
    task_events << "{\"records\":[";
    std::ostringstream range_events;
    bool first_event = true, first_range = true;
    uint64_t dropped_events = 0;
    std::unordered_map<uint64_t, std::pair<uint64_t, size_t>> deferred_entries;
    std::vector<std::pair<size_t, PercounterTaskEvent>> deferred_updates;
    kernels << "[";
    tasks << "{\"producer\":\"aicore\",\"records\":[";
    streams << "{\"streams\":[";
    bool first_task = true, first_stream = true;
    const auto epoch = owner.api->run_epoch();
    const auto task_token = [&](uint64_t id) {
        if (id >= layout.task_count) throw std::runtime_error("invalid percounter task id");
        if (id < layout.table_task_count) return TaskId::make_global(static_cast<int32_t>(id));
        auto invocation =
            std::upper_bound(invocations.begin(), invocations.end(), id, [](uint64_t task, const auto &record) {
                return task < record.body_base;
            });
        if (invocation == invocations.begin()) throw std::runtime_error("percounter task has no Graph");
        --invocation;
        return TaskId::make_sub_task(invocation->shell_task_id, id - invocation->body_base);
    };
    const auto emit_task_event = [&](size_t worker, const PercounterTaskEvent &event,
                                     const PercounterTaskTiming *timing = nullptr, size_t source_worker = SIZE_MAX) {
        const auto &header = headers[worker];
        const uint64_t id = percounter_order_task_id(event.entry);
        const auto block = percounter_order_block_idx(event.entry);
        const auto blocks = percounter_order_block_num(event.entry);
        if (event.cycles == 0 || id >= layout.task_count || block >= blocks)
            throw std::runtime_error("invalid percounter task event");
        const auto token = task_token(id);
        if (!first_event) task_events << ',';
        first_event = false;
        task_events << "\n{\"kind\":\"" << task_event_name(event.kind) << "\",\"cycles\":" << event.cycles
                    << ",\"run_epoch\":" << epoch << ",\"task_id\":" << TaskId::to_uint64(token)
                    << ",\"block_index\":" << block << ",\"block_count\":" << blocks << ",\"worker_id\":" << worker
                    << ",\"cluster_index\":" << header.cluster_index << ",\"cluster_lane\":" << header.cluster_lane;
        if (timing != nullptr) {
            task_events << ",\"dispatch_cycles\":" << timing->dispatch << ",\"kernel_start_cycles\":" << timing->start
                        << ",\"finish_cycles\":" << timing->finish;
        }
        if (event.kind == PercounterTaskEventKind::CompletionIssue) task_events << ",\"contribution\":" << event.detail;
        if (event.kind == PercounterTaskEventKind::DeferRegistered || source_worker != SIZE_MAX)
            task_events << ",\"deferred_group\":" << event.detail;
        if (source_worker != SIZE_MAX) task_events << ",\"source_worker_id\":" << source_worker;
        if (event.kind == PercounterTaskEventKind::DeferredCompletionIssue) task_events << ",\"contribution\":1";
        task_events << '}';
    };
    for (size_t worker = 0; worker < headers.size(); ++worker) {
        const auto &header = headers[worker];
        if (header.physical_core_id < 0) continue;
        const auto task_offset = layout.task_records_offset + worker * layout.task_record_stride;
        const auto scalar_offset =
            layout.scalar_records_offset + worker * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord);
        if (header.worker_id != worker || header.task_records_offset != task_offset ||
            header.task_committed > layout.task_record_capacity || header.task_dropped != 0)
            throw std::runtime_error("invalid or truncated percounter task timing");
        uint32_t registration = 0;
        for (const auto &task : read_records<PercounterTaskTiming>(owner, task_offset, header.task_committed)) {
            ++registration;
            if (task.dispatch == 0 || task.start < task.dispatch || task.finish < task.start)
                throw std::runtime_error("invalid percounter task interval");
            if (!first_task) {
                tasks << ',';
                kernels << ',';
            }
            first_task = false;
            kernels << '[' << worker << ',' << TaskId::to_uint64(task.task_token) << ',' << registration << ','
                    << task.start << ',' << task.kernel_end << ",0," << epoch << ','
                    << percounter_order_block_idx(task.entry) << ',' << percounter_order_block_num(task.entry) << ']';
            tasks << '[' << worker << ',' << registration << ',' << task.dispatch << ',' << task.finish << ',' << epoch
                  << ']';
            if (task.kernel_end < task.start || task.kernel_end > task.finish)
                throw std::runtime_error("invalid percounter kernel return timestamp");
            emit_task_event(
                worker, {task.kernel_end, task.entry, task.start, PercounterTaskEventKind::KernelEnd, 0}, &task
            );
        }
        const auto event_offset =
            layout.event_records_offset + worker * layout.event_record_capacity * sizeof(PercounterTaskEvent);
        if (header.event_records_offset != event_offset || header.event_capacity != layout.event_record_capacity ||
            header.event_committed > header.event_capacity)
            throw std::runtime_error("invalid percounter task event header");
        dropped_events += header.event_dropped;
        for (const auto &event : read_records<PercounterTaskEvent>(owner, event_offset, header.event_committed)) {
            if (event.kind == PercounterTaskEventKind::RangeEnterBegin ||
                event.kind == PercounterTaskEventKind::GraphReadyObserved ||
                event.kind == PercounterTaskEventKind::RangeEntered) {
                if (!first_range) range_events << ',';
                first_range = false;
                range_events << "{\"kind\":\"" << task_event_name(event.kind) << "\",\"cycles\":" << event.cycles
                             << ",\"run_epoch\":" << epoch << ",\"worker_id\":" << worker
                             << ",\"cluster_index\":" << header.cluster_index
                             << ",\"cluster_lane\":" << header.cluster_lane
                             << ",\"range_index\":" << static_cast<uint32_t>(event.entry);
                const auto shell = event.entry >> 32;
                if (shell != UINT32_MAX) range_events << ",\"graph_shell_id\":" << shell;
                if (event.kind == PercounterTaskEventKind::GraphReadyObserved)
                    range_events << ",\"watermark\":" << event.detail;
                range_events << '}';
                continue;
            }
            if (event.kind == PercounterTaskEventKind::DeferredCompletionIssue ||
                event.kind == PercounterTaskEventKind::DeferPollBegin ||
                event.kind == PercounterTaskEventKind::DeferSignalReady) {
                deferred_updates.emplace_back(worker, event);
                continue;
            }
            if (event.kind == PercounterTaskEventKind::DeferRegistered &&
                !deferred_entries.emplace(event.detail, std::pair{event.entry, worker}).second)
                throw std::runtime_error("duplicate percounter deferred group");
            emit_task_event(worker, event);
        }
        if (!capture_scalar) continue;
        if (header.scalar_records_offset != scalar_offset ||
            header.scalar_storage_words > PERCOUNTER_SCALAR_WORD_CAPACITY)
            throw std::runtime_error("invalid percounter scalar header");
        const auto slots = read_records<uint64_t>(owner, scalar_offset, header.scalar_storage_words);
        std::vector<SchedulerScalarRecord> records;
        records.reserve(header.scalar_storage_words);
        uint32_t cursor = 0;
        while (cursor < header.scalar_storage_words) {
            SchedulerScalarRecord record{};
            if (!percounter_decode_scalar_storage(
                    slots.data(), slots.size(), cursor, header.scalar_window_start, record
                ) ||
                (record.kind & 0xffffU) > static_cast<uint32_t>(ProfilePhase::CntNotifyDeferIssue) ||
                record.start_time == 0 || record.end_time < record.start_time)
                throw std::runtime_error("invalid percounter scalar interval");
            records.push_back(record);
        }
        if (cursor != slots.size()) throw std::runtime_error("trailing percounter scalar slots");
        std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
            if (a.start_time != b.start_time) return a.start_time < b.start_time;
            if (a.end_time != b.end_time) return a.end_time > b.end_time;
            return a.kind < b.kind;
        });
        if (!first_stream) streams << ',';
        first_stream = false;
        streams << "{\"platform\":\"a5\",\"producer\":\"aicore\",\"scheduler_id\":" << worker
                << ",\"worker_id\":" << worker << ",\"core_type\":\"" << (header.core_type == 0 ? "aic" : "aiv")
                << "\",\"physical_core_id\":" << header.physical_core_id
                << ",\"cluster_index\":" << header.cluster_index << ",\"cluster_lane\":" << header.cluster_lane
                << ",\"capture\":{\"committed\":" << records.size() << ",\"dropped\":" << header.scalar_dropped
                << ",\"truncated\":" << (header.scalar_dropped ? "true" : "false") << "},\"records\":[";
        for (size_t index = 0; index < records.size(); ++index) {
            const auto &record = records[index];
            if (index) streams << ',';
            streams << "\n{\"start_cycles\":" << record.start_time << ",\"end_cycles\":" << record.end_time
                    << ",\"run_epoch\":" << epoch << ",\"loop_iter\":0,\"kind\":\""
                    << profile_phase_name(static_cast<ProfilePhase>(record.kind & 0xffffU))
                    << "\",\"tasks_processed\":0,\"task_id\":";
            const auto phase = static_cast<ProfilePhase>(record.kind & 0xffffU);
            if (record.task_ref == -1 || percounter_profile_is_notify(phase)) streams << "null";
            else
                streams << TaskId::to_uint64(
                    task_token(percounter_order_task_id(static_cast<uint64_t>(record.task_ref)))
                );
            streams << '}';
        }
        streams << "],\"metrics\":[";
        for (size_t index = 0; index < records.size(); ++index) {
            const auto &record = records[index];
            if (index) streams << ',';
            const auto phase = static_cast<ProfilePhase>(record.kind & 0xffffU);
            const bool notify = percounter_profile_is_notify(phase);
            streams << "{\"record_index\":" << index << ",\"scalar_calls\":" << (notify ? 1 : record.count)
                    << ",\"scalar_depth\":" << (record.kind >> 16);
            if (notify) {
                streams << ",\"memory_space\":\"CntNotify SRAM\",\"address\":" << static_cast<uint64_t>(record.task_ref)
                        << (phase == ProfilePhase::CntNotifyRead ? ",\"observed_value\":" : ",\"operand\":")
                        << record.count;
            } else if (record.task_ref != -1) {
                const auto entry = static_cast<uint64_t>(record.task_ref);
                streams << ",\"block_index\":" << percounter_order_block_idx(entry)
                        << ",\"block_count\":" << percounter_order_block_num(entry);
            }
            if ((record.kind & 0xffffU) == static_cast<uint32_t>(ProfilePhase::Window))
                streams << ",\"batch_write_cycles\":" << header.scalar_write_cycles
                        << ",\"no_progress_loops\":" << header.scalar_poll_loops
                        << ",\"short_function_intervals_omitted\":" << header.scalar_short_intervals
                        << ",\"function_min_cycles\":" << header.scalar_function_min_cycles
                        << ",\"scalar_storage_word_bytes\":8,\"scalar_storage_bytes\":"
                        << slots.size() * sizeof(uint64_t) << ",\"scalar_logical_records\":" << records.size()
                        << ",\"iterations_detailed\":true";
            streams << '}';
        }
        streams << "]}";
    }
    tasks << "]}";
    kernels << "]";
    streams << "]}";
    uint64_t unmatched_updates = 0;
    for (auto &[worker, event] : deferred_updates) {
        const auto match = deferred_entries.find(event.detail);
        if (match == deferred_entries.end()) {
            ++unmatched_updates;
            continue;
        }
        event.entry = match->second.first;
        emit_task_event(worker, event, nullptr, match->second.second);
    }
    if (unmatched_updates != 0 && dropped_events == 0)
        throw std::runtime_error("percounter deferred completion has no producer task");
    task_events << "],\"range_records\":[" << range_events.str() << "],\"dropped\":" << dropped_events
                << ",\"unmatched_deferred_updates\":" << unmatched_updates;
    const auto emit_task_ids = [&](const char *name, const std::vector<int64_t> &indices) {
        task_events << ",\"" << name << "\":[";
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i != 0) task_events << ',';
            task_events << TaskId::to_uint64(TaskId::make_global(static_cast<int32_t>(indices[i])));
        }
        task_events << ']';
    };
    emit_task_ids("folded_task_ids", owner.folded_task_indices);
    emit_task_ids("inline_completed_task_ids", owner.inline_completed_task_indices);
    task_events << '}';
    const auto aicpu =
        read_records<PercounterAicpuTiming>(owner, layout.aicpu_profile_offset, PLATFORM_MAX_AICPU_THREADS);
    std::ostringstream lifecycle;
    lifecycle << '[';
    bool first_lifecycle = true;
    const auto begin_lifecycle = [&](size_t thread) {
        if (!first_lifecycle) lifecycle << ',';
        first_lifecycle = false;
        lifecycle << "{\"aicpu_thread_id\":" << thread;
    };
    for (size_t thread = 0; thread < aicpu.size(); ++thread) {
        const auto &timing = aicpu[thread];
        if (timing.ready_publish_start_cycles == 0 && timing.observe_start_cycles == 0 &&
            timing.shutdown_start_cycles == 0)
            continue;
        begin_lifecycle(thread);
        lifecycle << ",\"ready_publish_start_cycles\":" << timing.ready_publish_start_cycles
                  << ",\"ready_publish_end_cycles\":" << timing.ready_publish_end_cycles
                  << ",\"observe_start_cycles\":" << timing.observe_start_cycles
                  << ",\"sinks_observed_cycles\":" << timing.sinks_observed_cycles
                  << ",\"observe_end_cycles\":" << timing.observe_end_cycles
                  << ",\"shutdown_start_cycles\":" << timing.shutdown_start_cycles
                  << ",\"shutdown_end_cycles\":" << timing.shutdown_end_cycles << '}';
    }
    for (const auto &graph : invocations) {
        if (graph.materialize_start_cycles == 0 || graph.materialize_end_cycles == 0) continue;
        begin_lifecycle(graph.materializer_thread_id);
        lifecycle << ",\"graph_shell_id\":" << graph.shell_task_id
                  << ",\"materialize_start_cycles\":" << graph.materialize_start_cycles
                  << ",\"materialize_end_cycles\":" << graph.materialize_end_cycles << '}';
        const uint64_t count = (graph.calendar_ready + PERCOUNTER_GRAPH_SLICE_TASKS - 1) / PERCOUNTER_GRAPH_SLICE_TASKS;
        for (const auto &watermark :
             read_records<PercounterGraphReadyTiming>(owner, graph.ready_timing_offset, count)) {
            if (watermark.cycles == 0) throw std::runtime_error("missing percounter Graph ready timestamp");
            begin_lifecycle(graph.materializer_thread_id);
            lifecycle << ",\"graph_shell_id\":" << graph.shell_task_id
                      << ",\"graph_watermark_published_cycles\":" << watermark.cycles
                      << ",\"watermark\":" << watermark.watermark << '}';
        }
    }
    lifecycle << ']';
    return publish(owner, ChipSwimlaneExtensionSection::AicoreTasks, kernels) &&
           publish(owner, ChipSwimlaneExtensionSection::SchedulerTasks, tasks) &&
           publish(owner, ChipSwimlaneExtensionSection::PercounterTaskEvents, task_events) &&
           publish(owner, ChipSwimlaneExtensionSection::AicpuLifecycleRecords, lifecycle) &&
           (!capture_scalar || publish(owner, ChipSwimlaneExtensionSection::SchedulerRecords, streams));
}
}  // namespace

void retain_profile_state(
    Runtime *runtime, const HostApi *api, void *device, const PercounterLayout &layout,
    const std::vector<int64_t> &folded_task_indices, const std::vector<int64_t> &inline_completed_task_indices
) {
    if (layout.profiling_level < 2) return;
    std::scoped_lock lock(owners_mutex);
    owners.insert_or_assign(
        runtime, ProfileOwner{api, device, layout, folded_task_indices, inline_completed_task_indices}
    );
}
}  // namespace simpler::percounter

bool scheduler_extension_publish_profiling(Runtime *runtime) {
    using namespace simpler::percounter;
    ProfileOwner owner{};
    {
        std::scoped_lock lock(owners_mutex);
        const auto it = owners.find(runtime);
        if (it == owners.end()) return true;
        owner = it->second;
    }
    return export_profile(owner);
}

void scheduler_extension_release(Runtime *runtime) {
    using namespace simpler::percounter;
    std::scoped_lock lock(owners_mutex);
    owners.erase(runtime);
}
