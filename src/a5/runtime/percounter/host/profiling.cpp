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
    case PercounterTaskEventKind::Claim:
        return "claim";
    case PercounterTaskEventKind::Ready:
        return "ready";
    case PercounterTaskEventKind::CompletionIssue:
        return "completion_issue";
    case PercounterTaskEventKind::DeferRegistered:
        return "defer_registered";
    case PercounterTaskEventKind::DeferredCompletionIssue:
        return "deferred_completion_issue";
    case PercounterTaskEventKind::MixPublished:
        return "mix_publish";
    case PercounterTaskEventKind::InputsPending:
        return "inputs_pending";
    case PercounterTaskEventKind::InputsReady:
        return "inputs_ready";
    case PercounterTaskEventKind::EntryCommitted:
        return "entry_commit";
    case PercounterTaskEventKind::KernelEnd:
        return "kernel_end";
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
    bool first_event = true;
    uint64_t dropped_events = 0;
    std::unordered_map<uint64_t, uint64_t> deferred_entries;
    std::vector<std::pair<size_t, PercounterTaskEvent>> deferred_updates;
    kernels << "[";
    tasks << "{\"producer\":\"aicore\",\"records\":[";
    streams << "{\"streams\":[";
    bool first_task = true, first_stream = true;
    const auto epoch = owner.api->run_epoch();
    const auto emit_task_event = [&](size_t worker, const PercounterTaskEvent &event,
                                     const PercounterTaskTiming *timing = nullptr) {
        const auto &header = headers[worker];
        const uint64_t id = percounter_order_task_id(event.entry);
        const auto block = percounter_order_block_idx(event.entry);
        const auto blocks = percounter_order_block_num(event.entry);
        if (event.cycles == 0 || id >= layout.task_count || block >= blocks)
            throw std::runtime_error("invalid percounter task event");
        TaskId token = TaskId::make_global(static_cast<int32_t>(id));
        if (id >= layout.table_task_count) {
            auto invocation =
                std::upper_bound(invocations.begin(), invocations.end(), id, [](uint64_t task, const auto &record) {
                    return task < record.body_base;
                });
            if (invocation == invocations.begin()) throw std::runtime_error("percounter task event has no Graph");
            --invocation;
            token = TaskId::make_sub_task(invocation->shell_task_id, id - invocation->body_base);
        }
        if (!first_event) task_events << ',';
        first_event = false;
        task_events << "\n{\"kind\":\"" << task_event_name(event.kind) << "\",\"cycles\":" << event.cycles
                    << ",\"run_epoch\":" << epoch << ",\"task_id\":" << TaskId::to_uint64(token)
                    << ",\"block_index\":" << block << ",\"block_count\":" << blocks << ",\"worker_id\":" << worker
                    << ",\"cluster_index\":" << header.cluster_index << ",\"cluster_lane\":" << header.cluster_lane;
        if (event.kind == PercounterTaskEventKind::Claim) task_events << ",\"claim_start_cycles\":" << event.detail;
        if (layout.profiling_level == 2 && event.kind == PercounterTaskEventKind::Claim)
            task_events << ",\"slot\":" << event.slot
                        << ",\"participant_mask\":" << static_cast<unsigned>(event.participant_mask)
                        << ",\"mix\":" << (event.mix ? "true" : "false");
        if (layout.profiling_level == 2 && timing != nullptr)
            task_events << ",\"dispatch_cycles\":" << timing->dispatch << ",\"finish_cycles\":" << timing->finish;
        if (event.kind == PercounterTaskEventKind::KernelEnd)
            task_events << ",\"kernel_start_cycles\":" << event.detail;
        if (event.kind == PercounterTaskEventKind::CompletionIssue) task_events << ",\"contribution\":" << event.detail;
        if (event.kind == PercounterTaskEventKind::DeferRegistered ||
            event.kind == PercounterTaskEventKind::DeferredCompletionIssue)
            task_events << ",\"deferred_group\":" << event.detail;
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
                    << task.start << ',' << task.finish << ',' << task.start - task.dispatch << ',' << epoch << ']';
            tasks << '[' << worker << ',' << registration << ',' << task.dispatch << ',' << task.finish << ',' << epoch
                  << ']';
            if (task.kernel_end < task.start || task.kernel_end > task.finish)
                throw std::runtime_error("invalid percounter kernel return timestamp");
            emit_task_event(
                worker, {task.kernel_end, task.entry, task.start, PercounterTaskEventKind::KernelEnd, 0, 0, 0}, &task
            );
        }
        const auto event_offset =
            layout.event_records_offset + worker * layout.event_record_capacity * sizeof(PercounterTaskEvent);
        if (header.event_records_offset != event_offset || header.event_capacity != layout.event_record_capacity ||
            header.event_committed > header.event_capacity)
            throw std::runtime_error("invalid percounter task event header");
        dropped_events += header.event_dropped;
        for (const auto &event : read_records<PercounterTaskEvent>(owner, event_offset, header.event_committed)) {
            if (event.kind == PercounterTaskEventKind::DeferredCompletionIssue) {
                deferred_updates.emplace_back(worker, event);
                continue;
            }
            if (event.kind == PercounterTaskEventKind::DeferRegistered &&
                !deferred_entries.emplace(event.detail, event.entry).second)
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
                (record.kind & 0xffffU) > static_cast<uint32_t>(ProfilePhase::DsbDdr) || record.start_time == 0 ||
                record.end_time < record.start_time)
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
                    << "\",\"tasks_processed\":0,\"task_id\":null}";
        }
        streams << "],\"metrics\":[";
        for (size_t index = 0; index < records.size(); ++index) {
            const auto &record = records[index];
            if (index) streams << ',';
            streams << "{\"record_index\":" << index << ",\"scalar_calls\":" << record.count
                    << ",\"scalar_depth\":" << (record.kind >> 16);
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
        event.entry = match->second;
        emit_task_event(worker, event);
    }
    if (unmatched_updates != 0 && dropped_events == 0)
        throw std::runtime_error("percounter deferred completion has no producer task");
    task_events << "],\"dropped\":" << dropped_events << ",\"unmatched_deferred_updates\":" << unmatched_updates;
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
    return publish(owner, ChipSwimlaneExtensionSection::AicoreTasks, kernels) &&
           publish(owner, ChipSwimlaneExtensionSection::SchedulerTasks, tasks) &&
           publish(owner, ChipSwimlaneExtensionSection::PercounterTaskEvents, task_events) &&
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
