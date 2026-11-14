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

bool export_profile(const ProfileOwner &owner) {
    const auto &layout = owner.layout;
    const auto headers =
        read_records<PercounterProfileBuffer>(owner, layout.profile_buffers_offset, SCHEDULER_WORKER_CAPACITY);
    std::ostringstream tasks, kernels, streams;
    kernels << "[";
    tasks << "{\"producer\":\"aicore\",\"records\":[";
    streams << "{\"streams\":[";
    bool first_task = true, first_stream = true;
    const auto epoch = owner.api->run_epoch();
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
        }
        if (layout.profiling_level < 3) continue;
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
                streams << ",\"scalar_schema\":2,\"batch_write_cycles\":" << header.scalar_write_cycles
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
    return publish(owner, ChipSwimlaneExtensionSection::AicoreTasks, kernels) &&
           publish(owner, ChipSwimlaneExtensionSection::SchedulerTasks, tasks) &&
           (layout.profiling_level < 3 || publish(owner, ChipSwimlaneExtensionSection::SchedulerRecords, streams));
}
}  // namespace

void retain_profile_state(Runtime *runtime, const HostApi *api, void *device, const PercounterLayout &layout) {
    if (layout.profiling_level < 2) return;
    std::scoped_lock lock(owners_mutex);
    owners.insert_or_assign(runtime, ProfileOwner{api, device, layout});
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
