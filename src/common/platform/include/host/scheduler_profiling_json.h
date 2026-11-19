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

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <sstream>
#include <string>

#include "common/scheduler_profiling.h"
#include "host/collected_record.h"

// A stream carries no runtime name of its own. The runtime is a property of the whole
// capture -- one run compiles against one runtime -- so it is stated once, in the
// document's metadata. Repeating it per stream would offer a reader a choice of sources
// where only one exists, and it would be absent below SCHED_PHASES anyway, where a
// capture has no streams at all.
inline void chip_swimlane_write_scheduler_ring_records(
    std::ostream &out, const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> &streams,
    const std::vector<uint32_t> &dropped_records, bool counts_unknown, bool include_aicore
) {
    out << "{\n    \"streams\": [";
    bool first_stream = true;
    for (size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        using Participant = std::tuple<SchedPhaseParticipantKind, uint32_t, uint32_t>;
        std::set<Participant> participants;
        std::map<Participant, std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> groups;
        for (const auto &collected : streams[stream_index]) {
            const auto &record = collected.record;
            participants.insert(
                {record.participant_kind, record.participant_worker_id, record.participant_physical_core_id}
            );
            if (!include_aicore && record.participant_kind != SchedPhaseParticipantKind::Aicpu) continue;
            groups[{record.participant_kind,
                    record.participant_kind == SchedPhaseParticipantKind::Aicpu ? static_cast<uint32_t>(stream_index) :
                                                                                  record.participant_worker_id,
                    record.participant_kind == SchedPhaseParticipantKind::Aicpu ? UINT32_MAX :
                                                                                  record.participant_physical_core_id}]
                .push_back(collected);
        }
        for (const auto &group : groups) {
            const auto &[kind, worker, physical] = group.first;
            const bool aicpu = kind == SchedPhaseParticipantKind::Aicpu;
            const auto &records = group.second;
            const uint32_t dropped = stream_index < dropped_records.size() ? dropped_records[stream_index] : 0;
            const bool unknown = counts_unknown || participants.size() > 1 || (!include_aicore && dropped != 0);
            if (!first_stream) out << ",";
            out << "\n      {\"platform\": \"" << CHIP_SWIMLANE_ARCHITECTURE_NAME << "\", \"producer\": \""
                << (aicpu ? "aicpu" : "aicore") << "\", \"scheduler_id\": " << worker << ", \"worker_id\": " << worker
                << ", \"core_type\": \"" << (aicpu ? "aicpu" : (kind == SchedPhaseParticipantKind::Aic ? "aic" : "aiv"))
                << "\", \"physical_core_id\": ";
            if (aicpu || physical == UINT32_MAX) out << "null";
            else out << physical;
            out << ", \"capture\": {\"committed\": " << records.size() << ", \"dropped\": ";
            if (unknown) out << "null, \"truncated\": null";
            else out << dropped << ", \"truncated\": " << (dropped == 0 ? "false" : "true");
            out << "}, \"records\": [";
            for (size_t record_index = 0; record_index < records.size(); ++record_index) {
                const ChipSwimlaneAicpuSchedPhaseRecord &record = records[record_index].record;
                if (record_index != 0) out << ",";
                out << "\n        {\"start_cycles\": " << record.start_time << ", \"end_cycles\": " << record.end_time
                    << ", \"run_epoch\": " << records[record_index].run_epoch << ", \"loop_iter\": " << record.loop_iter
                    << ", \"kind\": \"" << sched_phase_kind_name(record.kind)
                    << "\", \"tasks_processed\": " << record.tasks_processed << ", \"task_id\": ";
                if ((!aicpu && record.participant_has_task != 0) ||
                    (aicpu && sched_phase_carries_task_id(record.kind))) {
                    out << TaskId::to_uint64(record.phase_data.task_id);
                } else {
                    out << "null";
                }
                out << "}";
            }
            if (!records.empty()) out << "\n      ";
            out << "], \"metrics\": [";
            for (size_t record_index = 0; record_index < records.size(); ++record_index) {
                const ChipSwimlaneAicpuSchedPhaseRecord &record = records[record_index].record;
                if (record_index != 0) out << ",";
                out << "\n        {\"record_index\": " << record_index;
                if (aicpu && sched_phase_carries_pop_counters(record.kind)) {
                    out << ", \"pop_hit\": " << record.phase_data.dispatch.pop_hit
                        << ", \"pop_miss\": " << record.phase_data.dispatch.pop_miss;
                }
                out << ", \"shared_at_start\": [" << record.shared_depth_at_start[0] << ","
                    << record.shared_depth_at_start[1] << "," << record.shared_depth_at_start[2]
                    << "], \"shared_at_end\": [" << record.shared_depth_at_end[0] << ","
                    << record.shared_depth_at_end[1] << "," << record.shared_depth_at_end[2] << "]}";
            }
            if (!records.empty()) out << "\n      ";
            out << "]}";
            first_stream = false;
        }
    }
    if (!first_stream) out << "\n    ";
    out << "]\n  }";
}

/** The eager snapshot owns AICore phases; ordinary AICPU phases remain independent. */
inline void chip_swimlane_write_scheduler_records(
    std::ostream &out, const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> &streams,
    const std::vector<uint32_t> &dropped_records, bool counts_unknown = false, const std::string *snapshot = nullptr
) {
    if (snapshot == nullptr) {
        chip_swimlane_write_scheduler_ring_records(out, streams, dropped_records, counts_unknown, true);
        return;
    }
    std::ostringstream ring;
    chip_swimlane_write_scheduler_ring_records(ring, streams, dropped_records, counts_unknown, false);
    const std::string filtered = ring.str();
    const size_t ring_begin = filtered.find('[') + 1;
    const size_t ring_end = filtered.rfind(']');
    const auto ring_body = filtered.substr(ring_begin, ring_end - ring_begin);
    if (ring_body.find_first_not_of(" \t\r\n") == std::string::npos) {
        out << *snapshot;
        return;
    }
    // The runtime snapshot may contain arrays outside its top-level streams.
    size_t snapshot_begin = std::string::npos, snapshot_end = std::string::npos;
    size_t depth = 0, string_begin = 0;
    bool in_string = false;
    for (size_t i = 0; i < snapshot->size(); ++i) {
        const char c = (*snapshot)[i];
        if (in_string) {
            if (c == '\\') {
                ++i;
                continue;
            }
            if (c != '"') continue;
            in_string = false;
            if (depth != 1 || snapshot->compare(string_begin, i - string_begin + 1, "\"streams\"") != 0) continue;
            const auto colon = snapshot->find_first_not_of(" \t\r\n", i + 1);
            if (colon == std::string::npos || (*snapshot)[colon] != ':') continue;
            const auto array = snapshot->find_first_not_of(" \t\r\n", colon + 1);
            if (array != std::string::npos && (*snapshot)[array] == '[') snapshot_begin = array + 1;
        } else if (c == '"') {
            in_string = true;
            string_begin = i;
        } else if (c == '{' || c == '[') {
            ++depth;
        } else if (c == '}' || c == ']') {
            if (depth == 0) break;
            --depth;
            if (c == ']' && depth == 1 && snapshot_begin != std::string::npos) {
                snapshot_end = i;
                break;
            }
        }
    }
    if (snapshot_end == std::string::npos) {
        out.setstate(std::ios::failbit);
        return;
    }
    out << snapshot->substr(0, snapshot_end);
    if (snapshot->substr(snapshot_begin, snapshot_end - snapshot_begin).find_first_not_of(" \t\r\n") !=
        std::string::npos)
        out << ",";
    out << ring_body << snapshot->substr(snapshot_end);
}
