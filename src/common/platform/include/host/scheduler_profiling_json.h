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

#include "common/scheduler_profiling.h"
#include "host/collected_record.h"

inline const char *chip_swimlane_scheduler_kind_name(ChipSwimlaneSchedPhaseKind kind) {
    switch (kind) {
    case ChipSwimlaneSchedPhaseKind::Complete:
        return "complete";
    case ChipSwimlaneSchedPhaseKind::Dispatch:
        return "dispatch";
    case ChipSwimlaneSchedPhaseKind::Release:
        return "release";
    case ChipSwimlaneSchedPhaseKind::Dummy:
        return "dummy";
    case ChipSwimlaneSchedPhaseKind::EarlyDispatch:
        return "early_dispatch";
    case ChipSwimlaneSchedPhaseKind::Resolve:
        return "resolve";
    case ChipSwimlaneSchedPhaseKind::ResolveStandalone:
        return "resolve_standalone";
    case ChipSwimlaneSchedPhaseKind::DummyTask:
        return "dummy_task";
    case ChipSwimlaneSchedPhaseKind::PredicatedSkip:
        return "predicated_skip";
    case ChipSwimlaneSchedPhaseKind::Drain:
        return "drain";
    case ChipSwimlaneSchedPhaseKind::DrainPrepare:
        return "drain_prepare";
    case ChipSwimlaneSchedPhaseKind::DrainPublish:
        return "drain_publish";
    case ChipSwimlaneSchedPhaseKind::AsyncPoll:
        return "async_poll";
    case ChipSwimlaneSchedPhaseKind::GraphPrepare:
        return "graph_prepare";
    }
    return "unknown";
}

// A stream carries no runtime name of its own. The runtime is a property of the whole
// capture -- one run compiles against one runtime -- so it is stated once, in the
// document's metadata. Repeating it per stream would offer a reader a choice of sources
// where only one exists, and it would be absent below SCHED_PHASES anyway, where a
// capture has no streams at all.
inline void chip_swimlane_write_scheduler_records(
    std::ostream &out, const std::vector<std::vector<CollectedRecord<ChipSwimlaneAicpuSchedPhaseRecord>>> &streams,
    const std::vector<uint32_t> &dropped_records
) {
    out << "{\n    \"schema_version\": 1,\n    \"streams\": [";
    bool first_stream = true;
    for (size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        const auto &records = streams[stream_index];
        if (records.empty()) continue;
        const uint32_t dropped = stream_index < dropped_records.size() ? dropped_records[stream_index] : 0;
        if (!first_stream) out << ",";
        out << "\n      {\"platform\": \"" << CHIP_SWIMLANE_ARCHITECTURE_NAME
            << "\", \"producer\": \"aicpu\", \"scheduler_id\": " << stream_index << ", \"worker_id\": " << stream_index
            << ", \"core_type\": \"aicpu\", \"physical_core_id\": null, \"capture\": {\"committed\": " << records.size()
            << ", \"dropped\": " << dropped << ", \"truncated\": " << (dropped == 0 ? "false" : "true")
            << "}, \"records\": [";
        for (size_t record_index = 0; record_index < records.size(); ++record_index) {
            const ChipSwimlaneAicpuSchedPhaseRecord &record = records[record_index].record;
            if (record_index != 0) out << ",";
            out << "\n        {\"start_cycles\": " << record.start_time << ", \"end_cycles\": " << record.end_time
                << ", \"loop_iter\": " << record.loop_iter << ", \"kind\": \""
                << chip_swimlane_scheduler_kind_name(record.kind)
                << "\", \"tasks_processed\": " << record.tasks_processed << ", \"task_id\": ";
            if (record.kind == ChipSwimlaneSchedPhaseKind::DummyTask ||
                record.kind == ChipSwimlaneSchedPhaseKind::PredicatedSkip ||
                record.kind == ChipSwimlaneSchedPhaseKind::GraphPrepare) {
                out << record.phase_data.task_id.raw;
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
            if (record.kind == ChipSwimlaneSchedPhaseKind::Dispatch) {
                out << ", \"pop_hit\": " << record.phase_data.dispatch.pop_hit
                    << ", \"pop_miss\": " << record.phase_data.dispatch.pop_miss;
            }
            out << ", \"shared_at_start\": [" << record.shared_depth_at_start[0] << ","
                << record.shared_depth_at_start[1] << "," << record.shared_depth_at_start[2]
                << "], \"shared_at_end\": [" << record.shared_depth_at_end[0] << "," << record.shared_depth_at_end[1]
                << "," << record.shared_depth_at_end[2] << "]}";
        }
        if (!records.empty()) out << "\n      ";
        out << "]}";
        first_stream = false;
    }
    if (!first_stream) out << "\n    ";
    out << "]\n  }";
}
