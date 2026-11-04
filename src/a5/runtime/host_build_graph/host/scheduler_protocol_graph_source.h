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

#include <cstring>
#include <map>

#include "scheduler_protocol_export.h"

struct SchedulerProtocolSourceTask {
    // Original submission kind: kernel, Graph, deferred dummy, Host completed.
    uint32_t kind;
    uint32_t definition_offset;
};

template <class T>
inline T scheduler_protocol_source_value(const std::vector<std::byte> &image, uint64_t offset) {
    if (offset > image.size() || sizeof(T) > image.size() - offset)
        throw std::invalid_argument("protocol source section outside original image");
    T value;
    std::memcpy(&value, image.data() + offset, sizeof(T));
    return value;
}

// Only original outer payloads and recorded Definition bytes enter this capture.
// Expanded bindings, child plans, sinks and completion maps are not oracle inputs.
inline std::string scheduler_protocol_freeze_source(
    const SchedulerGraphView &original, const std::vector<SchedulerProtocolSourceTask> &tasks,
    const std::vector<std::byte> &definitions
) {
    using namespace scheduler_protocol_export;
    if (original.native_header_offset != 0 || original.task_count != tasks.size())
        throw std::invalid_argument("protocol source must be the original graph");
    std::map<uint64_t, std::pair<uint64_t, GraphDefinition>> unique;
    std::ostringstream out;
    {
        JsonObject root(out);
        root.field("tasks") << '[';
        for (size_t task = 0; task < tasks.size(); ++task) {
            if (task != 0) out << ',';
            JsonObject row(out);
            row.number("task_id", task);
            row.number("kind", tasks[task].kind);
            uint64_t key = 0;
            if (tasks[task].kind == 1) {
                const auto offset = tasks[task].definition_offset;
                const auto def = scheduler_protocol_source_value<GraphDefinition>(definitions, offset);
                if (def.task_count <= 0 || def.task_count > SUB_TASK_MAX_NUM || offset > definitions.size() ||
                    def.total_bytes > definitions.size() - offset)
                    throw std::invalid_argument("invalid original Definition extent");
                key = def.graph_key;
                unique.emplace(key, std::make_pair(offset, def));
            }
            row.number("graph_key", key);
            uint32_t count = 0;
            if (scheduler_node_fanin_count(original, task, &count) != SchedulerGraphResult::OK)
                throw std::invalid_argument("invalid original fanin row");
            std::vector<int64_t> deps(count);
            for (uint32_t edge = 0; edge < count; ++edge)
                if (scheduler_node_fanin_ref(original, task, edge, &deps[edge]) != SchedulerGraphResult::OK)
                    throw std::invalid_argument("invalid original dependency");
            row.numbers("dependencies", deps.data(), deps.size());
        }
        out << ']';
        root.field("definitions") << '[';
        bool first = true;
        for (const auto &[key, entry] : unique) {
            if (!first) out << ',';
            first = false;
            const auto &[offset, def] = entry;
            JsonObject row(out);
            row.number("graph_key", key);
            row.field("children") << '[';
            for (int32_t child = 0; child < def.task_count; ++child) {
                if (child != 0) out << ',';
                const auto sub = scheduler_protocol_source_value<SubTaskDefinition>(
                    definitions, offset + def.off_sub_tasks + child * sizeof(SubTaskDefinition)
                );
                JsonObject item(out);
                item.lane_mask("active_mask", sub.active_mask);
                item.number("block_num", sub.logical_block_num);
                item.number("task_attrs", sub.task_attrs);
                item.number("predicate_slot", sub.predicate_slot);
                item.numbers("kernel_ids", sub.kernel_id, 3);
                const auto begin = scheduler_protocol_source_value<int32_t>(
                    definitions, offset + def.off_fanin_offsets + child * sizeof(int32_t)
                );
                const auto end = scheduler_protocol_source_value<int32_t>(
                    definitions, offset + def.off_fanin_offsets + (child + 1) * sizeof(int32_t)
                );
                if (begin < 0 || end < begin || end > def.edge_count)
                    throw std::invalid_argument("invalid original Definition fanin row");
                std::vector<uint16_t> deps;
                for (int32_t edge = begin; edge < end; ++edge)
                    deps.push_back(
                        scheduler_protocol_source_value<uint16_t>(
                            definitions, offset + def.off_fanin_indices + edge * sizeof(uint16_t)
                        )
                    );
                item.numbers("dependencies", deps.data(), deps.size());
            }
            out << ']';
        }
        out << ']';
    }
    return out.str();
}
