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

#include "host_build_graph/graph_host_state.h"

#include <stddef.h>

#include <atomic>
#include <memory>
#include <new>
#include <optional>
#include <vector>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/graph_recording.h"

// The GraphHostState container itself: its lifetime, and the reads the host performs on a
// finished one -- what to upload, how much arena to ship, which Definitions it holds.
//
// Nothing here records or submits. The Graph submit path in host/graph_submit.cpp is a
// caller of these, which is what keeps the container readable without it.

GraphHostStatePtr make_graph_host_state(const GraphDefinitionArena &arena) {
    return GraphHostStatePtr{new (std::nothrow) GraphHostState{arena}};
}

void GraphHostStateDeleter::operator()(GraphHostState *state) const noexcept { delete state; }

const GraphDefinition *graph_record_definition(const GraphHostState &state, const GraphDefinitionRecord &record) {
    if (record.object_offset == GRAPH_NO_OBJECT_OFFSET) {
        return graph_definition(record.spill.data(), record.spill.size());
    }
    if (state.arena.base == nullptr) return nullptr;
    return graph_definition(state.image_at(record.object_offset), record.bytes);
}

size_t graph_host_upload_count(const GraphHostState &state) { return state.pending_uploads.size(); }

std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index) {
    if (index >= state.pending_uploads.size()) return std::nullopt;
    GraphPendingUpload &upload = state.pending_uploads[index];
    if (upload.outer_slot == nullptr) return std::nullopt;
    return GraphHostUpload{upload.outer_slot, upload.full_key};
}

size_t graph_host_arena_used(const GraphHostState &state) { return state.arena_cursor.load(std::memory_order_acquire); }

GraphHostDefinitionList graph_host_definitions(GraphHostState &state) {
    GraphHostDefinitionList list;
    list.entries.reserve(state.definitions.size());
    for (const auto &[key, record] : state.definitions) {
        if (graph_record_definition(state, record) == nullptr) continue;
        const bool spilled = record.object_offset == GRAPH_NO_OBJECT_OFFSET;
        list.entries.push_back(
            GraphHostDefinition{
                key, record.object_offset, spilled ? record.spill.data() : nullptr,
                spilled ? record.spill.size() : record.bytes
            }
        );
    }
    return list;
}
