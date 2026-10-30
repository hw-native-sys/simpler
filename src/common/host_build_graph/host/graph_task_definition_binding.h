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

#include <unordered_map>
#include <utility>
#include <vector>

#include "common/unified_log.h"
#include "host_build_graph/host/definition_packing.h"
#include "host_build_graph/host/ready_queue_sizing.h"
#include "host_build_graph/shared_memory.h"

namespace hbg {

struct PreparedGraphDefinition {
    size_t object_offset;
    size_t image_bytes;
    const std::byte *copy;
    const GraphDefinition *definition;
    ReadyQueuePopulations ready_queue_populations;
    bool populations_ready;
};

// Borrows a completed build's records and slots. Neither planning nor queue
// measurement acquires device storage or publishes a task's Definition offset.
struct GraphTaskDefinitionBindings {
    std::unordered_map<uint64_t, PreparedGraphDefinition> objects;
    std::vector<std::pair<ChipTaskSlotState *, uint32_t>> bindings;
    ReadyQueuePopulations populations{};
    size_t bytes{0};
    size_t spilled{0};
};

inline bool resolve_graph_task_definitions(
    GraphHostState &graph_state, const ReadyQueuePopulations &initial_populations, GraphTaskDefinitionBindings &out
) {
    if (graph_state.inflight_count.load(std::memory_order_acquire) != 0 || !graph_state.inflight.empty()) return false;
    GraphHostDefinitionList definitions = graph_host_definitions(graph_state);
    if (definitions.entries.size() != graph_state.definitions.size()) return false;
    DefinitionPackingPlan packing;
    if (!measure_definition_packing(definitions, graph_host_arena_used(graph_state), packing)) return false;
    GraphTaskDefinitionBindings next;
    next.bytes = packing.bytes;
    next.spilled = packing.spilled;
    for (const auto &object : packing.objects) {
        const auto record = graph_state.definitions.find(object.graph_key);
        if (record == graph_state.definitions.end()) return false;
        const GraphDefinition *definition = graph_record_definition(graph_state, record->second);
        if (definition == nullptr || definition->graph_key != object.graph_key) return false;
        next.objects.emplace(
            object.graph_key,
            PreparedGraphDefinition{object.object_offset, object.image_bytes, object.copy, definition, {}, false}
        );
    }
    ReadyQueuePopulations populations = initial_populations;
    const size_t count = graph_host_upload_count(graph_state);
    for (size_t index = 0; index < count; ++index) {
        std::optional<GraphHostUpload> upload = graph_host_upload(graph_state, index);
        if (!upload.has_value() || upload->outer_slot == nullptr || upload->outer_slot->task_kind != TaskKind::GRAPH) {
            LOG_ERROR("host-orch: invalid pending Graph task");
            return false;
        }
        const GraphHostUpload pending = *upload;
        auto object_it = next.objects.find(pending.graph_key);
        if (object_it == next.objects.end()) {
            LOG_ERROR("host-orch: Graph task has no matching prepared Definition object");
            return false;
        }
        // Validate the prepared bytes that publication will copy to the device.
        const GraphDefinition *definition = object_it->second.definition;
        if (definition->total_bytes != object_it->second.image_bytes) {
            LOG_ERROR("host-orch: Graph task has no matching prepared Definition object");
            return false;
        }
        GraphExecutionStorageLayout storage_layout{};
        if (definition->task_count <= 0 || definition->task_count > SUB_TASK_MAX_NUM ||
            definition->graph_key != pending.graph_key ||
            !graph_execution_storage_layout(
                definition->task_count, definition->tensor_arg_count, definition->scalar_arg_count,
                definition->edge_count, &storage_layout
            ) ||
            storage_layout.total_bytes != definition->execution_storage_bytes ||
            pending.outer_slot->to_payload().tensor_count != definition->boundary_tensor_count ||
            pending.outer_slot->to_payload().scalar_count != definition->boundary_scalar_count) {
            LOG_ERROR("host-orch: invalid Graph Definition for task");
            return false;
        }
        const uintptr_t outer_base =
            reinterpret_cast<uintptr_t>(pending.outer_slot->to_descriptor().packed_buffer_base);
        const uintptr_t outer_end = reinterpret_cast<uintptr_t>(pending.outer_slot->to_descriptor().packed_buffer_end);
        if (outer_end < outer_base || definition->required_heap > UINTPTR_MAX - outer_base ||
            storage_layout.total_bytes > outer_end - outer_base ||
            definition->required_heap > outer_end - outer_base - storage_layout.total_bytes) {
            LOG_ERROR("host-orch: Graph runtime storage does not fit its outer task heap");
            return false;
        }
        const uintptr_t storage_addr = outer_base + definition->required_heap;
        if (storage_addr % alignof(ChipTaskStorage) != 0) {
            LOG_ERROR("host-orch: Graph runtime storage address is misaligned");
            return false;
        }
        PreparedGraphDefinition &packed_definition = object_it->second;
        if (!packed_definition.populations_ready) {
            const SubTaskDefinition *tasks = graph_definition_array<SubTaskDefinition>(
                *definition, definition->off_sub_tasks, definition->task_count
            );
            if (tasks == nullptr) {
                LOG_ERROR("host-orch: invalid Graph Definition sub-task array");
                return false;
            }
            for (int32_t i = 0; i < definition->task_count; ++i) {
                // Sizing takes the kind materialize will give this task. add_task
                // singles out GRAPH and routes everything else by shape, and a Graph
                // body member is never the shell, so the shape decides. Derived here
                // the same way the device derives it, so the two cannot drift.
                const LaneMask mask(tasks[i].active_mask);
                packed_definition.ready_queue_populations.add_task(
                    mask, TaskAttrs(tasks[i].task_attrs), mask.is_empty() ? TaskKind::DUMMY : TaskKind::KERNEL
                );
            }
            packed_definition.populations_ready = true;
        }
        populations.add(packed_definition.ready_queue_populations);
        next.bindings.emplace_back(
            pending.outer_slot, static_cast<uint32_t>(object_it->second.object_offset + sizeof(GraphDefinitionHeader))
        );
    }
    next.populations = populations;
    out = std::move(next);
    return true;
}

}  // namespace hbg
