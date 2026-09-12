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
#include "host_build_graph/graph_definition_pack.h"

#include <cstring>

#include "common/host_api.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host_graph_build.h"
#include "host_log.h"

namespace simpler::hbg {

GraphBuild::GraphBuild() = default;
GraphBuild::~GraphBuild() = default;

bool pack_graph_definitions(
    GraphHostState &graph_state, const GraphDefinitionArena &arena, GraphDefinitionPlan &plan,
    ReadyQueuePopulations &populations
) {
    const size_t count = graph_host_upload_count(graph_state);
    GraphHostDefinitionList definitions = graph_host_definitions(graph_state);
    const auto align_up = [](size_t value) {
        return (value + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
    };
    // Objects the recorders built already occupy the arena's used prefix at the
    // offsets they claimed, so the block starts out that long and the rest are
    // appended past them.
    size_t block_bytes = graph_host_arena_used(graph_state);
    if (block_bytes > arena.capacity || block_bytes % GRAPH_DEFINITION_OBJECT_ALIGN != 0 ||
        (block_bytes && (!arena.base || reinterpret_cast<uintptr_t>(arena.base) % GRAPH_DEFINITION_OBJECT_ALIGN != 0 ||
                         arena.object_prefix_bytes != sizeof(GraphDefinitionHeader) ||
                         arena.object_align != GRAPH_DEFINITION_OBJECT_ALIGN)))
        return false;
    for (const GraphHostDefinition &entry : definitions.entries) {
        if (entry.bytes < sizeof(GraphDefinition) ||
            entry.bytes > SIZE_MAX - sizeof(GraphDefinitionHeader) - (GRAPH_DEFINITION_OBJECT_ALIGN - 1))
            return false;
        if (entry.spill == nullptr) {
            if (!arena.base || entry.object_offset > block_bytes ||
                sizeof(GraphDefinitionHeader) > block_bytes - entry.object_offset ||
                entry.bytes > block_bytes - entry.object_offset - sizeof(GraphDefinitionHeader) ||
                entry.object_offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0)
                return false;
            plan.objects.emplace(
                entry.full_key, PackedDefinition{entry.object_offset, entry.bytes, nullptr, {}, false}
            );
            continue;
        }
        const size_t object_offset = block_bytes;
        const size_t padded = align_up(sizeof(GraphDefinitionHeader) + entry.bytes);
        if (padded > SIZE_MAX - block_bytes) return false;
        block_bytes += padded;
        plan.objects.emplace(entry.full_key, PackedDefinition{object_offset, entry.bytes, entry.spill, {}, false});
        plan.spilled++;
    }

    plan.bytes = block_bytes;
    for (size_t index = 0; index < count; ++index) {
        std::optional<GraphHostUpload> upload = graph_host_upload(graph_state, index);
        if (!upload.has_value() || upload->outer_slot == nullptr || upload->outer_slot->task_kind != TaskKind::GRAPH) {
            LOG_ERROR("host-orch: invalid pending Graph task");
            return false;
        }
        auto object_it = plan.objects.find(upload->full_key);
        if (object_it == plan.objects.end()) {
            LOG_ERROR("host-orch: Graph task has no matching Definition object");
            return false;
        }
        const auto &object = object_it->second;
        const auto *image =
            object.copy != nullptr ? object.copy : arena.base + object.object_offset + sizeof(GraphDefinitionHeader);
        const auto *definition = reinterpret_cast<const GraphDefinition *>(image);
        if (definition->total_bytes != object_it->second.image_bytes) {
            LOG_ERROR("host-orch: Graph task has no matching Definition object");
            return false;
        }
        GraphExecutionStorageLayout storage_layout{};
        if (definition->task_count <= 0 || definition->task_count > MAX_IN_GRAPH_TASKS ||
            definition->full_key != upload->full_key ||
            !graph_execution_storage_layout(
                definition->task_count, definition->tensor_arg_count, definition->scalar_arg_count, &storage_layout
            ) ||
            storage_layout.total_bytes != definition->execution_storage_bytes ||
            upload->outer_slot->to_payload().tensor_count != definition->boundary_count ||
            upload->outer_slot->to_payload().scalar_count != definition->boundary_scalar_count) {
            LOG_ERROR("host-orch: invalid Graph Definition for task");
            return false;
        }
        const uintptr_t outer_base =
            reinterpret_cast<uintptr_t>(upload->outer_slot->to_descriptor().packed_buffer_base);
        const uintptr_t outer_end = reinterpret_cast<uintptr_t>(upload->outer_slot->to_descriptor().packed_buffer_end);
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
        PackedDefinition &packed_definition = object_it->second;
        if (!packed_definition.populations_ready) {
            const InGraphTaskDefinition *tasks = graph_definition_array<InGraphTaskDefinition>(
                *definition, definition->off_in_graph_tasks, definition->task_count
            );
            if (tasks == nullptr) {
                LOG_ERROR("host-orch: invalid Graph Definition in-graph task array");
                return false;
            }
            for (int32_t i = 0; i < definition->task_count; ++i) {
                // Sizing takes the kind materialize will give this task. add_task
                // singles out GRAPH and routes everything else by shape, and a Graph
                // body member is never the shell, so the shape decides. Derived here
                // the same way the device derives it, so the two cannot drift.
                const ActiveMask mask(tasks[i].active_mask);
                packed_definition.ready_queue_populations.add_task(
                    mask, TaskAttrs(tasks[i].task_attrs), mask.is_dummy() ? TaskKind::DUMMY : TaskKind::KERNEL
                );
            }
            packed_definition.populations_ready = true;
        }
        populations.add(packed_definition.ready_queue_populations);
        plan.bindings.emplace_back(upload->outer_slot, object.object_offset);
    }
    return true;
}

bool upload_graph_definitions(const HostApi *api, GraphBuild &build) {
    auto &graph_state = *build.graph_state;
    const auto &plan = *build.definitions;
    const auto align_up = [](size_t value) {
        return (value + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
    };
    void *block = nullptr;
    std::byte *staging = nullptr;
    if (plan.bytes != 0) {
        void *staging_addr = nullptr;
        // Growing the staging preserves what the recorders wrote into it, and the
        // offsets above name positions rather than addresses, so a block that moves
        // here costs nothing. Nothing is recording by now, which is what makes the
        // move safe at all.
        if (api->acquire_graph_definition_block(plan.bytes, GRAPH_DEFINITION_OBJECT_ALIGN, &block, &staging_addr) !=
            0) {
            LOG_ERROR(
                "host-orch: failed to retain %zu bytes for %zu Graph Definition object(s)", plan.bytes,
                plan.objects.size()
            );
            return false;
        }
        staging = static_cast<std::byte *>(staging_addr);
        // The owner may have moved staging. Publish its new base even if a
        // later copy fails, so retry and repeated upload retain a valid source.
        if (!graph_host_rebind_staging(graph_state, staging, plan.bytes)) return false;
        build.workspace.definitions.base = staging;
        build.workspace.definitions.capacity = plan.bytes;
        if (!block) return false;
        for (const auto &[key, object] : plan.objects) {
            std::byte *base = staging + object.object_offset;
            std::byte *image = base + sizeof(GraphDefinitionHeader);
            if (object.copy != nullptr) std::memcpy(image, object.copy, object.image_bytes);
            // Built value-initialized and copied over the whole header, so every byte
            // of the object's framing — padding included — is defined by this write
            // rather than by what the retained staging held before it.
            const auto *definition = reinterpret_cast<const GraphDefinition *>(image);
            GraphDefinitionHeader framing{};
            framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
            framing.full_key = definition->full_key;
            framing.definition_bytes = definition->total_bytes;
            std::memcpy(base, &framing, sizeof(framing));
            const size_t object_bytes = sizeof(GraphDefinitionHeader) + object.image_bytes;
            const size_t padded = align_up(object_bytes);
            std::memset(base + object_bytes, 0, padded - object_bytes);
        }
        if (api->copy_to_device(block, staging, plan.bytes) != 0) {
            LOG_ERROR("host-orch: failed to upload the Graph Definition block");
            return false;
        }
    }

    for (const auto &[slot, offset] : plan.bindings) {
        slot->graph_context = reinterpret_cast<GraphDefinition *>(
            reinterpret_cast<uintptr_t>(block) + offset + sizeof(GraphDefinitionHeader)
        );
    }
    return true;
}

}  // namespace simpler::hbg
