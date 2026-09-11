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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/graph_host_state.h"
#include "host_build_graph/kernel_graph_slot_wire.h"
#include "worker/pipeline_contract.h"
#include "host/kernel_execution_state.h"
#include "utils/device_arena.h"

namespace hbg {

static_assert(GRAPH_DEFINITION_OBJECT_ALIGN <= DeviceArena::kDefaultBaseAlign);

// Host-only value snapshot, independent of the build's borrowed buffers. Sizes
// exclude caller tensor storage, code residency and CANN-owned capture packets.
struct GraphResourceRequirements {
    uint64_t gm_heap_bytes{0};
    uint64_t runtime_arena_bytes{0};  // Includes the compact SM image exactly once.
    uint64_t graph_definition_bytes{0};
    uint64_t scheduler_state_bytes{0};  // A5 upper bound; zero for the Graph fallback.

    // Logical storage requirements, not committed HBM or a frozen capacity.
    bool required_bytes(uint64_t &bytes) const {
        if (gm_heap_bytes == 0 || runtime_arena_bytes == 0 || gm_heap_bytes > UINT64_MAX - runtime_arena_bytes)
            return false;
        uint64_t total = gm_heap_bytes + runtime_arena_bytes;
        for (uint64_t extra : {graph_definition_bytes, scheduler_state_bytes}) {
            if (extra > UINT64_MAX - total) return false;
            total += extra;
        }
        bytes = total;
        return true;
    }
};

// A host-only capacity plan for one context execution slot. All graph-specific
// destinations fit inside one runtime arena, so the common contract accounts
// for Definition and A5 scheduler storage as well as the runtime/SM image.
// The owner must allocate and freeze these destinations before kernel launch;
// constructing a plan neither allocates device storage nor freezes a context.
// Inputs must share the context's architecture and runtime layout ABI. The
// runtime/SM region holds one complete per-invocation image; its internal offsets
// are never merged between graphs. The restore path must bind that image anew.
class KernelResourcePlan {
public:
    static int create(const GraphResourceRequirements *graphs, size_t count, KernelResourcePlan &out) {
        if (graphs == nullptr || count == 0) return PTO_RUNTIME_ERR_INTERNAL;
        KernelResourcePlan next;
        for (size_t i = 0; i < count; ++i) {
            uint64_t bytes = 0;
            if (!graphs[i].required_bytes(bytes)) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
            next.capacity_.gm_heap_bytes = std::max(next.capacity_.gm_heap_bytes, graphs[i].gm_heap_bytes);
            next.capacity_.runtime_arena_bytes =
                std::max(next.capacity_.runtime_arena_bytes, graphs[i].runtime_arena_bytes);
            next.capacity_.graph_definition_bytes =
                std::max(next.capacity_.graph_definition_bytes, graphs[i].graph_definition_bytes);
            next.capacity_.scheduler_state_bytes =
                std::max(next.capacity_.scheduler_state_bytes, graphs[i].scheduler_state_bytes);
        }
        // Recompute the packed layout from region capacities. Never merge old
        // offsets or sum/max the total sizes of already-packed graph layouts.
        uint64_t cursor = next.capacity_.runtime_arena_bytes;
        if (!append_region(next.capacity_.graph_definition_bytes, cursor, next.definition_offset_) ||
            !append_region(next.capacity_.scheduler_state_bytes, cursor, next.scheduler_offset_) ||
            !append_region(sizeof(GraphSlotRegistry), cursor, next.registry_offset_) ||
            cursor > UINT64_MAX - next.capacity_.gm_heap_bytes) {
            return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        }
        next.runtime_arena_bytes_ = cursor;
        out = next;
        return 0;
    }

    const GraphResourceRequirements &capacity() const { return capacity_; }
    uint64_t definition_offset() const { return definition_offset_; }
    uint64_t scheduler_offset() const { return scheduler_offset_; }
    uint64_t registry_offset() const { return registry_offset_; }
    uint64_t runtime_arena_bytes() const { return runtime_arena_bytes_; }

    bool admits(const GraphResourceRequirements &graph) const {
        uint64_t bytes = 0;
        return runtime_arena_bytes_ != 0 && graph.required_bytes(bytes) &&
               graph.gm_heap_bytes <= capacity_.gm_heap_bytes &&
               graph.runtime_arena_bytes <= capacity_.runtime_arena_bytes &&
               graph.graph_definition_bytes <= capacity_.graph_definition_bytes &&
               graph.scheduler_state_bytes <= capacity_.scheduler_state_bytes;
    }

    // Only context resource prepare receives allocator operations. Launch gets
    // stable views through bind_kernel_resources_for_launch below.
    int prepare(KernelExecutionState &context, const KernelResourceOps &ops) const {
        KernelResourceLayout layout{
            resource_schema,
            pipeline_contract(),
            {{PTO_PIPELINE_GM_HEAP, 0, capacity_.gm_heap_bytes},
             {PTO_PIPELINE_RUNTIME_IMAGE, 0, capacity_.runtime_arena_bytes},
             {PTO_PIPELINE_RUNTIME_IMAGE, definition_offset_, capacity_.graph_definition_bytes},
             {PTO_PIPELINE_RUNTIME_IMAGE, scheduler_offset_, capacity_.scheduler_state_bytes},
             {PTO_PIPELINE_RUNTIME_IMAGE, registry_offset_, sizeof(GraphSlotRegistry)}},
        };
        return context.prepare_resources(layout, ops);
    }

    static constexpr uint64_t resource_schema = 0x4842470000000002ULL;

    PipelineContract pipeline_contract() const {
        return {
            PTO_PIPELINE_CONTRACT_ABI_VERSION,
            4,
            1,
            {
                {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, capacity_.gm_heap_bytes},
                {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, runtime_arena_bytes_},
                {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
                {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            },
        };
    }

private:
    static bool append_region(uint64_t bytes, uint64_t &cursor, uint64_t &offset) {
        if (bytes == 0) return true;
        constexpr uint64_t alignment = DeviceArena::kDefaultBaseAlign;
        if (cursor > UINT64_MAX - (alignment - 1)) return false;
        const uint64_t aligned = (cursor + alignment - 1) & ~(alignment - 1);
        if (bytes > UINT64_MAX - aligned) return false;
        offset = aligned;
        cursor = aligned + bytes;
        return true;
    }

    GraphResourceRequirements capacity_{};
    uint64_t definition_offset_{0};
    uint64_t scheduler_offset_{0};
    uint64_t registry_offset_{0};
    uint64_t runtime_arena_bytes_{0};
};

// Borrowed addresses in the context's frozen execution slot. Device restore
// writes invocation-specific state here; the Host never mutates it in launch.
struct KernelWorkingBinding {
    KernelResourceView heap;
    KernelResourceView runtime_image;
    KernelResourceView definitions;
    KernelResourceView scheduler;
};

inline int bind_kernel_resources_for_launch(
    const KernelExecutionState &context, int device_id, uint64_t generation, const GraphResourceRequirements &graph,
    KernelWorkingBinding &out
) {
    uint64_t total = 0;
    if (!graph.required_bytes(total)) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    const uint64_t required[] = {
        graph.gm_heap_bytes, graph.runtime_arena_bytes, graph.graph_definition_bytes, graph.scheduler_state_bytes,
        sizeof(GraphSlotRegistry)
    };
    KernelResourceBinding binding;
    const int rc = context.bind_resources_for_launch(
        device_id, generation, KernelResourcePlan::resource_schema, required, 5, binding
    );
    if (rc != 0) return rc;
    out = {binding.regions[0], binding.regions[1], binding.regions[2], binding.regions[3]};
    return 0;
}

// Same packing as bind_graph_definitions: retained prefix plus aligned spill
// objects, each distinct Definition once regardless of its submission count.
inline bool
graph_definition_block_bytes(const GraphHostDefinitionList &definitions, uint64_t arena_used, uint64_t &bytes) {
    if (arena_used % GRAPH_DEFINITION_OBJECT_ALIGN != 0) return false;
    uint64_t total = arena_used;
    for (const GraphHostDefinition &entry : definitions.entries) {
        if (entry.bytes < sizeof(GraphDefinition) ||
            entry.bytes > UINT64_MAX - sizeof(GraphDefinitionHeader) - (GRAPH_DEFINITION_OBJECT_ALIGN - 1))
            return false;
        const uint64_t object_bytes =
            (sizeof(GraphDefinitionHeader) + entry.bytes + GRAPH_DEFINITION_OBJECT_ALIGN - 1) &
            ~(static_cast<uint64_t>(GRAPH_DEFINITION_OBJECT_ALIGN) - 1);
        if (entry.spill == nullptr) {
            if (entry.object_offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0 || entry.object_offset > arena_used ||
                object_bytes > arena_used - entry.object_offset)
                return false;
        } else {
            if (entry.object_offset != GRAPH_NO_OBJECT_OFFSET || object_bytes > UINT64_MAX - total) return false;
            total += object_bytes;
        }
    }
    bytes = total;
    return true;
}

}  // namespace hbg
