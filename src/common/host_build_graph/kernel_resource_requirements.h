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

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/kernel_graph_slot_protocol.h"
#include "host/kernel_execution_state.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime_status.h"
#include "worker/runtime_c_api.h"

namespace hbg {

struct RuntimeLayoutKey {
    RuntimeArchitecture architecture{};
    uint64_t task_capacity{0};
    uint64_t arena_bytes{0};
    uint64_t copied_begin{0};
    uint64_t copied_end{0};

    bool valid() const {
        return (architecture == RuntimeArchitecture::A2A3 || architecture == RuntimeArchitecture::A5) &&
               task_capacity != 0 && copied_begin < copied_end && copied_end == arena_bytes;
    }
    bool operator==(const RuntimeLayoutKey &other) const {
        return architecture == other.architecture && task_capacity == other.task_capacity &&
               arena_bytes == other.arena_bytes && copied_begin == other.copied_begin && copied_end == other.copied_end;
    }
};

// Logical storage, excluding caller tensor contents, callable residency and RTS
// capture packets. The compact SM image is counted once, in runtime_arena_bytes.
struct GraphResourceRequirements {
    RuntimeLayoutKey layout{};
    uint64_t gm_heap_bytes{0};
    uint64_t runtime_arena_bytes{0};
    uint64_t graph_definition_bytes{0};
    uint64_t scheduler_state_bytes{0};

    bool required_bytes(uint64_t &out) const {
        if (!layout.valid() || gm_heap_bytes == 0 || runtime_arena_bytes < layout.arena_bytes) return false;
        uint64_t total = 0;
        for (uint64_t bytes : {gm_heap_bytes, runtime_arena_bytes, graph_definition_bytes, scheduler_state_bytes}) {
            if (bytes > UINT64_MAX - total) return false;
            total += bytes;
        }
        out = total;
        return true;
    }
};

inline int32_t measure_graph_resources(
    const GraphBuild &build, const RuntimeArenaLayout &layout, RuntimeArchitecture architecture,
    uint64_t scheduler_state_bytes, GraphResourceRequirements &out
) {
    if (!build.build_complete || !build.graph_state) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (build.total_tasks < 0 || static_cast<uint64_t>(build.total_tasks) > build.task_capacity ||
        build.task_capacity != layout.task_capacity || build.heap_bytes == 0 || build.image_bytes == 0 ||
        build.bind_usage.submitted_tasks != static_cast<uint64_t>(build.total_tasks) ||
        build.definitions.bytes > UINT32_MAX)
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (build.image_bytes > UINT64_MAX - layout.off_copied_end)
        return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    GraphResourceRequirements next{
        {architecture, layout.task_capacity, layout.arena_size, layout.off_copied_begin, layout.off_copied_end},
        build.heap_bytes,
        layout.off_copied_end + build.image_bytes,
        build.definitions.bytes,
        scheduler_state_bytes
    };
    uint64_t total = 0;
    if (!next.required_bytes(total)) return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    out = next;
    return 0;
}

// One serial execution slot reserves each region's maximum independently.
// This value does not allocate, freeze a context, or authorize device execution preparation.
class KernelResourceCapacity {
public:
    static int32_t create(const GraphResourceRequirements *graphs, size_t count, KernelResourceCapacity &out) {
        if (graphs == nullptr || count == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        KernelResourceCapacity next;
        next.capacity_.layout = graphs[0].layout;
        for (size_t i = 0; i < count; ++i) {
            uint64_t total = 0;
            if (!(graphs[i].layout == next.capacity_.layout)) return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
            if (!graphs[i].required_bytes(total))
                return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
            next.capacity_.gm_heap_bytes = std::max(next.capacity_.gm_heap_bytes, graphs[i].gm_heap_bytes);
            next.capacity_.runtime_arena_bytes =
                std::max(next.capacity_.runtime_arena_bytes, graphs[i].runtime_arena_bytes);
            next.capacity_.graph_definition_bytes =
                std::max(next.capacity_.graph_definition_bytes, graphs[i].graph_definition_bytes);
            next.capacity_.scheduler_state_bytes =
                std::max(next.capacity_.scheduler_state_bytes, graphs[i].scheduler_state_bytes);
        }
        uint64_t total = 0;
        if (!next.capacity_.required_bytes(total))
            return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
        out = next;
        return 0;
    }
    const GraphResourceRequirements &capacity() const { return capacity_; }
    bool admits(const GraphResourceRequirements &graph) const {
        uint64_t total = 0;
        return graph.required_bytes(total) && graph.layout == capacity_.layout &&
               graph.gm_heap_bytes <= capacity_.gm_heap_bytes &&
               graph.runtime_arena_bytes <= capacity_.runtime_arena_bytes &&
               graph.graph_definition_bytes <= capacity_.graph_definition_bytes &&
               graph.scheduler_state_bytes <= capacity_.scheduler_state_bytes;
    }

    int32_t prepare(KernelExecutionState &context, const KernelResourceOps &ops) const {
        KernelResourceLayout layout;
        const int status = resource_layout(layout);
        return status == 0 ? context.prepare_resources(layout, ops) : status;
    }

    int32_t make_registration(
        const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t binary_id,
        GraphSlotRegistration &out
    ) const {
        if (binary_id == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        const uint64_t required[] = {
            capacity_.gm_heap_bytes, capacity_.runtime_arena_bytes, capacity_.graph_definition_bytes,
            capacity_.scheduler_state_bytes, sizeof(GraphSlotRegistry)
        };
        KernelResourceBinding binding;
        const int status =
            context.inspect_frozen_resources(device_id, generation, resource_schema, required, 5, binding);
        if (status != 0) return status;
        GraphSlotRegistration next{};
        next.magic = GRAPH_SLOT_MAGIC;
        next.version = GRAPH_SLOT_VERSION;
        next.bytes = sizeof(next);
        next.flags = GRAPH_SLOT_FROZEN_SERIAL;
        next.device_id = device_id;
        next.slot_generation = generation;
        next.runtime_binary_id = binary_id;
        for (size_t i = 0; i < 4; ++i)
            next.destinations[i] = {binding.regions[i].address, binding.regions[i].capacity};
        next.registry = {binding.regions[4].address, binding.regions[4].capacity};
        if (!calculate_graph_slot_packet_capacity(next.destinations, next.max_packet_bytes))
            return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
        next.checksum = calculate_graph_slot_registration_checksum(next);
        if (!is_valid_graph_slot_registration(next)) return PTO_RUNTIME_ERR_INTERNAL;
        out = next;
        return 0;
    }

private:
    int32_t resource_layout(KernelResourceLayout &out) const {
        uint64_t total = 0;
        if (!capacity_.required_bytes(total)) return PTO_RUNTIME_ERR_INVALID_STATE;
        constexpr uint64_t alignment = DeviceArena::kDefaultBaseAlign;
        uint64_t cursor = capacity_.runtime_arena_bytes;
        uint64_t offsets[3]{};
        const uint64_t sizes[] = {
            capacity_.graph_definition_bytes, capacity_.scheduler_state_bytes, sizeof(GraphSlotRegistry)
        };
        for (size_t i = 0; i < 3; ++i) {
            if (sizes[i] == 0) continue;
            if (cursor > UINT64_MAX - (alignment - 1)) return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
            const uint64_t offset = (cursor + alignment - 1) & ~(alignment - 1);
            if (sizes[i] > UINT64_MAX - offset) return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
            offsets[i] = offset;
            cursor = offset + sizes[i];
        }
        KernelResourceLayout next;
        next.schema = resource_schema;
        next.contract = {
            PTO_PIPELINE_CONTRACT_ABI_VERSION,
            4,
            1,
            {{PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, capacity_.gm_heap_bytes},
             {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, cursor},
             {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
             {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0}}
        };
        next.regions = {
            {PTO_PIPELINE_GM_HEAP, 0, capacity_.gm_heap_bytes},
            {PTO_PIPELINE_RUNTIME_IMAGE, 0, capacity_.runtime_arena_bytes},
            {PTO_PIPELINE_RUNTIME_IMAGE, offsets[0], sizes[0]},
            {PTO_PIPELINE_RUNTIME_IMAGE, offsets[1], sizes[1]},
            {PTO_PIPELINE_RUNTIME_IMAGE, offsets[2], sizes[2]}
        };
        out = std::move(next);
        return 0;
    }
    static constexpr uint64_t resource_schema = 0x4842470000000004ULL;
    GraphResourceRequirements capacity_{};
};

}  // namespace hbg
