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

#include "utils/device_arena.h"
#include "worker/runtime_c_api.h"

namespace simpler::hbg {

// Requirements are internal C++ values, but a plan may only combine snapshots
// whose images use the same runtime layout. Keep that condition explicit rather
// than relying on byte totals that can happen to match across architectures.
enum class RuntimeArchitecture : uint32_t {
    A2A3 = 1,
    A5 = 2,
};

inline constexpr uint32_t HBG_RUNTIME_LAYOUT_ABI_VERSION = 1;

struct RuntimeLayoutKey {
    uint32_t abi_version{0};
    RuntimeArchitecture architecture{};
    uint64_t task_capacity{0};
    uint64_t arena_bytes{0};
    uint64_t copied_begin{0};
    uint64_t copied_end{0};

    bool valid() const {
        return abi_version == HBG_RUNTIME_LAYOUT_ABI_VERSION &&
               (architecture == RuntimeArchitecture::A2A3 || architecture == RuntimeArchitecture::A5) &&
               task_capacity != 0 && copied_begin <= copied_end && copied_end == arena_bytes;
    }
};

inline bool operator==(const RuntimeLayoutKey &lhs, const RuntimeLayoutKey &rhs) {
    return lhs.abi_version == rhs.abi_version && lhs.architecture == rhs.architecture &&
           lhs.task_capacity == rhs.task_capacity && lhs.arena_bytes == rhs.arena_bytes &&
           lhs.copied_begin == rhs.copied_begin && lhs.copied_end == rhs.copied_end;
}

// Host-only snapshot produced from one completed H1 GraphBuild. These are
// required usable bytes, not committed HBM and not a context's frozen capacity.
// Caller tensors, resident code and CANN-owned capture packets are excluded.
struct GraphResourceRequirements {
    RuntimeLayoutKey layout{};
    uint64_t gm_heap_bytes{0};
    uint64_t runtime_image_bytes{0};
    uint64_t graph_definition_bytes{0};
    uint64_t scheduler_state_bytes{0};

    bool logical_device_bytes(uint64_t &bytes) const {
        if (!layout.valid() || gm_heap_bytes == 0 || runtime_image_bytes == 0 ||
            gm_heap_bytes > UINT64_MAX - runtime_image_bytes) {
            return false;
        }
        uint64_t total = gm_heap_bytes + runtime_image_bytes;
        for (uint64_t extra : {graph_definition_bytes, scheduler_state_bytes}) {
            if (extra > UINT64_MAX - total) return false;
            total += extra;
        }
        bytes = total;
        return true;
    }
};

// Host-only capacity plan for one prepared HBG kernel slot. Region maxima are
// taken independently and then repacked with the real device alignment. The
// plan does not allocate or freeze storage; the kernel-context prepare phase
// must commit runtime_slot_bytes() and gm_heap capacity before any launch.
class GraphCapacityPlan {
public:
    static int create(const GraphResourceRequirements *graphs, size_t count, GraphCapacityPlan &out) {
        if (graphs == nullptr || count == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;

        GraphCapacityPlan next;
        next.capacity_.layout = graphs[0].layout;
        for (size_t i = 0; i < count; ++i) {
            uint64_t ignored = 0;
            if (!graphs[i].logical_device_bytes(ignored) || !(graphs[i].layout == next.capacity_.layout)) {
                return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
            }
            next.capacity_.gm_heap_bytes = std::max(next.capacity_.gm_heap_bytes, graphs[i].gm_heap_bytes);
            next.capacity_.runtime_image_bytes =
                std::max(next.capacity_.runtime_image_bytes, graphs[i].runtime_image_bytes);
            next.capacity_.graph_definition_bytes =
                std::max(next.capacity_.graph_definition_bytes, graphs[i].graph_definition_bytes);
            next.capacity_.scheduler_state_bytes =
                std::max(next.capacity_.scheduler_state_bytes, graphs[i].scheduler_state_bytes);
        }

        uint64_t cursor = next.capacity_.runtime_image_bytes;
        if (!append_region(next.capacity_.graph_definition_bytes, cursor, next.definition_offset_) ||
            !append_region(next.capacity_.scheduler_state_bytes, cursor, next.scheduler_offset_) ||
            cursor > UINT64_MAX - next.capacity_.gm_heap_bytes) {
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        }
        next.runtime_slot_bytes_ = cursor;
        next.initialized_ = true;
        out = next;
        return 0;
    }

    const GraphResourceRequirements &capacity() const { return capacity_; }
    uint64_t definition_offset() const { return definition_offset_; }
    uint64_t scheduler_offset() const { return scheduler_offset_; }
    uint64_t runtime_slot_bytes() const { return runtime_slot_bytes_; }

    bool admits(const GraphResourceRequirements &graph) const {
        uint64_t ignored = 0;
        return initialized_ && graph.logical_device_bytes(ignored) && graph.layout == capacity_.layout &&
               graph.gm_heap_bytes <= capacity_.gm_heap_bytes &&
               graph.runtime_image_bytes <= capacity_.runtime_image_bytes &&
               graph.graph_definition_bytes <= capacity_.graph_definition_bytes &&
               graph.scheduler_state_bytes <= capacity_.scheduler_state_bytes;
    }

    PipelineContract pipeline_contract() const {
        if (!initialized_) return {};
        return {
            PTO_PIPELINE_CONTRACT_ABI_VERSION,
            4,
            1,
            {
                {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, capacity_.gm_heap_bytes},
                {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, runtime_slot_bytes_},
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
    uint64_t runtime_slot_bytes_{0};
    bool initialized_{false};
};

}  // namespace simpler::hbg
