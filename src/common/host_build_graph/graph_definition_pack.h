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

#include "host_build_graph/graph_host_state.h"
#include "host_build_graph/ready_queue_sizing.h"

struct HostApi;

namespace simpler::hbg {

struct PackedDefinition {
    size_t object_offset;
    size_t image_bytes;
    const std::byte *copy;
    ReadyQueuePopulations ready_queue_populations;
    bool populations_ready;
};

// Retained objects use offsets; spill images and task slots borrow GraphBuild's
// records and workspace. No device address is resolved while packing.
struct GraphDefinitionPlan {
    std::unordered_map<uint64_t, PackedDefinition> objects;
    std::vector<std::pair<ChipTaskSlotState *, size_t>> bindings;
    size_t bytes{0};
    size_t spilled{0};
};

bool pack_graph_definitions(
    GraphHostState &state, const GraphDefinitionArena &arena, GraphDefinitionPlan &plan,
    ReadyQueuePopulations &populations
);
struct GraphBuild;
bool upload_graph_definitions(const HostApi *api, GraphBuild &build);

}  // namespace simpler::hbg
