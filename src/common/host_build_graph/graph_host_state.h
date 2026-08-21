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
#include <memory>
#include <optional>
#include <vector>

struct PTO2TaskSlotState;
struct GraphHostState;

inline constexpr size_t GRAPH_MAX_DEFINITIONS = 16;

struct GraphHostStateDeleter {
    void operator()(GraphHostState *state) const noexcept;
};

using GraphHostStatePtr = std::unique_ptr<GraphHostState, GraphHostStateDeleter>;

struct GraphHostUpload {
    PTO2TaskSlotState *outer_slot;
    uint64_t full_key;
    uint64_t definition_hash;
};

// The run's distinct Definition images (already deduplicated by the host-side
// Definition cache), for upload as shared device objects ahead of submissions.
struct GraphHostDefinition {
    uint64_t full_key;
    const std::byte *data;
    size_t bytes;
};

struct GraphHostDefinitionList {
    std::vector<GraphHostDefinition> entries;
};

GraphHostStatePtr make_graph_host_state();
size_t graph_host_upload_count(const GraphHostState &state);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);
GraphHostDefinitionList graph_host_definitions(GraphHostState &state);
