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
#include <utility>
#include <vector>

#include "host_build_graph/host/graph_host_state.h"

namespace hbg {

struct DefinitionPackingObject {
    uint64_t graph_key;
    size_t object_offset;
    size_t image_bytes;
    const std::byte *copy;
};

// Offsets describe the launch section; spill pointers borrow the completed build.
// Measuring acquires no device storage and never reads tensor contents.
struct DefinitionPackingPlan {
    std::vector<DefinitionPackingObject> objects;
    size_t bytes{0};
    size_t spilled{0};
};

inline bool
measure_definition_packing(const GraphHostDefinitionList &definitions, size_t arena_used, DefinitionPackingPlan &out) {
    constexpr size_t alignment = GRAPH_DEFINITION_OBJECT_ALIGN;
    if (arena_used % alignment != 0 || arena_used > UINT32_MAX) return false;
    DefinitionPackingPlan next;
    next.bytes = arena_used;
    next.objects.reserve(definitions.entries.size());
    for (const auto &entry : definitions.entries) {
        if (entry.bytes < sizeof(GraphDefinition) || entry.bytes > UINT32_MAX ||
            entry.bytes > SIZE_MAX - sizeof(GraphDefinitionHeader) - (alignment - 1))
            return false;
        const size_t object_bytes = (sizeof(GraphDefinitionHeader) + entry.bytes + alignment - 1) & ~(alignment - 1);
        size_t offset = entry.object_offset;
        if (entry.spill != nullptr) {
            if (offset != GRAPH_NO_OBJECT_OFFSET || object_bytes > UINT32_MAX - next.bytes) return false;
            offset = next.bytes;
            next.bytes += object_bytes;
            ++next.spilled;
        } else if (offset % alignment != 0 || offset > arena_used || object_bytes > arena_used - offset) {
            return false;
        }
        for (const auto &previous : next.objects) {
            if (entry.graph_key == previous.graph_key) return false;
            const size_t previous_bytes =
                (sizeof(GraphDefinitionHeader) + previous.image_bytes + alignment - 1) & ~(alignment - 1);
            if (offset < previous.object_offset + previous_bytes && previous.object_offset < offset + object_bytes)
                return false;
        }
        next.objects.push_back({entry.graph_key, offset, entry.bytes, entry.spill});
    }
    out = std::move(next);
    return true;
}

}  // namespace hbg
