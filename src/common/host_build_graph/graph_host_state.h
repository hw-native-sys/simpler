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

struct ChipTaskSlotState;
struct GraphHostState;

inline constexpr size_t GRAPH_MAX_DEFINITIONS = 16;

// The host block a run builds its Definition objects into. An object is
// [prefix][image]: the upload owner declares how much room it fills ahead of
// every image and the alignment every object base carries, so a recorder writes
// only the image and never needs to know what the prefix holds. Objects are
// packed from the base by a bump cursor and padded to `object_align`. The
// platform retains the block across binds, so a run whose Definitions fit
// `capacity` acquires no host memory for them at all.
struct GraphDefinitionArena {
    std::byte *base{nullptr};
    size_t capacity{0};
    size_t object_prefix_bytes{0};
    size_t object_align{1};
};

// A Definition the arena had no room for. It carries its own image and the
// upload copies it into the object it assigns, which is what makes a bind
// correct — never merely slower — when it outgrows the retained capacity.
inline constexpr size_t GRAPH_NO_OBJECT_OFFSET = static_cast<size_t>(-1);

struct GraphHostStateDeleter {
    void operator()(GraphHostState *state) const noexcept;
};

using GraphHostStatePtr = std::unique_ptr<GraphHostState, GraphHostStateDeleter>;

struct GraphHostUpload {
    ChipTaskSlotState *outer_slot;
    uint64_t full_key;
};

// The run's distinct Definition images (already deduplicated by the host-side
// Definition cache), for upload as shared device objects ahead of submissions.
// `object_offset` locates the object in the arena rather than naming its address,
// so the arena may be reallocated — preserving its content — between publication
// and upload. It is GRAPH_NO_OBJECT_OFFSET exactly when `spill` is set, which is
// then the image the upload must copy into an object of its own choosing.
struct GraphHostDefinition {
    uint64_t full_key;
    size_t object_offset;
    const std::byte *spill;
    size_t bytes;
};

struct GraphHostDefinitionList {
    std::vector<GraphHostDefinition> entries;
};

GraphHostStatePtr make_graph_host_state(const GraphDefinitionArena &arena);

/**
 * Stand the calling thread's recording storage up — hazard map, in-graph task slots, the
 * flat per-task arrays and the task tensor pool — without recording anything.
 *
 * A recorder worker calls this once as it starts, so the allocations land at callable
 * registration rather than inside the first bind that worker serves, and a failure is
 * reported where the caller can still act on it. It is an optimization, not the only
 * stand-up point: a worker the pool creates after prewarm, and a thread whose storage was
 * dropped for overshooting the in-graph task cap, still stand up lazily on their next recording.
 *
 * @return false when an allocation failed; the failure is also counted for
 *         graph_recorder_storage_failures(), which is how the host notices across the
 *         .so boundary that carries no return value.
 */
bool graph_recorder_stand_up_storage();

/** Stand-up failures since the process started. Monotonic. */
size_t graph_recorder_storage_failures();
size_t graph_host_upload_count(const GraphHostState &state);
// Arena bytes this run has claimed: the prefix its objects occupy, and so the
// length of the region an upload must ship for the objects built in place.
size_t graph_host_arena_used(const GraphHostState &state);
// Rebind after the upload owner grows staging while preserving its contents.
// Recording must have completed; offsets and spill ownership remain unchanged.
bool graph_host_rebind_staging(GraphHostState &state, void *base, size_t capacity);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);
GraphHostDefinitionList graph_host_definitions(GraphHostState &state);

// Borrowed Definition image, valid under the same workspace lease as the build.
const std::byte *graph_host_definition_data(const GraphHostState &state, uint64_t full_key);
