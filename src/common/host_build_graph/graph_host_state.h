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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "host_build_graph/graph_recording.h"

struct ChipTaskSlotState;

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

struct GraphPendingUpload {
    ChipTaskSlotState *outer_slot{nullptr};
    uint64_t full_key{0};
    bool deferred_heap{false};
};

enum class GraphRecordingStatus : uint8_t { RECORDING = 0, READY = 1, FAILED = 2 };

// One Definition being recorded. Entries are keyed by Graph key and held by
// unique_ptr, so a rehash of the owning map never moves one: the recording
// thread is handed this address at graph_begin and dereferences it without
// taking recording_mutex.
//
// The entry carries the boundary and the status, not the recorded body: the body's
// storage belongs to the recorder thread that will fill it (recorder_recording()), so
// nothing here is sized by the graph.
struct GraphInflightRecording {
    uint64_t full_key{0};
    GraphBoundary boundary;
    // Atomic because graph_prepare reads it on the recording thread without
    // taking recording_mutex, by design: acquiring the mutex there lets a
    // main-thread burst of same-key submissions starve the thread before it can
    // bind its private recording state. Every other access holds the mutex.
    std::atomic<GraphRecordingStatus> recording_status{GraphRecordingStatus::RECORDING};

    GraphRecordingStatus status() const { return recording_status.load(std::memory_order_acquire); }
    void set_status(GraphRecordingStatus next) { recording_status.store(next, std::memory_order_release); }
};

// One published Definition image. It lives in the run's arena at `object_offset`
// — an offset, not an address, so the arena can be reallocated between
// publication and upload — unless the arena had no room, in which case `spill`
// holds the image and `object_offset` is GRAPH_NO_OBJECT_OFFSET.
struct GraphDefinitionRecord {
    size_t object_offset{GRAPH_NO_OBJECT_OFFSET};
    size_t bytes{0};
    std::vector<std::byte> spill;
    // The boundary this Definition was recorded against, reduced to what a later
    // invocation is checked against. Never reaches the device: materialize takes the
    // boundary from the outer task's own payload, which carries that invocation's
    // arguments.
    //
    // Each half holds one entry per parameter of its kind, sized to the boundary rather
    // than to the cap. The match path walks the whole of it on every same-key submission,
    // so its size is a per-submission cost: a boundary of N tensor parameters is N
    // contiguous cache lines.
    GraphBoundaryMatchInfo boundary_match_info;
};

struct GraphHostState {
    explicit GraphHostState(const GraphDefinitionArena &arena) :
        arena(arena) {}

    std::unordered_map<uint64_t, GraphDefinitionRecord> definitions;
    // Recordings in flight, at most one per Graph key. Several record at once,
    // each on its own thread; graph_commit drains and finalizes all of them.
    std::unordered_map<uint64_t, std::unique_ptr<GraphInflightRecording>> inflight;
    std::vector<GraphPendingUpload> pending_uploads;
    std::mutex recording_mutex;
    std::condition_variable recording_cv;
    // Mirrors inflight.size() so orchestration completion answers the common
    // "nothing is recording" case without taking recording_mutex.
    std::atomic<size_t> inflight_count{0};
    // Fixed for the run: recorder threads hold addresses inside it, so it must
    // not move while any of them is filling an image.
    GraphDefinitionArena arena;
    std::atomic<size_t> arena_cursor{0};

    // Claim room for one object of `image_bytes`, padded so the next object
    // starts aligned too. Returns the object offset, or nullopt when the run has
    // outgrown the retained arena — the caller then builds into its own buffer.
    // Several recording threads reserve at once and a losing exchange retries
    // rather than advancing the cursor past the capacity, so an object that does
    // not fit costs the run its own slot and no one else's.
    std::optional<size_t> reserve_object(size_t image_bytes) {
        if (arena.base == nullptr || arena.object_align == 0) return std::nullopt;
        if (image_bytes > SIZE_MAX - arena.object_prefix_bytes) return std::nullopt;
        const size_t object_bytes = arena.object_prefix_bytes + image_bytes;
        if (object_bytes > SIZE_MAX - (arena.object_align - 1)) return std::nullopt;
        const size_t claimed = (object_bytes + arena.object_align - 1) & ~(arena.object_align - 1);
        size_t offset = arena_cursor.load(std::memory_order_relaxed);
        while (true) {
            if (claimed > arena.capacity - offset) return std::nullopt;
            if (arena_cursor.compare_exchange_weak(
                    offset, offset + claimed, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                return offset;
            }
        }
    }

    // Where an object's image starts, for a record the arena holds.
    std::byte *image_at(size_t object_offset) const { return arena.base + object_offset + arena.object_prefix_bytes; }

    // Definitions this run can still admit: published plus in flight, against
    // the per-worker cache limit.
    size_t claimed_definitions() const { return definitions.size() + inflight.size(); }
    bool any_recording() const {
        for (const auto &entry : inflight) {
            if (entry.second->status() == GraphRecordingStatus::RECORDING) return true;
        }
        return false;
    }
};

struct GraphHostStateDeleter {
    void operator()(GraphHostState *state) const noexcept;
};

using GraphHostStatePtr = std::unique_ptr<GraphHostState, GraphHostStateDeleter>;

GraphHostStatePtr make_graph_host_state(const GraphDefinitionArena &arena);

// Where a published record's image is: in the arena at its object offset, or in
// the buffer it spilled to.
const GraphDefinition *graph_record_definition(const GraphHostState &state, const GraphDefinitionRecord &record);

size_t graph_host_upload_count(const GraphHostState &state);
// Arena bytes this run has claimed: the prefix its objects occupy, and so the
// length of the region an upload must ship for the objects built in place.
size_t graph_host_arena_used(const GraphHostState &state);
std::optional<GraphHostUpload> graph_host_upload(GraphHostState &state, size_t index);
GraphHostDefinitionList graph_host_definitions(GraphHostState &state);
