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
/**
 * A recording builds its Definition image directly into the arena the bind hands
 * the host state — the retained host staging its objects are uploaded from — so
 * the upload ships what the recorders wrote instead of copying it. Two properties
 * make that safe, and each is invisible from the scene tests, which pass either
 * way:
 *
 *   - the placement contract: object offsets are aligned and disjoint, and the
 *     claimed prefix covers every one of them, because the upload derives the
 *     region it ships from that prefix alone;
 *   - the fallback: a run the retained capacity cannot hold still publishes a
 *     valid image, in a buffer of its own, so outgrowing the arena costs a copy
 *     rather than the run.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"

class HbgGraphDefinitionArenaTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    std::vector<std::byte> staging;
    GraphDefinitionArena arena{};

    static constexpr size_t HEAP_BYTES = 64 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS));
    }

    void TearDown() override {
        orch.graph_host_state = nullptr;
        graph_state.reset();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // A bind's arena of `capacity` bytes. `capacity == 0` is the first bind of a
    // process, where the platform has nothing retained yet.
    void bind_arena(size_t capacity) {
        staging.assign(capacity, std::byte{0});
        arena = GraphDefinitionArena{};
        arena.base = capacity == 0 ? nullptr : staging.data();
        arena.capacity = capacity;
        arena.object_prefix_bytes = sizeof(GraphDefinitionHeader);
        arena.object_align = GRAPH_DEFINITION_OBJECT_ALIGN;
        graph_state = make_graph_host_state(arena);
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
    }

    const GraphDefinition *definition_image(const GraphHostDefinition &entry) const {
        const std::byte *image =
            entry.spill != nullptr ? entry.spill : arena.base + entry.object_offset + arena.object_prefix_bytes;
        return reinterpret_cast<const GraphDefinition *>(image);
    }

    // Record one Graph of `task_count` chained tasks under `graph_key`. The chain
    // makes the image's size a function of the count, so two keys recorded with
    // different counts cannot come out byte-identical and share one Definition.
    void record_graph(uint64_t graph_key, int task_count, const simpler::hbg::Tensor &boundary, const uint32_t *shape) {
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);
        const GraphScopeResult scope = orch.graph_begin(graph_key, boundary_args, 0x1736);
        ASSERT_TRUE(scope.recording);
        ASSERT_TRUE(scope.task_id.is_valid());
        ASSERT_TRUE(orch.graph_prepare(scope.recording_handle, boundary_args));
        // A body reads the entry's parameter list, not the caller's tensors: those carry
        // recording-space addresses and PARAM provenance, which is what the classifier
        // resolves against.
        simpler::hbg::Tensor input = scope.params->tensor(0).ref();
        for (int i = 0; i < task_count; ++i) {
            CoreTaskArgs task_args;
            task_args.add_input(input);
            TensorCreateInfo output(shape, 1, DataType::UINT32);
            task_args.add_output(output);
            TaskOutputTensors outputs = orch.submit_dummy_task(task_args);
            ASSERT_TRUE(outputs.task_id().is_valid());
            input = outputs.get_ref(0);
        }
        ASSERT_TRUE(orch.graph_end());
    }
};

TEST_F(HbgGraphDefinitionArenaTest, ObjectsAreBuiltInTheArenaAtAlignedDisjointOffsets) {
    bind_arena(256 * 1024);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    record_graph(0x1715, 1, boundary, shape);
    record_graph(0x1716, 3, boundary, shape);
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());

    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 2u);

    const size_t used = graph_host_arena_used(*graph_state);
    EXPECT_LE(used, arena.capacity);
    size_t claimed_total = 0;
    for (const GraphHostDefinition &entry : definitions.entries) {
        EXPECT_EQ(entry.spill, nullptr) << "an arena this size has room for every object";
        ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET);
        EXPECT_EQ(entry.object_offset % GRAPH_DEFINITION_OBJECT_ALIGN, 0u);
        const size_t object_bytes = arena.object_prefix_bytes + entry.bytes;
        const size_t claimed =
            (object_bytes + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
        claimed_total += claimed;
        EXPECT_LE(entry.object_offset + claimed, used) << "the claimed prefix must cover every object in it";
        const GraphDefinition *image = definition_image(entry);
        EXPECT_EQ(image->full_key, entry.full_key);
        EXPECT_EQ(image->total_bytes, entry.bytes);
        EXPECT_GT(image->task_count, 0u) << "the object holds a filled image, not just claimed bytes";
        EXPECT_EQ(reinterpret_cast<uintptr_t>(image) % GRAPH_DEFINITION_OBJECT_ALIGN, 0u)
            << "the image base carries the alignment its section offsets assume";
    }
    EXPECT_EQ(claimed_total, used) << "the prefix is exactly the objects, so the upload ships no unclaimed bytes";

    // Distinct offsets alone would pass if two objects overlapped, so pin the
    // disjointness the shared block depends on.
    const GraphHostDefinition &first = definitions.entries[0];
    const GraphHostDefinition &second = definitions.entries[1];
    const size_t first_end = first.object_offset + arena.object_prefix_bytes + first.bytes;
    const size_t second_end = second.object_offset + arena.object_prefix_bytes + second.bytes;
    EXPECT_TRUE(first_end <= second.object_offset || second_end <= first.object_offset)
        << "two objects must not share arena bytes";

    // Every shell resolves to exactly one published Definition, by the only name it
    // carries: an upload record that matched two entries, or none, would have the
    // upload validate a different image than the one the task was submitted against.
    for (size_t i = 0; i < graph_host_upload_count(*graph_state); ++i) {
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, i);
        ASSERT_TRUE(upload.has_value());
        size_t matches = 0;
        for (const GraphHostDefinition &entry : definitions.entries) {
            if (entry.full_key != upload->full_key) continue;
            EXPECT_EQ(definition_image(entry)->total_bytes, entry.bytes) << "shell " << i;
            matches++;
        }
        EXPECT_EQ(matches, 1u) << "shell " << i << " must name exactly one published Definition";
    }
}

TEST_F(HbgGraphDefinitionArenaTest, AnArenaWithNoRoomSpillsAndStillPublishesTheImage) {
    // What the first bind of a process sees: the platform has retained nothing yet.
    bind_arena(0);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    record_graph(0x1715, 2, boundary, shape);
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal()) << "outgrowing the arena must cost a copy, not the run";

    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    const GraphHostDefinition &entry = definitions.entries[0];
    ASSERT_NE(entry.spill, nullptr);
    EXPECT_EQ(entry.object_offset, GRAPH_NO_OBJECT_OFFSET);
    EXPECT_EQ(graph_host_arena_used(*graph_state), 0u) << "nothing was claimed, so the upload ships no prefix";

    const GraphDefinition *image = definition_image(entry);
    EXPECT_EQ(image->full_key, entry.full_key);
    EXPECT_EQ(image->total_bytes, entry.bytes);
    EXPECT_GT(image->task_count, 0u) << "the spill holds a filled image, not just sized bytes";

    const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, 0);
    ASSERT_TRUE(upload.has_value());
    EXPECT_EQ(upload->full_key, image->full_key);
}

TEST_F(HbgGraphDefinitionArenaTest, AnArenaTooSmallForAnObjectSpillsIt) {
    // Big enough to be a real region and far too small for any image: a Definition
    // carries its sections past the header, so no recording fits in this.
    bind_arena(sizeof(GraphDefinition));

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    record_graph(0x1715, 1, boundary, shape);
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());

    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    EXPECT_NE(definitions.entries[0].spill, nullptr);
    EXPECT_EQ(definitions.entries[0].object_offset, GRAPH_NO_OBJECT_OFFSET);
    EXPECT_EQ(graph_host_arena_used(*graph_state), 0u)
        << "a failed claim must leave the cursor where it was, so a later object still fits";
    EXPECT_EQ(definition_image(definitions.entries[0])->total_bytes, definitions.entries[0].bytes);
}
