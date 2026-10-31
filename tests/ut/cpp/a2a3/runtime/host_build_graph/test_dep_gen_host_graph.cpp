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

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <unistd.h>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host/dep_gen_host_graph.h"

namespace {

std::filesystem::path output_path(const char *name) {
    return std::filesystem::temp_directory_path() /
           (std::string("simpler_dep_gen_") + name + "_" + std::to_string(::getpid()) + ".json");
}

void capture_task(
    TaskId task_id, TaskId predecessor = TaskId::invalid(), bool sync_start = false, bool predicated = false
) {
    const int32_t kernel_ids[3] = {1, -1, -1};
    dep_gen_host_graph_begin_task(task_id, false, false, sync_start, predicated, kernel_ids, 1, 0, nullptr, nullptr);
    if (predecessor.is_valid()) dep_gen_host_graph_add_explicit_edge(predecessor);
    dep_gen_host_graph_end_task();
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(DepGenHostGraphTest, EmitWritesTheGraphCapturedOnTheSameThread) {
    const std::filesystem::path path = output_path("same_thread");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(11));
    capture_task(TaskId::make_global(12), TaskId::make_global(11));

    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);
    const std::string json = read_file(path);
    EXPECT_NE(json.find("\"task_id\":\"11\""), std::string::npos);
    EXPECT_NE(json.find("\"task_id\":\"12\""), std::string::npos);
    EXPECT_NE(json.find("\"pred\":\"11\",\"succ\":\"12\""), std::string::npos);
    std::filesystem::remove(path);
}

TEST(DepGenHostGraphTest, EmitOnAnotherThreadWritesNothingAndReports) {
    const std::filesystem::path path = output_path("cross_thread");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(21));

    // The graph is thread-local: a run whose orchestration and drain land on
    // different threads emits nothing rather than a partial deps.json.
    int emit_rc = 0;
    std::thread other([&]() {
        emit_rc = dep_gen_host_graph_emit(path.c_str());
    });
    other.join();

    EXPECT_EQ(emit_rc, -3);
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(DepGenHostGraphTest, BeginCaptureClearsThePreviousRunsGraph) {
    const std::filesystem::path path = output_path("reset");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(101));

    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(202));
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    const std::string json = read_file(path);
    EXPECT_EQ(json.find("\"task_id\":\"101\""), std::string::npos);
    EXPECT_NE(json.find("\"task_id\":\"202\""), std::string::npos);
    std::filesystem::remove(path);
}

TEST(DepGenHostGraphTest, EachTaskCarriesItsLaunchAttributes) {
    const std::filesystem::path path = output_path("attrs");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(41));
    capture_task(TaskId::make_global(42), TaskId::invalid(), /*sync_start=*/true, /*predicated=*/true);
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    const std::string json = read_file(path);
    EXPECT_NE(json.find("\"task_id\":\"41\""), std::string::npos);
    EXPECT_NE(json.find("\"block_num\":1,\"sync_start\":false,\"predicated\":false"), std::string::npos) << json;
    EXPECT_NE(json.find("\"block_num\":1,\"sync_start\":true,\"predicated\":true"), std::string::npos) << json;
    // A run that replayed no Graph writes no Graph sections.
    EXPECT_EQ(json.find("graph_definitions"), std::string::npos);
    EXPECT_EQ(json.find("graph_shells"), std::string::npos);
    std::filesystem::remove(path);
}

namespace {

// A three-task body: AIC root 0, MIX 1 with two blocks, and AIV 2 that is
// sync-start and predicated, with edges 0->1, 0->2 and 1->2. Laid out by hand
// the way graph_definition_array reads it: each section at an aligned offset
// from the Definition's own address.
struct HandBuiltDefinition {
    alignas(64) std::array<uint8_t, 16384> bytes{};

    template <typename T>
    static uint32_t place(uint32_t *cursor, int32_t count) {
        const uint32_t at = (*cursor + alignof(T) - 1) / alignof(T) * alignof(T);
        *cursor = at + static_cast<uint32_t>(sizeof(T) * static_cast<size_t>(count));
        return at;
    }

    explicit HandBuiltDefinition(uint64_t graph_key) {
        auto *definition = reinterpret_cast<GraphDefinition *>(bytes.data());
        uint32_t cursor = sizeof(GraphDefinition);
        definition->graph_key = graph_key;
        definition->task_count = 3;
        definition->edge_count = 3;
        definition->off_sub_tasks = place<SubTaskDefinition>(&cursor, 3);
        definition->off_fanin_offsets = place<int32_t>(&cursor, 4);
        definition->off_fanin_indices = place<uint16_t>(&cursor, 3);
        definition->total_bytes = cursor;

        auto *subs = reinterpret_cast<SubTaskDefinition *>(bytes.data() + definition->off_sub_tasks);
        const int32_t kernels[3][3] = {{5, -1, -1}, {6, 7, 8}, {-1, 9, -1}};
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k)
                subs[i].kernel_id[k] = kernels[i][k];
            subs[i].logical_block_num = 1;
        }
        subs[1].logical_block_num = 2;
        TaskAttrs attrs;
        attrs.set_sync_start();
        attrs.set_predicate();
        subs[2].task_attrs = attrs.raw();

        const int32_t fanin_offsets[4] = {0, 0, 1, 3};
        const uint16_t fanin_indices[3] = {0, 0, 1};
        std::memcpy(bytes.data() + definition->off_fanin_offsets, fanin_offsets, sizeof(fanin_offsets));
        std::memcpy(bytes.data() + definition->off_fanin_indices, fanin_indices, sizeof(fanin_indices));
    }

    const GraphDefinition &definition() const { return *reinterpret_cast<const GraphDefinition *>(bytes.data()); }
};

}  // namespace

// Graph tasks replaying one Definition — one graph_key, whatever object holds it —
// serialize it once, and each names it.
TEST(DepGenHostGraphTest, GraphTasksOfOneKeyShareOneSerializedBody) {
    const std::filesystem::path path = output_path("graph_body");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(31));
    capture_task(TaskId::make_global(32), TaskId::make_global(31));
    capture_task(TaskId::make_global(33), TaskId::make_global(32));
    const HandBuiltDefinition body(7);
    const HandBuiltDefinition same_key(7);
    const HandBuiltDefinition other_key(8);
    dep_gen_host_graph_add_graph_body(TaskId::make_global(31), body.definition());
    dep_gen_host_graph_add_graph_body(TaskId::make_global(32), same_key.definition());
    dep_gen_host_graph_add_graph_body(TaskId::make_global(33), other_key.definition());
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    const std::string json = read_file(path);
    const std::string tasks =
        "\"tasks\":["
        "{\"index\":0,\"kernel_ids\":[5,-1,-1],\"block_num\":1,\"sync_start\":false,\"predicated\":false},"
        "{\"index\":1,\"kernel_ids\":[6,7,8],\"block_num\":2,\"sync_start\":false,\"predicated\":false},"
        "{\"index\":2,\"kernel_ids\":[-1,9,-1],\"block_num\":1,\"sync_start\":true,\"predicated\":true}],"
        "\"edges\":[[0,1],[0,2],[1,2]]}";
    EXPECT_NE(
        json.find("\"graph_definitions\":[{\"definition\":0," + tasks + ",{\"definition\":1," + tasks + "]"),
        std::string::npos
    ) << json;
    EXPECT_NE(
        json.find(
            "\"graph_shells\":[{\"task_id\":\"31\",\"definition\":0},{\"task_id\":\"32\",\"definition\":0},"
            "{\"task_id\":\"33\",\"definition\":1}]}"
        ),
        std::string::npos
    ) << json;
    std::filesystem::remove(path);
}

TEST(DepGenHostGraphTest, NothingIsRecordedWhileCaptureIsOff) {
    const std::filesystem::path path = output_path("off");
    std::filesystem::remove(path);
    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(51));
    dep_gen_host_graph_set_enabled(false);
    const HandBuiltDefinition body(9);
    dep_gen_host_graph_add_graph_body(TaskId::make_global(51), body.definition());
    dep_gen_host_graph_set_enabled(true);
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    EXPECT_EQ(read_file(path).find("graph_shells"), std::string::npos);
    std::filesystem::remove(path);
}
