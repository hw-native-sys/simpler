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

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <unistd.h>

#include "host_build_graph/dep_gen_host_graph.h"

namespace {

std::filesystem::path output_path(const char *name) {
    return std::filesystem::temp_directory_path() /
           (std::string("simpler_dep_gen_") + name + "_" + std::to_string(::getpid()) + ".json");
}

void capture_task(uint64_t task_id, uint64_t predecessor = 0) {
    const int32_t kernel_ids[3] = {1, -1, -1};
    dep_gen_host_graph_begin_task(task_id, false, false, kernel_ids, 1, 0, nullptr, nullptr);
    if (predecessor != 0) dep_gen_host_graph_add_explicit_edge(predecessor);
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
    capture_task(11);
    capture_task(12, 11);

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
    capture_task(21);

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
    capture_task(101);

    dep_gen_host_graph_set_enabled(true);
    dep_gen_host_graph_begin_capture();
    capture_task(202);
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    const std::string json = read_file(path);
    EXPECT_EQ(json.find("\"task_id\":\"101\""), std::string::npos);
    EXPECT_NE(json.find("\"task_id\":\"202\""), std::string::npos);
    std::filesystem::remove(path);
}
