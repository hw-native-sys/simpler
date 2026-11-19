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

#include "host/host_dependency_capture_scope.h"
#include <stdexcept>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <unistd.h>

#include "host_build_graph/host/dep_gen_host_graph.h"

namespace {

std::filesystem::path output_path(const char *name) {
    return std::filesystem::temp_directory_path() /
           (std::string("simpler_dep_gen_") + name + "_" + std::to_string(::getpid()) + ".json");
}

void capture_task(TaskId task_id, TaskId predecessor = TaskId::invalid()) {
    const int32_t kernel_ids[3] = {1, -1, -1};
    dep_gen_host_graph_begin_task(task_id, false, false, kernel_ids, 1, 0, nullptr, nullptr);
    if (predecessor.is_valid()) dep_gen_host_graph_add_explicit_edge(predecessor);
    dep_gen_host_graph_end_task();
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

}  // namespace

TEST(DepGenHostGraphTest, ScopeConfigurationIsIndependentOfThreadHistory) {
    using simpler::dfx::host_graph::HostGraphExport;
    using simpler::dfx::host_graph::TakeOutcome;
    std::thread caller([] {
        for (bool enabled : {true, false, true}) {
            HostGraphExport graph;
            HostDependencyCaptureScope capture(enabled);
            EXPECT_EQ(dep_gen_host_graph_enabled(), enabled);
            EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::NotCaptured));
            dep_gen_host_graph_begin_capture();
            capture_task(TaskId::make_global(601));
            EXPECT_EQ(
                dep_gen_host_graph_take(&graph),
                static_cast<int>(enabled ? TakeOutcome::Complete : TakeOutcome::NotCaptured)
            );
            EXPECT_EQ(graph.tasks.size(), enabled ? 1u : 0u);
        }
    });
    caller.join();
}

TEST(DepGenHostGraphTest, EmitWritesTheGraphCapturedOnTheSameThread) {
    HostDependencyCaptureScope capture(true);
    const std::filesystem::path path = output_path("same_thread");
    std::filesystem::remove(path);
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
    HostDependencyCaptureScope capture(true);
    const std::filesystem::path path = output_path("cross_thread");
    std::filesystem::remove(path);
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
    HostDependencyCaptureScope capture(true);
    const std::filesystem::path path = output_path("reset");
    std::filesystem::remove(path);
    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(101));

    dep_gen_host_graph_begin_capture();
    capture_task(TaskId::make_global(202));
    ASSERT_EQ(dep_gen_host_graph_emit(path.c_str()), 0);

    const std::string json = read_file(path);
    EXPECT_EQ(json.find("\"task_id\":\"101\""), std::string::npos);
    EXPECT_NE(json.find("\"task_id\":\"202\""), std::string::npos);
    std::filesystem::remove(path);
}

TEST(DepGenHostGraphTest, EmptyAndIncompleteCapturesRemainDistinctFromNoSession) {
    using simpler::dfx::host_graph::HostGraphExport;
    using simpler::dfx::host_graph::TakeOutcome;
    HostGraphExport graph;
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::NotCaptured));
    {
        HostDependencyCaptureScope capture(true);
        dep_gen_host_graph_begin_capture();
        EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::Complete));
        EXPECT_TRUE(graph.tasks.empty());
        EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::NotCaptured));
        dep_gen_host_graph_begin_capture();
        dep_gen_host_graph_begin_task(TaskId::make_global(301), false, false, nullptr, 1, 0, nullptr, nullptr);
        EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::Incomplete));
    }
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::NotCaptured));
}

TEST(DepGenHostGraphTest, NestedCaptureIsRejectedAndExceptionDiscardsOwnedState) {
    using simpler::dfx::host_graph::HostGraphExport;
    using simpler::dfx::host_graph::TakeOutcome;
    HostGraphExport graph;
    try {
        HostDependencyCaptureScope capture(true);
        dep_gen_host_graph_begin_capture();
        capture_task(TaskId::make_global(401));
        EXPECT_THROW(HostDependencyCaptureScope nested(true), std::logic_error);
        throw 1;
    } catch (int) {}
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::NotCaptured));
    HostDependencyCaptureScope next(true);
    dep_gen_host_graph_begin_capture();
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(TakeOutcome::Complete));
    EXPECT_TRUE(graph.tasks.empty());
}

TEST(DepGenHostGraphTest, ParkedCaptureCanMoveToANewSessionOnAnotherThread) {
    using simpler::dfx::host_graph::HostGraphExport;
    using simpler::dfx::host_graph::TakeOutcome;
    HostGraphExport saved;
    int outcome;
    {
        HostDependencyCaptureScope capture(true);
        dep_gen_host_graph_begin_capture();
        capture_task(TaskId::make_global(501));
        outcome = dep_gen_host_graph_take(&saved);
    }
    HostGraphExport restored;
    std::thread other([&] {
        HostDependencyCaptureScope capture(true);
        dep_gen_host_graph_restore_for_export(&saved, outcome);
        EXPECT_EQ(dep_gen_host_graph_take(&restored), static_cast<int>(TakeOutcome::Complete));
    });
    other.join();
    ASSERT_EQ(restored.tasks.size(), 1u);
    EXPECT_EQ(restored.tasks[0].task_id, 501u);
    EXPECT_TRUE(saved.tasks.empty());
}
