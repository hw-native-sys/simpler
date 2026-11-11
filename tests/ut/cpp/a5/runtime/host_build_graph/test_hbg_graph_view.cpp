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

#include <cstdint>

#include "hbg_scheduler_test_support.h"

namespace {

SchedulerGraphView current_view(const scheduler_test::BasicGraphBuffer<8> &buffer) {
    auto view = buffer.graph();
    view.original_task_count = static_cast<uint32_t>(view.task_count);
    view.view_version = SCHEDULER_GRAPH_VIEW_VERSION;
    view.view_bytes = sizeof(SchedulerGraphView);
    return view;
}

TEST(AicoreGraphView, OrdinaryNodesKeepTheirOriginalDependencyReferences) {
    scheduler_test::BasicGraphBuffer<8> buffer(5);
    buffer.executable(0, 0);
    buffer.executable(1, 1);
    buffer.executable(2, 0, {0, 1});
    buffer.executable(3, 1, {2});
    buffer.executable(4, 0, {0, 2, 3});
    auto graph = current_view(buffer);
    ASSERT_TRUE(scheduler_graph_view_valid(graph));
    EXPECT_EQ(graph.native_header_offset, 0u);
    uint32_t count = UINT32_MAX;
    ASSERT_EQ(scheduler_node_fanin_count(graph, 4, &count), SchedulerGraphResult::OK);
    EXPECT_EQ(count, 3u);
    SchedulerFaninView fanin{};
    ASSERT_EQ(scheduler_node_fanin_view(graph, 4, &fanin), SchedulerGraphResult::OK);
    ASSERT_EQ(fanin.count, count);
    for (uint32_t index = 0; index < count; ++index) {
        int64_t producer = -1;
        ASSERT_EQ(scheduler_node_fanin_ref(graph, 4, index, &producer), SchedulerGraphResult::OK);
        EXPECT_EQ(producer, buffer.payload(4).fanin_data()[index]);
        ASSERT_EQ(scheduler_fanin_ref(fanin, 4, index, &producer), SchedulerGraphResult::OK);
        EXPECT_EQ(producer, buffer.payload(4).fanin_data()[index]);
    }
    int64_t producer = -1;
    EXPECT_EQ(scheduler_fanin_ref(fanin, 4, count, &producer), SchedulerGraphResult::INVALID_FANIN_ID);
    ASSERT_EQ(scheduler_node_fanin_count(graph, 0, &count), SchedulerGraphResult::OK);
    EXPECT_EQ(count, 0u);
}

TEST(AicoreGraphView, RejectsAnUnversionedOrMalformedViewBeforeReadingStorage) {
    scheduler_test::BasicGraphBuffer<8> buffer(2);
    auto graph = current_view(buffer);
    ASSERT_TRUE(scheduler_graph_view_valid(graph));
    graph.view_version = 0;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    uint32_t count = 0;
    EXPECT_EQ(scheduler_node_fanin_count(graph, 0, &count), SchedulerGraphResult::INVALID_ARGUMENTS);
    graph.view_version = SCHEDULER_GRAPH_VIEW_VERSION;
    graph.view_bytes = 32;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    graph.view_bytes = sizeof(SchedulerGraphView);
    graph.original_task_count = 1;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    graph.original_task_count = 2;
    graph.storage_address = 0;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
}

TEST(AicoreGraphView, RejectsReferencesAndRowsOutsideTheOrdinaryGraph) {
    scheduler_test::BasicGraphBuffer<8> buffer(2);
    buffer.executable(0, 0);
    buffer.executable(1, 1, {0});
    const auto graph = current_view(buffer);
    uint32_t count = 0;
    int64_t producer = -1;
    EXPECT_EQ(scheduler_node_fanin_count(graph, -1, &count), SchedulerGraphResult::INVALID_TASK_COUNT);
    EXPECT_EQ(scheduler_node_fanin_count(graph, 2, &count), SchedulerGraphResult::INVALID_TASK_COUNT);
    EXPECT_EQ(scheduler_node_fanin_count(graph, 0, nullptr), SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(scheduler_node_fanin_ref(graph, 1, 0, nullptr), SchedulerGraphResult::INVALID_ARGUMENTS);
    EXPECT_EQ(scheduler_node_fanin_ref(graph, 1, 1, &producer), SchedulerGraphResult::INVALID_FANIN_ID);
    EXPECT_EQ(scheduler_node_fanin_ref(graph, 1, UINT32_MAX, &producer), SchedulerGraphResult::INVALID_FANIN_ID);
    buffer.payload(1).fanin_count = -1;
    EXPECT_EQ(scheduler_node_fanin_count(graph, 1, &count), SchedulerGraphResult::INVALID_FANIN_COUNT);
    buffer.payload(1).fanin_count = 129;
    EXPECT_EQ(scheduler_node_fanin_count(graph, 1, &count), SchedulerGraphResult::INVALID_FANIN_COUNT);
    buffer.payload(1).fanin_count = 1;
    buffer.payload(1).fanin_data()[0] = 1;
    EXPECT_EQ(scheduler_node_fanin_ref(graph, 1, 0, &producer), SchedulerGraphResult::INVALID_FANIN_ID);
    buffer.payload(1).fanin_data()[0] = -1;
    EXPECT_EQ(scheduler_node_fanin_ref(graph, 1, 0, &producer), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(AicoreGraphView, HeaderOffsetCannotAuthorizeItsOwnRead) {
    scheduler_test::BasicGraphBuffer<8> buffer(2);
    auto graph = current_view(buffer);
    graph.scheduler_state_address = 1;
    graph.scheduler_state_bytes = 128;
    graph.native_header_offset = UINT64_MAX - 63;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    graph.native_header_offset = 128;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
    graph.scheduler_state_address = UINT64_MAX - 63;
    graph.native_header_offset = 64;
    EXPECT_FALSE(scheduler_graph_view_valid(graph));
}

}  // namespace
