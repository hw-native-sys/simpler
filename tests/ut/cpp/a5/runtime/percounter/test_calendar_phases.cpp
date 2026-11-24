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
#include <vector>

#include "host/calendar_image.h"
#include "host/graph_template.h"

namespace simpler::percounter {
namespace {

std::vector<TaskInput> cube_tasks(const std::vector<std::vector<int32_t>> &fanins) {
    std::vector<TaskInput> tasks(fanins.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        tasks[i].active_mask = 1;
        tasks[i].fanin_count = static_cast<int32_t>(fanins[i].size());
        tasks[i].fanin_ids = fanins[i].data();
    }
    return tasks;
}

TEST(CalendarPhases, SchedulesGraphAncestorsAndKeepsUnrelatedWorkInTail) {
    HostGraphTemplate body;
    const BodyTaskShape shapes[] = {{1, false, false, 1, 1}};
    const int32_t body_offsets[] = {0, 0};
    const int32_t empty_fanin[] = {0};
    ASSERT_EQ(build_graph_template(shapes, 1, body_offsets, empty_fanin, 0, 6, &body).status, TemplateStatus::OK);

    // 0 is already complete, 1 is independent, 2 -> 3 -> Graph 4.
    // 4 -> 5 -> Graph 6, and Graph 6 -> 7 -> 8. Task 8 also waits for 1.
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0}, {2}, {3}, {4}, {5}, {6}, {1, 7}};
    auto outer = cube_tasks(fanins);
    outer[0].inline_completed = true;
    // Invocations need not be stored in outer table order.
    const GraphInvocationInput invocations[] = {{6, 0}, {4, 0}};
    outer[4].graph_invocation = 1;
    outer[6].graph_invocation = 0;
    const GraphPlanInput graph{&body, 1, invocations, 2};

    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(outer, graph, &phases));
    ASSERT_EQ(phases.size(), 5U);

    EXPECT_EQ(phases[0].invocation, -1);
    EXPECT_EQ(phases[0].task_ids, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(phases[0].tasks[0].fanin_count, 0);
    ASSERT_EQ(phases[0].tasks[1].fanin_count, 1);
    EXPECT_EQ(phases[0].tasks[1].fanin_ids[0], 0);
    EXPECT_EQ(phases[1].invocation, 1);
    EXPECT_EQ(phases[1].task_ids, (std::vector<uint32_t>{0}));
    EXPECT_EQ(phases[2].invocation, -1);
    EXPECT_EQ(phases[2].task_ids, (std::vector<uint32_t>{5}));
    EXPECT_EQ(phases[2].tasks[0].fanin_count, 0);
    EXPECT_EQ(phases[3].invocation, 0);
    EXPECT_EQ(phases[3].task_ids, (std::vector<uint32_t>{0}));
    EXPECT_EQ(phases[4].invocation, -1);
    EXPECT_EQ(phases[4].task_ids, (std::vector<uint32_t>{1, 7, 8}));
    EXPECT_EQ(phases[4].tasks[1].fanin_count, 0);
    ASSERT_EQ(phases[4].tasks[2].fanin_count, 2);
    EXPECT_EQ(phases[4].tasks[2].fanin_ids[0], 0);
    EXPECT_EQ(phases[4].tasks[2].fanin_ids[1], 1);
}

TEST(CalendarPhases, SummarizesFaninAcrossDistinctWaitingLanes) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {}, {}, {0, 1, 2, 3}};
    auto tasks = cube_tasks(fanins);
    tasks.back().block_num = 3;
    const std::vector<CalendarRow> rows{
        {percounter_order_entry(0, 0, 1), CALENDAR_NO_JOIN, 0}, {percounter_order_entry(1, 0, 1), CALENDAR_NO_JOIN},
        {percounter_order_entry(4, 0, 3), CALENDAR_NO_JOIN, 0}, {percounter_order_entry(2, 0, 1), CALENDAR_NO_JOIN},
        {percounter_order_entry(3, 0, 1), CALENDAR_NO_JOIN, 0}, {percounter_order_entry(4, 1, 3), CALENDAR_NO_JOIN},
        {percounter_order_entry(4, 2, 3), CALENDAR_NO_JOIN, 0},
    };
    range_calendar_detail::CalendarWaitPlan plan;
    ASSERT_TRUE(range_calendar_detail::plan_calendar_waits(tasks, rows, {0, 3, 6, 7}, 0, plan));
    ASSERT_TRUE(plan.summarized_fanins.back());
    EXPECT_EQ(plan.row_assignments[2].begin, 0);
    EXPECT_EQ(plan.row_assignments[2].count, 1);
    EXPECT_EQ(plan.row_assignments[5].begin, 1);
    EXPECT_EQ(plan.row_assignments[5].count, 1);
    EXPECT_EQ(plan.row_assignments[6].begin, 2);
    EXPECT_EQ(plan.row_assignments[6].count, 2);
}

TEST(CalendarPhases, KeepsGraphBodyShapesAndOwnsDependenciesAfterMove) {
    HostGraphTemplate body;
    const BodyTaskShape shapes[] = {{7, true, false, 2, 6}, {0, false, false, 1, 0}, {2, false, false, 1, 1}};
    const int32_t offsets[] = {0, 0, 1, 2};
    const int32_t indices[] = {0, 1};
    ASSERT_EQ(build_graph_template(shapes, 3, offsets, indices, 2, 6, &body).status, TemplateStatus::OK);

    std::vector<TaskInput> outer(1);
    outer[0].graph_invocation = 0;
    const GraphInvocationInput invocation{0, 0};
    const GraphPlanInput graph{&body, 1, &invocation, 1};
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(outer, graph, &phases));
    ASSERT_EQ(phases.size(), 1U);
    body = HostGraphTemplate{};
    // Force relocation of the phase records after the source template dies.
    phases.reserve(phases.capacity() + 1);

    const auto &phase = phases.front();
    EXPECT_EQ(phase.task_ids, (std::vector<uint32_t>{0, 1, 2}));
    ASSERT_EQ(phase.tasks.size(), 3U);
    EXPECT_TRUE(phase.tasks[0].is_mix);
    EXPECT_TRUE(phase.tasks[0].sync_start);
    EXPECT_EQ(phase.tasks[0].active_mask, 7U);
    EXPECT_EQ(phase.tasks[0].block_num, 2);
    EXPECT_EQ(phase.tasks[0].subtask_count, 3);
    EXPECT_EQ(phase.tasks[1].active_mask, 0U);
    EXPECT_EQ(phase.tasks[1].core_type_index, 1U);
    EXPECT_EQ(phase.tasks[2].active_mask, 2U);
    ASSERT_EQ(phase.tasks[1].fanin_count, 1);
    EXPECT_EQ(phase.tasks[1].fanin_ids[0], 0);
    ASSERT_EQ(phase.tasks[2].fanin_count, 1);
    EXPECT_EQ(phase.tasks[2].fanin_ids[0], 1);
}

TEST(CalendarPhases, ReportsInvalidWorkerTopology) {
    HostGraphTemplate body;
    const BodyTaskShape shape{1, false, false, 1, 1};
    const int32_t offsets[] = {0, 0};
    const auto result = build_graph_template(&shape, 1, offsets, nullptr, 0, 2, &body);
    EXPECT_EQ(result.status, TemplateStatus::MALFORMED);
    EXPECT_STREQ(result.reason, "invalid worker topology");
}

TEST(CalendarPhases, RejectsInconsistentGraphsAndPreservesThePreviousResult) {
    HostGraphTemplate body;
    const BodyTaskShape shapes[] = {{1, false, false, 1, 1}};
    const int32_t offsets[] = {0, 0};
    const int32_t empty_fanin[] = {0};
    ASSERT_EQ(build_graph_template(shapes, 1, offsets, empty_fanin, 0, 6, &body).status, TemplateStatus::OK);

    std::vector<CalendarPhase> phases;
    const std::vector<std::vector<int32_t>> no_fanins{{}, {}};
    auto outer = cube_tasks(no_fanins);
    outer[0].inline_completed = true;
    const GraphInvocationInput invocation{1, 0};
    const GraphPlanInput graph{&body, 1, &invocation, 1};
    outer[1].graph_invocation = 0;
    ASSERT_TRUE(plan_calendar_phases(outer, graph, &phases));

    const int32_t forward_dependency = 1;
    outer[0].fanin_count = 1;
    outer[0].fanin_ids = &forward_dependency;
    EXPECT_FALSE(plan_calendar_phases(outer, graph, &phases));

    outer[0].fanin_count = 0;
    body.fanin_offsets = {0, -1};
    EXPECT_FALSE(plan_calendar_phases(outer, graph, &phases));

    ASSERT_EQ(phases.size(), 1U);
    EXPECT_EQ(phases[0].invocation, 0);
    EXPECT_EQ(phases[0].task_ids, (std::vector<uint32_t>{0}));
}

TEST(CalendarPhases, SharesGraphRowsAndRelocatesBindingsPerInvocation) {
    constexpr uint32_t count = 17;
    HostGraphTemplate body;
    std::vector<BodyTaskShape> shapes(count, {1, false, false, 1, 1});
    shapes[0] = {3, false, false, 1, 2};
    std::vector<int32_t> offsets(count + 1, 0);
    std::vector<int32_t> fanin;
    for (uint32_t task = 1; task < count; ++task) {
        fanin.push_back(task - 1);
        offsets[task + 1] = static_cast<int32_t>(fanin.size());
    }
    ASSERT_EQ(
        build_graph_template(shapes.data(), count, offsets.data(), fanin.data(), fanin.size(), 3, &body).status,
        TemplateStatus::OK
    );

    const std::vector<std::vector<int32_t>> outer_fanins{{}, {}, {0, 1}, {2}, {3}};
    auto outer = cube_tasks(outer_fanins);
    const GraphInvocationInput invocations[]{{2, 0}, {3, 0}};
    outer[2].graph_invocation = 0;
    outer[3].graph_invocation = 1;
    const GraphPlanInput graphs{&body, 1, invocations, 2};
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(outer, graphs, &phases));
    ASSERT_EQ(phases.size(), 4U);

    std::vector<RangeCalendarInput> inputs;
    for (size_t i = 0; i < phases.size(); ++i) {
        if (i == 2) {
            inputs.push_back({&phases[i], inputs[1].plan, true});
        } else {
            auto plan = std::make_shared<CalendarPlan>();
            ASSERT_TRUE(build_static_calendar(phases[i].tasks, 1, {}, plan.get()));
            inputs.push_back({&phases[i], std::move(plan), false});
        }
    }
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar(inputs, graphs, outer.size(), 1, &image));
    const auto header = image.load<CalendarRangeHeader>(0);
    ASSERT_EQ(header.worker_count, 3U);
    EXPECT_EQ(image.phase_slot_count, 1U);
    EXPECT_GE(header.notify_slots_count, image.phase_slot_count);
    EXPECT_EQ(image.phase_ends[0].contributors, 1U);
    EXPECT_EQ(image.phase_ends[1].task, 2U);
    EXPECT_EQ(image.phase_ends[2].task, 3U);
    EXPECT_EQ(image.sample_offsets[1], 0U);
    EXPECT_NE(image.sample_offsets[2], 0U);

    const auto patch = [&](uint32_t task) {
        return image.load<CalendarDispatchPatch>(header.patch_offset + task * sizeof(CalendarDispatchPatch));
    };
    const uint32_t first_graph = static_cast<uint32_t>(outer.size());
    const uint32_t second_graph = first_graph + count;
    EXPECT_NE(patch(first_graph).dispatch_offsets[0], patch(second_graph).dispatch_offsets[0]);
    // Two 32-byte records from different materialization batches must never
    // share a cache line, even when this batch contains an odd number of roles.
    EXPECT_NE(patch(first_graph + 15).dispatch_offsets[0] / 64, patch(first_graph + 16).dispatch_offsets[0] / 64);
    const uint32_t cube_ranges = image.load<uint32_t>(header.worker_index_offset + sizeof(uint32_t));
    ASSERT_EQ(cube_ranges, 6U);
    const auto range = [&](uint32_t i) {
        return image.load<CalendarRange>(header.range_offset + i * sizeof(CalendarRange));
    };
    EXPECT_EQ(range(1).rows_offset, range(3).rows_offset);
    EXPECT_EQ(range(2).rows_offset, range(4).rows_offset);
    EXPECT_NE(range(1).binding_offset, range(3).binding_offset);
    EXPECT_LT(range(1).sequence_base, range(3).sequence_base);
    EXPECT_EQ(range(1).entry_wait_count, 2U);
    EXPECT_EQ(range(2).entry_wait_count, 1U);
    EXPECT_EQ(range(2).exit_wait_count, 1U);
    EXPECT_EQ(range(4).exit_wait_count, 1U);
    EXPECT_NE(range(2).exit_wait_offset, range(4).exit_wait_offset);
}

}  // namespace
}  // namespace simpler::percounter
