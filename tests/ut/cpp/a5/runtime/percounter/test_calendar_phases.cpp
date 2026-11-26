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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "aicpu/peer_fanout.h"
#include "host/calendar_cache.h"
#include "host/calendar_event_score.h"
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

TEST(CalendarPhases, PeerDiamondUsesCubeCompletionForFanoutAndVectorNotificationsForJoin) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}, {0}, {1, 2}};
    auto tasks = cube_tasks(fanins);
    for (auto &task : tasks)
        task.known_synchronous = true;
    for (const int id : {1, 2}) {
        tasks[id].core_type_index = 1;
        tasks[id].active_mask = 2;
    }
    constexpr uint32_t clusters = 4;
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, clusters, {}, plan.get()));
    const std::vector<RangeCalendarInput> inputs{{&phases[0], plan, false}};
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar(inputs, {}, tasks.size(), clusters, &image));
    ASSERT_EQ(image.peer_single_waits, 0U);
    ASSERT_EQ(image.peer_counter_waits, 2U);
    ASSERT_EQ(image.peer_double_waits, 1U);
    ASSERT_NE(image.peer_endpoints_offset, 0U);
    ASSERT_EQ(image.fused_fanins.size(), 1U);
    struct Tables {
        PercounterControl control{};
        std::array<PercounterCounter, 4> counters{};
    } tables;
    PercounterLayout layout{};
    layout.counters_offset = offsetof(Tables, counters);
    layout.task_count = 4;
    std::vector<uint64_t> slots(image.load<CalendarRangeHeader>(0).notify_slots_count);
    for (size_t i = 0; i < slots.size(); ++i)
        slots[i] = 0x1000 + i * 32;
    ASSERT_TRUE(bind_range_calendar(image, 0, layout, &tables, slots));
    auto extra = [&](size_t id) {
        return image.load<CalendarPeerExtra>(image.extras[id].offset);
    };
    const auto join = extra(3);
    ASSERT_EQ(join.input_count, 2);
    ASSERT_EQ(join.join_target, 2U);
    ASSERT_NE(join.join_read, 0U);
    for (size_t id : {1U, 2U}) {
        EXPECT_EQ(tables.counters[id].completion_notify_address, join.join_read);
        const auto producer = extra(id);
        ASSERT_EQ(producer.send_count, 1);
        EXPECT_EQ(producer.input_count, 0);
        const auto send = image.load<CalendarPeerSend>(producer.sends_offset);
        EXPECT_EQ(send.ordinal, id - 1);
        EXPECT_EQ(send.join_add, 0U);  // The standard DONE already contributes.
    }

    // B and C observe A's original DONE threshold; the sink shares that counter.
    EXPECT_EQ(tables.counters[0].completion_notify_address, tables.counters[3].completion_notify_address);
    EXPECT_EQ(tables.counters[0].completion_notify_target, 1U);
    EXPECT_EQ(tables.counters[3].completion_notify_target, 2U);

    // Rebuilding the same placement with an undeclared branch must drop its shared join.
    phases[0].tasks[1].known_synchronous = false;
    ASSERT_TRUE(build_range_calendar(inputs, {}, tasks.size(), clusters, &image));
    EXPECT_EQ(image.peer_single_waits, 0U);
    EXPECT_EQ(image.peer_counter_waits, 1U);
    EXPECT_EQ(image.peer_double_waits, 0U);
    EXPECT_EQ(image.peer_endpoints_offset, 0U);
    EXPECT_TRUE(image.fused_fanins.empty());
}

TEST(CalendarPhases, TwoCubePublishersKeepIndividualCompletionWaits) {
    constexpr uint32_t clusters = 4;
    for (bool other_consumer : {false, true}) {
        SCOPED_TRACE(other_consumer);
        std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}};
        if (other_consumer) fanins.push_back({0});
        auto tasks = cube_tasks(fanins);
        for (auto &task : tasks)
            task.known_synchronous = true;
        std::vector<CalendarPhase> phases;
        ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
        // Each task occupies a different cube, so the join must wait for both producers.
        calendar_detail::PlacementState placement;
        placement.workers.resize(clusters * PLATFORM_CORES_PER_BLOCKDIM);
        for (uint32_t i = 0; i < tasks.size(); ++i)
            placement.workers[i].push_back({{percounter_order_entry(i, 0, 1), CALENDAR_NO_JOIN, 0}, i});
        auto plan = std::make_shared<CalendarPlan>();
        ASSERT_TRUE(calendar_detail::pack(placement, phases[0].tasks, tasks.size(), 0, plan.get()));
        ASSERT_EQ(plan->waits.task_fanins[2], fanins[2]);

        RangeCalendarImage image;
        ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, {}, tasks.size(), clusters, &image));
        const auto header = image.load<CalendarRangeHeader>(0);
        EXPECT_EQ(header.notify_slots_count, 2U);
        EXPECT_EQ(header.peer_endpoints_offset, 0U);
        EXPECT_EQ(image.fanin_slot_count, 0U);
        EXPECT_TRUE(image.fused_fanins.empty());

        struct Tables {
            PercounterControl control{};
            std::array<PercounterCounter, 4> counters{};
        } tables;
        PercounterLayout layout{};
        layout.task_count = tasks.size();
        layout.counters_offset = offsetof(Tables, counters);
        ASSERT_TRUE(bind_range_calendar(image, 0, layout, &tables, {0x1000, 0x1020}));
        const auto index = image.load<uint32_t>(header.worker_index_offset + 2 * sizeof(uint32_t));
        const auto range = image.load<CalendarRange>(header.range_offset + index * sizeof(CalendarRange));
        ASSERT_EQ(range.row_count, 1U);
        const auto row = image.load<CalendarExecutionRow>(range.rows_offset);
        EXPECT_EQ(row.flags & CALENDAR_EXEC_PEER, 0);
        ASSERT_EQ(row.wait_count, 2U);
        const auto dispatch = image.load<CalendarDispatch>(range.binding_offset + row.dispatch_offset);
        for (uint32_t i = 0; i < 2; ++i) {
            const auto wait =
                image.load<CalendarWait>(range.binding_offset + dispatch.waits_offset + i * sizeof(CalendarWait));
            EXPECT_EQ(wait.read_cmd, (0x1000U + i * 32) | CALENDAR_WAIT_DEFER);
            EXPECT_EQ(wait.target, (uint64_t{i} << 32) | (uint64_t{24} << 24) | 1);
        }
    }
}

TEST(CalendarPhases, SinglePeerConsumersKeepTheSharedPredecessorThresholdWhenTheirOwnDoneIsFused) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}, {1}, {1}, {2, 3}};
    auto tasks = cube_tasks(fanins);
    for (auto &task : tasks)
        task.known_synchronous = true;
    for (const auto id : {1, 2}) {
        tasks[id].active_mask = 2;
        tasks[id].core_type_index = 1;
    }
    constexpr uint32_t clusters = 4;
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, clusters, {}, plan.get()));
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, {}, tasks.size(), clusters, &image));
    ASSERT_EQ(image.peer_single_waits, 2U);
    ASSERT_EQ(image.fused_fanins.size(), 1U);
    struct Tables {
        PercounterControl control{};
        std::array<PercounterCounter, 5> counters{};
    } tables;
    PercounterLayout layout{};
    layout.task_count = tasks.size();
    layout.counters_offset = offsetof(Tables, counters);
    ASSERT_TRUE(bind_range_calendar(image, 0, layout, &tables, {0x1000, 0x1020}));

    const auto header = image.load<CalendarRangeHeader>(0);
    const auto range = image.load<CalendarRange>(header.range_offset);
    for (const uint32_t consumer : {2U, 3U}) {
        ASSERT_EQ(image.dispatches[consumer].global_task, consumer);
        const auto dispatch = image.load<CalendarDispatch>(image.dispatches[consumer].offset);
        const auto wait = image.load<CalendarWait>(range.binding_offset + dispatch.waits_offset);
        EXPECT_EQ(wait.read_cmd, 0x1000U | CALENDAR_WAIT_DEFER);
        EXPECT_EQ(wait.target, (uint64_t{1} << 32) | (uint64_t{25} << 24) | 2);
        EXPECT_EQ(tables.counters[consumer].completion_notify_address, 0x1020U);
    }
}

TEST(CalendarPhases, SpmdNotificationsPreserveIndependentStartAndOrdinaryInputWaits) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}, {1}};
    auto tasks = cube_tasks(fanins);
    constexpr uint32_t clusters = 4;
    for (auto &task : tasks) {
        task.known_synchronous = true;
        task.block_num = clusters;
    }
    for (const uint32_t id : {0U, 2U}) {
        tasks[id].active_mask = 2;
        tasks[id].core_type_index = 1;
    }
    tasks[2].sync_start = true;
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, clusters, {}, plan.get()));
    const std::vector<RangeCalendarInput> inputs{{&phases[0], plan, false}};
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar(inputs, {}, tasks.size(), clusters, &image));
    EXPECT_EQ(image.peer_spmd_edges, 1U);
    EXPECT_EQ(image.peer_counter_waits, 1U);
    const auto sender = image.load<CalendarSpmdPeerExtra>(image.extras[0].offset);
    EXPECT_GT(sender.target_count, 1U);
    EXPECT_NE(sender.fanout_offset, 0U);
    EXPECT_EQ(sender.target_count, image.peer_spmd_wait_rows);

    phases[0].tasks[1].known_synchronous = false;
    ASSERT_TRUE(build_range_calendar(inputs, {}, tasks.size(), clusters, &image));
    EXPECT_EQ(image.peer_spmd_edges, 0U);
    EXPECT_EQ(image.peer_endpoints_offset, 0U);
}

TEST(CalendarPhases, SharedFanoutRebindsAddressesAndHandlesDifferentInitialTokens) {
    RangeCalendarImage image;
    image.bytes.resize(sizeof(CalendarRangeHeader));
    std::vector<range_calendar_detail::PeerTask> peers(3);
    peers[0].spmd_targets = {0, 2};
    peers[1].spmd_targets = {1, 2};
    peers[2].spmd_targets = {0, 2};
    ASSERT_TRUE(range_calendar_detail::prepare_peer_fanouts(image, peers));
    EXPECT_EQ(peers[0].fanout_offset, peers[2].fanout_offset);
    EXPECT_NE(peers[0].fanout_offset, peers[1].fanout_offset);
    auto *calendar = reinterpret_cast<CalendarRangeHeader *>(image.bytes.data());
    calendar->peer_fanouts_offset = image.peer_fanouts_offset;
    calendar->worker_count = 3;
    CalendarPeerEndpoint endpoints[] = {
        {0x100, UINT32_MAX, CALENDAR_PEER_SPMD_ENDPOINT},
        {0x200, UINT32_MAX, CALENDAR_PEER_SPMD_ENDPOINT},
        {0x300, UINT32_MAX, CALENDAR_PEER_SPMD_ENDPOINT}
    };
    const auto offset = peers[0].fanout_offset;
    const auto addresses = static_cast<uint32_t>(offset + sizeof(CalendarPeerFanout));
    ASSERT_TRUE(percounter_bind_peer_fanouts(*calendar, image.bytes.size(), endpoints));
    EXPECT_EQ(image.load<CalendarPeerFanout>(offset).common_token_base, uint64_t{UINT32_MAX});
    EXPECT_EQ(image.load<uint64_t>(addresses), 0x100U);
    EXPECT_EQ(image.load<uint64_t>(addresses + sizeof(uint64_t)), 0x300U);
    endpoints[2].address = 0x400;
    endpoints[2].token_base = 17;
    ASSERT_TRUE(percounter_bind_peer_fanouts(*calendar, image.bytes.size(), endpoints));
    EXPECT_EQ(image.load<CalendarPeerFanout>(offset).common_token_base, UINT64_MAX);
    EXPECT_EQ(image.load<uint64_t>(addresses + sizeof(uint64_t)), 0x400U);
    EXPECT_EQ(image.load<uint32_t>(addresses + 2 * sizeof(uint64_t) + sizeof(uint32_t)), 17U);
    endpoints[2].input_count = 0;
    EXPECT_FALSE(percounter_bind_peer_fanouts(*calendar, image.bytes.size(), endpoints));
}

TEST(CalendarPhases, MulticastBurstsCoverOnlySubscribedAddresses) {
    const auto check = [](const std::vector<uint64_t> &addresses, uint32_t expected) {
        std::vector<CalendarPeerBurst> bursts(addresses.size() / 4);
        const auto count = percounter_pack_peer_bursts(addresses.data(), addresses.size(), bursts.data());
        ASSERT_EQ(count, expected);
        if (!count) return;
        std::vector<uint64_t> actual;
        for (uint16_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < bursts[i].targets; ++j)
                actual.push_back(bursts[i].start + uint64_t{j} * bursts[i].stride_bytes);
        }
        auto wanted = addresses;
        std::sort(wanted.begin(), wanted.end());
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, wanted);
    };
    std::vector<uint64_t> pairs;
    for (uint64_t i = 0; i < 16; ++i) {
        pairs.push_back(0x100000 - 0x1000 * i);
        pairs.push_back(0x100100 - 0x1000 * i);
    }
    check(pairs, 2);
    pairs.push_back(0x1000060);
    check(pairs, 3);
    check({4, 8, 12, 16}, 1);
    check({4, 12, 28, 60}, 0);
    check({4, (1ULL << 32) + 4, (2ULL << 32) + 4, (3ULL << 32) + 4}, 0);
}

TEST(CalendarPhases, FusesAnExclusivePairAndPreservesOtherCompletionCounters) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}, {2}};
    auto tasks = cube_tasks(fanins);
    tasks[2].block_num = 2;
    const GraphPlanInput graphs{};
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, graphs, &phases));
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, 2, {}, plan.get()));
    ASSERT_TRUE(plan->waits.summarized_fanins[2]);

    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, graphs, tasks.size(), 2, &image));
    ASSERT_EQ(image.fused_fanins.size(), 1U);
    EXPECT_EQ(image.fused_fanins[0].producers, (std::array<uint32_t, CALENDAR_FUSED_FANIN_MAX_PRODUCERS>{0, 1}));
    EXPECT_EQ(image.fused_fanins[0].producer_count, 2U);
    EXPECT_EQ(image.fused_fanins[0].credits, 2U);
    EXPECT_EQ(image.completion.tasks[0].slot, UINT32_MAX);
    EXPECT_EQ(image.completion.tasks[1].slot, UINT32_MAX);
    EXPECT_NE(image.completion.tasks[2].slot, UINT32_MAX);
    EXPECT_NE(image.completion.tasks[3].slot, UINT32_MAX);
    EXPECT_EQ(image.phase_ends[0].task, 3U);

    struct HostTables {
        PercounterControl control{};
        std::array<PercounterCounter, 4> counters{};
    } tables;
    PercounterLayout layout{};
    layout.task_count = tasks.size();
    layout.control_offset = offsetof(HostTables, control);
    layout.counters_offset = offsetof(HostTables, counters);
    std::vector<uint64_t> notifications(image.load<CalendarRangeHeader>(0).notify_slots_count);
    for (size_t i = 0; i < notifications.size(); ++i)
        notifications[i] = 0x1000 + i * 32;
    ASSERT_TRUE(bind_range_calendar(image, 0, layout, &tables, notifications));
    const auto &a = tables.counters[0];
    const auto &b = tables.counters[1];
    EXPECT_NE(a.completion_notify_address, 0U);
    EXPECT_EQ(a.completion_notify_address, b.completion_notify_address);
    EXPECT_NE(a.completion_notify_address, tables.counters[2].completion_notify_address);
    EXPECT_EQ(a.calendar_defer_bit, 1U << 24);
    EXPECT_EQ(b.calendar_defer_bit, 1U << 25);
    EXPECT_EQ(a.completion_notify_target, 2U);
    EXPECT_EQ(b.completion_notify_target, 2U);

    CalendarEventCosts costs;
    costs.fanin_credit_cycles = 4;
    costs.fanin_observe_cycles = 2;
    uint64_t ordinary_cycles = 0;
    uint64_t fused_cycles = 0;
    ASSERT_TRUE(evaluate_calendar_events(phases[0].tasks, 2, *plan, {}, costs, &ordinary_cycles));
    ASSERT_TRUE(evaluate_calendar_events(phases[0].tasks, 2, *plan, {}, costs, &fused_cycles, true));
    EXPECT_EQ(fused_cycles, 5U);
    EXPECT_EQ(ordinary_cycles, 9U);
}

TEST(CalendarPhases, FusesWiderGroupsOnlyWithOneHelperPerProducer) {
    constexpr uint32_t clusters = 4;
    for (const uint32_t producers : {3U, 4U}) {
        const uint32_t consumer = producers;
        const uint32_t task_count = producers + 2;
        std::vector<std::vector<int32_t>> fanins(task_count);
        for (uint32_t producer = 0; producer < producers; ++producer)
            fanins[consumer].push_back(static_cast<int32_t>(producer));
        fanins.back().push_back(static_cast<int32_t>(consumer));

        for (const uint32_t helpers : {producers, 2U}) {
            SCOPED_TRACE(testing::Message() << "producers=" << producers << ", helpers=" << helpers);
            auto tasks = cube_tasks(fanins);
            tasks[consumer].block_num = helpers;
            std::vector<CalendarPhase> phases;
            ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
            auto plan = std::make_shared<CalendarPlan>();
            ASSERT_TRUE(build_static_calendar(phases[0].tasks, clusters, {}, plan.get()));
            ASSERT_TRUE(plan->waits.summarized_fanins[consumer]);

            RangeCalendarImage image;
            ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, {}, task_count, clusters, &image));
            if (helpers != producers) {
                EXPECT_TRUE(
                    std::any_of(
                        plan->waits.row_assignments.begin(), plan->waits.row_assignments.end(),
                        [](const auto &assignment) {
                            return assignment.count > 1;
                        }
                    )
                );
                EXPECT_TRUE(image.fused_fanins.empty());
                EXPECT_EQ(image.fanin_slot_count, 1U);
                continue;
            }

            ASSERT_EQ(image.fused_fanins.size(), 1U);
            EXPECT_EQ(image.fused_fanins[0].producer_count, producers);
            EXPECT_EQ(image.fused_fanins[0].credits, producers);
            struct HostTables {
                PercounterControl control{};
                std::array<PercounterCounter, clusters + 2> counters{};
            } tables;
            PercounterLayout layout{};
            layout.task_count = task_count;
            layout.control_offset = offsetof(HostTables, control);
            layout.counters_offset = offsetof(HostTables, counters);
            std::vector<uint64_t> notifications(image.load<CalendarRangeHeader>(0).notify_slots_count);
            for (size_t i = 0; i < notifications.size(); ++i)
                notifications[i] = 0x1000 + i * 32;
            ASSERT_TRUE(bind_range_calendar(image, 0, layout, &tables, notifications));

            const auto group_wait = std::find_if(image.waits.begin(), image.waits.end(), [](const auto &wait) {
                return wait.kind == RangeCalendarImage::WaitKind::FUSED_FANIN;
            });
            ASSERT_NE(group_wait, image.waits.end());
            const auto header = image.load<CalendarWait>(group_wait->offset);
            EXPECT_EQ(header.read_cmd & CALENDAR_WAIT_TAG_MASK, CALENDAR_WAIT_FUSED_FANIN);
            EXPECT_EQ(header.target, (uint64_t{producers} << 32) | producers);
            for (uint32_t producer = 0; producer < producers; ++producer) {
                const auto &counter = tables.counters[producer];
                EXPECT_EQ(counter.completion_notify_address, header.read_cmd & ~CALENDAR_WAIT_TAG_MASK);
                EXPECT_EQ(counter.completion_notify_target, producers);
                EXPECT_EQ(counter.calendar_defer_bit, 1U << (24 + producer));
            }
        }
    }
}

TEST(CalendarPhases, ABypassCompletionObserverPreventsFusion) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}, {0}, {2, 3}};
    auto tasks = cube_tasks(fanins);
    tasks[2].block_num = 2;
    // Use a lane that never performed the original two-input wait.
    tasks[3].active_mask = 2;
    tasks[3].core_type_index = 1;
    const GraphPlanInput graphs{};
    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, graphs, &phases));
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, 2, {}, plan.get()));
    ASSERT_TRUE(plan->waits.summarized_fanins[2]);
    ASSERT_EQ(plan->waits.task_fanins[3], (std::vector<int32_t>{0}));
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, graphs, tasks.size(), 2, &image));
    EXPECT_TRUE(image.fused_fanins.empty());
    EXPECT_NE(image.completion.tasks[0].slot, image.completion.tasks[1].slot);
}

TEST(CalendarPhases, ClusterFaninsComposeUnlessTheFallbackObservesAPairedProducer) {
    for (const bool shares_producer : {false, true}) {
        const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}, {}, {shares_producer ? 0 : 3}, {2, 4}};
        auto tasks = cube_tasks(fanins);
        tasks[2].block_num = 2;
        tasks[4].active_mask = 2;
        tasks[4].core_type_index = 1;
        tasks[4].block_num = 4;
        const GraphPlanInput graphs{};
        std::vector<CalendarPhase> phases;
        ASSERT_TRUE(plan_calendar_phases(tasks, graphs, &phases));
        auto plan = std::make_shared<CalendarPlan>();
        ASSERT_TRUE(build_static_calendar(phases[0].tasks, 2, {}, plan.get()));
        ASSERT_TRUE(plan->waits.summarized_fanins[2]);
        RangeCalendarImage image;
        ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}}, graphs, tasks.size(), 2, &image));
        ASSERT_TRUE(image.uses_cluster_input_share);
        if (shares_producer) {
            EXPECT_TRUE(image.fused_fanins.empty());
            EXPECT_NE(image.completion.tasks[0].slot, UINT32_MAX);
        } else {
            ASSERT_EQ(image.fused_fanins.size(), 1U);
            EXPECT_EQ(
                image.fused_fanins[0].producers, (std::array<uint32_t, CALENDAR_FUSED_FANIN_MAX_PRODUCERS>{0, 1})
            );
        }
    }
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
    EXPECT_EQ(plan.row_input_owners, std::vector<uint32_t>(rows.size(), CALENDAR_NO_INPUT_OWNER));
}

TEST(CalendarPhases, SharesAivInputsWithAClusterSibling) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}};
    auto tasks = cube_tasks(fanins);
    tasks[1].active_mask = 2;
    tasks[1].core_type_index = 1;
    tasks[1].block_num = 4;

    std::vector<CalendarPhase> phases;
    ASSERT_TRUE(plan_calendar_phases(tasks, {}, &phases));
    ASSERT_EQ(phases.size(), 1U);
    auto plan = std::make_shared<CalendarPlan>();
    ASSERT_TRUE(build_static_calendar(phases[0].tasks, 1, {}, plan.get()));

    // Each lane's second block keeps the existing same-lane observation.
    // A second phase can reuse immutable rows, but never the first phase's token.
    CalendarPhase later;
    later.tasks = phases[0].tasks;
    later.fanins = phases[0].fanins;
    calendar_phase_detail::bind_fanins(later);
    later.task_ids = {2, 3};
    RangeCalendarImage image;
    ASSERT_TRUE(build_range_calendar({{&phases[0], plan, false}, {&later, plan, false}}, {}, 4, 1, &image));
    ASSERT_TRUE(image.uses_cluster_input_share);
    const auto header = image.load<CalendarRangeHeader>(0);
    const auto aiv_range = [&](uint32_t worker, uint32_t phase) {
        const auto index = image.load<uint32_t>(header.worker_index_offset + worker * sizeof(uint32_t));
        return image.load<CalendarRange>(header.range_offset + (index + phase) * sizeof(CalendarRange));
    };
    const auto aiv_row = [&](uint32_t worker, uint32_t row) {
        const auto range = aiv_range(worker, 0);
        EXPECT_EQ(range.row_count, 2U);
        return image.load<CalendarExecutionRow>(range.rows_offset + row * sizeof(CalendarExecutionRow));
    };
    const auto owner = aiv_row(1, 0);
    const auto follower = aiv_row(2, 0);
    EXPECT_NE(owner.flags & CALENDAR_EXEC_INPUT_OWNER, 0);
    EXPECT_NE(follower.flags & CALENDAR_EXEC_INPUT_FOLLOWER, 0);
    EXPECT_EQ(owner.wait_count, 1);
    EXPECT_EQ(follower.wait_count, 0);
    EXPECT_EQ(owner.mix_delta, 0U);
    EXPECT_EQ(follower.mix_delta, 1U);
    for (uint32_t worker : {1, 2}) {
        EXPECT_EQ(aiv_row(worker, 1).wait_count, 0);
        EXPECT_EQ(aiv_row(worker, 1).flags & (CALENDAR_EXEC_INPUT_OWNER | CALENDAR_EXEC_INPUT_FOLLOWER), 0);
        EXPECT_EQ(aiv_range(worker, 1).rows_offset, aiv_range(worker, 0).rows_offset);
        EXPECT_EQ(aiv_range(worker, 1).entry_wait_count, 1U);
        EXPECT_GT(aiv_range(worker, 1).sequence_base, aiv_range(worker, 0).sequence_base + follower.mix_delta);
    }
    PercounterLayout layout;
    ASSERT_TRUE(plan_layout(4, 8, &layout, 0, {}, 0, 0, image.uses_cluster_input_share));
    EXPECT_NE(layout.mix_signal_ready_offset, 0U);
    EXPECT_EQ(layout.mix_task_count, 0U);
}

TEST(CalendarPhases, SharesOnlyOrdinarySummaryFollowers) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}};
    auto tasks = cube_tasks(fanins);
    tasks[2].active_mask = 2;
    tasks[2].core_type_index = 1;
    tasks[2].block_num = 4;
    const std::vector<CalendarRow> rows{
        {percounter_order_entry(0, 0, 1), CALENDAR_NO_JOIN}, {percounter_order_entry(1, 0, 1), CALENDAR_NO_JOIN},
        {percounter_order_entry(2, 0, 4), CALENDAR_NO_JOIN}, {percounter_order_entry(2, 1, 4), CALENDAR_NO_JOIN},
        {percounter_order_entry(2, 2, 4), CALENDAR_NO_JOIN}, {percounter_order_entry(2, 3, 4), CALENDAR_NO_JOIN},
    };
    range_calendar_detail::CalendarWaitPlan waits;
    ASSERT_TRUE(range_calendar_detail::plan_calendar_waits(tasks, rows, {0, 1, 2, 3, 4, 5, 6}, 0, waits));
    ASSERT_TRUE(waits.summarized_fanins[2]);
    EXPECT_EQ(waits.row_assignments[2].count, 1U);
    EXPECT_EQ(waits.row_assignments[3].count, 1U);
    EXPECT_EQ(waits.row_input_owners[2], CALENDAR_NO_INPUT_OWNER);
    EXPECT_EQ(waits.row_input_owners[3], CALENDAR_NO_INPUT_OWNER);
    EXPECT_EQ(waits.row_input_owners[4], 4U);
    EXPECT_EQ(waits.row_input_owners[5], 4U);
}

TEST(CalendarPhases, KeepsInputSharingWithinAClusterAndOrdersItsOwnerFirst) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {1}, {0}};
    auto tasks = cube_tasks(fanins);
    for (size_t id : {2, 3}) {
        tasks[id].active_mask = 2;
        tasks[id].core_type_index = 1;
    }
    tasks[3].block_num = 2;
    const std::vector<CalendarRow> rows{
        {percounter_order_entry(0, 0, 1), CALENDAR_NO_JOIN}, {percounter_order_entry(1, 0, 1), CALENDAR_NO_JOIN},
        {percounter_order_entry(2, 0, 1), CALENDAR_NO_JOIN}, {percounter_order_entry(3, 0, 2), CALENDAR_NO_JOIN},
        {percounter_order_entry(3, 1, 2), CALENDAR_NO_JOIN},
    };
    range_calendar_detail::CalendarWaitPlan plan;
    // Task 3's first AIV has an earlier packed row but a later topological gate.
    ASSERT_TRUE(range_calendar_detail::plan_calendar_waits(tasks, rows, {0, 2, 4, 5}, 0, plan));
    EXPECT_EQ(plan.row_input_owners[3], 4U);
    EXPECT_EQ(plan.row_input_owners[4], 4U);

    // Leave the sibling in a different cluster. Both blocks still wait directly.
    range_calendar_detail::CalendarWaitPlan separate_clusters;
    ASSERT_TRUE(range_calendar_detail::plan_calendar_waits(tasks, rows, {0, 2, 2, 4, 4, 5, 5}, 0, separate_clusters));
    EXPECT_TRUE(separate_clusters.row_waits[3]);
    EXPECT_TRUE(separate_clusters.row_waits[4]);
    EXPECT_EQ(separate_clusters.row_input_owners, std::vector<uint32_t>(rows.size(), CALENDAR_NO_INPUT_OWNER));

    tasks[3].sync_start = true;
    range_calendar_detail::CalendarWaitPlan synchronized;
    ASSERT_TRUE(range_calendar_detail::plan_calendar_waits(tasks, rows, {0, 2, 4, 5}, 0, synchronized));
    EXPECT_EQ(synchronized.row_input_owners, std::vector<uint32_t>(rows.size(), CALENDAR_NO_INPUT_OWNER));
}

TEST(CalendarPhases, ReducesTransitiveWaitsButKeepsTheFirstObservers) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}, {0, 1, 2}};
    auto tasks = cube_tasks(fanins);
    tasks[3].block_num = 2;
    CalendarPlan plan;
    ASSERT_TRUE(build_static_calendar(tasks, 2, {}, &plan));
    EXPECT_EQ(plan.waits.task_fanins[2], (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(plan.waits.task_fanins[3], (std::vector<int32_t>{2}));
    EXPECT_FALSE(plan.waits.summarized_fanins[3]);
    EXPECT_EQ(tasks[3].fanin_count, 3);
    EXPECT_EQ(tasks[3].fanin_ids, fanins[3].data());
}

TEST(CalendarPhases, KeepsControlWaitOnTheLaneThatDidNotExecuteIt) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}};
    auto tasks = cube_tasks(fanins);
    tasks[0].active_mask = 0;
    tasks[0].core_type_index = 1;
    tasks[1].active_mask = 2;
    tasks[1].core_type_index = 1;
    tasks[1].block_num = 2;
    tasks[1].sync_start = true;

    CalendarPlan plan;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &plan));
    EXPECT_EQ(plan.waits.task_fanins[1], (std::vector<int32_t>{0}));
    EXPECT_EQ(plan.waits.row_waits, (std::vector<bool>{false, false, true}));
}

TEST(CalendarEvents, MeasuresCohortTailFromTheLastEntry) {
    CalendarTiming sample{};
    sample.start = 10;
    sample.end = 40;
    sample.defer_register = 42;
    sample.defer_completion = 58;

    const auto middle = calendar_sample_cost(sample, 30);
    ASSERT_TRUE(middle);
    EXPECT_EQ(middle->occupancy_cycles, 12U);
    EXPECT_EQ(middle->defer_tail_cycles, 16U);

    const auto late = calendar_sample_cost(sample, 50);
    ASSERT_TRUE(late);
    EXPECT_EQ(late->occupancy_cycles, 1U);
    EXPECT_EQ(late->defer_tail_cycles, 8U);
}

TEST(CalendarPhases, TransitiveWaitProofStopsAtTheBitmapBoundary) {
    std::vector<std::vector<int32_t>> fanins(66);
    fanins[63] = {0};
    fanins[64] = {0, 63};
    fanins[65] = {0, 64};
    const auto tasks = cube_tasks(fanins);
    std::vector<uint64_t> local;
    ASSERT_TRUE(range_calendar_detail::calendar_transitive_waits(tasks, local));
    EXPECT_EQ(local[63], 0U);
    EXPECT_EQ(local[64], UINT64_C(1) << 63);
    EXPECT_EQ(local[65], UINT64_C(1) << 1);
    CalendarPlan plan;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &plan));
    EXPECT_EQ(plan.waits.task_fanins[64], (std::vector<int32_t>{63}));
    EXPECT_EQ(plan.waits.task_fanins[65], (std::vector<int32_t>{64}));
}

TEST(CalendarPhases, ProvesLongWaitsUsingRealDependencyChains) {
    constexpr uint32_t chain_length = 72;
    constexpr uint32_t independent = 1;
    constexpr uint32_t consumer = chain_length + 2;
    std::vector<TaskInput> tasks(consumer + 1);
    std::vector<int32_t> previous(tasks.size());
    for (uint32_t id = 2; id < consumer; ++id) {
        previous[id] = id == 2 ? 0 : id - 1;
        tasks[id].fanin_count = 1;
        tasks[id].fanin_ids = &previous[id];
    }

    const int32_t fanin[]{0, static_cast<int32_t>(independent), static_cast<int32_t>(consumer - 1)};
    tasks[consumer].fanin_count = 3;
    tasks[consumer].fanin_ids = fanin;

    const range_calendar_detail::CalendarWaitChains chains(tasks);
    EXPECT_TRUE(chains.orders(consumer, 0));
    EXPECT_FALSE(chains.orders(consumer, independent));
    EXPECT_FALSE(chains.orders(consumer, consumer - 1));

    CalendarPlan plan;
    for (auto &task : tasks)
        task.active_mask = 1;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &plan));
    EXPECT_EQ(plan.waits.task_fanins[consumer], (std::vector<int32_t>{1, static_cast<int32_t>(consumer - 1)}));
}

TEST(CalendarPhases, ReusesACompletedWaitOnlyAfterAnUnconditionalRowOnThatLane) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}};
    auto tasks = cube_tasks(fanins);
    tasks[1].block_num = 2;
    CalendarPlan ordinary;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &ordinary));
    EXPECT_EQ(ordinary.waits.row_waits, (std::vector<bool>{false, true, false}));

    tasks[1].has_predicate = true;
    CalendarPlan predicated;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &predicated));
    EXPECT_EQ(predicated.waits.row_waits, (std::vector<bool>{false, true, true}));
}

TEST(CalendarPhases, ControlRetirementProofIsScopedToTheActualLane) {
    for (const uint32_t controls : {1, 65}) {
        std::vector<std::vector<int32_t>> predecessors(controls + 2);
        predecessors[controls] = predecessors[controls + 1] = {0};
        auto tasks = cube_tasks(predecessors);
        for (uint32_t id = 0; id < tasks.size(); ++id) {
            tasks[id].core_type_index = 1;
            tasks[id].active_mask = id < controls ? 0U : 2U;
        }
        calendar_detail::Shape shape;
        ASSERT_TRUE(calendar_detail::describe(tasks, 1, {}, &shape));
        calendar_detail::PlacementState state;
        state.workers.resize(3);
        state.free_at.resize(3);
        for (uint32_t id = 0; id < tasks.size(); ++id) {
            const uint32_t mask = id == controls + 1 ? 4U : 2U;
            calendar_detail::append_block(&state, 1, tasks[id], shape.tasks[id], id, 0, {0, mask, 0}, id, {});
        }
        CalendarPlan plan;
        ASSERT_TRUE(calendar_detail::pack(state, tasks, shape.row_count, 0, &plan));
        EXPECT_TRUE(plan.waits.task_fanins[controls].empty());
        EXPECT_EQ(plan.waits.task_fanins[controls + 1], (std::vector<int32_t>{0}));
    }
}

TEST(CalendarPhases, BackfillPreservesTheSelectedCubeGroupAgainstMix) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0}, {}, {}, {1}};
    auto tasks = cube_tasks(fanins);
    for (const auto task : {0, 1}) {
        tasks[task].active_mask = 2;
        tasks[task].core_type_index = 1;
    }
    for (const auto task : {4, 5}) {
        tasks[task].active_mask = 3;
        tasks[task].is_mix = true;
        tasks[task].subtask_count = 2;
    }
    const std::vector<CalendarRowCost> costs{{1, 9}, {1, 14}, {10, 0}, {7, 0}, {1, 0}, {1, 0}, {30, 0}, {30, 0}};
    CalendarPlan plan;
    ASSERT_TRUE(build_static_calendar(
        tasks, 1, costs, &plan, CalendarBlockOrder::SUBMISSION, 10, CalendarPriority::LONGEST_ROW, true
    ));
    calendar_detail::Shape shape;
    calendar_detail::CalendarLayout layout;
    ASSERT_TRUE(calendar_detail::describe(tasks, 1, costs, &shape));
    ASSERT_TRUE(calendar_detail::read_calendar_layout(tasks, 1, shape, plan, layout));
    ASSERT_GE(layout.indices[1], 2U);
    EXPECT_EQ(percounter_order_task_id(layout.rows[0].entry), 3);
    EXPECT_EQ(percounter_order_task_id(layout.rows[1].entry), 2);
}

TEST(CalendarEvents, ScoresTheWaitProtocolOfEachPlacement) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}};
    auto tasks = cube_tasks(fanins);
    tasks[2].block_num = 2;
    const std::vector<CalendarRowCost> costs{{8, 0}, {8, 0}, {8, 0}, {1, 0}};
    calendar_detail::Shape shape;
    ASSERT_TRUE(calendar_detail::describe(tasks, 2, costs, &shape));
    const auto pack = [&](uint32_t second_consumer_cluster, CalendarPlan *plan) {
        calendar_detail::PlacementState state;
        state.workers.resize(6);
        state.free_at.resize(6);
        const auto append = [&](uint32_t task, uint32_t block, uint32_t cluster) {
            calendar_detail::append_block(
                &state, 2, tasks[task], shape.tasks[task], task, block, {cluster, 1, 0}, 0, costs
            );
        };
        append(0, 0, 0);
        append(1, 0, 1);
        append(2, 0, 0);
        append(2, 1, second_consumer_cluster);
        return calendar_detail::pack(state, tasks, shape.row_count, 0, plan);
    };

    CalendarPlan summarized;
    CalendarPlan direct;
    ASSERT_TRUE(pack(1, &summarized));
    ASSERT_TRUE(pack(0, &direct));
    ASSERT_TRUE(summarized.waits.summarized_fanins[2]);
    ASSERT_FALSE(direct.waits.summarized_fanins[2]);
    uint64_t summarized_cycles = 0;
    uint64_t direct_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(tasks, 2, summarized, costs, &summarized_cycles));
    ASSERT_TRUE(evaluate_static_calendar(tasks, 2, direct, costs, &direct_cycles));
    EXPECT_EQ(summarized_cycles, 16U);
    EXPECT_EQ(direct_cycles, 17U);

    CalendarEventCosts timing;
    timing.direct_observe_cycles = 1;
    timing.fanin_credit_cycles = 4;
    timing.fanin_observe_cycles = 2;
    ASSERT_TRUE(evaluate_calendar_events(tasks, 2, summarized, costs, timing, &summarized_cycles));
    ASSERT_TRUE(evaluate_calendar_events(tasks, 2, direct, costs, timing, &direct_cycles));
    EXPECT_EQ(summarized_cycles, 22U);
    EXPECT_EQ(direct_cycles, 19U);  // The second direct consumer reuses its lane observation.
}

TEST(CalendarEvents, ADelayedOwnerDoesNotHoldTheFollowerAfterItsDoneProbe) {
    calendar_detail::CalendarEventGraph graph(1);
    const auto start = graph.add_event();
    const auto arrival = graph.add_event();
    const auto owner = graph.add_event();
    const auto done = graph.add_event();
    const auto ready = graph.add_event();
    const auto exit = graph.add_event();
    graph.link(start, owner, 200);
    graph.link(start, done, 66);
    graph.add_follower({arrival, owner, done, false}, ready);
    graph.link(ready, exit, 100);

    CalendarEventCosts costs;
    costs.direct_observe_cycles = 4;
    uint64_t cycles = 0;
    ASSERT_TRUE(graph.replay(costs, cycles));
    // DONE misses the first probe at cycle 64 and is observed at 64 + 4 + 64 + 4.
    EXPECT_EQ(cycles, 236U);
}

TEST(CalendarEvents, ChargesDirectProbesInWaitListOrder) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {0, 1}};
    auto tasks = cube_tasks(fanins);
    const std::vector<CalendarRowCost> row_costs{{2, 0}, {10, 0}, {1, 0}};
    calendar_detail::Shape shape;
    ASSERT_TRUE(calendar_detail::describe(tasks, 3, row_costs, &shape));
    calendar_detail::PlacementState state;
    state.workers.resize(9);
    state.free_at.resize(9);
    for (uint32_t task = 0; task < 3; ++task) {
        calendar_detail::append_block(&state, 3, tasks[task], shape.tasks[task], task, 0, {task, 1, 0}, 0, row_costs);
    }

    CalendarEventCosts costs;
    costs.direct_observe_cycles = 3;
    CalendarPlan plan;
    uint64_t cycles = 0;
    ASSERT_TRUE(calendar_detail::pack(state, tasks, shape.row_count, 0, &plan));
    ASSERT_TRUE(evaluate_calendar_events(tasks, 3, plan, row_costs, costs, &cycles));
    EXPECT_EQ(cycles, 14U);
    const int32_t reversed[] = {1, 0};
    tasks[2].fanin_ids = reversed;
    ASSERT_TRUE(calendar_detail::pack(state, tasks, shape.row_count, 0, &plan));
    ASSERT_TRUE(evaluate_calendar_events(tasks, 3, plan, row_costs, costs, &cycles));
    EXPECT_EQ(cycles, 17U);
}

TEST(CalendarEvents, AHelperServicesReleasedCreditsSerially) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {}, {}, {0, 1, 2, 3}};
    auto tasks = cube_tasks(fanins);
    for (size_t i = 0; i < 4; ++i) {
        tasks[i].active_mask = 2;
        tasks[i].core_type_index = 1;
    }
    tasks[4].block_num = 2;
    const std::vector<CalendarRowCost> row_costs{{10, 0}, {6, 0}, {9, 0}, {7, 0}, {1, 0}, {1, 0}};
    calendar_detail::Shape shape;
    ASSERT_TRUE(calendar_detail::describe(tasks, 2, row_costs, &shape));
    calendar_detail::PlacementState state;
    state.workers.resize(6);
    state.free_at.resize(6);
    for (uint32_t task = 0; task < 4; ++task) {
        const calendar_detail::Placement placement{task / 2, (task % 2 == 0) ? 2U : 4U, 0};
        calendar_detail::append_block(&state, 2, tasks[task], shape.tasks[task], task, 0, placement, 0, row_costs);
    }
    for (uint32_t block = 0; block < 2; ++block) {
        calendar_detail::append_block(&state, 2, tasks[4], shape.tasks[4], 4, block, {block, 1, 0}, 0, row_costs);
    }
    CalendarPlan plan;
    ASSERT_TRUE(calendar_detail::pack(state, tasks, shape.row_count, 0, &plan));
    ASSERT_TRUE(plan.waits.summarized_fanins[4]);

    CalendarEventCosts costs;
    costs.fanin_credit_cycles = 4;
    uint64_t cycles = 0;
    ASSERT_TRUE(evaluate_calendar_events(tasks, 2, plan, row_costs, costs, &cycles));
    EXPECT_EQ(cycles, 16U);
}

TEST(CalendarEvents, JoinsLocalMixRolesBeforeTheStartCohort) {
    const std::vector<std::vector<int32_t>> fanins{{}, {}, {}};
    auto tasks = cube_tasks(fanins);
    tasks[1].active_mask = 2;
    tasks[1].core_type_index = 1;
    tasks[2].active_mask = 3;
    tasks[2].is_mix = true;
    tasks[2].sync_start = true;
    tasks[2].subtask_count = 2;
    tasks[2].block_num = 2;
    const std::vector<CalendarRowCost> row_costs{{5, 0}, {2, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}};
    calendar_detail::Shape shape;
    ASSERT_TRUE(calendar_detail::describe(tasks, 2, row_costs, &shape));
    calendar_detail::PlacementState state;
    state.workers.resize(6);
    state.free_at.resize(6);
    const auto append = [&](uint32_t task, uint32_t block, uint32_t cluster) {
        calendar_detail::append_block(
            &state, 2, tasks[task], shape.tasks[task], task, block, {cluster, tasks[task].active_mask, 0}, 0, row_costs
        );
    };
    append(0, 0, 0);
    append(1, 0, 1);
    append(2, 0, 0);
    append(2, 1, 1);
    CalendarPlan plan;
    ASSERT_TRUE(calendar_detail::pack(state, tasks, shape.row_count, 0, &plan));

    CalendarEventCosts costs;
    costs.row_prepare_cycles = 1;
    costs.mix_arrival_cycles = 2;
    costs.mix_release_cycles = 4;
    costs.mix_follower_observe_cycles = 1;
    costs.entry_cycles = 5;
    uint64_t cycles = 0;
    ASSERT_TRUE(evaluate_calendar_events(tasks, 2, plan, row_costs, costs, &cycles));
    EXPECT_EQ(cycles, 25U);
}

TEST(CalendarEvents, RejectsALaneOrderThatInvertsAnExecutionDependency) {
    const std::vector<std::vector<int32_t>> fanins{{}, {0}};
    const auto tasks = cube_tasks(fanins);
    CalendarPlan plan;
    ASSERT_TRUE(build_static_calendar(tasks, 1, {}, &plan));
    CalendarHeader header{};
    std::memcpy(&header, plan.bytes.data(), sizeof(header));
    ASSERT_EQ(header.row_count, 2U);
    CalendarRow rows[2]{};
    std::memcpy(rows, plan.bytes.data() + header.row_offset, sizeof(rows));
    std::swap(rows[0], rows[1]);
    std::memcpy(plan.bytes.data() + header.row_offset, rows, sizeof(rows));
    std::swap(plan.logical_row_by_packed[0], plan.logical_row_by_packed[1]);
    plan.waits.row_waits = {true, false};

    uint64_t cycles = 123;
    EXPECT_FALSE(evaluate_calendar_events(tasks, 1, plan, {}, {}, &cycles));
    EXPECT_EQ(cycles, 123U);
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
    EXPECT_EQ(range(1).phase_row_offset, 0U);
    EXPECT_EQ(range(2).phase_row_offset, range(1).row_count);
    EXPECT_EQ(range(2).exit_wait_count, 1U);
    EXPECT_EQ(range(4).exit_wait_count, 1U);
    EXPECT_NE(range(2).exit_wait_offset, range(4).exit_wait_offset);
}

}  // namespace
}  // namespace simpler::percounter
