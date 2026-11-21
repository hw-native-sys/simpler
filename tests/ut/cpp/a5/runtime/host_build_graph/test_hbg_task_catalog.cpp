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
#include <stdexcept>
#include <vector>

#include "dependency_plan.h"
#include "scheduler/scheduler_issue.h"
#include "hbg_scheduler_test_support.h"
#include "task_catalog_plan.h"

namespace {

SchedulerTaskMetadata physical(LaneMask lanes, uint16_t blocks = 1) {
    SchedulerTaskMetadata result{};
    result.flags = SCHEDULER_TASK_EXECUTABLE | (lanes.is_mix() ? SCHEDULER_TASK_MIX : 0);
    result.active_mask = lanes;
    result.logical_block_num = blocks;
    return result;
}

struct CatalogImage {
    SchedulerTaskCatalogPlan plan;
    std::vector<uint8_t> storage;
    void *base{nullptr};

    explicit CatalogImage(const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters = 2) {
        if (!scheduler_plan_task_catalog(metadata, clusters, 64, &plan))
            throw std::logic_error("cannot plan test catalog");
        storage.resize(plan.total_size + SCHEDULER_STATE_BASE_ALIGNMENT);
        base = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_BASE_ALIGNMENT - 1) &
            ~(uintptr_t{SCHEDULER_STATE_BASE_ALIGNMENT} - 1)
        );
        scheduler_init_task_catalog(base, plan);
    }

    bool valid() {
        return scheduler_task_catalog_validate(
            base, plan.total_size, plan.header_offset, plan.header.task_count, plan.cluster_count
        );
    }
};

template <size_t MaxTasks>
bool prepare_ordinary(
    scheduler_test::BasicGraphBuffer<MaxTasks> &graph, std::vector<SchedulerTaskMetadata> &metadata,
    std::vector<int64_t> &completed, SchedulerTaskCatalogPlan &catalog, SchedulerDependencyPlan &dependencies,
    std::string *error = nullptr
) {
    const SchedulerOrdinaryFaninPool pool{graph.payload(0).fanin_data(), metadata.size() * SCHEDULER_GRAPH_MAX_FANIN};
    return scheduler_order_task_catalog(graph.graph(), metadata, completed, &catalog, error, &dependencies, &pool) &&
           scheduler_prepare_ordinary_fanin(
               graph.storage(), pool.data, pool.capacity, metadata, completed, catalog, dependencies, error
           );
}

template <size_t MaxTasks>
bool prepare_ordinary(scheduler_test::BasicGraphBuffer<MaxTasks> &graph, std::vector<int64_t> completed = {}) {
    std::vector<SchedulerTaskMetadata> metadata(graph.graph().task_count, physical(LaneMask::aic()));
    CatalogImage image(metadata);
    SchedulerDependencyPlan dependencies;
    return prepare_ordinary(graph, metadata, completed, image.plan, dependencies);
}

TEST(HbgTaskCatalog, PhysicalTasksUseTwoEngineListsAndControlsRemainSeparate) {
    std::vector<SchedulerTaskMetadata> metadata{
        physical(LaneMask::aic(), 4),
        physical(LaneMask::aiv0(), 2),
        physical(LaneMask(true, true, false)),
        physical(LaneMask::both_aiv()),
        physical(LaneMask::aiv0()),
        {},
        {}
    };
    metadata[4].flags |= SCHEDULER_TASK_INLINE;
    for (auto &kernel : metadata[4].kernel_ids)
        kernel = UINT16_MAX;
    metadata[5].logical_block_num = 1;
    CatalogImage image(metadata);
    ASSERT_TRUE(image.valid());
    ASSERT_EQ(image.plan.descriptors.size(), 2u);
    EXPECT_EQ(image.plan.header.task_count, 7u);
    EXPECT_EQ(image.plan.header.entry_count, 4u);
    EXPECT_EQ(image.plan.header.control_count, 2u);
    const auto cubes = scheduler_task_catalog(image.base, image.plan.header, 0);
    const auto vectors = scheduler_task_catalog(image.base, image.plan.header, 1);
    EXPECT_EQ(std::vector<int64_t>(cubes.refs, cubes.refs + cubes.count), (std::vector<int64_t>{0}));
    EXPECT_EQ(std::vector<int64_t>(vectors.refs, vectors.refs + vectors.count), (std::vector<int64_t>{1, 2, 3}));
    EXPECT_FALSE(scheduler_task_catalog_valid(scheduler_task_catalog(image.base, image.plan.header, 2)));
    EXPECT_EQ(metadata[2].active_mask.raw(), 3u);
    EXPECT_EQ(metadata[3].active_mask.raw(), 6u);
}

TEST(HbgTaskCatalog, OppositeSourceThresholdsUsePhysicalMembersAndProxyDuty) {
    std::vector<SchedulerTaskMetadata> metadata{physical(LaneMask::both_aiv()),        physical(LaneMask::aic(), 2),
                                                physical(LaneMask(true, false, true)), physical(LaneMask::aiv0(), 2),
                                                physical(LaneMask(true, true, false)), physical(LaneMask::aiv1())};
    for (const uint32_t ref : {1u, 3u})
        metadata[ref].flags |= SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_SPMD;
    CatalogImage image(metadata);
    const auto &header = *scheduler_state_at<SchedulerTaskCatalogHeader>(image.base, image.plan.header_offset);
    ASSERT_TRUE(image.valid());
    const uint32_t required[][3] = {{0, 0, 0}, {0, 0, 0}, {2, 0, 0}, {0, 3, 3}, {2, 4, 0}, {0, 5, 3}};
    for (uint32_t ref = 0; ref < metadata.size(); ++ref) {
        SCOPED_TRACE(ref);
        const auto &entry = scheduler_task_catalog_order(image.base, header, ref);
        EXPECT_EQ(entry.rank, ref);
        for (uint32_t lane = 0; lane < 3; ++lane)
            EXPECT_EQ(entry.required_floor[lane], required[ref][lane]);
    }
    EXPECT_EQ(header.first_mix_ref, 0u);
    EXPECT_EQ(scheduler_task_catalog_order(image.base, header, 0).next_mix_ref, 2u);
    EXPECT_EQ(scheduler_task_catalog_order(image.base, header, 2).next_mix_ref, 4u);
    EXPECT_EQ(scheduler_task_catalog_order(image.base, header, 4).next_mix_ref, UINT32_MAX);
    EXPECT_EQ(scheduler_task_claim_lanes(metadata[2].flags, metadata[2].active_mask), LaneMask::aiv0());
    EXPECT_FALSE(metadata[2].active_mask.has_aiv0());
    EXPECT_EQ(scheduler_task_claim_lanes(metadata[3].flags, metadata[3].active_mask), LaneMask::both_aiv());
}

TEST(HbgTaskCatalog, EmptyAndSingleEngineRunsKeepBothDirectories) {
    for (uint32_t count : {0u, 1u}) {
        CatalogImage image(std::vector<SchedulerTaskMetadata>(count, physical(LaneMask::aic())));
        ASSERT_TRUE(image.valid());
        ASSERT_EQ(image.plan.descriptors.size(), 2u);
        EXPECT_EQ(scheduler_task_catalog(image.base, image.plan.header, 0).count, count);
        EXPECT_EQ(scheduler_task_catalog(image.base, image.plan.header, 1).count, 0u);
        EXPECT_EQ(image.plan.header.control_count, 0u);
    }
}

TEST(HbgTaskCatalog, TopologicalGenerationsExposeIndependentBranchesBeforeDeeperTasks) {
    scheduler_test::BasicGraphBuffer<16> graph(8);
    graph.executable(0, 0);
    graph.executable(1, 0);
    graph.executable(2, 0, {0});
    graph.executable(3, 0, {2});
    graph.executable(4, 0, {3});
    graph.executable(5, 0, {0});
    graph.executable(6, 0, {5});
    graph.executable(7, 0);
    const std::vector<SchedulerTaskMetadata> metadata{
        physical(LaneMask::aiv0()),
        physical(LaneMask::aiv0()),
        physical(LaneMask::aic()),
        physical(LaneMask::aiv0()),
        physical(LaneMask(true, true, false)),
        physical(LaneMask::aic()),
        physical(LaneMask::aiv0()),
        physical(LaneMask::aic())
    };
    CatalogImage image(metadata);
    std::string error;
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error)) << error;
    scheduler_init_task_catalog(image.base, image.plan);
    ASSERT_TRUE(image.valid());
    const auto cubes = scheduler_task_catalog(image.base, image.plan.header, 0);
    const auto vectors = scheduler_task_catalog(image.base, image.plan.header, 1);
    EXPECT_EQ(std::vector<int64_t>(cubes.refs, cubes.refs + cubes.count), (std::vector<int64_t>{7, 2, 5}));
    EXPECT_EQ(std::vector<int64_t>(vectors.refs, vectors.refs + vectors.count), (std::vector<int64_t>{0, 1, 3, 6, 4}));
    EXPECT_EQ(image.plan.task_order, (std::vector<int64_t>{0, 1, 7, 2, 5, 3, 6, 4}));
    EXPECT_EQ(image.plan.task_levels, (std::vector<uint32_t>{0, 0, 1, 2, 3, 1, 2, 0}));
    const auto &mixed = scheduler_task_catalog_order(image.base, image.plan.header, 4);
    EXPECT_EQ(mixed.rank, 7u);
    EXPECT_EQ(image.plan.header.first_mix_ref, 4u);
}

SchedulerTaskMetadata spmd(LaneMask lanes, uint16_t blocks, bool sync_start = false) {
    auto task = physical(lanes, blocks);
    task.flags |= SCHEDULER_TASK_SPMD;
    if (sync_start) task.flags |= SCHEDULER_TASK_SYNC_START;
    return task;
}

TEST(HbgTaskCatalog, ElidesMixHandshakeOnlyForACompletedOrdinaryPrefix) {
    const std::vector<SchedulerTaskMetadata> tasks{
        spmd(LaneMask::aic(), 2, true), physical(LaneMask::aic()), physical(LaneMask::all())
    };
    const auto mix_floor = [&](bool dependent, bool asynchronous, bool predicated) {
        auto metadata = tasks;
        if (predicated) metadata[1].flags |= SCHEDULER_TASK_HAS_PREDICATE;
        SchedulerDependencyPlan deps;
        deps.fanin = {{}, dependent ? std::vector<int64_t>{0} : std::vector<int64_t>{}, {1}};
        deps.host_done.resize(metadata.size());
        std::vector<LaneMask> async_lanes(metadata.size());
        if (asynchronous) async_lanes[1] = LaneMask::aic();
        CatalogImage image(metadata);
        EXPECT_TRUE(scheduler_order_task_catalog(deps, metadata, &image.plan));
        EXPECT_NE(image.plan.order_rows[2].required_floor[0], 0u);
        scheduler_elide_proven_mix_floors(deps, metadata, async_lanes, image.plan);
        return image.plan.order_rows[2].required_floor[0];
    };
    EXPECT_EQ(mix_floor(true, false, false), 0u);
    EXPECT_NE(mix_floor(false, false, false), 0u);
    EXPECT_NE(mix_floor(true, true, false), 0u);
    EXPECT_NE(mix_floor(true, false, true), 0u);
}

void expect_wave_order(
    uint32_t clusters, const std::vector<SchedulerTaskMetadata> &tasks, const std::vector<std::vector<int64_t>> &fanin,
    const std::vector<int64_t> &expected, bool native = false, bool deferred = false, uint64_t controls = 0
) {
    SchedulerDependencyPlan deps;
    deps.fanin = fanin;
    CatalogImage image(tasks, clusters);
    std::string error;
    const bool promote = scheduler_can_defer_side_tasks(native, deferred, controls, tasks);
    ASSERT_TRUE(scheduler_order_task_catalog(deps, tasks, &image.plan, &error, promote)) << error;
    EXPECT_EQ(image.plan.task_order, expected);
    EXPECT_EQ(deps.fanin, fanin);
    for (size_t ref = 0; ref < tasks.size(); ++ref)
        for (const auto pred : fanin[ref])
            EXPECT_LT(image.plan.task_ranks[pred], image.plan.task_ranks[ref]);
    scheduler_init_task_catalog(image.base, image.plan);
    EXPECT_TRUE(image.valid());
}

TEST(HbgTaskCatalog, ShortIndependentBranchYieldsOnlyToAnOverlappingWaveOnItsOwnEngine) {
    const std::vector<std::vector<int64_t>> fanin{{}, {}, {1}, {2}};
    const std::vector<int64_t> original{1, 0, 2, 3}, delayed{1, 2, 0, 3};
    for (const auto lanes : {LaneMask::aic(), LaneMask::aiv0()}) {
        SCOPED_TRACE(static_cast<int>(lanes.raw()));
        const uint16_t capacity = lanes.has_aic() ? 8 : 16;
        const uint16_t side = capacity / 2 - 1;
        const uint16_t wave = capacity - side + 1;
        const auto other = lanes.has_aic() ? LaneMask::aiv0() : LaneMask::aic();
        std::vector<SchedulerTaskMetadata> tasks{
            spmd(lanes, side), physical(LaneMask::all()), spmd(lanes, wave), physical(other)
        };
        expect_wave_order(8, tasks, fanin, delayed);
        tasks[2] = spmd(lanes, wave - 1);  // Both fit at once.
        expect_wave_order(8, tasks, fanin, original);
        tasks[2] = spmd(lanes, capacity, true);
        expect_wave_order(8, tasks, fanin, delayed);  // The wave may use sync_start.
        tasks[0] = spmd(lanes, capacity / 2);
        expect_wave_order(8, tasks, fanin, original);
        tasks[0] = spmd(lanes, side, true);
        expect_wave_order(8, tasks, fanin, original);
        tasks[0] = physical(other);
        expect_wave_order(8, tasks, fanin, original);
        tasks[0] = physical(LaneMask::all());
        expect_wave_order(8, tasks, fanin, original);
    }
}

TEST(HbgTaskCatalog, MixedWaveCountsItsComputingLanesOnTheContendedEngine) {
    // Eight clusters have eight AICs and sixteen AIVs. A six-block MIX with one AIV
    // per block leaves ten vector cores available; with two per block it leaves four.
    const auto root = physical(LaneMask::all());
    const auto one_aiv = spmd(LaneMask(true, false, true), 6);
    const auto two_aivs = spmd(LaneMask::all(), 6, true);
    const std::vector<std::vector<int64_t>> fanin{{}, {}, {1}, {2}};
    expect_wave_order(8, {spmd(LaneMask::aic(), 3), root, one_aiv, root}, fanin, {1, 2, 0, 3});
    expect_wave_order(8, {spmd(LaneMask::aiv0(), 7), root, one_aiv, root}, fanin, {1, 0, 2, 3});
    expect_wave_order(8, {spmd(LaneMask::aiv0(), 7), root, two_aivs, root}, fanin, {1, 2, 0, 3});
    expect_wave_order(8, {spmd(LaneMask::aic(), 3), root, spmd(LaneMask::both_aiv(), 8), root}, fanin, {1, 0, 2, 3});
    // Ten MIX blocks execute in waves across eight clusters. A single AIV per cluster
    // leaves eight AIVs free, while a two-AIV MIX occupies all sixteen.
    const auto side = spmd(LaneMask::aiv0(), 7);
    expect_wave_order(8, {side, root, spmd(LaneMask(true, false, true), 10), root}, fanin, {1, 0, 2, 3});
    expect_wave_order(8, {side, root, spmd(LaneMask::all(), 10), root}, fanin, {1, 2, 0, 3});
    // A pure vector wave can use either AIV per cluster, so all ten blocks can be active.
    expect_wave_order(8, {side, root, spmd(LaneMask::aiv0(), 10), root}, fanin, {1, 2, 0, 3});
}

TEST(HbgTaskCatalog, DualAivSpmdRemainsAWideTargetBetweenVectorBranches) {
    const std::vector<SchedulerTaskMetadata> tasks{
        spmd(LaneMask::aiv0(), 7), spmd(LaneMask::both_aiv(), 7), physical(LaneMask::aic()), spmd(LaneMask::aiv0(), 16),
        physical(LaneMask::aic())
    };
    for (const auto &task : tasks)
        ASSERT_EQ(
            task.flags, scheduler_task_metadata_flags_from_submit_state(
                            task.active_mask, TaskAttrs{}, task.logical_block_num, false
                        )
        );
    // The seven-block dual-AIV task uses fourteen of the sixteen vector cores. It cannot be deferred.
    expect_wave_order(8, tasks, {{}, {}, {}, {2}, {3}}, {2, 0, 1, 3, 4});
    // Give that task a longer branch than task 0. Task 0 may move behind it, but cannot pass the next wave.
    expect_wave_order(8, tasks, {{}, {2}, {}, {2}, {1, 3}}, {2, 1, 0, 3, 4});
}

TEST(HbgTaskCatalog, NarrowBranchCannotPassAnotherCooperativeTaskOrAnEquallyCriticalBranch) {
    const auto lanes = LaneMask::aic();
    const auto side = spmd(lanes, 3), wave = spmd(lanes, 6);
    const auto root = physical(LaneMask::aiv0());
    const std::vector<std::vector<int64_t>> fork{{}, {}, {}, {2}, {3}};
    for (const auto barrier : {physical(LaneMask::all()), spmd(LaneMask::aiv0(), 2, true)})
        expect_wave_order(8, {side, barrier, root, wave, root}, fork, {2, 0, 1, 3, 4});
    // Both independent branches still have one edge remaining.
    expect_wave_order(8, {side, root, wave, root, root}, {{}, {}, {1}, {2}, {0}}, {1, 0, 2, 4, 3});
    // An eligible wave two generations away is outside the search.
    expect_wave_order(8, {side, root, root, wave, root}, {{}, {}, {1}, {2}, {3}}, {1, 0, 2, 3, 4});
    // The would-be delayed branch supplies a direct input to the wave.
    expect_wave_order(8, {side, root, wave, root}, {{}, {}, {0, 1}, {2}}, {0, 1, 2, 3});
}

TEST(HbgTaskCatalog, NarrowBranchStopsAtTheNearestCompetingWave) {
    const auto lanes = LaneMask::aic();
    const auto root = physical(LaneMask::aiv0());
    const auto side = spmd(lanes, 3), wave = spmd(lanes, 6);
    const std::vector<std::vector<int64_t>> fork{{}, {}, {1}, {1}, {2}, {3}};
    expect_wave_order(8, {side, root, wave, wave, root, root}, fork, {1, 2, 0, 3, 4, 5});
    // Leave the side task available alongside the first wave if both fit.
    expect_wave_order(8, {side, root, spmd(lanes, 4), wave, root, root}, fork, {1, 0, 2, 3, 4, 5});
    // A single wave can displace only the closest of two equally eligible side branches.
    expect_wave_order(8, {side, side, root, wave, root}, {{}, {}, {}, {2}, {3}}, {2, 0, 3, 1, 4});
}

TEST(HbgTaskCatalog, NativeDeferredAndLiveControlGraphsKeepTheGenerationalOrder) {
    const auto lanes = LaneMask::aic();
    std::vector<SchedulerTaskMetadata> tasks{
        spmd(lanes, 3), physical(LaneMask::all()), spmd(lanes, 6), physical(lanes)
    };
    const std::vector<std::vector<int64_t>> fanin{{}, {}, {1}, {2}};
    const std::vector<int64_t> original{1, 0, 2, 3};
    expect_wave_order(8, tasks, fanin, {1, 2, 0, 3});
    expect_wave_order(8, tasks, fanin, original, true);
    expect_wave_order(8, tasks, fanin, original, false, true);
    expect_wave_order(8, tasks, fanin, original, false, false, 1);
    tasks.back().flags |= SCHEDULER_TASK_HAS_PREDICATE;
    expect_wave_order(8, tasks, fanin, original);
}

TEST(HbgTaskCatalog, OrdinaryFaninConfirmsEarlierBranchWithoutChangingDependenciesOrIssueOrder) {
    scheduler_test::BasicGraphBuffer<8> graph(5);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0, {1});
    graph.executable(3, 0);
    graph.executable(4, 0, {2, 3, 3});
    const std::vector<SchedulerTaskMetadata> metadata(5, physical(LaneMask::aic()));
    CatalogImage image(metadata);
    std::string error;
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error)) << error;
    ASSERT_LT(image.plan.task_ranks[3], image.plan.task_ranks[2]);
    const auto task_order = image.plan.task_order;
    const auto catalogs = image.plan.catalogs;
    auto &payload = graph.payload(4);
    auto *const fanin_address = payload.fanin_data();
    std::vector<int32_t> original(fanin_address, fanin_address + payload.fanin_count);

    ASSERT_TRUE(prepare_ordinary(graph)) << error;
    ASSERT_EQ(payload.fanin_count, 3);
    EXPECT_EQ(payload.fanin_data(), fanin_address);
    std::vector<int32_t> reordered(fanin_address, fanin_address + payload.fanin_count);
    EXPECT_EQ(reordered, (std::vector<int32_t>{3, 3, 2}));
    std::sort(original.begin(), original.end());
    std::sort(reordered.begin(), reordered.end());
    EXPECT_EQ(reordered, original);
    // The same dependency graph still yields the same task issuance order.
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error)) << error;
    EXPECT_EQ(image.plan.task_order, task_order);
    EXPECT_EQ(image.plan.catalogs, catalogs);
}

TEST(HbgTaskCatalog, FoldingSingleConsumerControlsPreservesRealCompletionDebt) {
    scheduler_test::BasicGraphBuffer<8> graph(8);
    std::vector<SchedulerTaskMetadata> metadata(8, physical(LaneMask::aic()));
    for (uint32_t ref = 0; ref < metadata.size(); ++ref) {
        graph.executable(ref, 0);
        metadata[ref].kernel_ids[0] = 1;
        metadata[ref].kernel_ids[1] = metadata[ref].kernel_ids[2] = UINT16_MAX;
    }
    graph.executable(2, 0, {0, 1});
    graph.executable(3, 0, {2});
    graph.executable(4, 0, {3, 0});
    graph.executable(5, 0, {4});
    graph.executable(6, 0, {-1});
    std::vector<int64_t> completed{6};
    metadata[6] = {};
    for (const uint32_t ref : {2u, 3u, 5u}) {
        metadata[ref].flags |= SCHEDULER_TASK_INLINE;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            metadata[ref].kernel_ids[lane] = UINT16_MAX;
            graph.storage()[ref].task.kernel_id[lane] = -1;
        }
    }
    CatalogImage image(metadata);
    std::string error;
    SchedulerDependencyPlan dependencies;
    ASSERT_TRUE(prepare_ordinary(graph, metadata, completed, image.plan, dependencies, &error)) << error;
    EXPECT_EQ(completed, (std::vector<int64_t>{2, 3, 5, 6}));
    EXPECT_EQ(dependencies.sinks, (std::vector<uint32_t>{4, 7}));
    ASSERT_EQ(graph.payload(4).fanin_count, 2);
    EXPECT_EQ(
        std::vector<int32_t>(graph.payload(4).fanin_data(), graph.payload(4).fanin_data() + 2),
        (std::vector<int32_t>{0, 1})
    );
    for (const uint32_t ref : {2u, 3u, 5u}) {
        EXPECT_EQ(metadata[ref].flags, 0);
        EXPECT_EQ(metadata[ref].logical_block_num, 0);
        EXPECT_EQ(graph.payload(ref).fanin_count, 0);
    }
    scheduler_test::FixtureStorage storage(8, 3, 1);
    std::copy(metadata.begin(), metadata.end(), storage.metadata);
    storage.prepare_catalog();
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    auto *local = storage.local_context(&storage.contexts[0]);
    auto *base = storage.scheduler_state->base();
    for (const int64_t ref : completed)
        scheduler_task_control_at(base, local, ref)->completion_state = scheduler_completion_completed_state(1);
    scheduler_task_control_at(base, local, 0)->completion_state = scheduler_completion_encode(1, 0, 1);
    scheduler_task_control_at(base, local, 1)->completion_state = scheduler_completion_completed_state(1);
    SchedulerFaninView fanin{};
    ASSERT_EQ(scheduler_node_fanin_view(graph.graph(), 4, &fanin), SchedulerGraphResult::OK);
    uint32_t prefix = 0;
    ASSERT_TRUE(
        scheduler_observe_fanin_prefix(graph.graph(), base, local, storage.run_control, 4, fanin, prefix)
    );
    EXPECT_EQ(prefix, 0u);
    scheduler_task_control_at(base, local, 0)->completion_state = scheduler_completion_completed_state(1);
    ASSERT_TRUE(
        scheduler_observe_fanin_prefix(graph.graph(), base, local, storage.run_control, 4, fanin, prefix)
    );
    EXPECT_EQ(prefix, 2u);
}

TEST(HbgTaskCatalog, SharedAndOversizedControlJoinsRemainLive) {
    for (const bool shared : {true, false}) {
        const uint32_t count = shared ? 5 : 131;
        scheduler_test::BasicGraphBuffer<256> graph(count);
        std::vector<SchedulerTaskMetadata> metadata(count, physical(LaneMask::aic()));
        for (uint32_t ref = 0; ref < count; ++ref)
            graph.executable(ref, 0);
        const uint32_t join = shared ? 2 : 64;
        std::vector<int32_t> parents;
        for (uint32_t ref = 0; ref < join; ++ref)
            parents.push_back(ref);
        graph.executable(join, 0, parents);
        metadata[join].flags |= SCHEDULER_TASK_INLINE;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            metadata[join].kernel_ids[lane] = UINT16_MAX;
            graph.storage()[join].task.kernel_id[lane] = -1;
        }
        if (shared) {
            graph.executable(3, 0, {2});
            graph.executable(4, 0, {2});
        } else {
            std::vector<int32_t> dependencies;
            for (uint32_t ref = join; ref < count - 1; ++ref)
                dependencies.push_back(ref);
            graph.executable(count - 1, 0, dependencies);
        }
        std::vector<int64_t> completed;
        auto *const consumer_fanin = graph.payload(count - 1).fanin_data();
        CatalogImage image(metadata);
        SchedulerDependencyPlan dependencies;
        ASSERT_TRUE(prepare_ordinary(graph, metadata, completed, image.plan, dependencies));
        EXPECT_TRUE(completed.empty());
        EXPECT_TRUE(scheduler_task_is_inline(metadata[join].flags));
        EXPECT_EQ(graph.payload(join).fanin_count, static_cast<int32_t>(parents.size()));
        EXPECT_EQ(graph.payload(count - 1).fanin_data(), consumer_fanin);
    }
}

TEST(HbgTaskCatalog, CompletedNodesDoNotReadStaleIncomingDependenciesDuringOrdering) {
    scheduler_test::BasicGraphBuffer<8> graph(3);
    graph.executable(0, 0, {INT32_MAX});
    graph.executable(1, 0, {0, 0});
    graph.executable(2, 0);
    const std::vector<SchedulerTaskMetadata> metadata(3, physical(LaneMask::aic()));
    CatalogImage image(metadata);
    std::string error;
    EXPECT_FALSE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan, &error));
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {0}, &image.plan, &error)) << error;
    EXPECT_EQ(image.plan.catalogs[0], (std::vector<int64_t>{0, 2, 1}));
    ASSERT_TRUE(prepare_ordinary(graph, {0})) << error;
    EXPECT_EQ(graph.payload(0).fanin_data()[0], INT32_MAX);
    EXPECT_EQ(graph.payload(1).fanin_count, 2);
}

TEST(HbgTaskCatalog, ResidentFaninDropsOnlyPredecessorsProvenByAnotherLiveDependency) {
    scheduler_test::BasicGraphBuffer<8> graph(5);
    graph.executable(0, 0);
    graph.executable(1, 0);
    graph.executable(2, 0, {0});
    graph.executable(3, 0, {0, 2});
    graph.executable(4, 0, {0, 1, 2, 3});
    std::vector<SchedulerTaskMetadata> metadata(5, physical(LaneMask::aic()));
    CatalogImage image(metadata);
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &image.plan));
    const auto catalogs = image.plan.catalogs;
    const auto order = image.plan.task_order;
    std::vector<int64_t> completed;
    SchedulerDependencyPlan dependencies;
    ASSERT_TRUE(prepare_ordinary(graph, metadata, completed, image.plan, dependencies));
    ASSERT_EQ(graph.payload(3).fanin_count, 1);
    EXPECT_EQ(graph.payload(3).fanin_data()[0], 2);
    ASSERT_EQ(graph.payload(4).fanin_count, 2);
    EXPECT_EQ(graph.payload(4).fanin_data()[0], 1);
    EXPECT_EQ(graph.payload(4).fanin_data()[1], 3);
    EXPECT_EQ(image.plan.catalogs, catalogs);
    EXPECT_EQ(image.plan.task_order, order);
}

TEST(HbgTaskCatalog, HostCompletedNodeCannotProveItsHistoricalPredecessorDone) {
    scheduler_test::BasicGraphBuffer<4> graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0, {0, 1});
    ASSERT_TRUE(prepare_ordinary(graph, {1}));
    EXPECT_EQ(graph.payload(2).fanin_count, 2);
}

TEST(HbgTaskCatalog, AliasedFaninRowsKeepEveryOriginalDependency) {
    for (const bool transitive : {false, true}) {
        SCOPED_TRACE(transitive);
        scheduler_test::BasicGraphBuffer<8> graph(5);
        graph.executable(0, 0);
        graph.executable(1, 0, transitive ? std::vector<int32_t>{0} : std::vector<int32_t>{});
        graph.executable(2, 0);
        graph.executable(3, 0, {0, 1, 2});
        graph.executable(4, 0, {1});
        auto *const row = graph.payload(3).fanin_data();
        graph.payload(4).fanin.set(row + 1);

        ASSERT_TRUE(prepare_ordinary(graph));
        ASSERT_EQ(graph.payload(4).fanin_count, 1);
        EXPECT_EQ(graph.payload(4).fanin_data()[0], 1);
        EXPECT_EQ(graph.payload(3).fanin_count, 3);
        EXPECT_EQ(std::vector<int32_t>(row, row + 3), (std::vector<int32_t>{0, 1, 2}));
    }
}

TEST(HbgTaskCatalog, InsufficientAliasedPoolKeepsLiveJoinAndDuplicateDependencies) {
    scheduler_test::BasicGraphBuffer<8> graph(6);
    std::vector<SchedulerTaskMetadata> metadata(6, physical(LaneMask::aic()));
    auto *shared = graph.payload(0).fanin_data();
    shared[0] = 0;
    shared[1] = 0;
    for (uint32_t ref = 0; ref < 6; ++ref) {
        graph.executable(ref, 0);
        if (ref >= 2) {
            graph.payload(ref).fanin.set(shared);
            graph.payload(ref).fanin_count = 2;
        }
    }
    metadata[5].flags |= SCHEDULER_TASK_INLINE;
    for (uint32_t lane = 0; lane < 3; ++lane)
        metadata[5].kernel_ids[lane] = UINT16_MAX;
    CatalogImage image(metadata);
    SchedulerDependencyPlan plan;
    std::vector<int64_t> completed;
    const SchedulerOrdinaryFaninPool pool{shared, 2};
    ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, completed, &image.plan, nullptr, &plan, &pool));
    ASSERT_TRUE(scheduler_prepare_ordinary_fanin(graph.storage(), shared, 2, metadata, completed, image.plan, plan));
    EXPECT_TRUE(completed.empty());
    EXPECT_TRUE(scheduler_task_is_inline(metadata[5].flags));
    EXPECT_EQ(plan.sinks, (std::vector<uint32_t>{1, 2, 3, 4, 5}));
    for (uint32_t ref = 2; ref < 6; ++ref) {
        EXPECT_EQ(plan.fanin[ref], (std::vector<int64_t>{0, 0}));
        EXPECT_EQ(graph.payload(ref).fanin_data(), shared);
    }
}

TEST(HbgTaskCatalog, RejectsFaninPastPoolEndBeforeReadingIt) {
    scheduler_test::BasicGraphBuffer<4> graph(3);
    auto *pool_base = graph.payload(0).fanin_data();
    graph.executable(2, 0, {0, 1});
    graph.payload(2).fanin.set(pool_base + 1);
    const SchedulerOrdinaryFaninPool pool{pool_base, 2};
    SchedulerTaskShape shape;
    std::vector<int64_t> decoded;
    EXPECT_EQ(
        scheduler_classify_host_task(graph.graph(), 2, &shape, pool, &decoded), SchedulerGraphResult::INVALID_FANIN_ID
    );
    EXPECT_TRUE(decoded.empty());
    const SchedulerOrdinaryFaninPool valid_pool{pool_base, 3};
    pool_base[1] = 0;
    pool_base[2] = 0;
    EXPECT_EQ(
        scheduler_classify_host_task(graph.graph(), 2, &shape, valid_pool, &decoded),
        SchedulerGraphResult::INVALID_FANIN_ID
    );
    pool_base[2] = 1;
    EXPECT_EQ(scheduler_classify_host_task(graph.graph(), 2, &shape, valid_pool, &decoded), SchedulerGraphResult::OK);
    EXPECT_EQ(decoded, (std::vector<int64_t>{0, 1}));
    EXPECT_FALSE(pool.contains(reinterpret_cast<const int32_t *>(reinterpret_cast<uintptr_t>(pool_base) + 1), 1));
}

TEST(HbgTaskCatalog, BoundedAncestryRetainsDistantEdgesAndValidatesBeforeMutation) {
    scheduler_test::BasicGraphBuffer<128> graph(66);
    for (int32_t ref = 0; ref < 66; ++ref) {
        graph.executable(ref, 0);
    }
    graph.executable(1, 0, {0});
    graph.executable(64, 0, {0, 1});
    graph.executable(65, 0, {0, 1, 64, 65});
    EXPECT_FALSE(prepare_ordinary(graph));
    EXPECT_EQ(graph.payload(64).fanin_count, 2);
    graph.executable(65, 0, {0, 1, 64});
    ASSERT_TRUE(prepare_ordinary(graph));
    ASSERT_EQ(graph.payload(64).fanin_count, 1);
    EXPECT_EQ(graph.payload(64).fanin_data()[0], 1);
    ASSERT_EQ(graph.payload(65).fanin_count, 2);
    EXPECT_EQ(graph.payload(65).fanin_data()[0], 0);
    EXPECT_EQ(graph.payload(65).fanin_data()[1], 64);
}

TEST(HbgTaskCatalog, CrossEnginePredecessorRequiresIssuedSupplyRatherThanLogicalCompletion) {
    for (const bool sync_start : {false, true}) {
        SCOPED_TRACE(sync_start);
        scheduler_test::BasicGraphBuffer<4> graph(4);
        graph.executable(0, 0);
        graph.executable(1, 0, {0});
        graph.executable(2, 0);
        graph.executable(3, 0, {2});
        std::vector<SchedulerTaskMetadata> metadata{
            physical(LaneMask::aic(), 2), physical(LaneMask::aiv0()), physical(LaneMask::aic(), 2),
            physical(LaneMask::aiv0())
        };
        if (sync_start) {
            metadata[0].flags |= SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_SPMD;
            metadata[2].flags |= SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_SPMD;
        }
        SchedulerTaskCatalogPlan plan;
        scheduler_test::FixtureStorage storage(4, 3, 1, {}, false, [&](AicoreSchedulerLayout &layout) {
            ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 1, layout.total_size, &plan));
            layout.total_size = plan.total_size;
        });
        std::string error;
        ASSERT_TRUE(scheduler_order_task_catalog(graph.graph(), metadata, {}, &plan, &error)) << error;
        void *base = storage.scheduler_state->base();
        std::copy(metadata.begin(), metadata.end(), storage.metadata);
        scheduler_init_task_catalog(base, plan);
        storage.task_catalog_offset = plan.header_offset;
        auto *worker = storage.local_context(&storage.contexts[1]);
        auto *producer = scheduler_task_control_at(base, worker, 0);
        producer->completion_state = scheduler_completion_initial_state(2);
        const auto query = [&]() {
            int64_t ref = -1;
            bool progress = false;
            EXPECT_TRUE(
                scheduler_find_issuable_task(graph.graph(), base, worker, storage.run_control, &ref, &progress)
            );
            return ref;
        };
        EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
        producer->next_block = 2;
        EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
        producer->claim_seal = 2;
        EXPECT_EQ(query(), 1);
        EXPECT_EQ(query(), 1);
        EXPECT_FALSE(scheduler_completion_done(producer->completion_state));
        scheduler_task_control_at(base, worker, 1)->next_block = 1;
        scheduler_task_control_at(base, worker, 1)->claim_seal = 1;
        auto *next_producer = scheduler_task_control_at(base, worker, 2);
        next_producer->completion_state = scheduler_completion_initial_state(2);
        EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
        next_producer->next_block = 2;
        EXPECT_EQ(query(), SCHEDULER_TASK_REF_INVALID);
        next_producer->claim_seal = 2;
        EXPECT_EQ(query(), 3);
        EXPECT_FALSE(scheduler_completion_done(next_producer->completion_state));
        EXPECT_EQ(scheduler_read_error(storage.run_control), 0u);
    }
}

TEST(HbgTaskCatalog, MalformedImageCannotAuthorizeDescriptorAccess) {
    CatalogImage image({physical(LaneMask::aic()), physical(LaneMask::aiv0())});
    auto *header = scheduler_state_at<SchedulerTaskCatalogHeader>(image.base, image.plan.header_offset);
    auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(image.base, header->descriptors_offset);
    ASSERT_TRUE(image.valid());
    header->control_count = header->task_count;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->entry_count = header->task_count + 1;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    descriptors[0].entries_offset = image.plan.header_offset;
    EXPECT_FALSE(image.valid());
    descriptors[0] = image.plan.descriptors[0];
    descriptors[0].capacity = UINT64_MAX;
    EXPECT_FALSE(image.valid());
    descriptors[0] = image.plan.descriptors[0];
    header->first_mix_ref = static_cast<uint32_t>(header->task_count);
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->order_offset = header->descriptors_offset;
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->storage_end_offset = header->order_offset + sizeof(SchedulerTaskCatalogOrder);
    EXPECT_FALSE(image.valid());
    *header = image.plan.header;
    header->storage_end_offset = image.plan.total_size + 64;
    EXPECT_FALSE(image.valid());
}

TEST(HbgTaskCatalog, InvalidTopologyOrOverflowLeavesThePreviousPlanIntact) {
    const std::vector<SchedulerTaskMetadata> metadata{physical(LaneMask::aic())};
    SchedulerTaskCatalogPlan plan;
    ASSERT_TRUE(scheduler_plan_task_catalog(metadata, 1, 128, &plan));
    const auto size = plan.total_size;
    const auto catalogs = plan.catalogs;
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, 0, 128, &plan));
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, SCHEDULER_CLUSTER_CAPACITY + 1, 128, &plan));
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata, 1, UINT64_MAX - 64, &plan));
    EXPECT_EQ(plan.total_size, size);
    EXPECT_EQ(plan.catalogs, catalogs);
}

TEST(HbgHostDependencies, SinksFollowOnlyValidatedLiveEdges) {
    scheduler_test::BasicGraphBuffer<4> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0, 0});
    const std::vector<SchedulerTaskMetadata> metadata(2, physical(LaneMask::aic()));
    SchedulerDependencyPlan dependencies;
    ASSERT_TRUE(scheduler_plan_dependencies(graph.graph(), metadata, {}, dependencies));
    EXPECT_EQ(dependencies.sinks, (std::vector<uint32_t>{1}));
    ASSERT_TRUE(scheduler_plan_dependencies(graph.graph(), metadata, {1}, dependencies));
    EXPECT_EQ(dependencies.sinks, (std::vector<uint32_t>{0}));
    EXPECT_EQ(dependencies.host_done, (std::vector<bool>{false, true}));
    ASSERT_TRUE(scheduler_plan_dependencies(graph.graph(), metadata, {0, 1}, dependencies));
    EXPECT_TRUE(dependencies.sinks.empty());
    EXPECT_FALSE(scheduler_plan_dependencies(graph.graph(), metadata, {2}, dependencies));
    EXPECT_FALSE(scheduler_plan_dependencies(graph.graph(), {metadata[0]}, {}, dependencies));
    for (int32_t producer : {-1, 1, 2, INT32_MAX}) {
        SCOPED_TRACE(producer);
        graph.executable(1, 0, {producer});
        EXPECT_FALSE(scheduler_plan_dependencies(graph.graph(), metadata, {}, dependencies));
    }
}

}  // namespace
