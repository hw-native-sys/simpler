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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "ready_pool_plan.h"
#include "dependency_plan.h"
#include "scheduler/scheduler_completion_resolver.h"
#include "scheduler/scheduler_ready_publication.h"

namespace {
std::vector<SchedulerTaskMetadata> tasks(uint32_t count, uint32_t type) {
    std::vector<SchedulerTaskMetadata> result(count);
    for (auto &task : result) {
        task.flags = SCHEDULER_TASK_EXECUTABLE;
        task.logical_block_num = 1;
        task.active_mask = LaneMask{/*aic=*/type == 0, /*aiv0=*/type != 0, /*aiv1=*/false};
        if (type == SCHEDULER_MIX_QUEUE) {
            task.flags |= SCHEDULER_TASK_MIX;
            task.active_mask = LaneMask{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
        }
    }
    return result;
}

struct PoolFixture {
    AicoreSchedulerLayout layout{};
    SchedulerReadyPoolPlan plan{};
    SchedulerDependencyPlan dependency_plan{};
    std::unique_ptr<scheduler_test::SchedulerStateBuffer> storage;
    scheduler_test::BasicGraphBuffer<256> graph_storage;
    SchedulerGraphView graph{};
    std::vector<SchedulerLocalState> locals;
    SchedulerRunControl *run{nullptr};

    PoolFixture(
        const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters,
        const std::vector<std::vector<int32_t>> &fanins = {}, const std::vector<int64_t> &completed = {}
    ) :
        graph_storage(metadata.size()),
        locals(clusters * 3) {
        EXPECT_TRUE(scheduler_plan_layout(metadata.size(), metadata.size(), 0, &layout));
        EXPECT_TRUE(scheduler_plan_ready_pool(metadata, clusters, layout.total_size, &plan));
        layout.total_size = plan.total_size;
        if (!fanins.empty()) {
            EXPECT_EQ(fanins.size(), metadata.size());
            uint64_t edges = 0;
            for (size_t ref = 0; ref < fanins.size(); ++ref) {
                graph_storage.executable(ref, 0, fanins[ref]);
                edges += fanins[ref].size();
            }
            EXPECT_TRUE(scheduler_plan_dependencies(metadata.size(), edges, layout.total_size, &dependency_plan));
            plan.header.dependencies_offset = dependency_plan.header_offset;
            plan.header.storage_end_offset = dependency_plan.total_size;
            layout.total_size = dependency_plan.total_size;
        }
        storage = std::make_unique<scheduler_test::SchedulerStateBuffer>(layout);
        scheduler_init_ready_pool(base(), plan);
        std::copy(
            metadata.begin(), metadata.end(),
            scheduler_state_at<SchedulerTaskMetadata>(base(), layout.task_metadata_offset)
        );
        run = scheduler_state_at<SchedulerRunControl>(base(), layout.run_control_offset);
        graph = graph_storage.graph();
        graph.scheduler_state_address = reinterpret_cast<uint64_t>(base());
        graph.scheduler_state_bytes = layout.total_size;
        EXPECT_TRUE(scheduler_init_dependencies(base(), dependency_plan, graph, completed));
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &local = locals[worker];
            for (auto &slot : local.slots)
                scheduler_initialize_free_slot(&slot);
            for (auto &lane : local.mix_slots)
                for (auto &slot : lane)
                    scheduler_initialize_free_slot(&slot);
            auto &config = local.config;
            const uint32_t cluster = worker / 3;
            config.graph = &graph;
            config.self_lane = worker % 3;
            config.self_core_type = worker % 3 == 0 ? 0 : 1;
            config.cluster_count = clusters;
            config.scheduler_count = locals.size();
            config.scheduler_index = worker;
            config.runtime_worker_count = locals.size();
            config.ready_pool_offset = plan.header_offset;
            config.task_controls_offset = layout.task_controls_offset;
            config.task_metadata_offset = layout.task_metadata_offset;
            config.cluster_type_ranks[0] = cluster;
            config.cluster_type_ranks[1] = cluster * 2;
            config.cluster_type_ranks[2] = cluster * 2 + 1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                config.worker_ids[lane] = cluster * 3 + lane;
        }
    }
    void *base() const { return storage->base(); }
    bool push(uint32_t producer, uint32_t type, const std::vector<int64_t> &refs) {
        SchedulerReadyBatch batch{};
        for (int64_t ref : refs)
            if (!scheduler_ready_batch_append(base(), &locals[producer], ref, &batch, nullptr)) return false;
        return scheduler_ready_batch_push(base(), &locals[producer], type, &batch, nullptr);
    }
    bool
    pop(uint32_t worker, uint32_t type, uint64_t *cursor, SchedulerReadyClaim *claim,
        SchedulerMixPlacement *placement = nullptr) {
        return scheduler_claim_ready_for_slot(
            graph, base(), &locals[worker], run, locals.size(), type, cursor, nullptr, claim, placement, false
        );
    }
};

TEST(SchedulerReadyPoolTest, SpmdTasksEnterAdmissionOncePerLogicalTask) {
    for (uint32_t type : {0U, 1U}) {
        auto metadata = tasks(2, type);
        metadata[0].flags |= SCHEDULER_TASK_SPMD;
        metadata[0].logical_block_num = 500;
        ASSERT_TRUE(scheduler_task_requires_admission(metadata[0]));
        ASSERT_FALSE(scheduler_task_requires_admission(metadata[1]));
        PoolFixture fixture(metadata, 32);
        ASSERT_TRUE(fixture.push(0, type, {0, 1}));
        uint32_t counts[2] = {};
        for (uint32_t cluster = 0; cluster < 32; ++cluster) {
            uint64_t cursor = 0;
            SchedulerReadyClaim claim{};
            ASSERT_TRUE(fixture.pop(cluster * 3 + type, type, &cursor, &claim));
            if (claim.task_ref == SCHEDULER_TASK_REF_INVALID) continue;
            ASSERT_GE(claim.task_ref, 0);
            ASSERT_LT(claim.task_ref, 2);
            ++counts[claim.task_ref];
        }
        EXPECT_EQ(counts[0], 1u);
        EXPECT_EQ(counts[1], 1u);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[type], type));
    }
}

TEST(SchedulerDependencyTest, DeduplicatesEdgesAndExcludesHostCompletedNodes) {
    PoolFixture fixture(tasks(5, 0), 1, {{}, {0}, {0, 0, 1}, {1, 2}, {2, 3}}, {0, 1});
    const auto *header = scheduler_dependencies(fixture.base(), fixture.plan.header_offset);
    ASSERT_NE(header, nullptr);
    ASSERT_TRUE(
        scheduler_ready_pool_validate(fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 5, 1)
    );
    EXPECT_EQ(header->edge_count, 3);
    auto *remaining = scheduler_state_at<uint64_t>(fixture.base(), header->remaining_offset);
    EXPECT_EQ(std::vector<uint64_t>(remaining, remaining + 5), (std::vector<uint64_t>{0, 0, 0, 1, 2}));
    EXPECT_EQ(
        scheduler_bootstrap_route_task(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 2, nullptr),
        SchedulerRouteResult::READY_TO_ENQUEUE
    );
    EXPECT_EQ(
        scheduler_bootstrap_route_task(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 4, nullptr),
        SchedulerRouteResult::WAITING
    );
    int64_t successor = -1;
    ASSERT_TRUE(scheduler_dependency_successor(fixture.base(), *header, 2, 0, &successor));
    EXPECT_EQ(successor, 3);
    ASSERT_TRUE(scheduler_dependency_successor(fixture.base(), *header, 2, 1, &successor));
    EXPECT_EQ(successor, 4);
    ASSERT_TRUE(scheduler_dependency_successor(fixture.base(), *header, 2, 2, &successor));
    EXPECT_EQ(successor, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_FALSE(scheduler_dependency_successor(fixture.base(), *header, 2, 3, &successor));
}

TEST(SchedulerDependencyTest, ConcurrentProducerResolversPublishJoinExactlyOnce) {
    constexpr uint32_t producers = 32;
    std::vector<std::vector<int32_t>> edges(producers + 1);
    edges.back().resize(producers);
    std::iota(edges.back().begin(), edges.back().end(), 0);
    PoolFixture fixture(tasks(producers + 1, 0), 11, edges);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> failures{0};
    std::vector<std::thread> threads;
    for (uint32_t producer = 0; producer < producers; ++producer) {
        auto *control = scheduler_task_control_at(fixture.base(), &fixture.locals[producer], producer);
        control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
        threads.emplace_back([&, producer] {
            while (!start.load(std::memory_order_acquire)) {}
            SchedulerResolveCursor cursor;
            for (uint32_t step = 0; step < 1000; ++step) {
                auto result = scheduler_resolve_completion_step(
                    fixture.graph, fixture.base(), &fixture.locals[producer], fixture.run, producer, cursor, 1
                );
                if (result.result == SchedulerResolveResult::DONE) return;
                if (result.result == SchedulerResolveResult::ERROR) break;
            }
            ++failures;
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    ASSERT_EQ(failures.load(), 0);
    uint64_t pop_cursor = 0;
    SchedulerReadyClaim claim;
    ASSERT_TRUE(fixture.pop(0, 0, &pop_cursor, &claim));
    EXPECT_EQ(claim.task_ref, producers);
    ASSERT_TRUE(fixture.pop(0, 0, &pop_cursor, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    const auto *header = scheduler_dependencies(fixture.base(), fixture.plan.header_offset);
    EXPECT_EQ(scheduler_state_at<uint64_t>(fixture.base(), header->remaining_offset)[producers], 0);
    SchedulerResolveCursor duplicate;
    EXPECT_EQ(
        scheduler_resolve_completion_step(
            fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, duplicate, 1
        )
            .result,
        SchedulerResolveResult::ERROR
    );
    EXPECT_EQ(scheduler_state_at<uint64_t>(fixture.base(), header->remaining_offset)[producers], 0);
}

TEST(SchedulerDependencyTest, FanoutSurvivesEveryBudgetBoundaryAndOwnerTransfer) {
    constexpr uint32_t successors = 22;
    std::vector<std::vector<int32_t>> edges(successors + 1, {0, 0});
    edges.front().clear();
    PoolFixture fixture(tasks(successors + 1, 0), 3, edges);
    scheduler_task_control_at(fixture.base(), &fixture.locals[0], 0)->state =
        static_cast<int64_t>(SchedulerTaskState::DONE);
    SchedulerResolveCursor cursor;
    bool done = false;
    for (uint32_t step = 0; step < 1000; ++step) {
        SchedulerResolveCursor restored;
        std::memcpy(&restored, &cursor, sizeof(cursor));
        const auto result = scheduler_resolve_completion_step(
            fixture.graph, fixture.base(), &fixture.locals[step % fixture.locals.size()], fixture.run, 0, restored, 1
        );
        cursor = restored;
        ASSERT_NE(result.result, SchedulerResolveResult::ERROR);
        ASSERT_LE(result.work_units, 1);
        if (result.result == SchedulerResolveResult::DONE) {
            done = true;
            break;
        }
    }
    ASSERT_TRUE(done);
    EXPECT_EQ(cursor.visited_count, successors);
    std::vector<bool> seen(successors + 1, false);
    uint64_t pop_cursor = 0;
    for (uint32_t i = 0; i < successors; ++i) {
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &pop_cursor, &claim));
        ASSERT_GT(claim.task_ref, 0);
        ASSERT_LE(claim.task_ref, successors);
        EXPECT_FALSE(seen[claim.task_ref]);
        seen[claim.task_ref] = true;
    }
    SchedulerReadyClaim claim;
    ASSERT_TRUE(fixture.pop(0, 0, &pop_cursor, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerDependencyTest, RejectsMalformedStorageAndOutOfBoundsSuccessors) {
    PoolFixture fixture(tasks(2, 0), 1, {{}, {0}});
    auto *header = scheduler_state_at<SchedulerDependencyHeader>(fixture.base(), fixture.dependency_plan.header_offset);
    const auto valid = *header;
    header->remaining_offset = fixture.dependency_plan.header_offset;
    EXPECT_FALSE(
        scheduler_ready_pool_validate(fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 2, 1)
    );
    *header = valid;
    header->edge_count = header->edge_capacity + 1;
    EXPECT_FALSE(
        scheduler_ready_pool_validate(fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 2, 1)
    );
    *header = valid;
    auto *refs = scheduler_state_at<int32_t>(fixture.base(), header->fanout_refs_offset);
    refs[0] = 2;
    int64_t ref;
    EXPECT_FALSE(scheduler_dependency_successor(fixture.base(), *header, 0, 0, &ref));
    EXPECT_EQ(scheduler_dependency_notify(fixture.base(), *header, 2), SchedulerDependencyResult::ERROR);
    EXPECT_EQ(scheduler_dependency_notify(fixture.base(), *header, 1), SchedulerDependencyResult::READY);
    EXPECT_EQ(scheduler_dependency_notify(fixture.base(), *header, 1), SchedulerDependencyResult::ERROR);
}

TEST(SchedulerReadyPoolTest, TransferablePublicationsCompleteAfterOwnerHandoff) {
    PoolFixture fixture(tasks(24, 0), 9);
    SchedulerReadyPublication publication;
    uint32_t owner = 0;
    auto flush = [&] {
        for (uint32_t attempts = 0; attempts < 1000; ++attempts) {
            SchedulerReadyPublication moved;
            std::memcpy(&moved, &publication, sizeof(moved));
            auto &local = fixture.locals[++owner % fixture.locals.size()];
            const auto result = scheduler_ready_publication_step(fixture.graph, fixture.base(), &local, moved);
            publication = moved;
            EXPECT_NE(result, SchedulerReadyPublishResult::ERROR);
            if (result != SchedulerReadyPublishResult::YIELD) return result;
        }
        return SchedulerReadyPublishResult::ERROR;
    };
    for (int64_t ref = 0; ref < 24; ++ref) {
        auto result =
            scheduler_ready_publication_append(fixture.graph, fixture.base(), &fixture.locals[0], ref, publication);
        if (result == SchedulerReadyAppendResult::FULL) {
            ASSERT_EQ(flush(), SchedulerReadyPublishResult::DONE);
            publication = {};
            result =
                scheduler_ready_publication_append(fixture.graph, fixture.base(), &fixture.locals[0], ref, publication);
        }
        ASSERT_EQ(result, SchedulerReadyAppendResult::ADDED);
    }
    ASSERT_EQ(flush(), SchedulerReadyPublishResult::DONE);
    std::vector<bool> found(24, false);
    uint64_t cursor = 0;
    for (uint32_t count = 0; count < 24; ++count) {
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
        ASSERT_GE(claim.task_ref, 0);
        ASSERT_LT(claim.task_ref, 24);
        EXPECT_FALSE(found[claim.task_ref]);
        found[claim.task_ref] = true;
    }
    SchedulerReadyClaim empty;
    ASSERT_TRUE(fixture.pop(0, 0, &cursor, &empty));
    EXPECT_EQ(empty.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerReadyPoolTest, ConcurrentTransferablePublishersAndConsumersDeliverEachTaskOnce) {
    constexpr uint32_t task_count = 192;
    PoolFixture fixture(tasks(task_count, 1), 12);
    std::vector<std::atomic<uint32_t>> seen(task_count);
    for (auto &value : seen)
        value.store(0);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> consumed{0}, failures{0};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::vector<std::thread> threads;
    for (uint32_t producer = 0; producer < 6; ++producer) {
        threads.emplace_back([&, producer] {
            while (!start.load(std::memory_order_acquire)) {}
            SchedulerReadyPublication publication;
            auto &local = fixture.locals[producer * 3];
            auto flush = [&] {
                while (std::chrono::steady_clock::now() < deadline) {
                    SchedulerReadyPublication restored;
                    std::memcpy(&restored, &publication, sizeof(restored));
                    const auto result =
                        scheduler_ready_publication_step(fixture.graph, fixture.base(), &local, restored);
                    publication = restored;
                    if (result == SchedulerReadyPublishResult::ERROR) return false;
                    if (result == SchedulerReadyPublishResult::DONE) {
                        return scheduler_ready_publication_step(fixture.graph, fixture.base(), &local, publication) ==
                               SchedulerReadyPublishResult::DONE;
                    }
                }
                return false;
            };
            for (uint32_t index = 0; index < 32; ++index) {
                const int64_t ref = producer * 32 + index;
                auto result =
                    scheduler_ready_publication_append(fixture.graph, fixture.base(), &local, ref, publication);
                if (result == SchedulerReadyAppendResult::FULL) {
                    if (!flush()) {
                        ++failures;
                        return;
                    }
                    publication = {};
                    result =
                        scheduler_ready_publication_append(fixture.graph, fixture.base(), &local, ref, publication);
                }
                if (result != SchedulerReadyAppendResult::ADDED) {
                    ++failures;
                    return;
                }
            }
            if (!flush()) ++failures;
        });
    }
    for (uint32_t consumer = 0; consumer < 6; ++consumer) {
        threads.emplace_back([&, consumer] {
            while (!start.load(std::memory_order_acquire)) {}
            uint64_t cursor = 0;
            while (consumed.load() < task_count && failures.load() == 0 &&
                   std::chrono::steady_clock::now() < deadline) {
                SchedulerReadyClaim claim;
                if (!fixture.pop(consumer * 3 + 1, 1, &cursor, &claim)) {
                    ++failures;
                    return;
                }
                if (claim.task_ref == SCHEDULER_TASK_REF_INVALID) continue;
                if (claim.task_ref < 0 || claim.task_ref >= task_count) {
                    ++failures;
                    return;
                }
                if (seen[claim.task_ref].fetch_add(1) != 0) ++failures;
                ++consumed;
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(consumed.load(), task_count);
    for (const auto &value : seen)
        EXPECT_EQ(value.load(), 1u);
}

TEST(SchedulerReadyPoolTest, AllocatesExactlyOneEntryPerExecutableMetadataAcrossGroupBoundaries) {
    auto metadata = tasks(97, 0);
    for (uint32_t ref = 0; ref < metadata.size(); ++ref) {
        if (ref % 5 == 0) metadata[ref].flags = 0;
        else if (ref % 3 == 0) metadata[ref] = tasks(1, 1)[0];
        else if (ref % 3 == 1) metadata[ref] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    }
    for (uint32_t clusters : {1, 7, 8, 9, 11, 32}) {
        PoolFixture fixture(metadata, clusters);
        ASSERT_TRUE(scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, metadata.size(), clusters
        ));
        uint64_t total = 0;
        for (uint32_t queue = 0; queue < fixture.plan.header.queue_count; ++queue) {
            const auto view = scheduler_ready_pool_queue(fixture.base(), fixture.plan.header, queue);
            total += view.capacity;
            EXPECT_EQ(view.control->head, 0u);
            EXPECT_EQ(view.control->reserved_tail, 0u);
            for (uint64_t entry = 0; entry < view.capacity; ++entry)
                EXPECT_EQ(view.entries[entry], SCHEDULER_TASK_REF_INVALID);
        }
        EXPECT_EQ(total, 77u);
        EXPECT_EQ(total, fixture.plan.header.entry_count);
        EXPECT_EQ(fixture.plan.header.group_counts[1], (2 * clusters + 7) / 8);
    }
}

TEST(SchedulerReadyPoolTest, AnyEligibleWorkerClaimsOrdinaryWorkAcrossFormerShardBoundaries) {
    for (uint32_t type : {0U, 1U, SCHEDULER_MIX_QUEUE}) {
        PoolFixture fixture(tasks(1, type), 12);
        ASSERT_TRUE(fixture.push(0, type, {0}));
        const uint32_t worker = type == 1 ? 34 : 33;
        uint64_t cursor = 0;
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(worker, type, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, 0);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[worker], type));
        ASSERT_TRUE(fixture.pop(worker, type, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerReadyPoolTest, TypeEligibilityDoesNotDependOnWhichWorkerPublishes) {
    auto metadata = tasks(3, 0);
    metadata[1] = tasks(1, 1)[0];
    metadata[2] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    PoolFixture fixture(metadata, 9);
    for (uint32_t type = 0; type < 3; ++type)
        ASSERT_TRUE(fixture.push(1, type, {type}));
    SchedulerReadyClaim claim{};
    uint64_t cursor = 0;
    EXPECT_FALSE(fixture.pop(25, 0, &cursor, &claim));
    EXPECT_FALSE(fixture.pop(25, SCHEDULER_MIX_QUEUE, &cursor, &claim));
    EXPECT_FALSE(fixture.pop(24, 1, &cursor, &claim));
    ASSERT_TRUE(fixture.pop(25, 1, &cursor, &claim));
    EXPECT_EQ(claim.task_ref, 1);
    ASSERT_TRUE(fixture.pop(24, 0, &cursor, &claim));
    EXPECT_EQ(claim.task_ref, 0);
    ASSERT_TRUE(fixture.pop(24, SCHEDULER_MIX_QUEUE, &cursor, &claim));
    EXPECT_EQ(claim.task_ref, 2);
}

TEST(SchedulerReadyPoolTest, RemoteTasksProgressWhileTheLocalGroupRemainsNonempty) {
    PoolFixture fixture(tasks(128, 0), 24);
    std::vector<int64_t> own, remote;
    for (int64_t ref = 0; ref < 128; ++ref) {
        const auto group = scheduler_ready_pool_ref_group(ref, 3);
        if (group == 0) own.push_back(ref);
        else if (group == 1) remote.push_back(ref);
    }
    ASSERT_GT(own.size(), 8u);
    ASSERT_FALSE(remote.empty());
    ASSERT_TRUE(fixture.push(0, 0, own));
    ASSERT_TRUE(fixture.push(0, 0, {remote[0]}));
    uint64_t cursor = 1;
    bool remote_claimed = false;
    for (uint32_t poll = 0; poll < 4; ++poll) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
        remote_claimed = remote_claimed || claim.task_ref == remote[0];
    }
    EXPECT_TRUE(remote_claimed);
    EXPECT_TRUE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[0], 0));
}

TEST(SchedulerReadyPoolTest, RemotePriorityRotatesAcrossContinuouslyNonemptyGroups) {
    for (uint64_t initial_cursor : {0, 1}) {
        PoolFixture fixture(tasks(256, 0), 24);
        const uint32_t groups = fixture.plan.header.group_counts[0];
        std::vector<std::vector<int64_t>> refs(groups);
        for (int64_t ref = 0; ref < 256; ++ref)
            refs[scheduler_ready_pool_ref_group(ref, groups)].push_back(ref);
        ASSERT_GT(refs[0].size(), 6 * (groups - 1));
        for (const auto &group_refs : refs) {
            ASSERT_GT(group_refs.size(), 2u);
            ASSERT_TRUE(fixture.push(0, 0, group_refs));
        }
        uint64_t cursor = initial_cursor;
        for (uint32_t cycle = 0; cycle < 2; ++cycle) {
            std::vector<uint32_t> claims(groups);
            for (uint32_t poll = 0; poll < 4 * (groups - 1); ++poll) {
                SchedulerReadyClaim claim{};
                ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
                ASSERT_GE(claim.task_ref, 0);
                ++claims[scheduler_ready_pool_ref_group(claim.task_ref, groups)];
            }
            for (uint32_t group = 1; group < groups; ++group)
                EXPECT_EQ(claims[group], 1u);
        }
    }
}

TEST(SchedulerReadyPoolTest, UnplaceableMixRemainsSharedUntilAnotherCoordinatorClaimsIt) {
    PoolFixture fixture(tasks(1, SCHEDULER_MIX_QUEUE), 9);
    ASSERT_TRUE(fixture.push(1, SCHEDULER_MIX_QUEUE, {0}));
    for (auto &slot : fixture.locals[0].slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    SchedulerMixPlacement placement{};
    ASSERT_TRUE(fixture.pop(0, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(fixture.pop(24, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(placement.count, 2u);
}

TEST(SchedulerReadyPoolTest, ConcurrentProducersAndRemoteConsumersNeverLoseOrDuplicateTasks) {
    PoolFixture fixture(tasks(192, 1), 12);
    std::vector<std::atomic<uint32_t>> seen(192);
    for (auto &entry : seen)
        entry.store(0);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> consumed{0}, failures{0};
    std::vector<std::thread> threads;
    for (uint32_t producer = 0; producer < 6; ++producer) {
        threads.emplace_back([&, producer] {
            while (!start.load()) {}
            std::vector<int64_t> refs(32);
            std::iota(refs.begin(), refs.end(), producer * 32);
            if (!fixture.push(producer * 3, 1, refs)) ++failures;
        });
    }
    for (uint32_t consumer = 0; consumer < 12; ++consumer) {
        threads.emplace_back([&, consumer] {
            while (!start.load()) {}
            uint64_t cursor = consumer;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (consumed.load() < 192 && failures.load() == 0) {
                SchedulerReadyClaim claim{};
                if (!fixture.pop(consumer * 3 + 1, 1, &cursor, &claim)) {
                    ++failures;
                    break;
                }
                if (claim.task_ref < 0) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        ++failures;
                        break;
                    }
                    continue;
                }
                if (claim.task_ref >= 192 || ++seen[claim.task_ref] != 1) {
                    ++failures;
                    break;
                }
                ++consumed;
            }
        });
    }
    start.store(true);
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(consumed.load(), 192u);
    for (const auto &entry : seen)
        EXPECT_EQ(entry.load(), 1u);
}

TEST(SchedulerReadyPoolTest, RejectsCorruptDescriptorsBeforeAnyQueueAccess) {
    PoolFixture fixture(tasks(12, 0), 9);
    auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.plan.header_offset);
    auto *descriptors = scheduler_state_at<SchedulerReadyPoolDescriptor>(fixture.base(), pool->descriptors_offset);
    const auto valid = [&] {
        return scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 12, 9
        );
    };
    ASSERT_TRUE(valid());
    for (uint32_t width : {0U, 4U}) {
        pool->group_width = width;
        EXPECT_FALSE(valid());
    }
    *pool = fixture.plan.header;
    pool->version = 3;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->queue_count = UINT32_MAX;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    descriptors[0].entries_offset = pool->controls_offset;
    EXPECT_FALSE(valid());
    descriptors[0] = fixture.plan.descriptors[0];
    descriptors[0].capacity = UINT64_MAX;
    EXPECT_FALSE(valid());
    descriptors[0] = fixture.plan.descriptors[0];
    pool->storage_end_offset = fixture.layout.total_size + 128;
    EXPECT_FALSE(valid());
}

TEST(SchedulerReadyPoolTest, RejectsInvalidTopologyAndOverflowWithoutChangingThePreviousPlan) {
    SchedulerReadyPoolPlan plan{};
    ASSERT_TRUE(scheduler_plan_ready_pool(tasks(1, 0), 1, 128, &plan));
    const uint64_t size = plan.total_size;
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 0, 128, &plan));
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), SCHEDULER_CLUSTER_CAPACITY + 1, 128, &plan));
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 1, UINT64_MAX - 64, &plan));
    EXPECT_EQ(plan.total_size, size);
}

}  // namespace
