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
    }
    return result;
}

std::vector<SchedulerTaskMetadata> mixed_tasks(uint32_t count) {
    auto result = tasks(count, 0);
    for (auto &task : result) {
        task.flags |= SCHEDULER_TASK_MIX;
        task.active_mask = LaneMask{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
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

            for (uint32_t lane = 0; lane < 3; ++lane)
                config.worker_ids[lane] = cluster * 3 + lane;
        }
    }
    void *base() const { return storage->base(); }
    bool push(uint32_t producer, uint32_t type, const std::vector<int64_t> &refs) {
        SchedulerReadyBatch batch{};
        for (int64_t ref : refs)
            if (!scheduler_ready_batch_append(base(), &locals[producer], ref, &batch)) return false;
        return scheduler_ready_batch_push(base(), &locals[producer], type, &batch);
    }
    bool publish(uint32_t producer, int64_t ref) {
        SchedulerReadyPublication publication{};
        if (scheduler_ready_publication_append(graph, base(), &locals[producer], ref, publication) !=
            SchedulerReadyAppendResult::ADDED)
            return false;
        for (uint32_t step = 0; step < 8; ++step) {
            const auto result = scheduler_ready_publication_step(graph, base(), &locals[producer], publication);
            if (result == SchedulerReadyPublishResult::DONE) return true;
            if (result == SchedulerReadyPublishResult::ERROR) return false;
        }
        return false;
    }
    bool pop(uint32_t worker, uint32_t type, SchedulerReadyClaim *claim) {
        return scheduler_claim_ready_for_slot(graph, base(), &locals[worker], run, type, claim);
    }
};

TEST(SchedulerReadyPoolTest, TwoQueuesShareAicAndMixWithoutChangingExecutionMetadata) {
    auto metadata = tasks(3, 0);
    metadata[1].flags |= SCHEDULER_TASK_MIX;
    metadata[1].active_mask = LaneMask{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
    metadata[2].flags |= SCHEDULER_TASK_MIX;
    metadata[2].active_mask = LaneMask{/*aic=*/false, /*aiv0=*/true, /*aiv1=*/true};
    PoolFixture fixture(metadata, 32);
    for (int64_t ref : {0, 1, 2})
        ASSERT_TRUE(fixture.publish(1, ref));

    for (int64_t expected : {0, 1, 2}) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(93, 0, &claim));
        ASSERT_EQ(claim.task_ref, expected);
        const auto *observed = scheduler_task_metadata_at(fixture.base(), &fixture.locals[93], claim.task_ref);
        EXPECT_EQ(observed->active_mask.raw(), metadata[expected].active_mask.raw());
        EXPECT_EQ(observed->flags, metadata[expected].flags);
    }
    EXPECT_EQ(fixture.plan.descriptors.size(), 2u);
    EXPECT_EQ(fixture.plan.descriptors[0].capacity, 3u);
    EXPECT_EQ(fixture.plan.descriptors[1].capacity, 0u);
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[93], 0));
}

TEST(SchedulerReadyPoolTest, TwoQueuesRouteCrossTypeProducersAndRejectIneligibleConsumers) {
    auto metadata = tasks(3, 0);
    metadata[1].flags |= SCHEDULER_TASK_MIX;
    metadata[1].active_mask = LaneMask{/*aic=*/false, /*aiv0=*/true, /*aiv1=*/true};
    metadata[2] = tasks(1, 1)[0];
    PoolFixture fixture(metadata, 9);
    ASSERT_TRUE(fixture.publish(1, 0));
    ASSERT_TRUE(fixture.publish(2, 1));
    ASSERT_TRUE(fixture.publish(0, 2));

    SchedulerReadyClaim claim{};
    EXPECT_FALSE(fixture.pop(25, 0, &claim));
    EXPECT_FALSE(fixture.pop(24, 1, &claim));
    ASSERT_TRUE(fixture.pop(25, 1, &claim));
    ASSERT_EQ(claim.task_ref, 2);
    ASSERT_TRUE(fixture.pop(25, 1, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    for (int64_t expected : {0, 1}) {
        ASSERT_TRUE(fixture.pop(24, 0, &claim));
        ASSERT_EQ(claim.task_ref, expected);
    }
    EXPECT_EQ(fixture.plan.descriptors.size(), 2u);
}

TEST(SchedulerReadyPoolTest, TwoQueuesEveryWorkerCanClaimItsOwnClassBeyondFormerGroupRanks) {
    for (uint32_t worker = 0; worker < 96; ++worker) {
        SCOPED_TRACE(worker);
        const uint32_t type = worker % 3 == 0 ? 0 : 1;
        auto metadata = tasks(1, type);
        if (type == 0 && (worker / 3) % 2 != 0) {
            metadata[0].flags |= SCHEDULER_TASK_MIX;
            metadata[0].active_mask = LaneMask{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
        }
        PoolFixture fixture(metadata, 32);
        ASSERT_TRUE(fixture.publish(type == 0 ? 1 : 0, 0));
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(worker, type, &claim));
        ASSERT_EQ(claim.task_ref, 0);
        ASSERT_TRUE(fixture.pop(worker, type, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerReadyPoolTest, TwoQueuesKeepZeroCapacityClassAccessibleWithoutClaimingWork) {
    for (uint32_t clusters : {1, 9, 32}) {
        for (uint32_t type : {0U, 1U}) {
            SCOPED_TRACE(clusters);
            SCOPED_TRACE(type);
            PoolFixture fixture(tasks(1, type), clusters);
            const uint32_t empty_type = 1U - type;
            const uint32_t empty_worker = (clusters - 1) * 3 + empty_type;
            SchedulerReadyClaim claim{};
            EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[empty_worker], empty_type));
            ASSERT_TRUE(fixture.pop(empty_worker, empty_type, &claim));
            EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);

            ASSERT_TRUE(fixture.publish(empty_worker, 0));
            const uint32_t worker = (clusters - 1) * 3 + type;
            ASSERT_TRUE(fixture.pop(worker, type, &claim));
            ASSERT_EQ(claim.task_ref, 0);
            EXPECT_EQ(fixture.plan.descriptors.size(), 2u);
            EXPECT_EQ(fixture.plan.descriptors[empty_type].capacity, 0u);
            EXPECT_EQ(fixture.plan.header.entry_count, 1u);
        }
    }
}

TEST(SchedulerReadyPoolTest, EmptyGraphStillHasTwoEmptyQueues) {
    PoolFixture fixture({}, 32);
    ASSERT_EQ(fixture.plan.descriptors.size(), 2u);
    EXPECT_EQ(fixture.plan.header.entry_count, 0u);
    for (uint32_t type : {0U, 1U}) {
        EXPECT_EQ(fixture.plan.descriptors[type].capacity, 0u);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[type], type));
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(type, type, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerReadyPoolTest, ReservedHeadHoleDoesNotExposeLaterPublishedTask) {
    PoolFixture fixture(tasks(2, 0), 32);
    const auto queue = scheduler_ready_pool_queue(fixture.base(), fixture.plan.header, 0);
    SchedulerTicketRange first{}, second{};
    ASSERT_EQ(scheduler_ticket_reserve(queue, 1, &first), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_reserve(queue, 1, &second), SchedulerTicketResult::SUCCESS);
    const int64_t second_ref = 1;
    ASSERT_EQ(scheduler_ticket_publish(queue, second, &second_ref), SchedulerTicketResult::SUCCESS);
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[93], 0));
    ASSERT_TRUE(fixture.pop(93, 0, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(queue.control->head, 0u);

    const int64_t first_ref = 0;
    ASSERT_EQ(scheduler_ticket_publish(queue, first, &first_ref), SchedulerTicketResult::SUCCESS);
    for (int64_t ref : {0, 1}) {
        EXPECT_TRUE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[93], 0));
        ASSERT_TRUE(fixture.pop(93, 0, &claim));
        EXPECT_EQ(claim.task_ref, ref);
    }
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[93], 0));
}

TEST(SchedulerReadyPoolTest, WrongClassEntryFailsBeforeAdvancingHead) {
    for (uint32_t type : {0U, 1U}) {
        SCOPED_TRACE(type);
        auto metadata = tasks(2, 0);
        metadata[1] = tasks(1, 1)[0];
        PoolFixture fixture(metadata, 32);
        const auto queue = scheduler_ready_pool_queue(fixture.base(), fixture.plan.header, type);
        SchedulerTicketRange range{};
        ASSERT_EQ(scheduler_ticket_reserve(queue, 1, &range), SchedulerTicketResult::SUCCESS);
        const int64_t wrong_ref = 1U - type;
        ASSERT_EQ(scheduler_ticket_publish(queue, range, &wrong_ref), SchedulerTicketResult::SUCCESS);
        SchedulerReadyClaim claim{};
        EXPECT_FALSE(fixture.pop(type, type, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(queue.control->head, 0u);
        EXPECT_NE(scheduler_read_error(fixture.run), 0);
    }
}

TEST(SchedulerReadyPoolTest, InvalidReadyClassAndWorkerTypeCannotClaimTask) {
    PoolFixture fixture(tasks(1, 0), 1);
    ASSERT_TRUE(fixture.publish(1, 0));
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(fixture.pop(0, 2, &claim));
    fixture.locals[0].config.self_core_type = 2;
    EXPECT_FALSE(fixture.pop(0, 0, &claim));
    fixture.locals[0].config.self_core_type = 0;
    ASSERT_TRUE(fixture.pop(0, 0, &claim));
    EXPECT_EQ(claim.task_ref, 0);
}

TEST(SchedulerReadyPoolTest, InvalidSharedReferenceCannotAcquireATicket) {
    for (int64_t ref : {INT64_C(-2), INT64_C(1), INT64_MAX}) {
        SCOPED_TRACE(ref);
        PoolFixture fixture(tasks(1, 0), 1);
        ASSERT_TRUE(fixture.push(0, 0, {0}));
        const auto queue = scheduler_ready_pool_queue(fixture.base(), fixture.plan.header, 0);
        scheduler_gm_store(queue.entries[0].task_ref, ref);
        SchedulerReadyClaim claim;
        EXPECT_FALSE(fixture.pop(0, 0, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_gm_fetch_add(queue.control->head, UINT64_C(0)), 0u);
    }
}

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
            SchedulerReadyClaim claim{};
            ASSERT_TRUE(fixture.pop(cluster * 3 + type, type, &claim));
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

TEST(SchedulerReadyPool, PlanSeparatesQueueCursorsEvenAfterA64BytePrefix) {
    SchedulerReadyPoolPlan plan;
    ASSERT_TRUE(scheduler_plan_ready_pool(tasks(2, 0), 1, 64, &plan));
    EXPECT_EQ(plan.header.controls_offset % alignof(SchedulerTicketQueue), 0u);
    for (const auto &descriptor : plan.descriptors)
        EXPECT_EQ(descriptor.entries_offset % alignof(SchedulerTicketEntry), 0u);
    EXPECT_NE(
        offsetof(SchedulerTicketQueue, head) / SCHEDULER_GM_ATOMIC_SECTOR_BYTES,
        offsetof(SchedulerTicketQueue, reserved_tail) / SCHEDULER_GM_ATOMIC_SECTOR_BYTES
    );
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
        scheduler_bootstrap_route_task(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 2),
        SchedulerRouteResult::READY_TO_ENQUEUE
    );
    EXPECT_EQ(
        scheduler_bootstrap_route_task(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 4),
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

TEST(SchedulerDependencyTest, HostConstructionRejectsInvalidFaninReferences) {
    for (int32_t producer : {-1, 1, 2, INT32_MAX}) {
        SCOPED_TRACE(producer);
        PoolFixture fixture(tasks(2, 0), 1, {{}, {0}});
        fixture.graph_storage.executable(1, 0, {producer});
        EXPECT_FALSE(scheduler_init_dependencies(fixture.base(), fixture.dependency_plan, fixture.graph, {}));
    }
}

TEST(SchedulerDependencyTest, ConcurrentProducerResolversPublishJoinExactlyOnce) {
    constexpr uint32_t producers = 32;
    for (uint32_t direct_mode : {0u, 1u, 2u}) {
        SCOPED_TRACE(direct_mode);
        std::vector<std::vector<int32_t>> edges(2 * producers + 1);
        edges[producers].resize(producers);
        std::iota(edges[producers].begin(), edges[producers].end(), 0);
        PoolFixture fixture(tasks(2 * producers + 1, 0), producers, edges);
        std::array<SchedulerReadyClaim, producers> direct{};
        std::atomic<bool> start{false};
        std::atomic<uint32_t> failures{0};
        std::vector<std::thread> threads;
        for (uint32_t producer = 0; producer < producers; ++producer) {
            auto *local = &fixture.locals[producer * 3];
            auto *control = scheduler_task_control_at(fixture.base(), local, producer);
            control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
            if (direct_mode == 2) direct[producer].task_ref = producers + 1 + producer;
            threads.emplace_back([&, producer, local] {
                while (!start.load(std::memory_order_acquire)) {}
                if (!scheduler_resolve_completion(
                        fixture.graph, fixture.base(), local, fixture.run, producer, 0, false,
                        direct_mode == 0 ? nullptr : &direct[producer], 0
                    ))
                    ++failures;
            });
        }
        start.store(true, std::memory_order_release);
        for (auto &thread : threads)
            thread.join();
        ASSERT_EQ(failures.load(), 0);
        uint32_t owners = 0;
        for (uint32_t producer = 0; producer < producers; ++producer) {
            owners += direct[producer].task_ref == producers;
            if (direct_mode == 2) EXPECT_EQ(direct[producer].task_ref, producers + 1 + producer);
        }
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &claim));
        if (claim.task_ref != SCHEDULER_TASK_REF_INVALID) {
            EXPECT_EQ(claim.task_ref, producers);
            ++owners;
        }
        EXPECT_EQ(owners, 1u);
        EXPECT_EQ(direct_mode == 1, claim.task_ref == SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.pop(0, 0, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
        const auto *header = scheduler_dependencies(fixture.base(), fixture.plan.header_offset);
        EXPECT_EQ(scheduler_state_at<uint64_t>(fixture.base(), header->remaining_offset)[producers], 0);
    }
}

TEST(SchedulerDependencyTest, SynchronousFanoutPublishesEveryDeduplicatedSuccessor) {
    constexpr uint32_t successors = 22;
    std::vector<std::vector<int32_t>> edges(successors + 1, {0, 0});
    edges.front().clear();
    PoolFixture fixture(tasks(successors + 1, 0), 3, edges);
    scheduler_task_control_at(fixture.base(), &fixture.locals[0], 0)->state =
        static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, 0, true, nullptr, UINT32_MAX
    ));
    std::vector<bool> seen(successors + 1, false);
    for (uint32_t i = 0; i < successors; ++i) {
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &claim));
        ASSERT_GT(claim.task_ref, 0);
        ASSERT_LE(claim.task_ref, successors);
        EXPECT_FALSE(seen[claim.task_ref]);
        seen[claim.task_ref] = true;
    }
    SchedulerReadyClaim claim;
    ASSERT_TRUE(fixture.pop(0, 0, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerDependencyTest, ImmediateCounterResolutionDoesNotUseWakeLinks) {
    PoolFixture fixture(mixed_tasks(2), 2, {{}, {0}});
    auto *control = scheduler_task_control_at(fixture.base(), &fixture.locals[0], 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, 0, false, nullptr, 0
    ));
    EXPECT_EQ(control->wake_list_head, SCHEDULER_WAKE_LIST_OPEN);
    SchedulerReadyClaim claim;
    ASSERT_TRUE(fixture.pop(0, 0, &claim));
    EXPECT_EQ(claim.task_ref, 1);
    ASSERT_TRUE(fixture.pop(0, 0, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerReadyPoolTest, DirectOwnershipLeavesNoQueuedCopyAndPreservesTheExistingOwner) {
    PoolFixture fixture(tasks(2, 0), 1);
    auto *local = &fixture.locals[0];
    SchedulerReadyPublication publication{};
    SchedulerReadyClaim direct;
    ASSERT_EQ(
        scheduler_ready_publication_append(fixture.graph, fixture.base(), local, 0, publication, &direct, 0),
        SchedulerReadyAppendResult::DIRECT
    );
    EXPECT_EQ(direct.task_ref, 0);
    EXPECT_EQ(direct.source, SchedulerReadySource::DIRECT_RESOLVE);
    EXPECT_EQ(publication.count, 0u);
    EXPECT_EQ(publication.destination, SchedulerReadyDestination::INVALID);
    ASSERT_EQ(
        scheduler_ready_publication_append(fixture.graph, fixture.base(), local, 1, publication, &direct, 0),
        SchedulerReadyAppendResult::ADDED
    );
    EXPECT_EQ(direct.task_ref, 0);
    ASSERT_EQ(publication.count, 1u);
    ASSERT_EQ(
        scheduler_ready_publication_step(fixture.graph, fixture.base(), local, publication),
        SchedulerReadyPublishResult::DONE
    );
    SchedulerReadyClaim queued;
    ASSERT_TRUE(fixture.pop(0, 0, &queued));
    EXPECT_EQ(queued.task_ref, 1);
    ASSERT_TRUE(fixture.pop(0, 0, &queued));
    EXPECT_EQ(queued.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerReadyPoolTest, DirectOwnershipRejectsNonExecutableMetadata) {
    PoolFixture fixture(tasks(1, 0), 1);
    auto *local = &fixture.locals[0];
    SchedulerReadyPublication publication{};
    SchedulerReadyClaim direct;
    auto *metadata = scheduler_task_metadata_at(fixture.base(), local, 0);
    metadata->flags = 0;
    EXPECT_EQ(
        scheduler_ready_publication_append(fixture.graph, fixture.base(), local, 0, publication, &direct, 0),
        SchedulerReadyAppendResult::ERROR
    );
    EXPECT_EQ(direct.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(publication.count, 0u);
}

TEST(SchedulerDependencyTest, ImmediateFanoutRetainsOneDirectAndPublishesEveryOtherSuccessor) {
    constexpr uint32_t successors = 23;
    auto metadata = tasks(successors + 1, 0);
    for (uint32_t ref = 1; ref <= successors; ++ref)
        metadata[ref] = ref % 3 == 2 ? mixed_tasks(1)[0] : tasks(1, ref % 3)[0];
    metadata[1] = tasks(1, 0)[0];
    std::vector<std::vector<int32_t>> fanins(successors + 1, {0});
    fanins.front().clear();
    PoolFixture fixture(metadata, 17, fanins);
    auto *control = scheduler_task_control_at(fixture.base(), &fixture.locals[0], 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    SchedulerReadyClaim direct;
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, 0, false, &direct, 0
    ));
    ASSERT_EQ(direct.task_ref, 1);
    EXPECT_EQ(direct.source, SchedulerReadySource::DIRECT_RESOLVE);
    std::vector<bool> seen(successors + 1, false);
    seen[direct.task_ref] = true;
    for (uint32_t type = 0; type < 2; ++type) {
        for (;;) {
            SchedulerReadyClaim claim;
            ASSERT_TRUE(fixture.pop(type == 1 ? 1 : 0, type, &claim));
            if (claim.task_ref == SCHEDULER_TASK_REF_INVALID) break;
            ASSERT_GT(claim.task_ref, 0);
            ASSERT_LE(claim.task_ref, successors);
            EXPECT_FALSE(seen[claim.task_ref]);
            seen[claim.task_ref] = true;
            EXPECT_EQ(
                scheduler_task_ready_queue(metadata[claim.task_ref].flags, metadata[claim.task_ref].active_mask), type
            );
        }
    }
    for (uint32_t ref = 1; ref <= successors; ++ref)
        EXPECT_TRUE(seen[ref]);
}

TEST(SchedulerDependencyTest, OrderedVectorResolverRetainsMixForAdmission) {
    PoolFixture fixture(mixed_tasks(2), 2, {{}, {0}});
    auto *local = &fixture.locals[1];
    SchedulerAdmissionLocalState admission{};
    admission.initialized = true;
    local->admission = &admission;
    auto *control = scheduler_task_control_at(fixture.base(), local, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    SchedulerReadyClaim direct;
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph, fixture.base(), local, fixture.run, 0, 0, false, &direct, local->config.self_core_type
    ));
    EXPECT_EQ(direct.task_ref, 1);
    EXPECT_EQ(direct.source, SchedulerReadySource::DIRECT_RESOLVE);
    SchedulerReadyClaim queued;
    ASSERT_TRUE(fixture.pop(0, 0, &queued));
    EXPECT_EQ(queued.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerDependencyTest, ImmediateMixedFanoutPublishesEverySuccessorInAicQueue) {
    PoolFixture fixture(mixed_tasks(4), 2, {{}, {0}, {0}, {0}});
    auto *control = scheduler_task_control_at(fixture.base(), &fixture.locals[0], 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, 0, false, nullptr, 0
    ));
    for (int64_t ref = 1; ref < 4; ++ref) {
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &claim));
        EXPECT_EQ(claim.task_ref, ref);
    }
    SchedulerReadyClaim empty;
    ASSERT_TRUE(fixture.pop(0, 0, &empty));
    EXPECT_EQ(empty.task_ref, SCHEDULER_TASK_REF_INVALID);
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
    for (uint32_t count = 0; count < 24; ++count) {
        SchedulerReadyClaim claim;
        ASSERT_TRUE(fixture.pop(0, 0, &claim));
        ASSERT_GE(claim.task_ref, 0);
        ASSERT_LT(claim.task_ref, 24);
        EXPECT_FALSE(found[claim.task_ref]);
        found[claim.task_ref] = true;
    }
    SchedulerReadyClaim empty;
    ASSERT_TRUE(fixture.pop(0, 0, &empty));
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
            while (consumed.load() < task_count && failures.load() == 0 &&
                   std::chrono::steady_clock::now() < deadline) {
                SchedulerReadyClaim claim;
                if (!fixture.pop(consumer * 3 + 1, 1, &claim)) {
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

TEST(SchedulerReadyPoolTest, AllocatesTwoQueuesAndOneEntryPerExecutableMetadata) {
    auto metadata = tasks(97, 0);
    for (uint32_t ref = 0; ref < metadata.size(); ++ref) {
        if (ref % 5 == 0) metadata[ref].flags = 0;
        else if (ref % 3 == 0) metadata[ref] = tasks(1, 1)[0];
        else if (ref % 3 == 1) metadata[ref] = mixed_tasks(1)[0];
    }
    for (uint32_t clusters : {1, 7, 8, 9, 11, 32}) {
        PoolFixture fixture(metadata, clusters);
        ASSERT_TRUE(scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, metadata.size(), clusters
        ));
        uint64_t total = 0;
        for (uint32_t queue = 0; queue < fixture.plan.descriptors.size(); ++queue) {
            const auto view = scheduler_ready_pool_queue(fixture.base(), fixture.plan.header, queue);
            total += view.capacity;
            EXPECT_EQ(view.control->head, 0u);
            EXPECT_EQ(view.control->reserved_tail, 0u);
            for (uint64_t entry = 0; entry < view.capacity; ++entry)
                EXPECT_EQ(view.entries[entry].task_ref, SCHEDULER_TASK_REF_INVALID);
        }
        EXPECT_EQ(total, 77u);
        EXPECT_EQ(total, fixture.plan.header.entry_count);
        ASSERT_EQ(fixture.plan.descriptors.size(), 2u);
        EXPECT_EQ(fixture.plan.descriptors[0].capacity, 51u);
        EXPECT_EQ(fixture.plan.descriptors[1].capacity, 26u);
    }
}

TEST(SchedulerReadyPoolTest, AnyEligibleWorkerClaimsWorkBeyondFormerShardBoundaries) {
    for (uint32_t kind : {0U, 1U, 2U}) {
        const uint32_t queue = kind == 1 ? 1 : 0;
        PoolFixture fixture(kind == 2 ? mixed_tasks(1) : tasks(1, queue), 12);
        ASSERT_TRUE(fixture.publish(0, 0));
        const uint32_t worker = queue == 1 ? 34 : 33;
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(worker, queue, &claim));
        EXPECT_EQ(claim.task_ref, 0);
        EXPECT_EQ(claim.source, SchedulerReadySource::POOL);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[worker], queue));
        ASSERT_TRUE(fixture.pop(worker, queue, &claim));
        EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerReadyPoolTest, MixClaimDoesNotWaitForAivPlacement) {
    PoolFixture fixture(mixed_tasks(1), 9);
    ASSERT_TRUE(fixture.publish(1, 0));
    for (auto &slot : fixture.locals[0].slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    for (uint32_t worker : {1u, 2u})
        for (auto &slot : fixture.locals[worker].slots)
            slot.state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(fixture.pop(0, 0, &claim));
    EXPECT_EQ(claim.task_ref, 0);
    ASSERT_TRUE(fixture.pop(24, 0, &claim));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerReadyPoolTest, ConcurrentProducersAndConsumersNeverLoseOrDuplicateTasks) {
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
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (consumed.load() < 192 && failures.load() == 0) {
                SchedulerReadyClaim claim{};
                if (!fixture.pop(consumer * 3 + 1, 1, &claim)) {
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
    pool->version = 3;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->entry_count = pool->task_count + 1;
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
