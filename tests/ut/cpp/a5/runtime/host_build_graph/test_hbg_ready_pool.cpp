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
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "ready_pool_plan.h"

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
    std::unique_ptr<scheduler_test::SchedulerStateBuffer> storage;
    scheduler_test::BasicGraphBuffer<256> graph_storage;
    SchedulerGraphView graph{};
    std::vector<SchedulerLocalState> locals;
    SchedulerRunControl *run{nullptr};

    PoolFixture(
        const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters, uint32_t width,
        SchedulerReadyPoolKind kind = SchedulerReadyPoolKind::TICKET, uint32_t chunk_window = 0
    ) :
        graph_storage(metadata.size()),
        locals(clusters * 3) {
        EXPECT_TRUE(scheduler_plan_layout(metadata.size(), metadata.size(), 0, &layout));
        EXPECT_TRUE(scheduler_plan_ready_pool(metadata, clusters, width, layout.total_size, &plan, kind, chunk_window));
        layout.total_size = plan.total_size;
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

TEST(SchedulerReadyPoolTest, AllocatesExactlyOneEntryPerExecutableMetadataAcrossAllTopologies) {
    auto metadata = tasks(97, 0);
    for (uint32_t ref = 0; ref < metadata.size(); ++ref) {
        if (ref % 5 == 0) metadata[ref].flags = 0;
        else if (ref % 3 == 0) metadata[ref] = tasks(1, 1)[0];
        else if (ref % 3 == 1) metadata[ref] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    }
    for (uint32_t width : {0, 4, 8}) {
        PoolFixture fixture(metadata, 11, width);
        ASSERT_TRUE(scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, metadata.size(), 11
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
        EXPECT_EQ(fixture.plan.header.group_counts[1], width == 0 ? 1u : (22 + width - 1) / width);
    }
}

TEST(SchedulerReadyPoolTest, AnyEligibleWorkerClaimsOrdinaryWorkAcrossFormerShardBoundaries) {
    for (uint32_t width : {0, 4, 8}) {
        for (uint32_t type : {0U, 1U, SCHEDULER_MIX_QUEUE}) {
            PoolFixture fixture(tasks(1, type), 12, width);
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
}

TEST(SchedulerReadyPoolTest, TypeEligibilityDoesNotDependOnWhichWorkerPublishes) {
    auto metadata = tasks(3, 0);
    metadata[1] = tasks(1, 1)[0];
    metadata[2] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    PoolFixture fixture(metadata, 9, 4);
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
    PoolFixture fixture(tasks(128, 0), 12, 4);
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
    for (uint32_t width : {4, 8}) {
        for (uint64_t initial_cursor : {0, 1}) {
            PoolFixture fixture(tasks(256, 0), 24, width);
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
}

TEST(SchedulerReadyPoolTest, UnplaceableMixRemainsSharedUntilAnotherCoordinatorClaimsIt) {
    PoolFixture fixture(tasks(1, SCHEDULER_MIX_QUEUE), 9, 4);
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
    for (uint32_t variant = 0; variant < 6; ++variant) {
        const uint32_t widths[] = {0, 4, 8, 0, 0, 0};
        PoolFixture fixture(
            tasks(192, 1), 12, widths[variant],
            variant >= 4 ? SchedulerReadyPoolKind::CHUNK :
                           (variant == 3 ? SchedulerReadyPoolKind::LINKED : SchedulerReadyPoolKind::TICKET),
            variant >= 4 ? (variant == 4 ? 1 : 4) : 0
        );
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
                auto &local = fixture.locals[producer * 3];
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (scheduler_ready_pool_has_pending(&local) && failures.load() == 0) {
                    if (!scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, nullptr) ||
                        std::chrono::steady_clock::now() >= deadline)
                        ++failures;
                }
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
}

TEST(SchedulerReadyPoolTest, RejectsCorruptDescriptorsBeforeAnyQueueAccess) {
    PoolFixture fixture(tasks(12, 0), 9, 4);
    auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.plan.header_offset);
    auto *descriptors = scheduler_state_at<SchedulerReadyPoolDescriptor>(fixture.base(), pool->descriptors_offset);
    const auto valid = [&] {
        return scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 12, 9
        );
    };
    ASSERT_TRUE(valid());
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
    ASSERT_TRUE(scheduler_plan_ready_pool(tasks(1, 0), 1, 0, 128, &plan));
    const uint64_t size = plan.total_size;
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 0, 0, 128, &plan));
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 1, 3, 128, &plan));
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 1, 0, UINT64_MAX - 64, &plan));
    EXPECT_EQ(plan.total_size, size);
}

TEST(SchedulerReadyPoolTest, LinkedQueuesUsePersistentNodesAndDistinctTypeSentinels) {
    auto metadata = tasks(97, 0);
    metadata[1] = tasks(1, 1)[0];
    metadata[2] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    PoolFixture fixture(metadata, 12, 0, SchedulerReadyPoolKind::LINKED);
    ASSERT_TRUE(scheduler_ready_pool_validate(
        fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, metadata.size(), 12
    ));
    for (uint32_t type = 0; type < 3; ++type) {
        auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.plan.header, type);
        EXPECT_EQ(queue.control->head, 97 + type);
        EXPECT_EQ(queue.control->tail, 97 + type);
        EXPECT_EQ(queue.links[queue.sentinel], SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.push(1, type, {type}));
        const uint32_t worker = type == 1 ? 34 : 33;
        uint64_t cursor = 0;
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(worker, type, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, type);
        EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[worker], type));
    }
}

TEST(SchedulerReadyPoolTest, LinkedMixPlacementFailureLeavesTaskAvailableToRemoteCoordinator) {
    PoolFixture fixture(tasks(1, SCHEDULER_MIX_QUEUE), 12, 0, SchedulerReadyPoolKind::LINKED);
    ASSERT_TRUE(fixture.push(1, SCHEDULER_MIX_QUEUE, {0}));
    for (auto &slot : fixture.locals[0].slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    SchedulerMixPlacement placement{};
    ASSERT_TRUE(fixture.pop(0, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(fixture.pop(33, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(placement.count, 2u);
}

TEST(SchedulerReadyPoolTest, LinkedPendingBatchSurvivesRetryAndMergesCompletionWorkWhileSlotsAreFull) {
    PoolFixture fixture(tasks(8, 0), 12, 0, SchedulerReadyPoolKind::LINKED);
    auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.plan.header, 0);
    const int64_t initial[] = {0, 1, 2, 3};
    SchedulerLinkedBatch paused{};
    ASSERT_EQ(scheduler_linked_prepare(queue, initial, 4, &paused), SchedulerLinkedResult::SUCCESS);
    ASSERT_EQ(
        scheduler_gm_compare_exchange(queue.links[queue.sentinel], SCHEDULER_TASK_REF_INVALID, paused.first),
        SCHEDULER_TASK_REF_INVALID
    );
    // The first publisher pauses after its linearization point, before moving tail.
    auto &local = fixture.locals[0];
    for (auto &slot : local.slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    for (auto &lane : local.mix_slots)
        for (auto &slot : lane)
            slot.state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(fixture.push(0, 0, {4, 5}));
    ASSERT_EQ(local.pending_publications[0].count, 2u);
    ASSERT_TRUE(fixture.push(0, 0, {6, 7}));
    ASSERT_EQ(local.pending_publications[0].count, 4u);
    bool progress = false;
    ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress));
    EXPECT_TRUE(progress);
    EXPECT_FALSE(scheduler_ready_pool_has_pending(&local));
    uint64_t cursor = 0;
    for (int64_t ref = 0; ref < 8; ++ref) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(33, 0, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, ref);
    }
}

TEST(SchedulerReadyPoolTest, LinkedAbortKeepsPrivateBatchUnpublishedUntilRunStorageIsDiscarded) {
    PoolFixture fixture(tasks(2, 0), 1, 0, SchedulerReadyPoolKind::LINKED);
    auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.plan.header, 0);
    const int64_t refs[] = {0, 1};
    auto &local = fixture.locals[0];
    ASSERT_EQ(scheduler_linked_prepare(queue, refs, 2, &local.pending_publications[0]), SchedulerLinkedResult::SUCCESS);
    scheduler_publish_error(fixture.run, 1);
    bool progress = true;
    ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(local.pending_publications[0].count, 2u);
    EXPECT_EQ(queue.links[queue.sentinel], SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(queue.control->tail, queue.sentinel);
}

TEST(SchedulerReadyPoolTest, LinkedLayoutRejectsOverlappingOrTruncatedLinksAndUnsupportedTopology) {
    PoolFixture fixture(tasks(12, 0), 9, 0, SchedulerReadyPoolKind::LINKED);
    auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.plan.header_offset);
    const auto valid = [&] {
        return scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 12, 9
        );
    };
    ASSERT_TRUE(valid());
    pool->links_offset = pool->controls_offset;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->storage_end_offset = pool->links_offset + 14 * sizeof(int64_t);
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->group_width = 8;
    EXPECT_FALSE(valid());
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 9, 8, 128, &fixture.plan, SchedulerReadyPoolKind::LINKED));
}

TEST(SchedulerReadyPoolTest, LinkedCorruptNextFailsWithoutClaimingOrDereferencingInvalidTask) {
    PoolFixture fixture(tasks(2, 0), 1, 0, SchedulerReadyPoolKind::LINKED);
    auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.plan.header, 0);
    scheduler_gm_store(queue.links[queue.sentinel], INT64_MAX);
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(fixture.pop(0, 0, &cursor, &claim));
    EXPECT_EQ(queue.control->head, queue.sentinel);
    EXPECT_NE(scheduler_read_error(fixture.run), 0u);
}

TEST(SchedulerReadyPoolTest, LinkedServicePublishesAllPendingTypesAndReinitializesAfterQuiescence) {
    auto metadata = tasks(3, 0);
    metadata[1] = tasks(1, 1)[0];
    metadata[2] = tasks(1, SCHEDULER_MIX_QUEUE)[0];
    PoolFixture fixture(metadata, 2, 0, SchedulerReadyPoolKind::LINKED);
    for (uint32_t epoch = 0; epoch < 2; ++epoch) {
        if (epoch != 0) scheduler_init_ready_pool(fixture.base(), fixture.plan);
        auto &local = fixture.locals[0];
        for (uint32_t type = 0; type < 3; ++type) {
            auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.plan.header, type);
            const int64_t ref = type;
            ASSERT_EQ(
                scheduler_linked_prepare(queue, &ref, 1, &local.pending_publications[type]),
                SchedulerLinkedResult::SUCCESS
            );
        }
        // A service budget of zero models a yielded bootstrap publisher.
        bool progress = true;
        ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress, 0));
        EXPECT_FALSE(progress);
        ASSERT_TRUE(scheduler_ready_pool_has_pending(&local));
        ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress));
        EXPECT_TRUE(progress);
        EXPECT_FALSE(scheduler_ready_pool_has_pending(&local));
        for (uint32_t type = 0; type < 3; ++type) {
            uint64_t cursor = 0;
            SchedulerReadyClaim claim{};
            ASSERT_TRUE(fixture.pop(type == 1 ? 4 : 3, type, &cursor, &claim));
            EXPECT_EQ(claim.task_ref, type);
        }
    }
}

TEST(SchedulerReadyPoolTest, ChunkLayoutsInitializeEveryNodeAndRouteAllTypesGlobally) {
    auto metadata = tasks(6, 0);
    for (uint32_t type = 0; type < 3; ++type)
        for (uint32_t index = 0; index < 2; ++index)
            metadata[type * 2 + index] = tasks(1, type)[0];
    for (uint32_t window : {1, 4}) {
        PoolFixture fixture(metadata, 12, 0, SchedulerReadyPoolKind::CHUNK, window);
        ASSERT_TRUE(
            scheduler_ready_pool_validate(fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 6, 12)
        );
        auto *chunks = scheduler_state_at<SchedulerReadyChunk>(fixture.base(), fixture.plan.header.chunks_offset);
        for (uint32_t ref = 0; ref < 6; ++ref) {
            EXPECT_EQ(chunks[ref].count, 0u);
            EXPECT_EQ(chunks[ref].cursor, 0u);
            for (auto entry : chunks[ref].refs)
                EXPECT_EQ(entry, SCHEDULER_TASK_REF_INVALID);
        }
        for (uint32_t type = 0; type < 3; ++type) {
            ASSERT_TRUE(fixture.push(1, type, {type * 2, type * 2 + 1}));
            uint64_t cursor = 0;
            for (uint32_t index = 0; index < 2; ++index) {
                SchedulerReadyClaim claim{};
                ASSERT_TRUE(fixture.pop(type == 1 ? 34 : 33, type, &cursor, &claim));
                EXPECT_EQ(claim.task_ref, type * 2 + index);
            }
        }
    }
}

TEST(SchedulerReadyPoolTest, ChunkWindowPeriodicallyServicesTheOldestNonemptyChunk) {
    PoolFixture fixture(tasks(32, 0), 12, 0, SchedulerReadyPoolKind::CHUNK, 4);
    std::vector<int64_t> refs(32);
    std::iota(refs.begin(), refs.end(), 0);
    ASSERT_TRUE(fixture.push(0, 0, refs));
    uint64_t cursor = 0;
    for (uint32_t index = 0; index < 8; ++index) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
        ASSERT_GE(claim.task_ref, 0);
        if ((index & 3U) == 3) EXPECT_EQ(claim.task_ref, index / 4);
    }
}

TEST(SchedulerReadyPoolTest, ChunkPendingCountsDescriptorsAndPublishesWithFullSlotsAfterRetry) {
    PoolFixture fixture(tasks(8, 0), 12, 0, SchedulerReadyPoolKind::CHUNK, 1);
    auto queue = scheduler_ready_pool_chunk_queue(fixture.base(), fixture.plan.header, 0);
    SchedulerLinkedBatch paused{};
    for (int64_t ref = 0; ref < 4; ++ref) {
        SchedulerLinkedBatch incoming{};
        ASSERT_EQ(scheduler_chunk_prepare(queue, &ref, 1, &incoming), SchedulerLinkedResult::SUCCESS);
        ASSERT_EQ(scheduler_linked_append_private(queue.chain, &paused, &incoming), SchedulerLinkedResult::SUCCESS);
    }
    ASSERT_EQ(
        scheduler_gm_compare_exchange(
            queue.chain.links[queue.chain.sentinel], SCHEDULER_TASK_REF_INVALID, paused.first
        ),
        SCHEDULER_TASK_REF_INVALID
    );
    auto &local = fixture.locals[0];
    for (auto &slot : local.slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    for (auto &lane : local.mix_slots)
        for (auto &slot : lane)
            slot.state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(fixture.push(0, 0, {4, 5}));
    ASSERT_EQ(local.pending_publications[0].count, 1u);
    ASSERT_TRUE(fixture.push(0, 0, {6, 7}));
    ASSERT_EQ(local.pending_publications[0].count, 2u);
    EXPECT_EQ(queue.chunks[4].count, 2u);
    EXPECT_EQ(queue.chunks[6].count, 2u);
    bool progress = false;
    ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress));
    EXPECT_TRUE(progress);
    EXPECT_FALSE(scheduler_ready_pool_has_pending(&local));
    uint64_t cursor = 0;
    for (int64_t ref = 0; ref < 8; ++ref) {
        SchedulerReadyClaim claim{};
        ASSERT_TRUE(fixture.pop(33, 0, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, ref);
    }
}

TEST(SchedulerReadyPoolTest, ChunkAbortLeavesSealedPrivatePayloadUnclaimedAndUnpublished) {
    PoolFixture fixture(tasks(4, 0), 1, 0, SchedulerReadyPoolKind::CHUNK, 4);
    auto queue = scheduler_ready_pool_chunk_queue(fixture.base(), fixture.plan.header, 0);
    auto &local = fixture.locals[0];
    for (int64_t first : {0, 2}) {
        const int64_t refs[] = {first, first + 1};
        SchedulerLinkedBatch incoming{};
        ASSERT_EQ(scheduler_chunk_prepare(queue, refs, 2, &incoming), SchedulerLinkedResult::SUCCESS);
        ASSERT_EQ(
            scheduler_linked_append_private(queue.chain, &local.pending_publications[0], &incoming),
            SchedulerLinkedResult::SUCCESS
        );
    }
    scheduler_publish_error(fixture.run, 1);
    bool progress = true;
    ASSERT_TRUE(scheduler_ready_pool_service_pending(fixture.base(), &local, fixture.run, nullptr, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(local.pending_publications[0].count, 2u);
    EXPECT_EQ(queue.chain.links[queue.chain.sentinel], SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(queue.chain.control->head, queue.chain.sentinel);
    EXPECT_EQ(queue.chain.control->tail, queue.chain.sentinel);
    EXPECT_EQ(queue.chain.links[0], 2);
    for (int64_t first : {0, 2}) {
        EXPECT_EQ(queue.chunks[first].cursor, 0u);
        EXPECT_EQ(queue.chunks[first].count, 2u);
        EXPECT_EQ(queue.chunks[first].refs[1], first + 1);
    }
}

TEST(SchedulerReadyPoolTest, ChunkWorkHintSeesSuccessorsBehindAnExhaustedHead) {
    PoolFixture fixture(tasks(9, 0), 1, 0, SchedulerReadyPoolKind::CHUNK, 1);
    std::vector<int64_t> refs(9);
    std::iota(refs.begin(), refs.end(), 0);
    ASSERT_TRUE(fixture.push(0, 0, refs));
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    for (uint32_t index = 0; index < 8; ++index) {
        ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
        EXPECT_EQ(claim.task_ref, index);
    }
    ASSERT_TRUE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[0], 0));
    ASSERT_TRUE(fixture.pop(0, 0, &cursor, &claim));
    EXPECT_EQ(claim.task_ref, 8);
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(fixture.base(), &fixture.locals[0], 0));
}

TEST(SchedulerReadyPoolTest, ChunkMixWithoutPlacementRemainsGloballyAvailable) {
    PoolFixture fixture(tasks(9, SCHEDULER_MIX_QUEUE), 12, 0, SchedulerReadyPoolKind::CHUNK, 4);
    std::vector<int64_t> refs(9);
    std::iota(refs.begin(), refs.end(), 0);
    ASSERT_TRUE(fixture.push(1, SCHEDULER_MIX_QUEUE, refs));
    for (auto &slot : fixture.locals[0].slots)
        slot.state = SchedulerDispatchSlotState::FILLING;
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    SchedulerMixPlacement placement{};
    ASSERT_TRUE(fixture.pop(0, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(fixture.pop(33, SCHEDULER_MIX_QUEUE, &cursor, &claim, &placement));
    EXPECT_GE(claim.task_ref, 0);
    EXPECT_EQ(placement.count, 2u);
}

TEST(SchedulerReadyPoolTest, ChunkLayoutRejectsPayloadOverlapAndUnknownWindow) {
    PoolFixture fixture(tasks(12, 0), 9, 0, SchedulerReadyPoolKind::CHUNK, 4);
    auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(fixture.base(), fixture.plan.header_offset);
    const auto valid = [&] {
        return scheduler_ready_pool_validate(
            fixture.base(), fixture.layout.total_size, fixture.plan.header_offset, 12, 9
        );
    };
    ASSERT_TRUE(valid());
    pool->chunks_offset = pool->links_offset;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->chunk_window = 2;
    EXPECT_FALSE(valid());
    *pool = fixture.plan.header;
    pool->storage_end_offset = pool->chunks_offset + sizeof(SchedulerReadyChunk) * 11;
    EXPECT_FALSE(valid());
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 1, 0, 128, &fixture.plan, SchedulerReadyPoolKind::CHUNK, 0));
    EXPECT_FALSE(scheduler_plan_ready_pool(tasks(1, 0), 1, 0, 128, &fixture.plan, SchedulerReadyPoolKind::TICKET, 4));
}

}  // namespace
