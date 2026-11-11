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
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_ready_publication.h"
#include "scheduler/scheduler_completion_resolver.h"
#include "runtime_types.h"
#include "hbg_scheduler_test_support.h"

namespace {

using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;
using scheduler_test::kFixtureCallableCount;
using scheduler_test::SchedulerStateBuffer;

bool pop_ready(
    const SchedulerGraphAccess &graph, void *base, SchedulerLocalState *local, SchedulerRunControl *run, uint32_t queue,
    int64_t *task
) {
    SchedulerReadyClaim claim{};
    const bool ok =
        scheduler_claim_ready_for_slot(graph, base, local, run, local->config.runtime_worker_count, queue, &claim);
    *task = claim.task_ref;
    return ok;
}

TEST(SchedulerActivityBuffer, IsAllocatedOnlyWhenRequestedAndNeverWraps) {
    AicoreSchedulerLayout disabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &disabled));
    EXPECT_EQ(disabled.activity_buffers_offset, 0u);

    AicoreSchedulerLayout enabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &enabled, true));
    ASSERT_NE(enabled.activity_buffers_offset, 0u);
    EXPECT_EQ(
        enabled.total_size - disabled.total_size,
        static_cast<uint64_t>(SCHEDULER_WORKER_CAPACITY) * sizeof(SchedulerActivityBuffer)
    );
    SchedulerStateBuffer storage(enabled);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(storage.base(), enabled.worker_contexts_offset);
    auto *buffers = scheduler_state_at<SchedulerActivityBuffer>(storage.base(), enabled.activity_buffers_offset);
    contexts[0].worker_index = SCHEDULER_WORKER_CAPACITY - 1;
    contexts[0].is_scheduler = 1;
    contexts[0].scheduler_index = SCHEDULER_WORKER_CAPACITY - 1;
    SchedulerLocalState local{};
    SchedulerLocalProfilingState profile{};
    local.profiling = &profile;
    local.profiling->activity_buffers_offset = contexts[0].activity_buffers_offset;
    local.config.scheduler_index = contexts[0].scheduler_index;
    local.config.self_lane = 0;
    local.profiling->loop_iter = 17;
    SchedulerActivityBuffer &buffer = buffers[SCHEDULER_WORKER_CAPACITY - 1];
    buffer.committed = SCHEDULER_ACTIVITY_CAPACITY - 1;

    scheduler_append_idle_activity(storage.base(), &local, 10, 20);
    scheduler_append_idle_activity(storage.base(), &local, 30, 40);

    EXPECT_EQ(buffer.committed, SCHEDULER_ACTIVITY_CAPACITY);
    EXPECT_EQ(buffer.dropped, 1u);
    const SchedulerIdleRecord &last = buffer.records[SCHEDULER_ACTIVITY_CAPACITY - 1];
    EXPECT_EQ(last.start_time, 10u);
    EXPECT_EQ(last.end_time, 20u);
    EXPECT_EQ(last.loop_iter, 17u);
}

TEST(SchedulerActivityBuffer, RejectsCorruptCommittedCountOnHost) {
    EXPECT_TRUE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY));
    EXPECT_FALSE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY + 1));
}

TEST(SchedulerProfilingLevel, EnablesOnlyTheRequestedGranularity) {
    EXPECT_FALSE(scheduler_task_timing_enabled(0));
    EXPECT_TRUE(scheduler_task_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
}

TEST(SchedulerProfilingLevel, DispatchWritesTaskIdentityBeforePhaseDetails) {
    for (uint64_t level = 0; level <= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL; ++level) {
        FixtureStorage storage(1, 2);
        storage.prepare_ready_pool();
        GraphBuffer graph(1);
        graph.executable(0, 0);
        storage.contexts[1].core_type = static_cast<int32_t>(CoreType::AIC);
        SchedulerReadyClaim ready_claim{};
        ready_claim.task_ref = 0;
        ready_claim.source = SchedulerReadySource::POOL;
        ready_claim.state_probe_start_cycles = 100;
        ready_claim.state_probe_end_cycles = 123;

        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(),
            storage.local_context(&storage.contexts[1], &storage.scheduler_local_state), storage.run_control,
            SchedulerFreeSlotClaim{1, 0, 0, 0}, ready_claim, level
        ));
        auto *traces =
            scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
        EXPECT_EQ(traces[0].worker_id, level == 0 ? 0u : storage.contexts[1].worker_index);
        EXPECT_EQ(traces[0].task_ref, 0u);
        EXPECT_EQ(
            traces[0].state_probe_scheduler_worker_id,
            level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? storage.contexts[1].worker_index : 0u
        );
        EXPECT_EQ(traces[0].state_probe_start_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 100u : 0u);
        EXPECT_EQ(traces[0].state_probe_end_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 123u : 0u);
        EXPECT_EQ(
            traces[0].ready_source,
            level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? static_cast<uint64_t>(SchedulerReadySource::POOL) : 0u
        );
        EXPECT_EQ(
            traces[0].publication_mode, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ?
                                            static_cast<uint64_t>(SchedulerPublicationMode::DISPATCH) :
                                            0u
        );
    }
}

struct ReadyPublicationTraceCase {
    LaneMask active_mask;
    uint16_t blocks;
    uint64_t level;
    bool inline_task;
};

std::vector<ReadyPublicationTraceCase> ready_publication_trace_cases() {
    std::vector<ReadyPublicationTraceCase> cases;
    for (uint64_t level : {UINT64_C(1), UINT64_C(3), UINT64_C(4)}) {
        for (LaneMask mask :
             {LaneMask::aic(), LaneMask::aiv0(), LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false),
              LaneMask(/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true), LaneMask::both_aiv(), LaneMask::all()}) {
            for (uint16_t blocks : {1, 2, 512})
                cases.push_back({mask, blocks, level, false});
        }
        cases.push_back({LaneMask::aiv0(), 1, level, true});
    }
    return cases;
}

class SchedulerReadyPublicationTrace : public ::testing::TestWithParam<ReadyPublicationTraceCase> {
protected:
    void SetUp() override {
        const auto &param = GetParam();
        const uint32_t traces_per_task = param.blocks * param.active_mask.count();
        storage = std::make_unique<FixtureStorage>(3, 3, traces_per_task);
        graph.executable(0, 0);
        for (uint32_t worker = 0; worker < 3; ++worker) {
            storage->contexts[worker].scheduler_count = 3;
            for (uint32_t lane = 0; lane < 3; ++lane)
                storage->contexts[worker].cluster_worker_ids[lane] = lane;
        }
        for (uint32_t task = 1; task < 3; ++task) {
            auto &metadata = storage->metadata[task];
            metadata.active_mask = param.active_mask;
            metadata.logical_block_num = param.blocks;
            metadata.trace_index_base = task * traces_per_task;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (param.active_mask.is_mix()) metadata.flags |= SCHEDULER_TASK_MIX;
            if (param.blocks > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
            if (param.inline_task) metadata.flags |= SCHEDULER_TASK_INLINE;
            if (!param.inline_task) graph.mixed(task, param.active_mask);
            for (uint32_t slot = 0; slot < 3; ++slot)
                metadata.kernel_ids[slot] =
                    !param.inline_task && param.active_mask.has(static_cast<SubtaskSlot>(slot)) ? 1 : UINT16_MAX;
        }
        storage->prepare_ready_pool();
        queue = scheduler_task_ready_queue(storage->metadata[1].flags, param.active_mask);
        local = storage->local_context(&storage->contexts[queue == 1 ? 1 : 0]);
        traces = scheduler_state_at<SchedulerTaskTrace>(base(), storage->layout.trace_cells_offset);
        trace_snapshot.assign(3 * traces_per_task * sizeof(SchedulerTaskTrace), 0xa5);
        ASSERT_FALSE(trace_snapshot.empty());
        ASSERT_LE(storage->layout.trace_cells_offset + trace_snapshot.size(), storage->layout.total_size);
        std::memcpy(traces, trace_snapshot.data(), trace_snapshot.size());
    }

    void *base() const { return storage->scheduler_state->base(); }

    void expect_traces_unchanged() const {
        EXPECT_EQ(std::memcmp(traces, trace_snapshot.data(), trace_snapshot.size()), 0);
    }

    void append_and_pop() {
        SchedulerReadyBatch batch{};

        for (int64_t task : {1, 2}) {
            const bool appended = scheduler_ready_batch_append(base(), local, task, &batch);
            ASSERT_TRUE(appended);
        }
        EXPECT_EQ(batch.head, 1);
        EXPECT_EQ(batch.tail, 2);
        EXPECT_EQ(batch.count, 2u);

        expect_traces_unchanged();
        ASSERT_TRUE(scheduler_ready_batch_push(base(), local, queue, &batch));

        EXPECT_EQ(batch.count, 0u);
        for (int64_t expected : {INT64_C(1), INT64_C(2), SCHEDULER_TASK_REF_INVALID}) {
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            ASSERT_TRUE(pop_ready(graph.graph(), base(), local, storage->run_control, queue, &task));
            EXPECT_EQ(task, expected);
        }
        EXPECT_EQ(storage->run_control->scheduler_error, 0u);
        expect_traces_unchanged();
    }

    std::unique_ptr<FixtureStorage> storage;
    scheduler_test::BasicGraphBuffer<4> graph{3};
    SchedulerLocalState *local{nullptr};
    SchedulerTaskTrace *traces{nullptr};
    uint32_t queue{0};
    std::vector<uint8_t> trace_snapshot;
};

TEST_P(SchedulerReadyPublicationTrace, AppendPublishesLogicalTasksWithoutWritingExecutionTraces) { append_and_pop(); }

TEST_P(SchedulerReadyPublicationTrace, OrdinaryDirectAndQueuedMixLeaveExecutionTracesUntouched) {
    graph.payload(1).fanin_count = 1;
    graph.payload(1).fanin_data()[0] = 0;
    storage->metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;

    ASSERT_EQ(
        scheduler_route_task(graph.graph(), base(), local, storage->run_control, 1), SchedulerRouteResult::WAITING
    );
    auto *producer = scheduler_task_control_at(base(), local, 0);
    scheduler_gm_store(producer->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    SchedulerReadyClaim direct{};
    const bool mix = scheduler_task_is_mix(storage->metadata[1].flags);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), base(), local, storage->run_control, 0, GetParam().level, true, &direct,
        local->config.self_core_type
    ));
    EXPECT_EQ(direct.task_ref, mix ? SCHEDULER_TASK_REF_INVALID : 1);
    if (!mix) {
        EXPECT_EQ(direct.source, SchedulerReadySource::DIRECT_RESOLVE);
        EXPECT_EQ(direct.publication_mode, SchedulerPublicationMode::REFILL);
    }
    EXPECT_EQ(producer->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);

    EXPECT_EQ(storage->run_control->scheduler_error, 0u);
    int64_t ref = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(pop_ready(graph.graph(), base(), local, storage->run_control, queue, &ref));
    EXPECT_EQ(ref, mix ? 1 : SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(pop_ready(graph.graph(), base(), local, storage->run_control, queue, &ref));
    EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
    expect_traces_unchanged();
}

INSTANTIATE_TEST_SUITE_P(
    AllTaskShapes, SchedulerReadyPublicationTrace, ::testing::ValuesIn(ready_publication_trace_cases())
);

class SchedulerMixReadyPublicationTrace : public SchedulerReadyPublicationTrace {};

std::vector<ReadyPublicationTraceCase> mix_ready_fallback_trace_cases() {
    std::vector<ReadyPublicationTraceCase> cases;
    for (const auto &param : ready_publication_trace_cases()) {
        if (param.level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL && param.active_mask.is_mix()) cases.push_back(param);
    }
    return cases;
}

TEST_P(SchedulerMixReadyPublicationTrace, MixUsesSharedQueueAndKeepsOlderWorkAhead) {
    graph.payload(1).fanin_count = 1;
    graph.payload(1).fanin_data()[0] = 0;
    storage->metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;

    ASSERT_EQ(
        scheduler_route_task(graph.graph(), base(), local, storage->run_control, 1), SchedulerRouteResult::WAITING
    );
    SchedulerReadyBatch older{};
    ASSERT_TRUE(scheduler_ready_batch_append(base(), local, 2, &older));
    ASSERT_TRUE(scheduler_ready_batch_push(base(), local, queue, &older));
    scheduler_gm_store(
        scheduler_task_control_at(base(), local, 0)->state, static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    SchedulerReadyClaim direct{};
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), base(), local, storage->run_control, 0, GetParam().level, true, &direct,
        local->config.self_core_type
    ));
    EXPECT_EQ(direct.task_ref, SCHEDULER_TASK_REF_INVALID);

    for (int64_t expected : {INT64_C(2), INT64_C(1), SCHEDULER_TASK_REF_INVALID}) {
        int64_t ref = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(pop_ready(graph.graph(), base(), local, storage->run_control, queue, &ref));
        EXPECT_EQ(ref, expected);
    }
    expect_traces_unchanged();
}

INSTANTIATE_TEST_SUITE_P(
    AllMixShapes, SchedulerMixReadyPublicationTrace, ::testing::ValuesIn(mix_ready_fallback_trace_cases())
);

TEST(SchedulerBootstrap, RegistersOnlyOnFirstExecutableProducer) {
    FixtureStorage storage(4, 2);
    storage.prepare_ready_pool();
    GraphBuffer graph(4);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    graph.executable(3, 1, {2, 1});
    storage.metadata[1].active_mask = LaneMask::aiv0();
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.metadata[2].flags = 0;
    storage.metadata[3].active_mask = LaneMask::aiv0();
    storage.metadata[3].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[2].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    controls[2].wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;

    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_EQ(controls[1].waiting_producer, 0);

    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[1]),
            storage.run_control, 3
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[1].wake_list_head, 3);
    EXPECT_EQ(controls[3].next_fanin_index, 1);
    EXPECT_EQ(controls[3].waiting_producer, 1);
}

TEST(SchedulerRouteContinuation, BoundsFaninWorkAndResumesFromAnotherWorker) {
    constexpr uint32_t fanin_count = SCHEDULER_GRAPH_MAX_FANIN;
    FixtureStorage storage(fanin_count + 1, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(fanin_count + 1);
    std::vector<int32_t> fanins;
    auto *base = storage.scheduler_state->base();
    auto *first = storage.local_context(&storage.contexts[0]);
    for (uint32_t i = 0; i < fanin_count; ++i) {
        graph.executable(i, 0);
        fanins.push_back(i);
        scheduler_task_control_at(base, first, i)->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    }
    graph.executable(fanin_count, 0, fanins);
    SchedulerRouteCursor cursor;

    EXPECT_EQ(
        scheduler_route_task_step(graph.graph(), base, first, storage.run_control, fanin_count, cursor, 0).result,
        SchedulerRouteResult::YIELD
    );
    EXPECT_EQ(cursor.task_ref, SCHEDULER_TASK_REF_INVALID);
    uint32_t total_units = 0;
    for (uint32_t call = 0; cursor.result == SchedulerRouteResult::YIELD && call < fanin_count + 2; ++call) {
        auto *local = storage.local_context(&storage.contexts[call % 3]);
        SchedulerRouteCursor restored;
        std::memcpy(&restored, &cursor, sizeof(cursor));
        const auto progress =
            scheduler_route_task_step(graph.graph(), base, local, storage.run_control, fanin_count, restored, 8);
        EXPECT_LE(progress.work_units, 8u);
        total_units += progress.work_units;
        cursor = restored;
    }
    EXPECT_EQ(cursor.result, SchedulerRouteResult::READY_TO_ENQUEUE);
    EXPECT_EQ(total_units, fanin_count + 1);

    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerRouteContinuation, ProducerClosingBetweenStepsAdvancesWithoutLostRegistration) {
    FixtureStorage storage(3, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0);
    graph.executable(2, 1, {0, 1});
    auto *base = storage.scheduler_state->base();
    auto *local = storage.local_context(&storage.contexts[0]);
    SchedulerRouteCursor cursor;

    auto progress = scheduler_route_task_step(graph.graph(), base, local, storage.run_control, 2, cursor, 2);
    ASSERT_EQ(progress.result, SchedulerRouteResult::YIELD);
    ASSERT_EQ(cursor.producer_ref, 0);
    auto *producer = scheduler_task_control_at(base, local, 0);
    scheduler_gm_store(producer->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    scheduler_gm_exchange(producer->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    local = storage.local_context(&storage.contexts[1]);
    progress = scheduler_route_task_step(graph.graph(), base, local, storage.run_control, 2, cursor, 1);
    ASSERT_EQ(progress.result, SchedulerRouteResult::YIELD);

    progress = scheduler_route_task_step(graph.graph(), base, local, storage.run_control, 2, cursor, 2);
    ASSERT_EQ(progress.result, SchedulerRouteResult::WAITING);
    EXPECT_EQ(scheduler_task_control_at(base, local, 1)->wake_list_head, 2);

    auto *waiter = scheduler_task_control_at(base, local, 2);
    waiter->next_fanin_index = INT32_MIN;
    waiter->next_waiter = 71;
    waiter->state = static_cast<int64_t>(SchedulerTaskState::DONE);
    const auto retry = scheduler_route_task_step(graph.graph(), base, local, storage.run_control, 2, cursor, 8);
    EXPECT_EQ(retry.result, SchedulerRouteResult::WAITING);
    EXPECT_EQ(retry.work_units, 0u);
    EXPECT_EQ(waiter->next_fanin_index, INT32_MIN);
    EXPECT_EQ(waiter->next_waiter, 71);

    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerRouteContinuation, ConcurrentPublishersRetainEveryWaiterWithinPerCallBudgets) {
    constexpr uint32_t publishers = 16;
    FixtureStorage storage(publishers + 1, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(publishers + 1);
    graph.executable(0, 0);
    for (uint32_t i = 1; i <= publishers; ++i)
        graph.executable(i, 1, {0});
    auto *base = storage.scheduler_state->base();
    auto *local = storage.local_context(&storage.contexts[0]);
    std::vector<SchedulerRouteCursor> cursors(publishers);
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < publishers; ++index) {
        threads.emplace_back([&, index] {
            SchedulerLocalState own = *local;

            while (!start.load(std::memory_order_acquire)) {}
            for (uint32_t attempts = 0; attempts < 10000 && cursors[index].result == SchedulerRouteResult::YIELD;
                 ++attempts) {
                auto progress = scheduler_route_task_step(
                    graph.graph(), base, &own, storage.run_control, index + 1, cursors[index], 1
                );
                EXPECT_LE(progress.work_units, 1u);
            }
            EXPECT_EQ(cursors[index].result, SchedulerRouteResult::WAITING);
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    std::vector<bool> found(publishers + 1, false);
    int64_t waiter =
        scheduler_gm_exchange(scheduler_task_control_at(base, local, 0)->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    uint32_t count = 0;
    while (waiter >= 0 && count <= publishers) {
        ASSERT_GE(waiter, 1);
        ASSERT_LE(waiter, publishers);
        EXPECT_FALSE(found[waiter]);
        found[waiter] = true;
        waiter = scheduler_observe_next_waiter(scheduler_task_control_at(base, local, waiter));
        ++count;
    }
    EXPECT_EQ(waiter, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_EQ(count, publishers);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerReadyPublicationTest, TransferKeepsTheSharedQueueAndNeverRewritesPublishedTasks) {
    FixtureStorage storage(9, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(9);
    auto *base = storage.scheduler_state->base();
    auto *first = storage.local_context(&storage.contexts[0]);
    SchedulerReadyPublication publication{};
    for (int64_t ref = 0; ref < 8; ++ref) {
        graph.executable(ref, 0);
        ASSERT_EQ(
            scheduler_ready_publication_append(graph.graph(), base, first, ref, publication),
            SchedulerReadyAppendResult::ADDED
        );
    }
    EXPECT_EQ(
        scheduler_ready_publication_append(graph.graph(), base, first, 8, publication), SchedulerReadyAppendResult::FULL
    );
    EXPECT_EQ(publication.destination, SchedulerReadyDestination::AIC);
    SchedulerReadyPublication moved{};
    std::memcpy(&moved, &publication, sizeof(moved));
    auto *second = storage.local_context(&storage.contexts[3]);
    ASSERT_EQ(scheduler_ready_publication_step(graph.graph(), base, second, moved), SchedulerReadyPublishResult::DONE);
    for (int64_t expected = 0; expected < 8; ++expected) {
        int64_t ref = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(pop_ready(graph.graph(), base, first, storage.run_control, 0, &ref));
        ASSERT_EQ(ref, expected);
        scheduler_task_control_at(base, first, ref)->next_waiter = 71;
    }
    EXPECT_EQ(scheduler_ready_publication_step(graph.graph(), base, first, moved), SchedulerReadyPublishResult::DONE);
    for (int64_t ref = 0; ref < 8; ++ref)
        EXPECT_EQ(scheduler_task_control_at(base, first, ref)->next_waiter, 71);
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(base, first, 0));
}

TEST(SchedulerReadyPublicationTest, CorruptContinuationCannotRedirectBusinessPublicationToAnInbox) {
    FixtureStorage storage(1, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    auto *base = storage.scheduler_state->base();
    auto *local = storage.local_context(&storage.contexts[0]);
    SchedulerReadyPublication publication{};
    ASSERT_EQ(
        scheduler_ready_publication_append(graph.graph(), base, local, 0, publication),
        SchedulerReadyAppendResult::ADDED
    );
    publication.inbox_offset = sizeof(SchedulerReadyInbox);
    EXPECT_EQ(
        scheduler_ready_publication_step(graph.graph(), base, local, publication), SchedulerReadyPublishResult::ERROR
    );
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(base, local, 0));
}

TEST(SchedulerCompletionContinuation, PublishesReadyPrefixBeforeLongFaninCanYield) {
    FixtureStorage storage(11, 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(11);
    auto *base = storage.scheduler_state->base();
    auto *local = storage.local_context(&storage.contexts[0]);
    for (int64_t ref = 0; ref < 11; ++ref)
        graph.executable(ref, 0);
    graph.executable(1, 0, {0});
    graph.executable(10, 0, {0, 2, 3, 4, 5, 6, 7, 8, 9});
    for (int64_t ref : {10, 1}) {
        ASSERT_EQ(
            scheduler_route_task(graph.graph(), base, local, storage.run_control, ref), SchedulerRouteResult::WAITING
        );
    }
    for (int64_t ref = 0; ref < 11; ++ref) {
        if (ref == 1 || ref == 10) continue;
        scheduler_gm_store(
            scheduler_task_control_at(base, local, ref)->state, static_cast<int64_t>(SchedulerTaskState::DONE)
        );
    }
    SchedulerResolveCursor cursor;
    bool reached_long_fanin = false;
    for (uint32_t call = 0; call < 32; ++call) {
        const auto progress =
            scheduler_resolve_completion_step(graph.graph(), base, local, storage.run_control, 0, cursor, 1);
        ASSERT_NE(progress.result, SchedulerResolveResult::ERROR);
        EXPECT_LE(progress.work_units, 1u);
        if (cursor.waiter_ref == 10 && cursor.phase == SchedulerResolvePhase::ROUTE) {
            reached_long_fanin = true;
            break;
        }
    }
    ASSERT_TRUE(reached_long_fanin);
    int64_t ref = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(pop_ready(graph.graph(), base, local, storage.run_control, 0, &ref));
    EXPECT_EQ(ref, 1);
    for (uint32_t call = 0; call < 64 && cursor.phase != SchedulerResolvePhase::DONE; ++call) {
        const auto progress =
            scheduler_resolve_completion_step(graph.graph(), base, local, storage.run_control, 0, cursor, 1);
        ASSERT_NE(progress.result, SchedulerResolveResult::ERROR);
        EXPECT_LE(progress.work_units, 1u);
    }
    ASSERT_EQ(cursor.phase, SchedulerResolvePhase::DONE);
    ASSERT_TRUE(pop_ready(graph.graph(), base, local, storage.run_control, 0, &ref));
    EXPECT_EQ(ref, 10);
    ASSERT_TRUE(pop_ready(graph.graph(), base, local, storage.run_control, 0, &ref));
    EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerCompletionContinuation, SavesTheDetachedNextBeforeAnotherResolverReusesTheWaiter) {
    FixtureStorage storage(11, 3);
    storage.prepare_ready_pool();
    for (uint32_t worker = 0; worker < 3; ++worker)
        for (uint32_t lane = 0; lane < 3; ++lane)
            storage.contexts[worker].cluster_worker_ids[lane] = lane;
    GraphBuffer graph(11);
    graph.executable(0, 0);
    graph.executable(1, 0);
    auto *base = storage.scheduler_state->base();
    auto *local = storage.local_context(&storage.contexts[0]);
    for (int64_t ref = 2; ref < 11; ++ref) {
        graph.executable(ref, 0, {0, 1});
        ASSERT_EQ(
            scheduler_route_task(graph.graph(), base, local, storage.run_control, ref), SchedulerRouteResult::WAITING
        );
    }
    scheduler_gm_store(
        scheduler_task_control_at(base, local, 0)->state, static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    SchedulerResolveCursor cursor;

    bool migrated = false;
    std::vector<bool> seen(11, false);
    for (uint32_t attempt = 0; attempt < 1000 && cursor.phase != SchedulerResolvePhase::DONE; ++attempt) {
        SchedulerResolveCursor transferred;
        std::memcpy(&transferred, &cursor, sizeof(cursor));
        local = storage.local_context(&storage.contexts[attempt % 3]);
        const auto progress =
            scheduler_resolve_completion_step(graph.graph(), base, local, storage.run_control, 0, transferred, 1);
        ASSERT_NE(progress.result, SchedulerResolveResult::ERROR);
        EXPECT_LE(progress.work_units, 1u);
        cursor = transferred;
        if (scheduler_gm_load(scheduler_task_control_at(base, local, 1)->wake_list_head) >= 0 && !migrated) {
            migrated = true;
            scheduler_gm_store(
                scheduler_task_control_at(base, local, 1)->state, static_cast<int64_t>(SchedulerTaskState::DONE)
            );
            SchedulerResolveCursor peer;
            for (uint32_t call = 0; call < 32 && peer.phase != SchedulerResolvePhase::DONE; ++call) {
                const auto outcome =
                    scheduler_resolve_completion_step(graph.graph(), base, local, storage.run_control, 1, peer, 8);
                ASSERT_NE(outcome.result, SchedulerResolveResult::ERROR)
                    << "site=" << storage.run_control->error_site << " task=" << storage.run_control->error_task_ref
                    << " waiter=" << peer.waiter_ref << " next=" << peer.next_waiter
                    << " publish=" << static_cast<uint32_t>(peer.publication.phase)
                    << " target=" << peer.publication.target;
                EXPECT_LE(outcome.work_units, 8u);
            }
            ASSERT_EQ(peer.phase, SchedulerResolvePhase::DONE);
            int64_t ref = SCHEDULER_TASK_REF_INVALID;
            ASSERT_TRUE(pop_ready(
                graph.graph(), base, storage.local_context(&storage.contexts[0]), storage.run_control, 0, &ref
            ));
            ASSERT_GE(ref, 2);
            ASSERT_LT(ref, 11);
            seen[ref] = true;
            scheduler_task_control_at(base, local, ref)->next_waiter = INT64_MAX;
        }
    }
    EXPECT_TRUE(migrated);
    ASSERT_EQ(cursor.phase, SchedulerResolvePhase::DONE);
    EXPECT_EQ(cursor.visited_count, 9u);

    for (uint32_t count = 0; count < 8; ++count) {
        int64_t ref = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(
            pop_ready(graph.graph(), base, storage.local_context(&storage.contexts[0]), storage.run_control, 0, &ref)
        );
        ASSERT_GE(ref, 2);
        ASSERT_LT(ref, 11);
        EXPECT_FALSE(seen[ref]);
        seen[ref] = true;
    }
    for (uint32_t ref = 2; ref < 11; ++ref)
        EXPECT_TRUE(seen[ref]);
    const auto repeated =
        scheduler_resolve_completion_step(graph.graph(), base, local, storage.run_control, 0, cursor, 8);
    EXPECT_EQ(repeated.result, SchedulerResolveResult::DONE);
    EXPECT_EQ(repeated.work_units, 0u);

    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

// Bootstrap publishes through the same type-routed push the steady state uses,
// so a core must not leave work of another type in its own inbox: an AIC task
// an AIV lane found during its graph scan belongs to the cluster's AIC.
TEST(SchedulerBootstrap, AivPublisherRoutesAicBatchToSharedAicQueue) {
    FixtureStorage storage(2, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    auto *base = storage.scheduler_state->base();
    auto *publisher = storage.local_context(&storage.contexts[1]);
    auto *consumer = storage.local_context(&storage.contexts[3]);
    SchedulerReadyBatch batch{};

    for (int64_t ref : {0, 1})
        ASSERT_TRUE(scheduler_ready_batch_append(base, publisher, ref, &batch));
    ASSERT_TRUE(scheduler_ready_batch_push(base, publisher, 0, &batch));

    EXPECT_FALSE(scheduler_ready_pool_may_have_work(base, publisher, 1));
    for (int64_t expected : {0, 1}) {
        int64_t ref = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(pop_ready(graph.graph(), base, consumer, storage.run_control, 0, &ref));
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerReadyPool, RejectsReadyQueueBoundary) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    auto *local = storage.local_context(&storage.contexts[0]);
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, 0, &batch));
    EXPECT_FALSE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), local, SCHEDULER_READY_QUEUE_COUNT, &batch)
    );
    EXPECT_TRUE(scheduler_ready_queue_holder(0, 0));
    EXPECT_FALSE(scheduler_ready_queue_holder(1, 0));
    EXPECT_FALSE(scheduler_ready_queue_holder(2, 0));
}

TEST(SchedulerReadyPool, RequiresOwnerStateForPublishAndClaim) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);

    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 0, &batch
    ));

    EXPECT_FALSE(scheduler_ready_batch_push(storage.scheduler_state->base(), nullptr, 0, &batch));
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), nullptr, storage.run_control, 1, 0, &claim
    ));
    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), nullptr, storage.run_control, 0, 0, true, nullptr, UINT32_MAX
    ));
}

TEST(SchedulerReadyPool, BatchPushPreservesFifoAndDrainsTheSharedQueue) {
    constexpr uint64_t count = 4;
    FixtureStorage storage(count, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(count);
    auto *base = storage.scheduler_state->base();
    auto *producer = storage.local_context(&storage.contexts[0]);
    auto *consumer = storage.local_context(&storage.contexts[3]);
    SchedulerReadyBatch batch{};

    for (uint64_t task = 0; task < count; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(scheduler_ready_batch_append(base, producer, task, &batch));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(base, producer, 0, &batch));
    EXPECT_TRUE(scheduler_ready_pool_may_have_work(base, consumer, 0));
    for (uint64_t expected = 0; expected < count; ++expected) {
        int64_t ref = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(pop_ready(graph.graph(), base, consumer, storage.run_control, 0, &ref));
        EXPECT_EQ(ref, expected);
    }
    EXPECT_FALSE(scheduler_ready_pool_may_have_work(base, consumer, 0));
}

TEST(SchedulerDispatch, RejectsKernelIdBeforeCallableTableAccess) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(kFixtureCallableCount);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, RejectsInvalidSingleSubtaskShapeBeforeMetadataIndex) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].active_mask = LaneMask::none();
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsCoreTypeMismatch) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIV);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsUnknownTargetCoreType) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 1);
    storage.contexts[0].core_type = 2;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = LaneMask::aiv0();
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, DispatchSequenceIsIndependentOfSlotGenerationAndRejectsZeroCallable) {
    FixtureStorage storage(2, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.scheduler_local_state.slots[0].generation, 1u);
    uint32_t pending_slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&storage.scheduler_local_state, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, scheduler_dispatch_publication(1, SchedulerDispatchSlotState::READY));

    storage.callable_addresses[1] = 0;
    storage.run_control->error_claimed = 0;
    storage.run_control->scheduler_error = 0;
    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

// The bound is the count this worker context publishes, not the capacity the
// scheduler state used to reserve. Both ids below are mapped to a non-zero
// address, so a zero entry cannot be what refuses the second one: only the
// length can. An implementation that kept the former fixed 1024 bound accepts
// it and fails here.
TEST(SchedulerDispatch, RejectsAKernelIdPastThePublishedCallableCount) {
    FixtureStorage storage(2, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.contexts[0].callable_addresses_count = 2;
    storage.callable_addresses[1] = UINT64_C(0x1000);
    storage.callable_addresses[2] = UINT64_C(0x3000);
    storage.metadata[0].kernel_ids[0] = 1;
    storage.metadata[1].kernel_ids[0] = 2;

    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, UINT64_C(0x1000)) << "the last id inside the count still dispatches";

    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, AcceptsLastCallableAndInlineSentinel) {
    FixtureStorage storage(2, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(kFixtureCallableCount - 1);
    storage.callable_addresses[kFixtureCallableCount - 1] = UINT64_C(0x2000);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, UINT64_C(0x2000));

    storage.metadata[1].kernel_ids[0] = UINT16_MAX;
    storage.metadata[1].flags |= SCHEDULER_TASK_INLINE;
    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset + sizeof(DispatchPayload)
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerPredicate, DistinguishesFailedAndMalformedPredicates) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(8) int64_t value = 5;
    struct PredicateCase {
        PredicateOp op;
        int64_t pass_target;
        int64_t fail_target;
    };
    constexpr std::array<PredicateCase, 6> cases{{
        {PredicateOp::EQ, 5, 6},
        {PredicateOp::NE, 6, 5},
        {PredicateOp::GT, 4, 5},
        {PredicateOp::LT, 6, 5},
        {PredicateOp::GE, 5, 6},
        {PredicateOp::LE, 5, 4},
    }};
    constexpr std::array<uint8_t, 4> element_sizes{1, 2, 4, 8};
    for (uint8_t element_size : element_sizes) {
        for (const PredicateCase &predicate_case : cases) {
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.pass_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.fail_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::FAIL);
        }
    }

    graph.predicate(0, 0, 0, static_cast<uint8_t>(PredicateOp::NONE));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);

    graph.predicate(0, 0, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 3, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 8, UINT8_C(0xff));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value) + 1, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
}

TEST(SchedulerPredicate, MalformedPredicateStopsDispatchWithoutPublishingSlot) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    graph.predicate(0, 0, 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_PREDICATE));
    EXPECT_NE(storage.scheduler_local_state.slots[0].state, SchedulerDispatchSlotState::READY);
}

TEST(SchedulerPredicate, FailedPredicatePublishesDependencyOnlyDispatch) {
    FixtureStorage storage(1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(4) int32_t value = 0;
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0
    ));
    const auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerReadyPool, ConcurrentConsumersNeverDuplicateTask) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kConsumerCount = 8;
    FixtureStorage storage(kTasks, kConsumerCount * 3);
    storage.prepare_ready_pool();
    GraphBuffer graph(kTasks);
    SchedulerReadyBatch batch{};
    for (uint64_t task = 0; task < kTasks; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), task, &batch
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &storage.owner_states[0]), 0,
        &batch
    ));
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};
    auto consume = [&](uint64_t worker) {
        while (!failed.load(std::memory_order_relaxed) && claimed.load(std::memory_order_relaxed) < kTasks) {
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            if (!pop_ready(
                    graph.graph(), storage.scheduler_state->base(),
                    storage.local_context(&storage.contexts[worker], nullptr), storage.run_control, 0, &task
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);
    for (uint64_t worker = 0; worker < kConsumerCount; ++worker)
        consumers.emplace_back(consume, worker * 3);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyPool, ConcurrentPublicationAndConsumptionNeverLoseTasks) {
    constexpr uint64_t kTasks = 2048;
    constexpr uint64_t kSpinLimit = 10000000;
    FixtureStorage storage(kTasks, 6);
    storage.prepare_ready_pool();
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerLocalState &owner_state = storage.owner_states[0];
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};

    std::thread owner([&] {
        for (uint64_t task = 0; task < kTasks && !failed.load(std::memory_order_relaxed); ++task) {
            SchedulerReadyBatch batch{};
            if (!scheduler_ready_batch_append(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), task, &batch
                ) ||
                !scheduler_ready_batch_push(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0,
                    &batch
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        for (uint64_t spin = 0; spin < kSpinLimit && claimed.load(std::memory_order_relaxed) < kTasks; ++spin) {
            std::this_thread::yield();
        }
        if (claimed.load(std::memory_order_relaxed) != kTasks) failed.store(true, std::memory_order_relaxed);
    });
    std::thread consumer([&] {
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            if (!pop_ready(
                    graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[3]),
                    storage.run_control, 0, &task
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    });
    owner.join();
    consumer.join();

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyWake, ConcurrentRegistrationAndCloseResolveEveryConsumerExactlyOnce) {
    constexpr uint64_t kConsumers = 64;
    FixtureStorage storage(kConsumers + 1, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(kConsumers + 1);
    graph.executable(0, 0);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        graph.executable(task, 0, {0});
        storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    std::vector<std::atomic<uint32_t>> seen(kConsumers + 1);
    std::atomic<uint64_t> started{0};
    std::atomic<bool> failed{false};
    SchedulerLocalState *reader = storage.local_context(&storage.contexts[0]);
    SchedulerLocalState *owner = storage.local_context(&storage.contexts[0], &storage.owner_states[0]);
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumers);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        consumers.emplace_back([&, task] {
            started.fetch_add(1, std::memory_order_release);
            const SchedulerRouteResult route = scheduler_route_task(
                graph.graph(), storage.scheduler_state->base(), reader, storage.run_control, static_cast<int64_t>(task)
            );
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                seen[task].fetch_add(1, std::memory_order_relaxed);
            } else if (route != SchedulerRouteResult::WAITING) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    std::thread closer([&] {
        while (started.load(std::memory_order_acquire) != kConsumers)
            std::this_thread::yield();
        scheduler_gm_store(controls[0].state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph.graph(), storage.scheduler_state->base(), owner, storage.run_control, 0, 0, true, nullptr,
                UINT32_MAX
            ))
            failed.store(true, std::memory_order_relaxed);
    });
    for (auto &consumer : consumers)
        consumer.join();
    closer.join();

    while (!failed.load(std::memory_order_relaxed)) {
        int64_t task = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(pop_ready(graph.graph(), storage.scheduler_state->base(), reader, storage.run_control, 0, &task));
        if (task == SCHEDULER_TASK_REF_INVALID) break;
        seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        EXPECT_EQ(seen[task].load(), 1u) << "consumer=" << task;
        EXPECT_EQ(controls[task].next_fanin_index, 0) << "consumer=" << task;
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED)) << "consumer=" << task;
    }
}

TEST(SchedulerReadyWake, WakeResolvePublishesConsumerToSharedQueue) {
    FixtureStorage storage(2, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;

    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.owner_states[0]), storage.run_control, 0, 0, true, nullptr,
        UINT32_MAX
    ));

    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(pop_ready(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, &task
    ));
    EXPECT_EQ(task, 1);
}

// A push appends, so work a completion resolves is consumed after work that was
// already queued: the queue is one FIFO per ready class.
TEST(SchedulerReadyWake, WakeResolveQueuesBehindOlderPublishedWork) {
    FixtureStorage storage(3, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0);
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;

    SchedulerLocalState &owner_state = storage.owner_states[0];
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );
    SchedulerReadyBatch older{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 2, &older
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0, &older
    ));
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state),
        storage.run_control, 0, 0, true, nullptr, UINT32_MAX
    ));

    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(pop_ready(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, &task
    ));
    EXPECT_EQ(task, 2);
    ASSERT_TRUE(pop_ready(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, &task
    ));
    EXPECT_EQ(task, 1);
}

TEST(SchedulerReadyWake, RejectsInvalidWaiterShapeBeforeReadyBatchIndex) {
    FixtureStorage storage(2, 1);
    storage.prepare_ready_pool();
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].active_mask = LaneMask::none();
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);

    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.owner_states[0]), storage.run_control, 0, 0, true, nullptr,
        UINT32_MAX
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_INVALID_SHAPE));
}

struct MixPrefixFixture {
    explicit MixPrefixFixture(uint32_t successors = 4, uint32_t producer_worker = 0) :
        count(successors),
        storage(successors + 3, 9, 3),
        graph(successors + 3) {
        graph.executable(0, 0);
        graph.executable(1, 0);
        for (uint32_t task = 2; task < count + 3; ++task) {
            graph.mixed(task, LaneMask::all());
            graph.executable(task, 0, task < count + 2 ? std::vector<int32_t>{0} : std::vector<int32_t>{});
            auto &metadata = storage.metadata[task];
            metadata.active_mask = LaneMask::all();
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
            if (task < count + 2) metadata.flags |= SCHEDULER_TASK_HAS_FANIN;
            metadata.trace_index_base = task * 3;
            metadata.kernel_ids[0] = metadata.kernel_ids[1] = metadata.kernel_ids[2] = 1;
        }
        storage.prepare_ready_pool();
        for (uint32_t worker = 0; worker < 9; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 9;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = worker / 3 * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
            locals[worker] = storage.local_context(&context);
        }
        producer = locals[producer_worker];
        controls = scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset);
    }

    void *base() { return storage.scheduler_state->base(); }
    void register_successors() {
        // Registration prepends, so reverse registration gives the resolver
        // the deterministic chain 2 -> 3 -> ... -> count + 1.
        for (uint32_t task = count + 2; task-- > 2;)
            ASSERT_EQ(
                scheduler_route_task(graph.graph(), base(), producer, storage.run_control, task),
                SchedulerRouteResult::WAITING
            );
        ASSERT_EQ(controls[0].wake_list_head, 2);
    }
    bool resolve(uint32_t task = 0, SchedulerReadyClaim *direct = nullptr) {
        scheduler_gm_store(controls[task].state, static_cast<int64_t>(SchedulerTaskState::DONE));
        return scheduler_resolve_completion(
            graph.graph(), base(), producer, storage.run_control, task, 0, true, direct, producer->config.self_core_type
        );
    }
    bool pop(int64_t *task, uint32_t worker = 3) {
        return pop_ready(graph.graph(), base(), locals[worker], storage.run_control, 0, task);
    }
    SchedulerTicketQueueView queue() {
        const auto *pool = scheduler_state_at<SchedulerReadyPoolHeader>(base(), producer->config.ready_pool_offset);
        return scheduler_ready_pool_queue(base(), *pool, 0);
    }
    bool empty() {
        return scheduler_gm_query(queue().control->head) == scheduler_gm_query(queue().control->reserved_tail);
    }

    uint32_t count;
    FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<128> graph;
    std::array<SchedulerLocalState *, 9> locals{};
    SchedulerLocalState *producer{};
    SchedulerTaskControl *controls{};
};

TEST(SchedulerReadyWake, MixFanoutUsesSharedQueueAndPreservesFifoWithoutDirectHandoff) {
    for (uint32_t producer_worker : {0u, 1u, 2u}) {
        for (uint32_t count : {1u, 2u, 8u, 9u}) {
            for (bool older_queued : {false, true}) {
                SCOPED_TRACE(producer_worker);
                SCOPED_TRACE(count);
                SCOPED_TRACE(older_queued);
                MixPrefixFixture fixture(count, producer_worker);
                fixture.register_successors();
                if (older_queued) {
                    SchedulerReadyBatch older{};
                    ASSERT_TRUE(scheduler_ready_batch_append(fixture.base(), fixture.producer, count + 2, &older));
                    ASSERT_TRUE(scheduler_ready_batch_push(fixture.base(), fixture.producer, 0, &older));
                }
                SchedulerReadyClaim direct{};
                ASSERT_TRUE(fixture.resolve(0, producer_worker == 0 ? &direct : nullptr));

                EXPECT_EQ(direct.task_ref, SCHEDULER_TASK_REF_INVALID);
                int64_t popped = SCHEDULER_TASK_REF_INVALID;
                if (older_queued) {
                    ASSERT_TRUE(fixture.pop(&popped));
                    EXPECT_EQ(popped, count + 2);
                }
                for (uint32_t task = 2; task < count + 2; ++task) {
                    ASSERT_TRUE(fixture.pop(&popped));
                    EXPECT_EQ(popped, task);
                }
                ASSERT_TRUE(fixture.pop(&popped));
                EXPECT_EQ(popped, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
            }
        }
    }
}

TEST(SchedulerReadyWake, InvalidSuffixLeavesOnlyTheAlreadyPublishedMixPrefixReachable) {
    for (uint32_t corruption : {0u, 1u, 2u}) {
        SCOPED_TRACE(corruption);
        MixPrefixFixture fixture(10);
        fixture.register_successors();
        if (corruption == 1) fixture.controls[9].next_waiter = fixture.graph.graph().task_count;
        else if (corruption == 2) {
            fixture.graph.executable(10, 0, {0, 10});
            ASSERT_EQ(fixture.graph.payload(10).fanin_count, 2);
        } else {
            fixture.storage.metadata[10].active_mask = LaneMask::none();
            fixture.storage.metadata[10].flags &= ~SCHEDULER_TASK_MIX;
        }
        SchedulerReadyClaim direct{};
        EXPECT_FALSE(fixture.resolve(0, &direct));

        EXPECT_EQ(direct.task_ref, SCHEDULER_TASK_REF_INVALID);
        const auto queue = fixture.queue();
        EXPECT_EQ(queue.control->reserved_tail, 8u);
        EXPECT_EQ(queue.control->head, 0u);
        for (uint64_t index = 0; index < 8; ++index)
            EXPECT_EQ(queue.entries[index].task_ref, static_cast<int64_t>(index + 2));
        EXPECT_EQ(
            fixture.storage.run_control->error_site,
            static_cast<uint64_t>(
                corruption == 1 ? SchedulerErrorSite::COMPLETION_INVALID_WAITER :
                corruption == 2 ? SchedulerErrorSite::UNKNOWN :
                                  SchedulerErrorSite::COMPLETION_INVALID_SHAPE
            )
        );
        if (corruption == 2)
            EXPECT_EQ(
                fixture.storage.run_control->scheduler_error,
                static_cast<uint64_t>(SchedulerGraphResult::INVALID_FANIN_ID)
            );
        // Inspect terminal state only: no consumer or kernel runs after the error.
    }
}

TEST(SchedulerReadyWake, ErrorsExposeOnlyThePrefixPublishedBeforeFaninTraversal) {
    for (uint32_t bad_position : {0u, 1u}) {
        for (uint32_t corruption : {0u, 1u, 2u}) {
            SCOPED_TRACE(bad_position);
            SCOPED_TRACE(corruption);
            MixPrefixFixture fixture;
            fixture.register_successors();
            if (corruption == 1) {
                if (bad_position == 0) fixture.controls[0].wake_list_head = fixture.graph.graph().task_count;
                else fixture.controls[2].next_waiter = fixture.graph.graph().task_count;
            } else if (corruption == 2) {
                const uint32_t task = 2 + bad_position;
                fixture.graph.executable(task, 0, {0, static_cast<int32_t>(task)});
                ASSERT_EQ(fixture.graph.payload(task).fanin_count, 2);
            } else {
                fixture.storage.metadata[2 + bad_position].active_mask = LaneMask::none();
                fixture.storage.metadata[2 + bad_position].flags &= ~SCHEDULER_TASK_MIX;
            }
            SchedulerReadyClaim direct{};
            EXPECT_FALSE(fixture.resolve(0, &direct));
            // A pending prefix is published before a multi-fanin traversal,
            // even when that traversal subsequently finds a malformed edge.
            const uint64_t published = bad_position == 1 && corruption == 2 ? 1 : 0;

            EXPECT_EQ(fixture.queue().control->reserved_tail, published);
            EXPECT_EQ(fixture.queue().control->head, 0u);
            if (published != 0) EXPECT_EQ(fixture.queue().entries[0].task_ref, 2);

            EXPECT_NE(fixture.storage.run_control->scheduler_error, 0u);
            if (corruption == 2)
                EXPECT_EQ(
                    fixture.storage.run_control->scheduler_error,
                    static_cast<uint64_t>(SchedulerGraphResult::INVALID_FANIN_ID)
                );
        }
    }
}

TEST(SchedulerReadyWake, MixPrefixMigrationKeepsTheRemainingWakeChainAndPublishesEachReadyTaskOnce) {
    for (uint32_t migrating : {2u, 3u, 4u, 5u}) {
        SCOPED_TRACE(migrating);
        MixPrefixFixture fixture;
        fixture.graph.executable(migrating, 0, {0, 1});
        fixture.register_successors();
        SchedulerReadyClaim direct{};
        ASSERT_TRUE(fixture.resolve(0, &direct));
        const uint64_t initial_batches = (migrating > 2 ? 1u : 0u) + (migrating < 5 ? 1u : 0u);

        EXPECT_EQ(direct.task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.controls[1].wake_list_head, migrating);
        EXPECT_EQ(fixture.controls[migrating].waiting_producer, 1);
        EXPECT_EQ(fixture.controls[migrating].next_fanin_index, 1);
        int64_t popped = SCHEDULER_TASK_REF_INVALID;
        for (uint32_t task = 2; task < 6; ++task) {
            if (task == migrating) continue;
            ASSERT_TRUE(fixture.pop(&popped));
            EXPECT_EQ(popped, task);
        }
        ASSERT_TRUE(fixture.pop(&popped));
        EXPECT_EQ(popped, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.resolve(1));

        ASSERT_TRUE(fixture.pop(&popped));
        EXPECT_EQ(popped, migrating);
        ASSERT_TRUE(fixture.pop(&popped));
        EXPECT_EQ(popped, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
    }
}

TEST(SchedulerReadyWake, ConcurrentConsumersReceiveEveryEarlyMixPrefixAndSuffixTaskExactlyOnce) {
    constexpr uint32_t count = 64;
    MixPrefixFixture fixture(count, 1);
    fixture.register_successors();
    std::array<std::atomic<uint32_t>, count> seen{};
    std::atomic<bool> resolved{false};
    std::atomic<bool> failed{false};
    std::atomic<uint32_t> started{0};
    std::vector<std::thread> consumers;
    for (uint32_t worker : {0u, 3u, 6u}) {
        consumers.emplace_back([&, worker] {
            started.fetch_add(1, std::memory_order_release);
            for (uint32_t spin = 0; spin < 10000000; ++spin) {
                const bool producer_done = resolved.load(std::memory_order_acquire);
                int64_t task = SCHEDULER_TASK_REF_INVALID;
                if (!fixture.pop(&task, worker)) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
                if (task >= 0) {
                    if (task < 2 || task >= count + 2) failed.store(true, std::memory_order_release);
                    else seen[task - 2].fetch_add(1, std::memory_order_relaxed);
                } else if (producer_done) {
                    return;
                }
                std::this_thread::yield();
            }
            failed.store(true, std::memory_order_release);
        });
    }
    while (started.load(std::memory_order_acquire) != 3)
        std::this_thread::yield();
    if (!fixture.resolve()) failed.store(true, std::memory_order_release);
    resolved.store(true, std::memory_order_release);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_FALSE(failed.load());
    // All MIX refs use the same bounded publication as other ready work.

    for (const auto &occurrences : seen)
        EXPECT_EQ(occurrences.load(), 1u);
    EXPECT_TRUE(fixture.empty());
    EXPECT_EQ(fixture.storage.run_control->scheduler_error, 0u);
}

}  // namespace
