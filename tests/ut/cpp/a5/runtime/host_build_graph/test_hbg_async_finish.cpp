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
#include <cstring>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "ready_pool_plan.h"
#include "scheduler/scheduler_async_finish.h"
#include "scheduler/scheduler_graph_control.h"

namespace {
constexpr uint64_t kEpoch = 23;

struct FinishFixture {
    AicoreSchedulerLayout layout{};
    SchedulerReadyPoolPlan pool{};
    std::unique_ptr<scheduler_test::SchedulerStateBuffer> storage;
    scheduler_test::BasicGraphBuffer<256> graph_storage;
    SchedulerGraphView graph{};
    std::vector<SchedulerLocalState> locals{6};
    SchedulerRunControl *run{nullptr};
    SchedulerAsyncFinishIdentity identity{};
    SchedulerAsyncTaskJoin join{};
    SchedulerAsyncLease lease{};
    SchedulerAsyncDirectoryWord directory{};
    SchedulerAsyncFinishProgress saved{};
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    uint64_t owned{0};

    FinishFixture(
        uint32_t count, SchedulerReadyPoolKind kind = SchedulerReadyPoolKind::TICKET, bool legacy = false,
        bool control_node = false, bool mixed_types = false
    ) :
        graph_storage(count) {
        std::vector<SchedulerTaskMetadata> metadata(count);
        for (uint32_t task = 0; task < count; ++task) {
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
            metadata[task].active_mask = LaneMask::aic();
            metadata[task].logical_block_num = 1;
            metadata[task].timing_slot = -1;
            graph_storage.executable(task, 0);
            if (mixed_types && task != 0) {
                const uint32_t type = (task - 1) % 3;
                metadata[task].active_mask = type == 0 ? LaneMask::aic() :
                                             type == 1 ? LaneMask::aiv0() :
                                                         LaneMask::all();
                if (type == SCHEDULER_MIX_QUEUE) metadata[task].flags |= SCHEDULER_TASK_MIX;
            }
        }
        if (control_node) metadata[1] = {};
        EXPECT_TRUE(scheduler_plan_layout(count, count, 0, &layout));
        if (!legacy) {
            EXPECT_TRUE(scheduler_plan_ready_pool(
                metadata, 2, 0, layout.total_size, &pool, kind, kind == SchedulerReadyPoolKind::CHUNK ? 4 : 0
            ));
            layout.total_size = pool.total_size;
        }
        SchedulerNativeGraphHeader native{};
        uint64_t native_offset = 0;
        if (control_node) {
            // Only TOP/CONTROL_DUMMY bindings are reached in this fixture. The
            // native header still has its complete, checked region envelope.
            auto &cursor = layout.total_size;
            EXPECT_TRUE(scheduler_layout_reserve(&cursor, sizeof(native), 128, &native_offset));
            EXPECT_TRUE(
                scheduler_layout_reserve(&cursor, count * sizeof(SchedulerNodeBinding), 128, &native.bindings_offset)
            );
            EXPECT_TRUE(
                scheduler_layout_reserve(&cursor, count * sizeof(int64_t), 128, &native.completion_refs_offset)
            );
            EXPECT_TRUE(
                scheduler_layout_reserve(&cursor, sizeof(SchedulerGraphTemplate), 128, &native.templates_offset)
            );
            EXPECT_TRUE(
                scheduler_layout_reserve(&cursor, sizeof(SchedulerGraphInstance), 128, &native.instances_offset)
            );
            EXPECT_TRUE(scheduler_layout_checked_align(cursor, 128, &native.readonly_end_offset));
            cursor = native.readonly_end_offset;
            EXPECT_TRUE(scheduler_layout_reserve(
                &cursor, locals.size() * sizeof(SchedulerReadyInbox), 128, &native.control_inboxes_offset
            ));
            EXPECT_TRUE(scheduler_layout_checked_align(cursor, 128, &layout.total_size));
            native.magic = SCHEDULER_NATIVE_GRAPH_MAGIC;
            native.version = SCHEDULER_NATIVE_GRAPH_VERSION;
            native.header_bytes = sizeof(native);
            native.image_bytes = layout.total_size;
            native.node_count = native.original_task_count = count;
            native.definition_count = native.instance_count = 1;
            native.control_inbox_count = locals.size();
            native.run_epoch = kEpoch;
        }
        storage = std::make_unique<scheduler_test::SchedulerStateBuffer>(layout);
        if (!legacy) scheduler_init_ready_pool(base(), pool);
        std::copy(
            metadata.begin(), metadata.end(),
            scheduler_state_at<SchedulerTaskMetadata>(base(), layout.task_metadata_offset)
        );
        run = scheduler_state_at<SchedulerRunControl>(base(), layout.run_control_offset);
        scheduler_state_at<SchedulerGraphSummary>(base(), layout.graph_summary_offset)->run_epoch = kEpoch;
        scheduler_gm_store(run->bootstrap_scan_complete, uint64_t{1});
        graph = graph_storage.graph();
        graph.scheduler_state_address = reinterpret_cast<uint64_t>(base());
        graph.scheduler_state_bytes = layout.total_size;
        graph.native_header_offset = native_offset;
        if (control_node) {
            *scheduler_state_at<SchedulerNativeGraphHeader>(base(), native_offset) = native;
            auto *bindings = scheduler_state_at<SchedulerNodeBinding>(base(), native.bindings_offset);
            auto *completions = scheduler_state_at<int64_t>(base(), native.completion_refs_offset);
            for (uint32_t task = 0; task < count; ++task) {
                bindings[task] = {
                    task,
                    SCHEDULER_NATIVE_NO_INSTANCE,
                    task == 1 ? SchedulerNodeKind::CONTROL_DUMMY : SchedulerNodeKind::TOP_KERNEL,
                    {}
                };
                completions[task] = task;
            }
            auto *inboxes = scheduler_state_at<SchedulerReadyInbox>(base(), native.control_inboxes_offset);
            for (size_t worker = 0; worker < locals.size(); ++worker)
                scheduler_gm_store(inboxes[worker].endpoints, SCHEDULER_READY_ENDPOINTS_EMPTY);
        }
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &local = locals[worker];
            auto &config = local.config;
            config.graph = &graph;
            config.graph_summary_offset = layout.graph_summary_offset;
            config.task_controls_offset = layout.task_controls_offset;
            config.task_metadata_offset = layout.task_metadata_offset;
            config.ready_inboxes_offset = layout.ready_inboxes_offset;
            config.ready_directory_offset = layout.ready_directory_offset;
            config.control_inboxes_offset = native.control_inboxes_offset;
            config.ready_pool_offset = legacy ? 0 : pool.header_offset;
            config.cluster_count = 2;
            config.runtime_worker_count = config.scheduler_count = locals.size();
            config.scheduler_index = worker;
            config.self_lane = worker % 3;
            config.self_core_type = worker % 3 == 0 ? 0 : 1;
            config.cluster_type_ranks[0] = worker / 3;
            config.cluster_type_ranks[1] = (worker / 3) * 2;
            config.cluster_type_ranks[2] = (worker / 3) * 2 + 1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                config.worker_ids[lane] = (worker / 3) * 3 + lane;
        }
        identity.run_epoch = kEpoch;
        identity.task_ref = 0;
        identity.group_base = 0;
        identity.group_count = 1;
        EXPECT_EQ(scheduler_async_lease_publish(lease).result, SchedulerAsyncLeaseResult::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_publish_once(directory, 0), SchedulerAsyncDirectoryResult::SUCCESS);
    }
    void *base() const { return storage->base(); }
    SchedulerTaskControl *control(int64_t ref) { return scheduler_task_control_at(base(), &locals[0], ref); }
    void fanout(uint32_t begin, uint32_t end) {
        scheduler_gm_store(control(0)->wake_list_head, static_cast<int64_t>(begin));
        for (uint32_t ref = begin; ref < end; ++ref) {
            graph_storage.executable(ref, 0, {0});
            scheduler_gm_store(
                control(ref)->next_waiter, ref + 1 == end ? SCHEDULER_TASK_REF_INVALID : static_cast<int64_t>(ref + 1)
            );
        }
    }
    uint64_t completed() const {
        uint64_t count = 0;
        for (const auto &local : locals)
            count += local.pending_completed;
        return count;
    }
    template <typename Ops = SchedulerAsyncFinishQueueOps, typename Clock = SchedulerAsyncFinishClock>
    SchedulerAsyncFinishResult slice(uint32_t worker, uint32_t budget, uint64_t profiling_level = 0) {
        const auto claim = scheduler_async_lease_try_claim(lease, scheduler_async_lease_load(lease));
        EXPECT_EQ(claim.result, SchedulerAsyncLeaseResult::SUCCESS);
        owned = claim.word;
        auto progress = saved;
        const auto result = scheduler_async_finish_step<Ops, SchedulerResolveLink, Clock>(
            graph, base(), &locals[worker], run, identity, &join, &progress, &budget, &wake, &ready, profiling_level
        );
        saved = progress;
        for (const auto &local : locals)
            EXPECT_FALSE(scheduler_ready_pool_has_pending(&local));
        if (result == SchedulerAsyncFinishResult::PENDING)
            EXPECT_EQ(
                scheduler_async_lease_handoff(lease, owned, SchedulerAsyncPhase::RUNNABLE).result,
                SchedulerAsyncLeaseResult::SUCCESS
            );
        return result;
    }
    bool count_drained(uint32_t worker) {
        return scheduler_async_finish_contribute(&locals[worker], run, join, lease, directory, 0, &saved, &completion);
    }
    void terminal(uint32_t worker) {
        EXPECT_FALSE(count_drained(worker));
        EXPECT_EQ(
            scheduler_async_lease_handoff(lease, owned, SchedulerAsyncPhase::TERMINAL).result,
            SchedulerAsyncLeaseResult::SUCCESS
        );
        EXPECT_FALSE(count_drained(worker));
        EXPECT_EQ(scheduler_async_directory_clear_once(directory, 0), SchedulerAsyncDirectoryResult::SUCCESS);
        EXPECT_TRUE(count_drained(worker));
    }
    std::vector<int64_t> take_all() {
        std::vector<int64_t> result;
        for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
            if (locals[0].config.ready_pool_offset == 0) {
                for (uint64_t owner = 0; owner < scheduler_ready_queue_owner_count(&locals[0], type); ++owner) {
                    auto *inbox = scheduler_ready_inbox_at(base(), &locals[0], type, owner);
                    for (uint32_t n = 0; n < graph.task_count; ++n) {
                        int64_t ref = SCHEDULER_TASK_REF_INVALID;
                        EXPECT_TRUE(
                            scheduler_ready_pop_from_queue(graph, base(), &locals[0], run, inbox, &ref, nullptr)
                        );
                        if (ref < 0) break;
                        result.push_back(ref);
                    }
                }
            } else if (pool.header.kind == SchedulerReadyPoolKind::TICKET) {
                for (uint32_t group = 0; group < pool.header.group_counts[type]; ++group) {
                    const auto queue =
                        scheduler_ready_pool_queue(base(), pool.header, pool.header.queue_begins[type] + group);
                    for (uint32_t n = 0; n < graph.task_count; ++n) {
                        int64_t refs[2]{};
                        uint32_t count = 0;
                        auto status = scheduler_ticket_try_pop_batch(queue, refs, 2, &count);
                        if (status == SchedulerTicketResult::NOT_READY) break;
                        EXPECT_EQ(status, SchedulerTicketResult::SUCCESS);
                        result.insert(result.end(), refs, refs + count);
                    }
                }
            } else {
                const auto queue =
                    scheduler_ready_pool_linked_queue(base(), pool.header, pool.header.queue_begins[type]);
                for (uint32_t n = 0; n < graph.task_count; ++n) {
                    int64_t refs[2]{};
                    uint32_t count = 0;
                    SchedulerLinkedResult status;
                    if (pool.header.kind == SchedulerReadyPoolKind::LINKED) {
                        status = scheduler_linked_try_pop_batch(queue, refs, 2, &count);
                    } else {
                        const auto chunks =
                            scheduler_ready_pool_chunk_queue(base(), pool.header, pool.header.queue_begins[type]);
                        int64_t window[1]{};
                        uint32_t found = 0;
                        status = scheduler_chunk_window(chunks, window, &found);
                        if (status == SchedulerLinkedResult::SUCCESS) {
                            SchedulerChunkCandidate candidate{};
                            status = scheduler_chunk_peek(chunks, window[0], &candidate);
                            if (status == SchedulerLinkedResult::SUCCESS)
                                status = scheduler_chunk_try_claim(chunks, candidate, 2, refs, &count);
                        }
                    }
                    if (status == SchedulerLinkedResult::NOT_READY) break;
                    EXPECT_EQ(status, SchedulerLinkedResult::SUCCESS);
                    result.insert(result.end(), refs, refs + count);
                }
            }
        }
        return result;
    }
};

TEST(HbgAsyncFinish, CompletedCursorForAnotherTaskCannotSkipCurrentWaiter) {
    FinishFixture fixture(2);
    scheduler_gm_store(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    scheduler_gm_store(fixture.control(0)->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    fixture.saved.phase = static_cast<uint64_t>(SchedulerAsyncFinishPhase::RESOLVE);
    fixture.saved.current_waiter = 1;
    fixture.saved.detached_next = SCHEDULER_TASK_REF_INVALID;
    fixture.saved.resolve = scheduler_resolve_begin(0);
    fixture.saved.resolve.phase = SchedulerResolvePhase::COMPLETED;

    EXPECT_EQ(fixture.slice(0, 8), SchedulerAsyncFinishResult::ERROR);
    EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::FAILED));
    EXPECT_NE(scheduler_read_error(fixture.run), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_TRUE(fixture.take_all().empty());
    EXPECT_EQ(fixture.completed(), 0u);
}

TEST(HbgAsyncFinish, ZeroBudgetAndEmptyFanoutRespectPublicationAndTerminalOrder) {
    FinishFixture fixture(1);
    EXPECT_EQ(fixture.slice(0, 0), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    EXPECT_EQ(fixture.slice(3, 1), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(fixture.completed(), 0u);
    EXPECT_EQ(fixture.slice(1, 1), SchedulerAsyncFinishResult::DONE);
    fixture.terminal(1);
    EXPECT_EQ(fixture.completed(), 1u);
    EXPECT_TRUE(fixture.count_drained(4));
    EXPECT_EQ(fixture.completed(), 1u);
    EXPECT_EQ(fixture.completion.resolve_count, 1u);
    EXPECT_EQ(fixture.wake.wake_close_count, 1u);
}

TEST(HbgAsyncFinish, NonzeroFailedJoinAndRunErrorNeverPublishTaskDone) {
    for (uint32_t scenario = 0; scenario < 3; ++scenario) {
        FinishFixture fixture(1);
        if (scenario == 0) fixture.join.word = UINT64_C(1) << 16;
        if (scenario == 1) scheduler_async_join_abort(fixture.join);
        if (scenario == 2)
            scheduler_record_error(
                fixture.run, 0, SchedulerGraphResult::INVALID_ARGUMENTS, &fixture.graph, &fixture.locals[0]
            );
        EXPECT_EQ(fixture.slice(0, 20), SchedulerAsyncFinishResult::ERROR);
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->state), static_cast<int64_t>(SchedulerTaskState::BLOCKED));
        EXPECT_EQ(scheduler_gm_load(fixture.control(0)->wake_list_head), SCHEDULER_WAKE_LIST_OPEN);
        EXPECT_EQ(fixture.completed(), 0u);
    }
}

TEST(HbgAsyncFinish, AllReadyPoolsPublishEachSuccessorOnceAcrossSingleUnitHandoffs) {
    for (auto kind : {SchedulerReadyPoolKind::TICKET, SchedulerReadyPoolKind::LINKED, SchedulerReadyPoolKind::CHUNK}) {
        FinishFixture fixture(26, kind);
        fixture.fanout(1, 26);
        SchedulerAsyncFinishResult result = SchedulerAsyncFinishResult::PENDING;
        uint32_t worker = 0;
        for (uint32_t slice = 0; slice < 400 && result == SchedulerAsyncFinishResult::PENDING; ++slice) {
            worker = slice % fixture.locals.size();
            result = fixture.slice(worker, 1);
            EXPECT_EQ(fixture.completed(), 0u);
        }
        ASSERT_EQ(result, SchedulerAsyncFinishResult::DONE);
        auto actual = fixture.take_all();
        std::sort(actual.begin(), actual.end());
        std::vector<int64_t> expected(25);
        std::iota(expected.begin(), expected.end(), 1);
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(fixture.ready.enqueue_count, 25u);
        EXPECT_EQ(fixture.ready.batch_count, 4u);
        fixture.terminal(worker);
        EXPECT_EQ(fixture.completed(), 1u);
        EXPECT_EQ(scheduler_read_error(fixture.run), 0u);
    }
}

struct RetryReserve : SchedulerAsyncFinishQueueOps {
    inline static uint32_t remaining{0};
    inline static uint32_t reservations{0};
    inline static uint64_t largest{0};
    static SchedulerTicketResult
    reserve(const SchedulerTicketQueueView &queue, uint64_t count, SchedulerTicketRange *range) {
        if (remaining != 0) {
            --remaining;
            return SchedulerTicketResult::RETRY;
        }
        ++reservations;
        largest = std::max(largest, count);
        return scheduler_ticket_reserve(queue, count, range);
    }
};

TEST(HbgAsyncFinish, TicketRetryRetainsBatchAndSuccessfulReservationPublishesBeforeYield) {
    FinishFixture fixture(10);
    fixture.fanout(1, 10);
    RetryReserve::remaining = 3;
    RetryReserve::reservations = 0;
    RetryReserve::largest = 0;
    for (uint32_t worker : {0u, 3u, 1u}) {
        ASSERT_EQ(fixture.slice<RetryReserve>(worker, 200), SchedulerAsyncFinishResult::PENDING);
        EXPECT_EQ(fixture.saved.pending_count, 8u);
        EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH));
        EXPECT_EQ(RetryReserve::reservations, 0u);
        EXPECT_TRUE(fixture.take_all().empty());
    }
    // One unit remains: reservation and all eight stores must complete together.
    ASSERT_EQ(fixture.slice<RetryReserve>(4, 1), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(fixture.saved.pending_count, 0u);
    EXPECT_EQ(RetryReserve::reservations, 1u);
    EXPECT_EQ(fixture.take_all().size(), 8u);
    ASSERT_EQ(fixture.slice<RetryReserve>(2, 200), SchedulerAsyncFinishResult::DONE);
    EXPECT_EQ(fixture.take_all(), std::vector<int64_t>{9});
    EXPECT_EQ(RetryReserve::reservations, 2u);
    EXPECT_EQ(RetryReserve::largest, 8u);
    fixture.terminal(2);
}

TEST(HbgAsyncFinish, LinkedAndChunkRetryKeepPreparedBatchOutsideWorkerPrivateState) {
    for (auto kind : {SchedulerReadyPoolKind::LINKED, SchedulerReadyPoolKind::CHUNK}) {
        FinishFixture fixture(13, kind);
        fixture.fanout(1, 10);
        const auto queue = scheduler_ready_pool_linked_queue(fixture.base(), fixture.pool.header, 0);
        const int64_t prefix[] = {10, 11, 12};
        SchedulerLinkedBatch batch{};
        if (kind == SchedulerReadyPoolKind::LINKED) {
            ASSERT_EQ(scheduler_linked_prepare(queue, prefix, 3, &batch), SchedulerLinkedResult::SUCCESS);
        } else {
            const auto chunks = scheduler_ready_pool_chunk_queue(fixture.base(), fixture.pool.header, 0);
            for (const auto ref : prefix) {
                SchedulerLinkedBatch one{};
                ASSERT_EQ(scheduler_chunk_prepare(chunks, &ref, 1, &one), SchedulerLinkedResult::SUCCESS);
                ASSERT_EQ(scheduler_linked_append_private(queue, &batch, &one), SchedulerLinkedResult::SUCCESS);
            }
        }
        // A valid publisher pauses after its publication CAS and before optional
        // tail helping. Each FINISH slice may help only one of these old links.
        ASSERT_EQ(
            scheduler_gm_compare_exchange(queue.links[queue.sentinel], SCHEDULER_TASK_REF_INVALID, batch.first),
            SCHEDULER_TASK_REF_INVALID
        );
        for (uint32_t worker : {0u, 3u, 1u}) {
            ASSERT_EQ(fixture.slice(worker, 200), SchedulerAsyncFinishResult::PENDING);
            EXPECT_EQ(fixture.saved.pending_count, 8u);
            EXPECT_EQ(fixture.saved.pending_first, 1);
            EXPECT_EQ(fixture.saved.pending_last, kind == SchedulerReadyPoolKind::CHUNK ? 1 : 8);
            EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH));
        }
        ASSERT_EQ(fixture.slice(4, 200), SchedulerAsyncFinishResult::DONE);
        auto actual = fixture.take_all();
        std::sort(actual.begin(), actual.end());
        std::vector<int64_t> expected(12);
        std::iota(expected.begin(), expected.end(), 1);
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(fixture.ready.contention_giveup_count, 3u);
        fixture.terminal(4);
    }
}

TEST(HbgAsyncFinish, LongFaninAndFanoutShareTheActualWorkBudget) {
    FinishFixture fixture(201);
    fixture.fanout(128, 201);
    std::vector<int32_t> fanins(128);
    std::iota(fanins.begin(), fanins.end(), 0);
    fixture.graph_storage.executable(128, 0, fanins);
    for (uint32_t producer = 1; producer < 128; ++producer) {
        scheduler_gm_store(fixture.control(producer)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        scheduler_gm_store(fixture.control(producer)->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    }
    SchedulerAsyncFinishResult result = SchedulerAsyncFinishResult::PENDING;
    uint32_t worker = 0;
    for (uint32_t slice = 0; slice < 300 && result == SchedulerAsyncFinishResult::PENDING; ++slice) {
        const auto before = fixture.wake.fanin_state_load_count;
        worker = slice % fixture.locals.size();
        result = fixture.slice(worker, 4);
        EXPECT_LE(fixture.wake.fanin_state_load_count - before, 4u);
        if (fixture.wake.fanin_state_load_count < 128) EXPECT_TRUE(fixture.take_all().empty());
    }
    ASSERT_EQ(result, SchedulerAsyncFinishResult::DONE);
    auto actual = fixture.take_all();
    std::sort(actual.begin(), actual.end());
    std::vector<int64_t> expected(73);
    std::iota(expected.begin(), expected.end(), 128);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(fixture.wake.fanin_state_load_count, 200u);
    EXPECT_EQ(fixture.wake.wake_close_count, 1u);
    fixture.terminal(worker);
}

TEST(HbgAsyncFinish, RelinkedWaiterCannotOverwriteTheSavedDetachedSuccessor) {
    FinishFixture fixture(5);
    fixture.graph_storage.executable(2, 0, {0, 1});
    fixture.graph_storage.executable(3, 0, {1});
    fixture.graph_storage.executable(4, 0, {0});
    scheduler_gm_store(fixture.control(0)->wake_list_head, int64_t{2});
    scheduler_gm_store(fixture.control(2)->next_waiter, int64_t{4});
    scheduler_gm_store(fixture.control(1)->wake_list_head, int64_t{3});
    ASSERT_EQ(fixture.slice(0, 5), SchedulerAsyncFinishResult::PENDING);
    ASSERT_EQ(fixture.saved.resolve.phase, SchedulerResolvePhase::LINK);
    ASSERT_EQ(fixture.slice(3, 1), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->wake_list_head), 2);
    EXPECT_EQ(scheduler_gm_load(fixture.control(2)->next_waiter), 3);
    ASSERT_EQ(fixture.slice(1, 40), SchedulerAsyncFinishResult::DONE);
    EXPECT_EQ(fixture.take_all(), std::vector<int64_t>{4});
    fixture.terminal(1);
}

struct RetryInbox : SchedulerAsyncFinishQueueOps {
    inline static uint32_t remaining{0};
    static uint64_t compare_exchange(volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        if (remaining != 0) {
            --remaining;
            return expected ^ 1;
        }
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

TEST(HbgAsyncFinish, LegacyInboxTargetAndBatchSurviveRetryOnAnotherCluster) {
    FinishFixture fixture(4, SchedulerReadyPoolKind::TICKET, true);
    fixture.fanout(1, 4);
    RetryInbox::remaining = 1;
    ASSERT_EQ(fixture.slice<RetryInbox>(0, 100), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(fixture.saved.pending_target, 0u);
    ASSERT_EQ(fixture.slice<RetryInbox>(3, 100), SchedulerAsyncFinishResult::DONE);
    EXPECT_TRUE(scheduler_ready_endpoints_empty(
        scheduler_gm_query(scheduler_ready_inbox_at(fixture.base(), &fixture.locals[0], 0, 1)->endpoints)
    ));
    EXPECT_EQ(fixture.take_all(), (std::vector<int64_t>{1, 2, 3}));
    fixture.terminal(3);
}

TEST(HbgAsyncFinish, ControlSuccessorKeepsItsTargetAcrossRetryThenUsesExistingControlService) {
    FinishFixture fixture(3, SchedulerReadyPoolKind::TICKET, false, true);
    fixture.graph_storage.executable(1, 0, {0});
    fixture.graph_storage.executable(2, 0, {1});
    scheduler_gm_store(fixture.control(0)->wake_list_head, int64_t{1});
    scheduler_gm_store(fixture.control(1)->wake_list_head, int64_t{2});
    RetryInbox::remaining = 1;
    ASSERT_EQ(fixture.slice<RetryInbox>(0, 100), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(fixture.saved.pending_kind, static_cast<uint64_t>(SchedulerAsyncFinishQueue::CONTROL));
    EXPECT_EQ(fixture.saved.pending_target, 0u);
    ASSERT_EQ(fixture.slice<RetryInbox>(3, 100), SchedulerAsyncFinishResult::DONE);
    auto *inboxes =
        scheduler_state_at<SchedulerReadyInbox>(fixture.base(), fixture.locals[0].config.control_inboxes_offset);
    EXPECT_FALSE(scheduler_ready_endpoints_empty(scheduler_gm_query(inboxes[0].endpoints)));
    EXPECT_TRUE(scheduler_ready_endpoints_empty(scheduler_gm_query(inboxes[3].endpoints)));
    fixture.terminal(3);
    bool progress = false;
    ASSERT_TRUE(
        scheduler_service_controls(fixture.graph, fixture.base(), &fixture.locals[0], fixture.run, 0, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.take_all(), std::vector<int64_t>{2});
    EXPECT_EQ(fixture.completed(), 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run), 0u);
}

TEST(HbgAsyncFinish, ErrorAfterPrivatePreparationNeverPublishesOrContributes) {
    FinishFixture fixture(2, SchedulerReadyPoolKind::CHUNK);
    fixture.fanout(1, 2);
    SchedulerAsyncFinishResult result = SchedulerAsyncFinishResult::PENDING;
    for (uint32_t slice = 0;
         slice < 20 && fixture.saved.phase != static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH); ++slice)
        ASSERT_EQ(result = fixture.slice(slice % 6, 1), SchedulerAsyncFinishResult::PENDING);
    ASSERT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH));
    scheduler_record_error(fixture.run, 0, SchedulerGraphResult::INVALID_ARGUMENTS, &fixture.graph, &fixture.locals[0]);
    EXPECT_EQ(fixture.slice(3, 100), SchedulerAsyncFinishResult::ERROR);
    EXPECT_TRUE(fixture.take_all().empty());
    EXPECT_EQ(fixture.completed(), 0u);
    EXPECT_FALSE(fixture.count_drained(3));
}

TEST(HbgAsyncFinish, QueueTypeChangesFlushOnlyTheOwnedPrefixBeforeResumingReadyTask) {
    for (uint32_t variant = 0; variant < 4; ++variant) {
        const auto kind = variant < 3 ? static_cast<SchedulerReadyPoolKind>(variant) : SchedulerReadyPoolKind::TICKET;
        FinishFixture fixture(13, kind, variant == 3, false, true);
        fixture.fanout(1, 13);
        SchedulerAsyncFinishResult result = SchedulerAsyncFinishResult::PENDING;
        uint32_t worker = 0;
        for (uint32_t slice = 0; slice < 400 && result == SchedulerAsyncFinishResult::PENDING; ++slice) {
            worker = slice % 6;
            result = fixture.slice(worker, 1);
        }
        ASSERT_EQ(result, SchedulerAsyncFinishResult::DONE);
        auto actual = fixture.take_all();
        std::sort(actual.begin(), actual.end());
        std::vector<int64_t> expected(12);
        std::iota(expected.begin(), expected.end(), 1);
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(fixture.ready.enqueue_count, 12u);
        EXPECT_EQ(fixture.ready.batch_count, 12u);
        fixture.terminal(worker);
    }
}

TEST(HbgAsyncFinish, CooperativeTaskDoneUsesPersistentOrderOnAnUnrelatedObserverCore) {
    scheduler_test::FixtureStorage storage(1, 6, 3, {6, 2, 1, 0}, true);
    scheduler_test::BasicGraphBuffer<1> graph_storage(1);
    graph_storage.executable(0, 0);
    auto &metadata = storage.metadata[0];
    metadata.logical_block_num = 2;
    metadata.flags |= SCHEDULER_TASK_SPMD;
    metadata.timing_slot = -1;
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, kEpoch));
    const uint32_t worker = 3;
    auto &context = storage.contexts[worker];
    context.scheduler_count = 6;
    for (uint32_t lane = 0; lane < 3; ++lane)
        context.cluster_worker_ids[lane] = 3 + lane;
    storage.test_graphs[worker] = graph_storage.graph();
    auto *local = storage.local_context(&context);
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    SchedulerAdmissionLocalState admission{};
    admission.epoch = kEpoch;
    admission.records_offset = storage.layout.admission_records_offset;
    admission.record_stride = storage.layout.admission_record_stride;
    admission.capacity = 1;
    // The observer has already advanced its local admission frontier. TaskDone
    // must use the immutable task record, not require this old task locally.
    admission.scan_cursor = 1;
    admission.current_task_ref = SCHEDULER_TASK_REF_INVALID;
    local->admission = &admission;
    auto *record = scheduler_state_at<SchedulerAdmissionRecord>(base, admission.records_offset);
    record->run_epoch = kEpoch;
    record->order_key = 0;
    SchedulerAsyncTaskJoin join{};
    SchedulerAsyncFinishIdentity identity{};
    identity.run_epoch = kEpoch;
    identity.task_ref = 0;
    SchedulerAsyncFinishProgress progress{};
    uint32_t budget = 2;
    ASSERT_EQ(
        scheduler_async_finish_step(
            graph_storage.graph(), base, local, storage.run_control, identity, &join, &progress, &budget
        ),
        SchedulerAsyncFinishResult::DONE
    );
    auto *protocol = scheduler_protocol_layout_at(base, local);
    const auto *trace = scheduler_state_at<SchedulerProtocolTaskTrace>(base, protocol->tasks_offset);
    EXPECT_EQ(trace->completion.order_key, 0u);
    EXPECT_EQ(trace->completion.worker_index, worker);
    EXPECT_GT(trace->completion.before.sequence, 0u);
    EXPECT_GT(trace->completion.after.sequence, trace->completion.before.sequence);
    EXPECT_EQ(scheduler_protocol_worker_at(base, local, protocol)->error_flags, 0u);
    EXPECT_EQ(scheduler_read_error(storage.run_control), 0u);
}

struct ContendedDirectory : SchedulerAsyncFinishQueueOps {
    inline static volatile uint64_t *directory_word{nullptr};
    inline static uint32_t remaining{0};
    inline static uint32_t endpoint_attempts{0};
    inline static uint32_t directory_attempts{0};
    static uint64_t compare_exchange(volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        if (&word == directory_word) {
            ++directory_attempts;
            if (remaining != 0) {
                --remaining;
                // A neighbouring inbox changes its directory bit after this
                // publisher's load. The real comparison must fail once.
                scheduler_gm_store(word, expected ^ UINT64_C(2));
            }
        } else {
            ++endpoint_attempts;
        }
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

TEST(HbgAsyncFinish, DirectoryContentionYieldsAfterPublicationWithoutTouchingTransferredNodes) {
    FinishFixture fixture(2, SchedulerReadyPoolKind::TICKET, true);
    fixture.fanout(1, 2);
    auto *directory = scheduler_ready_directory_at(fixture.base(), &fixture.locals[0]);
    ContendedDirectory::directory_word = &directory->core_types[0][0].bits;
    ContendedDirectory::remaining = 3;
    ContendedDirectory::endpoint_attempts = ContendedDirectory::directory_attempts = 0;
    ASSERT_EQ(fixture.slice<ContendedDirectory>(0, 100), SchedulerAsyncFinishResult::PENDING);
    ASSERT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH_DIRECTORY));
    EXPECT_EQ(ContendedDirectory::endpoint_attempts, 1u);
    EXPECT_EQ(ContendedDirectory::directory_attempts, 1u);
    EXPECT_EQ(fixture.take_all(), std::vector<int64_t>{1});
    // The consumer owns this node already. Continuation may only mark the saved
    // queue's directory, even if its previous metadata has become unusable.
    scheduler_gm_store(fixture.control(1)->next_waiter, int64_t{-77});
    fixture.control(1)->next_fanin_index = -88;
    for (uint32_t worker : {3u, 1u}) {
        const auto attempts = ContendedDirectory::directory_attempts;
        ASSERT_EQ(fixture.slice<ContendedDirectory>(worker, 100), SchedulerAsyncFinishResult::PENDING);
        EXPECT_EQ(ContendedDirectory::directory_attempts, attempts + 1);
        EXPECT_EQ(ContendedDirectory::endpoint_attempts, 1u);
        EXPECT_EQ(fixture.completed(), 0u);
    }
    ASSERT_EQ(fixture.slice<ContendedDirectory>(4, 100), SchedulerAsyncFinishResult::DONE);
    EXPECT_EQ(ContendedDirectory::endpoint_attempts, 1u);
    EXPECT_EQ(ContendedDirectory::directory_attempts, 4u);
    EXPECT_EQ(fixture.control(1)->next_fanin_index, -88);
    EXPECT_EQ(scheduler_gm_load(fixture.control(1)->next_waiter), -77);
    EXPECT_TRUE(fixture.take_all().empty());
    EXPECT_EQ(fixture.ready.batch_count, 1u);
    fixture.terminal(4);
    ContendedDirectory::directory_word = nullptr;
}

struct OffsetSliceClock {
    inline static uint64_t stamp{0};
    static uint64_t now() {
        stamp += 7;
        return stamp;
    }
};

TEST(HbgAsyncFinish, PhaseTimingAccumulatesActualSlicesWithoutCrossCoreBusySpan) {
    FinishFixture fixture(4);
    fixture.fanout(1, 4);
    const auto original_worker = fixture.control(0)->scheduler_worker_id;
    const auto original_loop = fixture.control(0)->completion_resolve_loop_iter;
    SchedulerAsyncFinishResult result = SchedulerAsyncFinishResult::PENDING;
    uint32_t worker = 0;
    uint32_t calls = 0;
    for (; calls < 100 && result == SchedulerAsyncFinishResult::PENDING; ++calls) {
        worker = calls % 6;
        // Clock origins differ and wall time includes gaps. Only each call's
        // own seven-cycle interval belongs to FINISH's active work.
        OffsetSliceClock::stamp = UINT64_C(10000000) * worker + UINT64_C(1000) * calls;
        result = fixture.slice<SchedulerAsyncFinishQueueOps, OffsetSliceClock>(
            worker, calls == 0 ? 0 : 1, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL
        );
        EXPECT_EQ(fixture.saved.active_slice_cycles, 7 * (calls + 1));
        EXPECT_EQ(fixture.saved.slice_count, calls + 1);
        EXPECT_EQ(fixture.saved.work_units, calls);
    }
    ASSERT_EQ(result, SchedulerAsyncFinishResult::DONE);
    EXPECT_GT(calls, 6u);
    EXPECT_EQ(fixture.control(0)->completion_resolve_start_cycles, 0u);
    EXPECT_EQ(fixture.control(0)->completion_resolve_end_cycles, 0u);
    EXPECT_EQ(fixture.control(0)->scheduler_worker_id, original_worker);
    EXPECT_EQ(fixture.control(0)->completion_resolve_loop_iter, original_loop);
    fixture.terminal(worker);

    FinishFixture unprofiled(1);
    ASSERT_EQ(unprofiled.slice(0, 8), SchedulerAsyncFinishResult::DONE);
    EXPECT_EQ(unprofiled.saved.active_slice_cycles, 0u);
    EXPECT_EQ(unprofiled.saved.slice_count, 0u);
    EXPECT_EQ(unprofiled.saved.work_units, 0u);
    uint64_t saturated = UINT64_MAX - 3;
    scheduler_async_finish_accumulate(saturated, 7);
    EXPECT_EQ(saturated, UINT64_MAX);
}

struct ClearedDirectory : SchedulerAsyncFinishQueueOps {
    inline static volatile uint64_t *directory_word{nullptr};
    inline static bool clear_before_confirmation{true};
    inline static uint32_t confirmations{0};
    static uint64_t compare_exchange(volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        if (&word == directory_word) {
            ++confirmations;
            if (clear_before_confirmation) {
                clear_before_confirmation = false;
                // The bypass snapshot said one, but a consumer's atomic clear
                // has reached authoritative storage before confirmation.
                scheduler_gm_store(word, uint64_t{0});
            }
        }
        return scheduler_gm_compare_exchange(word, expected, desired);
    }
};

TEST(HbgAsyncFinish, DirectoryBitThatAppearsSetStillRequiresAtomicConfirmation) {
    FinishFixture fixture(2, SchedulerReadyPoolKind::TICKET, true);
    fixture.fanout(1, 2);
    auto *directory = scheduler_ready_directory_at(fixture.base(), &fixture.locals[0]);
    auto &word = directory->core_types[0][0].bits;
    scheduler_gm_store(word, uint64_t{1});
    ClearedDirectory::directory_word = &word;
    ClearedDirectory::clear_before_confirmation = true;
    ClearedDirectory::confirmations = 0;
    ASSERT_EQ(fixture.slice<ClearedDirectory>(0, 100), SchedulerAsyncFinishResult::PENDING);
    EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::PUBLISH_DIRECTORY));
    EXPECT_EQ(ClearedDirectory::confirmations, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(word, uint64_t{0}), 0u);
    ASSERT_EQ(fixture.slice<ClearedDirectory>(3, 100), SchedulerAsyncFinishResult::DONE);
    EXPECT_EQ(ClearedDirectory::confirmations, 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(word, uint64_t{0}), 1u);
    EXPECT_EQ(fixture.take_all(), std::vector<int64_t>{1});
    fixture.terminal(3);
    ClearedDirectory::directory_word = nullptr;
}
struct ForeignAbortAfterTaskDoneBefore : SchedulerAsyncFinishProtocol {
    static inline SchedulerRunControl *run = nullptr;
    static void
    task_done(void *base, const SchedulerLocalState *local, int64_t task, uint64_t order, uint32_t flags, bool before) {
        SchedulerAsyncFinishProtocol::task_done(base, local, task, order, flags, before);
        if (before) scheduler_record_error(run, 1, SchedulerGraphResult::INVALID_ARGUMENTS);
    }
};

TEST(HbgAsyncFinish, ForeignAbortAfterRealTaskDoneBeforeDoesNotInventLocalFailureCode) {
    scheduler_test::FixtureStorage storage(2, 6, 1, {6, 2, 0, 0}, true);
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, kEpoch));
    auto &context = storage.contexts[0];
    context.scheduler_count = 6;
    for (uint32_t lane = 0; lane < 3; ++lane)
        context.cluster_worker_ids[lane] = lane;
    storage.test_graphs[0] = graph.graph();
    auto *local = storage.local_context(&context);
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    SchedulerAsyncFinishIdentity identity{};
    identity.run_epoch = kEpoch;
    identity.task_ref = 0;
    SchedulerAsyncTaskJoin join{};
    SchedulerAsyncFinishProgress progress{};
    uint32_t budget = 8;
    ForeignAbortAfterTaskDoneBefore::run = storage.run_control;
    const auto result = scheduler_async_finish_step<
        SchedulerAsyncFinishQueueOps, SchedulerResolveLink, SchedulerAsyncFinishClock, ForeignAbortAfterTaskDoneBefore>(
        graph.graph(), base, local, storage.run_control, identity, &join, &progress, &budget
    );
    ForeignAbortAfterTaskDoneBefore::run = nullptr;
    EXPECT_EQ(result, SchedulerAsyncFinishResult::ERROR);
    EXPECT_EQ(progress.phase, static_cast<uint64_t>(SchedulerAsyncFinishPhase::FAILED));
    EXPECT_EQ(progress.failure_code, 0);
    EXPECT_EQ(storage.run_control->error_task_ref, 1u);
    EXPECT_EQ(local->pending_completed, 0u);
    const auto *trace = scheduler_state_at<SchedulerProtocolTaskTrace>(base, storage.layout.protocol.tasks_offset);
    EXPECT_NE(trace[0].completion.before.sequence, 0u);
    EXPECT_EQ(trace[0].completion.after.sequence, 0u);
    EXPECT_EQ(
        scheduler_gm_load(scheduler_task_control_at(base, local, 0)->state),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
}

}  // namespace
