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

#include "async_completion_plan.h"
#include "hbg_gm_mix_test_support.h"
#include "scheduler/scheduler_async_finish.h"

namespace {

using namespace pto2::resident;
using namespace scheduler_test::gm_mix;

TEST(HbgAdmissionProgress, CompletedPureSyncMemberClosesFromItsOwnClaim) {
    GmMixFixture fixture({{2, LaneMask::aic(), true}}, true, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.fill(0));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(0, &execution));
    ASSERT_TRUE(fixture.complete(execution));
    bool closed = false;
    ASSERT_TRUE(scheduler_close_ordered_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &closed
    ));
    EXPECT_TRUE(closed);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(0)->claim_seal, 0u);
    EXPECT_FALSE(fixture.task_done(0));
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_NE(members[0].flags & SCHEDULER_PROTOCOL_MEMBER_FROM_CLAIM, 0u);
    EXPECT_EQ(members[0].closed_cursor, UINT64_MAX);
    EXPECT_EQ(members[0].claim_cursor_seen, 0u);
    ASSERT_TRUE(fixture.refresh(3));
    ASSERT_TRUE(scheduler_close_ordered_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), &closed
    ));
    EXPECT_FALSE(closed);
}

TEST(HbgAdmissionProgress, ExistingReadyWindowRunsBeforeTheFirstCohortClaim) {
    for (const bool mixed : {false, true}) {
        SCOPED_TRACE(mixed);
        std::vector<GmShape> shapes(SCHEDULER_EXECUTION_TOKEN_COUNT, {1, LaneMask::aic()});
        const uint32_t cohort = shapes.size();
        shapes.push_back({2, mixed ? LaneMask::all() : LaneMask::aic(), true});
        GmMixFixture fixture(shapes, true, 4);
        ASSERT_TRUE(fixture.prepare_calls());
        for (uint32_t task = 0; task < cohort; ++task) {
            auto *local = fixture.locals[0];
            for (uint32_t slice = 0; slice <= cohort && !scheduler_execution_token_has_task(local, task); ++slice) {
                const uint64_t head = local->issue_cursor;
                const uint32_t count = scheduler_execution_token_count(local);
                bool progress = false;
                ASSERT_TRUE(scheduler_refill_execution_tokens(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
                ));
                ASSERT_TRUE(progress);
                ASSERT_TRUE(local->issue_cursor > head || scheduler_execution_token_count(local) > count);
            }
            ASSERT_TRUE(scheduler_execution_token_has_task(local, task));
            EXPECT_EQ(fixture.control(task)->next_block, 1u);
            EXPECT_EQ(scheduler_execution_token_count(local), task + 1);
        }
        ASSERT_TRUE(fixture.admit(cohort, 3));
        for (uint32_t task = 0; task < cohort; ++task) {
            ASSERT_TRUE(fixture.fill(0));
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(0, &execution));
            ASSERT_EQ(execution.task, task);
            EXPECT_FALSE(fixture.admissions[0].sync_claimed);
            EXPECT_EQ(fixture.control(cohort)->next_block, 0u);
            const auto *layout = scheduler_protocol_layout_at(fixture.base(), fixture.locals[0]);
            ASSERT_NE(layout, nullptr);
            const auto *record = scheduler_protocol_execution_at(fixture.base(), fixture.locals[0], layout, task, 0, 0);
            ASSERT_NE(record, nullptr);
            EXPECT_NE(record->flags & SCHEDULER_PROTOCOL_PERMIT_PRECLAIM, 0u);
            EXPECT_EQ(record->permit_tail, 1u);
            ASSERT_TRUE(fixture.complete(execution));
        }
        ASSERT_TRUE(fixture.fill(0));
        if (mixed) {
            EXPECT_TRUE(fixture.admissions[0].sync_claimed);
            EXPECT_EQ(fixture.control(cohort)->next_block, 1u);
            EXPECT_NE(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            EXPECT_EQ(scheduler_execution_token_count(fixture.locals[0]), 0u);
            EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
            continue;
        }
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        EXPECT_EQ(execution.task, cohort);
        EXPECT_TRUE(fixture.admissions[0].sync_claimed);
        EXPECT_EQ(fixture.control(cohort)->next_block, 1u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgAdmissionProgress, ReadyOrdinarySlotMayRunBeforeButNotAfterItsGmGroupIsOwned) {
    for (const uint32_t worker : {0u, 1u}) {
        for (const bool own_before_entry : {false, true}) {
            SCOPED_TRACE(worker);
            SCOPED_TRACE(own_before_entry);
            GmMixFixture fixture(
                {{1, worker == 0 ? LaneMask::aic() : LaneMask::aiv0()}, {2, LaneMask::all(), true}}, true, 2
            );
            ASSERT_TRUE(fixture.prepare_calls());
            ASSERT_TRUE(fixture.claim_ordinary(0, worker));
            ASSERT_TRUE(fixture.progress_ordinary(worker));
            auto *local = fixture.locals[worker];
            uint32_t slot = UINT32_MAX;
            ASSERT_TRUE(scheduler_local_ready_peek(local, &slot));
            const auto prepared = local->slots[slot].prepared;
            ASSERT_TRUE(fixture.admit(1));
            if (worker != 0) {
                ASSERT_TRUE(fixture.refresh(0));
                ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
                ASSERT_TRUE(fixture.publish(0));
                ASSERT_GT(fixture.mailbox(0)->lanes[worker].assigned_seq, 0u);
            }
            GmExecution execution{};
            EXPECT_FALSE(fixture.begin(worker, &execution));
            ASSERT_TRUE(fixture.service_cooperative(worker));
            ASSERT_TRUE(scheduler_local_ready_peek(local, &slot));
            EXPECT_EQ(local->slots[slot].prepared.execution_index, prepared.execution_index);
            EXPECT_EQ(local->slots[slot].prepared.claimed_state, prepared.claimed_state);
            EXPECT_EQ(local->slots[slot].preclaim_order, 0u);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            EXPECT_TRUE(fixture.admissions[worker].preclaim_initialized);
            EXPECT_EQ(fixture.admissions[worker].preclaim_mask, 0u);
            if (own_before_entry) {
                if (worker == 0) {
                    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
                } else {
                    bool observed = false;
                    ASSERT_TRUE(fixture.observe(worker, &observed));
                    ASSERT_TRUE(observed);
                }
                EXPECT_FALSE(fixture.begin(worker, &execution));
                ASSERT_TRUE(fixture.service_cooperative(worker));
                EXPECT_TRUE(scheduler_execution_token_has_task(local, 0));
                EXPECT_TRUE(fixture.admissions[worker].preclaim_initialized);
                EXPECT_EQ(fixture.admissions[worker].preclaim_mask, 0u);
                EXPECT_NE(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            } else {
                ASSERT_TRUE(fixture.begin(worker, &execution));
                EXPECT_EQ(execution.task, 0);
                EXPECT_EQ(execution.worker, worker);
                ASSERT_TRUE(fixture.complete(execution));
                EXPECT_TRUE(fixture.task_done(0));
                ASSERT_TRUE(fixture.service_cooperative(worker));
                EXPECT_NE(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            }
            EXPECT_EQ(fixture.control(0)->next_block, 1u);
            EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        }
    }
}

TEST(HbgAdmissionProgress, GmPreclaimWindowDoesNotWaitForAsyncDebtOrRefillVacatedTokens) {
    GmMixFixture fixture(
        {{1, LaneMask::aiv0()},
         {1, LaneMask::aic()},
         {1, LaneMask::aic()},
         {2, LaneMask::all(), true},
         {1, LaneMask::aic()}},
        false, 2
    );
    fixture.graph.executable(1, 0, {0});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 1));
    // Physical work is finished, but its registered completion condition is not.
    fixture.control(0)->completion_state = scheduler_completion_encode(1, 0, 1);
    ASSERT_TRUE(fixture.claim_ordinary(1, 0));
    ASSERT_TRUE(fixture.claim_ordinary(2, 0));
    ASSERT_TRUE(fixture.admit(3));
    ASSERT_TRUE(fixture.service_cooperative(0));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(0, &execution));
    EXPECT_EQ(execution.task, 2);
    ASSERT_TRUE(fixture.complete(execution));
    EXPECT_TRUE(scheduler_execution_token_has_task(fixture.locals[0], 1));
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_TRUE(fixture.service_cooperative(0));
    EXPECT_TRUE(fixture.admissions[0].sync_claimed);
    EXPECT_NE(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(fixture.admissions[0].preclaim_mask, 0u);
    EXPECT_EQ(fixture.control(4)->next_block, 0u);
    EXPECT_TRUE(scheduler_execution_token_has_task(fixture.locals[0], 1));
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST(HbgAdmissionProgress, DeferredPredecessorLeavesIndependentCounterPublisherRunnable) {
    constexpr uint32_t successor_count = SCHEDULER_EXECUTION_TOKEN_COUNT + 1;
    constexpr uint32_t publisher = successor_count + 1;
    constexpr uint32_t task_count = publisher + 1;
    for (const bool sync_successors : {false, true}) {
        SCOPED_TRACE(sync_successors ? "sync successors" : "ordinary successors");
        std::vector<GmShape> shapes(task_count, {2, LaneMask::aiv0(), sync_successors});
        shapes[0].sync_start = true;
        shapes[publisher] = {1, LaneMask::aiv0()};
        AsyncCompletionHostPlan plan;
        GmMixFixture fixture(shapes, false, 1, [&](AicoreSchedulerLayout &layout) {
            AsyncTaskPlanInput input;
            input.task_ref = 0;
            input.block_count = 2;
            input.active_mask = LaneMask::aiv0();
            input.capacities[1] = 1;
            std::string error;
            EXPECT_TRUE(plan_async_completion(layout.total_size, 47, task_count, {input}, &plan, &error)) << error;
            layout.total_size = plan.total_bytes;
            layout.async_layout_offset = plan.layout_offset;
        });
        ASSERT_TRUE(initialize_async_completion(fixture.base(), fixture.storage.layout.total_size, plan));
        scheduler_state_at<SchedulerAsyncLayout>(fixture.base(), plan.layout_offset)->image_bytes =
            fixture.storage.layout.total_size;
        std::array<SchedulerAsyncLocalState, 3> async_workers{};
        for (uint32_t worker = 0; worker < async_workers.size(); ++worker) {
            ASSERT_TRUE(scheduler_initialize_async(
                fixture.base(), fixture.storage.layout.total_size, plan.layout_offset, 47, task_count,
                fixture.storage.layout.task_controls_offset, async_workers[worker]
            ));
            fixture.locals[worker]->async = &async_workers[worker];
        }

        // Every B depends on W. Independent P is submitted after the full
        // waiting window; Host generations place it before those consumers.
        for (uint32_t task = 1; task <= successor_count; ++task) {
            fixture.graph.executable(task, 1, {0});
        }
        const std::vector<SchedulerTaskMetadata> metadata(
            fixture.storage.metadata, fixture.storage.metadata + task_count
        );
        SchedulerTaskCatalogPlan catalog;
        ASSERT_TRUE(scheduler_plan_task_catalog(
            metadata, fixture.storage.catalog_clusters(), fixture.storage.catalog_begin, &catalog
        ));
        ASSERT_TRUE(scheduler_order_task_catalog(fixture.graph.graph(), metadata, {}, &catalog));
        ASSERT_LT(catalog.task_ranks[publisher], catalog.task_ranks[1]);
        scheduler_init_task_catalog(fixture.base(), catalog);
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_NE(fixture.locals[1]->admission, nullptr);
        ASSERT_TRUE(fixture.claim_pending(0, 1));
        ASSERT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        EXPECT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);

        alignas(128) uint32_t signal = 0;
        for (uint32_t worker = 1; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            ASSERT_EQ(execution.task, 0);
            auto &context = fixture.storage.prepared_global_context(0, execution.block, execution.subtask);
            AsyncWriter writer;
            ASSERT_TRUE(scheduler_async_begin_prebound_execution(&async_workers[worker], 0, context, writer));
            CompletionToken token{
                reinterpret_cast<uint64_t>(&signal), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
            };
            ASSERT_TRUE(register_completion_condition(writer, token));
            ASSERT_TRUE(
                scheduler_async_seal_execution(&async_workers[worker], 0, execution.block, execution.subtask, writer)
            );
            ASSERT_TRUE(fixture.complete(execution));
        }
        ASSERT_NE(
            scheduler_completion_task_state(fixture.control(0)->completion_state),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
        // Real ordered refill reaches the full waiting window without any
        // caller assigning tokens or skipping an issuance head.
        if (!sync_successors) {
            for (uint32_t slice = 0; slice < task_count * SCHEDULER_EXECUTION_TOKEN_COUNT; ++slice) {
                if (scheduler_execution_token_count(fixture.locals[1]) == SCHEDULER_EXECUTION_TOKEN_COUNT &&
                    scheduler_execution_token_count(fixture.locals[2]) == SCHEDULER_EXECUTION_TOKEN_COUNT)
                    break;
                bool advanced = false;
                for (uint32_t worker = 1; worker < 3; ++worker) {
                    bool progress = false;
                    ASSERT_TRUE(scheduler_refill_execution_tokens(
                        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
                    ));
                    advanced = advanced || progress;
                }
                ASSERT_TRUE(advanced);
            }
        }
        for (uint32_t worker = 1; worker < 3; ++worker) {
            auto *local = fixture.locals[worker];
            EXPECT_EQ(scheduler_execution_token_count(local), sync_successors ? 0u : SCHEDULER_EXECUTION_TOKEN_COUNT);
            EXPECT_EQ(local->local_ready_mask, 0u);
            for (const auto &slot : local->slots)
                EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
        }
        for (uint32_t task = 1; task <= successor_count; ++task) {
            EXPECT_FALSE(scheduler_completion_done(fixture.control(task)->completion_state));
            if (sync_successors) {
                EXPECT_EQ(fixture.control(task)->next_block, 0u);
                EXPECT_EQ(
                    fixture.control(task)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
                );
            }
            for (uint32_t block = 0; block < 2; ++block) {
                uint64_t index = UINT64_MAX;
                ASSERT_TRUE(scheduler_prepared_execution_index(
                    fixture.base(), fixture.locals[1]->config.prepared, task, block, 1, &index
                ));
                EXPECT_NE(
                    scheduler_prepared_call_at(fixture.base(), fixture.locals[1]->config.prepared, index)
                        ->function_bin_addr,
                    0u
                );
            }
        }
        EXPECT_EQ(shared->commit_tail, 1u);

        uint32_t publishers = 0;
        std::set<std::pair<int64_t, uint32_t>> successors;
        // B must not reserve either AIV while W is waiting for P's COUNTER.
        for (uint32_t step = 0; step < task_count * 16; ++step) {
            for (uint32_t worker = 0; worker < 3; ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution execution{};
                if (fixture.begin(worker, &execution)) {
                    if (execution.task == publisher) {
                        for (uint32_t task = 1; task <= successor_count; ++task) {
                            EXPECT_FALSE(scheduler_completion_done(fixture.control(task)->completion_state));
                            if (sync_successors) {
                                EXPECT_EQ(fixture.control(task)->next_block, 0u);
                                EXPECT_EQ(
                                    fixture.control(task)->admission_state,
                                    static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
                                );
                            }
                        }
                        for (const auto *local : fixture.locals)
                            for (const auto &slot : local->slots)
                                if (slot.state != SchedulerDispatchSlotState::FREE) EXPECT_EQ(slot.task_ref, publisher);
                        ++publishers;
                        __atomic_store_n(&signal, 1u, __ATOMIC_RELEASE);
                    } else {
                        ASSERT_GE(execution.task, 1);
                        ASSERT_LE(execution.task, successor_count);
                        EXPECT_LT(execution.block, 2u);
                        EXPECT_TRUE(successors.emplace(execution.task, execution.block).second);
                        EXPECT_EQ(signal, 1u);
                        EXPECT_EQ(
                            scheduler_completion_task_state(fixture.control(0)->completion_state),
                            static_cast<int64_t>(SchedulerTaskState::DONE)
                        );
                    }
                    ASSERT_TRUE(fixture.complete(execution));
                }
                bool progress = false;
                SchedulerAsyncCompletion completion{};
                ASSERT_TRUE(scheduler_async_check_step(
                    async_workers[worker], worker == 0 ? 1 : 2, 8, SchedulerAsyncCompletionObserver{}, progress,
                    completion
                ));
                if (completion.result == SchedulerAsyncCompletionResult::FINISH_OWNER)
                    ASSERT_TRUE(scheduler_async_complete_task(
                        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), completion
                    ));
            }
        }
        EXPECT_EQ(publishers, 1u);
        EXPECT_EQ(successors.size(), successor_count * 2);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        for (uint64_t task = 0; task < task_count; ++task)
            EXPECT_EQ(
                scheduler_completion_task_state(fixture.control(task)->completion_state),
                static_cast<int64_t>(SchedulerTaskState::DONE)
            );
        for (uint32_t worker = 0; worker < 3; ++worker) {
            bool done = false;
            ASSERT_TRUE(fixture.drain(worker, &done));
            EXPECT_TRUE(done);
        }
    }
}

}  // namespace
