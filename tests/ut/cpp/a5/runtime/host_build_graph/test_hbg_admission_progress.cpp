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

        // Every B depends on W; independent P publishes W's COUNTER and is
        // beyond a full candidate window in the immutable AIV catalog.
        for (uint32_t task = 1; task <= successor_count; ++task) {
            fixture.graph.executable(task, 1, {0});
            fixture.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
        }
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.bootstrap_admission());
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
            GlobalContext context{};
            AsyncWriter writer;
            ASSERT_TRUE(scheduler_async_begin_execution(
                &async_workers[worker], 0, execution.block, execution.subtask, context, writer
            ));
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
        // Ordinary consumers hold actual execution tickets while the independent
        // publisher remains beyond their finite local token window.
        for (uint32_t worker = 1; worker < 3; ++worker) {
            auto *local = fixture.locals[worker];
            local->execution_scan_cursor = 1;
            for (uint32_t index = 0; index < SCHEDULER_EXECUTION_TOKEN_COUNT; ++index) {
                const uint32_t task = index + 1;
                if (sync_successors) continue;
                auto &token = local->execution_tokens[index];
                const uint64_t block = scheduler_gm_fetch_add(fixture.control(task)->next_block, UINT64_C(1));
                ASSERT_LT(block, 2u);
                token.task_ref = task;
                token.block_idx = static_cast<uint32_t>(block);
                token.subtask_slot = 1;
                token.phase = SchedulerExecutionTokenPhase::TICKETED;
                ASSERT_EQ(
                    scheduler_node_fanin_view(fixture.graph.graph(), task, &token.fanin), SchedulerGraphResult::OK
                );
                ASSERT_TRUE(scheduler_execution_token_receive(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), token, 0
                ));
            }
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
                ASSERT_TRUE(scheduler_async_check_step(
                    async_workers[worker], worker == 0 ? 1 : 2, 8, SchedulerAsyncCompletionObserver{}, progress
                ));
                ASSERT_TRUE(scheduler_async_finish_step(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), async_workers[worker],
                    progress
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
