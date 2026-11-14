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
    {
        AsyncCompletionHostPlan plan;
        GmMixFixture fixture(
            {{2, LaneMask::aiv0()}, {2, LaneMask::aiv0()}, {1, LaneMask::aiv0()}}, false, 1,
            [&](AicoreSchedulerLayout &layout) {
                AsyncTaskPlanInput input;
                input.task_ref = 0;
                input.block_count = 2;
                input.active_mask = LaneMask::aiv0();
                std::string error;
                EXPECT_TRUE(plan_async_completion(layout.total_size, 47, 3, {input}, &plan, &error)) << error;
                layout.total_size = plan.total_bytes;
                layout.async_layout_offset = plan.layout_offset;
            }
        );
        ASSERT_TRUE(initialize_async_completion(fixture.base(), fixture.storage.layout.total_size, plan));
        std::array<SchedulerAsyncLocalState, 3> async_workers{};
        for (uint32_t worker = 0; worker < async_workers.size(); ++worker) {
            ASSERT_TRUE(scheduler_initialize_async(
                fixture.base(), plan.total_bytes, plan.layout_offset, 47, 3, async_workers[worker]
            ));
            fixture.locals[worker]->async = &async_workers[worker];
        }

        // A -> B is the only DAG edge. A's COUNTER is published by independent P.
        fixture.graph.executable(1, 1, {0});
        fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 47));
        ASSERT_TRUE(fixture.bootstrap_admission());
        ASSERT_EQ(
            scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1),
            SchedulerRouteResult::WAITING
        );
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
            EXPECT_EQ(fixture.admissions[worker].single_wave_claim_confirmed, execution.block + 1);
            EXPECT_FALSE(fixture.admissions[worker].single_wave_retired);
            ASSERT_TRUE(fixture.complete(execution));
            EXPECT_TRUE(fixture.admissions[worker].single_wave_retired);
            bool closed = false;
            ASSERT_TRUE(scheduler_close_ordered_obligation(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &closed
            ));
            EXPECT_TRUE(closed);
            EXPECT_EQ(fixture.admissions[worker].single_wave_claim_confirmed, 0u);
            EXPECT_FALSE(fixture.admissions[worker].single_wave_retired);
            EXPECT_NE(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
            EXPECT_EQ(shared->commit_tail, 1u);
        }
        ASSERT_NE(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        ASSERT_TRUE(fixture.enqueue(2, 1));

        uint32_t publishers = 0;
        uint32_t successors = 0;
        // B must not reserve either AIV while A is waiting for P's COUNTER.
        for (uint32_t step = 0; step < 24; ++step) {
            for (uint32_t worker = 0; worker < 3; ++worker) {
                ASSERT_TRUE(fixture.fill(worker));
                GmExecution execution{};
                if (fixture.begin(worker, &execution)) {
                    if (execution.task == 2) {
                        ++publishers;
                        signal = 1;
                    } else {
                        ASSERT_EQ(execution.task, 1);
                        ++successors;
                        EXPECT_EQ(signal, 1u);
                        EXPECT_EQ(fixture.control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
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
        EXPECT_EQ(successors, 2u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
        for (uint64_t task = 0; task < 3; ++task)
            EXPECT_EQ(fixture.control(task)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            bool done = false;
            ASSERT_TRUE(fixture.drain(worker, &done));
            EXPECT_TRUE(done);
        }
    }
}

}  // namespace
