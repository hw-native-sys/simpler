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

#include "pto_runtime2.h"

extern "C" void unified_log_error(const char *, const char *, ...) {}
extern "C" void unified_log_warn(const char *, const char *, ...) {}
extern "C" void unified_log_info(const char *, const char *, ...) {}
extern "C" void unified_log_debug(const char *, const char *, ...) {}
extern "C" void unified_log_always(const char *, const char *, ...) {}

namespace {

class ManualScopeRuntimeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rt_ = pto2_runtime_create(PTO2_MODE_GRAPH_ONLY);
        ASSERT_NE(rt_, nullptr);
    }

    void TearDown() override {
        if (rt_ != nullptr) {
            pto2_runtime_destroy(rt_);
        }
    }

    static TensorCreateInfo make_create_info() {
        static const uint32_t kShape[1] = {1};
        return TensorCreateInfo(kShape, 1, DataType::FLOAT32);
    }

    PTO2Runtime *rt_{nullptr};
};

TEST_F(ManualScopeRuntimeTest, ExplicitDepAddsProducerFaninAtSubmitTime) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo producer_ci = make_create_info();
    Arg producer_args;
    producer_args.add_output(producer_ci);
    TaskSubmitResult producer = pto2_alloc_tensors(&rt_->orchestrator, producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    TensorCreateInfo consumer_ci = make_create_info();
    Arg consumer_args;
    consumer_args.add_output(consumer_ci);
    consumer_args.add_dep(producer.task_id());
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    TaskSubmitResult consumer = pto2_submit_mixed_task(&rt_->orchestrator, kernels, consumer_args);
    ASSERT_FALSE(rt_->orchestrator.fatal);
    ASSERT_TRUE(consumer.task_id().is_valid());

    PTO2TaskSlotState &producer_slot =
        rt_->scheduler.ring_sched_states[producer.task_id().ring()].get_slot_state_by_task_id(producer.task_id().local());
    PTO2TaskSlotState &consumer_slot =
        rt_->scheduler.ring_sched_states[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());

    ASSERT_NE(consumer_slot.payload, nullptr);
    EXPECT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_slot_states[0], &producer_slot);
    EXPECT_EQ(producer_slot.fanout_count, 2);
}

TEST_F(ManualScopeRuntimeTest, RuntimeOutputsRecordManualScopeMetadata) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo output_ci = make_create_info();
    Arg args;
    args.add_output(output_ci);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    TaskSubmitResult outputs = pto2_submit_mixed_task(&rt_->orchestrator, kernels, args);
    ASSERT_FALSE(rt_->orchestrator.fatal);
    ASSERT_EQ(outputs.size(), 1u);

    const Tensor &out = outputs.get_ref(0);
    EXPECT_TRUE(out.owner_task_id.is_valid());
    EXPECT_EQ(out.producer_scope_depth, 0);
    EXPECT_EQ(out.producer_manual_scope_depth, 0);
}

TEST_F(ManualScopeRuntimeTest, TaskSlotRecordsSubmissionScopeDepth) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo outer_ci = make_create_info();
    Arg outer_args;
    outer_args.add_output(outer_ci);
    TaskSubmitResult outer = pto2_alloc_tensors(&rt_->orchestrator, outer_args);
    ASSERT_TRUE(outer.task_id().is_valid());

    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::AUTO);

    TensorCreateInfo inner_ci = make_create_info();
    Arg inner_args;
    inner_args.add_output(inner_ci);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    TaskSubmitResult inner = pto2_submit_mixed_task(&rt_->orchestrator, kernels, inner_args);
    ASSERT_FALSE(rt_->orchestrator.fatal);
    ASSERT_TRUE(inner.task_id().is_valid());

    PTO2TaskSlotState &outer_slot =
        rt_->scheduler.ring_sched_states[outer.task_id().ring()].get_slot_state_by_task_id(outer.task_id().local());
    PTO2TaskSlotState &inner_slot =
        rt_->scheduler.ring_sched_states[inner.task_id().ring()].get_slot_state_by_task_id(inner.task_id().local());

    EXPECT_EQ(outer_slot.scope_depth, 0);
    EXPECT_EQ(inner_slot.scope_depth, 1);
}

TEST_F(ManualScopeRuntimeTest, ExplicitDepRejectsTaskFromOuterScope) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo outer_ci = make_create_info();
    Arg outer_args;
    outer_args.add_output(outer_ci);
    TaskSubmitResult outer = pto2_alloc_tensors(&rt_->orchestrator, outer_args);
    ASSERT_TRUE(outer.task_id().is_valid());

    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::AUTO);

    TensorCreateInfo inner_ci = make_create_info();
    Arg inner_args;
    inner_args.add_output(inner_ci);
    inner_args.add_dep(outer.task_id());
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    TaskSubmitResult inner = pto2_submit_mixed_task(&rt_->orchestrator, kernels, inner_args);
    EXPECT_TRUE(rt_->orchestrator.fatal);
    EXPECT_TRUE(inner.empty());
}

TEST_F(ManualScopeRuntimeTest, ExplicitDepRejectsTaskFromClosedScopeAtSameDepth) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo first_ci = make_create_info();
    Arg first_args;
    first_args.add_output(first_ci);
    TaskSubmitResult first = pto2_alloc_tensors(&rt_->orchestrator, first_args);
    ASSERT_TRUE(first.task_id().is_valid());

    pto2_scope_end(&rt_->orchestrator);
    ASSERT_FALSE(rt_->orchestrator.fatal);

    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    TensorCreateInfo second_ci = make_create_info();
    Arg second_args;
    second_args.add_output(second_ci);
    second_args.add_dep(first.task_id());
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    TaskSubmitResult second = pto2_submit_mixed_task(&rt_->orchestrator, kernels, second_args);
    EXPECT_TRUE(rt_->orchestrator.fatal);
    EXPECT_TRUE(second.empty());
}

TEST_F(ManualScopeRuntimeTest, ViewCachesStartOffsetEagerly) {
    pto2_scope_begin(&rt_->orchestrator, PTO2ScopeMode::MANUAL);

    uint32_t shapes[2] = {8, 8};
    TensorCreateInfo create_info(shapes, 2, DataType::FLOAT32);
    Arg args;
    args.add_output(create_info);
    TaskSubmitResult outputs = pto2_alloc_tensors(&rt_->orchestrator, args);
    ASSERT_EQ(outputs.size(), 1u);

    uint32_t view_shapes[2] = {2, 3};
    uint32_t view_offsets[2] = {2, 3};
    Tensor view = outputs.get_ref(0).view(view_shapes, view_offsets);

    EXPECT_EQ(view.start_offset, 19u);
}

}  // namespace
