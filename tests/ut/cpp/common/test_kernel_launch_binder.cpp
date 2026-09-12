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

#include "kernel_binder_test_support.h"

namespace kernel_binder_test {
TEST(KernelBinder, ThreeStreamSuccessAndSteadyStateHaveExactTrace) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    ASSERT_EQ(f.launch().status, 0);
    EXPECT_EQ(f.fake.trace, success);
    for (int i = 0; i < 10; ++i) {
        f.fake.clear_trace();
        const auto result = f.launch();
        EXPECT_EQ(result.status, 0);
        EXPECT_TRUE(result.tail_recorded);
        EXPECT_EQ(f.fake.trace, (std::vector<Step>(success.begin() + 1, success.end())));
    }
    EXPECT_EQ(f.fake.queries, 0);
    EXPECT_EQ(f.fake.acquisitions, 11);
    EXPECT_EQ(f.fake.finishes, 11);
}

TEST(KernelBinder, StreamSwitchQueriesButNeverWaitsOldTail) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    ASSERT_EQ(f.launch().status, 0);
    f.fake.clear_trace();
    f.fake.complete = false;
    EXPECT_NE(f.launch(ptr(101)).status, 0);
    EXPECT_TRUE(f.fake.trace.empty());
    EXPECT_FALSE(f.fake.poisoned);
    f.fake.complete = true;
    f.fake.query_error = -22;
    EXPECT_EQ(f.launch(ptr(101)).status, -22);
    EXPECT_TRUE(f.fake.trace.empty());
    f.fake.query_error = 0;
    EXPECT_EQ(f.launch(ptr(101)).status, 0);
    EXPECT_EQ(f.fake.queries, 3);
    EXPECT_EQ(f.fake.trace, (std::vector<Step>(success.begin() + 1, success.end())));
    f.fake.clear_trace();
    EXPECT_EQ(f.launch(ptr(101)).status, 0);
    EXPECT_EQ(f.fake.queries, 3);
}

TEST(KernelBinder, ReadOnlyRejectionsReleaseOwnerAndPreservePrepareDependency) {
    for (int fault = 0; fault < 10; ++fault) {
        SCOPED_TRACE(fault);
        Fixture f;
        f.initialize();
        auto gate = f.fake.gate();
        auto ops = f.fake.ops();
        auto binding = f.binding;
        void *caller = ptr(100);
        switch (fault) {
        case 0:
            f.fake.validation_error = -1234;
            break;
        case 1:
            f.fake.ready = false;
            break;
        case 2:
            f.fake.frozen = false;
            break;
        case 3:
            f.fake.handles.aicpu_done = nullptr;
            break;
        case 4:
            f.fake.handles.aicpu_done = ptr(5);
            break;
        case 5:
            caller = ptr(1);
            break;
        case 6:
            binding.packet_bytes = 0;
            break;
        case 7:
            gate.acquire = nullptr;
            break;
        case 8:
            gate.finish = nullptr;
            break;
        case 9:
            ops.launch_aicore = nullptr;
            break;
        }
        EXPECT_NE(launch_bound_kernel(binding, caller, gate, ops).status, 0);
        EXPECT_TRUE(f.fake.trace.empty());
        EXPECT_FALSE(f.fake.poisoned);
        EXPECT_EQ(f.fake.acquisitions, f.fake.finishes);
        EXPECT_EQ(f.fake.previous_caller, 0u);
        EXPECT_TRUE(f.fake.pending_prepare);
        f.fake.validation_error = 0;
        f.fake.ready = f.fake.frozen = true;
        f.fake.handles.aicpu_done = ptr(6);
        EXPECT_EQ(f.launch().status, 0);
        EXPECT_EQ(f.fake.trace, success);
    }
}

TEST(KernelBinder, EveryEnqueueFailurePoisonsWithExactCompensationTrace) {
    for (int fail = 1; fail <= static_cast<int>(success.size()); ++fail) {
        SCOPED_TRACE(fail);
        Fixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize());
        f.fake.fail_at = fail;
        const auto result = f.launch();
        EXPECT_EQ(result.status, -1700 - fail);
        EXPECT_EQ(result.failed_step, success[fail - 1]);
        auto expected = std::vector<Step>(success.begin(), success.begin() + fail);
        if (fail == 6) expected.insert(expected.end(), {Cancel, Step::AicoreDone, Step::JoinAicore, Step::SerialTail});
        if (fail == 7) expected.insert(expected.end(), {Cancel, Step::JoinAicore, Step::SerialTail});
        if (fail == 8)
            expected.insert(
                expected.end(), {Cancel, Step::AicpuDone, Step::JoinAicpu, Step::JoinAicore, Step::SerialTail}
            );
        EXPECT_EQ(f.fake.trace, expected);
        EXPECT_TRUE(f.fake.poisoned);
        EXPECT_EQ(f.fake.runtime_error, result.status);
        f.fake.clear_trace();
        EXPECT_NE(f.launch().status, 0);
        EXPECT_TRUE(f.fake.trace.empty());
        EXPECT_EQ(f.fake.acquisitions, f.fake.finishes);
    }
}

TEST(KernelBinder, CompensationFailuresKeepBothErrorsAndStopAtFailure) {
    for (int primary : {6, 7, 8}) {
        const int cleanup_steps = primary == 6 ? 4 : primary == 7 ? 3 : 5;
        for (int step = 1; step <= cleanup_steps; ++step) {
            SCOPED_TRACE(primary * 100 + step);
            Fixture f;
            ASSERT_NO_FATAL_FAILURE(f.initialize());
            f.fake.fail_at = primary;
            f.fake.second_fail_at = primary + step;
            const auto result = f.launch();
            EXPECT_EQ(result.status, -1700 - primary);
            EXPECT_EQ(result.cleanup_status, -1700 - primary - step);
            EXPECT_FALSE(result.tail_recorded);
            EXPECT_EQ(f.fake.calls, primary + step);
            EXPECT_EQ(f.fake.runtime_error, result.status);
            EXPECT_EQ(f.fake.cleanup_error, result.cleanup_status);
            EXPECT_EQ(f.fake.acquisitions, f.fake.finishes);
        }
    }
}

TEST(KernelBinder, ConcurrentHostLaunchIsRejectedWithoutTouchingArgs) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    std::promise<void> entered, release;
    f.fake.entered = &entered;
    f.fake.release = release.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        return f.launch();
    });
    entered.get_future().wait();
    auto second = f.launch();
    EXPECT_NE(second.status, 0);
    EXPECT_FALSE(second.enqueue_started);
    release.set_value();
    EXPECT_EQ(first.get().status, 0);
    EXPECT_EQ(f.fake.acquisitions, 1);
    EXPECT_EQ(f.fake.finishes, 1);
}

TEST(KernelBinder, NewPrepareTailIsConsumedExactlyOnce) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    ASSERT_EQ(f.launch().status, 0);
    f.fake.pending_prepare = true;
    f.fake.clear_trace();
    EXPECT_EQ(f.launch().status, 0);
    EXPECT_EQ(f.fake.trace, success);
}
}  // namespace kernel_binder_test
