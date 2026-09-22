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

#include "aicpu/thread_scheduling.h"

namespace {
int current_policy = SCHED_FIFO;
int get_error = 0;
int set_error = 0;
int set_calls = 0;

class ThreadScheduling : public ::testing::Test {
    void SetUp() override {
        current_policy = SCHED_FIFO;
        get_error = 0;
        set_error = 0;
        set_calls = 0;
    }
};
}  // namespace

extern "C" int __wrap_sched_getscheduler(pid_t pid) {
    EXPECT_EQ(pid, 0);
    errno = get_error;
    return get_error ? -1 : current_policy;
}

extern "C" int __wrap_sched_setscheduler(pid_t pid, int policy, const sched_param *param) {
    EXPECT_EQ(pid, 0);
    EXPECT_EQ(policy, SCHED_OTHER);
    EXPECT_EQ(param->sched_priority, 0);
    ++set_calls;
    errno = set_error;
    if (set_error) return -1;
    current_policy = policy;
    return 0;
}

TEST_F(ThreadScheduling, ConvertsRealtimePolicy) {
    EXPECT_EQ(use_normal_aicpu_scheduling(), 0);
    EXPECT_EQ(current_policy, SCHED_OTHER);
    EXPECT_EQ(set_calls, 1);
}

TEST_F(ThreadScheduling, RechecksReusedWorker) {
    EXPECT_EQ(use_normal_aicpu_scheduling(), 0);
    EXPECT_EQ(use_normal_aicpu_scheduling(), 0);
    EXPECT_EQ(set_calls, 1);
    current_policy = SCHED_FIFO;
    EXPECT_EQ(use_normal_aicpu_scheduling(), 0);
    EXPECT_EQ(set_calls, 2);
}

TEST_F(ThreadScheduling, PreservesPolicyWhenDenied) {
    set_error = EPERM;
    EXPECT_EQ(use_normal_aicpu_scheduling(), EPERM);
    EXPECT_EQ(current_policy, SCHED_FIFO);
}

TEST_F(ThreadScheduling, ReportsQueryFailure) {
    get_error = EINVAL;
    EXPECT_EQ(use_normal_aicpu_scheduling(), EINVAL);
    EXPECT_EQ(set_calls, 0);
}
