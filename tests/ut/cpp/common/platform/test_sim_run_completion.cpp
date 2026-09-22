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

#include <future>
#include <thread>

#include "sim_run_completion.h"

using simpler::common::sim_host::SimRunCompletion;

TEST(SimRunCompletionTest, SubmissionRemainsPendingUntilTheFinalTaskCompletes) {
    SimRunCompletion completion;
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_ERROR);
    std::promise<void> task_started;
    std::promise<void> release_task;
    std::shared_future<void> release = release_task.get_future().share();

    completion.reset(1);
    std::thread task([&]() {
        task_started.set_value();
        release.wait();
        completion.task_finished();
    });

    task_started.get_future().wait();
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_NOT_READY);

    release_task.set_value();
    task.join();
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(completion.first_error(), 0);
}

TEST(SimRunCompletionTest, CompletionWaitsForEveryTaskAndKeepsTheFirstError) {
    SimRunCompletion completion;
    completion.reset(2);

    completion.task_finished(-7);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(completion.first_error(), -7);

    completion.task_finished(-9);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(completion.first_error(), -7);

    completion.reset(1);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(completion.first_error(), 0);
}

TEST(SimRunCompletionTest, AbandonPublishesStickyErrorAndLateCompletionCannotOverwriteIt) {
    SimRunCompletion completion;
    completion.reset(1);

    completion.abandon();
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_ERROR);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_ERROR);

    completion.task_finished();
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_ERROR);

    completion.reset(1);
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    completion.task_finished();
    EXPECT_EQ(completion.poll(), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
}
