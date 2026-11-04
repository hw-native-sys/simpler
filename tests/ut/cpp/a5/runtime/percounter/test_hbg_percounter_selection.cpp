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

#include <cstdlib>

#include "percounter/percounter_selection.h"

namespace {

namespace pc = simpler::hbg::percounter;

TEST(PercounterSelection, UnsetAndEmptyKeepTheDefault) {
    pc::SchedulerRequest request = pc::SchedulerRequest::PERCOUNTER;
    ASSERT_TRUE(pc::parse_scheduler_request(nullptr, &request));
    EXPECT_EQ(request, pc::SchedulerRequest::DEFAULT);

    request = pc::SchedulerRequest::PERCOUNTER;
    ASSERT_TRUE(pc::parse_scheduler_request("", &request));
    EXPECT_EQ(request, pc::SchedulerRequest::DEFAULT);
}

TEST(PercounterSelection, PercounterIsTheOnlyAcceptedValue) {
    pc::SchedulerRequest request = pc::SchedulerRequest::DEFAULT;
    ASSERT_TRUE(pc::parse_scheduler_request("percounter", &request));
    EXPECT_EQ(request, pc::SchedulerRequest::PERCOUNTER);

    for (const char *text : {"Percounter", "percounter ", "auto", "resident", "legacy", "per-counter"}) {
        EXPECT_FALSE(pc::parse_scheduler_request(text, &request)) << text;
    }
}

TEST(PercounterSelection, ResolveReadsTheEnvironmentAndRejectsOnlyUnrecognisedValues) {
    pc::SchedulerRequest request = pc::SchedulerRequest::PERCOUNTER;
    ASSERT_EQ(unsetenv(pc::SCHEDULER_TYPE_ENV), 0);
    ASSERT_TRUE(pc::resolve_scheduler_request(&request));
    EXPECT_EQ(request, pc::SchedulerRequest::DEFAULT);

    ASSERT_EQ(setenv(pc::SCHEDULER_TYPE_ENV, "", 1), 0);
    ASSERT_TRUE(pc::resolve_scheduler_request(&request));
    EXPECT_EQ(request, pc::SchedulerRequest::DEFAULT);

    ASSERT_EQ(setenv(pc::SCHEDULER_TYPE_ENV, "percounter", 1), 0);
    ASSERT_TRUE(pc::resolve_scheduler_request(&request));
    EXPECT_EQ(request, pc::SchedulerRequest::PERCOUNTER);

    ASSERT_EQ(setenv(pc::SCHEDULER_TYPE_ENV, "resident", 1), 0);
    EXPECT_FALSE(pc::resolve_scheduler_request(&request));

    ASSERT_EQ(unsetenv(pc::SCHEDULER_TYPE_ENV), 0);
}

}  // namespace
