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
#include <cstdint>
#include "scheduler/scheduler_types.h"

namespace {
int clock_reads = 0;
}

uint64_t get_sys_cnt_aicpu() { return ++clock_reads; }

TEST(SchedulerCycleAccounting, UnconsumedAccumulatorsDoNotSampleTheClock) {
    uint64_t idle = 19, complete = 23, dispatch = 29, async = 31;
    CYCLE_COUNT_START();
    CYCLE_COUNT_LAP(idle);
    CYCLE_COUNT_LAP(complete);
    CYCLE_COUNT_LAP(dispatch);
    CYCLE_COUNT_LAP(async);
    EXPECT_EQ(clock_reads, 0);
    EXPECT_EQ(idle, 19u);
    EXPECT_EQ(complete, 23u);
    EXPECT_EQ(dispatch, 29u);
    EXPECT_EQ(async, 31u);
}
