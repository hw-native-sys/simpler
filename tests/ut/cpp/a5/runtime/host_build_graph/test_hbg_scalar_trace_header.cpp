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

#include "scheduler/scheduler_scalar_record.h"

#include <gtest/gtest.h>

TEST(HbgScalarTraceHeader, SimulationClockUsesPlatformTicks) {
    const uint64_t before = sys_cnt_now_ticks();
    const uint64_t observed = scheduler_scalar_clock();
    const uint64_t after = sys_cnt_now_ticks();
    EXPECT_GT(observed, 0u);
    EXPECT_GE(observed, before);
    EXPECT_LE(observed, after);
}
