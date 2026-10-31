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
#pragma once
#include <gtest/gtest.h>

inline void check_peer_failure_retirement() {
    for (bool shutdown : {false, true}) {
        TestedExecutor executor;
        executor.terminal_.record_participant(shutdown ? 0 : -42, shutdown ? -43 : 0);
        executor.terminal_.record_participant(0, 0);
        executor.terminal_publisher_.take(run_terminal_select(true, 0, executor.terminal_), nullptr, 0);
        executor.terminal_.reset();
        const int expected = shutdown ? -43 : -42;
        EXPECT_EQ(executor.terminal_execution_status(0, 0), expected);
        DeviceRunResultRegion region{};
        ASSERT_TRUE(executor.terminal_publisher_.publish(reinterpret_cast<uint64_t>(&region), 7));
        EXPECT_EQ(device_run_result_terminal(region, 7).code, expected);
    }
}

inline void check_local_failure_retirement() {
    TestedExecutor executor;
    executor.terminal_publisher_.take(run_terminal_select(true, 0, executor.terminal_), nullptr, 0);
    EXPECT_EQ(executor.terminal_execution_status(-51, -52), -51);
    EXPECT_EQ(executor.terminal_execution_status(0, -52), -52);
    EXPECT_EQ(executor.terminal_execution_status(0, 0), 0);
}
