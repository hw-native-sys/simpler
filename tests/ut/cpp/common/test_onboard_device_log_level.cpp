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

#include "aicpu/device_log.h"
#include "common/unified_log.h"
#include "dlog_pub.h"

TEST(OnboardDeviceLogLevelTest, WarnThresholdSuppressesTiming) {
    init_log_switch();
    ASSERT_TRUE(is_log_enable_warn());

    set_log_level(30);

    EXPECT_FALSE(is_log_enable_timing());
    EXPECT_TRUE(is_log_enable_warn());
}

TEST(OnboardDeviceLogLevelTest, TimingThresholdKeepsTiming) {
    init_log_switch();

    set_log_level(25);

    EXPECT_TRUE(is_log_enable_timing());
    EXPECT_TRUE(is_log_enable_warn());
}

TEST(OnboardDeviceLogLevelTest, TimingFormatsOnlyWhenHostAndCannThresholdsPermitIt) {
    for (bool cann_warn : {false, true}) {
        dlog_stub::warn_enabled = cann_warn;
        init_log_switch();
        for (int host_threshold : {25, 30}) {
            SCOPED_TRACE(::testing::Message() << "CANN WARN=" << cann_warn << " host=" << host_threshold);
            set_log_level(host_threshold);
            const bool timing = cann_warn && host_threshold == 25;
            EXPECT_EQ(is_log_enable_warn(), cann_warn);
            EXPECT_EQ(is_log_enable_timing(), timing);
            int formatted = -1;
            unified_log_timing(__func__, "payload%n", &formatted);
            EXPECT_EQ(formatted, timing ? 7 : -1);
        }
    }
    dlog_stub::warn_enabled = true;
    init_log_switch();
    set_log_level(25);
}
