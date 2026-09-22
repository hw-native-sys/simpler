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

#include "device_phase_capture.h"
#include "host_log.h"

using simpler::log::LogLevel;

namespace {

class DevicePhaseCaptureTest : public testing::Test {
protected:
    void TearDown() override { HostLogger::get_instance().set_level(LogLevel::TIMING); }
};

}  // namespace

TEST_F(DevicePhaseCaptureTest, FollowsTraceGatesAndTimingMarkerVisibility) {
    HostLogger::get_instance().set_level(LogLevel::TIMING);
    EXPECT_EQ(device_phase_capture_enabled(), EXPECT_CAPTURE_ENABLED != 0);

    HostLogger::get_instance().set_level(LogLevel::INFO);
    EXPECT_EQ(device_phase_capture_enabled(), EXPECT_CAPTURE_ENABLED != 0);

    HostLogger::get_instance().set_level(LogLevel::WARN);
    EXPECT_FALSE(device_phase_capture_enabled());

    HostLogger::get_instance().set_level(LogLevel::ERROR);
    EXPECT_FALSE(device_phase_capture_enabled());

    HostLogger::get_instance().set_level(LogLevel::NUL);
    EXPECT_FALSE(device_phase_capture_enabled());
}
