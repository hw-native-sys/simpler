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

#include "common/host_span.h"
#include "common/log_level.h"
#include "common/strace.h"
#include "host_log.h"

TEST(HostLogUnboundTest, PrivateModuleStateStartsSilent) {
    EXPECT_EQ(HostLogger::get_instance().level(), static_cast<int>(simpler::log::LogLevel::NUL));
    EXPECT_EQ(unified_log_host_span_enabled(), 0);

    const SimplerHostSpan span{1, 0, 0, 0, 100, 25, "node.dispatch", "run_id=1"};
    testing::internal::CaptureStderr();
    HostLogger::get_instance().log(simpler::log::LogLevel::ERROR, "unbound", "must stay silent");
    unified_log_host_span(&span);
    {
        STRACE_CONTEXT(7, 9, 2);
        STRACE("unbound");
        STRACE_HOST_SPAN_AT("unbound_explicit", 100, 25, 0);
        EXPECT_EQ(STRACE_NEW_INV(), 0u);
        EXPECT_EQ(STRACE_ALLOC_INV(), 0u);
        STRACE_SET_HID(19);
        EXPECT_EQ(simpler::strace::strace_state(), nullptr);
    }
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
}
