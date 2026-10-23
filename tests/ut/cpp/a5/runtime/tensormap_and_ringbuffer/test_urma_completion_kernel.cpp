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
#define __CPU_SIM
#include "backend/urma/urma_completion_kernel.h"

namespace {
struct Event {
    uint64_t handle = 0;
    pto::comm::DmaEngine engine = pto::comm::DmaEngine::URMA;
    uint16_t urmaJettyBase = 0, urmaJettyCount = 1;
    uint32_t urmaTargetBbPerJetty[1]{0}, urmaTargetCqePerJetty[1]{7};
};
struct Session {
    uint32_t qpIdxBase = 0, qpCount = 1;
};
}  // namespace

TEST(A5UrmaCompletionKernel, PreserveTargetsEvenWhenHandleWrapsToZero) {
    AsyncCtx ctx{};
    ctx.task_token = TaskId::make(0, 42);
    uint32_t count = 0;
    int32_t error = 0;
    DeferredCompletionEntry entries[2]{};
    ctx.completion_count = &count;
    ctx.completion_error_code = &error;
    ctx.completion_entries = entries;
    ctx.completion_capacity = 2;
    uint8_t workspace[8]{};
    ASSERT_TRUE(pto2::detail::register_urma_async_event(ctx, Event{}, Session{}, workspace));
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(entries[0].addr, 0u);
    EXPECT_EQ(entries[0].expected_value, 7u);
    EXPECT_EQ(entries[0].backend_cookie, reinterpret_cast<uint64_t>(workspace));
    ASSERT_TRUE(pto2::detail::register_urma_async_event(ctx, Event{}, Session{}, workspace));
    EXPECT_EQ(count, 2u);
    EXPECT_FALSE(pto2::detail::register_urma_async_event(ctx, Event{}, Session{}, workspace));
    EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
}

TEST(A5UrmaCompletionKernel, InvalidOrUntrackedEventFailsWithoutWaiting) {
    AsyncCtx ctx{};
    EXPECT_FALSE(pto2::detail::register_urma_async_event(ctx, Event{}, Session{}, nullptr));
    ctx.task_token = TaskId::make(0, 42);
    uint32_t count = 0;
    int32_t error = 0;
    DeferredCompletionEntry entries[1]{};
    ctx.completion_count = &count;
    ctx.completion_error_code = &error;
    ctx.completion_entries = entries;
    ctx.completion_capacity = 1;
    uint8_t workspace[8]{};
    Event event;
    for (uint16_t jets : {0, 2}) {
        event.urmaJettyCount = jets;
        EXPECT_FALSE(pto2::detail::register_urma_async_event(ctx, event, Session{}, workspace));
        EXPECT_EQ(count, 0u);
        EXPECT_EQ(error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
}
