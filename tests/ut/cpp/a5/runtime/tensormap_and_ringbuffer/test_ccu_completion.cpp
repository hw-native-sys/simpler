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
#include <array>
#include <cstddef>
#include <new>

#include "async_wait.h"
#include "runtime.h"
namespace {
TEST(TmrCcuCompletion, OrdinaryRuntimeDoesNotInheritWorkspaceFromReusedStorage) {
    alignas(Runtime) std::array<std::byte, sizeof(Runtime)> storage;
    storage.fill(std::byte{0xa5});
    auto *runtime = new (storage.data()) Runtime;
    EXPECT_EQ(runtime->dev.ccu_workspace_addr, 0U);
    runtime->~Runtime();
}

CompletionCondition condition(uint64_t &event) {
    CompletionCondition cond;
    cond.engine = ASYNC_ENGINE_CCU;
    cond.completion_type = COMPLETION_TYPE_CCU;
    cond.addr = reinterpret_cast<uint64_t>(&event);
    cond.backend_cookie = 1;
    return cond;
}
TEST(TmrCcuCompletion, WaitsForIndependentEventsAndConfirmsClearBeforeRetirement) {
    uint64_t a{}, b{1};
    auto first = condition(a);
    auto second = condition(b);
    EXPECT_EQ(first.test().state, CompletionPollState::PENDING);
    EXPECT_EQ(second.test().state, CompletionPollState::READY);
    EXPECT_EQ(b, 0U);
    a = 1;
    EXPECT_EQ(first.test().state, CompletionPollState::READY);
    EXPECT_EQ(a, 0U);
    second.retire();
    EXPECT_TRUE(second.retired);
    EXPECT_FALSE(first.retired);
}
TEST(TmrCcuCompletion, RejectsInvalidEventMask) {
    uint64_t event{1};
    auto cond = condition(event);
    cond.backend_cookie = 2;
    EXPECT_EQ(cond.test().state, CompletionPollState::FAILED);
}

}  // namespace
