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
#include "async_condition_writer.h"
#include "platform_comm/ccu/ccu_completion.h"
#include "runtime.h"
namespace {
TEST(HbgCcuCompletion, OrdinaryRuntimeDoesNotInheritWorkspaceFromReusedStorage) {
    alignas(Runtime) std::array<std::byte, sizeof(Runtime)> storage;
    storage.fill(std::byte{0xa5});
    auto *runtime = new (storage.data()) Runtime;
    EXPECT_EQ(runtime->dev.ccu_workspace_addr, 0U);
    runtime->~Runtime();
}

TEST(HbgCcuCompletion, PollsIndependentEventsAndClearsReadyEvents) {
    uint64_t a{}, b{1};
    EXPECT_EQ(
        simpler::ccu::poll_completion(reinterpret_cast<uint64_t>(&a), 1).state,
        simpler::ccu::CompletionPollState::pending
    );
    EXPECT_EQ(
        simpler::ccu::poll_completion(reinterpret_cast<uint64_t>(&b), 1).state, simpler::ccu::CompletionPollState::ready
    );
    EXPECT_EQ(b, 0U);
    a = 1;
    EXPECT_EQ(
        simpler::ccu::poll_completion(reinterpret_cast<uint64_t>(&a), 1).state, simpler::ccu::CompletionPollState::ready
    );
    EXPECT_EQ(a, 0U);
}
TEST(HbgCcuCompletion, RejectsInvalidEventMask) {
    uint64_t event{1};
    EXPECT_EQ(
        simpler::ccu::poll_completion(reinterpret_cast<uint64_t>(&event), 2).state,
        simpler::ccu::CompletionPollState::protocol_error
    );
}
TEST(HbgCcuCompletion, ResidentWriterAcceptsCanonicalEventToken) {
    uint64_t event{};
    CompletionToken token{reinterpret_cast<uint64_t>(&event), 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, 1};
    EXPECT_TRUE(pto2::resident::detail::token_valid(token));
    token.backend_cookie = 2;
    EXPECT_FALSE(pto2::resident::detail::token_valid(token));
}
}  // namespace
