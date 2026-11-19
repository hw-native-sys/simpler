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

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "scheduler/scheduler.h"
#include "support/test_task_id.h"

TEST(AsyncWaitPollHint, ReadsCountWhileAnotherDispatcherDrainsMailbox) {
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    std::atomic_init(&mailbox->head, uint64_t{0});
    std::atomic_init(&mailbox->tail, uint64_t{0});
    std::atomic_init(&mailbox->entries[0].seq, uint64_t{0});
    auto waits = std::make_unique<AsyncWaitList>();
    EXPECT_FALSE(waits->needs_poll(*mailbox));

    const auto token = simpler::ut::test_task_id(42);
    ASSERT_TRUE(mailbox->try_push_condition(token, 1, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER));
    ASSERT_TRUE(waits->needs_poll(*mailbox));

    std::atomic<bool> start{false};
    int32_t drained = 0;
    int32_t error = SIMPLER_ERROR_NONE;
    std::thread drainer([&] {
        ASSERT_TRUE(waits->try_lock());
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();
        AsyncWaitList::DrainCompletionSink sink{};
        drained = waits->drain_aicore_completion_mailbox_locked(mailbox.get(), sink, error);
        waits->unlock();
    });

    // Release the writer without ordering its count update against the reader.
    // Either hint is valid here, so keep the read observable for TSAN.
    start.store(true, std::memory_order_release);
    const volatile bool hint = waits->needs_poll(*mailbox);
    (void)hint;
    drainer.join();

    ASSERT_EQ(error, SIMPLER_ERROR_NONE);
    ASSERT_EQ(drained, 1);
    EXPECT_FALSE(mailbox->has_pending());
    EXPECT_TRUE(waits->needs_poll(*mailbox));
}
