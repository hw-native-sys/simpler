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
#include <cstring>
#include "backend/async_backend.h"

namespace {
SchedulerAsyncGroupIdentity identity(int64_t task = 1) {
    SchedulerAsyncGroupIdentity result{};
    result.run_epoch = 19;
    result.task_ref = task;
    result.capacity = 64;
    return result;
}
DeferredCompletionEntry counter(uint64_t address, uint32_t expected = 1) {
    return {address, 0, expected, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
}
DeferredCompletionEntry event(uint64_t address, uint64_t expected = UINT64_C(0x100000001)) {
    return {address, expected, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
}

struct Reader {
    uint32_t *reads;
    int32_t signed_value;
    uint64_t post;
    int32_t counter(uint64_t) {
        ++*reads;
        return signed_value;
    }
    uint64_t sdma_post_done(uint64_t) {
        ++*reads;
        return post;
    }
};
struct Barrier {
    uint32_t *count;
    void finish() { ++*count; }
};
TEST(HbgAsyncBackend, CheckUsesOneObservationAndFullWidthThresholdWithoutResourceTables) {
    uint32_t reads = 0, barriers = 0;
    AsyncCompletionBackend<Reader, Barrier> pending({&reads, -1, UINT64_C(0x100000000)}, {&barriers});
    EXPECT_EQ(pending.observe_once(identity(), 0, counter(0x1000, 0)).state, SchedulerAsyncConditionState::PENDING);
    EXPECT_EQ(pending.observe_once(identity(), 0, event(0x2000)).state, SchedulerAsyncConditionState::PENDING);
    EXPECT_EQ(reads, 2u);
    EXPECT_EQ(barriers, 0u);
    AsyncCompletionBackend<Reader, Barrier> ready({&reads, 1, UINT64_C(0x100000001)}, {&barriers});
    EXPECT_EQ(ready.observe_once(identity(), 0, event(0x2000)).state, SchedulerAsyncConditionState::READY);
    EXPECT_EQ(reads, 3u);
    EXPECT_EQ(barriers, 1u);
    EXPECT_EQ(ready.observe_once(identity(), 0, counter(0)).state, SchedulerAsyncConditionState::FAILED);
    EXPECT_EQ(reads, 3u);
}
}  // namespace
