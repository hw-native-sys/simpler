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
#include <cstdint>
#include <limits>

#include "backend/async_observer.h"

namespace {

DeferredCompletionEntry counter_entry(uint64_t address, uint32_t expected) {
    return {address, 0, expected, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
}

DeferredCompletionEntry sdma_entry(uint64_t address, uint64_t expected) {
    return {address, expected, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0};
}

struct CountingReader {
    int32_t counter_value{0};
    uint64_t post_value{0};
    uint32_t counter_reads{0};
    uint32_t post_reads{0};
    uint64_t address{0};

    int32_t counter(uint64_t at) {
        ++counter_reads;
        address = at;
        return counter_value;
    }
    uint64_t sdma_post_done(uint64_t at) {
        ++post_reads;
        address = at;
        return post_value;
    }
};

TEST(HbgAsyncObserver, CounterUsesSignedComparisonAtNonnegativeThresholdBoundaries) {
    struct Case {
        int32_t value;
        uint32_t expected;
        CompletionPollState state;
    };
    const Case cases[] = {
        {0, 0, CompletionPollState::READY},
        {0, 1, CompletionPollState::PENDING},
        {1, 1, CompletionPollState::READY},
        {2, 1, CompletionPollState::READY},
        {INT32_MAX - 1, INT32_MAX, CompletionPollState::PENDING},
        {INT32_MAX, INT32_MAX, CompletionPollState::READY},
        {-1, 0, CompletionPollState::PENDING},
        {INT32_MIN, 0, CompletionPollState::PENDING},
    };
    for (const auto &item : cases) {
        CountingReader reader;
        reader.counter_value = item.value;
        const auto result = async_observe_once(counter_entry(0x1004, item.expected), reader);
        EXPECT_EQ(result.state, item.state);
        EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(reader.counter_reads, 1u);
        EXPECT_EQ(reader.post_reads, 0u);
        EXPECT_EQ(reader.address, 0x1004u);
    }
}

TEST(HbgAsyncObserver, SdmaComparesTheFull64BitPostIdAndReadsOnlyOnce) {
    const uint64_t expected = (UINT64_C(1) << 40) + 7;
    for (uint64_t value : {expected - 1, expected, expected + 1}) {
        CountingReader reader;
        reader.post_value = value;
        const auto result = async_observe_once(sdma_entry(0x1080, expected), reader);
        EXPECT_EQ(result.state, value < expected ? CompletionPollState::PENDING : CompletionPollState::READY);
        EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(reader.post_reads, 1u);
        EXPECT_EQ(reader.counter_reads, 0u);
        EXPECT_EQ(reader.address, 0x1080u);
    }
    CountingReader reader;
    reader.post_value = UINT32_MAX;
    EXPECT_EQ(async_observe_once(sdma_entry(0x1080, expected), reader).state, CompletionPollState::PENDING);
    EXPECT_EQ(reader.post_reads, 1u);
}

TEST(HbgAsyncObserver, MalformedEntriesNeverReadExternalMemory) {
    const DeferredCompletionEntry invalid[] = {
        counter_entry(0, 0),
        counter_entry(0x1002, 1),
        counter_entry(0x1004, static_cast<uint32_t>(INT32_MAX) + 1),
        {0x1004, 1, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0},
        {0x1004, 0, 1, COMPLETION_ENGINE_ROCE, COMPLETION_TYPE_COUNTER, 0},
        {0x1004, 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 1},
        sdma_entry(0, 1),
        sdma_entry(0x1004, 1),
        sdma_entry(0x1080, 0),
        {0x1080, 1, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 0},
        {0x1080, 0, 0, COMPLETION_ENGINE_SDMA, 99, 0},
    };
    for (const auto &entry : invalid) {
        CountingReader reader;
        const auto result = async_observe_once(entry, reader);
        EXPECT_EQ(result.state, CompletionPollState::FAILED);
        EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(reader.counter_reads + reader.post_reads, 0u);
    }
}

TEST(HbgAsyncObserver, ActualCounterLoadsPreserveTheConditionAndNeighbours) {
    alignas(64) std::array<int32_t, 16> words{};
    words.fill(73);
    words[3] = 0;
    const auto entry = counter_entry(reinterpret_cast<uint64_t>(&words[3]), 1);
    EXPECT_EQ(async_observe_once(entry).state, CompletionPollState::PENDING);
    EXPECT_EQ(words[3], 0);
    words[3] = 1;
    EXPECT_EQ(async_observe_once(entry).state, CompletionPollState::READY);
    EXPECT_EQ(words[3], 1);
    for (size_t i = 0; i < words.size(); ++i)
        if (i != 3) EXPECT_EQ(words[i], 73);
}

TEST(HbgAsyncObserver, ActualSdmaLoadNeverClearsOrRetiresTheSharedRecord) {
    alignas(64) std::array<uint64_t, 8> words{};
    words.fill(UINT64_C(0x123456789abcdef));
    const uint64_t expected = (UINT64_C(1) << 48) + 9;
    words[0] = expected - 1;
    const auto entry = sdma_entry(reinterpret_cast<uint64_t>(words.data()), expected);
    EXPECT_EQ(async_observe_once(entry).state, CompletionPollState::PENDING);
    EXPECT_EQ(words[0], expected - 1);
    words[0] = expected + 1;
    EXPECT_EQ(async_observe_once(entry).state, CompletionPollState::READY);
    EXPECT_EQ(async_observe_once(entry).state, CompletionPollState::READY);
    EXPECT_EQ(words[0], expected + 1);
    for (size_t i = 1; i < words.size(); ++i)
        EXPECT_EQ(words[i], UINT64_C(0x123456789abcdef));
}

}  // namespace
