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
#include <cstring>
#include <functional>

#include "async_condition_writer.h"

namespace {

using namespace pto2::resident;

struct WriterImage {
    alignas(128) std::array<uint8_t, 8192> image{};
    AsyncWriter writer{image.data(), image.size(), 91, 128, 7};

    explicit WriterImage(uint16_t capacity = 4) {
        auto &data = group();
        data.run_epoch = writer.run_epoch;
        data.task_ref = 11;
        data.group_index = writer.group_index;
        data.block_index = 9;
        data.subtask_slot = 1;
        data.capacity = capacity;
        data.entries_bytes = async_group_entry_bytes(capacity);
        data.entries_offset = 4096;
    }

    AsyncGroupData &group() { return *reinterpret_cast<AsyncGroupData *>(image.data() + writer.group_offset); }

    DeferredCompletionEntry &entry(uint32_t index) {
        return reinterpret_cast<DeferredCompletionEntry *>(image.data() + group().entries_offset)[index];
    }
    static CompletionToken counter_token() { return {0x10080, 42, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0}; }
    static CompletionToken sdma_token() {
        return {0x20200, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 19};
    }
};

TEST(HbgAsyncWriter, CopiesShareReservationAndFirstError) {
    WriterImage fixture;
    const auto copy = fixture.writer;
    const auto reservation = reserve_conditions(fixture.writer, 4);
    ASSERT_NE(reservation.serial, 0u);
    EXPECT_EQ(fixture.group().count, 0u);
    ASSERT_TRUE(commit_condition(copy, reservation, WriterImage::counter_token()));
    ASSERT_TRUE(commit_condition(fixture.writer, reservation, WriterImage::sdma_token()));
    EXPECT_EQ(fixture.group().count, 2u);
    ASSERT_TRUE(finish_reservation(copy, reservation, 2));
    ASSERT_TRUE(register_completion_condition(fixture.writer, WriterImage::counter_token()));
    EXPECT_EQ(fixture.group().count, 3u);
    EXPECT_EQ(fixture.entry(0).expected_value, 42u);
    EXPECT_EQ(fixture.entry(1).backend_cookie, 19u);
    EXPECT_EQ(fixture.entry(1)._pad, 0u);
    EXPECT_TRUE(seal_conditions(copy));
}

TEST(HbgAsyncWriter, All64ConditionsFitWithoutOverlappingNextExecution) {
    WriterImage fixture(64);
    const uint64_t end = fixture.group().entries_offset + fixture.group().entries_bytes;
    fixture.image[end] = 0xa5;
    const auto reservation = reserve_conditions(fixture.writer, 64);
    for (uint32_t index = 0; index < 64; ++index) {
        auto token = WriterImage::counter_token();
        token.expected_value = index;
        ASSERT_TRUE(commit_condition(fixture.writer, reservation, token));
    }
    ASSERT_TRUE(finish_reservation(fixture.writer, reservation, 64));
    ASSERT_TRUE(seal_conditions(fixture.writer));
    for (uint32_t index = 0; index < 64; ++index) {
        EXPECT_EQ(fixture.entry(index).expected_value, index);
        EXPECT_EQ(fixture.entry(index)._pad, 0u);
    }
    EXPECT_EQ(fixture.image[end], 0xa5);
}

TEST(HbgAsyncWriter, EmptyAndUnusedReservationsSealButAnOpenReservationFails) {
    WriterImage empty;
    EXPECT_TRUE(seal_conditions(empty.writer));
    EXPECT_EQ(empty.group().count, 0u);
    WriterImage unused;
    const auto reservation = reserve_conditions(unused.writer, 4);
    EXPECT_TRUE(finish_reservation(unused.writer, reservation, 0));
    EXPECT_TRUE(seal_conditions(unused.writer));
    WriterImage open;
    ASSERT_NE(reserve_conditions(open.writer, 1).serial, 0u);
    EXPECT_FALSE(seal_conditions(open.writer));
    EXPECT_EQ(static_cast<AsyncPublicationState>(open.group().state), AsyncPublicationState::FAILED);
    EXPECT_EQ(open.group().error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(open.group().count, 0u);
}

TEST(HbgAsyncWriter, IgnoredRegistrationFailurePoisonsSealAndPreservesFirstError) {
    WriterImage fixture(1);
    EXPECT_TRUE(register_completion_condition(fixture.writer, WriterImage::counter_token()));
    EXPECT_FALSE(register_completion_condition(fixture.writer, WriterImage::counter_token()));
    EXPECT_EQ(fixture.group().error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_FALSE(seal_conditions(fixture.writer));
    EXPECT_EQ(fixture.group().error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(fixture.group().count, 1u);
}

TEST(HbgAsyncWriter, NestedAndStaleReservationsCannotClaimAnotherTransaction) {
    WriterImage nested;
    ASSERT_NE(reserve_conditions(nested.writer, 2).serial, 0u);
    EXPECT_EQ(reserve_conditions(nested.writer, 1).serial, 0u);
    EXPECT_FALSE(seal_conditions(nested.writer));
    WriterImage stale;
    const auto old = reserve_conditions(stale.writer, 1);
    ASSERT_TRUE(finish_reservation(stale.writer, old, 0));
    const auto current = reserve_conditions(stale.writer, 1);
    ASSERT_NE(old.serial, current.serial);
    EXPECT_FALSE(commit_condition(stale.writer, old, WriterImage::counter_token()));
    EXPECT_EQ(stale.group().count, 0u);
    EXPECT_FALSE(commit_condition(stale.writer, current, WriterImage::counter_token()));
    EXPECT_FALSE(seal_conditions(stale.writer));
}

TEST(HbgAsyncWriter, ReservationBoundAndActualCountCannotLoseSubmittedConditions) {
    for (bool overcommit : {false, true}) {
        WriterImage fixture;
        const auto reservation = reserve_conditions(fixture.writer, 1);
        ASSERT_TRUE(commit_condition(fixture.writer, reservation, WriterImage::counter_token()));
        if (overcommit) EXPECT_FALSE(commit_condition(fixture.writer, reservation, WriterImage::counter_token()));
        else EXPECT_FALSE(finish_reservation(fixture.writer, reservation, 0));
        EXPECT_EQ(fixture.group().count, 1u);
        EXPECT_FALSE(seal_conditions(fixture.writer));
    }
}

TEST(HbgAsyncWriter, NullAndMisalignedCompletionAddressesFail) {
    for (uint64_t address : {UINT64_C(0), UINT64_C(0x10001), UINT64_MAX}) {
        WriterImage fixture;
        auto token = WriterImage::counter_token();
        token.addr = address;
        EXPECT_FALSE(register_completion_condition(fixture.writer, token));
        EXPECT_EQ(fixture.group().count, 0u);
        EXPECT_FALSE(seal_conditions(fixture.writer));
    }
}

TEST(HbgAsyncWriter, ExpectedWidthEngineTypeAndCookieAreCheckedBeforeWriting) {
    const std::array<std::function<void(CompletionToken &)>, 4> mutations{
        [](auto &token) {
            token.expected_value = UINT32_MAX;
        },
        [](auto &token) {
            token.engine = COMPLETION_ENGINE_URMA;
        },
        [](auto &token) {
            token.completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
        },
        [](auto &token) {
            token.backend_cookie = 1;
        }
    };
    for (const auto &mutate : mutations) {
        WriterImage fixture;
        auto token = WriterImage::counter_token();
        mutate(token);
        EXPECT_FALSE(register_completion_condition(fixture.writer, token));
        EXPECT_EQ(fixture.group().count, 0u);
    }
    for (uint64_t post : {UINT64_C(0)}) {
        WriterImage fixture;
        auto token = WriterImage::sdma_token();
        token.backend_cookie = post;
        EXPECT_FALSE(register_completion_condition(fixture.writer, token));
        EXPECT_FALSE(seal_conditions(fixture.writer));
    }
}

TEST(HbgAsyncWriter, InvalidImageSpansAndCapacityAreRejectedBeforeArrayAccess) {
    const std::array<std::function<void(AsyncGroupData &)>, 9> mutations{
        [](auto &g) {
            g.capacity = 0;
        },
        [](auto &g) {
            g.capacity = 65;
        },
        [](auto &g) {
            g.entries_bytes = UINT32_MAX;
        },
        [](auto &g) {
            g.subtask_slot = 3;
        },
        [](auto &g) {
            g.count = 5;
        },
        [](auto &g) {
            g.entries_offset = 128;
        },
        [](auto &g) {
            ++g.entries_offset;
        },
        [](auto &g) {
            g.entries_offset = 8192;
        },
        [](auto &g) {
            g.entries_bytes += sizeof(AsyncGroupData);
        }
    };
    for (const auto &mutate : mutations) {
        WriterImage fixture;
        mutate(fixture.group());
        EXPECT_EQ(reserve_conditions(fixture.writer, 1).serial, 0u);
        EXPECT_FALSE(seal_conditions(fixture.writer));
    }
}

TEST(HbgAsyncWriter, StaleOrAbsentContextCannotModifyAnotherRunOrGroup) {
    WriterImage fixture;
    for (uint32_t mutation = 0; mutation < 5; ++mutation) {
        auto writer = fixture.writer;
        if (mutation == 0) ++writer.run_epoch;
        if (mutation == 1) ++writer.group_index;
        if (mutation == 2) writer.image_bytes = 128;
        if (mutation == 3) writer.group_offset = UINT64_MAX - 127;
        if (mutation == 4) writer.image = nullptr;
        EXPECT_EQ(reserve_conditions(writer, 1).serial, 0u);
        EXPECT_FALSE(seal_conditions(writer));
    }
    EXPECT_EQ(fixture.group().error_code, 0);
    EXPECT_EQ(static_cast<AsyncPublicationState>(fixture.group().state), AsyncPublicationState::OPEN);
}

TEST(HbgAsyncWriter, SealedPublicationIsImmutableAndCannotBeReplayed) {
    WriterImage fixture;
    ASSERT_TRUE(register_completion_condition(fixture.writer, WriterImage::counter_token()));
    ASSERT_TRUE(seal_conditions(fixture.writer));
    const auto snapshot = fixture.image;
    EXPECT_FALSE(register_completion_condition(fixture.writer, WriterImage::counter_token()));
    EXPECT_FALSE(seal_conditions(fixture.writer));
    EXPECT_EQ(fixture.image, snapshot);
}

TEST(HbgAsyncWriter, ReservationSerialNeverWraps) {
    WriterImage fixture;
    fixture.group().reservation_serial = UINT32_MAX;
    EXPECT_EQ(reserve_conditions(fixture.writer, 1).serial, 0u);
    EXPECT_EQ(fixture.group().reservation_serial, UINT32_MAX);
    EXPECT_FALSE(seal_conditions(fixture.writer));
}

}  // namespace
