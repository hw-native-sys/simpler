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

#include <array>
#include <climits>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "aicpu/region_instance_view.h"

namespace {

struct RegionStorage {
    alignas(64) std::array<uint8_t, 128> payload{};
    alignas(64) std::array<int32_t, 32> counters{};
};

RegionPartLocalSpan payload_span(RegionStorage *storage) {
    return RegionPartLocalSpan{reinterpret_cast<uint64_t>(storage->payload.data()), storage->payload.size()};
}

RegionPartLocalSpan counter_span(RegionStorage *storage) {
    return RegionPartLocalSpan{
        reinterpret_cast<uint64_t>(storage->counters.data()), storage->counters.size() * sizeof(int32_t)
    };
}

RegionInstanceView make_view(RegionStorage *storage) {
    return RegionInstanceView(payload_span(storage), counter_span(storage));
}

TEST(RegionInstanceViewTest, AcceptsIndependentNonAdjacentZeroAndReversedBases) {
    alignas(64) std::array<uint8_t, 64> first{};
    alignas(64) std::array<uint8_t, 64> second{};
    RegionPartLocalSpan payload{reinterpret_cast<uint64_t>(first.data()), first.size()};
    RegionPartLocalSpan counter{reinterpret_cast<uint64_t>(second.data()), second.size()};

    RegionInstanceView adjacent_or_not(payload, counter);
    ASSERT_FALSE(adjacent_or_not.failed()) << adjacent_or_not.error().message;
    EXPECT_EQ(adjacent_or_not.payload_span().base, payload.base);
    EXPECT_EQ(adjacent_or_not.counter_span().base, counter.base);

    RegionInstanceView reversed(counter, payload);
    ASSERT_FALSE(reversed.failed()) << reversed.error().message;
    EXPECT_EQ(reversed.payload_span().base, counter.base);
    EXPECT_EQ(reversed.counter_span().base, payload.base);

    RegionInstanceView zero_payload(RegionPartLocalSpan{0, 64}, RegionPartLocalSpan{256, 64});
    ASSERT_FALSE(zero_payload.failed()) << zero_payload.error().message;
}

TEST(RegionInstanceViewTest, InvalidConstructionIsStickyAndHasNoIdentityCookie) {
    static_assert(sizeof(RegionViewError::kind) == sizeof(uint32_t), "error kind is not an identity cookie");
    RegionInstanceView uninitialized;
    EXPECT_TRUE(uninitialized.failed());
    EXPECT_EQ(uninitialized.error().kind, RegionViewErrorKind::INVALID_VIEW);

    RegionInstanceView overlap(RegionPartLocalSpan{0, 128}, RegionPartLocalSpan{64, 64});
    EXPECT_TRUE(overlap.failed());
    EXPECT_EQ(overlap.error().kind, RegionViewErrorKind::INVALID_VIEW);
    EXPECT_EQ(overlap.error().op, RegionViewOp::CONSTRUCT);
    EXPECT_STRNE(overlap.error().message, "");

    RegionInstanceView unaligned(RegionPartLocalSpan{64, 64}, RegionPartLocalSpan{1, 64});
    EXPECT_TRUE(unaligned.failed());
    EXPECT_EQ(unaligned.error().kind, RegionViewErrorKind::INVALID_VIEW);

    RegionInstanceView odd_bytes(RegionPartLocalSpan{64, 64}, RegionPartLocalSpan{128, 6});
    EXPECT_TRUE(odd_bytes.failed());
    EXPECT_EQ(odd_bytes.error().kind, RegionViewErrorKind::INVALID_VIEW);

    RegionPayloadView view{1, 1};
    EXPECT_FALSE(overlap.payload().write(0, "x", 1));
    EXPECT_FALSE(overlap.payload().read(0, 1, view));
    EXPECT_EQ(view.local_addr, 0u);
    EXPECT_EQ(overlap.error().kind, RegionViewErrorKind::INVALID_VIEW);
}

TEST(RegionInstanceViewTest, RejectsOverflowUnalignedAndNonMultipleCounterSpans) {
    EXPECT_FALSE(region_validate_payload_span(RegionPartLocalSpan{UINT64_MAX - 7, 16}));
    EXPECT_FALSE(region_validate_counter_span(RegionPartLocalSpan{1, 64}));
    EXPECT_FALSE(region_validate_counter_span(RegionPartLocalSpan{64, 6}));
    EXPECT_FALSE(region_validate_counter_span(RegionPartLocalSpan{64, 0}));
    EXPECT_FALSE(region_validate_independent_spans(RegionPartLocalSpan{0, 128}, RegionPartLocalSpan{64, 64}));
}

TEST(RegionInstanceViewTest, PayloadWriteAndReadUseOffsetZeroSpans) {
    RegionStorage storage{};
    RegionInstanceView view = make_view(&storage);
    ASSERT_FALSE(view.failed()) << view.error().message;
    EXPECT_EQ(view.payload().span().logical_bytes, storage.payload.size());
    EXPECT_EQ(view.counter().span().logical_bytes, storage.counters.size() * sizeof(int32_t));

    const uint32_t marker = 0xA5B6C7D8u;
    ASSERT_TRUE(view.payload().write(12, &marker, sizeof(marker))) << view.error().message;
    uint32_t observed = 0;
    std::memcpy(&observed, storage.payload.data() + 12, sizeof(observed));
    EXPECT_EQ(observed, marker);

    RegionPayloadView inner{};
    ASSERT_TRUE(view.payload().read(12, sizeof(marker), inner)) << view.error().message;
    EXPECT_EQ(inner.nbytes, sizeof(marker));
    EXPECT_EQ(*reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(inner.local_addr)), marker);
}

TEST(RegionInstanceViewTest, PreIssuePayloadBoundsDoNotPoison) {
    RegionStorage storage{};
    RegionInstanceView view = make_view(&storage);
    RegionPayloadView inner{0xCAFE, 0xBEEF};

    EXPECT_FALSE(view.payload().read(120, 16, inner));
    EXPECT_EQ(inner.local_addr, 0u);
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::OUT_OF_BOUNDS);
    EXPECT_FALSE(view.failed());

    EXPECT_FALSE(view.payload().write(0, nullptr, 1));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::OUT_OF_BOUNDS);
    EXPECT_FALSE(view.failed());

    const uint8_t byte = 0x11;
    EXPECT_TRUE(view.payload().write(0, &byte, 1)) << view.error().message;
    EXPECT_EQ(storage.payload[0], 0x11);
}

TEST(RegionInstanceViewTest, NotifyTestAndWaitFollowNeutralPoisonRules) {
    RegionStorage storage{};
    RegionInstanceView view = make_view(&storage);
    uint64_t addr = 0;
    ASSERT_TRUE(view.counter().addr(0, addr)) << view.error().message;
    EXPECT_TRUE(view.counter().contains_addr(addr));

    ASSERT_TRUE(view.counter().notify(0, 5, RegionNotifyOp::Set)) << view.error().message;
    EXPECT_EQ(storage.counters[0], 5);
    ASSERT_TRUE(view.counter().notify(0, -2, RegionNotifyOp::Add)) << view.error().message;
    EXPECT_EQ(storage.counters[0], 3);

    RegionSignalTestResult result{};
    storage.counters[1] = 7;
    ASSERT_TRUE(view.counter().test(4, 7, RegionWaitCmp::EQ, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    EXPECT_EQ(result.observed, 7);
    ASSERT_TRUE(view.counter().test(4, 8, RegionWaitCmp::NE, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    ASSERT_TRUE(view.counter().test(4, 6, RegionWaitCmp::GT, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    ASSERT_TRUE(view.counter().test(4, 7, RegionWaitCmp::GE, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    ASSERT_TRUE(view.counter().test(4, 8, RegionWaitCmp::LT, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    ASSERT_TRUE(view.counter().test(4, 7, RegionWaitCmp::LE, result)) << view.error().message;
    EXPECT_TRUE(result.matched);
    ASSERT_TRUE(view.counter().test(4, 8, RegionWaitCmp::EQ, result)) << view.error().message;
    EXPECT_FALSE(result.matched);
    EXPECT_EQ(result.observed, 7);
    EXPECT_FALSE(view.failed());
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::NONE);

    int32_t observed = 0;
    EXPECT_FALSE(view.counter().wait(0, 99, RegionWaitCmp::EQ, 1, observed));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::TIMEOUT);
    EXPECT_FALSE(view.failed());
    EXPECT_EQ(observed, 3);
    EXPECT_TRUE(view.counter().notify(0, 9, RegionNotifyOp::Set)) << view.error().message;
    EXPECT_TRUE(view.counter().wait(0, 8, RegionWaitCmp::GE, 1'000'000, observed)) << view.error().message;
    EXPECT_EQ(observed, 9);
}

TEST(RegionInstanceViewTest, InvalidEnumAndUnalignedOffsetDoNotPoison) {
    RegionStorage storage{};
    RegionInstanceView view = make_view(&storage);

    EXPECT_FALSE(view.counter().notify(0, 1, static_cast<RegionNotifyOp>(99)));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::INVALID_ENUM);
    EXPECT_FALSE(view.failed());

    uint64_t addr = 0xCAFE;
    EXPECT_FALSE(view.counter().addr(2, addr));
    EXPECT_EQ(addr, 0u);
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::OUT_OF_BOUNDS);
    EXPECT_FALSE(view.failed());
    EXPECT_TRUE(view.counter().notify(0, 1, RegionNotifyOp::Set)) << view.error().message;
}

struct FailingCopyOps : LocalMemoryOps {
    bool copy_bytes(void *, const void *, size_t) const { return false; }
};

struct FailingLoadOps : LocalMemoryOps {
    bool load_i32(uint64_t, int32_t &) const { return false; }
};

TEST(RegionInstanceViewTest, IssuedLocalOpsFailureIsSticky) {
    RegionStorage storage{};
    RegionInstanceViewImpl<FailingCopyOps> view(payload_span(&storage), counter_span(&storage));
    ASSERT_FALSE(view.failed()) << view.error().message;
    const uint32_t marker = 1;
    EXPECT_FALSE(view.payload().write(0, &marker, sizeof(marker)));
    EXPECT_TRUE(view.failed());
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
    EXPECT_EQ(view.error().op, RegionViewOp::PAYLOAD_WRITE);

    RegionPayloadView inner{};
    EXPECT_FALSE(view.payload().read(0, 4, inner));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
}

TEST(RegionInstanceViewTest, DefaultInvalidAndIssuedFailedViewsHaveNoReset) {
    RegionInstanceView uninitialized;
    EXPECT_TRUE(uninitialized.failed());
    EXPECT_EQ(uninitialized.error().kind, RegionViewErrorKind::INVALID_VIEW);
    RegionPayloadView unread{};
    EXPECT_FALSE(uninitialized.payload().read(0, 1, unread));
    EXPECT_EQ(uninitialized.error().kind, RegionViewErrorKind::INVALID_VIEW);

    RegionInstanceView overlap(RegionPartLocalSpan{0, 128}, RegionPartLocalSpan{64, 64});
    EXPECT_TRUE(overlap.failed());
    const uint8_t byte = 1;
    EXPECT_FALSE(overlap.payload().write(0, &byte, 1));
    EXPECT_EQ(overlap.error().kind, RegionViewErrorKind::INVALID_VIEW);
    EXPECT_EQ(overlap.error().op, RegionViewOp::CONSTRUCT);
}

TEST(RegionInstanceViewTest, AddUsesDefined32BitWrap) {
    RegionStorage storage{};
    RegionInstanceView view = make_view(&storage);
    ASSERT_FALSE(view.failed()) << view.error().message;

    ASSERT_TRUE(view.counter().notify(0, INT32_MAX, RegionNotifyOp::Set)) << view.error().message;
    ASSERT_TRUE(view.counter().notify(0, 1, RegionNotifyOp::Add)) << view.error().message;
    EXPECT_EQ(storage.counters[0], INT32_MIN);

    ASSERT_TRUE(view.counter().notify(0, INT32_MIN, RegionNotifyOp::Set)) << view.error().message;
    ASSERT_TRUE(view.counter().notify(0, -1, RegionNotifyOp::Add)) << view.error().message;
    EXPECT_EQ(storage.counters[0], INT32_MAX);

    ASSERT_TRUE(view.counter().notify(0, 10, RegionNotifyOp::Set)) << view.error().message;
    ASSERT_TRUE(view.counter().notify(0, 4, RegionNotifyOp::Add)) << view.error().message;
    EXPECT_EQ(storage.counters[0], 14);

    ASSERT_TRUE(view.counter().notify(0, 10, RegionNotifyOp::Set)) << view.error().message;
    ASSERT_TRUE(view.counter().notify(0, -3, RegionNotifyOp::Add)) << view.error().message;
    EXPECT_EQ(storage.counters[0], 7);
}

struct FailingInvalidateOps : LocalMemoryOps {
    bool invalidate(const void *, size_t) const { return false; }
};

TEST(RegionInstanceViewTest, IssuedPayloadReadFailureIsSticky) {
    RegionStorage storage{};
    RegionInstanceViewImpl<FailingInvalidateOps> view(payload_span(&storage), counter_span(&storage));
    ASSERT_FALSE(view.failed()) << view.error().message;
    RegionPayloadView inner{};
    EXPECT_FALSE(view.payload().read(0, 4, inner));
    EXPECT_TRUE(view.failed());
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
    EXPECT_EQ(view.error().op, RegionViewOp::PAYLOAD_READ);
    EXPECT_FALSE(view.payload().write(0, "x", 1));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
}

TEST(RegionInstanceViewTest, IssuedCounterLoadFailureIsSticky) {
    RegionStorage storage{};
    RegionInstanceViewImpl<FailingLoadOps> view(payload_span(&storage), counter_span(&storage));
    ASSERT_FALSE(view.failed()) << view.error().message;
    EXPECT_FALSE(view.counter().notify(0, 1, RegionNotifyOp::Add));
    EXPECT_TRUE(view.failed());
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
    EXPECT_EQ(view.error().op, RegionViewOp::NOTIFY);
    EXPECT_FALSE(view.counter().notify(0, 1, RegionNotifyOp::Set));
    EXPECT_EQ(view.error().kind, RegionViewErrorKind::ISSUED_FAILURE);
}

}  // namespace
