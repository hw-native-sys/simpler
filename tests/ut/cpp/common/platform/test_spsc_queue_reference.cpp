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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "spsc_queue.h"

namespace {

struct LayoutCase {
    uint64_t depth;
    uint64_t input_arena_bytes;
    uint64_t output_arena_bytes;
    uint64_t output_desc_offset;
    uint64_t input_arena_offset;
    uint64_t output_arena_offset;
    uint64_t payload_bytes;
};

constexpr std::array<LayoutCase, 4> kLayoutGolden{{
    {1, 64, 64, 32, 64, 128, 192},
    {4, 128, 192, 128, 256, 384, 576},
    {8, 192, 64, 256, 512, 704, 768},
    {2, 64, 128, 64, 128, 192, 320},
}};

constexpr uint8_t kDescriptorGoldenBytes[32] = {
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

}  // namespace

TEST(SpscQueueReference, LayoutGoldenVectors) {
    for (const auto &test_case : kLayoutGolden) {
        spsc_queue::SpscQueueLayout layout{};
        ASSERT_TRUE(
            spsc_queue::SpscQueueLayout::create(
                test_case.depth, test_case.input_arena_bytes, test_case.output_arena_bytes, &layout
            )
        );
        EXPECT_EQ(layout.input_desc_offset, 0u);
        EXPECT_EQ(layout.output_desc_offset, test_case.output_desc_offset);
        EXPECT_EQ(layout.input_arena_offset, test_case.input_arena_offset);
        EXPECT_EQ(layout.output_arena_offset, test_case.output_arena_offset);
        EXPECT_EQ(layout.payload_bytes, test_case.payload_bytes);
        EXPECT_EQ(layout.input_desc_tail_offset, 0u);
        EXPECT_EQ(layout.input_desc_head_offset, 64u);
        EXPECT_EQ(layout.output_desc_tail_offset, 128u);
        EXPECT_EQ(layout.output_desc_head_offset, 192u);
        EXPECT_EQ(layout.initiator_abort_offset, 256u);
        EXPECT_EQ(layout.peer_abort_offset, 320u);
        EXPECT_EQ(layout.counter_bytes, 384u);
    }
}

TEST(SpscQueueReference, LayoutRejectsInvalidDepthAndLeavesOutputUntouched) {
    spsc_queue::SpscQueueLayout layout{
        7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    };
    const spsc_queue::SpscQueueLayout original = layout;
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(3, 64, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, 65, 64, &layout));
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(2, std::numeric_limits<uint64_t>::max() - 63, 64, &layout));
    EXPECT_EQ(layout.depth, original.depth);
    EXPECT_EQ(layout.payload_bytes, original.payload_bytes);
    EXPECT_FALSE(spsc_queue::SpscQueueLayout::create(1, 64, 64, nullptr));
}

TEST(SpscQueueReference, OpcodeAndDescriptorBytesGoldenVector) {
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::INVALID), 0u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::DATA), 1u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::STOP), 2u);
    EXPECT_EQ(static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::ERROR), 3u);
    EXPECT_EQ(spsc_queue::kDescriptorBytes, 32u);

    spsc_queue::SpscQueueDescriptor descriptor{
        7,
        static_cast<uint64_t>(spsc_queue::SpscQueueOpcode::ERROR),
        128,
        16,
    };
    uint8_t encoded[32] = {};
    ASSERT_TRUE(spsc_queue::encode_descriptor(descriptor, encoded, sizeof(encoded)));
    EXPECT_EQ(std::memcmp(encoded, kDescriptorGoldenBytes, sizeof(kDescriptorGoldenBytes)), 0);
    spsc_queue::SpscQueueDescriptor decoded{};
    ASSERT_TRUE(spsc_queue::decode_descriptor(kDescriptorGoldenBytes, sizeof(kDescriptorGoldenBytes), &decoded));
    EXPECT_EQ(decoded.seq, 7u);
    EXPECT_EQ(decoded.opcode, 3u);
    EXPECT_EQ(decoded.payload_offset, 128u);
    EXPECT_EQ(decoded.payload_nbytes, 16u);
    EXPECT_FALSE(spsc_queue::decode_descriptor(kDescriptorGoldenBytes, 31, &decoded));
}

TEST(SpscQueueReference, TaskArgsAreRegionScalarsPlusThreeConfigValues) {
    static_assert(spsc_queue::kSpscQueueConfigScalarCount == 3, "application config is depth and two arenas");
    static_assert(REGION_TASK_SCALAR_COUNT + spsc_queue::kSpscQueueConfigScalarCount == 11, "11 task scalars");
}

namespace {

uint64_t g_now_ns = 0;

uint64_t test_now_ns() { return g_now_ns; }

struct FakePart {
    RegionPartLocalSpan span_value;
    RegionPartLocalSpan span() const { return span_value; }
};

struct FakeView {
    bool sticky_failed{false};
    RegionViewError last_error{};
    RegionPartLocalSpan payload_span{0x1000, 192};
    RegionPartLocalSpan counter_span{0x2000, 384};

    bool failed() const { return sticky_failed; }
    const RegionViewError &error() const { return last_error; }
    FakePart payload() const { return FakePart{payload_span}; }
    FakePart counter() const { return FakePart{counter_span}; }
};

spsc_queue::SpscQueueConfig config_for(uint64_t depth, uint64_t input_arena, uint64_t output_arena) {
    return spsc_queue::SpscQueueConfig{depth, input_arena, output_arena};
}

}  // namespace

TEST(SpscQueueReference, NullClockDoesNotReadTheView) {
    FakeView view{};
    view.sticky_failed = true;
    spsc_queue::SpscQueueEndpoint<FakeView> queue(view, config_for(1, 64, 64), spsc_queue::MonotonicClock{nullptr});
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::BAD_ARGUMENT);
    EXPECT_EQ(queue.region_error(), nullptr);
    EXPECT_NE(std::string(queue.error().message).find("SPSC reference example"), std::string::npos);
    EXPECT_EQ(std::string(queue.error().message).find("SPSC queue endpoint error"), std::string::npos);
}

TEST(SpscQueueReference, StickyViewFailureIsSeparateFromProtocolError) {
    FakeView view{};
    view.sticky_failed = true;
    view.last_error.kind = RegionViewErrorKind::ISSUED_FAILURE;
    view.last_error.op = RegionViewOp::NOTIFY;
    view.last_error.observed = 4;
    std::snprintf(view.last_error.message, sizeof(view.last_error.message), "issued copy failed");
    spsc_queue::SpscQueueEndpoint<FakeView> queue(
        view, config_for(1, 64, 64), spsc_queue::MonotonicClock{&test_now_ns}
    );
    EXPECT_FALSE(queue.live());
    EXPECT_EQ(queue.error().kind, spsc_queue::SpscQueueErrorKind::ENDPOINT_ERROR);
    ASSERT_NE(queue.region_error(), nullptr);
    EXPECT_EQ(queue.region_error()->kind, RegionViewErrorKind::ISSUED_FAILURE);
    EXPECT_EQ(queue.region_error()->observed, 4);
    EXPECT_NE(std::string(queue.region_error()->message).find("issued copy failed"), std::string::npos);

    FakeView healthy{};
    spsc_queue::SpscQueueEndpoint<FakeView, 4> too_wide(
        healthy, config_for(1, 64, 64), spsc_queue::MonotonicClock{&test_now_ns}
    );
    EXPECT_FALSE(too_wide.live());
    EXPECT_EQ(too_wide.error().kind, spsc_queue::SpscQueueErrorKind::BAD_ARGUMENT);
    EXPECT_EQ(too_wide.region_error(), nullptr);
}
