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
#include <vector>

#include "host/l0_trace_records.h"
#include "common/l0_profiling.h"

using simpler::dfx::l0::BusyInterval;
using simpler::dfx::l0::BusyIntervalBuilder;
using simpler::dfx::l0::TraceDecoder;
using simpler::dfx::l0::TraceEvent;
using simpler::dfx::l0::TraceKind;

namespace {
void append_word(std::vector<uint8_t> &bytes, uint32_t word) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(word >> shift));
}
}  // namespace

TEST(L0TraceDecoder, SplitStartAndStateAcrossEveryByte) {
    std::vector<uint8_t> bytes;
    append_word(bytes, 0xE0120004u);
    append_word(bytes, 0xE0340003u);
    append_word(bytes, 0xE0000002u);
    append_word(bytes, 0xE0000001u);
    append_word(bytes, 0xF0030005u);
    append_word(bytes, 0x40070006u);
    append_word(bytes, 0xEFFFFFFFu);
    append_word(bytes, 0xDDDDDDDDu);

    TraceDecoder decoder;
    std::vector<TraceEvent> events;
    for (const uint8_t byte : bytes)
        decoder.feed(&byte, 1, [&](const TraceEvent &event) {
            events.push_back(event);
        });
    const auto &stats = decoder.finish();
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events[0].kind, TraceKind::StartStamp);
    EXPECT_EQ(events[0].anchor_syscnt, 0x0001000200030004ull);
    EXPECT_EQ(events[0].elapsed_cycles, 0u);
    EXPECT_EQ(events[0].block_id, 0x3412u);
    EXPECT_EQ(events[1].kind, TraceKind::State);
    EXPECT_EQ(events[1].anchor_syscnt, events[0].anchor_syscnt);
    EXPECT_EQ(events[1].elapsed_cycles, 5u);
    EXPECT_EQ(events[2].kind, TraceKind::MarkStamp);
    EXPECT_EQ(events[2].payload, 7u);
    EXPECT_EQ(events[2].elapsed_cycles, 11u);
    EXPECT_EQ(events[3].kind, TraceKind::EndFlag);
    EXPECT_EQ(stats.end_flags, 1u);
    EXPECT_EQ(stats.fillers, 1u);
    EXPECT_TRUE(stats.structurally_complete());
}

TEST(L0TraceDecoder, ClipsStateIntervalsAtObservedSegmentBoundaries) {
    BusyIntervalBuilder builder;
    std::vector<BusyInterval> intervals;
    const auto emit = [&](const BusyInterval &interval) {
        intervals.push_back(interval);
    };
    const uint64_t anchor = 1234;
    builder.accept({TraceKind::StartStamp, anchor, 0, 0, 7, 0, 14}, emit);
    builder.accept({TraceKind::State, anchor, 5, 0, 7, 0x2, 15}, emit);  // VEC is already busy.
    builder.accept({TraceKind::State, anchor, 8, 0, 7, 0x3, 15}, emit);  // SCALAR becomes busy.
    builder.accept({TraceKind::MarkStamp, anchor, 10, 0, 7, 9, 0}, emit);
    builder.accept({TraceKind::State, anchor, 15, 0, 7, 0x2, 15}, emit);  // SCALAR closes.
    builder.accept({TraceKind::EndFlag, anchor, 15, 0, 7, 0, 14}, emit);
    const auto &stats = builder.finish(emit);
    ASSERT_EQ(intervals.size(), 2u);
    EXPECT_EQ(intervals[0].anchor_syscnt, anchor);
    EXPECT_EQ(intervals[0].start_cycles, 8u);
    EXPECT_EQ(intervals[0].end_cycles, 15u);
    EXPECT_EQ(intervals[0].lane, 0u);
    EXPECT_EQ(intervals[0].block_id, 7u);
    EXPECT_EQ(intervals[0].clip_flags, 0u);
    EXPECT_EQ(intervals[1].start_cycles, 5u);
    EXPECT_EQ(intervals[1].end_cycles, 15u);
    EXPECT_EQ(intervals[1].lane, 1u);
    EXPECT_EQ(intervals[1].clip_flags, 3u);
    EXPECT_EQ(stats.emitted, 2u);
    EXPECT_EQ(stats.clipped_start, 1u);
    EXPECT_EQ(stats.clipped_end, 1u);
}

TEST(L0TraceDecoder, ClipsOpenEndToLastObservedTimestampWithoutEndFlag) {
    BusyIntervalBuilder builder;
    std::vector<BusyInterval> intervals;
    const auto emit = [&](const BusyInterval &interval) {
        intervals.push_back(interval);
    };
    builder.accept({TraceKind::StartStamp, 99, 0, 0, 2, 0, 14}, emit);
    builder.accept({TraceKind::State, 99, 2, 0, 2, 0, 15}, emit);
    builder.accept({TraceKind::State, 99, 4, 0, 2, 1, 15}, emit);
    builder.accept({TraceKind::MarkStamp, 99, 9, 0, 2, 7, 0}, emit);
    const auto &stats = builder.finish(emit);
    ASSERT_EQ(intervals.size(), 1u);
    EXPECT_EQ(intervals[0].start_cycles, 4u);
    EXPECT_EQ(intervals[0].end_cycles, 9u);
    EXPECT_EQ(intervals[0].clip_flags, 2u);
    EXPECT_EQ(stats.clipped_end, 1u);
}

TEST(L0TraceDecoder, EndMarkerRequiresANewStartAndFlagsUnexpectedFiller) {
    std::vector<uint8_t> bytes;
    for (uint32_t word :
         {0xE0000001u, 0xE0000002u, 0xE0000003u, 0xE0000004u, 0xF0010005u, 0xEFFFFFFFu, 0xDDDDDDDDu, 0xF0010006u})
        append_word(bytes, word);
    TraceDecoder decoder;
    std::vector<TraceEvent> events;
    decoder.feed(bytes.data(), bytes.size(), [&](const TraceEvent &event) {
        events.push_back(event);
    });
    const auto &stats = decoder.finish();
    EXPECT_EQ(events.size(), 3u);
    EXPECT_EQ(stats.end_flags, 1u);
    EXPECT_EQ(stats.fillers, 1u);
    EXPECT_EQ(stats.unexpected_fillers, 0u);
    EXPECT_EQ(stats.words_without_start, 1u);
    EXPECT_FALSE(stats.structurally_complete());
}

TEST(L0TraceDecoder, MalformedAndTruncatedGroupsFailCompleteness) {
    std::vector<uint8_t> bytes;
    append_word(bytes, 0x00010002u);
    append_word(bytes, 0xE0000003u);
    append_word(bytes, 0xF0000004u);
    append_word(bytes, 0xE0000005u);
    TraceDecoder decoder;
    std::vector<TraceEvent> events;
    decoder.feed(bytes.data(), bytes.size(), [&](const TraceEvent &event) {
        events.push_back(event);
    });
    const auto &stats = decoder.finish();
    EXPECT_TRUE(events.empty());
    EXPECT_EQ(stats.words_without_start, 2u);
    EXPECT_EQ(stats.malformed_start_groups, 1u);
    EXPECT_EQ(stats.incomplete_start_groups, 1u);
    EXPECT_FALSE(stats.structurally_complete());
}

TEST(L0RingRange, DetectsOverwriteAndSplitsWrappedCopy) {
    using simpler::dfx::l0::plan_ring_range;
    const auto first = plan_ring_range(0, 0, 256, 256, 1024);
    EXPECT_TRUE(first.trustworthy());
    EXPECT_EQ(first.start, 0u);
    EXPECT_EQ(first.first_bytes, 256u);
    EXPECT_EQ(first.second_bytes, 0u);

    const auto nonzero_start = plan_ring_range(128, 100, 356, 384, 1024);
    EXPECT_TRUE(nonzero_start.trustworthy());
    EXPECT_EQ(nonzero_start.start, 128u);
    EXPECT_EQ(nonzero_start.first_bytes, 256u);
    EXPECT_EQ(nonzero_start.second_bytes, 0u);

    const auto full = plan_ring_range(128, 0, 1024, 128, 1024);
    EXPECT_TRUE(full.trustworthy());
    EXPECT_EQ(full.bytes, 1024u);
    EXPECT_EQ(full.start, 128u);

    const auto overwritten = plan_ring_range(128, 0, 1280, 384, 1024);
    EXPECT_FALSE(overwritten.trustworthy());
    EXPECT_EQ(overwritten.lost_bytes, 256u);
    EXPECT_EQ(overwritten.start, 384u);
    EXPECT_EQ(overwritten.first_bytes, 640u);
    EXPECT_EQ(overwritten.second_bytes, 384u);

    const auto starts_near_end = plan_ring_range(900, 100, 400, 176, 1024);
    EXPECT_TRUE(starts_near_end.trustworthy());
    EXPECT_EQ(starts_near_end.start, 900u);
    EXPECT_EQ(starts_near_end.first_bytes, 124u);
    EXPECT_EQ(starts_near_end.second_bytes, 176u);

    const auto short_tail = plan_ring_range(900, 0, 200, 100, 1024);
    EXPECT_TRUE(short_tail.trustworthy());
    EXPECT_TRUE(short_tail.wrapped);
    EXPECT_EQ(short_tail.tail_unused_bytes, 24u);
    EXPECT_EQ(short_tail.start, 900u);
    EXPECT_EQ(short_tail.first_bytes, 100u);
    EXPECT_EQ(short_tail.second_bytes, 100u);

    // The counter can advance 32 bytes farther than the programmed ring length
    // when a record reaches the guard. Those bytes are still part of GM.
    const auto guarded_tail = plan_ring_range(900, 0, 160, 4, 1024, 1088);
    EXPECT_TRUE(guarded_tail.trustworthy());
    EXPECT_EQ(guarded_tail.first_bytes, 156u);
    EXPECT_EQ(guarded_tail.second_bytes, 4u);
    EXPECT_EQ(guarded_tail.bytes, 160u);

    const auto early_overwrite = plan_ring_range(900, 0, 970, 920, 1024);
    EXPECT_FALSE(early_overwrite.trustworthy());
    EXPECT_EQ(early_overwrite.tail_unused_bytes, 74u);
    EXPECT_EQ(early_overwrite.lost_bytes, 20u);
    EXPECT_EQ(early_overwrite.start, 920u);
    EXPECT_EQ(early_overwrite.first_bytes, 30u);
    EXPECT_EQ(early_overwrite.second_bytes, 920u);

    EXPECT_TRUE(plan_ring_range(128, 0, 256, 300, 1024).pointer_mismatch);
    EXPECT_TRUE(plan_ring_range(0, 0, 2500, 452, 1024).pointer_mismatch);
    EXPECT_TRUE(plan_ring_range(0, UINT32_MAX - 3, 4, 8, 1024).counter_wrapped);
    EXPECT_TRUE(plan_ring_range(0, 0, 256, 1025, 1024).invalid_pointer);
    EXPECT_TRUE(plan_ring_range(1025, 0, 256, 256, 1024).invalid_pointer);
}

TEST(L0RingRange, DerivesWptrOnlyBeforeTheWrapWindow) {
    using simpler::dfx::l0::derive_linear_wptr;
    constexpr uint32_t ring_len = 1u << 20;
    constexpr uint32_t ring_capacity = ring_len + L0_DFX_TAIL_HEADROOM_BYTES;
    constexpr uint32_t anchor = ring_len - L0_DFX_TAIL_HEADROOM_BYTES - 100;
    uint32_t wptr = 0;
    EXPECT_TRUE(derive_linear_wptr(anchor, 0, 50, ring_len, ring_capacity, &wptr));
    EXPECT_EQ(wptr, anchor + 50);
    EXPECT_FALSE(derive_linear_wptr(anchor, 0, 100, ring_len, ring_capacity, &wptr));
    EXPECT_FALSE(derive_linear_wptr(anchor, 0, 200, ring_len, ring_capacity, &wptr));

    EXPECT_TRUE(derive_linear_wptr(0, UINT32_MAX - 15, 16, ring_len, ring_capacity, &wptr));
    EXPECT_EQ(wptr, 32u);
    EXPECT_TRUE(derive_linear_wptr(ring_len + 32, 100, 100, ring_len, ring_capacity, &wptr));
    EXPECT_EQ(wptr, ring_len + 32);
    EXPECT_FALSE(derive_linear_wptr(0, 0, 1, ring_len, ring_len, &wptr));
}
