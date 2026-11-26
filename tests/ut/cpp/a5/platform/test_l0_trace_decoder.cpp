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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "host/l0_trace_records.h"
#include "common/core_type.h"
#include "common/platform_config.h"

static void append_word(std::vector<uint8_t> &bytes, uint32_t word) {
    for (int shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(word >> shift));
}

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

    L0TraceDecoder decoder;
    std::vector<L0TraceEvent> events;
    for (const uint8_t byte : bytes)
        decoder.feed(&byte, 1, [&](const L0TraceEvent &event) {
            events.push_back(event);
        });
    const auto &stats = decoder.finish();
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events[0].kind, L0TraceKind::StartStamp);
    EXPECT_EQ(events[0].anchor_syscnt, 0x0001000200030004ull);
    EXPECT_EQ(events[0].elapsed_cycles, 0u);
    EXPECT_EQ(events[0].block_id, 0x3412u);
    EXPECT_EQ(events[1].kind, L0TraceKind::State);
    EXPECT_EQ(events[1].anchor_syscnt, events[0].anchor_syscnt);
    EXPECT_EQ(events[1].elapsed_cycles, 5u);
    EXPECT_EQ(events[2].kind, L0TraceKind::MarkStamp);
    EXPECT_EQ(events[2].payload, 7u);
    EXPECT_EQ(events[2].elapsed_cycles, 11u);
    EXPECT_EQ(events[3].kind, L0TraceKind::EndFlag);
    EXPECT_EQ(stats.end_flags, 1u);
    EXPECT_EQ(stats.fillers, 1u);
    EXPECT_TRUE(stats.structurally_complete());
}

TEST(L0TraceDecoder, ClipsStateIntervalsAtObservedSegmentBoundaries) {
    L0BusyIntervalBuilder builder;
    std::vector<L0BusyInterval> intervals;
    const auto emit = [&](const L0BusyInterval &interval) {
        intervals.push_back(interval);
    };
    const uint64_t anchor = 1234;
    builder.accept({L0TraceKind::StartStamp, anchor, 0, 0, 7, 0, 14}, emit);
    builder.accept({L0TraceKind::State, anchor, 5, 0, 7, 0x2, 15}, emit);  // VEC is already busy.
    builder.accept({L0TraceKind::State, anchor, 8, 0, 7, 0x3, 15}, emit);  // SCALAR becomes busy.
    builder.accept({L0TraceKind::MarkStamp, anchor, 10, 0, 7, 9, 0}, emit);
    builder.accept({L0TraceKind::State, anchor, 15, 0, 7, 0x2, 15}, emit);  // SCALAR closes.
    builder.accept({L0TraceKind::EndFlag, anchor, 15, 0, 7, 0, 14}, emit);
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
    L0BusyIntervalBuilder builder;
    std::vector<L0BusyInterval> intervals;
    const auto emit = [&](const L0BusyInterval &interval) {
        intervals.push_back(interval);
    };
    builder.accept({L0TraceKind::StartStamp, 99, 0, 0, 2, 0, 14}, emit);
    builder.accept({L0TraceKind::State, 99, 2, 0, 2, 0, 15}, emit);
    builder.accept({L0TraceKind::State, 99, 4, 0, 2, 1, 15}, emit);
    builder.accept({L0TraceKind::MarkStamp, 99, 9, 0, 2, 7, 0}, emit);
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
    L0TraceDecoder decoder;
    std::vector<L0TraceEvent> events;
    decoder.feed(bytes.data(), bytes.size(), [&](const L0TraceEvent &event) {
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
    L0TraceDecoder decoder;
    std::vector<L0TraceEvent> events;
    decoder.feed(bytes.data(), bytes.size(), [&](const L0TraceEvent &event) {
        events.push_back(event);
    });
    const auto &stats = decoder.finish();
    EXPECT_TRUE(events.empty());
    EXPECT_EQ(stats.words_without_start, 2u);
    EXPECT_EQ(stats.malformed_start_groups, 1u);
    EXPECT_EQ(stats.incomplete_start_groups, 1u);
    EXPECT_FALSE(stats.structurally_complete());
}

class L0TraceExportTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto pattern = (std::filesystem::temp_directory_path() / "simpler_l0_export_ut_XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const char *created = ::mkdtemp(name.data());
        ASSERT_NE(created, nullptr);
        root = created;
        const uint32_t words[]{
            0xE0120004u, 0xE0340003u, 0xE0000002u, 0xE0000001u, 0xF0000005u,
            0xF0010003u, 0xF0000007u, 0x40070006u, 0xEFFFFFFFu,
        };
        std::ofstream raw(root / "core_84.bin", std::ios::binary);
        for (uint32_t word : words)
            raw.write(reinterpret_cast<const char *>(&word), sizeof(word));
        raw.close();
        ASSERT_TRUE(raw);
    }

    void TearDown() override {
        std::error_code ec;
        if (!root.empty()) std::filesystem::remove_all(root, ec);
    }

    std::filesystem::path root;
};

TEST_F(L0TraceExportTest, StructuredEventExportFromRawFile) {
    std::vector<uint32_t> core_types(PLATFORM_MAX_CORES, 0);
    core_types[84] = static_cast<uint32_t>(CoreType::AIV) + 1u;
    ASSERT_TRUE(export_l0_event_records(root, 42, 1500000000ull, 1200000000ull, core_types));
    const auto json_path = root / "core_84.events.json";
    std::ifstream json_file(json_path);
    ASSERT_TRUE(json_file);
    const std::string json(std::istreambuf_iterator<char>{json_file}, {});
    EXPECT_NE(json.find("\"event_count\":6"), std::string::npos);
    EXPECT_NE(json.find("\"core_frequency_hz\":1200000000"), std::string::npos);
}

TEST_F(L0TraceExportTest, ExportsTheDiscoveredRawPath) {
    std::filesystem::rename(root / "core_84.bin", root / "core_084.bin");
    std::vector<uint32_t> core_types(PLATFORM_MAX_CORES, 0);
    core_types[84] = static_cast<uint32_t>(CoreType::AIV) + 1u;
    ASSERT_TRUE(export_l0_event_records(root, 42, 1500000000ull, 1200000000ull, core_types));
    std::ifstream json_file(root / "core_84.events.json");
    ASSERT_TRUE(json_file);
    const std::string json(std::istreambuf_iterator<char>{json_file}, {});
    EXPECT_NE(json.find("\"raw_file\":\"core_084.bin\""), std::string::npos);
    EXPECT_NE(json.find("\"event_count\":6"), std::string::npos);
}

TEST_F(L0TraceExportTest, LeavesAnotherInvocationsTemporaryFileUntouched) {
    const auto foreign_path = root / "core_84.events.json.tmp";
    std::ofstream(foreign_path) << "owned by another invocation";
    std::vector<uint32_t> core_types(PLATFORM_MAX_CORES, 0);
    core_types[84] = static_cast<uint32_t>(CoreType::AIV) + 1u;
    ASSERT_TRUE(export_l0_event_records(root, 42, 1500000000ull, 1200000000ull, core_types));
    std::ifstream foreign(foreign_path);
    ASSERT_TRUE(foreign);
    const std::string content(std::istreambuf_iterator<char>{foreign}, {});
    EXPECT_EQ(content, "owned by another invocation");
    size_t files = 0;
    for (const auto &entry : std::filesystem::directory_iterator(root)) {
        EXPECT_TRUE(entry.is_regular_file());
        ++files;
    }
    EXPECT_EQ(files, 3u);
}
