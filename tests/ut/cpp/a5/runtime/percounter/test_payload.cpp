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

#include "inner_kernel.h"
#include "runtime_types.h"

uint64_t get_aicore_report_epoch();
#include "lane_loop.h"
#include "calendar_cache.h"
#include "calendar_plan.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using namespace simpler::percounter;

alignas(64) std::array<uint8_t, 4096> registers{};
alignas(64) std::array<uint8_t, PLATFORM_SSBUF_SIZE> startup_ssbuf{};

TaskInput task(uint32_t mask, int blocks = 1, bool sync = false) {
    TaskInput result{};
    result.active_mask = mask;
    result.is_mix = __builtin_popcount(mask) > 1;
    result.core_type_index = mask == 1 ? 0 : 1;
    result.subtask_count = __builtin_popcount(mask);
    result.block_num = blocks;
    result.sync_start = sync;
    return result;
}

struct CalendarDeferTest : testing::Test {
    static constexpr uint32_t defer_bit = 1U << 24;
    alignas(32) uint32_t done{0};
    alignas(32) uint32_t shell_done{0};
    uint32_t signal{0};
    CalendarTiming sample{100, 110, 0, 0};
    PercounterControl control{};
    PercounterCounter counter{};
    pto2::resident::AsyncGroupData scratch{};
    std::array<DeferredCompletionEntry, 1> entries{};
    std::array<PercounterAsyncCompletion, 2> groups{};
    LaneConfig producer{};
    CalendarExecutionExtra extra{};
    CalendarWait wait{};

    CalendarDeferTest() {
        const uint64_t set = reinterpret_cast<uint64_t>(&done);
        counter.completion_notify_address = set;
        counter.calendar_defer_bit = defer_bit;
        extra.shell_add_cmd = percounter_notify_add_address(reinterpret_cast<uint64_t>(&shell_done));
        wait = {set | CALENDAR_WAIT_DEFER, (uint64_t{24} << 24) | 1};
        producer.control = &control;
        producer.counters = &counter;
        producer.task_count = 1;
        producer.async_group = &scratch;
        producer.async_entries = entries.data();
        producer.async_completions = groups.data();
        producer.async_completion_capacity = groups.size();
        entries[0].addr = reinterpret_cast<uint64_t>(&signal);
        entries[0].expected_value = 1;
        entries[0].engine = COMPLETION_ENGINE_SDMA;
        entries[0].completion_type = COMPLETION_TYPE_COUNTER;
    }

    void seal(CalendarTiming &timing) {
        scratch.count = 1;
        bool deferred = false;
        ASSERT_TRUE(percounter_calendar_seal(producer, &extra, true, &timing, 0, &deferred));
        ASSERT_TRUE(deferred);
    }
};

TEST_F(CalendarDeferTest, RegistrationSetsFlagWithoutReleasingTheTaskOrGraph) {
    seal(sample);
    EXPECT_EQ(done, defer_bit);
    EXPECT_EQ(shell_done, 0U);
    EXPECT_EQ(control.async_registered, 1U);
    EXPECT_NE(sample.defer_register, 0U);
    EXPECT_EQ(sample.defer_completion, 0U);

    LaneConfig successor = producer;
    successor.cluster_lane = 1;
    uint64_t cursor = 0;
    ASSERT_TRUE(percounter_calendar_service_defer(successor, 0, &cursor));
    EXPECT_EQ(done, defer_bit);
    EXPECT_EQ(shell_done, 0U);

    signal = 1;
    ASSERT_TRUE(percounter_calendar_wait_list(successor, &wait, 1));
    EXPECT_EQ(done, defer_bit | 1);
    EXPECT_EQ(shell_done, 1U);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_GE(sample.defer_completion, sample.defer_register);
}

TEST_F(CalendarDeferTest, TwoSuccessorsSettleEveryDeferredGroupOnce) {
    CalendarTiming second_sample{100, 110, 0, 0};
    seal(sample);
    seal(second_sample);
    wait.target = (uint64_t{24} << 24) | 2;
    signal = 1;
    auto consume = [&](uint32_t lane) {
        LaneConfig successor = producer;
        successor.cluster_lane = lane;
        EXPECT_TRUE(percounter_calendar_wait_list(successor, &wait, 1));
    };
    std::thread first(consume, 1);
    std::thread second(consume, 2);
    first.join();
    second.join();
    EXPECT_EQ(done, defer_bit | 2);
    EXPECT_EQ(shell_done, 2U);
    EXPECT_EQ(control.async_registered, 2U);
    EXPECT_EQ(control.async_settled, 2U);
    ASSERT_GT(sample.defer_register, 0U);
    ASSERT_GT(second_sample.defer_register, 0U);
    EXPECT_GE(sample.defer_completion, sample.defer_register);
    EXPECT_GE(second_sample.defer_completion, second_sample.defer_register);
}

TEST(PercounterCalendar, SeparatesCoreOccupancyFromDeferredTail) {
    const auto ordinary = calendar_sample_cost({100, 104, 0, 0});
    ASSERT_TRUE(ordinary.has_value());
    EXPECT_EQ(ordinary->occupancy_cycles, 4U);
    EXPECT_EQ(ordinary->defer_tail_cycles, 0U);

    const auto deferred = calendar_sample_cost({100, 104, 108, 119});
    ASSERT_TRUE(deferred.has_value());
    EXPECT_EQ(deferred->occupancy_cycles, 8U);
    EXPECT_EQ(deferred->defer_tail_cycles, 11U);
    EXPECT_FALSE(calendar_sample_cost({100, 104, 108, 0}).has_value());
    EXPECT_FALSE(calendar_sample_cost({100, 104, 108, 105}).has_value());
}

TEST(PercounterCalendar, SyncMixReservesDistinctClustersAndWaitsForWholePredecessor) {
    std::vector<TaskInput> inputs{task(1), task(5, 2, true), task(2)};
    const int32_t first[] = {0};
    const int32_t second[] = {1};
    inputs[1].fanin_count = inputs[2].fanin_count = 1;
    inputs[1].fanin_ids = first;
    inputs[2].fanin_ids = second;

    CalendarPlan plan;
    const std::vector<CalendarRowCost> durations{{5, 0}, {2, 0}, {3, 0}, {4, 0}, {7, 0}, {2, 0}};
    ASSERT_TRUE(build_static_calendar(inputs, 2, durations, &plan));

    CalendarHeader header{};
    std::memcpy(&header, plan.bytes.data(), sizeof(header));
    ASSERT_EQ(header.worker_count, 6);
    ASSERT_EQ(header.row_count, 6);
    EXPECT_EQ(header.join_count, 2);
    std::array<uint32_t, 7> indices{};
    std::array<CalendarRow, 6> rows{};
    ASSERT_LE(header.index_offset + sizeof(indices), plan.bytes.size());
    ASSERT_LE(header.row_offset + sizeof(rows), plan.bytes.size());
    std::memcpy(indices.data(), plan.bytes.data() + header.index_offset, sizeof(indices));
    std::memcpy(rows.data(), plan.bytes.data() + header.row_offset, sizeof(rows));
    ASSERT_TRUE(std::is_sorted(indices.begin(), indices.end()));
    ASSERT_EQ(indices.back(), rows.size());
    std::array<int32_t, 2> cubes{-1, -1}, vectors{-1, -1};
    for (uint32_t worker = 0; worker < header.worker_count; ++worker) {
        for (uint32_t i = indices[worker]; i < indices[worker + 1]; ++i) {
            const auto entry = rows[i].entry;
            if (percounter_order_task_id(entry) != 1) continue;
            const auto block = percounter_order_block_idx(entry);
            ASSERT_GE(block, 0);
            ASSERT_LT(block, 2);
            if (worker < 2) {
                EXPECT_EQ(cubes[block], -1);
                cubes[block] = static_cast<int32_t>(worker);
            } else {
                EXPECT_EQ((worker - 2) % 2, 1);  // This MIX uses V1.
                EXPECT_EQ(vectors[block], -1);
                vectors[block] = static_cast<int32_t>((worker - 2) / 2);
            }
        }
    }
    EXPECT_GE(cubes[0], 0);
    EXPECT_GE(cubes[1], 0);
    EXPECT_NE(cubes[0], cubes[1]);
    EXPECT_EQ(cubes, vectors);

    uint64_t replay = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 2, plan, durations, &replay));
    EXPECT_EQ(replay, 14);  // Predecessor 5, slowest co-starting role 7, successor 2.
}

TEST(PercounterCalendar, RejectsSyncMixWiderThanAvailableClusters) {
    CalendarPlan plan;
    EXPECT_FALSE(build_static_calendar({task(5, 2, true)}, 1, {}, &plan));
}

TEST(PercounterCalendar, MeasuredPriorityAdvancesCubeThatUnblocksVector) {
    std::vector<TaskInput> inputs{task(1), task(1), task(2)};
    const int32_t predecessor[] = {1};
    inputs[2].fanin_count = 1;
    inputs[2].fanin_ids = predecessor;

    CalendarPlan cold;
    const std::vector<CalendarRowCost> durations{{5, 0}, {7, 0}, {10, 0}};
    ASSERT_TRUE(build_static_calendar(inputs, 1, {}, &cold));
    uint64_t cold_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, cold, durations, &cold_cycles));
    EXPECT_EQ(cold_cycles, 22);  // Cold Cube order runs both tasks before Vector can start.

    CalendarPlan measured;
    ASSERT_TRUE(build_static_calendar(inputs, 1, durations, &measured));
    uint64_t measured_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, measured, durations, &measured_cycles));
    EXPECT_EQ(measured_cycles, 17);  // Vector overlaps the shorter Cube task.
    EXPECT_EQ(measured.logical_row_by_packed.front(), 1);
    EXPECT_EQ(measured.predicted_cycles, measured_cycles);
}

}  // namespace

volatile uint8_t *sim_get_reg_base() { return registers.data(); }
uint32_t sim_get_physical_core_id() { return 0; }
uint64_t platform_ssbuf_hardware_base_address() { return reinterpret_cast<uint64_t>(startup_ssbuf.data()); }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }

TEST(PercounterScalarStorage, PreservesMixedIntervalsWithoutLogicalCount) {
    std::array<uint64_t, 8> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 100, 117);
    trace.write({200, uint64_t{UINT32_MAX} + 300, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 7});
    trace.append_memory(ProfilePhase::DsbAll, 500, 500);

    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 100);
    EXPECT_EQ(record.end_time, 117);
    EXPECT_EQ(record.task_ref, -1);
    EXPECT_EQ(record.count, 1);
    EXPECT_EQ(record.kind, static_cast<uint32_t>(ProfilePhase::SsbufLoad) | (3U << 16));
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 200);
    EXPECT_EQ(record.end_time, uint64_t{UINT32_MAX} + 300);
    EXPECT_EQ(record.task_ref, 42);
    EXPECT_EQ(record.count, 7);
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 500);
    EXPECT_EQ(record.end_time, 500);
    EXPECT_EQ(cursor, trace.storage_words);
    EXPECT_EQ(trace.dropped, 0);
}

TEST(PercounterScalarStorage, RejectsWholeExtendedRecordAtCapacityBoundary) {
    constexpr uint32_t capacity = PERCOUNTER_SCALAR_WORD_CAPACITY;
    std::vector<uint64_t> slots(capacity + 1);
    slots.back() = 0xfeed;
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.storage_words = capacity - 1;
    trace.write({100, 200, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 1});
    EXPECT_EQ(trace.storage_words, capacity - 1);
    EXPECT_EQ(trace.dropped, 1);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 0);
    trace.append_memory(ProfilePhase::SsbufLoad, 300, 400);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    trace.append_memory(ProfilePhase::SsbufLoad, 500, 600);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(trace.dropped, 2);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    EXPECT_EQ(slots.back(), 0xfeed);
}

TEST(PercounterScalarStorage, InvalidIntervalsNeverConsumeStorage) {
    std::array<uint64_t, 4> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.storage_words, 0);
    EXPECT_EQ(trace.dropped, 1);
    trace.dropped = UINT32_MAX;
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.dropped, UINT32_MAX);
}

TEST(PercounterScalarStorage, CompactWordsPreserveTicksAndUseFullWidthForLongIntervals) {
    std::array<uint64_t, 24> words{};
    PercounterScalarTrace trace{};
    trace.records = words.data();
    trace.window_start = 1000;
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 1007, 1012);
    EXPECT_EQ(words[0], UINT64_C(0x031b000500000007));
    trace.append_memory(ProfilePhase::SsbufLoad, 2000, 2000 + uint64_t{UINT16_MAX} + 1);
    trace.append_memory(ProfilePhase::SsbufLoad, uint64_t{UINT32_MAX} + 1001, uint64_t{UINT32_MAX} + 1002);
    trace.depth = 127;
    trace.append_memory(ProfilePhase::SsbufLoad, 3000, 3010);
    EXPECT_EQ(trace.storage_words, 16);
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, 1007);
    EXPECT_EQ(record.end_time, 1012);
    EXPECT_EQ(record.kind >> 16, 3);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.end_time, 2000 + uint64_t{UINT16_MAX} + 1);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, uint64_t{UINT32_MAX} + 1001);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.kind >> 16, 128);
    EXPECT_EQ(cursor, trace.storage_words);
}

TEST(PercounterScalarStorage, RejectsPartialExtendedRowsAndOverflowingAbsoluteTime) {
    const std::array<uint64_t, 4> partial{PERCOUNTER_SCALAR_EXTENDED, 1, 2, 3};
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    EXPECT_FALSE(percounter_decode_scalar_storage(partial.data(), partial.size(), cursor, 0, record));
    const uint64_t offset = 1;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&offset, 1, cursor, UINT64_MAX, record));
    const uint64_t duration = UINT64_C(1) << 32;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&duration, 1, cursor, UINT64_MAX, record));
}
