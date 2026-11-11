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

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "scheduler/scheduler_async_control.h"
#include "scheduler/scheduler_scalar_trace.h"
#include "scalar_policy_test_support.h"

namespace {
uint32_t kind(const SchedulerScalarRecord &record) { return record.kind & 0xffffU; }
std::vector<SchedulerScalarRecord> decode(const SchedulerScalarTrace &trace) {
    std::vector<SchedulerScalarRecord> rows;
    uint32_t cursor = 0;
    for (uint32_t index = 0; index < trace.committed; ++index) {
        SchedulerScalarRecord row{};
        EXPECT_TRUE(scheduler_decode_scalar_storage(trace.records, trace.storage_slots, cursor, row));
        rows.push_back(row);
    }
    EXPECT_EQ(cursor, trace.storage_slots);
    return rows;
}

}  // namespace

TEST(HbgScalarTrace, NoProgressLoopsKeepEveryFunctionInterval) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = 1;
    for (uint64_t begin : {10ULL, 50ULL}) {
        trace.append(SchedPhaseKind::ScalarAtomicClaim, begin + 2, begin + 5, -1, 2);
        trace.append(SchedPhaseKind::ScalarFillSlots, begin, begin + 20, -1, 1);
        trace.end_iteration(false);
    }
    EXPECT_EQ(trace.committed, 4U);
    trace.finish(100);
    ASSERT_EQ(buffer->scalar_committed, 8U);
    EXPECT_EQ(buffer->scalar_dropped, 0U);
    EXPECT_EQ(buffer->scalar_poll_loops, 2U);
    const auto decoded = decode(trace);
    for (uint32_t i = 0; i < 2; ++i) {
        const uint64_t begin = 10 + i * 40;
        const auto &atomic = decoded[i * 2];
        const auto &fill = decoded[i * 2 + 1];
        EXPECT_EQ(kind(atomic), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicClaim));
        EXPECT_EQ(atomic.start_time, begin + 2);
        EXPECT_EQ(atomic.end_time, begin + 5);
        EXPECT_EQ(atomic.kind >> 16, 2U);
        EXPECT_EQ(kind(fill), static_cast<uint32_t>(SchedPhaseKind::ScalarFillSlots));
        EXPECT_EQ(fill.start_time, begin);
        EXPECT_EQ(fill.end_time, begin + 20);
    }
}

TEST(HbgScalarTrace, ProgressKeepsTaskIdentityAndNestedDepth) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = 1;
    trace.append(SchedPhaseKind::ScalarAtomicClaim, 12, 15, 7, 2);
    trace.append(SchedPhaseKind::ScalarReadyClaim, 10, 20, 7, 1);
    trace.end_iteration(true);
    trace.finish(25);
    EXPECT_EQ(trace.poll_loops, 0U);
    ASSERT_GE(trace.committed, 2U);
    EXPECT_EQ(decode(trace)[0].task_ref, 7);
    EXPECT_EQ(decode(trace)[0].kind >> 16, 2U);
    EXPECT_EQ(decode(trace)[1].kind >> 16, 1U);
}

TEST(HbgScalarTrace, LargeSliceWritesEveryRecordWithoutStaging) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    for (uint32_t i = 0; i < 259; ++i)
        trace.append(SchedPhaseKind::ScalarResolve, 10 + i * 2, 11 + i * 2, i, 1);
    trace.end_iteration(true);
    trace.finish(1000);
    EXPECT_EQ(trace.dropped, 0U);
    const auto decoded = decode(trace);
    uint32_t found = 0;
    for (uint32_t i = 0; i < trace.committed; ++i)
        if (kind(decoded[i]) == static_cast<uint32_t>(SchedPhaseKind::ScalarResolve)) ++found;
    EXPECT_EQ(found, 259);
}

TEST(HbgScalarTrace, FullPartitionReportsTruncationWithoutOverwriting) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    for (uint32_t i = 0; i < SCHEDULER_SCALAR_RECORD_CAPACITY + 2; ++i)
        trace.write({i + 1, i + 2, i, static_cast<uint32_t>(SchedPhaseKind::ScalarResolve), 1});
    EXPECT_EQ(trace.committed, SCHEDULER_SCALAR_RECORD_CAPACITY);
    EXPECT_EQ(trace.dropped, 2U);
    EXPECT_EQ(decode(trace)[0].task_ref, 0);
    EXPECT_EQ(decode(trace).back().task_ref, SCHEDULER_SCALAR_RECORD_CAPACITY - 1);
}

TEST(HbgScalarTrace, InitializationPublishesHeadersWithoutTouchingRecordPayload) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout, true));
    const uint64_t initialized_bytes = layout.total_size;
    ASSERT_TRUE(scheduler_plan_scalar_storage(&layout));
    std::vector<uint8_t> storage(layout.total_size + SCHEDULER_STATE_ALIGNMENT - 1, 0xa5);
    auto *base = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_ALIGNMENT - 1) &
        ~(uintptr_t{SCHEDULER_STATE_ALIGNMENT} - 1)
    );
    ASSERT_TRUE(scheduler_init_data_from_layout(base, layout, UINT64_C(0x100000), initialized_bytes));
    auto *headers = scheduler_state_at<SchedulerActivityBuffer>(base, layout.activity_buffers_offset);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        EXPECT_EQ(headers[worker].scalar_committed, 0U);
        EXPECT_EQ(headers[worker].scalar_dropped, 0U);
        const auto *records = scheduler_state_at<SchedulerScalarRecord>(base, headers[worker].scalar_records_offset);
        EXPECT_EQ(records[0].start_time, UINT64_C(0xa5a5a5a5a5a5a5a5));
        EXPECT_EQ(records[SCHEDULER_SCALAR_RECORD_CAPACITY - 1].end_time, UINT64_C(0xa5a5a5a5a5a5a5a5));
    }
}

TEST(HbgScalarTrace, ResultTimingCallsTheOperationOnceWithAndWithoutRecording) {
    struct Profile {
        SchedulerScalarTrace scalar;
    };
    struct Local {
        Profile *profiling;
    };
    for (bool enabled : {false, true}) {
        Profile profile{};
        Local local{&profile};
        auto buffer = std::make_unique<SchedulerActivityBuffer>();
        auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
        profile.scalar.buffer = enabled ? buffer.get() : nullptr;
        profile.scalar.records = records.get();
        uint32_t calls = 0;
        const bool result = scheduler_scalar_result(&local, SchedPhaseKind::ScalarPollReadyPool, [&] {
            ++calls;
            return true;
        });
        EXPECT_TRUE(result);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(profile.scalar.committed, enabled ? 1u : 0u);
        EXPECT_EQ(profile.scalar.depth, 0u);
        if (enabled)
            EXPECT_EQ(kind(decode(profile.scalar)[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarPollReadyPool));
    }
}

TEST(HbgScalarTrace, UnboundPrimitivesPreserveOperandsAndPerformOneUpdate) {
    SchedulerScalarBinding binding;
    binding.attach(nullptr);
    int64_t signed_word = -7;
    uint64_t unsigned_word = UINT64_C(0xfedcba9876543210);
    uint32_t narrow_word = 0;
    EXPECT_EQ(scheduler_gm_query(unsigned_word), UINT64_C(0xfedcba9876543210));
    EXPECT_EQ(scheduler_gm_compare_exchange(signed_word, -7, 13), -7);
    EXPECT_EQ(signed_word, 13);
    EXPECT_EQ(scheduler_gm_compare_exchange(signed_word, -7, 99), 13);
    EXPECT_EQ(signed_word, 13);
    EXPECT_EQ(scheduler_gm_exchange(signed_word, -11), 13);
    EXPECT_EQ(signed_word, -11);
    EXPECT_EQ(scheduler_gm_fetch_add(unsigned_word, UINT64_C(3)), UINT64_C(0xfedcba9876543210));
    EXPECT_EQ(unsigned_word, UINT64_C(0xfedcba9876543213));
    scheduler_gm_fetch_add_discard(unsigned_word, UINT64_C(2));
    EXPECT_EQ(unsigned_word, UINT64_C(0xfedcba9876543215));
    scheduler_gm_store(narrow_word, UINT32_MAX);
    EXPECT_EQ(narrow_word, UINT32_MAX);
    EXPECT_EQ(scheduler_scalar_current(), nullptr);
}

TEST(HbgScalarTrace, MemoryObservationPreservesResultsAndRecordsDistinctPrimitives) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        int64_t signed_word = 7;
        uint64_t unsigned_word = 10;
        EXPECT_EQ(scheduler_gm_load(signed_word), 7);
        EXPECT_EQ(scheduler_gm_compare_exchange(signed_word, 7, 8), 7);
        EXPECT_EQ(scheduler_gm_exchange(signed_word, 9), 8);
        EXPECT_EQ(scheduler_gm_fetch_add(unsigned_word, UINT64_C(2)), 10u);
        scheduler_gm_fetch_add_discard(unsigned_word, UINT64_C(3));
        scheduler_gm_store(signed_word, 12);
        EXPECT_EQ(scheduler_gm_query(signed_word), 12);
        EXPECT_EQ(unsigned_word, 15u);
        scheduler_writeback_cache_line(&signed_word);
        scheduler_invalidate_cache_line(&signed_word);
        scheduler_cache_barrier();
        scheduler_ddr_barrier();
        EXPECT_EQ(trace.committed, 11u);
        EXPECT_EQ(trace.depth, 0u);
    }
    EXPECT_EQ(scheduler_scalar_current(), nullptr);
    trace.finish(scheduler_cycles());
    const SchedPhaseKind expected[] = {
        SchedPhaseKind::ScalarAtomicLoad,
        SchedPhaseKind::ScalarAtomicCas,
        SchedPhaseKind::ScalarAtomicExchange,
        SchedPhaseKind::ScalarAtomicFetchAdd,
        SchedPhaseKind::ScalarAtomicFetchAddIssue,
        SchedPhaseKind::ScalarAtomicStoreIssue,
        SchedPhaseKind::ScalarBypassLoad,
        SchedPhaseKind::ScalarDcciOut,
        SchedPhaseKind::ScalarDcciSingle,
        SchedPhaseKind::ScalarDsbAll,
        SchedPhaseKind::ScalarDsbDdr,
    };
    const auto decoded = decode(trace);
    for (uint32_t i = 0; i < 11; ++i) {
        EXPECT_EQ(kind(decoded[i]), static_cast<uint32_t>(expected[i]));
        EXPECT_EQ(decoded[i].count, 1u);
        EXPECT_EQ(decoded[i].kind >> 16, 1u);
    }
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgScalarTrace, NestedSemanticSpanOwnsItsPrimitiveAndBindingEndsWithItsScope) {
    struct Profile {
        SchedulerScalarTrace scalar;
    } profile;
    struct Local {
        Profile *profiling;
    } local{&profile};
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    profile.scalar.buffer = buffer.get();
    profile.scalar.records = records.get();
    profile.scalar.window_start = scheduler_cycles();
    uint64_t word = 3;
    {
        SchedulerScalarBinding binding;
        binding.attach(&profile.scalar);
        EXPECT_EQ(
            scheduler_scalar_result(
                &local, SchedPhaseKind::ScalarClaimSealConfirm,
                [&] {
                    return scheduler_gm_fetch_add(word, UINT64_C(0));
                }
            ),
            3u
        );
    }
    EXPECT_EQ(scheduler_gm_fetch_add(word, UINT64_C(1)), 3u);
    EXPECT_EQ(profile.scalar.committed, 2u);
    EXPECT_EQ(decode(profile.scalar)[0].kind >> 16, 2u);
    EXPECT_EQ(decode(profile.scalar)[1].kind >> 16, 1u);
    EXPECT_LE(decode(profile.scalar)[1].start_time, decode(profile.scalar)[0].start_time);
    EXPECT_LE(decode(profile.scalar)[0].end_time, decode(profile.scalar)[1].end_time);
}

TEST(HbgScalarTrace, FunctionThresholdKeepsLongSpansAndEveryPrimitive) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = 100;
    trace.window_start = 1;
    trace.append_function(SchedPhaseKind::ScalarOrderedPoll, 10, 109, 7, 1);
    trace.append_function(SchedPhaseKind::ScalarOrderedFill, 10, 110, 7, 1);
    trace.append(SchedPhaseKind::ScalarAtomicCas, 20, 21, 7, 3);
    trace.append(SchedPhaseKind::ScalarDcciOut, 21, 22, 7, 3);
    trace.append(SchedPhaseKind::ScalarDsbAll, 22, 23, 7, 3);
    trace.append(SchedPhaseKind::ScalarBypassLoad, 23, 24, 7, 3);
    EXPECT_EQ(trace.committed, 5u);
    EXPECT_EQ(trace.short_intervals, 1u);
    trace.finish(200);
    EXPECT_EQ(buffer->scalar_short_intervals, 1u);
    EXPECT_EQ(buffer->scalar_function_min_cycles, 100u);
    EXPECT_EQ(buffer->scalar_dropped, 0u);
    EXPECT_EQ(kind(decode(trace)[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarOrderedFill));
    EXPECT_EQ(kind(decode(trace)[1]), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicCas));
    EXPECT_EQ(kind(decode(trace)[2]), static_cast<uint32_t>(SchedPhaseKind::ScalarDcciOut));
    EXPECT_EQ(kind(decode(trace)[3]), static_cast<uint32_t>(SchedPhaseKind::ScalarDsbAll));
    EXPECT_EQ(kind(decode(trace)[4]), static_cast<uint32_t>(SchedPhaseKind::ScalarBypassLoad));
}

TEST(HbgScalarTrace, PayloadCacheLinesHaveIndividualEventsAndSeparateBarrier) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto payload = std::make_unique<DispatchPayload>();
    SchedulerScalarTrace trace;
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = UINT64_MAX;
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    scheduler_writeback_dispatch_payload(payload.get());
    scheduler_observe_dispatch_payload_control(payload.get());
    scheduler_observe_dispatch_payload_arguments(payload.get());
    scheduler_observe_dispatch_payload_barrier();
    const uint32_t lines = sizeof(DispatchPayload) / 64;
    ASSERT_EQ(trace.committed, 2 * lines + 1);
    for (uint32_t index = 0; index < trace.committed; ++index) {
        const auto expected = index < lines      ? SchedPhaseKind::ScalarDcciOut :
                              index == 2 * lines ? SchedPhaseKind::ScalarDsbAll :
                                                   SchedPhaseKind::ScalarDcciSingle;
        EXPECT_EQ(kind(decode(trace)[index]), static_cast<uint32_t>(expected));
        EXPECT_EQ(decode(trace)[index].count, 1u);
    }
    EXPECT_EQ(trace.short_intervals, 0u);
}

TEST(HbgScalarTrace, AsyncCompletionObserversRetainEveryLoadAndItsOriginalWidth) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    SchedulerScalarTrace trace;
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = UINT64_MAX;
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    alignas(64) uint32_t counters[16]{};
    counters[1] = 0x87654321;
    counters[2] = 0x12345678;
    uint64_t event_record = 0;
    const CompletionToken counter{
        reinterpret_cast<uint64_t>(&counters[1]), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    const CompletionToken event{
        reinterpret_cast<uint64_t>(&event_record), 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, 7
    };
    for (const auto &token : {counter, event}) {
        EXPECT_EQ(SchedulerAsyncCompletionObserver{}(token, 0).state, CompletionPollState::FAILED);
        EXPECT_EQ(SchedulerAsyncCompletionObserver{}(token, 1).state, CompletionPollState::PENDING);
    }
    counters[1] = 1;
    event_record = 7;
    for (const auto &token : {counter, event})
        EXPECT_EQ(SchedulerAsyncCompletionObserver{}(token, 3).state, CompletionPollState::READY);
    EXPECT_EQ(counters[2], 0x12345678u);
    ASSERT_EQ(trace.committed, 4u);
    for (uint32_t index = 0; index < trace.committed; ++index) {
        EXPECT_EQ(kind(decode(trace)[index]), static_cast<uint32_t>(SchedPhaseKind::ScalarBypassLoad));
        EXPECT_EQ(decode(trace)[index].count, 1u);
    }
    EXPECT_EQ(trace.short_intervals, 0u);
}

TEST(HbgScalarTrace, FilteringSemanticParentsDoesNotSuppressAtomicChildren) {
    struct Profile {
        SchedulerScalarTrace scalar;
    } profile;
    struct Local {
        Profile *profiling;
    } local{&profile};
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = profile.scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = UINT64_MAX;
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    uint64_t word = 3;
    EXPECT_EQ(
        scheduler_scalar_result(
            &local, SchedPhaseKind::ScalarClaimSealConfirm,
            [&] {
                return scheduler_gm_fetch_add(word, UINT64_C(0));
            }
        ),
        3u
    );
    EXPECT_EQ(trace.committed, 1u);
    EXPECT_EQ(decode(trace)[0].kind >> 16, 2u);
    EXPECT_EQ(kind(decode(trace)[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicFetchAdd));
    EXPECT_EQ(trace.short_intervals, 1u);
    EXPECT_EQ(trace.depth, 0u);
}

TEST(HbgScalarTrace, CompactStoragePreservesWideClocksTasksCountsAndDepth) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto slots = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = slots.get();
    const uint64_t start = UINT64_C(1) << 52;
    const uint32_t phase = static_cast<uint32_t>(SchedPhaseKind::ScalarBypassLoad) | UINT32_C(0xffff0000);
    const SchedulerScalarRecord inputs[] = {
        {start, start + 17, -1, phase, 1},
        {start, start + UINT32_MAX, -1, phase, 1},
        {start, start + UINT64_C(0x100000000), -1, phase, 1},
        {start, start + 2, INT64_MAX, phase, UINT32_MAX},
        {start, start + 2, INT64_MIN, phase, 0},
        {UINT64_MAX, UINT64_MAX, -1, phase, 1},
    };
    for (const auto &input : inputs)
        trace.write(input);
    EXPECT_EQ(trace.storage_slots, 9u);
    EXPECT_EQ(slots[0].time, start);
    EXPECT_EQ(slots[0].kind, phase);
    EXPECT_EQ(slots[0].value, 17u);
    EXPECT_EQ(slots[2].kind, phase | SCHEDULER_SCALAR_STORAGE_EXTENDED);
    const auto actual = decode(trace);
    ASSERT_EQ(actual.size(), 6u);
    for (uint32_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].start_time, inputs[i].start_time);
        EXPECT_EQ(actual[i].end_time, inputs[i].end_time);
        EXPECT_EQ(actual[i].task_ref, inputs[i].task_ref);
        EXPECT_EQ(actual[i].kind, inputs[i].kind);
        EXPECT_EQ(actual[i].count, inputs[i].count);
    }
}

TEST(HbgScalarTrace, CompactDecoderRejectsTruncatedAndOverflowingIntervals) {
    SchedulerScalarStorageSlot rows[] = {{1, SCHEDULER_SCALAR_STORAGE_EXTENDED | 90, 1}, {2, 0, 0}};
    SchedulerScalarRecord row{};
    uint32_t cursor = 0;
    EXPECT_FALSE(scheduler_decode_scalar_storage(rows, 1, cursor, row));
    cursor = 0;
    rows[0] = {UINT64_MAX, 90, 1};
    EXPECT_FALSE(scheduler_decode_scalar_storage(rows, 1, cursor, row));
    cursor = 0;
    rows[0] = {3, SCHEDULER_SCALAR_STORAGE_EXTENDED | 90, 1};
    EXPECT_FALSE(scheduler_decode_scalar_storage(rows, 2, cursor, row));
}

TEST(HbgScalarTrace, FinishCoversAllDirectWritesAndReportsTerminalPublication) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto slots = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = slots.get();
    trace.window_start = scheduler_cycles();
    const uint32_t input_count = 515;
    for (uint32_t i = 0; i < input_count; ++i) {
        const uint64_t start = scheduler_cycles();
        trace.append(SchedPhaseKind::ScalarBypassLoad, start, scheduler_cycles(), -1, 1);
    }
    const uint32_t stored_slots = trace.storage_slots;
    EXPECT_EQ(stored_slots, input_count);
    EXPECT_EQ(buffer->scalar_committed, 0u);
    const uint64_t finish_begin = scheduler_cycles();
    trace.finish(finish_begin);
    const auto rows = decode(trace);
    uint32_t loads = 0;
    for (const auto &row : rows) {
        if (kind(row) == static_cast<uint32_t>(SchedPhaseKind::ScalarBypassLoad)) ++loads;
    }
    EXPECT_EQ(loads, input_count);
    ASSERT_EQ(rows.size(), input_count + 4u);
    const auto &parent = rows[rows.size() - 4];
    EXPECT_EQ(kind(parent), static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinish));
    EXPECT_EQ(parent.start_time, finish_begin);
    EXPECT_EQ(parent.kind >> 16, 1u);
    const SchedPhaseKind children[] = {
        SchedPhaseKind::ScalarTraceFinishClean, SchedPhaseKind::ScalarTraceFinishBarrier
    };
    uint64_t cursor = parent.start_time;
    for (uint32_t i = 0; i < 2; ++i) {
        const auto &child = rows[rows.size() - 3 + i];
        EXPECT_EQ(kind(child), static_cast<uint32_t>(children[i]));
        EXPECT_EQ(child.kind >> 16, 2u);
        EXPECT_EQ(child.start_time, cursor);
        EXPECT_GE(child.end_time, child.start_time);
        cursor = child.end_time;
    }
    EXPECT_EQ(cursor, parent.end_time);
    EXPECT_EQ(rows[rows.size() - 3].count, (stored_slots + 3) / 4);
    EXPECT_EQ(kind(rows.back()), static_cast<uint32_t>(SchedPhaseKind::ScalarWindow));
    EXPECT_EQ(rows.back().end_time, parent.end_time);
    EXPECT_EQ(buffer->scalar_storage_format, 1u);
    EXPECT_EQ(buffer->scalar_storage_slots, trace.storage_slots);
    EXPECT_EQ(buffer->scalar_committed, trace.committed);
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgScalarTrace, LinkedTranslationUnitsRetainTheirInstrumentationPolicy) {
    for (int invocation = 0; invocation < 2; ++invocation) {
        const auto plain = scalar_policy_plain();
        EXPECT_EQ(plain.value, 37u);
        EXPECT_EQ(plain.join_word, scheduler_async_pending::REGISTERED_UNIT + 1);
        EXPECT_EQ(plain.records, 0u);
        const auto profiled = scalar_policy_profiled();
        EXPECT_EQ(profiled.value, plain.value);
        EXPECT_EQ(profiled.join_word, plain.join_word);
        EXPECT_EQ(profiled.bypass_loads, 1u);
        EXPECT_EQ(profiled.atomic_adds, 1u);
        EXPECT_EQ(profiled.ordered_polls, 1u);
        EXPECT_EQ(scheduler_scalar_current(), nullptr);
    }
}
