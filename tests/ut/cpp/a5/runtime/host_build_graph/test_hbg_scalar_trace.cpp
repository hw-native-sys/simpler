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

#include "scheduler/scheduler_scalar_policy.h"

namespace {
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = true;
};
}  // namespace

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "scheduler/scheduler_async_control.h"
#include "scheduler/scheduler_prepared_layout.h"
#include "scheduler/scheduler_scalar_trace.h"
#include "scalar_policy_test_support.h"
#include "hbg_gm_mix_test_support.h"

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
        trace.append(SchedPhaseKind::ScalarAtomicCas, begin + 2, begin + 5, -1, 2);
        trace.append(SchedPhaseKind::ScalarFillOwnSlots, begin, begin + 20, -1, 1);
        trace.end_iteration(false);
    }
    const uint32_t before_finish = trace.committed;
    trace.finish(scheduler_scalar_clock());
    ASSERT_EQ(buffer->scalar_committed, before_finish + 4);
    EXPECT_EQ(buffer->scalar_dropped, 0U);
    EXPECT_EQ(buffer->scalar_poll_loops, 2U);
    const auto decoded = decode(trace);
    uint32_t cursor = 0;
    for (uint64_t begin : {10ULL, 50ULL}) {
        ASSERT_LE(cursor + 3, before_finish);
        const auto &atomic = decoded[cursor++];
        const auto &fill = decoded[cursor++];
        EXPECT_EQ(kind(atomic), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicCas));
        EXPECT_EQ(atomic.start_time, begin + 2);
        EXPECT_EQ(atomic.end_time, begin + 5);
        EXPECT_EQ(atomic.kind >> 16, 2U);
        EXPECT_EQ(kind(fill), static_cast<uint32_t>(SchedPhaseKind::ScalarFillOwnSlots));
        EXPECT_EQ(fill.start_time, begin);
        EXPECT_EQ(fill.end_time, begin + 20);
        const auto &iteration = decoded[cursor++];
        EXPECT_EQ(kind(iteration), static_cast<uint32_t>(SchedPhaseKind::ScalarEndIteration));
        EXPECT_EQ(iteration.kind >> 16, 1U);
        if (cursor < before_finish &&
            kind(decoded[cursor]) == static_cast<uint32_t>(SchedPhaseKind::ScalarTraceAppend)) {
            const auto &submission = decoded[cursor++];
            EXPECT_GE(submission.start_time, iteration.end_time);
            EXPECT_GT(submission.end_time, submission.start_time);
            EXPECT_EQ(submission.kind >> 16, 1U);
        }
    }
    EXPECT_EQ(cursor, before_finish);
}

TEST(HbgScalarTrace, ProgressKeepsTaskIdentityAndNestedDepth) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = 1;
    trace.append(SchedPhaseKind::ScalarAtomicCas, 12, 15, 7, 2);
    trace.append(SchedPhaseKind::ScalarClusterMixPrepareMember, 10, 20, 7, 1);
    trace.end_iteration(true);
    trace.finish(25);
    EXPECT_EQ(trace.poll_loops, 0U);
    ASSERT_GE(trace.committed, 2U);
    EXPECT_EQ(decode(trace)[0].task_ref, 7);
    EXPECT_EQ(decode(trace)[0].kind >> 16, 2U);
    EXPECT_EQ(decode(trace)[1].kind >> 16, 1U);
}

TEST(HbgScalarTrace, L3RecordsOnlyClaimLoadsAndIdentifiesTheirSource) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.capture_trace_control = false;
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    uint64_t gm = 19;
    uint32_t notify = 7;
    EXPECT_EQ(scheduler_gm_query(gm), 19u);
    {
        SchedulerExecutionClaimLoads claim;
        EXPECT_EQ(scheduler_gm_query(gm), 19u);
        EXPECT_EQ(scheduler_notify_query(notify), 7u);
        scheduler_writeback_cache_line(&gm);
        scheduler_invalidate_cache_line(&gm);
        scheduler_cache_barrier();
    }
    EXPECT_EQ(scheduler_notify_query(notify), 7u);
    auto rows = decode(trace);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(kind(rows[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarGmLoad));
    EXPECT_EQ(kind(rows[1]), static_cast<uint32_t>(SchedPhaseKind::ScalarCntNotifyLoad));
    EXPECT_EQ(kind(rows[2]), static_cast<uint32_t>(SchedPhaseKind::ScalarDsbAll));
}

TEST(HbgScalarTrace, L3OmitsTraceControlAndTheirLeavesButKeepsNestedBusinessWork) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.capture_trace_control = false;
    trace.window_start = scheduler_cycles();
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    volatile uint64_t word = 0;
    {
        SchedulerScalarScope check(SchedPhaseKind::ScalarLoopControl);
        {
            SchedulerScalarScope observer(SchedPhaseKind::ScalarTraceBeforeExecution);
            EXPECT_EQ(scheduler_gm_fetch_add(word, UINT64_C(1)), 0u);
        }
        {
            SchedulerScalarScope fanin(SchedPhaseKind::ScalarExecutionTokenFanin);
            EXPECT_EQ(scheduler_gm_fetch_add(word, UINT64_C(1)), 1u);
        }
        scheduler_cluster_barrier();
    }
    trace.append_function(SchedPhaseKind::ScalarFillOwnSlots, 10, 11, -1, 1);
    trace.end_iteration(false);
    trace.finish(scheduler_cycles());
    const auto rows = decode(trace);
    ASSERT_EQ(rows.size(), 4u);
    EXPECT_EQ(kind(rows[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicFetchAdd));
    EXPECT_EQ(kind(rows[1]), static_cast<uint32_t>(SchedPhaseKind::ScalarExecutionTokenFanin));
    EXPECT_EQ(kind(rows[2]), static_cast<uint32_t>(SchedPhaseKind::ScalarFillOwnSlots));
    EXPECT_EQ(kind(rows[3]), static_cast<uint32_t>(SchedPhaseKind::ScalarWindow));
    EXPECT_EQ(buffer->scalar_poll_loops, 1u);
    EXPECT_EQ(buffer->scalar_dropped, 0u);
}

TEST(HbgScalarTrace, LargeSliceWritesEveryRecordWithoutStaging) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    for (uint32_t i = 0; i < 259; ++i)
        trace.append(SchedPhaseKind::ScalarCompletions, 10 + i * 2, 11 + i * 2, i, 1);
    trace.end_iteration(true);
    trace.finish(1000);
    EXPECT_EQ(trace.dropped, 0U);
    const auto decoded = decode(trace);
    uint32_t found = 0;
    for (uint32_t i = 0; i < trace.committed; ++i)
        if (kind(decoded[i]) == static_cast<uint32_t>(SchedPhaseKind::ScalarCompletions)) ++found;
    EXPECT_EQ(found, 259);
}

TEST(HbgScalarTrace, OptionalIterationsPreserveTheTerminalWindowAndOnlyRequiredLossReportsTruncation) {
    constexpr uint32_t detail_capacity = SCHEDULER_SCALAR_RECORD_CAPACITY - 4;
    for (const uint32_t excess : {0u, 2u}) {
        SCOPED_TRACE(excess);
        auto buffer = std::make_unique<SchedulerActivityBuffer>();
        auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
        SchedulerScalarTrace trace;
        trace.buffer = buffer.get();
        trace.records = records.get();
        trace.window_start = 1;
        const auto required = [&](uint32_t index) {
            trace.append(SchedPhaseKind::ScalarCompletions, index + 1, index + 2, index);
        };
        for (uint32_t i = 0; i < detail_capacity - 1; ++i)
            required(i);
        trace.end_iteration(false);
        ASSERT_EQ(trace.committed, detail_capacity - 1);
        required(detail_capacity - 1);
        trace.end_iteration(false);
        ASSERT_EQ(trace.committed, detail_capacity);
        for (uint32_t i = 0; i < excess; ++i)
            required(detail_capacity + i);
        trace.finish(scheduler_scalar_clock());
        EXPECT_EQ(buffer->scalar_poll_loops, 2u);
        EXPECT_EQ(buffer->scalar_committed, SCHEDULER_SCALAR_RECORD_CAPACITY);
        EXPECT_EQ(buffer->scalar_dropped, excess);
        const auto decoded = decode(trace);
        ASSERT_EQ(decoded.size(), SCHEDULER_SCALAR_RECORD_CAPACITY);
        EXPECT_EQ(decoded.front().task_ref, 0);
        EXPECT_EQ(decoded[detail_capacity - 1].task_ref, detail_capacity - 1);
        EXPECT_EQ(kind(decoded[detail_capacity]), static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinish));
        EXPECT_EQ(kind(decoded[detail_capacity + 1]), static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinishClean));
        EXPECT_EQ(kind(decoded[detail_capacity + 2]), static_cast<uint32_t>(SchedPhaseKind::ScalarTraceFinishBarrier));
        EXPECT_EQ(kind(decoded.back()), static_cast<uint32_t>(SchedPhaseKind::ScalarWindow));
    }
}

TEST(HbgScalarTrace, MixPublicationAndAccountingAreAttributedOnlyToLiveWork) {
    using namespace scheduler_test::gm_mix;
    GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = fixture.locals[1]->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    bool armed = false;
    ASSERT_TRUE(
        scheduler_claim_for_execute(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &armed)
    );
    ASSERT_TRUE(armed);
    for (uint32_t pass = 0; pass < 3; ++pass)
        for (uint32_t lane = 0; lane < 3; ++lane)
            ASSERT_TRUE(fixture.service_cooperative(lane));
    for (uint32_t lane : {0u, 2u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_TRUE(fixture.task_done(0));
    const auto before_drain = trace.committed;
    bool done = false;
    ASSERT_TRUE(fixture.drain(1, &done));
    EXPECT_TRUE(done);
    uint32_t builds = 0;
    uint32_t accounting_polls = 0;
    const auto rows = decode(trace);
    for (uint32_t i = 0; i < rows.size(); ++i) {
        if (kind(rows[i]) == static_cast<uint32_t>(SchedPhaseKind::ScalarMixNotifyMembers)) {
            EXPECT_EQ(rows[i].task_ref, 0);
            ++builds;
        }
        if (kind(rows[i]) == static_cast<uint32_t>(SchedPhaseKind::ScalarClusterMixAccount)) {
            EXPECT_LT(i, before_drain);
            EXPECT_EQ(rows[i].task_ref, 0);
            ++accounting_polls;
        }
    }
    EXPECT_EQ(builds, 1u);
    EXPECT_GT(accounting_polls, 0u);
}

TEST(HbgScalarTrace, InitializationPublishesHeadersWithoutTouchingRecordPayload) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, &layout, true));
    const uint64_t initialized_bytes = layout.total_size;
    ASSERT_TRUE(scheduler_plan_scalar_storage(&layout));
    std::vector<uint8_t> storage(layout.total_size + SCHEDULER_STATE_BASE_ALIGNMENT - 1, 0xa5);
    auto *base = reinterpret_cast<uint8_t *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_BASE_ALIGNMENT - 1) &
        ~(uintptr_t{SCHEDULER_STATE_BASE_ALIGNMENT} - 1)
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
        const bool result = scheduler_scalar_result(&local, SchedPhaseKind::ScalarAdvanceMixFloor, [&] {
            ++calls;
            return true;
        });
        EXPECT_TRUE(result);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(profile.scalar.committed, enabled ? 1u : 0u);
        EXPECT_EQ(profile.scalar.depth, 0u);
        if (enabled)
            EXPECT_EQ(kind(decode(profile.scalar)[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarAdvanceMixFloor));
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
        EXPECT_EQ(scheduler_gm_compare_exchange(signed_word, 7, 8), 7);
        EXPECT_EQ(scheduler_gm_fetch_add(unsigned_word, UINT64_C(2)), 10u);
        scheduler_gm_fetch_add_discard(unsigned_word, UINT64_C(3));
        scheduler_gm_store(signed_word, 12);
        EXPECT_EQ(scheduler_gm_query(signed_word), 12);
        EXPECT_EQ(unsigned_word, 15u);
        scheduler_writeback_cache_line(&signed_word);
        scheduler_invalidate_cache_line(&signed_word);
        scheduler_cache_barrier();
        scheduler_ddr_barrier();
        EXPECT_EQ(trace.committed, 9u);
        EXPECT_EQ(trace.depth, 0u);
    }
    EXPECT_EQ(scheduler_scalar_current(), nullptr);
    trace.finish(scheduler_cycles());
    const SchedPhaseKind expected[] = {
        SchedPhaseKind::ScalarAtomicCas,
        SchedPhaseKind::ScalarAtomicFetchAdd,
        SchedPhaseKind::ScalarAtomicFetchAddIssue,
        SchedPhaseKind::ScalarAtomicStoreIssue,
        SchedPhaseKind::ScalarGmLoad,
        SchedPhaseKind::ScalarDcciOut,
        SchedPhaseKind::ScalarDcciSingle,
        SchedPhaseKind::ScalarDsbAll,
        SchedPhaseKind::ScalarDsbDdr,
    };
    const auto decoded = decode(trace);
    for (uint32_t i = 0; i < std::size(expected); ++i) {
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
                &local, SchedPhaseKind::ScalarMixLaneOrderReady,
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

TEST(HbgScalarTrace, BoundFunctionScopeNamesItsPrimitiveWithoutAWorkerArgument) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    uint64_t word = 3;
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        {
            SchedulerScalarScope scope(SchedPhaseKind::ScalarPreparedReceive);
            scope.set_task_ref(7);
            EXPECT_EQ(scheduler_gm_fetch_add(word, UINT64_C(0)), 3u);
        }
        EXPECT_EQ(trace.depth, 0u);
    }
    const auto rows = decode(trace);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(kind(rows[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicFetchAdd));
    EXPECT_EQ(kind(rows[1]), static_cast<uint32_t>(SchedPhaseKind::ScalarPreparedReceive));
    EXPECT_EQ(rows[0].kind >> 16, 2u);
    EXPECT_EQ(rows[1].kind >> 16, 1u);
    EXPECT_EQ(rows[1].task_ref, 7);
    EXPECT_LE(rows[1].start_time, rows[0].start_time);
    EXPECT_LE(rows[0].end_time, rows[1].end_time);
    {
        SchedulerScalarScope unbound(SchedPhaseKind::ScalarPreparedReceive);
        EXPECT_EQ(scheduler_gm_fetch_add(word, UINT64_C(1)), 3u);
    }
    EXPECT_EQ(trace.committed, 2u);
}

TEST(HbgScalarTrace, FunctionThresholdKeepsLongSpansAndEveryPrimitive) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = 100;
    trace.window_start = 1;
    trace.append_function(SchedPhaseKind::ScalarMixLaneOrderReady, 10, 109, 7, 1);
    trace.append_function(SchedPhaseKind::ScalarAdvanceMixFloor, 10, 110, 7, 1);
    trace.append(SchedPhaseKind::ScalarAtomicCas, 20, 21, 7, 3);
    trace.append(SchedPhaseKind::ScalarDcciOut, 21, 22, 7, 3);
    trace.append(SchedPhaseKind::ScalarDsbAll, 22, 23, 7, 3);
    trace.append(SchedPhaseKind::ScalarBypassLoad, 23, 24, 7, 3);
    trace.append_function(SchedPhaseKind::ScalarCanEnterExecution, 110, 111, -1, 1);
    trace.append_function(SchedPhaseKind::ScalarCanEnterExecution, 115, 116, -1, 2);
    const auto retained = decode(trace);
    ASSERT_GE(retained.size(), 6u);
    if (retained.size() != 6u) {
        ASSERT_EQ(retained.size(), 7u);
        EXPECT_EQ(kind(retained[6]), static_cast<uint32_t>(SchedPhaseKind::ScalarTraceAppend));
    }
    EXPECT_EQ(trace.short_intervals, 2u);
    trace.finish(scheduler_scalar_clock());
    EXPECT_EQ(buffer->scalar_short_intervals, 2u);
    EXPECT_EQ(buffer->scalar_function_min_cycles, 100u);
    EXPECT_EQ(buffer->scalar_dropped, 0u);
    EXPECT_EQ(kind(decode(trace)[0]), static_cast<uint32_t>(SchedPhaseKind::ScalarAdvanceMixFloor));
    EXPECT_EQ(kind(decode(trace)[1]), static_cast<uint32_t>(SchedPhaseKind::ScalarAtomicCas));
    EXPECT_EQ(kind(decode(trace)[2]), static_cast<uint32_t>(SchedPhaseKind::ScalarDcciOut));
    EXPECT_EQ(kind(decode(trace)[3]), static_cast<uint32_t>(SchedPhaseKind::ScalarDsbAll));
    EXPECT_EQ(kind(decode(trace)[4]), static_cast<uint32_t>(SchedPhaseKind::ScalarBypassLoad));
    EXPECT_EQ(kind(decode(trace)[5]), static_cast<uint32_t>(SchedPhaseKind::ScalarCanEnterExecution));
}

TEST(HbgScalarTrace, PreparedCallPublicationRecordsEachWritebackAndItsBarrier) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto call = std::make_unique<SchedulerPreparedCall>();
    SchedulerScalarTrace trace;
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.function_min_cycles = UINT64_MAX;
    SchedulerScalarBinding binding;
    binding.attach(&trace);
    for (uint64_t offset = 0; offset < sizeof(SchedulerPreparedCall); offset += 64)
        scheduler_writeback_cache_line(reinterpret_cast<uint8_t *>(call.get()) + offset);
    scheduler_cache_barrier();
    const uint32_t lines = sizeof(SchedulerPreparedCall) / 64;
    ASSERT_EQ(trace.committed, lines + 1);
    for (uint32_t index = 0; index < trace.committed; ++index) {
        const auto expected = index < lines ? SchedPhaseKind::ScalarDcciOut : SchedPhaseKind::ScalarDsbAll;
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
            &local, SchedPhaseKind::ScalarMixLaneOrderReady,
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

TEST(HbgScalarTrace, CompactStoragePreservesBootstrapAndClaimPhases) {
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto slots = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    SchedulerScalarTrace trace;
    trace.buffer = buffer.get();
    trace.records = slots.get();
    const SchedPhaseKind phases[] = {
        SchedPhaseKind::ScalarBootstrapInitWait, SchedPhaseKind::ScalarBootstrapPeerWait,
        SchedPhaseKind::ScalarBootstrapTopology, SchedPhaseKind::ScalarPreparedClaim,
        SchedPhaseKind::ScalarCounterAccounting,
    };
    const char *names[] = {
        "scalar_bootstrap_init_wait", "scalar_bootstrap_peer_wait", "scalar_bootstrap_topology",
        "scalar_prepared_claim", "scalar_counter_accounting"
    };
    for (auto phase : phases)
        trace.append(phase, 10, 20, -1, 2);
    const auto rows = decode(trace);
    ASSERT_EQ(rows.size(), 5u);
    for (uint32_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(kind(rows[i]), 139u + i);
        EXPECT_STREQ(sched_phase_kind_name(static_cast<SchedPhaseKind>(kind(rows[i]))), names[i]);
        EXPECT_EQ(rows[i].start_time, 10u);
        EXPECT_EQ(rows[i].end_time, 20u);
        EXPECT_EQ(rows[i].kind >> 16, 2u);
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
        EXPECT_EQ(profiled.floor_checks, 1u);
        EXPECT_EQ(scheduler_scalar_current(), nullptr);
    }
}

TEST(SchedulerActivityBuffer, IsAllocatedOnlyWhenRequestedAndNeverWraps) {
    AicoreSchedulerLayout disabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, &disabled));
    EXPECT_EQ(disabled.activity_buffers_offset, 0u);

    AicoreSchedulerLayout enabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, &enabled, true));
    ASSERT_NE(enabled.activity_buffers_offset, 0u);
    EXPECT_EQ(
        enabled.total_size - disabled.total_size,
        static_cast<uint64_t>(SCHEDULER_WORKER_CAPACITY) * sizeof(SchedulerActivityBuffer)
    );
    scheduler_test::SchedulerStateBuffer storage(enabled);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(storage.base(), enabled.worker_contexts_offset);
    auto *buffers = scheduler_state_at<SchedulerActivityBuffer>(storage.base(), enabled.activity_buffers_offset);
    contexts[0].worker_index = SCHEDULER_WORKER_CAPACITY - 1;
    SchedulerLocalState local{};
    SchedulerLocalProfilingState profile{};
    local.profiling = &profile;
    local.profiling->activity_buffers_offset = contexts[0].activity_buffers_offset;
    local.config.self_lane = 0;
    local.config.worker_index = SCHEDULER_WORKER_CAPACITY - 1;
    local.profiling->loop_iter = 17;
    SchedulerActivityBuffer &buffer = buffers[SCHEDULER_WORKER_CAPACITY - 1];
    buffer.committed = SCHEDULER_ACTIVITY_CAPACITY - 1;

    scheduler_append_idle_activity(storage.base(), &local, 10, 20);
    scheduler_append_idle_activity(storage.base(), &local, 30, 40);

    EXPECT_EQ(buffer.committed, SCHEDULER_ACTIVITY_CAPACITY);
    EXPECT_EQ(buffer.dropped, 1u);
    const SchedulerIdleRecord &last = buffer.records[SCHEDULER_ACTIVITY_CAPACITY - 1];
    EXPECT_EQ(last.start_time, 10u);
    EXPECT_EQ(last.end_time, 20u);
    EXPECT_EQ(last.loop_iter, 17u);
}

TEST(SchedulerActivityBuffer, RejectsCorruptCommittedCountOnHost) {
    EXPECT_TRUE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY));
    EXPECT_FALSE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY + 1));
}

TEST(SchedulerProfilingLevel, EnablesOnlyTheRequestedGranularity) {
    EXPECT_FALSE(scheduler_task_timing_enabled(0));
    EXPECT_TRUE(scheduler_task_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
}
