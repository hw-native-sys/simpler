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
#include <vector>
#include "backend/sdma/sdma_submission.h"

namespace {
using namespace pto2::detail;
struct SubmissionData {
    struct alignas(128) Region {
        SchedulerAsyncLayout layout{};
        AsyncSdmaGuardContext header{};
        AsyncSdmaSubmissionGuard guards[ASYNC_SDMA_CHANNELS]{};
    } region;
    SchedulerAsyncGroupWriter writer{};
    AsyncExecutionContext execution{};
    DeferredCompletionEntry entries[64]{};
    alignas(64) uint8_t workspace[4096]{};
    std::vector<uint32_t> order;
    uint32_t builds{0}, submits{0}, waits{0};
    uint32_t queues{0}, group{0};
    uint32_t actual_count{1};
    uint32_t fault{0};
    bool records{true};
    bool deferred{true};
    uint32_t queue_count{1};
    SubmissionData() {
        region.layout.total_bytes = sizeof(region);
        region.layout.run_epoch = 11;
        region.layout.task_count = 1;
        region.layout.sdma_guard_count = ASYNC_SDMA_CHANNELS;
        region.layout.sdma_guard_context_offset = offsetof(Region, header);
        region.header.run_epoch = 11;
        region.header.guard_count = ASYNC_SDMA_CHANNELS;
        region.header.bytes = (ASYNC_SDMA_CHANNELS + 1) * 128;
        execution.region_address = reinterpret_cast<uint64_t>(&region);
        execution.writer_state = reinterpret_cast<uint64_t>(&writer.reservation);
        execution.subtask_slot = 1;
    }
    uint64_t workspace_address() { return reinterpret_cast<uint64_t>(workspace); }
    AsyncCtx ctx() {
        AsyncCtx value{};
        value.task_token = deferred ? TaskId::make_global(0) : TaskId::invalid();
        value.execution_context = &execution;
        value.completion_error_code = deferred ? &writer.error_code : &execution.error_code;
        if (deferred) {
            value.completion_count = &writer.count;
            value.completion_capacity = 64;
            value.completion_entries = entries;
            value.completion_reservation = &writer.reservation;
        }
        return value;
    }
    bool preflight() { return fault != 1; }
    bool build_session(uint32_t q, uint32_t g) {
        ++builds;
        queues = q;
        group = g;
        for (uint32_t i = 0; i < queues; ++i)
            EXPECT_EQ(region.guards[group * queues + i].word, 1u);
        return fault != 2;
    }
    bool session_valid(uint32_t q, uint32_t g) { return fault != 3 && q == queues && g == group; }
    bool submit() {
        ++submits;
        return fault != 4;
    }
    bool event_valid(uint64_t) { return fault != 5; }
    bool has_records() { return records; }
    uint32_t record_count() { return actual_count; }
    bool record(uint32_t index, SdmaCompletionReceiptRecord *out) {
        if (fault == 6) return false;
        out->address = workspace_address() + (fault == 7 ? 0 : index * 64);
        if (fault == 8) ++out->address;
        if (fault == 9) out->address = 0;
        out->expected = fault == 10 ? 0 : UINT64_C(0x100000001);
        return true;
    }
    bool wait() {
        ++waits;
        order.push_back(2);
        for (uint32_t i = 0; i < queues; ++i)
            EXPECT_EQ(region.guards[group * queues + i].word, 0u);
        return fault != 11;
    }
};
struct Cache {
    void flush_range(volatile void *, uint32_t) {}
    void finish() {}
};
struct Metadata {
    SubmissionData *data;
    void publish(uint8_t *) {
        data->order.push_back(1);
        for (uint32_t i = 0; i < data->queues; ++i)
            EXPECT_EQ(data->region.guards[data->group * data->queues + i].word, 1u);
    }
};
struct Sink {
    SubmissionData *data;
    std::vector<SdmaCompletionReceiptRecord> values;
    void operator()(uint32_t index, uint32_t count, const SdmaCompletionReceiptRecord &record) {
        data->order.push_back(3);
        EXPECT_EQ(values.size(), index);
        EXPECT_GT(count, index);
        for (uint32_t i = 0; i < data->queues; ++i)
            EXPECT_EQ(data->region.guards[data->group * data->queues + i].word, 0u);
        values.push_back(record);
    }
};
bool submit(SubmissionData &data, AsyncCtx &ctx, Sink &sink, uint64_t bytes = 64, SdmaOp operation = SdmaOp::TGET) {
    Cache cache;
    return sdma_submit(
        ctx, operation, true, data.workspace_address(), 0x20000, 0x30000, bytes, 47, data, cache, sink, Metadata{&data},
        data.queue_count
    );
}

TEST(HbgSdmaSubmission, SingleAndMultipleQueuesUseOneRegistrationAndPublicationProtocol) {
    for (uint32_t queues : {1u, 2u}) {
        SubmissionData data;
        data.queue_count = queues;
        data.actual_count = queues;
        auto ctx = data.ctx();
        const CompletionToken prefix{0x8000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
        ASSERT_TRUE(async_completion_append(ctx, prefix));
        Sink sink{&data, {}};
        ASSERT_TRUE(submit(data, ctx, sink));
        EXPECT_EQ(data.writer.count, data.actual_count + 1);
        EXPECT_EQ(data.entries[0].addr, 0x8000u);
        EXPECT_EQ(data.entries[1].backend_cookie, UINT64_C(0x100000001));
        EXPECT_EQ(data.writer.reservation.limit, 0u);
        EXPECT_EQ(data.waits, 0u);
        ASSERT_EQ(sink.values.size(), data.actual_count);
        EXPECT_EQ(data.order.front(), 1u);
        EXPECT_EQ(data.order.back(), 3u);
        EXPECT_EQ(data.group * data.queues, queues == 2 ? 46u : 47u);
    }
}

TEST(HbgSdmaSubmission, CapacityAndPreflightFailuresPreventSdkSideEffects) {
    for (uint32_t fault = 0; fault < 5; ++fault) {
        SubmissionData data;
        data.queue_count = 2;
        auto ctx = data.ctx();
        if (fault == 0) ctx.completion_capacity = 1;
        if (fault == 1) data.fault = 1;
        if (fault == 2) data.queue_count = 0;
        if (fault == 4) data.queue_count = ASYNC_SDMA_CHANNELS + 1;
        if (fault == 3) data.execution.subtask_slot = 0;
        Sink sink{&data, {}};
        EXPECT_FALSE(submit(data, ctx, sink));
        EXPECT_EQ(data.builds, 0u);
        EXPECT_EQ(data.submits, 0u);
        EXPECT_TRUE(sink.values.empty());
        EXPECT_NE(data.writer.error_code, 0);
    }
}

TEST(HbgSdmaSubmission, InvalidWriterCannotReachSdkForEmptyOrNonemptyTransfers) {
    for (uint64_t bytes : {0u, 64u}) {
        for (uint32_t fault = 0; fault < 4; ++fault) {
            SubmissionData data;
            auto ctx = data.ctx();
            if (fault == 0) ctx.completion_count = nullptr;
            if (fault == 1) ctx.completion_capacity = 0;
            if (fault == 2) data.writer.reservation.sealed = 1;
            if (fault == 3) data.writer.error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
            Sink sink{&data, {}};
            EXPECT_FALSE(submit(data, ctx, sink, bytes));
            EXPECT_EQ(data.builds, 0u);
            EXPECT_EQ(data.submits, 0u);
            EXPECT_EQ(data.writer.count, 0u);
            EXPECT_TRUE(sink.values.empty());
            if (fault == 3) EXPECT_EQ(data.writer.error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
        }
    }
}

TEST(HbgSdmaSubmission, SdkOrRecordFailuresPoisonHeldQueuesAndKeepTheFirstError) {
    for (uint32_t fault = 2; fault <= 10; ++fault) {
        SubmissionData data;
        data.queue_count = 2;
        data.actual_count = 2;
        data.fault = fault;
        auto ctx = data.ctx();
        Sink sink{&data, {}};
        EXPECT_FALSE(submit(data, ctx, sink)) << fault;
        EXPECT_TRUE(sink.values.empty());
        EXPECT_NE(data.writer.error_code, 0);
        EXPECT_EQ(data.region.guards[46].word, 2u);
        EXPECT_EQ(data.region.guards[47].word, 2u);
        const auto attempts = data.builds;
        data.fault = 0;
        EXPECT_FALSE(submit(data, ctx, sink));
        EXPECT_EQ(data.builds, attempts);
    }
}

TEST(HbgSdmaSubmission, InvalidRecordCountsNeverCompleteTheReservation) {
    for (uint32_t count : {0u, 3u, UINT32_MAX}) {
        SubmissionData data;
        data.queue_count = 2;
        data.actual_count = count;
        auto ctx = data.ctx();
        Sink sink{&data, {}};
        EXPECT_FALSE(submit(data, ctx, sink));
        EXPECT_EQ(data.writer.count, 0u);
        EXPECT_NE(data.writer.reservation.limit, 0u);
    }
}

TEST(HbgSdmaSubmission, EmptyTransferHasNoAttemptAndSynchronousFallbackHasNoCondition) {
    SubmissionData data;
    auto ctx = data.ctx();
    Sink sink{&data, {}};
    ASSERT_TRUE(submit(data, ctx, sink, 0));
    EXPECT_EQ(data.builds, 0u);
    EXPECT_EQ(data.writer.reservation.serial, 0u);
    EXPECT_TRUE(sink.values.empty());
    data.records = false;
    ASSERT_TRUE(submit(data, ctx, sink, 64, SdmaOp::TPUT));
    EXPECT_EQ(data.writer.count, 0u);
    EXPECT_EQ(data.writer.reservation.limit, 0u);
    EXPECT_TRUE(sink.values.empty());
}

TEST(HbgSdmaSubmission, SynchronousEntryPublishesAndUnlocksBeforeWaitAndReceipt) {
    SubmissionData data;
    data.queue_count = 2;
    data.deferred = false;
    auto ctx = data.ctx();
    Sink sink{&data, {}};
    ASSERT_TRUE(submit(data, ctx, sink));
    EXPECT_EQ(data.waits, 1u);
    EXPECT_EQ(data.order, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(data.writer.count, 0u);
    data.fault = 11;
    EXPECT_FALSE(submit(data, ctx, sink));
    EXPECT_EQ(data.execution.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(data.region.guards[46].word, 0u);
}

TEST(HbgSdmaSubmission, ScratchBoundaryChecksPrecedeSessionConstruction) {
    EXPECT_TRUE(sdma_scratch_valid(128, 128, 128, 7));
    // TASSIGN(scratch, 0) names the beginning of UB, not a null GM buffer.
    EXPECT_TRUE(sdma_scratch_valid(0, 128, 128, 0));
    EXPECT_FALSE(sdma_scratch_valid(129, 128, 128, 0));
    EXPECT_FALSE(sdma_scratch_valid(128, UINT64_C(1) << 32, 128, 0));
    EXPECT_FALSE(sdma_scratch_valid(128, 128, 128, 8));
}
}  // namespace
