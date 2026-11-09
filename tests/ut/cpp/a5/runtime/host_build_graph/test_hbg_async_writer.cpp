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
#include <set>
#include <vector>

#include "async_completion_writer.h"
#include "backend/sdma/sdma_submission_contract.h"
#include "dispatch_payload.h"

namespace {
struct Writer {
    uint32_t count{0};
    int32_t error{0};
    std::array<DeferredCompletionEntry, 64> entries{};
    uint32_t sealed{0};
    AsyncCtx context(uint32_t capacity = 64) {
        return {&count, &error, entries.data(), capacity, TaskId::make_global(3), 0, &sealed};
    }
};

CompletionToken token(uint64_t post = (UINT64_C(1) << 44) + 7) {
    return {0x10000, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD, post};
}

TEST(HbgAsyncWriter, ExistingContextAndPayloadWireLayoutsArePreserved) {
    EXPECT_EQ(sizeof(AsyncCtxStorage), 40u);
    EXPECT_EQ(sizeof(AsyncCtx), 56u);
    EXPECT_EQ(sizeof(LocalContext), 48u);
    EXPECT_EQ(sizeof(DispatchPayload), 512u);
    EXPECT_EQ(offsetof(DispatchPayload, args), 64u);
}

TEST(HbgAsyncWriter, CapacityPreflightBoundsEveryPossibleCompletionRecord) {
    Writer writer;
    auto ctx = writer.context(4);
    ASSERT_TRUE(async_completion_has_capacity(ctx, 4));
    EXPECT_EQ(writer.count, 0u);
    ASSERT_TRUE(async_completion_append(ctx, token()));
    ASSERT_TRUE(async_completion_append(ctx, token(9)));
    EXPECT_EQ(writer.count, 2u);
    EXPECT_EQ(writer.entries[0].backend_cookie, (UINT64_C(1) << 44) + 7);
    EXPECT_EQ(writer.entries[1].backend_cookie, 9u);
    ASSERT_TRUE(async_completion_has_capacity(ctx, 2));
    EXPECT_EQ(writer.error, 0);
}

TEST(HbgAsyncWriter, ExhaustionRefusesBeforeCallerCanSubmitAndPreservesOldEntries) {
    Writer writer;
    auto ctx = writer.context();
    for (uint32_t index = 0; index < 64; ++index)
        ASSERT_TRUE(async_completion_append(ctx, token(index + 1)));
    uint32_t submits = 0;
    if (async_completion_has_capacity(ctx, 1)) ++submits;
    EXPECT_EQ(submits, 0u);
    EXPECT_EQ(writer.count, 64u);
    EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(writer.entries[63].backend_cookie, 64u);
    async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_FALSE(async_completion_append(ctx, token()));
    EXPECT_EQ(writer.count, 64u);
}

TEST(HbgAsyncWriter, CapacityPreflightConsumesNoConditionAndPreservesTheFirstError) {
    Writer writer;
    auto ctx = writer.context();
    EXPECT_TRUE(async_completion_has_capacity(ctx, 48));
    EXPECT_EQ(writer.count, 0u);
    async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_FALSE(async_completion_has_capacity(ctx, 1));
    EXPECT_EQ(writer.count, 0u);
}

TEST(HbgAsyncWriter, InvalidCapacitiesAndCorruptCountsCannotAppendOrGrant) {
    for (uint32_t capacity : {0u, 65u, UINT32_MAX}) {
        Writer writer;
        auto ctx = writer.context(capacity);
        EXPECT_FALSE(async_completion_has_capacity(ctx, 1));
        EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(writer.count, 0u);
    }
    Writer writer;
    writer.count = UINT32_MAX;
    auto ctx = writer.context();
    EXPECT_FALSE(async_completion_has_capacity(ctx, 1));
    EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(writer.count, UINT32_MAX);
    auto absent_entries = writer.context();
    writer.count = 0;
    writer.error = 0;
    absent_entries.completion_entries = nullptr;
    EXPECT_FALSE(async_completion_append(absent_entries, token()));
    EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

struct RecordingCache {
    std::vector<std::pair<uintptr_t, uint32_t>> ranges;
    uint32_t fences{0};
    void flush_range(volatile void *address, uint32_t bytes) {
        ranges.emplace_back(reinterpret_cast<uintptr_t>(address), bytes);
    }
    void finish() { ++fences; }
};

TEST(HbgAsyncWriter, ResidentRegistrationKeepsDirtyWriterPrivateUntilSeal) {
    for (uint32_t count : {0u, 1u, 8u, 64u}) {
        Writer writer;
        auto ctx = writer.context();
        RecordingCache cache;
        for (uint32_t index = 0; index < count; ++index) {
            ASSERT_TRUE(async_completion_append(ctx, token(index + 1)));
            async_completion_flush(ctx, cache);
        }
        EXPECT_TRUE(cache.ranges.empty());
        EXPECT_EQ(cache.fences, 0u);
        EXPECT_EQ(writer.count, count);
        ASSERT_TRUE(async_completion_seal(ctx));
        EXPECT_EQ(writer.sealed, 1u);
    }
}

TEST(HbgAsyncWriter, MalformedTokensFailBeforeWritingAndRemainStickyAtSeal) {
    for (uint32_t fault = 0; fault < 8; ++fault) {
        Writer writer;
        auto ctx = writer.context();
        auto value = token();
        if (fault == 0) value.addr = 0;
        if (fault == 1) value.addr += 4;
        if (fault == 2) value.engine = UINT32_MAX;
        if (fault == 3) value.completion_type = -1;
        if (fault == 4) value.backend_cookie = 0;
        if (fault == 5) value.expected_value = 1;
        if (fault == 6 || fault == 7) {
            value.completion_type = COMPLETION_TYPE_COUNTER;
            value.backend_cookie = fault == 6 ? 1 : 0;
            value.expected_value = fault == 7 ? UINT32_MAX : 1;
        }
        EXPECT_FALSE(async_completion_append(ctx, value));
        EXPECT_EQ(writer.count, 0u);
        EXPECT_EQ(writer.entries[0].addr, 0u);
        EXPECT_EQ(writer.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_FALSE(async_completion_append(ctx, token()));
        EXPECT_FALSE(async_completion_seal(ctx));
    }
}

TEST(HbgAsyncWriter, MissingLegacyCountStillPublishesTheAvailableErrorField) {
    Writer writer;
    auto ctx = writer.context();
    ctx.completion_count = nullptr;
    ctx.completion_sealed = nullptr;
    EXPECT_FALSE(async_completion_has_capacity(ctx, 1));
    ASSERT_EQ(writer.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    RecordingCache cache;
    async_completion_flush(ctx, cache);
    ASSERT_EQ(cache.ranges.size(), 1u);
    EXPECT_EQ(cache.ranges.front().first, reinterpret_cast<uintptr_t>(&writer.error));
    EXPECT_EQ(cache.ranges.front().second, sizeof(writer.error));
    EXPECT_EQ(cache.fences, 1u);
}

TEST(HbgAsyncWriter, SplitPublicationNeverTouchesTheInterveningAtomicLine) {
    struct alignas(128) Image {
        uint32_t count{2};
        int32_t error{0};
        uint8_t writer_pad[120]{};
        uint64_t atomic_control[16]{};
        DeferredCompletionEntry entries[64]{};
    } image;
    AsyncCtx ctx{&image.count, &image.error, image.entries, 64, TaskId::make_global(1)};
    RecordingCache cache;
    async_completion_flush(ctx, cache);
    ASSERT_EQ(cache.ranges.size(), 3u);
    EXPECT_EQ(cache.fences, 1u);
    const uintptr_t base = reinterpret_cast<uintptr_t>(&image);
    std::set<uintptr_t> lines;
    for (const auto &[address, bytes] : cache.ranges)
        for (auto line = address & ~uintptr_t{63}; line < address + bytes; line += 64)
            lines.insert(line);
    EXPECT_EQ(lines, (std::set<uintptr_t>{base, base + 256}));
    EXPECT_FALSE(lines.count(base + 128));
    EXPECT_FALSE(lines.count(base + 192));
}

TEST(HbgAsyncWriter, LegacyContiguousSlabAndEmptyFlushPublishOnlyTheirOwnFields) {
    DeferredCompletionSlab slab{};
    AsyncCtx ctx = AsyncCtx::make(TaskId::make_global(1), &slab);
    RecordingCache empty;
    async_completion_flush(ctx, empty);
    EXPECT_EQ(empty.ranges.size(), 2u);
    EXPECT_EQ(empty.fences, 1u);
    ASSERT_TRUE(async_completion_append(ctx, token()));
    RecordingCache filled;
    async_completion_flush(ctx, filled);
    ASSERT_EQ(filled.ranges.size(), 3u);
    EXPECT_EQ(filled.ranges.back().first, reinterpret_cast<uintptr_t>(&slab.entries[0]));
    EXPECT_EQ(filled.ranges.back().second, sizeof(DeferredCompletionEntry));
    slab.count = UINT32_MAX;
    ctx.completion_capacity = UINT32_MAX;
    RecordingCache corrupt;
    async_completion_flush(ctx, corrupt);
    EXPECT_EQ(corrupt.ranges.back().second, 64 * sizeof(DeferredCompletionEntry));
}

TEST(HbgAsyncWriter, CopiesShareCountErrorAndSeal) {
    Writer writer;
    auto first = writer.context();
    auto copy = first;
    ASSERT_TRUE(async_completion_append(first, token()));
    ASSERT_TRUE(async_completion_append(copy, token(9)));
    EXPECT_EQ(writer.count, 2u);
    EXPECT_TRUE(async_completion_seal(copy));
    EXPECT_FALSE(async_completion_append(first, token(10)));
    EXPECT_FALSE(async_completion_seal(first));
    EXPECT_EQ(writer.count, 2u);
    EXPECT_EQ(writer.error, 0);
}

TEST(HbgAsyncWriter, EmptySealAndPostSealCopiesLeavePublishedStateUnchanged) {
    Writer writer;
    auto ctx = writer.context(), copy = ctx;
    ASSERT_TRUE(async_completion_seal(ctx));
    EXPECT_FALSE(async_completion_append(copy, token()));
    EXPECT_EQ(writer.error, 0);
    EXPECT_EQ(writer.count, 0u);
}

struct TensorShape {
    using RawDType = uint32_t;
    int64_t shapes[5]{1, 1, 1, 1, 8};
    int64_t strides[5]{8, 8, 8, 8, 1};
    uint64_t address{0x10000};
    int64_t GetShape(int dim) { return shapes[dim]; }
    int64_t GetStride(int dim) { return strides[dim]; }
    uint32_t *data() { return reinterpret_cast<uint32_t *>(address); }
    void count(int64_t value) {
        shapes[4] = value;
        for (int i = 0; i < 4; ++i)
            strides[i] = value;
    }
};

TEST(HbgAsyncWriter, SdmaPublicShapeValidationDistinguishesEmptyFromOverflowAndInvalidStorage) {
    TensorShape source, target;
    uint64_t bytes = 0;
    ASSERT_TRUE(async_sdma_transfer_bytes(target, source, &bytes));
    EXPECT_EQ(bytes, 32u);
    source.count(0);
    ASSERT_TRUE(async_sdma_transfer_bytes(target, source, &bytes));
    EXPECT_EQ(bytes, 0u);
    for (int64_t count : {int64_t{-1}, static_cast<int64_t>(UINT32_MAX) + 1}) {
        source.count(count);
        EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
        EXPECT_EQ(bytes, 0u);
    }
    source = {};
    target.count(7);
    EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
    target = {};
    source.strides[4] = 2;
    EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
    source = {};
    source.shapes[3] = 2;
    EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
    source = {};
    source.address = UINT64_MAX - 15;
    EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
    source.address = 0;
    EXPECT_FALSE(async_sdma_transfer_bytes(target, source, &bytes));
}

TEST(HbgAsyncWriter, FailedNonemptyTgetMustNotMasqueradeAsAnEmptySuccessfulEvent) {
    EXPECT_FALSE(async_sdma_event_valid(SdmaOp::TGET, 32, 0, false, 0, 0));
    EXPECT_TRUE(async_sdma_event_valid(SdmaOp::TGET, 0, 0, false, 0, 0));
    EXPECT_TRUE(async_sdma_event_valid(SdmaOp::TGET, 32, 17, false, 0, 0));
    EXPECT_TRUE(async_sdma_event_valid(SdmaOp::TPUT, 32, 0, false, 0, 0));
    EXPECT_FALSE(async_sdma_event_valid(SdmaOp::TPUT, 32, 0, true, 0, 0));
    EXPECT_FALSE(async_sdma_event_valid(SdmaOp::TGET, 0, 0, true, 0, 0));
    EXPECT_FALSE(async_sdma_event_valid(SdmaOp::TGET, 0, 0, false, 1, 0));
}
}  // namespace
