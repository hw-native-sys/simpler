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
#include "async_execution_context.h"
#include "async_completion_writer.h"

namespace {
struct ContextData {
    DispatchPayload payload{};
    SchedulerAsyncGroupWriter writer{};
    DeferredCompletionEntry entries[4]{};
    ContextData(bool deferred = true) {
        payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
        payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
        payload.async_context_address = reinterpret_cast<uint64_t>(&payload.async_context);
        payload.async_context.region_address = 0x10000;
        payload.async_context.task_ref = 3;
        payload.async_context.subtask_slot = 1;
        auto &ctx = payload.local_context.async_ctx;
        ctx.task_token = deferred ? TaskId::make_sub_task(2, 3) : TaskId::invalid();
        ctx.completion_error_code = deferred ? &writer.error_code : &payload.async_context.error_code;
        if (deferred) {
            ctx.completion_count = &writer.count;
            ctx.completion_entries = entries;
            ctx.completion_capacity = 4;
            payload.async_context.writer_state = reinterpret_cast<uint64_t>(&writer.sealed);
        }
    }
};

TEST(HbgAsyncExecutionContext, EmbeddedContextPreservesThePublicWireLayout) {
    EXPECT_EQ(sizeof(AsyncCtxStorage), 40u);
    EXPECT_EQ(sizeof(AsyncCtx), 56u);
    EXPECT_EQ(sizeof(LocalContext), 48u);
    EXPECT_EQ(sizeof(DispatchPayload), 512u);
    EXPECT_EQ(offsetof(DispatchPayload, args), 64u);
    EXPECT_EQ(offsetof(DispatchPayload, global_context), 464u);
    EXPECT_EQ(offsetof(DispatchPayload, async_context_address), 472u);
    EXPECT_EQ(offsetof(DispatchPayload, async_context), 480u);
    EXPECT_EQ(sizeof(AsyncExecutionContext), 32u);
}

TEST(HbgAsyncExecutionContext, CopiesAndRepeatedGettersShareCountAndPreserveDirtyRegistration) {
    ContextData data;
    AsyncCtx first{}, second{};
    ASSERT_TRUE(async_execution_context_load(data.payload, &first));
    const CompletionToken token{0x8000, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    ASSERT_TRUE(async_completion_append(first, token));
    ASSERT_TRUE(async_execution_context_load(data.payload, &second));
    EXPECT_EQ(second.completion_sealed, first.completion_sealed);
    EXPECT_EQ(*second.completion_count, 1u);
    EXPECT_EQ(TaskId::to_uint64(second.task_token), TaskId::to_uint64(TaskId::make_sub_task(2, 3)));
    ASSERT_TRUE(async_completion_append(second, token));
    EXPECT_EQ(data.writer.count, 2u);
    ASSERT_TRUE(async_completion_seal(second));
    EXPECT_FALSE(async_completion_append(first, token));
    EXPECT_EQ(data.writer.count, 2u);
}

TEST(HbgAsyncExecutionContext, SynchronousExecutionsRetainIgnoredSubmissionErrorsWithoutConditionStorage) {
    ContextData data(false);
    AsyncCtx ctx{};
    ASSERT_TRUE(async_execution_context_load(data.payload, &ctx));
    EXPECT_FALSE(ctx.task_token.is_valid());
    EXPECT_EQ(ctx.completion_count, nullptr);
    EXPECT_EQ(ctx.completion_sealed, nullptr);
    async_completion_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    async_completion_error(ctx, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(data.payload.async_context.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(HbgAsyncExecutionContext, CorruptFrameCannotExposeForeignWriterPointers) {
    for (uint32_t fault = 0; fault < 6; ++fault) {
        ContextData data;
        if (fault == 0) data.payload.async_context_address += 8;
        if (fault == 1) data.payload.async_context.magic = 0;
        if (fault == 2) data.payload.async_context.subtask_slot = 3;
        if (fault == 3) data.payload.async_context.writer_state += 8;
        if (fault == 4) data.payload.local_context.async_ctx.completion_error_code = reinterpret_cast<int32_t *>(0x40);
        if (fault == 5) data.payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = 0;
        AsyncCtx ctx{};
        EXPECT_FALSE(async_execution_context_load(data.payload, &ctx));
        EXPECT_FALSE(ctx.task_token.is_valid());
        EXPECT_EQ(ctx.completion_count, nullptr);
        EXPECT_EQ(ctx.completion_error_code, &data.payload.async_context.error_code);
        EXPECT_EQ(data.payload.async_context.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(data.writer.error_code, 0);
    }
}

TEST(HbgAsyncExecutionContext, ArgsEntryRejectsInvalidCarriersWithoutFallingBackToLegacy) {
    ContextData data;
    AsyncCtx ctx{};
    auto *args = reinterpret_cast<int64_t *>(data.payload.args);
    ASSERT_TRUE(async_execution_context_load(args, &ctx));
    EXPECT_EQ(ctx.completion_count, &data.writer.count);
    EXPECT_FALSE(async_execution_context_load(static_cast<int64_t *>(nullptr), &ctx));
    EXPECT_EQ(ctx.completion_count, nullptr);
    EXPECT_FALSE(async_execution_context_load(args + 1, &ctx));
    EXPECT_EQ(data.writer.error_code, 0);
    data.payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = 0;
    EXPECT_FALSE(async_execution_context_load(args, &ctx));
    EXPECT_EQ(data.payload.async_context.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(data.writer.error_code, 0);
    ContextData legacy;
    legacy.payload.async_context_address = 0;
    EXPECT_FALSE(async_execution_context_load(reinterpret_cast<int64_t *>(legacy.payload.args), &ctx));
}
}  // namespace
