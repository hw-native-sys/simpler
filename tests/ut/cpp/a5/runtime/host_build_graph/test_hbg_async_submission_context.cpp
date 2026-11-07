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
#include <cstring>
#include <vector>

#include "async_submission_context.h"
#include "scheduler/scheduler_async_runtime.h"

namespace {
struct Fixture {
    struct alignas(128) Image {
        uint8_t prefix[128]{};
        uint8_t submission[128]{};
        AsyncSubmissionContext context{};
    } image;
    DispatchPayload payload{};
    AsyncSubmissionContext &ctx() { return image.context; }
    int64_t *args() { return reinterpret_cast<int64_t *>(payload.args); }
    Fixture() {
        auto &h = ctx().header;
        h.run_epoch = 19;
        h.region_address = reinterpret_cast<uint64_t>(&image);
        h.region_bytes = sizeof(image);
        h.submission_layout_offset = offsetof(Image, submission);
        h.task_token = TaskId::to_uint64(TaskId::make_global(3));
        h.execution_index = 7;
        h.dispatch_generation = 11;
        h.frame_address = reinterpret_cast<uint64_t>(&payload);
        h.block_idx = 2;
        h.subtask_slot = 1;
        h.frame_sub_block_id = 0;
        h.completion_mode = CoreAsyncMode::DEFERRED;
        auto &e = ctx().evidence;
        e.run_epoch = h.run_epoch;
        e.task_ref = 3;
        e.execution_index = h.execution_index;
        e.dispatch_generation = h.dispatch_generation;
        e.block_idx = h.block_idx;
        e.subtask_slot = h.subtask_slot;
        payload.async_submission_context = reinterpret_cast<uint64_t>(&ctx());
        payload.local_context.block_idx = 2;
        payload.local_context.async_ctx.task_token = TaskId::make_global(3);
        payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
        payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
    }
};

struct Publisher {
    AsyncSubmissionContext *context;
    std::vector<int> operations;
    bool metadata_ready{true};
    int32_t metadata_error{0};
    bool publish_backend_metadata() {
        EXPECT_EQ(context->evidence.flags, 0u);
        EXPECT_EQ(context->evidence.attempted, context->evidence.committed + 1);
        operations.push_back(1);
        if (metadata_error != 0) async_submission_error(*context, metadata_error);
        return metadata_ready;
    }
    void flush_evidence(volatile void *address, uint32_t bytes) {
        EXPECT_EQ(address, &context->evidence);
        EXPECT_EQ(bytes, sizeof(AsyncSubmissionEvidence));
        EXPECT_EQ(context->evidence.attempted, context->evidence.committed);
        EXPECT_EQ(context->evidence.flags, ASYNC_SUBMISSION_METADATA_PUBLISHED);
        operations.push_back(2);
    }
    void finish() { operations.push_back(3); }
};

TEST(HbgAsyncSubmissionContext, ExplicitCarrierPreservesExistingWireOffsetsAndSizes) {
    EXPECT_EQ(sizeof(AsyncCtxStorage), 40u);
    EXPECT_EQ(sizeof(AsyncCtx), 48u);
    EXPECT_EQ(sizeof(LocalContext), 48u);
    EXPECT_EQ(sizeof(DispatchPayload), 512u);
    EXPECT_EQ(offsetof(DispatchPayload, args), 64u);
    EXPECT_EQ(offsetof(DispatchPayload, global_context), 464u);
    EXPECT_EQ(offsetof(DispatchPayload, async_submission_context), 472u);
    EXPECT_EQ(sizeof(AsyncSubmissionContext), 256u);
    EXPECT_EQ(offsetof(AsyncSubmissionContext, evidence), 128u);
}

TEST(HbgAsyncSubmissionContext, GetterRequiresTheCurrentFrameIdentityAndCompletionMode) {
    Fixture f;
    ASSERT_TRUE(async_submission_context_valid(&f.ctx()));
    EXPECT_EQ(get_async_submission_context(f.args()), &f.ctx());
    f.payload.local_context.block_idx = 1;
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.payload.local_context.block_idx = 2;
    f.payload.local_context.async_ctx.task_token = TaskId::make_global(4);
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.ctx().header.completion_mode = CoreAsyncMode::SYNCHRONOUS;
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.payload.local_context.async_ctx.task_token = TaskId::invalid();
    EXPECT_EQ(get_async_submission_context(f.args()), &f.ctx());
    f.payload.async_submission_context = 0;
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    EXPECT_EQ(get_async_submission_context(nullptr), nullptr);
}

TEST(HbgAsyncSubmissionContext, InvalidSchemaBoundsAndEvidenceAreRejectedWithoutMutation) {
    for (uint32_t change = 0; change < 8; ++change) {
        Fixture f;
        if (change == 0) ++f.ctx().header.version;
        if (change == 1) f.ctx().header.context_bytes = 128;
        if (change == 2) f.ctx().header.region_bytes = 128;
        if (change == 3) f.ctx().header.region_address = UINT64_MAX - 127;
        if (change == 4) f.ctx().header.submission_layout_offset = 0;
        if (change == 5) ++f.ctx().evidence.dispatch_generation;
        if (change == 6) f.ctx().evidence.reserved[0] = 1;
        if (change == 7) f.ctx().evidence.flags = 8;
        std::array<uint8_t, sizeof(AsyncSubmissionContext)> before{};
        std::memcpy(before.data(), &f.ctx(), before.size());
        EXPECT_FALSE(async_submission_context_valid(&f.ctx()));
        EXPECT_FALSE(async_submission_begin(f.ctx()));
        EXPECT_EQ(std::memcmp(before.data(), &f.ctx(), before.size()), 0);
    }
}

TEST(HbgAsyncSubmissionContext, SameTaskAndBlockCannotBorrowAnotherExecutionContext) {
    Fixture f;
    Fixture sibling;
    ASSERT_EQ(f.ctx().header.task_token, sibling.ctx().header.task_token);
    ASSERT_EQ(f.ctx().header.block_idx, sibling.ctx().header.block_idx);
    f.payload.async_submission_context = reinterpret_cast<uint64_t>(&sibling.ctx());
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.ctx().header.completion_mode = CoreAsyncMode::SYNCHRONOUS;
    sibling.ctx().header.completion_mode = CoreAsyncMode::SYNCHRONOUS;
    f.payload.local_context.async_ctx.task_token = TaskId::invalid();
    sibling.payload.local_context.async_ctx.task_token = TaskId::invalid();
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
}

TEST(HbgAsyncSubmissionContext, ContextArgumentPointersCannotNameAnotherFrame) {
    Fixture f;
    Fixture sibling;
    f.payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&sibling.payload.local_context);
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&f.payload.local_context);
    f.payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&sibling.payload.global_context);
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
}

TEST(HbgAsyncSubmissionContext, PhysicalSubblockComesFromTheFrameRatherThanTheLogicalSubslot) {
    Fixture f;
    f.ctx().header.subtask_slot = 2;
    f.ctx().evidence.subtask_slot = 2;
    EXPECT_EQ(get_async_submission_context(f.args()), &f.ctx());
    f.ctx().header.frame_sub_block_id = 1;
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    f.payload.global_context.sub_block_id = 1;
    EXPECT_EQ(get_async_submission_context(f.args()), &f.ctx());
}

TEST(HbgAsyncSubmissionContext, BackendPublicationPrecedesCommittedEvidence) {
    Fixture f;
    Publisher publisher{&f.ctx(), {}};
    for (uint64_t i = 0; i < 3; ++i) {
        ASSERT_TRUE(async_submission_begin(f.ctx()));
        EXPECT_EQ(f.ctx().evidence.attempted, i + 1);
        EXPECT_EQ(f.ctx().evidence.committed, i);
        EXPECT_EQ(f.ctx().evidence.flags, 0u);
        ASSERT_TRUE(async_submission_commit(f.ctx(), publisher));
        EXPECT_EQ(f.ctx().evidence.committed, i + 1);
    }
    EXPECT_EQ(publisher.operations, (std::vector<int>{1, 2, 3, 1, 2, 3, 1, 2, 3}));
    EXPECT_EQ(f.ctx().evidence.error_code, 0);
}

TEST(HbgAsyncSubmissionContext, FailedMetadataPublicationKeepsTheAttemptWithoutPublishingSuccess) {
    for (int32_t error : {0, -37}) {
        Fixture f;
        Publisher publisher{&f.ctx(), {}};
        ASSERT_TRUE(async_submission_begin(f.ctx()));
        ASSERT_TRUE(async_submission_commit(f.ctx(), publisher));
        publisher.operations.clear();
        publisher.metadata_ready = false;
        publisher.metadata_error = error;
        ASSERT_TRUE(async_submission_begin(f.ctx()));
        EXPECT_FALSE(async_submission_commit(f.ctx(), publisher));
        EXPECT_EQ(f.ctx().evidence.attempted, 2u);
        EXPECT_EQ(f.ctx().evidence.committed, 1u);
        EXPECT_EQ(f.ctx().evidence.flags, 0u);
        EXPECT_EQ(f.ctx().evidence.error_code, error != 0 ? error : SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(publisher.operations, (std::vector<int>{1}));
        EXPECT_FALSE(async_submission_begin(f.ctx()));
        EXPECT_FALSE(async_submission_commit(f.ctx(), publisher));
        EXPECT_EQ(publisher.operations, (std::vector<int>{1}));
        EXPECT_EQ(f.ctx().evidence.attempted, 2u);
        EXPECT_EQ(f.ctx().evidence.committed, 1u);
    }
}

TEST(HbgAsyncSubmissionContext, UncommittedOrFailedAttemptsNeverBecomeNoSubmission) {
    Fixture f;
    ASSERT_TRUE(async_submission_begin(f.ctx()));
    EXPECT_FALSE(async_submission_begin(f.ctx()));
    EXPECT_EQ(f.ctx().evidence.attempted, 1u);
    EXPECT_EQ(f.ctx().evidence.committed, 0u);
    EXPECT_EQ(f.ctx().evidence.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    Publisher publisher{&f.ctx(), {}};
    EXPECT_FALSE(async_submission_commit(f.ctx(), publisher));
    EXPECT_TRUE(publisher.operations.empty());
    async_submission_error(f.ctx(), -37);
    EXPECT_EQ(f.ctx().evidence.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(HbgAsyncSubmissionContext, OverflowAndCommitWithoutAttemptFailBeforePublication) {
    Fixture f;
    f.ctx().evidence.attempted = UINT64_MAX;
    f.ctx().evidence.committed = UINT64_MAX;
    EXPECT_FALSE(async_submission_begin(f.ctx()));
    EXPECT_EQ(f.ctx().evidence.attempted, UINT64_MAX);
    Fixture fresh;
    Publisher publisher{&fresh.ctx(), {}};
    EXPECT_FALSE(async_submission_commit(fresh.ctx(), publisher));
    EXPECT_TRUE(publisher.operations.empty());
    EXPECT_EQ(fresh.ctx().evidence.attempted, 0u);
}

TEST(HbgAsyncSubmissionContext, SealedEvidenceCannotBeReopenedByTheKernelSdk) {
    Fixture f;
    f.ctx().evidence.flags |= ASYNC_SUBMISSION_SEALED;
    EXPECT_EQ(get_async_submission_context(f.args()), nullptr);
    EXPECT_FALSE(async_submission_begin(f.ctx()));
    Publisher publisher{&f.ctx(), {}};
    EXPECT_FALSE(async_submission_commit(f.ctx(), publisher));
    EXPECT_TRUE(publisher.operations.empty());
    EXPECT_EQ(f.ctx().evidence.flags, ASYNC_SUBMISSION_FLAGS);
    EXPECT_EQ(f.ctx().evidence.error_code, 0);
}

TEST(HbgAsyncSubmissionContext, RebindingAnOrdinarySlotClearsThePriorSubmissionPointer) {
    Fixture f;
    SchedulerAsyncRuntimeView disabled;
    SchedulerGraphView empty_graph{};
    SchedulerGraphAccess graph{empty_graph};
    bool bound = true;
    ASSERT_TRUE(scheduler_async_runtime_bind_payload(disabled, graph, nullptr, nullptr, 0, 0, 0, &f.payload, &bound));
    EXPECT_FALSE(bound);
    EXPECT_EQ(f.payload.async_submission_context, 0u);
    EXPECT_FALSE(f.payload.local_context.async_ctx.task_token.is_valid());
}
}  // namespace
