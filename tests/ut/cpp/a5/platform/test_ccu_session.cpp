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
#include <fstream>
#include <iterator>

#include "support/ccu_session_fake.h"
#include "platform_comm/ccu/ccu_device.h"

namespace {
class CcuSessionTest : public testing::Test {
protected:
    std::shared_ptr<ccu_test::SessionControl> control = std::make_shared<ccu_test::SessionControl>();
    CcuSession session{Config()};
    CcuDeviceContext *ctx{};
    CcuProgramRef program{};
    CcuPositionalArguments args{};

    static CcuSessionConfig Config() {
        CcuSessionConfig config;
        config.allocation_id = 11;
        config.epoch = 29;
        config.device_id = 0;
        config.parent_owner = std::make_shared<int>(0);
        config.exchange = [](uint64_t, uint32_t, const void *, uint32_t, void *, uint32_t, uint32_t) {
            return 0;
        };
        return config;
    }

    void SetUp() override {
        ccu_test::QueueBackend(control);
        ASSERT_EQ(session.initialize(), CcuStatus::OK);
        std::ifstream file(CCU_TEST_POSITIONAL_MODULE, std::ios::binary);
        ASSERT_TRUE(file);
        CcuEmbeddedFunction function;
        function.binary.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        function.entry = "broadcast_root0";
        ASSERT_EQ(session.register_functions({function}), CcuStatus::OK);
        ASSERT_EQ(session.prepare_memory(1, 0x10000, 256), CcuStatus::OK);
        CcuDeviceContextView view{};
        ASSERT_EQ(session.get_device_context(view), CcuStatus::OK);
        ASSERT_EQ(ccu_get_device_context(view.addr, view.size, ctx), CcuStatus::OK);
        ASSERT_EQ(ccu_resolve_program(ctx, 0, program), CcuStatus::OK);
        args.tensor_count = 1;
        args.tensors[0] = {0x10000, 0, 4};
    }
    void TearDown() override { EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK); }
};
}  // namespace

TEST_F(CcuSessionTest, OutOfRangeArgumentDoesNotPublishAndNextRequestCompletes) {
    CcuRequestRef request{};
    args.tensors[0] = {0x100fc, 0, 8};
    EXPECT_EQ(ccu_reserve_request(ctx, program, request, args), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(request.record_addr, 0U);
    EXPECT_EQ(control->gate, 0U);
    EXPECT_EQ(ctx->next_sequence, 1U);
    args.tensors[0].byte_size = 4;
    ASSERT_EQ(ccu_reserve_request(ctx, program, request, args), CcuStatus::OK);
    ASSERT_EQ(ccu_submit_request(request), CcuStatus::OK);
    control->Complete(request);
    EXPECT_EQ(ccu_poll_request(request).state, CcuRequestState::SUCCEEDED);
    ccu_retire_request(request);
}

TEST_F(CcuSessionTest, SlotWaitsForMatchingCompletionAndReaderRetirement) {
    CcuRequestRef first{}, second{};
    ASSERT_EQ(session.retain_run(), CcuStatus::OK);
    ASSERT_EQ(ccu_reserve_request(ctx, program, first, args), CcuStatus::OK);
    ASSERT_EQ(ccu_submit_request(first), CcuStatus::OK);
    auto *record = ccu_detail::At<CcuRequestRecord>(ctx, ctx->record_offset);
    record->completed_sequence = first.sequence - 1;
    EXPECT_EQ(ccu_poll_request(first).state, CcuRequestState::PENDING);
    ccu_retire_request(first);
    EXPECT_EQ(ccu_reserve_request(ctx, program, second, args), CcuStatus::BUSY);

    control->Complete(first);
    ASSERT_EQ(ccu_poll_request(first).state, CcuRequestState::SUCCEEDED);
    EXPECT_EQ(session.poll_idle().state, CcuRequestState::PENDING);
    EXPECT_EQ(ccu_reserve_request(ctx, program, second, args), CcuStatus::BUSY);

    ASSERT_EQ(session.release_run(), CcuStatus::OK);
    ccu_retire_request(first);
    ASSERT_EQ(ccu_reserve_request(ctx, program, second, args), CcuStatus::OK);
    EXPECT_GT(second.sequence, first.sequence);
    EXPECT_EQ(ccu_poll_request(first).state, CcuRequestState::UNCONFIRMED);
    ccu_cancel_reserved_request(second);
    ASSERT_EQ(session.retain_run(), CcuStatus::OK);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::BUSY);
    ASSERT_EQ(session.release_run(), CcuStatus::OK);
}

TEST_F(CcuSessionTest, OverlappingImageRegionsAreRejectedBeforeUse) {
    ctx->arguments_offset = ctx->record_offset;
    CcuDeviceContext *resolved = ctx;
    EXPECT_EQ(
        ccu_get_device_context(reinterpret_cast<uint64_t>(ctx), ctx->image_size, resolved), CcuStatus::ABI_MISMATCH
    );
    EXPECT_EQ(resolved, nullptr);
}
