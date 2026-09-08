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

#include <cstring>

#include "backend/rdma/rdma_completion_scheduler.h"

namespace {

using pto2::rdma_backend::encode_rdma_event_handle;
using pto2::rdma_backend::Hns1825Cqe;
using pto2::rdma_backend::is_hns1825_cqe_owner_ready;
using pto2::rdma_backend::is_rdma_error_handle;
using pto2::rdma_backend::kCqeBytes;
using pto2::rdma_backend::poll_rdma_event_handle;
using pto2::rdma_backend::RdmaCqCtx;
using pto2::rdma_backend::RdmaWqCtx;

constexpr uint32_t kTestCqDepth = 1024;

struct alignas(kCqeBytes) TestRdmaCqe : Hns1825Cqe {};

struct TestRdmaWorkspace {
    pto2::rdma_backend::RdmaInfo info;
    RdmaWqCtx sq[2];
    RdmaCqCtx scq[2];
    uint32_t sq_tail[2];
    uint32_t cq_tail[2];
    uint32_t cq_sw_doorbell[2];
    TestRdmaCqe scq_entries[2][kTestCqDepth];
};

inline bool test_cqe_ready_owner(uint32_t cqe_seq) { return (cqe_seq & kTestCqDepth) != 0; }

inline void encode_test_cqe(TestRdmaCqe &cqe, uint32_t cqe_seq, bool error) {
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOptypeSend = 0;
    constexpr uint32_t kCqeOptypeError = 0x1e;
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 1u : 0u;
    cqe.owner_id_qpn = owner << kOwnerShift;
    cqe.op_sr_wqebb = (error ? kCqeOptypeError : kCqeOptypeSend) << kCqeOpcodeShift;
    cqe.syndrome = error ? 9 : 0;
}

inline void encode_test_pending_cqe(TestRdmaCqe &cqe, uint32_t cqe_seq) {
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOptypeSend = 0;
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 0u : 1u;
    cqe.owner_id_qpn = owner << kOwnerShift;
    cqe.op_sr_wqebb = kCqeOptypeSend << kCqeOpcodeShift;
}

inline void encode_test_invalid_cqe(TestRdmaCqe &cqe, uint32_t cqe_seq) {
    constexpr uint32_t kOwnerShift = 31;
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOptypeInvalid = 0x1f;
    const uint32_t owner = test_cqe_ready_owner(cqe_seq) ? 1u : 0u;
    cqe.owner_id_qpn = owner << kOwnerShift;
    cqe.op_sr_wqebb = kCqeOptypeInvalid << kCqeOpcodeShift;
}

struct SchedulerWorkspaceFixture {
    TestRdmaWorkspace ws{};

    SchedulerWorkspaceFixture() {
        std::memset(&ws, 0, sizeof(ws));
        ws.info.magic = pto2::rdma_backend::kRdmaWorkspaceMagic;
        ws.info.version = pto2::rdma_backend::kRdmaWorkspaceVersion;
        ws.info.backend = pto2::rdma_backend::kRdmaBackendHns1825;
        ws.info.qp_num = 1;
        ws.info.rank_count = 2;
        ws.info.sq_ptr = reinterpret_cast<uint64_t>(&ws.sq[0]);
        ws.info.scq_ptr = reinterpret_cast<uint64_t>(&ws.scq[0]);
        for (uint32_t rank = 0; rank < ws.info.rank_count; ++rank) {
            RdmaWqCtx &sq = ws.sq[rank];
            sq.wqe_size = kCqeBytes;
            sq.depth = kTestCqDepth;
            sq.tail_addr = reinterpret_cast<uint64_t>(&ws.sq_tail[rank]);

            RdmaCqCtx &cq = ws.scq[rank];
            cq.buf_addr = reinterpret_cast<uint64_t>(&ws.scq_entries[rank][0]);
            cq.cqe_size = kCqeBytes;
            cq.depth = kTestCqDepth;
            cq.tail_addr = reinterpret_cast<uint64_t>(&ws.cq_tail[rank]);
            cq.db_sw_addr = reinterpret_cast<uint64_t>(&ws.cq_sw_doorbell[rank]);
        }
    }

    uint64_t addr() { return reinterpret_cast<uint64_t>(&ws); }
};

}  // namespace

TEST(A5RdmaCompletionScheduler, ErrorHandleIsDetected) {
    EXPECT_TRUE(is_rdma_error_handle(encode_rdma_event_handle(0xffffffffu, 7)));
    EXPECT_FALSE(is_rdma_error_handle(encode_rdma_event_handle(1, 7)));
}

TEST(A5RdmaCompletionScheduler, ZeroHandleIsReady) {
    auto result = poll_rdma_event_handle(0, 0);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, ErrorHandleFails) {
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(0xffffffffu, 9), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, NullWorkspaceFails) {
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, OwnerReadyMatchesHns1825Backend) {
    EXPECT_TRUE(is_hns1825_cqe_owner_ready(false, 0, kTestCqDepth));
    EXPECT_FALSE(is_hns1825_cqe_owner_ready(true, 0, kTestCqDepth));
    EXPECT_TRUE(is_hns1825_cqe_owner_ready(true, kTestCqDepth, kTestCqDepth));
    EXPECT_FALSE(is_hns1825_cqe_owner_ready(false, kTestCqDepth, kTestCqDepth));
}

TEST(A5RdmaCompletionScheduler, InvalidWorkspaceMagicFails) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.info.magic = 0;

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, InvalidRankFails) {
    SchedulerWorkspaceFixture fixture;
    auto result = poll_rdma_event_handle(encode_rdma_event_handle(3, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, TailAlreadyAtTargetIsReady) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.cq_tail[1] = 7;

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 7), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, OwnerNotReadyReturnsPending) {
    SchedulerWorkspaceFixture fixture;
    encode_test_pending_cqe(fixture.ws.scq_entries[1][0], 0);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 0u);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, InvalidOpcodeReturnsPending) {
    SchedulerWorkspaceFixture fixture;
    encode_test_invalid_cqe(fixture.ws.scq_entries[1][0], 0);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 0u);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], 0u);
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, ReadyCqeAdvancesCqDoorbellAndSqTail) {
    SchedulerWorkspaceFixture fixture;
    encode_test_cqe(fixture.ws.scq_entries[1][0], 0, false);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], 1u);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], __builtin_bswap32(1u));
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

TEST(A5RdmaCompletionScheduler, ReadyCqeAfterWrapAdvancesCqDoorbellAndSqTail) {
    SchedulerWorkspaceFixture fixture;
    fixture.ws.cq_tail[1] = kTestCqDepth;
    fixture.ws.sq_tail[1] = kTestCqDepth;
    encode_test_cqe(fixture.ws.scq_entries[1][0], kTestCqDepth, false);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, kTestCqDepth + 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.ws.cq_tail[1], kTestCqDepth + 1);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], __builtin_bswap32(kTestCqDepth + 1));
    EXPECT_EQ(fixture.ws.sq_tail[1], kTestCqDepth);
}

TEST(A5RdmaCompletionScheduler, CqeErrorFailsAfterRetiringCqe) {
    SchedulerWorkspaceFixture fixture;
    encode_test_cqe(fixture.ws.scq_entries[1][0], 0, true);

    auto result = poll_rdma_event_handle(encode_rdma_event_handle(1, 1), fixture.addr());
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.ws.cq_tail[1], 1u);
    EXPECT_EQ(fixture.ws.cq_sw_doorbell[1], __builtin_bswap32(1u));
    EXPECT_EQ(fixture.ws.sq_tail[1], 0u);
}

static_assert(sizeof(TestRdmaCqe) == kCqeBytes);
