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
#include "backend/urma/urma_completion_scheduler.h"

namespace {
using namespace pto2::urma_backend;
struct Queue {
    UrmaInfo info{};
    UrmaWqCtx wq[2]{};
    UrmaCqCtx cq[2]{};
    alignas(64) uint32_t entries[4][16]{};
    uint32_t bb = 0, cqe = 0, db = 0;
    Queue() {
        info.qp_num = 1;
        info.jetty_count = 2;
        info.jetties_per_core = 1;
        info.rank_count = 2;
        info.sq_ptr = reinterpret_cast<uint64_t>(wq);
        info.scq_ptr = reinterpret_cast<uint64_t>(cq);
        wq[1].depth = 4;
        wq[1].tail_addr = reinterpret_cast<uint64_t>(&bb);
        cq[1].depth = 4;
        cq[1].cqe_shift_size = 6;
        cq[1].buf_addr = reinterpret_cast<uint64_t>(entries);
        cq[1].tail_addr = reinterpret_cast<uint64_t>(&cqe);
        cq[1].db_addr = reinterpret_cast<uint64_t>(&db);
    }
    uint64_t addr() { return reinterpret_cast<uint64_t>(&info); }
};
}  // namespace

TEST(A5UrmaCompletionScheduler, ZeroTargetAfterWrapIsNotEmpty) {
    Queue q;
    q.cqe = q.bb = UINT32_MAX;
    // At UINT32_MAX the next valid CQE has owner zero; owner one is pending.
    q.entries[3][0] = 4;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 0), q.addr(), 0).state, CompletionPollState::PENDING);
}

TEST(A5UrmaCompletionScheduler, MultipleRowsPerPeerAreRejected) {
    Queue q;
    q.info.qp_num = 2;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(0, 1), q.addr(), 1).state, CompletionPollState::FAILED);
}

TEST(A5UrmaCompletionScheduler, PendingPrefixDoesNotPublishEitherTail) {
    Queue q;
    q.entries[0][0] = 4;
    auto result = poll_urma_event_handle(encode_urma_event_handle(1, 3), q.addr(), 2);
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(q.cqe, 0u);
    EXPECT_EQ(q.bb, 0u);
    q.entries[1][0] = 4;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 3), q.addr(), 2).state, CompletionPollState::READY);
    EXPECT_EQ(q.cqe, 2u);
    EXPECT_EQ(q.bb, 3u);
    EXPECT_EQ(q.db, 2u);
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 3), q.addr(), 2).state, CompletionPollState::READY);
    EXPECT_EQ(q.bb, 3u);
}

TEST(A5UrmaCompletionScheduler, FailedPrefixCannotBecomeSuccessfulOnRetry) {
    Queue q;
    q.entries[0][0] = 4 | (1u << 24);
    for (int retry = 0; retry < 2; ++retry) {
        auto result = poll_urma_event_handle(encode_urma_event_handle(1, 1), q.addr(), 1);
        EXPECT_EQ(result.state, CompletionPollState::FAILED);
        EXPECT_NE(result.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(q.cqe, 0u);
        EXPECT_EQ(q.bb, 0u);
    }
}

TEST(A5UrmaCompletionScheduler, CounterAndOwnerWrap) {
    Queue q;
    q.cqe = UINT32_MAX;
    q.bb = UINT32_MAX - 1;
    q.entries[3][0] = 0;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 0), q.addr(), 0).state, CompletionPollState::READY);
    EXPECT_EQ(q.bb, 0u);
    EXPECT_EQ(q.cqe, 0u);
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 1), q.addr(), 1).state, CompletionPollState::PENDING);
    q.entries[0][0] = 4;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 1), q.addr(), 1).state, CompletionPollState::READY);
}

TEST(A5UrmaCompletionScheduler, BothTargetsMustBeReached) {
    Queue q;
    q.cqe = 2;
    q.bb = 1;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 3), q.addr(), 2).state, CompletionPollState::PENDING);
}

TEST(A5UrmaCompletionScheduler, InvalidWorkspaceLayoutAndTargetsFail) {
    EXPECT_EQ(poll_urma_event_handle(0, 0, 0).state, CompletionPollState::FAILED);
    Queue q;
    q.info.layout = 1;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 1), q.addr(), 1).state, CompletionPollState::FAILED);
    q.info.layout = 0;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 5), q.addr(), 5).state, CompletionPollState::FAILED);
    q.cq[1].cqe_shift_size = 5;
    EXPECT_EQ(poll_urma_event_handle(encode_urma_event_handle(1, 1), q.addr(), 1).state, CompletionPollState::FAILED);
}
