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
#include <cstdint>

#include <gtest/gtest.h>

#include "backend/rdma/rdma_completion_scheduler.h"

namespace {

using pto2::rdma_backend::kRdmaCqeOpcodeError;
using pto2::rdma_backend::kRdmaCqeOpcodeInvalid;
using pto2::rdma_backend::kRdmaCqeOpcodeShift;
using pto2::rdma_backend::kRdmaCqeOwnerBit;
using pto2::rdma_backend::poll_rdma_hns1825_cqe_record;

// The scheduler reads one aligned CQE slot; its first two dwords carry the owner
// bit and the opcode, so the fake below only has to materialize those.
constexpr uint32_t kTestCqeBytes = 64;

struct alignas(kTestCqeBytes) TestRdmaCqe {
    uint64_t dw01{0};
    uint64_t padding[7]{};
};

inline uint64_t cqe_addr(TestRdmaCqe &cqe) { return reinterpret_cast<uint64_t>(&cqe); }

inline void encode_test_cqe(TestRdmaCqe &cqe, uint32_t owner, uint32_t opcode) {
    cqe.dw01 =
        (static_cast<uint64_t>(owner) << kRdmaCqeOwnerBit) | (static_cast<uint64_t>(opcode) << kRdmaCqeOpcodeShift);
}

}  // namespace

TEST(A5RdmaCompletionScheduler, NullRecordAddressFails) {
    auto result = poll_rdma_hns1825_cqe_record(0, 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, InvalidExpectedOwnerFails) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 0, 0);
    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 2);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, OwnerNotReadyReturnsPending) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 1, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, InvalidOpcodeReturnsPending) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 0, kRdmaCqeOpcodeInvalid);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::PENDING);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, ErrorOpcodeFails) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 0, kRdmaCqeOpcodeError);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5RdmaCompletionScheduler, ReadyOpcodeReturnsReady) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 0, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 0);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}

TEST(A5RdmaCompletionScheduler, OwnerFlippedRoundReturnsReady) {
    TestRdmaCqe cqe{};
    encode_test_cqe(cqe, 1, 0);

    auto result = poll_rdma_hns1825_cqe_record(cqe_addr(cqe), 1);
    EXPECT_EQ(result.state, CompletionPollState::READY);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
}
