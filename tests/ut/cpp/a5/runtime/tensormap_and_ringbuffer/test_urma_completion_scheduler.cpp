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

using pto2::urma_backend::poll_urma_cqe_record;

TEST(UrmaCompletionScheduler, PollsOneCqeDw0WithoutWritingIt) {
    alignas(CHIP_ALIGN_SIZE) uint32_t dw0 = 0;

    EXPECT_EQ(poll_urma_cqe_record(/*cqe_addr=*/0, /*expected_owner=*/0).state, CompletionPollState::FAILED);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/2).state, CompletionPollState::FAILED
    );

    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::PENDING
    );
    EXPECT_EQ(dw0, 0U);

    dw0 = 1U << 2U;
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::READY
    );
    EXPECT_EQ(dw0, 1U << 2U);

    dw0 = (1U << 2U) | (1U << 24U);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::FAILED
    );
    EXPECT_EQ(dw0, (1U << 2U) | (1U << 24U));

    dw0 = (1U << 2U) | (1U << 16U);
    EXPECT_EQ(
        poll_urma_cqe_record(reinterpret_cast<uint64_t>(&dw0), /*expected_owner=*/1).state, CompletionPollState::FAILED
    );
    EXPECT_EQ(dw0, (1U << 2U) | (1U << 16U));
}
