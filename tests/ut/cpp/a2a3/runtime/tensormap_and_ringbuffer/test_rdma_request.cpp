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

#include <cstdint>
#include <limits>

#include "rdma/rdma_request.h"

TEST(A3RdmaRequest, RegisteredWindowBoundsUseSubtraction) {
    RdmaTransferArgsV1 request{0x1000, 1, 64, 128, 256, 0};
    EXPECT_TRUE(rdma_request_in_bounds(request, 0, 2, 1, 1024));
    request.local_offset = 1024;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, 1024));
    request.local_offset = std::numeric_limits<uint64_t>::max() - 255;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, 1024));
    request.local_offset = 64;
    request.remote_offset = std::numeric_limits<uint64_t>::max() - 255;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, 1024));
}

TEST(A3RdmaRequest, PeerQueueAndNonemptyTransferAreRequired) {
    RdmaTransferArgsV1 request{0x1000, 1, 0, 0, 64, 1};
    EXPECT_TRUE(rdma_request_in_bounds(request, 0, 2, 2, 64));
    EXPECT_FALSE(rdma_request_in_bounds(request, 1, 2, 2, 64));
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 1, 2, 64));
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, 64));
    request.nbytes = 0;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 2, 64));
    request.nbytes = uint64_t{1} << 32;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 2, uint64_t{1} << 33));
}

TEST(A3RdmaRequest, ByteRangesRespectTheAlignedSignedCountContract) {
    RdmaTransferArgsV1 request{0x1000, 1, 0, 0, 64, 0};
    for (auto *field : {&request.local_offset, &request.remote_offset, &request.nbytes}) {
        const auto value = *field;
        *field = value + 1;
        EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, 4096));
        *field = value;
    }
    request.nbytes = uint64_t{1} << 31;
    EXPECT_FALSE(rdma_request_in_bounds(request, 0, 2, 1, uint64_t{1} << 32));
    request.nbytes -= 64;
    EXPECT_TRUE(rdma_request_in_bounds(request, 0, 2, 1, uint64_t{1} << 32));
}
