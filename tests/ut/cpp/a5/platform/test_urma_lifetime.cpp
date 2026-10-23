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
#include <vector>
#include "host/urma_lifetime.h"

TEST(A5UrmaLifetime, StopDrainDestroyMustAllSucceedBeforeReclaim) {
    for (int fail = 0; fail <= 3; ++fail) {
        std::vector<int> calls;
        auto phase = [&](int n) {
            calls.push_back(n);
            return fail != n;
        };
        bool ok = urma_lifetime::shutdown(
            [&] {
                return phase(1);
            },
            [&] {
                return phase(2);
            },
            [&] {
                return phase(3);
            },
            [&] {
                calls.push_back(4);
            }
        );
        EXPECT_EQ(ok, fail == 0);
        EXPECT_EQ(calls.size(), fail == 0 ? 4u : static_cast<size_t>(fail));
        if (ok) EXPECT_EQ(calls, (std::vector<int>{1, 2, 3, 4}));
    }
}

TEST(A5UrmaLifetime, IndependentBbAndCqeDrainAndReadFailure) {
    using namespace pto2::urma_backend;
    UrmaInfo info{};
    UrmaWqCtx sq[2]{};
    UrmaCqCtx cq[2]{};
    uint32_t head = 3, tail = 2, completed = 2;
    info.qp_num = info.jetties_per_core = 1;
    info.rank_count = info.jetty_count = 2;
    info.sq_ptr = reinterpret_cast<uint64_t>(sq);
    info.scq_ptr = reinterpret_cast<uint64_t>(cq);
    sq[1].head_addr = reinterpret_cast<uint64_t>(&head);
    sq[1].tail_addr = reinterpret_cast<uint64_t>(&tail);
    sq[1].submitted_wqe_count = 2;
    cq[1].tail_addr = reinterpret_cast<uint64_t>(&completed);
    auto read = [](void *dst, uint64_t src, size_t size) {
        std::memcpy(dst, reinterpret_cast<void *>(src), size);
        return true;
    };
    auto addr = reinterpret_cast<uint64_t>(&info);
    EXPECT_FALSE(urma_lifetime::queues_drained(addr, read, 2));
    tail = head;
    EXPECT_TRUE(urma_lifetime::queues_drained(addr, read, 2));
    completed = 1;
    EXPECT_FALSE(urma_lifetime::queues_drained(addr, read, 2));
    sq[1].head_addr = sq[1].tail_addr = cq[1].tail_addr = 0;
    EXPECT_FALSE(urma_lifetime::queues_drained(addr, read, 2));
    EXPECT_FALSE(
        urma_lifetime::queues_drained(
            addr,
            [](void *, uint64_t, size_t) {
                return false;
            },
            2
        )
    );
}
