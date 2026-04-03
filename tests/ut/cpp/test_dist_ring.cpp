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

#include <thread>

#include "dist_ring.h"

TEST(DistRing, InvalidWindowSizeThrows) {
    DistRing r;
    EXPECT_THROW(r.init(0), std::invalid_argument);
    EXPECT_THROW(r.init(3), std::invalid_argument);  // not power-of-2
    EXPECT_THROW(r.init(-1), std::invalid_argument);
}

TEST(DistRing, AllocReturnsValidSlots) {
    DistRing r;
    r.init(8);
    std::vector<DistTaskSlot> slots;
    for (int i = 0; i < 8; ++i) {
        DistTaskSlot s = r.alloc();
        EXPECT_GE(s, 0);
        EXPECT_LT(s, 8);
        slots.push_back(s);
    }
    // All 8 slots should be distinct
    std::sort(slots.begin(), slots.end());
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(slots[i], i);
}

TEST(DistRing, BackPressureAndRelease) {
    DistRing r;
    r.init(4);

    // Fill the ring
    std::vector<DistTaskSlot> held;
    for (int i = 0; i < 4; ++i)
        held.push_back(r.alloc());
    EXPECT_EQ(r.active_count(), 4);

    // Release one slot from another thread, then alloc should succeed
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        r.release(held[0]);
    });

    DistTaskSlot s = r.alloc();  // blocks until releaser runs
    EXPECT_NE(s, DIST_INVALID_SLOT);
    releaser.join();

    r.shutdown();
}

TEST(DistRing, ShutdownUnblocksAlloc) {
    DistRing r;
    r.init(2);
    r.alloc();
    r.alloc();  // ring full

    std::thread t([&] {
        DistTaskSlot s = r.alloc();  // should unblock when shutdown
        EXPECT_EQ(s, DIST_INVALID_SLOT);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    r.shutdown();
    t.join();
}
