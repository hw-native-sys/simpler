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

#include "tensormap.h"

// Helper: host key (worker=-1)
static TensorKey hk(uint64_t ptr) { return {ptr, -1}; }

// Helper: child key (worker=w)
static TensorKey ck(uint64_t ptr, int8_t w) { return {ptr, w}; }

TEST(TensorMap, LookupEmptyReturnsInvalid) {
    TensorMap tm;
    EXPECT_EQ(tm.lookup(hk(0xDEADBEEF)), INVALID_SLOT);
}

TEST(TensorMap, InsertAndLookup) {
    TensorMap tm;
    tm.insert(hk(0x1000), 5);
    EXPECT_EQ(tm.lookup(hk(0x1000)), 5);
    EXPECT_EQ(tm.lookup(hk(0x2000)), INVALID_SLOT);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, OverwriteExistingEntry) {
    TensorMap tm;
    tm.insert(hk(0x1000), 3);
    tm.insert(hk(0x1000), 7);  // new producer reuses same buffer
    EXPECT_EQ(tm.lookup(hk(0x1000)), 7);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, EraseTaskOutputs) {
    TensorMap tm;
    tm.insert(hk(0x1000), 0);
    tm.insert(hk(0x2000), 0);
    tm.insert(hk(0x3000), 1);

    tm.erase_task_outputs({hk(0x1000), hk(0x2000)});

    EXPECT_EQ(tm.lookup(hk(0x1000)), INVALID_SLOT);
    EXPECT_EQ(tm.lookup(hk(0x2000)), INVALID_SLOT);
    EXPECT_EQ(tm.lookup(hk(0x3000)), 1);
    EXPECT_EQ(tm.size(), 1);
}

TEST(TensorMap, EraseWithEmptyKeyList) {
    TensorMap tm;
    tm.insert(hk(0x1000), 2);
    tm.erase_task_outputs({});
    EXPECT_EQ(tm.lookup(hk(0x1000)), 2);
}

TEST(TensorMap, MultipleEntries) {
    TensorMap tm;
    for (int i = 0; i < 100; ++i)
        tm.insert(hk(static_cast<uint64_t>(i) * 0x1000), i % 16);
    EXPECT_EQ(tm.size(), 100);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(tm.lookup(hk(static_cast<uint64_t>(i) * 0x1000)), i % 16);
}

// --- TensorKey compound key tests ---

TEST(TensorMap, SamePtrDifferentWorkerAreDistinct) {
    TensorMap tm;
    tm.insert(ck(0xABC, 0), 10);
    tm.insert(ck(0xABC, 1), 20);
    EXPECT_EQ(tm.lookup(ck(0xABC, 0)), 10);
    EXPECT_EQ(tm.lookup(ck(0xABC, 1)), 20);
    EXPECT_EQ(tm.size(), 2);
}

TEST(TensorMap, HostAndChildKeyAreDistinct) {
    TensorMap tm;
    tm.insert(hk(0x1000), 5);     // host (worker=-1)
    tm.insert(ck(0x1000, 0), 6);  // child worker 0
    EXPECT_EQ(tm.lookup(hk(0x1000)), 5);
    EXPECT_EQ(tm.lookup(ck(0x1000, 0)), 6);
    EXPECT_EQ(tm.size(), 2);
}

TEST(TensorMap, EraseChildKeyLeavesHostKey) {
    TensorMap tm;
    tm.insert(hk(0x1000), 5);
    tm.insert(ck(0x1000, 0), 6);
    tm.erase_task_outputs({ck(0x1000, 0)});
    EXPECT_EQ(tm.lookup(hk(0x1000)), 5);
    EXPECT_EQ(tm.lookup(ck(0x1000, 0)), INVALID_SLOT);
    EXPECT_EQ(tm.size(), 1);
}
