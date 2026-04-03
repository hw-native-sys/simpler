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

#include "dist_tensormap.h"

TEST(DistTensorMap, LookupEmptyReturnsInvalid) {
    DistTensorMap tm;
    EXPECT_EQ(tm.lookup(0xDEADBEEF), DIST_INVALID_SLOT);
}

TEST(DistTensorMap, InsertAndLookup) {
    DistTensorMap tm;
    tm.insert(0x1000, 5);
    EXPECT_EQ(tm.lookup(0x1000), 5);
    EXPECT_EQ(tm.lookup(0x2000), DIST_INVALID_SLOT);
    EXPECT_EQ(tm.size(), 1);
}

TEST(DistTensorMap, OverwriteExistingEntry) {
    DistTensorMap tm;
    tm.insert(0x1000, 3);
    tm.insert(0x1000, 7);  // new producer reuses same buffer
    EXPECT_EQ(tm.lookup(0x1000), 7);
    EXPECT_EQ(tm.size(), 1);
}

TEST(DistTensorMap, EraseTaskOutputs) {
    DistTensorMap tm;
    tm.insert(0x1000, 0);
    tm.insert(0x2000, 0);
    tm.insert(0x3000, 1);

    tm.erase_task_outputs({0x1000, 0x2000});

    EXPECT_EQ(tm.lookup(0x1000), DIST_INVALID_SLOT);
    EXPECT_EQ(tm.lookup(0x2000), DIST_INVALID_SLOT);
    EXPECT_EQ(tm.lookup(0x3000), 1);
    EXPECT_EQ(tm.size(), 1);
}

TEST(DistTensorMap, EraseWithEmptyKeyList) {
    DistTensorMap tm;
    tm.insert(0x1000, 2);
    tm.erase_task_outputs({});
    EXPECT_EQ(tm.lookup(0x1000), 2);
}

TEST(DistTensorMap, MultipleEntries) {
    DistTensorMap tm;
    for (int i = 0; i < 100; ++i)
        tm.insert(static_cast<uint64_t>(i) * 0x1000, i % 16);
    EXPECT_EQ(tm.size(), 100);
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(tm.lookup(static_cast<uint64_t>(i) * 0x1000), i % 16);
}
