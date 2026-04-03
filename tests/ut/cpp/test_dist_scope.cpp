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

#include "dist_scope.h"

TEST(DistScope, InitialDepthIsZero) {
    DistScope sc;
    EXPECT_EQ(sc.depth(), 0);
}

TEST(DistScope, ScopeEndWithoutBeginThrows) {
    DistScope sc;
    EXPECT_THROW(sc.scope_end([](DistTaskSlot) {}), std::runtime_error);
}

TEST(DistScope, SingleScope_ReleasesRegisteredTasks) {
    DistScope sc;
    sc.scope_begin();
    EXPECT_EQ(sc.depth(), 1);
    sc.register_task(10);
    sc.register_task(20);

    std::vector<DistTaskSlot> released;
    sc.scope_end([&](DistTaskSlot s) {
        released.push_back(s);
    });

    EXPECT_EQ(sc.depth(), 0);
    ASSERT_EQ(released.size(), 2u);
    EXPECT_EQ(released[0], 10);
    EXPECT_EQ(released[1], 20);
}

TEST(DistScope, RegisterOutsideScopeIsNoop) {
    DistScope sc;
    sc.register_task(5);  // no open scope — should not throw
    EXPECT_EQ(sc.depth(), 0);
}

TEST(DistScope, NestedScopes) {
    DistScope sc;
    sc.scope_begin();
    sc.register_task(1);
    sc.scope_begin();
    sc.register_task(2);
    EXPECT_EQ(sc.depth(), 2);

    std::vector<DistTaskSlot> inner_released;
    sc.scope_end([&](DistTaskSlot s) {
        inner_released.push_back(s);
    });
    EXPECT_EQ(sc.depth(), 1);
    ASSERT_EQ(inner_released.size(), 1u);
    EXPECT_EQ(inner_released[0], 2);

    std::vector<DistTaskSlot> outer_released;
    sc.scope_end([&](DistTaskSlot s) {
        outer_released.push_back(s);
    });
    EXPECT_EQ(sc.depth(), 0);
    ASSERT_EQ(outer_released.size(), 1u);
    EXPECT_EQ(outer_released[0], 1);
}

TEST(DistScope, EmptyScopeReleasesNothing) {
    DistScope sc;
    sc.scope_begin();
    int calls = 0;
    sc.scope_end([&](DistTaskSlot) {
        ++calls;
    });
    EXPECT_EQ(calls, 0);
}
