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

#include <condition_variable>
#include <mutex>
#include <thread>

#include "scope.h"

TEST(Scope, InitialDepthIsZero) {
    Scope sc;
    EXPECT_EQ(sc.depth(), 0);
}

TEST(Scope, ScopeEndWithoutBeginThrows) {
    Scope sc;
    EXPECT_THROW(sc.scope_end([](TaskSlot) {}), std::runtime_error);
}

TEST(Scope, SingleScope_ReleasesRegisteredTasks) {
    Scope sc;
    sc.scope_begin();
    EXPECT_EQ(sc.depth(), 1);
    sc.register_task(10);
    sc.register_task(20);

    std::vector<TaskSlot> released;
    sc.scope_end([&](TaskSlot s) {
        released.push_back(s);
    });

    EXPECT_EQ(sc.depth(), 0);
    ASSERT_EQ(released.size(), 2u);
    EXPECT_EQ(released[0], 10);
    EXPECT_EQ(released[1], 20);
}

TEST(Scope, RegisterOutsideScopeIsNoop) {
    Scope sc;
    sc.register_task(5);  // no open scope -- should not throw
    EXPECT_EQ(sc.depth(), 0);
}

TEST(Scope, NestedScopes) {
    Scope sc;
    sc.scope_begin();
    sc.register_task(1);
    sc.scope_begin();
    sc.register_task(2);
    EXPECT_EQ(sc.depth(), 2);

    std::vector<TaskSlot> inner_released;
    sc.scope_end([&](TaskSlot s) {
        inner_released.push_back(s);
    });
    EXPECT_EQ(sc.depth(), 1);
    ASSERT_EQ(inner_released.size(), 1u);
    EXPECT_EQ(inner_released[0], 2);

    std::vector<TaskSlot> outer_released;
    sc.scope_end([&](TaskSlot s) {
        outer_released.push_back(s);
    });
    EXPECT_EQ(sc.depth(), 0);
    ASSERT_EQ(outer_released.size(), 1u);
    EXPECT_EQ(outer_released[0], 1);
}

TEST(Scope, EmptyScopeReleasesNothing) {
    Scope sc;
    sc.scope_begin();
    int calls = 0;
    sc.scope_end([&](TaskSlot) {
        ++calls;
    });
    EXPECT_EQ(calls, 0);
}

TEST(Scope, ConcurrentThreadsOwnIndependentFrames) {
    Scope sc;
    std::mutex mu;
    std::condition_variable cv;
    int stage = 0;
    std::vector<TaskSlot> released_a;
    std::vector<TaskSlot> released_b;

    std::thread a([&]() {
        sc.scope_begin();
        sc.register_task(10);
        {
            std::unique_lock<std::mutex> lk(mu);
            stage = 1;
            cv.notify_all();
            cv.wait(lk, [&]() {
                return stage >= 2;
            });
        }
        sc.scope_end([&](TaskSlot slot) {
            released_a.push_back(slot);
        });
        {
            std::lock_guard<std::mutex> lk(mu);
            stage = 3;
            cv.notify_all();
        }
    });
    std::thread b([&]() {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&]() {
                return stage >= 1;
            });
        }
        sc.scope_begin();
        sc.register_task(20);
        {
            std::unique_lock<std::mutex> lk(mu);
            stage = 2;
            cv.notify_all();
            cv.wait(lk, [&]() {
                return stage >= 3;
            });
        }
        sc.scope_end([&](TaskSlot slot) {
            released_b.push_back(slot);
        });
    });

    a.join();
    b.join();
    ASSERT_EQ(released_a.size(), 1u);
    ASSERT_EQ(released_b.size(), 1u);
    EXPECT_EQ(released_a[0], 10);
    EXPECT_EQ(released_b[0], 20);
}
