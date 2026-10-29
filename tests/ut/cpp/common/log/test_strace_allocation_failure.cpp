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

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <thread>
#include <pthread.h>
#include <gtest/gtest.h>

#include "common/strace.h"
#include "host_log.h"

namespace {
std::atomic<bool> fail_allocation{false};
std::atomic<bool> fail_assignment{false};
std::atomic<bool> track_allocation{false};
std::atomic<void *> tracked{nullptr};
std::atomic<int> live_allocations{0};
}  // namespace

extern "C" void *__real_calloc(size_t, size_t);
extern "C" void __real_free(void *);
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);

extern "C" void *__wrap_calloc(size_t n, size_t size) {
    if (fail_allocation.exchange(false)) return nullptr;
    void *p = __real_calloc(n, size);
    if (p != nullptr && track_allocation.exchange(false)) {
        tracked.store(p);
        live_allocations.fetch_add(1);
    }
    return p;
}

extern "C" void __wrap_free(void *p) {
    if (p != nullptr && p == tracked.load()) {
        tracked.store(nullptr);
        live_allocations.fetch_sub(1);
    }
    __real_free(p);
}

extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *p) {
    if (fail_assignment.exchange(false)) return ENOMEM;
    return __real_pthread_setspecific(key, p);
}

TEST(StraceAllocationFailure, FailedAllocationLeavesNoUnbalancedScope) {
    HostLogger::get_instance().set_level(simpler::log::LogLevel::NUL, true);
    std::thread worker([] {
        fail_allocation.store(true);
        { STRACE("allocation_failed"); }
        auto *state = simpler::strace::strace_state();
        ASSERT_NE(state, nullptr);
        EXPECT_EQ(state->depth, 0);
        EXPECT_EQ(state->inv, 0u);
    });
    worker.join();
}

TEST(StraceAllocationFailure, FailedAssignmentFreesValueAndThreadExitFreesRetry) {
    HostLogger::get_instance().set_level(simpler::log::LogLevel::NUL, true);
    std::thread worker([] {
        track_allocation.store(true);
        fail_assignment.store(true);
        { STRACE_CONTEXT(7, 9, 2); }
        EXPECT_EQ(live_allocations.load(), 0);
        track_allocation.store(true);
        {
            STRACE_CONTEXT(17, 19, 3);
            auto *state = simpler::strace::strace_state();
            ASSERT_NE(state, nullptr);
            EXPECT_EQ(state->inv, 17u);
            EXPECT_EQ(state->depth, 3);
            EXPECT_EQ(live_allocations.load(), 1);
        }
    });
    worker.join();
    EXPECT_EQ(live_allocations.load(), 0);
}
