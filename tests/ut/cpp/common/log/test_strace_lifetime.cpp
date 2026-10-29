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

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <gtest/gtest.h>

#include "common/strace.h"
#include "host_log.h"

namespace {
size_t available_keys() {
    std::vector<pthread_key_t> keys;
    pthread_key_t key;
    while (pthread_key_create(&key, nullptr) == 0)
        keys.push_back(key);
    for (auto value : keys)
        pthread_key_delete(value);
    return keys.size();
}

int reload_with_surviving_threads() {
    // libc++ initializes a process-lifetime pthread key on the first std::thread.
    // The key-count baseline includes this standard-library state.
    std::thread([] {}).join();
    auto &owner = HostLogger::get_instance();
    owner.set_level(simpler::log::LogLevel::NUL, true);
    const size_t before = available_keys();
    for (int round = 0; round < 8; ++round) {
        void *handle = dlopen(TEST_STRACE_CONSUMER_PATH, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return 10;
        auto bind = reinterpret_cast<int (*)(SimplerHostLogState *)>(dlsym(handle, "test_strace_bind"));
        auto touch = reinterpret_cast<bool (*)()>(dlsym(handle, "test_strace_touch"));
        if (!bind || !touch || bind(owner.state()) != 0) return 11;
        std::mutex mutex;
        std::condition_variable cv;
        bool ready = false, leave = false, held = false;
        std::thread survivor([&] {
            held = touch();
            std::unique_lock<std::mutex> lock(mutex);
            ready = true;
            cv.notify_all();
            cv.wait(lock, [&] {
                return leave;
            });
        });
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] {
                return ready;
            });
        }
        const int close_rc = dlclose(handle);
        Dl_info info{};
        const bool still_mapped = dladdr(reinterpret_cast<void *>(touch), &info) != 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            leave = true;
        }
        cv.notify_all();
        survivor.join();
        if (close_rc != 0 || still_mapped) return 12;
        const size_t after = available_keys();
        if (after != before) {
            std::fprintf(stderr, "STRACE reload round %d: available keys before=%zu after=%zu\n", round, before, after);
            return 13;
        }
        if (!held) return 14;
    }
    return 0;
}
}  // namespace

TEST(StraceLifetimeDeathTest, UnloadBeforeThreadExitAndRepeatedReload) {
    ASSERT_EXIT(_exit(reload_with_surviving_threads()), ::testing::ExitedWithCode(0), "");
}

TEST(StraceLifetimeDeathTest, KeyExhaustionDisablesTracing) {
    ASSERT_EXIT(
        {
            std::vector<pthread_key_t> keys;
            pthread_key_t key;
            while (pthread_key_create(&key, nullptr) == 0)
                keys.push_back(key);
            HostLogger::get_instance().set_level(simpler::log::LogLevel::NUL, true);
            {
                STRACE_CONTEXT(17, 19, 2);
                STRACE("exhausted");
            }
            for (auto value : keys)
                pthread_key_delete(value);
            _exit(simpler::strace::strace_state() == nullptr ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(StraceLifetime, ContextCrossesModulesAndRestoresAfterExceptions) {
    auto &owner = HostLogger::get_instance();
    owner.set_level(simpler::log::LogLevel::NUL, true);
    void *handle = dlopen(TEST_STRACE_CONSUMER_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << dlerror();
    auto bind = reinterpret_cast<int (*)(SimplerHostLogState *)>(dlsym(handle, "test_strace_bind"));
    auto context = reinterpret_cast<bool (*)(uint64_t, uint64_t, int)>(dlsym(handle, "test_strace_context"));
    auto allocate = reinterpret_cast<uint64_t (*)()>(dlsym(handle, "test_strace_allocate_inv"));
    ASSERT_NE(bind, nullptr);
    ASSERT_NE(context, nullptr);
    ASSERT_NE(allocate, nullptr);
    ASSERT_EQ(bind(owner.state()), 0);
    {
        STRACE_CONTEXT(71, 83, 2);
        EXPECT_TRUE(context(71, 83, 2));
        const uint64_t first = STRACE_ALLOC_INV();
        EXPECT_GT(first, 0u);
        EXPECT_GT(allocate(), first);
        EXPECT_TRUE(context(71, 83, 2));
        try {
            STRACE_CONTEXT(91, 93, 4);
            STRACE("nested");
            EXPECT_TRUE(context(91, 93, 5));
            throw 1;
        } catch (int) {}
        EXPECT_TRUE(context(71, 83, 2));
        bool isolated = false;
        std::thread other([&] {
            isolated = context(0, 0, 0);
        });
        other.join();
        EXPECT_TRUE(isolated);
    }
    EXPECT_TRUE(context(0, 0, 0));
    EXPECT_EQ(dlclose(handle), 0);
}
