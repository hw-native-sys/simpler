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
#include <mutex>
#include <cstdint>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <pthread.h>
#include <gtest/gtest.h>

namespace {
size_t available_keys() {
    std::vector<pthread_key_t> keys;
    pthread_key_t key;
    while (pthread_key_create(&key, nullptr) == 0)
        keys.push_back(key);
    for (auto owned : keys)
        pthread_key_delete(owned);
    return keys.size();
}
}  // namespace

TEST(ModuleThreadValueLifetime, ReloadReclaimsKeysWhileCallerSurvives) {
    // libc++'s first std::thread initializes a process-lifetime pthread key.
    std::thread([] {}).join();
    const size_t before = available_keys();
    for (int round = 0; round < 8; ++round) {
        SCOPED_TRACE(round);
        void *module = dlopen(TEST_MODULE_THREAD_VALUE_PATH, RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(module, nullptr) << dlerror();
        auto touch = reinterpret_cast<bool (*)(uintptr_t, uintptr_t)>(dlsym(module, "test_module_value"));
        ASSERT_NE(touch, nullptr);
        EXPECT_TRUE(touch(17, 31));
        EXPECT_TRUE(touch(31, 43));
        bool isolated = false, ready = false, leave = false;
        std::mutex mutex;
        std::condition_variable cv;
        std::thread other([&] {
            isolated = touch(17, 59);
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
        EXPECT_TRUE(isolated);
        EXPECT_TRUE(touch(43, 71));
        EXPECT_EQ(dlclose(module), 0);
        Dl_info info{};
        EXPECT_EQ(dladdr(reinterpret_cast<void *>(touch), &info), 0);
        EXPECT_EQ(available_keys(), before);
        {
            std::lock_guard<std::mutex> lock(mutex);
            leave = true;
        }
        cv.notify_all();
        other.join();
        EXPECT_EQ(available_keys(), before);
    }
}

TEST(ModuleThreadValueLifetime, ModulesKeepIndependentKeysAndDefaults) {
    const size_t before = available_keys();
    void *first = dlopen(TEST_MODULE_THREAD_VALUE_PATH, RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(first, nullptr) << dlerror();
    void *second = dlopen(TEST_MODULE_THREAD_VALUE_PEER_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(second, nullptr) << dlerror();
    auto touch_first = reinterpret_cast<bool (*)(uintptr_t, uintptr_t)>(dlsym(first, "test_module_value"));
    auto touch_second = reinterpret_cast<bool (*)(uintptr_t, uintptr_t)>(dlsym(second, "test_module_value"));
    ASSERT_NE(touch_first, nullptr);
    ASSERT_NE(touch_second, nullptr);
    EXPECT_TRUE(touch_first(17, 31));
    EXPECT_TRUE(touch_second(17, 43));
    EXPECT_EQ(dlclose(first), 0);
    EXPECT_TRUE(touch_second(43, 59));
    EXPECT_EQ(dlclose(second), 0);
    EXPECT_EQ(available_keys(), before);
}
