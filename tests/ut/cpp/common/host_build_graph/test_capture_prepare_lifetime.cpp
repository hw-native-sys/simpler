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

#include <cerrno>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <dlfcn.h>
#include <exception>
#include <unistd.h>
#include <gtest/gtest.h>

TEST(CapturePrepareLifetimeDeathTest, CallingThreadSurvivesModuleUnload) {
    ASSERT_EXIT(
        {
            void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
            if (!module) _exit(1);
            auto touch = reinterpret_cast<void (*)()>(dlsym(module, "touch_capture_prepare"));
            if (!touch) _exit(2);
            std::mutex mutex;
            std::condition_variable cv;
            bool touched = false;
            bool leave = false;
            std::thread caller([&] {
                touch();
                std::unique_lock<std::mutex> lock(mutex);
                touched = true;
                cv.notify_all();
                cv.wait(lock, [&] {
                    return leave;
                });
            });
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] {
                    return touched;
                });
            }
            if (dlclose(module) != 0) _exit(3);
            Dl_info info{};
            const bool mapped = dladdr(reinterpret_cast<void *>(touch), &info) != 0;
            {
                std::scoped_lock lock(mutex);
                leave = true;
            }
            cv.notify_all();
            caller.join();
            _exit(mapped ? 4 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

// Exercise ordinary process teardown too, so LeakSanitizer observes scope cleanup.
TEST(CapturePrepareLifetime, RepeatedScopesReleaseStorage) {
    void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(module, nullptr);
    auto touch = reinterpret_cast<void (*)()>(dlsym(module, "touch_capture_prepare"));
    ASSERT_NE(touch, nullptr);
    for (int i = 0; i < 100; ++i)
        touch();
    EXPECT_EQ(dlclose(module), 0);
    Dl_info info{};
    EXPECT_EQ(dladdr(reinterpret_cast<void *>(touch), &info), 0);
}

#if defined(__linux__)
TEST(CapturePrepareLifetime, FailedKeyCreationRollsBack) {
    for (int position : {1, 2}) {
        void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(module, nullptr);
        auto probe = reinterpret_cast<int (*)(int, bool)>(dlsym(module, "capture_tls_probe"));
        auto keys = reinterpret_cast<int (*)()>(dlsym(module, "capture_live_keys"));
        ASSERT_NE(probe, nullptr);
        ASSERT_NE(keys, nullptr);
        const int before = keys();
        EXPECT_EQ(probe(position, false), EAGAIN);
        EXPECT_EQ(keys(), before);
        EXPECT_EQ(dlclose(module), 0);
    }
}

TEST(CapturePrepareLifetime, FailedBindingDoesNotPublishSession) {
    void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(module, nullptr);
    auto probe = reinterpret_cast<int (*)(int, bool)>(dlsym(module, "capture_tls_probe"));
    ASSERT_NE(probe, nullptr);
    EXPECT_EQ(probe(0, true), ENOMEM);
    EXPECT_EQ(probe(0, false), 0);
    EXPECT_EQ(dlclose(module), 0);
}

TEST(CapturePrepareLifetime, ReloadReturnsKeyBudgetWhileCallerSurvives) {
    auto available_keys = [] {
        std::vector<pthread_key_t> keys;
        pthread_key_t key;
        int error;
        while ((error = pthread_key_create(&key, nullptr)) == 0)
            keys.push_back(key);
        EXPECT_EQ(error, EAGAIN);
        for (auto owned : keys)
            EXPECT_EQ(pthread_key_delete(owned), 0);
        return keys.size();
    };
    const auto before = available_keys();
    for (int round = 0; round < 100; ++round) {
        void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
        ASSERT_NE(module, nullptr);
        auto touch = reinterpret_cast<void (*)()>(dlsym(module, "touch_capture_prepare"));
        ASSERT_NE(touch, nullptr);
        touch();
        EXPECT_EQ(dlclose(module), 0);
        EXPECT_EQ(available_keys(), before) << round;
    }
}

TEST(CapturePrepareLifetime, FailedArmingPreservesActiveSession) {
    void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(module, nullptr);
    auto probe = reinterpret_cast<bool (*)()>(dlsym(module, "capture_failed_arming_preserves_state"));
    ASSERT_NE(probe, nullptr);
    EXPECT_TRUE(probe());
    EXPECT_EQ(dlclose(module), 0);
}

TEST(CapturePrepareLifetimeDeathTest, FailedClearCannotReturnWithDanglingSelector) {
    ASSERT_EXIT(
        {
            void *module = dlopen(CONTEXT_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
            if (!module) _exit(1);
            auto probe = reinterpret_cast<void (*)()>(dlsym(module, "capture_fail_clear"));
            if (!probe) _exit(2);
            std::set_terminate([] {
                _exit(73);
            });
            probe();
        },
        ::testing::ExitedWithCode(73), ""
    );
}
#endif
