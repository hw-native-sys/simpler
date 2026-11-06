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

#include <dlfcn.h>
#include <unistd.h>
#include <memory>
#include <future>
#include <chrono>
#include <gtest/gtest.h>
#include "runtime_module.h"

TEST(HbgRecorderLifetimeDeathTest, LastModuleOwnerStopsPoolBeforeUnload) {
    ASSERT_EXIT(
        {
            for (int round = 0; round < 3; ++round) {
                auto first = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
                auto second = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
                auto prewarm = first->required<bool (*)()>("test_recorder_prewarm");
                if (!prewarm()) _exit(1);
                first.reset();
                if (!prewarm()) _exit(2);
                second.reset();
                Dl_info info{};
                if (dladdr(reinterpret_cast<void *>(prewarm), &info) != 0) _exit(3);
            }
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(HbgRecorderLifetime, SurvivingLoaderReferenceCanAcquireANewPool) {
    void *loader = dlopen(HBG_RECORDER_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(loader, nullptr);
    for (int round = 0; round < 3; ++round) {
        RuntimeModule owner(HBG_RECORDER_FIXTURE_PATH);
        EXPECT_TRUE(owner.required<bool (*)()>("test_recorder_prewarm")());
    }
    EXPECT_EQ(dlclose(loader), 0);
}

TEST(HbgRecorderLifetimeDeathTest, LastOwnerWaitsForRunningJobBeforeUnloading) {
    ASSERT_EXIT(
        {
            auto owner = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto start = owner->required<bool (*)(void (*)(void *), void *)>("test_recorder_start");
            auto wait = owner->required<void (*)()>("test_recorder_wait");
            struct Job {
                std::promise<void> entered;
                std::shared_future<void> release;
                void (*wait)();
            };
            std::promise<void> release;
            Job job;
            job.release = release.get_future().share();
            job.wait = wait;
            if (!start(
                    [](void *p) {
                        auto &job = *static_cast<Job *>(p);
                        job.entered.set_value();
                        job.release.wait();
                        job.wait();  // A worker must not wait for itself during shutdown.
                    },
                    &job
                ))
                _exit(1);
            job.entered.get_future().wait();
            auto closing = std::async(std::launch::async, [&] {
                owner.reset();
            });
            if (closing.wait_for(std::chrono::milliseconds(20)) != std::future_status::timeout) _exit(2);
            release.set_value();
            closing.get();
            Dl_info info{};
            if (dladdr(reinterpret_cast<void *>(start), &info) != 0) _exit(3);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(HbgRecorderLifetimeDeathTest, TerminalRetentionKeepsHostServicesAlive) {
    ASSERT_EXIT(
        {
            auto retained = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto other = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto prewarm = retained->required<bool (*)()>("test_recorder_prewarm");
            if (!prewarm()) _exit(1);
            retained->retain_until_process_exit();
            retained.reset();
            other.reset();
            if (!prewarm()) _exit(2);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
