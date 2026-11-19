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
#include <atomic>
#include <cerrno>
#include <csignal>
#include <memory>
#include <future>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <pthread.h>
#include <gtest/gtest.h>
#include "runtime_module.h"

namespace {
size_t available_keys() {
    std::vector<pthread_key_t> keys;
    pthread_key_t key;
    int error;
    while ((error = pthread_key_create(&key, nullptr)) == 0)
        keys.push_back(key);
    EXPECT_EQ(error, EAGAIN);
    for (auto owned : keys)
        EXPECT_EQ(pthread_key_delete(owned), 0);
    return keys.size();
}
}  // namespace

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

TEST(HbgRecorderLifetime, LoadDuringCloseGetsAFreshPool) {
    auto owner = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
    auto start = owner->required<bool (*)(void (*)(void *), void *)>("test_recorder_start");
    auto prewarm = owner->required<bool (*)()>("test_recorder_prewarm");
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
    ASSERT_TRUE(start(
        [](void *p) {
            auto &job = *static_cast<Job *>(p);
            job.entered.set_value();
            job.release.wait();
            job.wait();
        },
        &job
    ));
    job.entered.get_future().wait();
    auto closing = std::async(std::launch::async, [&] {
        owner.reset();
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (prewarm() && std::chrono::steady_clock::now() < deadline) {}
    ASSERT_FALSE(prewarm());
    std::promise<void> started;
    auto reload = std::async(std::launch::async, [&] {
        started.set_value();
        RuntimeModule fresh(HBG_RECORDER_FIXTURE_PATH);
        return fresh.required<bool (*)()>("test_recorder_prewarm")();
    });
    started.get_future().wait();
    release.set_value();
    closing.get();
    EXPECT_TRUE(reload.get());
}

TEST(HbgRecorderLifetime, RepeatedReloadReturnsKeyBudget) {
    const auto before = available_keys();
    for (int round = 0; round < 10; ++round) {
        RuntimeModule owner(HBG_RECORDER_FIXTURE_PATH);
        ASSERT_TRUE(owner.required<bool (*)()>("test_recorder_prewarm")());
    }
    EXPECT_EQ(available_keys(), before);
}

TEST(HbgRecorderLifetime, PartialHostServicePairRejectsLoad) {
    EXPECT_THROW(RuntimeModule{HBG_RECORDER_ACQUIRE_ONLY_FIXTURE_PATH}, std::runtime_error);
}

TEST(HbgRecorderLifetime, FailedHostServiceAcquireRejectsLoad) {
    EXPECT_THROW(RuntimeModule{HBG_RECORDER_FAILED_ACQUIRE_FIXTURE_PATH}, std::runtime_error);
}

TEST(HbgRecorderLifetimeDeathTest, LastOwnerWaitsForRunningJobBeforeUnloading) {
    ASSERT_EXIT(
        {
            auto owner = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto start = owner->required<bool (*)(void (*)(void *), void *)>("test_recorder_start");
            auto wait = owner->required<void (*)()>("test_recorder_wait");
            auto prewarm = owner->required<bool (*)()>("test_recorder_prewarm");
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
            // Shutdown closing admission is the explicit event; the gated job
            // then proves release cannot complete. Do not infer the block from
            // elapsed time: an unscheduled closer would look identical.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (prewarm() && std::chrono::steady_clock::now() < deadline) {}
            if (prewarm()) _exit(2);
            if (closing.wait_for(std::chrono::seconds(0)) != std::future_status::timeout) _exit(3);
            release.set_value();
            closing.get();
            Dl_info info{};
            if (dladdr(reinterpret_cast<void *>(start), &info) != 0) _exit(4);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(HbgRecorderLifetimeDeathTest, AcquireWaitsForClosingPool) {
    ASSERT_EXIT(
        {
            void *loader = dlopen(HBG_RECORDER_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
            if (!loader) _exit(1);
            auto owner = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto start = owner->required<bool (*)(void (*)(void *), void *)>("test_recorder_start");
            auto prewarm = owner->required<bool (*)()>("test_recorder_prewarm");
            auto wait = owner->required<void (*)()>("test_recorder_wait");
            auto acquire = owner->required<int (*)()>("simpler_runtime_acquire_host_services");
            auto release = owner->required<void (*)()>("simpler_runtime_release_host_services");
            auto acquire_waiting = owner->required<int (*)()>("test_recorder_acquire_waiting");
            struct Job {
                std::promise<void> entered;
                std::shared_future<void> release;
                void (*wait)();
            };
            std::promise<void> release_job;
            Job job;
            job.release = release_job.get_future().share();
            job.wait = wait;
            if (!start(
                    [](void *p) {
                        auto &job = *static_cast<Job *>(p);
                        job.entered.set_value();
                        job.release.wait();
                        job.wait();
                    },
                    &job
                ))
                _exit(2);
            job.entered.get_future().wait();
            auto closing = std::async(std::launch::async, [&] {
                owner.reset();
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (prewarm() && std::chrono::steady_clock::now() < deadline) {}
            if (prewarm()) _exit(3);
            std::atomic<int> returned{0};
            std::promise<int> acquired;
            auto waiter = std::async(std::launch::async, [&] {
                const int rc = acquire();
                returned.store(1, std::memory_order_release);
                acquired.set_value(rc);
            });
            // The fixture's observer is set inside the production acquire right
            // before it waits; seeing it proves the closer is busy, not merely
            // unscheduled.
            while (acquire_waiting() == 0 && std::chrono::steady_clock::now() < deadline) {}
            if (acquire_waiting() == 0) _exit(4);
            if (returned.load(std::memory_order_acquire) != 0) _exit(5);
            release_job.set_value();
            closing.get();
            if (acquired.get_future().get() != 0) _exit(6);
            waiter.get();
            release();
            if (dlclose(loader) != 0) _exit(7);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(HbgRecorderLifetimeDeathTest, ShutdownFromRecorderWorkerTerminates) {
    ASSERT_EXIT(
        {
            auto owner = std::make_unique<RuntimeModule>(HBG_RECORDER_FIXTURE_PATH);
            auto start = owner->required<bool (*)(void (*)(void *), void *)>("test_recorder_start");
            auto shutdown = owner->required<void (*)()>("test_recorder_shutdown");
            struct Job {
                void (*shutdown)();
            };
            Job job{shutdown};
            if (!start(
                    [](void *p) {
                        static_cast<Job *>(p)->shutdown();
                    },
                    &job
                ))
                _exit(1);
            owner->required<void (*)()>("test_recorder_wait")();
            _exit(0);
        },
        ::testing::KilledBySignal(SIGABRT), ""
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
