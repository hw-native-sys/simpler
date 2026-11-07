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
#include <thread>
#include <unistd.h>
#include <sched.h>
#include <gtest/gtest.h>
#include "aicpu/platform_aicpu_affinity.h"
#include <pthread.h>
namespace {
std::atomic<bool> fail_binding{false};
std::atomic<bool> fail_creation{false};
std::atomic<bool> fail_nonnull_binding{false};
}  // namespace
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __real_pthread_key_create(pthread_key_t *, void (*)(void *));
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (fail_binding.exchange(false) || (value != nullptr && fail_nonnull_binding.exchange(false))) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
extern "C" int __wrap_pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
    if (fail_creation.exchange(false)) return EAGAIN;
    return __real_pthread_key_create(key, destructor);
}
TEST(AicpuAffinityTls, BoundCellKeepsThreadsIndependent) {
    int32_t index = -1;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&index), 0);
    platform_aicpu_affinity_set_thread_idx(3);
    std::thread other([] {
        EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
        int32_t local = 7;
        ASSERT_EQ(platform_aicpu_affinity_bind_thread(&local), 0);
        EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 7);
        platform_aicpu_affinity_set_thread_idx(8);
        EXPECT_EQ(local, 8);
    });
    other.join();
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), 3);
    EXPECT_EQ(index, 3);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}
TEST(AicpuAffinityTlsDeathTest, BindingFailureRejectsWholeCohortWithoutHanging) {
    ASSERT_EXIT(
        {
            fail_binding.store(true);
            int32_t allowed[2];
            allowed[0] = sched_getcpu();
            allowed[1] = allowed[0];
            std::atomic<int> admitted{0};
            int errors[2]{};
            std::thread a([&] {
                admitted += platform_aicpu_affinity_gate_filter(allowed, 2, 2, nullptr, &errors[0]);
            });
            std::thread b([&] {
                admitted += platform_aicpu_affinity_gate_filter(allowed, 2, 2, nullptr, &errors[1]);
            });
            a.join();
            b.join();
            _exit(admitted == 0 && errors[0] == ENOMEM && errors[1] == ENOMEM ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(AicpuAffinityTlsDeathTest, KeyFailureIsReportedWithoutInvalidKeyUse) {
    ASSERT_EXIT(
        {
            fail_creation.store(true);
            int32_t cell = -1;
            _exit(platform_aicpu_affinity_bind_thread(&cell) == EAGAIN ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(AicpuAffinityTls, RepeatedCohortsKeepDistinctSlots) {
    const int32_t allowed[] = {sched_getcpu(), sched_getcpu()};
    for (int round = 0; round < 20; ++round) {
        std::atomic<int> mask{0};
        std::thread a([&] {
            if (platform_aicpu_affinity_gate_filter(allowed, 2, 2))
                mask.fetch_or(1 << platform_aicpu_affinity_thread_idx());
        });
        std::thread b([&] {
            if (platform_aicpu_affinity_gate_filter(allowed, 2, 2))
                mask.fetch_or(1 << platform_aicpu_affinity_thread_idx());
        });
        a.join();
        b.join();
        EXPECT_EQ(mask.load(), 3);
    }
}

TEST(AicpuAffinityTlsDeathTest, KeyCreationFailureCanRetryInSameModule) {
    ASSERT_EXIT(
        {
            int32_t cell = 7;
            fail_creation = true;
            if (platform_aicpu_affinity_bind_thread(&cell) != EAGAIN) _exit(1);
            if (platform_aicpu_affinity_bind_thread(&cell) != 0) _exit(2);
            if (platform_aicpu_affinity_thread_idx() != 7) _exit(3);
            _exit(platform_aicpu_affinity_bind_thread(nullptr) == 0 ? 0 : 4);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicpuAffinityTlsDeathTest, FailedCohortDoesNotPoisonNextLaunch) {
    ASSERT_EXIT(
        {
            const int32_t allowed[] = {sched_getcpu()};
            fail_binding = true;
            if (platform_aicpu_affinity_gate_filter(allowed, 1, 1)) _exit(1);
            if (!platform_aicpu_affinity_gate_filter(allowed, 1, 1)) _exit(2);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicpuAffinityTlsDeathTest, UnboundSetterExplainsFatalContractViolation) {
    ASSERT_DEATH(platform_aicpu_affinity_set_thread_idx(3), "requires a bound cell");
}

TEST(AicpuAffinityTls, NativeFailurePreservesCallbackIdentityAndNextLaunchRecovers) {
    const int32_t allowed[] = {sched_getcpu(), sched_getcpu()};
    for (int round = 0; round < 3; ++round) {
        std::atomic<int> initializations{0};
        AicpuEntryGateResult results[2];
        int32_t indices[2]{};
        fail_binding = round == 0;
        fail_nonnull_binding = round == 1;
        auto enter = [&](int i) {
            results[i] = platform_aicpu_entry_gate_filter(
                allowed, 2, 2,
                {[](void *context) -> int32_t {
                     ++*static_cast<std::atomic<int> *>(context);
                     return 0;
                 },
                 &initializations}
            );
            indices[i] = platform_aicpu_affinity_thread_idx();
        };
        std::thread a(enter, 0), b(enter, 1);
        a.join();
        b.join();
        EXPECT_EQ(initializations, 1);
        for (int32_t idx : indices) {
            if (round < 2) EXPECT_EQ(idx, -1);
            else EXPECT_GE(idx, 0);
        }
        EXPECT_NE(results[0].callback_index, results[1].callback_index);
        for (const auto &result : results) {
            EXPECT_GE(result.callback_index, 0);
            EXPECT_LT(result.callback_index, 2);
            EXPECT_EQ(result.status, round < 2 ? -1 : 0);
            EXPECT_EQ(result.execute, round == 2);
        }
    }
}

TEST(AicpuAffinityTls, ParameterRejectionInvalidatesPreviousIdentity) {
    const int32_t allowed[] = {sched_getcpu()};
    ASSERT_TRUE(platform_aicpu_affinity_gate_filter(allowed, 1, 1));
    ASSERT_EQ(platform_aicpu_affinity_thread_idx(), 0);
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(allowed, 0, 1));
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
    ASSERT_TRUE(platform_aicpu_affinity_gate_filter(allowed, 1, 1));
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(allowed, 1, 0));
    EXPECT_EQ(platform_aicpu_affinity_thread_idx(), -1);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}

TEST(AicpuAffinityTls, ExcessParticipantInvalidatesPreviousIdentity) {
    const int32_t allowed[] = {sched_getcpu()};
    bool checked = false;
    auto result = platform_aicpu_entry_gate_filter(
        allowed, 1, 1,
        {[](void *context) -> int32_t {
             int32_t previous = 7;
             if (platform_aicpu_affinity_bind_thread(&previous) != 0) return -1;
             const int32_t cpus[] = {sched_getcpu()};
             int error = 0;
             const bool accepted = platform_aicpu_affinity_gate_filter(cpus, 1, 1, nullptr, &error);
             *static_cast<bool *>(context) = !accepted && error == EINVAL && platform_aicpu_affinity_thread_idx() == -1;
             // The temporary cell must not outlive this initializer.
             return platform_aicpu_affinity_bind_thread(nullptr);
         },
         &checked}
    );
    EXPECT_TRUE(checked);
    EXPECT_TRUE(result.execute);
    EXPECT_EQ(result.status, 0);
    EXPECT_EQ(platform_aicpu_affinity_bind_thread(nullptr), 0);
}
