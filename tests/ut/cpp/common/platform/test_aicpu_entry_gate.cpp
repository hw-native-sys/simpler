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

#include "support/aicpu_entry_gate_cases.h"
#include "host/raii_scope_guard.h"
#include "host/sim_native_launch.h"
#include "host/sim_thread_start_gate.h"
#include <system_error>
TEST(AicpuEntryGate, BasicAndFilterInitializeBeforeSurvivorsAcrossSerialInvocations) {
    entry_gate_test::check_cycles(platform_aicpu_affinity_gate);
    entry_gate_test::check_cycles(entry_gate_test::filter);
}
TEST(AicpuEntryGate, SingleCallbackStillInitializes) {
    entry_gate_test::initialized = 0;
    entry_gate_test::release_initializer = true;
    EXPECT_TRUE(platform_aicpu_affinity_gate(1, 1, entry_gate_test::initialize_entry));
    EXPECT_EQ(entry_gate_test::initialized.load(), 1);
}
TEST(AicpuEntryGate, InvalidInputNeverInvokesInitializer) {
    entry_gate_test::initialized = 0;
    EXPECT_FALSE(platform_aicpu_affinity_gate(2, 1, entry_gate_test::initialize_entry));
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(nullptr, 2, 3, entry_gate_test::initialize_entry));
    EXPECT_EQ(entry_gate_test::initialized.load(), 0);
}

TEST(AicpuEntryGate, NativeIndicesIncludeFilteredCallbacksAndFinishExactlyOnce) {
    entry_gate_test::check_native(platform_aicpu_entry_gate, false, 0);
}
TEST(AicpuEntryGate, OneInvalidParticipantRejectsEveryoneBeforeWork) {
    entry_gate_test::check_native(platform_aicpu_entry_gate, true, 0);
}
TEST(AicpuEntryGate, InitializationFailureReachesEveryCallback) {
    entry_gate_test::check_native(platform_aicpu_entry_gate, false, -29);
}
TEST(AicpuEntryGate, UntrustedCountDoesNotInitializeOrClaimIdentity) {
    entry_gate_test::NativeState state;
    const auto result =
        platform_aicpu_entry_gate(4, MAX_GATE_THREADS + 1, {entry_gate_test::initialize_native, &state});
    EXPECT_EQ(result.callback_index, -1);
    EXPECT_NE(result.status, 0);
    EXPECT_FALSE(result.execute);
    EXPECT_EQ(state.initializations, 0);
}

TEST(AicpuEntryGate, SimFilterAlsoProvidesNativeIdentitiesAndRejectsInvalidParticipants) {
    entry_gate_test::check_native(entry_gate_test::native_filter, false, 0);
    entry_gate_test::check_native(entry_gate_test::native_filter, true, 0);
    entry_gate_test::check_native(entry_gate_test::native_filter, false, -29);
}

TEST(AicpuEntryGate, IndependentSimCohortsDoNotShareIndicesOrInitializers) {
    for (int round = 0; round != 20; ++round) {
        SimNativeEntryGate first(3), second(5);
        std::atomic<int> initialized[2]{};
        std::atomic<int> survivors[2]{};
        std::atomic<unsigned> indices[2]{};
        std::vector<std::thread> threads;
        for (int i = 0; i != 8; ++i)
            threads.emplace_back([&, i] {
                const int cohort = i % 2 == 0 && i < 6 ? 0 : 1;
                auto &gate = cohort == 0 ? first : second;
                const auto init = [](void *p) -> int32_t {
                    static_cast<std::atomic<int> *>(p)->fetch_add(1);
                    return 0;
                };
                const auto result = gate.enter(cohort == 0 ? 2 : 4, {init, &initialized[cohort]});
                EXPECT_GE(result.callback_index, 0);
                indices[cohort].fetch_or(1u << result.callback_index);
                if (result.execute) ++survivors[cohort];
                platform_aicpu_affinity_bind_thread(nullptr);
            });
        for (auto &thread : threads)
            thread.join();
        EXPECT_EQ(initialized[0], 1);
        EXPECT_EQ(initialized[1], 1);
        EXPECT_EQ(survivors[0], 2);
        EXPECT_EQ(survivors[1], 4);
        EXPECT_EQ(indices[0], 7u);
        EXPECT_EQ(indices[1], 31u);
    }
}

TEST(AicpuEntryGate, PartialSimThreadCreationCancelsEveryStartedWorkerBeforeDeviceWork) {
    for (int fail_at = 0; fail_at <= 8; ++fail_at) {
        SimThreadStartGate start;
        SimNativeEntryGate entry(4);
        std::vector<std::thread> threads;
        std::atomic<int> executed{0};
        const auto join = [&] {
            for (auto &thread : threads)
                if (thread.joinable()) thread.join();
        };
        int rc = 0;
        {
            auto startup_cleanup = RAIIScopeGuard([&] {
                start.cancel();
                join();
            });
            try {
                for (int i = 0; i != 8; ++i) {
                    if (i == fail_at)
                        throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
                    threads.emplace_back([&, i] {
                        if (!start.wait()) return;
                        if (i >= 4) {
                            const auto init = [](void *) -> int32_t {
                                return 0;
                            };
                            EXPECT_TRUE(entry.enter(4, {init, nullptr}).execute);
                            platform_aicpu_affinity_bind_thread(nullptr);
                        }
                        ++executed;
                    });
                }
            } catch (...) {
                rc = -1;
            }
            EXPECT_EQ(executed, 0);
            if (rc == 0) {
                start.release();
                startup_cleanup.dismiss();
            }
        }
        EXPECT_EQ(rc, fail_at == 8 ? 0 : -1);
        join();
        EXPECT_EQ(executed, fail_at == 8 ? 8 : 0);
    }
}
