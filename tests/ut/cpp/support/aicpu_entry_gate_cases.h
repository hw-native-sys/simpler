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

#pragma once
#include <gtest/gtest.h>
#include <atomic>
#include <array>
#include <chrono>
#include <future>
#include <thread>
#include <vector>
#include "aicpu/native_invocation_completion.h"
#include "aicpu/platform_aicpu_affinity.h"

namespace entry_gate_test {
inline std::atomic<int> initialized{0};
inline std::atomic<bool> initializer_entered{false};
inline std::atomic<bool> release_initializer{false};
inline int entry_state = -1;
inline void initialize_entry() {
    ++initialized;
    initializer_entered.store(true, std::memory_order_release);
    while (!release_initializer.load(std::memory_order_acquire)) {}
    entry_state = 0;
}
using Gate = bool (*)(int, int, AicpuEntryInitializer);
inline void check_cycles(Gate gate) {
    using namespace std::chrono_literals;
    initialized = 0;
    for (int round = 0; round < 3; ++round) {
        initializer_entered = false;
        release_initializer = false;
        entry_state = 99;
        std::vector<std::future<bool>> threads;
        auto enter = [&] {
            const bool survives = gate(2, 3, initialize_entry);
            if (survives) EXPECT_EQ(entry_state, 0);
            platform_aicpu_affinity_bind_thread(nullptr);
            return survives;
        };
        threads.push_back(std::async(std::launch::async, enter));
        threads.push_back(std::async(std::launch::async, enter));
        EXPECT_EQ(threads[0].wait_for(20ms), std::future_status::timeout);
        EXPECT_FALSE(initializer_entered.load());
        threads.push_back(std::async(std::launch::async, enter));
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (!initializer_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {}
        EXPECT_TRUE(initializer_entered.load());
        for (auto &thread : threads)
            EXPECT_EQ(thread.wait_for(0ms), std::future_status::timeout);
        release_initializer.store(true, std::memory_order_release);
        int survivors = 0;
        for (auto &thread : threads)
            survivors += thread.get();
        EXPECT_EQ(survivors, 2);
        EXPECT_EQ(initialized.load(), round + 1);
    }
}
inline bool filter(int logical, int launched, AicpuEntryInitializer init) {
    const int32_t cpus[] = {0, 1};
    return platform_aicpu_affinity_gate_filter(cpus, logical, launched, init);
}
}  // namespace entry_gate_test

namespace entry_gate_test {
using NativeGate = AicpuEntryGateResult (*)(int32_t, int32_t, AicpuEntryGateInitializer);
inline AicpuEntryGateResult native_filter(int32_t logical, int32_t launched, AicpuEntryGateInitializer initialize) {
    const int32_t cpus[] = {0, 1, 2, 3};
    return platform_aicpu_entry_gate_filter(cpus, logical, launched, initialize);
}
struct NativeState {
    NativeInvocationCompletion completion;
    int initializations{0};
    int32_t init_status{0};
};
inline int32_t initialize_native(void *context) {
    auto &state = *static_cast<NativeState *>(context);
    ++state.initializations;
    EXPECT_TRUE(state.completion.begin(6));
    EXPECT_TRUE(
        state.completion.accept_runtime_terminal({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0)
    );
    return state.init_status;
}
inline void check_native(NativeGate gate, bool invalid_participant, int32_t init_status) {
    NativeState state;
    state.init_status = init_status;
    for (int round = 0; round < 3; ++round) {
        std::atomic<int> published{0};
        std::atomic<int> returned{0};
        std::atomic<int> filtered_last{-1};
        std::array<AicpuEntryGateResult, 6> entries{};
        std::array<int32_t, 6> returns{};
        std::vector<std::thread> threads;
        for (int i = 0; i < 6; ++i) {
            threads.emplace_back([&, i] {
                auto result = gate(invalid_participant && i == 1 ? 0 : 4, 6, {initialize_native, &state});
                entries[i] = result;
                EXPECT_EQ(state.initializations, round + 1);
                int expected = -1;
                const bool last = !result.execute && filtered_last.compare_exchange_strong(expected, i);
                if (last) {
                    while (returned.load(std::memory_order_acquire) != 5)
                        std::this_thread::yield();
                }
                returns[i] = state.completion.finish(
                    result.callback_index, result.status, [&](RunTerminalSelection terminal, const void *, size_t) {
                        EXPECT_TRUE(last);
                        EXPECT_EQ(terminal.code, init_status != 0 ? init_status : invalid_participant ? -1 : 0);
                        published.fetch_add(1);
                        return NativeTerminalPublication::Published;
                    }
                );
                returned.fetch_add(1, std::memory_order_release);
                platform_aicpu_affinity_bind_thread(nullptr);
            });
        }
        for (auto &thread : threads)
            thread.join();
        std::array<bool, 6> seen{};
        int survivors = 0;
        for (int i = 0; i < 6; ++i) {
            ASSERT_GE(entries[i].callback_index, 0);
            ASSERT_LT(entries[i].callback_index, 6);
            EXPECT_FALSE(seen[entries[i].callback_index]);
            seen[entries[i].callback_index] = true;
            survivors += entries[i].execute;
            EXPECT_EQ(entries[i].status, init_status != 0 ? init_status : invalid_participant ? -1 : 0);
            if (i != filtered_last.load()) EXPECT_EQ(returns[i], 0);
        }
        EXPECT_EQ(survivors, invalid_participant || init_status != 0 ? 0 : 4);
        EXPECT_EQ(published.load(), 1);
        EXPECT_EQ(state.initializations, round + 1);
    }
}
}  // namespace entry_gate_test
