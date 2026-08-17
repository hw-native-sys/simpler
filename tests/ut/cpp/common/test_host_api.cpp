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

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "common/host_api.h"

namespace {

struct FakeRunner {
    void record_pipeline_slot(uint32_t slot) {
        std::scoped_lock lock(mutex);
        pipeline_slots.push_back(slot);
    }

    void record_arena_bank(uint32_t bank) {
        std::scoped_lock lock(mutex);
        arena_banks.push_back(bank);
    }

    void record_metadata_slot(uint32_t slot) {
        std::scoped_lock lock(mutex);
        metadata_slots.push_back(slot);
    }

    std::mutex mutex;
    std::vector<uint32_t> pipeline_slots;
    std::vector<uint32_t> arena_banks;
    std::vector<uint32_t> metadata_slots;
};

void set_retained_temp_buffer(void *runner_ctx, uint32_t pipeline_slot, void *, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_pipeline_slot(pipeline_slot);
}

bool retained_temp_metadata_matches(void *runner_ctx, uint32_t pipeline_slot, const void *, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_metadata_slot(pipeline_slot);
    return true;
}

void set_retained_temp_metadata(void *runner_ctx, uint32_t pipeline_slot, const void *, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_metadata_slot(pipeline_slot);
}

int setup_static_arena(void *runner_ctx, uint32_t arena_bank, size_t, size_t, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_arena_bank(arena_bank);
    return 0;
}

const HostApiOps &fake_ops() {
    static const HostApiOps ops = []() {
        HostApiOps result{};
        result.set_retained_temp_buffer = set_retained_temp_buffer;
        result.retained_temp_metadata_matches = retained_temp_metadata_matches;
        result.set_retained_temp_metadata = set_retained_temp_metadata;
        result.setup_static_arena = setup_static_arena;
        return result;
    }();
    return ops;
}

class StartGate {
public:
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrivals_;
        if (arrivals_ == 2) {
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [this]() {
            return arrivals_ == 2;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int arrivals_{0};
};

bool all_equal(const std::vector<uint32_t> &values, uint32_t expected) {
    return std::all_of(values.begin(), values.end(), [expected](uint32_t value) {
        return value == expected;
    });
}

}  // namespace

TEST(HostApiTest, BoundRunnerSlotAndBankSurviveConcurrentCrossThreadCalls) {
    constexpr uint32_t kRunnerASlot = 1;
    constexpr uint32_t kRunnerABank = 3;
    constexpr uint32_t kRunnerBSlot = 7;
    constexpr uint32_t kRunnerBBank = 5;
    constexpr int kCallsPerThread = 128;

    FakeRunner runner_a;
    FakeRunner runner_b;
    const HostApi api_a(&runner_a, kRunnerASlot, kRunnerABank, &fake_ops());
    const HostApi api_b(&runner_b, kRunnerBSlot, kRunnerBBank, &fake_ops());
    StartGate start_gate;

    std::thread first([&]() {
        start_gate.arrive_and_wait();
        for (int i = 0; i < kCallsPerThread; ++i) {
            api_a.set_retained_temp_buffer(nullptr, 0);
            api_a.set_retained_temp_metadata(nullptr, 0);
            api_b.setup_static_arena(0, 0, 0);
        }
    });
    std::thread second([&]() {
        start_gate.arrive_and_wait();
        for (int i = 0; i < kCallsPerThread; ++i) {
            api_b.set_retained_temp_buffer(nullptr, 0);
            (void)api_b.retained_temp_metadata_matches("key", 3);
            api_a.setup_static_arena(0, 0, 0);
        }
    });

    first.join();
    second.join();

    EXPECT_EQ(runner_a.pipeline_slots.size(), kCallsPerThread);
    EXPECT_EQ(runner_a.arena_banks.size(), kCallsPerThread);
    EXPECT_EQ(runner_a.metadata_slots.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.pipeline_slots.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.arena_banks.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.metadata_slots.size(), kCallsPerThread);
    EXPECT_TRUE(all_equal(runner_a.pipeline_slots, kRunnerASlot));
    EXPECT_TRUE(all_equal(runner_a.arena_banks, kRunnerABank));
    EXPECT_TRUE(all_equal(runner_a.metadata_slots, kRunnerASlot));
    EXPECT_TRUE(all_equal(runner_b.pipeline_slots, kRunnerBSlot));
    EXPECT_TRUE(all_equal(runner_b.arena_banks, kRunnerBBank));
    EXPECT_TRUE(all_equal(runner_b.metadata_slots, kRunnerBSlot));
}
