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
#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/host_api.h"
#include "common/chip_swimlane_extension.h"

namespace {

struct FakeRunner {
    void record_pipeline_slot(uint32_t slot) {
        std::lock_guard<std::mutex> lock(mutex);
        pipeline_slots.push_back(slot);
    }

    void record_arena_bank(uint32_t bank) {
        std::lock_guard<std::mutex> lock(mutex);
        arena_banks.push_back(bank);
    }

    void record_bank_set(RegionBanks banks) {
        std::lock_guard<std::mutex> lock(mutex);
        bank_sets.push_back(banks);
    }

    void record_pooled_bank(uint32_t bank) {
        std::lock_guard<std::mutex> lock(mutex);
        pooled_banks.push_back(bank);
    }

    void record_run_epoch(uint64_t epoch) {
        std::lock_guard<std::mutex> lock(mutex);
        run_epochs.push_back(epoch);
    }

    std::mutex mutex;
    std::vector<uint32_t> pipeline_slots;
    std::vector<uint32_t> arena_banks;
    std::vector<uint64_t> run_epochs;
    std::vector<RegionBanks> bank_sets;
    std::vector<uint32_t> pooled_banks;
};

void set_retained_temp_buffer(void *runner_ctx, uint32_t pipeline_slot, void *, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_pipeline_slot(pipeline_slot);
}

int setup_static_arena(void *runner_ctx, uint32_t /*pipeline_slot*/, RegionBanks banks, size_t, size_t, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_arena_bank(banks.heap);
    return 0;
}

int setup_static_arena_banks(void *runner_ctx, uint32_t /*pipeline_slot*/, RegionBanks banks, size_t, size_t, size_t) {
    static_cast<FakeRunner *>(runner_ctx)->record_bank_set(banks);
    return 0;
}

void *acquire_pooled_bank(void *runner_ctx, uint32_t /*pipeline_slot*/, uint32_t bank) {
    static_cast<FakeRunner *>(runner_ctx)->record_pooled_bank(bank);
    return nullptr;
}

void *acquire_pooled_sm_or_image(void *runner_ctx, uint32_t bank) {
    static_cast<FakeRunner *>(runner_ctx)->record_pooled_bank(bank);
    return nullptr;
}

const HostApiOps &bank_routing_ops() {
    static const HostApiOps ops = []() {
        HostApiOps result{};
        result.setup_static_arena = setup_static_arena_banks;
        result.acquire_pooled_gm_heap = acquire_pooled_bank;
        result.acquire_pooled_gm_sm = acquire_pooled_sm_or_image;
        result.acquire_pooled_runtime_arena = acquire_pooled_sm_or_image;
        return result;
    }();
    return ops;
}

const void *get_run_result(void *runner_ctx, uint32_t, uint64_t run_epoch, size_t *bytes_out) {
    static_cast<FakeRunner *>(runner_ctx)->record_run_epoch(run_epoch);
    if (bytes_out != nullptr) *bytes_out = 0;
    return nullptr;
}

const HostApiOps &fake_ops() {
    static const HostApiOps ops = []() {
        HostApiOps result{};
        result.set_retained_temp_buffer = set_retained_temp_buffer;
        result.setup_static_arena = setup_static_arena;
        result.get_run_result = get_run_result;
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

template <typename T>
bool all_equal(const std::vector<T> &values, T expected) {
    return std::all_of(values.begin(), values.end(), [expected](T value) {
        return value == expected;
    });
}

bool throwing_extension_callback(void *, uint64_t, ChipSwimlaneExtensionSection, const char *, size_t) {
    throw std::runtime_error("publication failed");
}

/** Records the run epoch a publication was addressed with. */
uint64_t g_published_epoch = 0;

bool recording_extension_callback(void *, uint64_t run_epoch, ChipSwimlaneExtensionSection, const char *, size_t) {
    g_published_epoch = run_epoch;
    return true;
}

}  // namespace

TEST(HostApiTest, BoundRunnerSlotBankAndEpochSurviveConcurrentCrossThreadCalls) {
    constexpr uint32_t kRunnerASlot = 1;
    constexpr uint32_t kRunnerABank = 3;
    constexpr uint64_t kRunnerAEpoch = 41;
    constexpr uint32_t kRunnerBSlot = 7;
    constexpr uint32_t kRunnerBBank = 5;
    constexpr uint64_t kRunnerBEpoch = 97;
    constexpr int kCallsPerThread = 128;

    FakeRunner runner_a;
    FakeRunner runner_b;
    const HostApi api_a(&runner_a, kRunnerASlot, kRunnerABank, kRunnerAEpoch, &fake_ops());
    const HostApi api_b(&runner_b, kRunnerBSlot, kRunnerBBank, kRunnerBEpoch, &fake_ops());
    StartGate start_gate;

    std::thread first([&]() {
        start_gate.arrive_and_wait();
        for (int i = 0; i < kCallsPerThread; ++i) {
            api_a.set_retained_temp_buffer(nullptr, 0);
            api_b.setup_static_arena(0, 0, 0);
            size_t bytes = 0;
            api_a.run_result(&bytes);
        }
    });
    std::thread second([&]() {
        start_gate.arrive_and_wait();
        for (int i = 0; i < kCallsPerThread; ++i) {
            api_b.set_retained_temp_buffer(nullptr, 0);
            api_a.setup_static_arena(0, 0, 0);
            size_t bytes = 0;
            api_b.run_result(&bytes);
        }
    });

    first.join();
    second.join();

    EXPECT_EQ(runner_a.pipeline_slots.size(), kCallsPerThread);
    EXPECT_EQ(runner_a.arena_banks.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.pipeline_slots.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.arena_banks.size(), kCallsPerThread);
    EXPECT_TRUE(all_equal(runner_a.pipeline_slots, kRunnerASlot));
    EXPECT_TRUE(all_equal(runner_a.arena_banks, kRunnerABank));
    EXPECT_TRUE(all_equal(runner_b.pipeline_slots, kRunnerBSlot));
    EXPECT_TRUE(all_equal(runner_b.arena_banks, kRunnerBBank));
    EXPECT_EQ(runner_a.run_epochs.size(), kCallsPerThread);
    EXPECT_EQ(runner_b.run_epochs.size(), kCallsPerThread);
    EXPECT_TRUE(all_equal(runner_a.run_epochs, kRunnerAEpoch));
    EXPECT_TRUE(all_equal(runner_b.run_epochs, kRunnerBEpoch));
}

TEST(HostApiTest, RoutesEachPooledRegionToItsSelectedBank) {
    FakeRunner runner;
    const HostApi api(&runner, 2, RegionBanks{3, 4, 5}, 9, &bank_routing_ops());
    EXPECT_EQ(api.setup_static_arena(1, 2, 3), 0);
    (void)api.acquire_pooled_gm_heap();
    (void)api.acquire_pooled_gm_sm();
    (void)api.acquire_pooled_runtime_arena();

    ASSERT_EQ(runner.bank_sets.size(), 1u);
    EXPECT_EQ(runner.bank_sets[0].heap, 3u);
    EXPECT_EQ(runner.bank_sets[0].sm, 4u);
    EXPECT_EQ(runner.bank_sets[0].image, 5u);
    EXPECT_EQ(runner.pooled_banks, (std::vector<uint32_t>{3, 4, 5}));
}

TEST(HostApiTest, RunResultIsAbsentWhenTheBackendProvidesNoRegion) {
    const HostApiOps ops{};
    const HostApi api(nullptr, 0, 0, 1, &ops);

    size_t bytes = 123;
    EXPECT_EQ(api.run_result(&bytes), nullptr);
    EXPECT_EQ(bytes, 0u);
}

TEST(HostApiTest, PublicationExceptionsBecomeFailureResults) {
    HostApiOps ops{};
    ops.publish_chip_swimlane_extension = throwing_extension_callback;
    const HostApi api(nullptr, 0, 0, 0, &ops);

    EXPECT_FALSE(api.publish_chip_swimlane_extension(ChipSwimlaneExtensionSection::SchedulerRecords, "{}", 2));
}

// A publication is addressed by the caller's own run, not by the runner's
// state: a retaining collector keeps one bucket per open run, and a predecessor
// publishes after its successor has been admitted.
TEST(HostApiTest, PublicationCarriesTheCallersOwnRunEpoch) {
    constexpr uint64_t kEpoch = 4242;
    HostApiOps ops{};
    ops.publish_chip_swimlane_extension = recording_extension_callback;
    const HostApi api(nullptr, /*pipeline_slot=*/0, /*arena_bank=*/0, kEpoch, &ops);

    g_published_epoch = 0;
    EXPECT_TRUE(api.publish_chip_swimlane_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]", 2));
    EXPECT_EQ(g_published_epoch, kEpoch);
}

TEST(ChipSwimlaneExtensionTest, ExposesOnlyFixedSectionNamesAndShapes) {
    EXPECT_STREQ(chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::AicoreTasks), "aicore_tasks");
    EXPECT_STREQ(
        chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerRecords), "scheduler_records"
    );
    EXPECT_STREQ(chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::SchedulerTasks), "scheduler_tasks");
    EXPECT_STREQ(
        chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::AicpuLifecycleRecords),
        "aicpu_lifecycle_records"
    );
    EXPECT_TRUE(chip_swimlane_extension_section_is_object(ChipSwimlaneExtensionSection::SchedulerTasks));
    EXPECT_TRUE(chip_swimlane_extension_section_is_object(ChipSwimlaneExtensionSection::SchedulerRecords));
    EXPECT_FALSE(chip_swimlane_extension_section_is_object(ChipSwimlaneExtensionSection::AicoreTasks));
    EXPECT_FALSE(chip_swimlane_extension_section_is_valid(ChipSwimlaneExtensionSection::Count));
    EXPECT_EQ(chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection::Count), nullptr);
    EXPECT_TRUE(chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection::SchedulerRecords, " {} "));
    EXPECT_TRUE(chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection::SchedulerTasks, " {} "));
    EXPECT_TRUE(chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection::AicoreTasks, " [] "));
    EXPECT_FALSE(chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection::SchedulerRecords, "[]"));
    EXPECT_FALSE(chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection::AicoreTasks, "{}"));
}

TEST(HostApi, OrchestrationScratchIsIsolatedByRunnerAndPipelineSlot) {
    using ScratchSlots = std::array<std::shared_ptr<void>, 2>;
    ScratchSlots first{}, second{};
    HostApiOps ops{};
    ops.get_orchestration_scratch = [](void *ctx, uint32_t slot) {
        return (*static_cast<ScratchSlots *>(ctx))[slot];
    };
    ops.set_orchestration_scratch = [](void *ctx, uint32_t slot, std::shared_ptr<void> scratch) {
        (*static_cast<ScratchSlots *>(ctx))[slot] = std::move(scratch);
    };
    const HostApi a(&first, 0, 0, 1, &ops);
    const HostApi b(&first, 1, 0, 2, &ops);
    const HostApi other(&second, 0, 0, 1, &ops);
    auto value = std::make_shared<int>(7);
    std::weak_ptr<int> lifetime = value;
    a.set_orchestration_scratch(value);
    value.reset();
    EXPECT_FALSE(lifetime.expired());
    EXPECT_EQ(*std::static_pointer_cast<int>(a.get_orchestration_scratch()), 7);
    EXPECT_EQ(b.get_orchestration_scratch(), nullptr);
    EXPECT_EQ(other.get_orchestration_scratch(), nullptr);
    a.set_orchestration_scratch({});
    EXPECT_TRUE(lifetime.expired());
    HostApiOps unsupported{};
    const HostApi fallback(nullptr, 0, 0, 0, &unsupported);
    EXPECT_EQ(fallback.get_orchestration_scratch(), nullptr);
    fallback.set_orchestration_scratch({});
}

TEST(HostApiTest, SchedulerScratchAndKernelQueriesDefaultOffWhenAbsent) {
    HostApiOps ops{};
    const HostApi absent(nullptr, 0, 0, 0, &ops);
    EXPECT_FALSE(absent.execution_is_kernel_mode());
    void *device = reinterpret_cast<void *>(0x1);
    void *host = reinterpret_cast<void *>(0x2);
    EXPECT_EQ(absent.acquire_scheduler_scratch(64, 64, &device, &host), -1);
    EXPECT_EQ(device, nullptr);
    EXPECT_EQ(host, nullptr);
    struct Mode {
        int kernel;
    } mode{0};
    ops.execution_is_kernel_mode = [](void *ctx) {
        return static_cast<Mode *>(ctx)->kernel;
    };
    const HostApi bound(&mode, 0, 0, 0, &ops);
    EXPECT_FALSE(bound.execution_is_kernel_mode());
    mode.kernel = 1;
    EXPECT_TRUE(bound.execution_is_kernel_mode());
}

TEST(HostApiTest, LegacyDiagnosticQueriesKeepRunnerFallbackAndAbsentDefaults) {
    HostApiOps ops{};
    HostApi absent(nullptr, 0, 0, 0, &ops);
    EXPECT_TRUE(absent.pmu_enabled());
    EXPECT_EQ(absent.chip_swimlane_level(), 0u);
    struct Selection {
        int pmu;
        uint32_t level;
    } selection{0, 0};
    ops.get_pmu_enabled = [](void *ctx) {
        return static_cast<Selection *>(ctx)->pmu;
    };
    ops.get_chip_swimlane_level = [](void *ctx) {
        return static_cast<Selection *>(ctx)->level;
    };
    HostApi legacy(&selection, 0, 0, 0, &ops);
    EXPECT_FALSE(legacy.pmu_enabled());
    EXPECT_EQ(legacy.chip_swimlane_level(), 0u);
    selection = {1, 3};
    EXPECT_TRUE(legacy.pmu_enabled());
    EXPECT_EQ(legacy.chip_swimlane_level(), 3u);
}
