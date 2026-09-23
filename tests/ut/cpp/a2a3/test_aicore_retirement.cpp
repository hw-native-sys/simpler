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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicpu/platform_regs.h"
#include "aicpu/device_time.h"
#pragma push_macro("OUT_OF_ORDER_STORE_BARRIER")
#undef OUT_OF_ORDER_STORE_BARRIER
#include "common/memory_barrier.h"
#pragma pop_macro("OUT_OF_ORDER_STORE_BARRIER")
#include "aicore_teardown.h"
#include "runtime.h"

void aicore_execute(Runtime *runtime, int block_idx, CoreType core_type);

namespace {
alignas(64) std::array<uint32_t, 0x500 / sizeof(uint32_t)> registers{};

uint32_t load_register(RegId reg) {
    return __atomic_load_n(&registers[reg_offset(reg) / sizeof(uint32_t)], __ATOMIC_ACQUIRE);
}

void store_register(RegId reg, uint32_t value) {
    __atomic_store_n(&registers[reg_offset(reg) / sizeof(uint32_t)], value, __ATOMIC_RELEASE);
}

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// The subprocess bounds a deliberately withheld CLOSE without leaking a
// blocked worker into the test process. The executor is the production TU.
void check_no_return_before_close(CoreType core_type) {
    Runtime runtime;
    std::atomic<bool> returned{false};
    store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
    store_register(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    std::thread worker([&] {
        aicore_execute(&runtime, 0, core_type);
        returned.store(true, std::memory_order_release);
    });
    if (!wait_until(
            [] {
                return load_register(RegId::COND) == AICORE_EXITED_VALUE;
            },
            std::chrono::seconds(5)
        )) {
        std::_Exit(2);
    }
    const bool returned_early = wait_until(
        [&] {
            return returned.load(std::memory_order_acquire);
        },
        std::chrono::milliseconds(100)
    );
    std::_Exit(returned_early ? 1 : 0);
}

TEST(AicoreRetirementDeathTest, AivCannotReturnBeforeFastPathClose) {
    EXPECT_EXIT(check_no_return_before_close(CoreType::AIV), testing::ExitedWithCode(0), "");
}

TEST(AicoreRetirementDeathTest, AicCannotReturnBeforeFastPathClose) {
    EXPECT_EXIT(check_no_return_before_close(CoreType::AIC), testing::ExitedWithCode(0), "");
}

void check_repeated_retirement(CoreType core_type) {
    Runtime runtime;
    auto &control = runtime.get_teardown_gates()[0];
    for (int generation = 0; generation < 32; ++generation) {
        registers.fill(0);
        __atomic_store_n(&control.post_close_release, 0U, __ATOMIC_RELEASE);
        std::atomic<bool> returned{false};
        store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
        store_register(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
        std::thread worker([&] {
            aicore_execute(&runtime, 0, core_type);
            returned.store(true, std::memory_order_release);
        });
        if (!wait_until(
                [] {
                    return load_register(RegId::COND) == AICORE_IDLE_VALUE;
                },
                std::chrono::seconds(5)
            ))
            std::_Exit(2);
        const AicoreExitTarget target{reinterpret_cast<uint64_t>(registers.data()), &control};
        if (platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()) != 0) std::_Exit(3);
        if (!wait_until(
                [&] {
                    return returned.load(std::memory_order_acquire);
                },
                std::chrono::seconds(5)
            ))
            std::_Exit(4);
        worker.join();
        if (load_register(RegId::FAST_PATH_ENABLE) != REG_SPR_FAST_PATH_CLOSE) std::_Exit(5);
        if (__atomic_load_n(&control.post_close_release, __ATOMIC_ACQUIRE) != AICORE_POST_CLOSE_RELEASE) {
            std::_Exit(6);
        }
    }
    std::_Exit(0);
}

TEST(AicoreRetirementDeathTest, AivReturnsAfterCloseAcrossRepeatedLaunches) {
    EXPECT_EXIT(check_repeated_retirement(CoreType::AIV), testing::ExitedWithCode(0), "");
}

TEST(AicoreRetirementDeathTest, AicReturnsAfterCloseAcrossRepeatedLaunches) {
    EXPECT_EXIT(check_repeated_retirement(CoreType::AIC), testing::ExitedWithCode(0), "");
}

TEST(AicoreRetirement, GroupWaitsForEveryAckBeforeClosingAnyWindow) {
    alignas(64) std::array<uint32_t, 0x500 / sizeof(uint32_t)> peer_registers{};
    AicoreTeardownControl controls[2]{};
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &controls[0]},
        {reinterpret_cast<uint64_t>(peer_registers.data()), &controls[1]},
    };
    registers.fill(0);
    for (const auto &target : targets)
        platform_init_aicore_regs(target.reg_addr);
    std::atomic<int32_t> result{-2};
    std::thread retiring([&] {
        result.store(
            platform_retire_aicore_group(targets, 2, platform_aicore_exit_deadline()), std::memory_order_release
        );
    });
    EXPECT_TRUE(wait_until(
        [&] {
            return read_reg(targets[0].reg_addr, RegId::DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL &&
                   read_reg(targets[1].reg_addr, RegId::DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL;
        },
        std::chrono::seconds(5)
    ));
    write_reg(targets[0].reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    EXPECT_FALSE(wait_until(
        [&] {
            return read_reg(targets[0].reg_addr, RegId::FAST_PATH_ENABLE) == REG_SPR_FAST_PATH_CLOSE;
        },
        std::chrono::milliseconds(100)
    ));
    EXPECT_EQ(__atomic_load_n(&controls[0].post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(__atomic_load_n(&controls[1].post_close_release, __ATOMIC_ACQUIRE), 0U);
    write_reg(targets[1].reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    retiring.join();
    EXPECT_EQ(result.load(std::memory_order_acquire), 0);
    for (const auto &target : targets) {
        EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_CLOSE);
        EXPECT_EQ(__atomic_load_n(&target.teardown->post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
    }
}

TEST(AicoreRetirement, MissingAckDoesNotCloseOrReleaseThatCore) {
    alignas(64) std::array<uint32_t, 0x500 / sizeof(uint32_t)> peer_registers{};
    AicoreTeardownControl controls[2]{};
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &controls[0]},
        {reinterpret_cast<uint64_t>(peer_registers.data()), &controls[1]},
    };
    registers.fill(0);
    for (const auto &target : targets)
        platform_init_aicore_regs(target.reg_addr);
    write_reg(targets[1].reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, 0), -1);
    EXPECT_EQ(read_reg(targets[0].reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    EXPECT_EQ(__atomic_load_n(&controls[0].post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(read_reg(targets[1].reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_CLOSE);
    EXPECT_EQ(__atomic_load_n(&controls[1].post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
}

// An unreleased worker blocks on a gate it cannot log about, so the caller's
// only way to name it is this per-target report.
TEST(AicoreRetirement, ReportsWhichTargetsWereReleased) {
    alignas(64) std::array<uint32_t, 0x500 / sizeof(uint32_t)> peer_registers{};
    AicoreTeardownControl controls[2]{};
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &controls[0]},
        {reinterpret_cast<uint64_t>(peer_registers.data()), &controls[1]},
    };
    registers.fill(0);
    for (const auto &target : targets)
        platform_init_aicore_regs(target.reg_addr);
    write_reg(targets[1].reg_addr, RegId::COND, AICORE_EXITED_VALUE);

    bool released[2] = {true, false};
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, 0, released), -1);
    EXPECT_FALSE(released[0]);
    EXPECT_TRUE(released[1]);

    write_reg(targets[0].reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    bool all_released[2] = {};
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, platform_aicore_exit_deadline(), all_released), 0);
    EXPECT_TRUE(all_released[0]);
    EXPECT_TRUE(all_released[1]);
}

TEST(AicoreRetirement, RejectsInvalidGroupBeforeTouchingRegisters) {
    AicoreTeardownControl control{};
    registers.fill(0);
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &control},
        {0, &control},
    };
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, 0), -1);
    EXPECT_EQ(load_register(RegId::DATA_MAIN_BASE), 0U);
    EXPECT_EQ(platform_retire_aicore_group(nullptr, 1, 0), -1);
    EXPECT_EQ(platform_retire_aicore_group(nullptr, 0, 0), 0);
    EXPECT_EQ(platform_retire_aicore_group(targets, PLATFORM_MAX_CORES + 1, 0), -1);
}

// The caller reads `released` whenever the call returns non-zero and leaves the
// buffer uninitialized, so a rejection must still fill every entry.
TEST(AicoreRetirement, FillsReleasedOnRejectedGroups) {
    AicoreTeardownControl control{};
    registers.fill(0);
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &control},
        {0, &control},
    };
    bool released[2] = {true, true};
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, 0, released), -1);
    EXPECT_FALSE(released[0]);
    EXPECT_FALSE(released[1]);

    bool null_targets[1] = {true};
    EXPECT_EQ(platform_retire_aicore_group(nullptr, 1, 0, null_targets), -1);
    EXPECT_FALSE(null_targets[0]);
}
}  // namespace

// Only handshake storage is used by these idle-worker tests. Keep host-side
// orchestration and the profiling service out of this executor-level fixture.
Runtime::Runtime() {
    std::memset(get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
    std::memset(get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * RUNTIME_MAX_WORKER);
}

volatile uint8_t *sim_get_reg_base() { return reinterpret_cast<volatile uint8_t *>(registers.data()); }
uint32_t sim_get_physical_core_id() { return 0; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }

// This copy executes the production control flow against observable MMIO and
// barrier operations. The executor tests above link the normal platform TU.
namespace retirement_order_probe {
constexpr size_t kWindowCount = 3;
using RegisterWindow = std::array<uint32_t, 0x500 / sizeof(uint32_t)>;
alignas(64) std::array<RegisterWindow, kWindowCount> windows{};
AicoreTeardownControl controls[kWindowCount]{};
AicoreExitTarget targets[kWindowCount]{};
std::array<bool, kWindowCount> expected_ack{};
std::array<bool, kWindowCount> readback_seen{};
size_t active_count = 0;
size_t readbacks = 0;
bool drained = false;

void prepare(const std::array<bool, kWindowCount> &acks, size_t count = kWindowCount) {
    expected_ack = acks;
    readback_seen.fill(false);
    active_count = count;
    readbacks = 0;
    drained = false;
    for (size_t i = 0; i < kWindowCount; ++i) {
        windows[i].fill(0);
        controls[i].post_close_release = 0;
        windows[i][reg_offset(RegId::FAST_PATH_ENABLE) / sizeof(uint32_t)] = REG_SPR_FAST_PATH_OPEN;
        windows[i][reg_offset(RegId::COND) / sizeof(uint32_t)] = acks[i] ? AICORE_EXITED_VALUE : AICORE_IDLE_VALUE;
        targets[i] = {reinterpret_cast<uint64_t>(windows[i].data()), &controls[i]};
    }
}

uint32_t observe_reg_load(const volatile uint32_t *ptr) {
    for (size_t i = 0; i < active_count; ++i) {
        if (ptr != &windows[i][reg_offset(RegId::FAST_PATH_ENABLE) / sizeof(uint32_t)]) continue;
        EXPECT_TRUE(expected_ack[i]);
        EXPECT_FALSE(readback_seen[i]);
        readback_seen[i] = true;
        EXPECT_FALSE(drained);
        for (size_t j = 0; j < active_count; ++j) {
            EXPECT_EQ(
                windows[j][reg_offset(RegId::FAST_PATH_ENABLE) / sizeof(uint32_t)],
                expected_ack[j] ? REG_SPR_FAST_PATH_CLOSE : REG_SPR_FAST_PATH_OPEN
            ) << "Window "
              << j << " at readback of window " << i;
            EXPECT_EQ(controls[j].post_close_release, 0U);
        }
        ++readbacks;
    }
    return *ptr;
}

void observe_write_reg(uint64_t base, RegId reg, uint32_t value) {
    *reinterpret_cast<uint32_t *>(base + reg_offset(reg)) = value;
}

void observe_read_barrier() {
    size_t acknowledged_count = 0;
    for (size_t i = 0; i < active_count; ++i) {
        acknowledged_count += expected_ack[i] ? 1 : 0;
        EXPECT_EQ(readback_seen[i], expected_ack[i]);
        EXPECT_EQ(controls[i].post_close_release, 0U);
    }
    EXPECT_EQ(readbacks, acknowledged_count);
    EXPECT_FALSE(drained);
    drained = true;
}

void observe_write_barrier() {}

#pragma push_macro("rmb")
#pragma push_macro("wmb")
#undef rmb
#undef wmb
#define rmb() observe_read_barrier()
#define wmb() observe_write_barrier()
#define get_reg_ptr observed_reg_ptr
#define read_reg observed_read_reg
#define reg_load_acquire observe_reg_load
#define write_reg observe_write_reg
#include "../../../../src/a2a3/platform/shared/aicpu/platform_regs.cpp"
#undef write_reg
#undef reg_load_acquire
#undef read_reg
#undef get_reg_ptr
#pragma pop_macro("wmb")
#pragma pop_macro("rmb")

TEST(AicoreRetirementOrdering, ClosesWholeGroupBeforeFirstReadbackAndDrainsBeforeRelease) {
    prepare({true, true, true});
    bool released[kWindowCount]{};
    EXPECT_EQ(retirement_order_probe::platform_retire_aicore_group(targets, kWindowCount, 0, released), 0);
    EXPECT_TRUE(drained);
    EXPECT_EQ(readbacks, kWindowCount);
    for (size_t i = 0; i < kWindowCount; ++i) {
        EXPECT_TRUE(released[i]);
        EXPECT_EQ(controls[i].post_close_release, AICORE_POST_CLOSE_RELEASE);
        EXPECT_EQ(windows[i][reg_offset(RegId::DATA_MAIN_BASE) / sizeof(uint32_t)], AICPU_IDLE_TASK_ID);
    }
}

TEST(AicoreRetirementOrdering, TimeoutClosesAndReadsOnlyAcknowledgedWindowsBeforeRelease) {
    prepare({true, false, true});
    bool released[kWindowCount]{};
    EXPECT_EQ(retirement_order_probe::platform_retire_aicore_group(targets, kWindowCount, 0, released), -1);
    EXPECT_TRUE(drained);
    EXPECT_EQ(readbacks, 2U);
    for (size_t i = 0; i < kWindowCount; ++i) {
        EXPECT_EQ(released[i], expected_ack[i]);
        EXPECT_EQ(controls[i].post_close_release, expected_ack[i] ? AICORE_POST_CLOSE_RELEASE : 0U);
        EXPECT_EQ(
            windows[i][reg_offset(RegId::DATA_MAIN_BASE) / sizeof(uint32_t)],
            expected_ack[i] ? AICPU_IDLE_TASK_ID : AICORE_EXIT_SIGNAL
        );
    }
}

TEST(AicoreRetirementOrdering, SingleWindowCloseIncludesReadbackAndLeavesDrainToCaller) {
    prepare({true, false, false}, 1);
    platform_close_aicore_window(targets[0].reg_addr);
    EXPECT_EQ(readbacks, 1U);
    EXPECT_FALSE(drained);
    EXPECT_EQ(controls[0].post_close_release, 0U);
    EXPECT_EQ(windows[0][reg_offset(RegId::DATA_MAIN_BASE) / sizeof(uint32_t)], AICPU_IDLE_TASK_ID);
}
}  // namespace retirement_order_probe
