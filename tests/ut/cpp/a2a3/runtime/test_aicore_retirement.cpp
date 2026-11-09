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
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicpu/platform_regs.h"
#include "aicpu/device_time.h"
#include "../../support/aicore_exit_responder.h"
#include "runtime.h"

void aicore_execute(Runtime *runtime, int block_idx, CoreType core_type);

#if defined(__linux__) && defined(SIMPLER_TEST_EPOCH_RETIREMENT)
namespace {
struct ReleasePollProbe {
    volatile uint64_t *ack;
    uint64_t epoch;
    std::mutex mutex;
    std::condition_variable changed;
    unsigned visits = 0;
    unsigned resumed = 0;
    bool returned = false;
};
thread_local ReleasePollProbe *release_poll_probe = nullptr;
}  // namespace

extern "C" int __real_sched_yield();
extern "C" int __wrap_sched_yield() {
    auto *probe = release_poll_probe;
    if (probe == nullptr || __atomic_load_n(probe->ack, __ATOMIC_ACQUIRE) != probe->epoch) return __real_sched_yield();
    // Only the executor thread is intercepted, after ACK publication. Each
    // visit proves that the production loop rejected the release it read.
    std::unique_lock<std::mutex> lock(probe->mutex);
    const unsigned visit = ++probe->visits;
    probe->changed.notify_all();
    probe->changed.wait(lock, [&] {
        return probe->resumed >= visit;
    });
    return 0;
}
#endif

AicoreExitTarget executor_exit_target(uint64_t reg, AicoreTeardownControl *gate, Handshake *report, uint64_t epoch) {
#ifdef SIMPLER_TEST_EPOCH_RETIREMENT
    return {reg, gate, &report->retired_epoch, epoch};
#else
    (void)report;
    (void)epoch;
    return {reg, gate};
#endif
}

namespace {

// The launch argument that selects the report protocol. The onboard entry
// publishes it into per-core state before calling the executor; this fixture
// stands in for that entry, so a case sets it the way a launch would.
uint64_t g_report_epoch = 0;
std::atomic<bool> g_pause_first_register{false};
std::atomic<bool> g_first_register_paused{false};
std::atomic<bool> g_resume_first_register{false};
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
    runtime.get_teardown_gates()[0].exit_request = AICORE_EXIT_NONCE_0;
    std::atomic<bool> returned{false};
    store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
    store_register(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    std::thread worker([&] {
        aicore_execute(&runtime, 0, core_type);
        returned.store(true, std::memory_order_release);
    });
    if (!wait_until(
            [] {
                return load_register(RegId::COND) == AICORE_EXIT_NONCE_0;
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
    registers.fill(0);
    for (int generation = 0; generation < 32; ++generation) {
        __atomic_store_n(&control.post_close_release, 0U, __ATOMIC_RELEASE);
        __atomic_store_n(&control.exit_request, 0U, __ATOMIC_RELEASE);
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
    AicoreExitResponder responder_0(targets[0]);
    EXPECT_FALSE(wait_until(
        [&] {
            return read_reg(targets[0].reg_addr, RegId::FAST_PATH_ENABLE) == REG_SPR_FAST_PATH_CLOSE;
        },
        std::chrono::milliseconds(100)
    ));
    EXPECT_EQ(__atomic_load_n(&controls[0].post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(__atomic_load_n(&controls[1].post_close_release, __ATOMIC_ACQUIRE), 0U);
    AicoreExitResponder responder_1(targets[1]);
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
    AicoreExitResponder responder_2(targets[1]);
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, get_sys_cnt_aicpu() + PLATFORM_PROF_SYS_CNT_FREQ / 10), -1);
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
    AicoreExitResponder responder_3(targets[1]);

    bool released[2] = {true, false};
    EXPECT_EQ(
        platform_retire_aicore_group(targets, 2, get_sys_cnt_aicpu() + PLATFORM_PROF_SYS_CNT_FREQ / 10, released), -1
    );
    EXPECT_FALSE(released[0]);
    EXPECT_TRUE(released[1]);
}

TEST(AicoreRetirement, RejectsInvalidGroupBeforeTouchingRegisters) {
    AicoreTeardownControl control{};
    registers.fill(0);
    const AicoreExitTarget targets[] = {
        {reinterpret_cast<uint64_t>(registers.data()), &control},
        {0, &control},
    };
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, get_sys_cnt_aicpu() + PLATFORM_PROF_SYS_CNT_FREQ / 10), -1);
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
    EXPECT_EQ(
        platform_retire_aicore_group(targets, 2, get_sys_cnt_aicpu() + PLATFORM_PROF_SYS_CNT_FREQ / 10, released), -1
    );
    EXPECT_FALSE(released[0]);
    EXPECT_FALSE(released[1]);

    bool null_targets[1] = {true};
    EXPECT_EQ(platform_retire_aicore_group(nullptr, 1, 0, null_targets), -1);
    EXPECT_FALSE(null_targets[0]);
}

// Run one real launch of this runtime's AICore executor and hand back the line
// it published. `epoch` is what the launch block carries, so these cases drive
// the production producer rather than a restatement of it.
//
// Host threads with ordinary stores: this observes *what* the producer wrote,
// never when another processor could see it.
Handshake publish_one_report(uint64_t epoch, CoreType core_type) {
    Runtime runtime;
    g_report_epoch = epoch;
    auto &control = runtime.get_teardown_gates()[0];
    registers.fill(0);
    __atomic_store_n(&control.post_close_release, 0U, __ATOMIC_RELEASE);
    store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
    store_register(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);

    std::atomic<bool> returned{false};
    std::thread worker([&] {
        aicore_execute(&runtime, 0, core_type);
        returned.store(true, std::memory_order_release);
    });
    EXPECT_TRUE(wait_until(
        [] {
            return load_register(RegId::COND) == AICORE_IDLE_VALUE;
        },
        std::chrono::seconds(5)
    ));
    const AicoreExitTarget target =
        executor_exit_target(reinterpret_cast<uint64_t>(registers.data()), &control, &runtime.get_workers()[0], epoch);
    EXPECT_EQ(platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()), 0);
    EXPECT_TRUE(wait_until(
        [&] {
            return returned.load(std::memory_order_acquire);
        },
        std::chrono::seconds(5)
    ));
    worker.join();
    g_report_epoch = 0;

    Handshake published{};
    std::memcpy(&published, &runtime.get_workers()[0], sizeof(published));
    return published;
}

// A native launch stamps its own report, and only that run's epoch accepts it.
TEST(AicoreReportEpoch, ANativeLaunchStampsTheReportItPublishes) {
    constexpr uint64_t kEpoch = 0x0000'0000'0000'2a02ULL;
    const Handshake published = publish_one_report(kEpoch, CoreType::AIC);

    EXPECT_EQ(published.aicore_done, 1u) << "block_idx + 1, unchanged by the stamp";
    EXPECT_EQ(published.report_epoch, kEpoch);
    EXPECT_TRUE(aicore_report_accepted(&published, kEpoch));
    EXPECT_FALSE(aicore_report_accepted(&published, kEpoch + 1)) << "another run must not accept this report";
}

// A kernel/persistent launch carries 0, writes no stamp, and is accepted on the
// marker alone — the protocol that predates this change, unchanged.
TEST(AicoreReportEpoch, AnUnstampedLaunchLeavesTheMarkerProtocolIntact) {
    const Handshake published = publish_one_report(0, CoreType::AIC);

    EXPECT_EQ(published.aicore_done, 1u);
    EXPECT_EQ(published.report_epoch, 0u) << "an unstamped launch must write no stamp";
    EXPECT_TRUE(aicore_report_accepted(&published, 0));
}

// Pause the production executor at its first register read, after its report
// but before IDLE. The AICPU must retain EXIT until this same core answers it.
void check_retirement_before_idle(uint64_t epoch, CoreType core_type, uint32_t retained_cond = AICORE_EXITED_VALUE) {
    Runtime runtime;
    auto &control = runtime.get_teardown_gates()[0];
    g_report_epoch = epoch;
    registers.fill(0);
    store_register(RegId::COND, retained_cond);
    g_first_register_paused.store(false, std::memory_order_release);
    g_resume_first_register.store(false, std::memory_order_release);
    g_pause_first_register.store(true, std::memory_order_release);
    std::atomic<bool> returned{false};
    std::thread worker([&] {
        aicore_execute(&runtime, 0, core_type);
        returned.store(true, std::memory_order_release);
    });
    if (!wait_until(
            [&] {
                return g_first_register_paused.load(std::memory_order_acquire);
            },
            std::chrono::seconds(5)
        ))
        std::_Exit(10);
    if (!aicore_report_accepted(&runtime.get_workers()[0], epoch)) std::_Exit(11);
    const AicoreExitTarget target =
        executor_exit_target(reinterpret_cast<uint64_t>(registers.data()), &control, &runtime.get_workers()[0], epoch);
    std::atomic<int32_t> rc{-2};
    std::thread retiring([&] {
        rc.store(platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()), std::memory_order_release);
    });
    if (!wait_until(
            [&] {
                return load_register(RegId::DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL;
            },
            std::chrono::seconds(2)
        ))
        std::_Exit(13);
    if (wait_until(
            [&] {
                return rc.load(std::memory_order_acquire) != -2;
            },
            std::chrono::milliseconds(20)
        ))
        std::_Exit(14);
    if (load_register(RegId::COND) != retained_cond || load_register(RegId::DATA_MAIN_BASE) != AICORE_EXIT_SIGNAL ||
        __atomic_load_n(&control.post_close_release, __ATOMIC_ACQUIRE) != 0)
        std::_Exit(15);
    // Independent resume: no release gate or artificial ACK lets the core out.
    g_resume_first_register.store(true, std::memory_order_release);
    if (!wait_until(
            [&] {
                return returned.load(std::memory_order_acquire);
            },
            std::chrono::seconds(5)
        ))
        std::_Exit(16);
    retiring.join();
    worker.join();
    if (rc.load(std::memory_order_acquire) != 0 || load_register(RegId::FAST_PATH_ENABLE) != REG_SPR_FAST_PATH_CLOSE ||
        (target.expected_epoch != 0 ?
             *target.retired_epoch != epoch :
             load_register(RegId::COND) != __atomic_load_n(&control.exit_request, __ATOMIC_ACQUIRE)))
        std::_Exit(17);
    std::_Exit(0);
}

TEST(AicoreFreshnessDeathTest, StampedAicRetiresAfterFailureBeforeIdle) {
    EXPECT_EXIT(check_retirement_before_idle(42, CoreType::AIC), testing::ExitedWithCode(0), "");
}

TEST(AicoreFreshnessDeathTest, StampedAivRetiresAfterFailureBeforeIdle) {
    EXPECT_EXIT(check_retirement_before_idle(42, CoreType::AIV), testing::ExitedWithCode(0), "");
}

TEST(AicoreFreshnessDeathTest, UnstampedAicRetiresAfterFailureBeforeIdle) {
    EXPECT_EXIT(check_retirement_before_idle(0, CoreType::AIC), testing::ExitedWithCode(0), "");
}

TEST(AicoreFreshnessDeathTest, UnstampedAivRetiresAfterFailureBeforeIdle) {
    EXPECT_EXIT(check_retirement_before_idle(0, CoreType::AIV), testing::ExitedWithCode(0), "");
}

}  // namespace

// Only handshake storage is used by these idle-worker tests. Keep host-side
// orchestration and the profiling service out of this executor-level fixture.
Runtime::Runtime() {
    std::memset(get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
    std::memset(get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * RUNTIME_MAX_WORKER);
}

volatile uint8_t *sim_get_reg_base() {
    if (g_pause_first_register.exchange(false, std::memory_order_acq_rel)) {
        g_first_register_paused.store(true, std::memory_order_release);
        while (!g_resume_first_register.load(std::memory_order_acquire))
            std::this_thread::yield();
    }
    return reinterpret_cast<volatile uint8_t *>(registers.data());
}
uint32_t sim_get_physical_core_id() { return 0; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }

uint64_t get_aicore_report_epoch() { return g_report_epoch; }

// A report can precede this run's first COND write. The retained EXITED is
// therefore not an acknowledgement when this run's core never answers EXIT.
TEST(ExitAckFreshnessRegression2520, PreviousExitedWithoutCurrentAckStaysUnreleased) {
    alignas(64) std::array<uint8_t, 65536> stale_registers{};
    AicoreTeardownControl stale_teardown{};
    const AicoreExitTarget stale_target{reinterpret_cast<uint64_t>(stale_registers.data()), &stale_teardown};
    platform_init_aicore_regs(stale_target.reg_addr);
    write_reg(stale_target.reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    bool released = true;

    EXPECT_EQ(platform_retire_aicore_group(&stale_target, 1, 0, &released), -1);
    EXPECT_FALSE(released);
    EXPECT_EQ(__atomic_load_n(&stale_teardown.post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(read_reg(stale_target.reg_addr, RegId::DATA_MAIN_BASE), AICORE_EXIT_SIGNAL);
}

TEST(AicoreRetirement, AckAfterDeadlineDoesNotCompleteRetirementWithoutAnOwner) {
    alignas(64) std::array<uint8_t, 65536> regs{};
    AicoreTeardownControl gate{};
    const AicoreExitTarget target{reinterpret_cast<uint64_t>(regs.data()), &gate};
    platform_init_aicore_regs(target.reg_addr);
    bool released = true;
    ASSERT_EQ(platform_retire_aicore_group(&target, 1, 0, &released), -1);
    ASSERT_FALSE(released);
    const uint32_t request = gate.exit_request;
    ASSERT_TRUE(request == AICORE_EXIT_NONCE_0 || request == AICORE_EXIT_NONCE_1);

    write_reg(target.reg_addr, RegId::COND, request);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::COND), request);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::DATA_MAIN_BASE), AICORE_EXIT_SIGNAL);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    EXPECT_EQ(gate.post_close_release, 0U);
    EXPECT_FALSE(released);
}

TEST(ExitAckFreshnessRegression2520, RetainedNoncesCannotReleaseASilentCore) {
    for (uint32_t old_cond : {AICORE_EXIT_NONCE_0, AICORE_EXIT_NONCE_1}) {
        alignas(64) std::array<uint8_t, 65536> stale_registers{};
        AicoreTeardownControl control{};
        const AicoreExitTarget target{reinterpret_cast<uint64_t>(stale_registers.data()), &control};
        platform_init_aicore_regs(target.reg_addr);
        write_reg(target.reg_addr, RegId::COND, old_cond);
        bool released = true;

        EXPECT_EQ(platform_retire_aicore_group(&target, 1, 0, &released), -1);
        EXPECT_FALSE(released);
        const uint32_t request = __atomic_load_n(&control.exit_request, __ATOMIC_ACQUIRE);
        EXPECT_NE(request, 0U);
        EXPECT_NE(request, old_cond);
        EXPECT_EQ(read_reg(target.reg_addr, RegId::COND), old_cond);
        EXPECT_EQ(__atomic_load_n(&control.post_close_release, __ATOMIC_ACQUIRE), 0U);
        EXPECT_EQ(read_reg(target.reg_addr, RegId::DATA_MAIN_BASE), AICORE_EXIT_SIGNAL);
    }
}

TEST(ExitAckFreshnessRegression2520, AnAnswerToThisRequestCanRetireWithoutObservingIdle) {
    alignas(64) std::array<uint8_t, 65536> stale_registers{};
    AicoreTeardownControl control{};
    const AicoreExitTarget target{reinterpret_cast<uint64_t>(stale_registers.data()), &control};
    platform_init_aicore_regs(target.reg_addr);
    write_reg(target.reg_addr, RegId::COND, AICORE_EXITED_VALUE);
    std::atomic<int32_t> rc{-2};
    std::thread retiring([&] {
        rc.store(platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()), std::memory_order_release);
    });
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (__atomic_load_n(&control.exit_request, __ATOMIC_ACQUIRE) == 0 && std::chrono::steady_clock::now() < limit)
        std::this_thread::yield();
    EXPECT_NE(__atomic_load_n(&control.exit_request, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(rc.load(std::memory_order_acquire), -2);
    EXPECT_EQ(__atomic_load_n(&control.post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::COND), AICORE_EXITED_VALUE);
    AicoreExitResponder responder(target);
    retiring.join();
    EXPECT_EQ(rc.load(std::memory_order_acquire), 0);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::COND), __atomic_load_n(&control.exit_request, __ATOMIC_ACQUIRE));
    EXPECT_EQ(__atomic_load_n(&control.post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
}

TEST(AicoreFreshnessDeathTest, NativeAicBeforeIdleRejectsRetainedNonce0) {
    EXPECT_EXIT(check_retirement_before_idle(42, CoreType::AIC, AICORE_EXIT_NONCE_0), testing::ExitedWithCode(0), "");
}
TEST(AicoreFreshnessDeathTest, NativeAivBeforeIdleRejectsRetainedNonce1) {
    EXPECT_EXIT(check_retirement_before_idle(42, CoreType::AIV, AICORE_EXIT_NONCE_1), testing::ExitedWithCode(0), "");
}
TEST(ExitAckFreshnessRegression2520, CondNonceRejectsEveryOldExitEncodingWithoutAResponder) {
    for (uint32_t old : {AICORE_EXIT_NONCE_0, AICORE_EXIT_NONCE_1, uint32_t(AICORE_EXITED_VALUE)}) {
        alignas(64) std::array<uint8_t, 65536> regs{};
        AicoreTeardownControl control{};
        const AicoreExitTarget target{reinterpret_cast<uint64_t>(regs.data()), &control};
        platform_init_aicore_regs(target.reg_addr);
        write_reg(target.reg_addr, RegId::COND, old);
        bool released = true;
        EXPECT_EQ(platform_retire_aicore_group(&target, 1, 0, &released), -1);
        EXPECT_FALSE(released);
        EXPECT_NE(control.exit_request, old);
        EXPECT_EQ(control.post_close_release, 0U);
        EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    }
}

TEST(AicoreFreshnessDeathTest, UnstampedAicBeforeIdleRejectsRetainedNonce0) {
    EXPECT_EXIT(check_retirement_before_idle(0, CoreType::AIC, AICORE_EXIT_NONCE_0), testing::ExitedWithCode(0), "");
}
TEST(AicoreFreshnessDeathTest, UnstampedAivBeforeIdleRejectsRetainedNonce1) {
    EXPECT_EXIT(check_retirement_before_idle(0, CoreType::AIV, AICORE_EXIT_NONCE_1), testing::ExitedWithCode(0), "");
}

TEST(EpochRetirement, SilentOldAckAndLateAckDoNotClose) {
    alignas(64) std::array<uint8_t, 65536> regs{};
    AicoreTeardownControl gate{};
    alignas(64) uint64_t ack = 7;
    const AicoreExitTarget target{reinterpret_cast<uint64_t>(regs.data()), &gate, &ack, 0x100000007ULL};
    platform_init_aicore_regs(target.reg_addr);
    bool released = true;
    EXPECT_EQ(platform_retire_aicore_group(&target, 1, 0, &released), -1);
    EXPECT_FALSE(released);
    EXPECT_EQ(gate.exit_request, 0U);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    __atomic_store_n(&ack, target.expected_epoch, __ATOMIC_RELEASE);
    EXPECT_EQ(gate.release_epoch, 0U);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
}

TEST(EpochRetirement, PartialAckOnlyReleasesMatchingCore) {
    alignas(64) std::array<uint8_t, 65536> regs[2]{};
    AicoreTeardownControl gates[2]{};
    uint64_t ack[2] = {UINT64_MAX, UINT64_MAX - 1};
    AicoreExitTarget targets[2] = {
        {reinterpret_cast<uint64_t>(regs[0].data()), &gates[0], &ack[0], UINT64_MAX},
        {reinterpret_cast<uint64_t>(regs[1].data()), &gates[1], &ack[1], UINT64_MAX}
    };
    for (auto &t : targets)
        platform_init_aicore_regs(t.reg_addr);
    bool released[2] = {};
    EXPECT_EQ(platform_retire_aicore_group(targets, 2, 0, released), -1);
    EXPECT_TRUE(released[0]);
    EXPECT_FALSE(released[1]);
    EXPECT_EQ(gates[0].release_epoch, UINT64_MAX);
    EXPECT_EQ(gates[1].release_epoch, 0U);
    EXPECT_EQ(read_reg(targets[0].reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_CLOSE);
    EXPECT_EQ(read_reg(targets[1].reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
}

#ifdef SIMPLER_TEST_EPOCH_RETIREMENT
void check_epoch_runs(CoreType type) {
    Runtime runtime;
    auto &gate = runtime.get_teardown_gates()[0];
    auto &ack = runtime.get_workers()[0].retired_epoch;
    for (uint64_t epoch : {uint64_t(1), uint64_t(1), uint64_t(2), uint64_t(0x100000002ULL), UINT64_MAX, UINT64_MAX}) {
        registers.fill(0);
        g_report_epoch = epoch;
        gate = {};
        aicore_invalidate_retired_report(&runtime.get_workers()[0]);
        g_first_register_paused.store(false, std::memory_order_release);
        g_resume_first_register.store(false, std::memory_order_release);
        g_pause_first_register.store(true, std::memory_order_release);
        std::atomic<bool> returned{false};
        store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
        store_register(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        std::thread worker([&] {
            aicore_execute(&runtime, 0, type);
            returned.store(true, std::memory_order_release);
        });
        if (!wait_until(
                [&] {
                    return g_first_register_paused.load(std::memory_order_acquire);
                },
                std::chrono::seconds(3)
            ))
            std::_Exit(36);
        if (__atomic_load_n(&ack, __ATOMIC_ACQUIRE) != 0) std::_Exit(37);
        g_resume_first_register.store(true, std::memory_order_release);
        if (!wait_until(
                [&] {
                    return __atomic_load_n(&ack, __ATOMIC_ACQUIRE) == epoch;
                },
                std::chrono::seconds(3)
            ))
            std::_Exit(31);
        const AicoreExitTarget target{reinterpret_cast<uint64_t>(registers.data()), &gate, &ack, epoch};
        if (platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()) != 0) std::_Exit(33);
        if (!wait_until(
                [&] {
                    return returned.load(std::memory_order_acquire);
                },
                std::chrono::seconds(3)
            ))
            std::_Exit(34);
        worker.join();
        if (gate.release_epoch != epoch || gate.exit_request != 0) std::_Exit(35);
    }
    std::_Exit(0);
}
TEST(EpochRetirementDeathTest, AicRetiresAcrossRepeatedAndWideEpochs) {
    EXPECT_EXIT(check_epoch_runs(CoreType::AIC), testing::ExitedWithCode(0), "");
}
TEST(EpochRetirementDeathTest, AivRetiresAcrossRepeatedAndWideEpochs) {
    EXPECT_EXIT(check_epoch_runs(CoreType::AIV), testing::ExitedWithCode(0), "");
}

#if defined(__linux__) && defined(SIMPLER_TEST_EPOCH_RETIREMENT)
void check_epoch_release_wait(CoreType type) {
    Runtime runtime;
    auto &gate = runtime.get_teardown_gates()[0];
    auto &ack = runtime.get_workers()[0].retired_epoch;
    for (uint64_t epoch : {uint64_t(1), uint64_t(0x100000002ULL), UINT64_MAX}) {
        registers.fill(0);
        gate = {};
        g_report_epoch = epoch;
        aicore_invalidate_retired_report(&runtime.get_workers()[0]);
        store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
        store_register(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        ReleasePollProbe probe{&ack, epoch};
        std::thread worker([&] {
            release_poll_probe = &probe;
            aicore_execute(&runtime, 0, type);
            release_poll_probe = nullptr;
            std::lock_guard<std::mutex> lock(probe.mutex);
            probe.returned = true;
            probe.changed.notify_all();
        });
        std::unique_lock<std::mutex> lock(probe.mutex);
        auto await_poll = [&](unsigned visit) {
            if (!probe.changed.wait_for(lock, std::chrono::seconds(3), [&] {
                    return probe.visits >= visit || probe.returned;
                }))
                std::_Exit(51);
            if (probe.returned || probe.visits != visit) std::_Exit(52);
            if (__atomic_load_n(&ack, __ATOMIC_ACQUIRE) != epoch ||
                load_register(RegId::FAST_PATH_ENABLE) != REG_SPR_FAST_PATH_OPEN)
                std::_Exit(53);
        };
        // No retirement owner has closed the window or published release.
        await_poll(1);
        if (__atomic_load_n(&gate.release_epoch, __ATOMIC_ACQUIRE) != 0) std::_Exit(54);
        // For wide epochs, the wrong value has the same low 32 bits.
        const uint64_t wrong_epoch = epoch > UINT32_MAX ? uint32_t(epoch) : epoch + 1;
        __atomic_store_n(&gate.release_epoch, wrong_epoch, __ATOMIC_RELEASE);
        probe.resumed = 1;
        probe.changed.notify_all();
        await_poll(2);
        if (__atomic_load_n(&gate.release_epoch, __ATOMIC_ACQUIRE) != wrong_epoch) std::_Exit(55);

        const AicoreExitTarget target{reinterpret_cast<uint64_t>(registers.data()), &gate, &ack, epoch};
        if (platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()) != 0) std::_Exit(56);
        if (load_register(RegId::FAST_PATH_ENABLE) != REG_SPR_FAST_PATH_CLOSE ||
            load_register(RegId::DATA_MAIN_BASE) != AICPU_IDLE_TASK_ID ||
            __atomic_load_n(&gate.release_epoch, __ATOMIC_ACQUIRE) != epoch)
            std::_Exit(57);
        probe.resumed = 2;
        probe.changed.notify_all();
        if (!probe.changed.wait_for(lock, std::chrono::seconds(3), [&] {
                return probe.returned;
            }))
            std::_Exit(58);
        lock.unlock();
        worker.join();
    }
    std::_Exit(0);
}
#endif

TEST(EpochRetirementDeathTest, AicWaitsForMatchingReleaseAfterAck) {
#if defined(__linux__) && defined(SIMPLER_TEST_EPOCH_RETIREMENT)
    EXPECT_EXIT(check_epoch_release_wait(CoreType::AIC), testing::ExitedWithCode(0), "");
#else
    GTEST_SKIP() << "Requires GNU ld sched_yield wrapping on Linux";
#endif
}
TEST(EpochRetirementDeathTest, AivWaitsForMatchingReleaseAfterAck) {
#if defined(__linux__) && defined(SIMPLER_TEST_EPOCH_RETIREMENT)
    EXPECT_EXIT(check_epoch_release_wait(CoreType::AIV), testing::ExitedWithCode(0), "");
#else
    GTEST_SKIP() << "Requires GNU ld sched_yield wrapping on Linux";
#endif
}

// Captured executions reuse the prepared epoch. Hold the real executor after
// report publication and before it can observe this replay's EXIT.
void check_replayed_epoch_before_idle(CoreType type) {
    Runtime runtime;
    constexpr uint64_t epoch = 0x100000007ULL;
    auto &gate = runtime.get_teardown_gates()[0];
    auto &report = runtime.get_workers()[0];
    report.retired_epoch = epoch;
    g_report_epoch = epoch;
    registers.fill(0);
    store_register(RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
    g_first_register_paused.store(false, std::memory_order_release);
    g_resume_first_register.store(false, std::memory_order_release);
    g_pause_first_register.store(true, std::memory_order_release);
    std::thread worker([&] {
        aicore_execute(&runtime, 0, type);
    });
    if (!wait_until(
            [&] {
                return g_first_register_paused.load(std::memory_order_acquire);
            },
            std::chrono::seconds(5)
        ))
        std::_Exit(41);
    if (!aicore_report_accepted(&report, epoch)) std::_Exit(42);
    const AicoreExitTarget target{reinterpret_cast<uint64_t>(registers.data()), &gate, &report.retired_epoch, epoch};
    bool released = true;
    const int rc = platform_retire_aicore_group(&target, 1, 0, &released);
    if (rc != -1 || released || gate.release_epoch != 0 ||
        load_register(RegId::FAST_PATH_ENABLE) != REG_SPR_FAST_PATH_OPEN)
        std::_Exit(43);
    std::_Exit(0);
}
TEST(EpochRetirementDeathTest, ReplayedAicRejectsSameEpochAckBeforeIdle) {
    EXPECT_EXIT(check_replayed_epoch_before_idle(CoreType::AIC), testing::ExitedWithCode(0), "");
}
TEST(EpochRetirementDeathTest, ReplayedAivRejectsSameEpochAckBeforeIdle) {
    EXPECT_EXIT(check_replayed_epoch_before_idle(CoreType::AIV), testing::ExitedWithCode(0), "");
}

#endif

TEST(EpochRetirement, MissingEpochAckAddressRejectsBeforeExit) {
    alignas(64) std::array<uint8_t, 65536> regs{};
    AicoreTeardownControl gate{};
    const AicoreExitTarget target{reinterpret_cast<uint64_t>(regs.data()), &gate, nullptr, 7};
    platform_init_aicore_regs(target.reg_addr);
    bool released = true;
    EXPECT_EQ(platform_retire_aicore_group(&target, 1, 0, &released), -1);
    EXPECT_FALSE(released);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::DATA_MAIN_BASE), AICPU_IDLE_TASK_ID);
    EXPECT_EQ(read_reg(target.reg_addr, RegId::FAST_PATH_ENABLE), REG_SPR_FAST_PATH_OPEN);
    EXPECT_EQ(gate.release_epoch, 0U);
}
