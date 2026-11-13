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
#include <memory>
#include <thread>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "runtime.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_error.h"

void aicore_execute(Runtime *runtime, int block_idx, CoreType core_type);

namespace {
alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
uint64_t report_epoch = 0;
std::atomic<bool> first_read{false};
std::atomic<bool> pause_first_read{false};
std::atomic<bool> resume_read{false};

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// The subprocess bounds withheld EXIT/request/release without leaking a spinner.
enum class BootstrapBase { VALID, ABSENT, MISALIGNED, OVERFLOWING };

void check_exit(uint32_t mode, BootstrapBase base, uint64_t epoch, CoreType type, uint32_t old_cond) {
    auto runtime = std::make_unique<Runtime>();
    SchedulerWorkerContext context{};
    SchedulerRunControl run_control{};
    runtime->set_worker_count(1);
    report_epoch = epoch;
    registers.fill(0);
    const uint64_t address = reinterpret_cast<uint64_t>(registers.data());
    write_reg(address, RegId::COND, old_cond);
    first_read.store(false);
    pause_first_read.store(true);
    resume_read.store(false);
    auto &gate = runtime->dev.teardown_gates[0];
    gate.exit_request = AICORE_EXIT_NONCE_0;
    gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
    context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&run_control);
    context.run_control_offset = 0;
    scheduler_publish_error(&run_control, 1);
    const uint64_t context_address = reinterpret_cast<uint64_t>(&context);
    const uint64_t bootstrap_address = base == BootstrapBase::ABSENT      ? 0 :
                                       base == BootstrapBase::MISALIGNED  ? context_address + 1 :
                                       base == BootstrapBase::OVERFLOWING ? UINT64_MAX - 63 :
                                                                            context_address;
    runtime->publish_scheduler_bootstrap(mode, bootstrap_address);
    std::atomic<bool> returned{false};
    std::thread worker([&] {
        aicore_execute(runtime.get(), 0, type);
        returned.store(true, std::memory_order_release);
    });
    if (!wait_until(
            [] {
                return first_read.load(std::memory_order_acquire);
            },
            std::chrono::seconds(2)
        ))
        std::_Exit(10);
    if (!aicore_report_accepted(&runtime->dev.workers[0], epoch)) std::_Exit(11);
    if (wait_until(
            [&] {
                return returned.load(std::memory_order_acquire);
            },
            std::chrono::milliseconds(20)
        ))
        std::_Exit(12);
    if (read_reg(address, RegId::COND) != old_cond) std::_Exit(13);

    // The production owners reset both words before READY or their first EXIT.
    std::memset(&gate, 0, sizeof(gate));
    wmb();
    const AicoreExitTarget target{address, &gate};
    std::atomic<int32_t> result{-2};
    std::thread caller([&] {
        result.store(
            platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()), std::memory_order_release
        );
    });
    if (!wait_until(
            [&] {
                return read_reg(address, RegId::DATA_MAIN_BASE) == AICORE_EXIT_SIGNAL;
            },
            std::chrono::seconds(2)
        ))
        std::_Exit(14);
    if (result.load(std::memory_order_acquire) != -2 || gate.post_close_release != 0) std::_Exit(15);
    resume_read.store(true, std::memory_order_release);
    if (!wait_until(
            [&] {
                return returned.load(std::memory_order_acquire);
            },
            std::chrono::seconds(5)
        ))
        std::_Exit(16);
    caller.join();
    worker.join();
    if (result.load() != 0 || read_reg(address, RegId::DATA_MAIN_BASE) != AICPU_IDLE_TASK_ID ||
        gate.post_close_release != AICORE_POST_CLOSE_RELEASE || read_reg(address, RegId::COND) != gate.exit_request)
        std::_Exit(17);
    std::_Exit(0);
}
// The first register read pauses the real Core after its first READY poll.
// All test-side state changes happen while that Core is stopped at the seam.
void check_ready_error_publication(bool publish_ready, bool stale_error) {
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(1);
    SchedulerWorkerContext context{};
    SchedulerRunControl control{};
    context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&control);
    control.scheduler_timeout_cycles = 1;
    scheduler_publish_error(&control, stale_error ? 7 : 0);
    runtime->publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, reinterpret_cast<uint64_t>(&context));
    report_epoch = 42;
    registers.fill(0);
    first_read.store(false);
    pause_first_read.store(true);
    resume_read.store(false);
    const uint64_t address = reinterpret_cast<uint64_t>(registers.data());
    std::thread core([&] {
        aicore_execute(runtime.get(), 0, CoreType::AIV);
    });
    if (!wait_until(
            [] {
                return first_read.load();
            },
            std::chrono::seconds(2)
        ))
        std::_Exit(30);
    if (control.scheduler_error != (stale_error ? 7u : 0u) || control.error_claimed != 0) std::_Exit(31);
    if (publish_ready) {
        scheduler_publish_error(&control, 0);
        runtime->dev.workers[0].task = reinterpret_cast<uint64_t>(&context);
        runtime->dev.workers[0].aicpu_ready = SCHEDULER_RUNTIME_MODE_RESIDENT_READY;
    }
    resume_read.store(true, std::memory_order_release);
    if (publish_ready && !wait_until(
                             [&] {
                                 return __atomic_load_n(&control.scheduler_error, __ATOMIC_ACQUIRE) != 0;
                             },
                             std::chrono::seconds(2)
                         ))
        std::_Exit(32);
    const AicoreExitTarget target{address, &runtime->dev.teardown_gates[0]};
    if (platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()) != 0) std::_Exit(33);
    core.join();
    if (publish_ready) {
        if (control.error_site != static_cast<uint64_t>(SchedulerErrorSite::CONTEXT_READY_TIMEOUT)) std::_Exit(34);
    } else if (control.scheduler_error != (stale_error ? 7u : 0u) || control.error_claimed != 0) {
        std::_Exit(35);
    }
    std::_Exit(0);
}
}  // namespace

Runtime::Runtime() {
    std::memset(get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
    std::memset(get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * RUNTIME_MAX_WORKER);
    dev.scheduler_bootstrap = {};
}

volatile uint8_t *sim_get_reg_base() {
    if (!first_read.exchange(true, std::memory_order_acq_rel) && pause_first_read.load()) {
        while (!resume_read.load(std::memory_order_acquire))
            std::this_thread::yield();
    }
    return registers.data();
}
uint32_t sim_get_physical_core_id() { return 0; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }
PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
uint64_t get_aicore_pmu_reg_base() { return 0; }
uint64_t get_aicore_report_epoch() { return report_epoch; }

TEST(HbgNonceExitDeathTest, ResidentFailureBeforeReadyWaitsForCurrentExit) {
    for (auto epoch : {0ULL, 42ULL})
        for (auto type : {CoreType::AIC, CoreType::AIV})
            for (auto old : {AICORE_EXIT_NONCE_0, AICORE_EXIT_NONCE_1})
                EXPECT_EXIT(
                    check_exit(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, BootstrapBase::VALID, epoch, type, old),
                    testing::ExitedWithCode(0), ""
                );
}

TEST(HbgNonceExitDeathTest, InvalidModeReportsIdentityAndRequiresCurrentNonce) {
    for (uint32_t mode : {0U, 3U, 4U, UINT32_MAX})
        for (auto epoch : {0ULL, 42ULL})
            for (auto type : {CoreType::AIC, CoreType::AIV})
                for (auto old : {AICORE_EXIT_NONCE_0, AICORE_EXIT_NONCE_1})
                    EXPECT_EXIT(
                        check_exit(mode, BootstrapBase::OVERFLOWING, epoch, type, old), testing::ExitedWithCode(0), ""
                    );
}

TEST(HbgNonceExitDeathTest, InvalidResidentBaseRetiresWithoutReadingSchedulerBacking) {
    for (uint32_t mode : {SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, SCHEDULER_RUNTIME_MODE_RESIDENT_READY})
        for (auto base : {BootstrapBase::ABSENT, BootstrapBase::MISALIGNED, BootstrapBase::OVERFLOWING})
            for (auto epoch : {0ULL, 42ULL})
                for (auto type : {CoreType::AIC, CoreType::AIV})
                    EXPECT_EXIT(
                        check_exit(mode, base, epoch, type, AICORE_EXIT_NONCE_1), testing::ExitedWithCode(0), ""
                    );
}

TEST(HbgNonceExitDeathTest, ContextTimeoutDoesNotWriteBeforeReadyOrOnFailedInitExit) {
    EXPECT_EXIT(check_ready_error_publication(false, false), testing::ExitedWithCode(0), "");
    EXPECT_EXIT(check_ready_error_publication(false, true), testing::ExitedWithCode(0), "");
}

TEST(HbgNonceExitDeathTest, ContextTimeoutPublishesAfterReadyIncludingWithAStaleError) {
    EXPECT_EXIT(check_ready_error_publication(true, false), testing::ExitedWithCode(0), "");
    EXPECT_EXIT(check_ready_error_publication(true, true), testing::ExitedWithCode(0), "");
}
