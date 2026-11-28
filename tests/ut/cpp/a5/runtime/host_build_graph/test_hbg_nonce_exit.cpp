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
#include "scheduler/scheduler_graph.h"
#include "scheduler/wire_types.h"

void aicore_execute(Runtime *runtime, int block_idx, CoreType core_type);

namespace {
alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
uint64_t report_epoch = 0;

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
enum class ControlBase { VALID, ABSENT, MISALIGNED };

void check_exit(uint64_t epoch, CoreType type, uint32_t old_cond, ControlBase base) {
    auto runtime = std::make_unique<Runtime>();
    simpler::percounter::PercounterControl control{};
    runtime->set_worker_count(1);
    report_epoch = epoch;
    registers.fill(0);
    const uint64_t address = reinterpret_cast<uint64_t>(registers.data());
    write_reg(address, RegId::COND, old_cond);
    auto &gate = runtime->get_teardown_gates()[0];
    gate.exit_request = AICORE_EXIT_NONCE_0;
    gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
    uint64_t control_address = reinterpret_cast<uint64_t>(&control);
    if (base == ControlBase::ABSENT) control_address = 0;
    if (base == ControlBase::MISALIGNED) ++control_address;
    alignas(32) uint32_t lifecycle = UINT32_MAX;
    runtime->publish_scheduler_bootstrap(
        SCHEDULER_RUNTIME_MODE_CALENDAR, control_address, reinterpret_cast<uint64_t>(&lifecycle),
        reinterpret_cast<uint64_t>(control.rejected_exit)
    );
    std::atomic<bool> returned{false};
    std::thread worker([&] {
        aicore_execute(runtime.get(), 0, type);
        returned.store(true, std::memory_order_release);
    });
    if (!wait_until(
            [&] {
                return aicore_report_accepted(&runtime->get_workers()[0], epoch);
            },
            std::chrono::seconds(2)
        ))
        std::_Exit(10);
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
    const AicoreExitTarget target{address, &gate, false};
    std::atomic<int32_t> result{-2};
    std::thread caller([&] {
        result.store(
            platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline()), std::memory_order_release
        );
    });
    if (result.load(std::memory_order_acquire) != -2 || gate.post_close_release != 0) std::_Exit(15);
    publish_aicore_lifecycle_word(&lifecycle, AICORE_LIFECYCLE_EXIT_BIT);
    if (!wait_until(
            [&] {
                return returned.load(std::memory_order_acquire);
            },
            std::chrono::seconds(5)
        ))
        std::_Exit(16);
    caller.join();
    worker.join();
    if (result.load() != 0 || read_reg(address, RegId::DATA_MAIN_BASE) != 0U ||
        gate.post_close_release != AICORE_POST_CLOSE_RELEASE || read_reg(address, RegId::COND) != gate.exit_request)
        std::_Exit(17);
    std::_Exit(0);
}
}  // namespace

Runtime::Runtime() :
    dev{} {
    dev.shared_state_addr = reinterpret_cast<uint64_t>(&host_shared_state_);
    std::memset(get_workers(), 0, sizeof(Handshake) * RUNTIME_MAX_WORKER);
    std::memset(get_teardown_gates(), 0, sizeof(AicoreTeardownControl) * RUNTIME_MAX_WORKER);
    dev.scheduler_bootstrap = {};
}

volatile uint8_t *sim_get_reg_base() { return registers.data(); }
uint32_t sim_get_physical_core_id() { return 0; }
uint64_t platform_ssbuf_hardware_base_address() { return 0; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }
PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
uint64_t get_aicore_pmu_reg_base() { return 0; }
uint64_t get_aicore_report_epoch() { return report_epoch; }

TEST(HbgNonceExitDeathTest, CalendarBeforeReadyWaitsForCurrentExit) {
    for (auto epoch : {0ULL, 42ULL})
        for (auto type : {CoreType::AIC, CoreType::AIV})
            for (auto old : {AICORE_EXIT_NONCE_0, AICORE_EXIT_NONCE_1})
                EXPECT_EXIT(check_exit(epoch, type, old, ControlBase::VALID), testing::ExitedWithCode(0), "");
}

TEST(HbgNonceExitDeathTest, MissingOrMisalignedControlStillAcknowledgesCurrentExit) {
    for (auto base : {ControlBase::ABSENT, ControlBase::MISALIGNED})
        EXPECT_EXIT(check_exit(42, CoreType::AIC, AICORE_EXIT_NONCE_0, base), testing::ExitedWithCode(0), "");
}
