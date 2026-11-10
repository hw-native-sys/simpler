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
#include <cerrno>
#include <cstdlib>
#include <memory>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include "aicore/aicore_profiling_state.h"
#include "aicpu/abort_aicore_startup.h"
#include "aicpu/native_invocation_entry.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/native_invocation_evidence.h"
#include "aicore/aicore_profiling_state.h"
#ifdef TEST_RESIDENT_STARTUP
#include "scheduler/scheduler_types.h"
#endif

void aicore_execute(Runtime *, int, CoreType);
Runtime::Runtime() { dev = {}; }
namespace {
alignas(64) std::array<uint8_t, 0x10000> registers{};
std::atomic<bool> reject_binding{false};
std::atomic<int> finishes{0};
std::atomic<bool> retired{false};
uint64_t report_epoch = 42;
bool expire_immediately = false;
int32_t finish(int32_t index, int32_t rc) {
    if (finishes.fetch_add(1) == 2 && !retired.load()) std::_Exit(20);
    return aicpu_finish_native_invocation(index, rc);
}
// The subprocess bounds a broken startup/retirement wait without leaving live
// core threads behind. This runs the actual executor and register protocol.
void rejected_launch(bool resident) {
    auto runtime = std::make_unique<Runtime>();
    runtime->dev.worker_count = 1;
#ifdef TEST_RESIDENT_STARTUP
    SchedulerWorkerContext context{};
    SchedulerRunControl control{};
    if (resident) {
        context.scheduler_state_base_address = reinterpret_cast<uint64_t>(&control);
        runtime->publish_scheduler_bootstrap(
            SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, reinterpret_cast<uint64_t>(&context)
        );
    }
#else
    (void)resident;
#endif
    std::array<uint64_t, PLATFORM_MAX_CORES> regs{};
    regs[0] = reinterpret_cast<uint64_t>(registers.data());
    NativeInvocationResultStorage result;
    for (int round = 0; round < 2; ++round) {
        registers.fill(0);
        finishes = 0;
        retired = false;
        reject_binding = true;
        std::atomic<bool> core_returned{false};
        std::thread core([&] {
            aicore_execute(runtime.get(), 0, CoreType::AIC);
            core_returned = true;
        });
        const int32_t allowed[] = {sched_getcpu()};
        std::array<std::thread, 3> callbacks;
        for (auto &callback : callbacks)
            callback = std::thread([&] {
                NativeInvocationEntry invocation{
                    3, reinterpret_cast<uint64_t>(&result), report_epoch, aicpu_begin_native_invocation, [] {}, finish
                };
                const auto gate = platform_aicpu_entry_gate_filter(allowed, 1, 3, invocation.initializer());
                if (gate.status == 0 || gate.execute) std::_Exit(21);
                invocation.complete(
                    gate,
                    [] {
                        std::_Exit(23);
                        return 0;
                    },
                    [&] {
                        if (abort_aicore_startup(
                                runtime.get(), reinterpret_cast<uint64_t>(regs.data()), report_epoch
                            ) != 0)
                            std::_Exit(22);
                        retired = true;
                    }
                );
            });
        for (auto &callback : callbacks)
            callback.join();
        core.join();
        const auto terminal = device_run_result_terminal(result.result, report_epoch);
        if (terminal.state != DeviceRunTerminalState::Failed || terminal.code != -1) std::_Exit(27);
        if (!core_returned || finishes != 3 || runtime->dev.workers[0].aicore_done != 0) std::_Exit(24);
    }
    std::_Exit(0);
}
bool final_bind_only = false;
}  // namespace
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if ((!final_bind_only || value != nullptr) && reject_binding.exchange(false)) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
volatile uint8_t *sim_get_reg_base() { return registers.data(); }
uint32_t sim_get_physical_core_id() { return 0; }
uint32_t get_aicore_profiling_flag() { return SIMPLER_DFX_FLAG_CHIP_SWIMLANE; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { std::_Exit(25); }
struct PmuAicoreRing;
PmuAicoreRing *get_aicore_pmu_ring() { std::_Exit(26); }
uint64_t get_aicore_pmu_reg_base() { return 0; }
uint64_t get_aicore_report_epoch() { return report_epoch; }
uint64_t platform_ssbuf_hardware_base_address() { return 0; }
TEST(AbortAicoreStartupDeathTest, BindingFailureRetiresRealCoreBeforeLastCallback) {
    for (bool final_bind : {false, true}) {
        final_bind_only = final_bind;
        EXPECT_EXIT(rejected_launch(false), testing::ExitedWithCode(0), "");
#ifdef TEST_RESIDENT_STARTUP
        EXPECT_EXIT(rejected_launch(true), testing::ExitedWithCode(0), "");
#endif
    }
}

// Shorten only negative-path deadlines; successful protocol tests retain the
// production budget. No wall-clock sleep is needed to cover recovery debt.
extern "C" uint64_t __real__Z29platform_aicore_exit_deadlinev();
extern "C" uint64_t __wrap__Z29platform_aicore_exit_deadlinev() {
    return expire_immediately ? 0 : __real__Z29platform_aicore_exit_deadlinev();
}
TEST(AbortAicoreStartup, MissingReportOrAckDoesNotReleaseOrInvalidateCore) {
    auto runtime = std::make_unique<Runtime>();
    runtime->dev.worker_count = 1;
    registers.fill(0);
    std::array<uint64_t, PLATFORM_MAX_CORES> regs{};
    regs[0] = reinterpret_cast<uint64_t>(registers.data());
    auto &report = runtime->dev.workers[0];
    report.physical_core_id = 0;
    report.aicore_done = 1;
    report.report_epoch = report_epoch - 1;
    expire_immediately = true;
    EXPECT_EQ(abort_aicore_startup(runtime.get(), reinterpret_cast<uint64_t>(regs.data()), report_epoch), -1);
    EXPECT_EQ(read_reg(regs[0], RegId::DATA_MAIN_BASE), 0u);
    EXPECT_EQ(report.aicore_done, 1);
    report.report_epoch = report_epoch;
    EXPECT_EQ(abort_aicore_startup(runtime.get(), reinterpret_cast<uint64_t>(regs.data()), report_epoch), -1);
    EXPECT_EQ(read_reg(regs[0], RegId::DATA_MAIN_BASE), AICORE_EXIT_SIGNAL);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0u);
    EXPECT_EQ(report.aicore_done, 1);
    expire_immediately = false;
}

TEST(AbortAicoreStartup, DuplicatePhysicalReportNeverCreatesTwoRetirementOwners) {
    auto runtime = std::make_unique<Runtime>();
    runtime->dev.worker_count = 2;
    registers.fill(0);
    std::array<uint64_t, PLATFORM_MAX_CORES> regs{};
    regs[0] = reinterpret_cast<uint64_t>(registers.data());
    for (int32_t i = 0; i < 2; ++i) {
        runtime->dev.workers[i].physical_core_id = 0;
        runtime->dev.workers[i].aicore_done = 1;
        runtime->dev.workers[i].report_epoch = report_epoch;
    }
    expire_immediately = true;
    EXPECT_EQ(abort_aicore_startup(runtime.get(), reinterpret_cast<uint64_t>(regs.data()), report_epoch), -1);
    // Both reports were acknowledged, but only one register window was retired:
    // the duplicate never became a second owner and no ACK released a report.
    EXPECT_EQ(read_reg(regs[0], RegId::DATA_MAIN_BASE), AICORE_EXIT_SIGNAL);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0u);
    EXPECT_EQ(runtime->dev.workers[0].aicore_done, 1);
    EXPECT_EQ(runtime->dev.workers[1].aicore_done, 1);
    expire_immediately = false;
}
