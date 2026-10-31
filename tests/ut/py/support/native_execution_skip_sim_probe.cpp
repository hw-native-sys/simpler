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
// Exercises the real simulator AICPU executor; only the answering cores are synthetic.
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "runtime.h"
#include "aicpu/platform_entry_args.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/device_run_result.h"

extern "C" int32_t aicpu_execute(Runtime *);
extern "C" int32_t aicpu_release_callables();

int main() {
    constexpr int cores = 3;
    constexpr int32_t failure = -71;
    auto runtime = std::make_unique<Runtime>();
    runtime->set_worker_count(cores);
    runtime->dev.aicpu_thread_num = 2;
    // A skipped run must not attach an arena or invoke a callable. These are
    // deliberately unavailable: an accidental normal boot cannot be successful.
    alignas(64) std::array<std::array<uint8_t, SIM_REG_BLOCK_SIZE>, cores> registers{};
    std::array<uint64_t, PLATFORM_MAX_CORES> addresses{};
    DeviceRunResultRegion result{};
    for (int i = 0; i < cores; ++i)
        addresses[i] = reinterpret_cast<uint64_t>(registers[i].data());
    set_platform_regs(reinterpret_cast<uint64_t>(addresses.data()));
    platform_aicpu_affinity_set_thread_idx(0);
    PlatformEntryArgs entry{};
    entry.unstamped_reports = true;
    entry.dispatch_error = std::getenv("SIMPLER_TEST_DISABLE_SKIP") == nullptr ? failure : 0;
    set_platform_entry_args(0, entry);
    set_platform_entry_args(1, entry);
    for (uint64_t round = 1; round <= 3; ++round) {
        result.published = 0;
        set_platform_run_result(reinterpret_cast<uint64_t>(&result), round);
        std::atomic<bool> stop{false};
        std::atomic<unsigned> dispatched{0}, drained{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < cores; ++i) {
            registers[i].fill(0);
            auto &report = runtime->dev.workers[i];
            report.physical_core_id = i;
            report.core_type = i == 0 ? CoreType::AIC : CoreType::AIV;
            report.report_epoch = 0;
            report.aicore_done = 1;
            workers.emplace_back([&, i] {
                auto &gate = runtime->dev.teardown_gates[i];
                bool acknowledged = false;
                while (!stop.load(std::memory_order_acquire)) {
                    const uint32_t task = read_reg(addresses[i], RegId::DATA_MAIN_BASE);
                    const uint32_t request = __atomic_load_n(&gate.exit_request, __ATOMIC_ACQUIRE);
                    if (task == AICORE_EXIT_SIGNAL &&
                        (request == AICORE_EXIT_NONCE_0 || request == AICORE_EXIT_NONCE_1) && !acknowledged) {
                        write_reg(addresses[i], RegId::COND, request);
                        acknowledged = true;
                    } else if (task != 0 && task != AICPU_IDLE_TASK_ID && task != AICORE_EXIT_SIGNAL) {
                        dispatched.fetch_add(1);
                    }
                    if (acknowledged &&
                        __atomic_load_n(&gate.post_close_release, __ATOMIC_ACQUIRE) == AICORE_POST_CLOSE_RELEASE) {
                        drained.fetch_add(1);
                        return;
                    }
                    std::this_thread::yield();
                }
            });
        }
        std::array<int32_t, 2> statuses{};
        std::array<std::thread, 2> callbacks;
        for (int i = 0; i < 2; ++i)
            callbacks[i] = std::thread([&, i] {
                platform_aicpu_affinity_set_thread_idx(i);
                statuses[i] = aicpu_execute(runtime.get());
            });
        for (auto &callback : callbacks)
            callback.join();
        const int32_t rc = statuses[0];
        // Bounded by the subprocess timeout if either the executor or a Core
        // gets stuck. Join proves the post-close return gates were released.
        for (auto &worker : workers)
            worker.join();
        assert(dispatched.load() == 0);
        assert(drained.load() == cores);
        const auto terminal = device_run_result_terminal(result, round);
        assert(terminal.state == DeviceRunTerminalState::Failed);
        if (entry.dispatch_error == 0) {
#ifdef SIMPLER_TEST_TMR
            // The scheduler has no runtime to dispatch. The real orchestrator
            // rejects the unregistered callable and publishes its signed error.
            assert(statuses[0] == 0);
            assert(statuses[1] == -1);
            assert(terminal.code == -1);
            assert(result.code_source == static_cast<uint32_t>(DeviceRunCodeSource::ThreadRc));
            std::puts("negative-control scheduler=0 orchestrator=-1 drained=3 terminal=-1 source=ThreadRc");
#endif
            assert(rc != failure || statuses[1] != failure);
            assert(aicpu_release_callables() == 0);
            return 2;
        }
        assert(rc == failure && statuses[1] == failure);
        assert(terminal.code == failure);
        std::printf(
            "round=%llu rc=%d dispatched=%u drained=%u terminal=%d\n", static_cast<unsigned long long>(round), rc,
            dispatched.load(), drained.load(), terminal.code
        );
    }
    assert(aicpu_release_callables() == 0);
}
