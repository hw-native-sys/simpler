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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "runtime.h"
#include "runtime_types.h"
#include "task_interface/tmr_kernel_context.h"
#include "task_interface/tmr_kernel_control.h"

namespace {
void kernel_test_spin();
}

// Build the actual architecture executor in this translation unit. Replacing
// only the simulator's wait hint exposes deterministic wait points without
// copying the dispatch algorithm or adding a production feature switch.
#undef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() kernel_test_spin()
#include "aicore_executor.cpp"

namespace {
using namespace simpler::tmr;

static_assert(offsetof(TaskPayload, tensor_count) == TASKPAYLOAD_TENSOR_COUNT_OFFSET);
static_assert(offsetof(TaskPayload, scalar_count) == TASKPAYLOAD_SCALAR_COUNT_OFFSET);
static_assert(offsetof(TaskPayload, scalars) == TASKPAYLOAD_SCALARS_OFFSET);

class Signal {
public:
    void set() {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = true;
        condition_.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return ready_;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool ready_{false};
};

struct SimState {
    // Covers a2a3's contiguous register block and a5's packed sparse pages.
    // Test reads/writes use the selected platform API, not hardcoded offsets.
    alignas(64) std::array<uint32_t, 0x6000 / sizeof(uint32_t)> registers{};
    std::atomic<uint32_t> register_accesses{0};
    std::atomic<uint32_t> calls{0};
    std::atomic<uint64_t> last_argument{0};
    std::function<void()> on_spin;
    std::function<void(uint32_t)> on_register;
};

SimState *active = nullptr;
thread_local bool is_worker = false;

void kernel_test_spin() {
    if (active->on_spin) active->on_spin();
}

void counting_kernel(int64_t *args) {
    active->last_argument.store(static_cast<uint64_t>(args[0]), std::memory_order_relaxed);
    active->calls.fetch_add(1, std::memory_order_release);
}

class TmrKernelAicoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        active = &model;
        runtime.dev.worker_count = 1;
        runtime.dev.workers[0].task = reinterpret_cast<uint64_t>(payloads.data());
        context.version = kTmrKernelContextVersion;
        context.bytes = sizeof(context);
        context.context_generation = 11;
        context.self_address = reinterpret_cast<uint64_t>(&context);
        context.resident_runtime = reinterpret_cast<uint64_t>(&runtime);
        context.control_address = reinterpret_cast<uint64_t>(&control);
        context.control_bytes = sizeof(control);
        context.reports_address = reinterpret_cast<uint64_t>(&report);
        context.reports_bytes = sizeof(report);
        context.worker_count = 1;
        for (auto &payload : payloads) {
            payload.function_bin_addr = reinterpret_cast<uint64_t>(&counting_kernel);
            payload.args[0] = 73;
        }
    }
    void TearDown() override { active = nullptr; }

    void clear_launch_words() {
        control = {};
        report = {};
        returned = false;
        release_at_return = 0;
        model.register_accesses = 0;
        model.calls = 0;
        model.on_spin = {};
        model.on_register = {};
    }

    std::thread launch(CoreType core_type, bool kernel = true) {
        return std::thread([&, core_type, kernel] {
            is_worker = true;
            if (kernel) aicore_execute_kernel(&runtime, &context, 0, core_type);
            else aicore_execute(&runtime, 0, core_type);
            returned.store(true, std::memory_order_release);
            is_worker = false;
        });
    }

    void release() {
        store_kernel_gm_word(&runtime.dev.teardown_gates[0].post_close_release, AICORE_POST_CLOSE_RELEASE);
    }

    SimState model;
    Runtime runtime;
    TmrKernelContextDescriptor context{};
    TmrLaunchControl control{};
    TmrCoreReport report{};
    std::array<DispatchPayload, 2> payloads{};
    TaskPayload source{};
    std::atomic<bool> returned{false};
    std::atomic<uint32_t> release_at_return{0};
};

TEST_F(TmrKernelAicoreTest, BothEntriesPublishIdentityBeforeRegisterOpenAndExecuteRepeatedRounds) {
    for (bool kernel : {false, true}) {
        for (uint32_t token : {uint32_t{2}, uint32_t{3}}) {
            clear_launch_words();
            runtime.dev.workers[0] = {};
            runtime.dev.teardown_gates[0].post_close_release = 0;
            write_reg(RegId::DATA_MAIN_BASE, 0);
            write_reg(RegId::COND, 0);
            auto &handshake = kernel ? report : runtime.dev.workers[0];
            Signal reported;
            Signal open;
            Signal executed;
            Signal exit;
            bool saw_report = false;
            bool saw_execution = false;
            model.on_spin = [&] {
                if (!saw_report) {
                    saw_report = true;
                    reported.set();
                    open.wait();
                } else if (!saw_execution && model.calls.load(std::memory_order_acquire) != 0) {
                    saw_execution = true;
                    executed.set();
                    exit.wait();
                }
            };
            auto worker = launch(CoreType::AIV, kernel);
            reported.wait();
            EXPECT_EQ(handshake.aicore_done, 1u);
            EXPECT_EQ(handshake.physical_core_id, 7u);
            EXPECT_EQ(static_cast<CoreType>(handshake.core_type), CoreType::AIV);
            EXPECT_EQ(model.calls.load(), 0u);
            EXPECT_FALSE(returned.load());
            handshake.task = reinterpret_cast<uint64_t>(payloads.data());
            write_reg(RegId::DATA_MAIN_BASE, token);
            open.set();
            executed.wait();
            EXPECT_EQ(model.calls.load(), 1u);
            EXPECT_EQ(model.last_argument.load(), 73u);
            EXPECT_EQ(read_reg(RegId::COND), MAKE_FIN_VALUE(token));
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
            release();
            exit.set();
            worker.join();
            EXPECT_TRUE(returned.load());
            EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
        }
    }
}

TEST_F(TmrKernelAicoreTest, RegisterExitReleasesEarlyDispatchWithoutDoorbellOrExecution) {
    constexpr uint32_t token = 2;
    source.tensor_count = 0;
    source.scalar_count = 1;
    source.scalars[0] = 84;
    payloads[token & 1u].src_payload = reinterpret_cast<uint64_t>(&source);
    report.task = reinterpret_cast<uint64_t>(payloads.data());
    write_reg(RegId::DATA_MAIN_BASE, token);
    Signal gated;
    Signal exit;
    model.on_spin = [&] {
        gated.set();
        exit.wait();
    };
    auto worker = launch(CoreType::AIV);
    gated.wait();
    EXPECT_EQ(payloads[token & 1u].args[0], 84u);
    EXPECT_EQ(model.calls.load(), 0u);
    EXPECT_EQ(read_dmb_high32(), 0u);
    write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    release();
    exit.set();
    worker.join();
    EXPECT_EQ(model.calls.load(), 0u);
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
}

}  // namespace

// Host-side construction and per-core platform identity/profiling are outside
// the device wrapper under test. No runtime allocation or DFX service is linked.
Runtime::Runtime() :
    dev{} {}

volatile uint8_t *sim_get_reg_base() {
    if (is_worker) {
        const uint32_t access = active->register_accesses.fetch_add(1) + 1;
        if (active->on_register) active->on_register(access);
    }
    return reinterpret_cast<volatile uint8_t *>(active->registers.data());
}
uint32_t sim_get_physical_core_id() { return 7; }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }
struct PmuAicoreRing;
PmuAicoreRing *get_aicore_pmu_ring() { return nullptr; }
uint64_t get_aicore_pmu_reg_base() { return 0; }
