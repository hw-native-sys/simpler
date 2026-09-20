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

/**
 * Retirement behaves the same whether every core answers or none does, and the
 * cases that matter are the ones where a core does not. A run that ends cleanly
 * exercises none of them, and a host-side test cannot see them either: the
 * device is force-reset on the path that produces them, so the per-core report
 * does not reliably reach the host log. These drive the group directly against
 * simulated register blocks, where a core's acknowledgement is something the
 * test decides rather than something it waits for.
 *
 * a5 has no fast-path window control and no return gate, so the properties a2a3
 * checks around those have no counterpart here. What is left is what the group
 * promises: every core signalled before any is waited on, no window closed until
 * the whole group has acknowledged, an unacknowledged core left alone and named,
 * and one shared deadline for the group rather than one per core.
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

namespace {

// One sparse register block per core, laid out the way sparse_reg_ptr expects.
class CoreRegs {
public:
    CoreRegs() { block_.fill(0); }
    uint64_t addr() { return reinterpret_cast<uint64_t>(block_.data()); }
    uint32_t dispatch() { return static_cast<uint32_t>(read_reg(addr(), RegId::DATA_MAIN_BASE)); }
    void ack() { write_reg(addr(), RegId::COND, AICORE_EXITED_VALUE); }
    bool signalled() { return dispatch() == AICORE_EXIT_SIGNAL; }
    bool closed() { return dispatch() == AICPU_IDLE_TASK_ID; }

private:
    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> block_{};
};

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// Short enough to keep the timeout cases quick, long enough that a sweep over a
// handful of cores cannot exhaust it on a loaded machine.
uint64_t short_deadline() { return get_sys_cnt_aicpu() + PLATFORM_PROF_SYS_CNT_FREQ / 10; }

}  // namespace

// The broadcast is what lets the cores drain concurrently; if it were interleaved
// with the waiting, a core late to answer would hold up signalling the rest.
TEST(AicoreRetirement, SignalsEveryCoreBeforeWaitingOnAny) {
    std::array<CoreRegs, 4> cores;
    uint64_t addrs[4];
    for (size_t i = 0; i < cores.size(); ++i)
        addrs[i] = cores[i].addr();

    // A budget far longer than the window this test watches is what makes the
    // assertion below discriminating: a retirement that signalled one core and
    // waited on it before signalling the next could not have reached the last
    // core within a fraction of a single budget.
    const uint64_t long_budget = PLATFORM_PROF_SYS_CNT_FREQ * 2;
    std::atomic<bool> done{false};
    std::thread caller([&] {
        platform_retire_aicore_group(addrs, cores.size(), get_sys_cnt_aicpu() + long_budget);
        done.store(true, std::memory_order_release);
    });
    // Nothing has acknowledged yet, so the call is still sweeping. Every core
    // must already carry the exit signal.
    EXPECT_TRUE(wait_until(
        [&] {
            return cores[cores.size() - 1].signalled();
        },
        std::chrono::milliseconds(200)
    ));
    for (auto &core : cores)
        EXPECT_TRUE(core.signalled());
    for (auto &core : cores)
        core.ack();
    EXPECT_TRUE(wait_until(
        [&] {
            return done.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));
    caller.join();
}

// A core is never quiesced while a peer is still being waited on.
TEST(AicoreRetirement, ClosesNoWindowUntilTheGroupHasAcknowledged) {
    std::array<CoreRegs, 2> cores;
    uint64_t addrs[2] = {cores[0].addr(), cores[1].addr()};

    std::atomic<int32_t> result{1};
    std::thread caller([&] {
        result.store(platform_retire_aicore_group(addrs, 2, short_deadline()), std::memory_order_release);
    });
    ASSERT_TRUE(wait_until(
        [&] {
            return cores[0].signalled() && cores[1].signalled();
        },
        std::chrono::seconds(2)
    ));
    cores[0].ack();
    // The second core has not answered, so neither window may close yet.
    EXPECT_FALSE(wait_until(
        [&] {
            return cores[0].closed();
        },
        std::chrono::milliseconds(50)
    ));
    cores[1].ack();
    EXPECT_TRUE(wait_until(
        [&] {
            return result.load(std::memory_order_acquire) == 0;
        },
        std::chrono::seconds(2)
    ));
    caller.join();
    EXPECT_TRUE(cores[0].closed());
    EXPECT_TRUE(cores[1].closed());
}

// The core that never answers keeps its exit signal: closing its window would
// hand it back while it is still whatever state it is stuck in.
TEST(AicoreRetirement, LeavesAnUnacknowledgedCoreUnclosed) {
    std::array<CoreRegs, 2> cores;
    uint64_t addrs[2] = {cores[0].addr(), cores[1].addr()};
    cores[1].ack();

    EXPECT_EQ(platform_retire_aicore_group(addrs, 2, get_sys_cnt_aicpu()), -1);
    EXPECT_TRUE(cores[0].signalled());
    EXPECT_FALSE(cores[0].closed());
    EXPECT_TRUE(cores[1].closed());
}

// An unretired core leaves the host nothing but a stream timeout, so the caller
// needs the group to say which ones they were.
TEST(AicoreRetirement, ReportsWhichCoresWereReleased) {
    std::array<CoreRegs, 2> cores;
    uint64_t addrs[2] = {cores[0].addr(), cores[1].addr()};
    cores[1].ack();

    bool released[2] = {true, false};
    EXPECT_EQ(platform_retire_aicore_group(addrs, 2, get_sys_cnt_aicpu(), released), -1);
    EXPECT_FALSE(released[0]);
    EXPECT_TRUE(released[1]);

    std::array<CoreRegs, 2> answered;
    uint64_t answered_addrs[2] = {answered[0].addr(), answered[1].addr()};
    for (auto &core : answered)
        core.ack();
    bool all_released[2] = {};
    EXPECT_EQ(platform_retire_aicore_group(answered_addrs, 2, short_deadline(), all_released), 0);
    EXPECT_TRUE(all_released[0]);
    EXPECT_TRUE(all_released[1]);
}

// One budget for the group, not one per core: a wedged core must not cost every
// core behind it a timeout of its own.
TEST(AicoreRetirement, SpendsOneBudgetOnTheWholeGroup) {
    constexpr size_t kCores = 8;
    std::array<CoreRegs, kCores> cores;
    uint64_t addrs[kCores];
    for (size_t i = 0; i < kCores; ++i)
        addrs[i] = cores[i].addr();

    const uint64_t budget_ticks = PLATFORM_PROF_SYS_CNT_FREQ / 10;
    const auto started = std::chrono::steady_clock::now();
    EXPECT_EQ(platform_retire_aicore_group(addrs, kCores, get_sys_cnt_aicpu() + budget_ticks), -1);
    const auto elapsed = std::chrono::steady_clock::now() - started;

    // Two budgets of headroom absorbs scheduling noise; eight would be the cost
    // of spending one budget per core.
    const auto budget = std::chrono::milliseconds(100);
    EXPECT_LT(elapsed, 2 * budget);
    for (auto &core : cores)
        EXPECT_FALSE(core.closed());
}

// The caller may pass an uninitialized buffer and read it on any return, so a
// rejected group still has to fill it, and must not touch a register first.
TEST(AicoreRetirement, RejectsInvalidGroupsWithoutTouchingRegisters) {
    std::array<CoreRegs, 2> cores;
    uint64_t addrs[2] = {cores[0].addr(), 0};

    bool released[2] = {true, true};
    EXPECT_EQ(platform_retire_aicore_group(addrs, 2, short_deadline(), released), -1);
    EXPECT_FALSE(released[0]);
    EXPECT_FALSE(released[1]);
    EXPECT_EQ(cores[0].dispatch(), 0U);

    EXPECT_EQ(platform_retire_aicore_group(nullptr, 1, short_deadline()), -1);
    EXPECT_EQ(platform_retire_aicore_group(nullptr, 0, short_deadline()), 0);
    EXPECT_EQ(platform_retire_aicore_group(addrs, PLATFORM_MAX_CORES + 1, short_deadline()), -1);
}

// The single-core entry point is the group over one target; callers retiring
// several are expected to use the group so they share a deadline and a sweep.
TEST(AicoreRetirement, SingleCoreEntryPointMatchesTheGroup) {
    CoreRegs acked;
    acked.ack();
    EXPECT_EQ(platform_deinit_aicore_regs(acked.addr()), 0);
    EXPECT_TRUE(acked.closed());

    // The single-core entry goes through the group path, so it owes the same
    // deferred close: signalling is not licence to quiesce a core that has not
    // answered yet. Watching the window stay open across a settle interval is
    // what separates that from a close issued on the signal alone.
    CoreRegs core;
    std::atomic<bool> done{false};
    std::thread caller([&] {
        platform_deinit_aicore_regs(core.addr());
        done.store(true, std::memory_order_release);
    });
    EXPECT_TRUE(wait_until(
        [&] {
            return core.signalled();
        },
        std::chrono::milliseconds(200)
    ));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(core.closed());
    EXPECT_FALSE(done.load(std::memory_order_acquire));

    core.ack();
    EXPECT_TRUE(wait_until(
        [&] {
            return done.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));
    caller.join();
    EXPECT_TRUE(core.closed());
}
