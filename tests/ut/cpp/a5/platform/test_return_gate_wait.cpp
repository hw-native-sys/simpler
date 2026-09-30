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
 * The post-close return gate has an AICore half and an AICPU half. The AICPU
 * half is covered by test_aicore_retirement.cpp; this case covers the AICore
 * half: the real a5sim startup EXIT and post-close waits, run on a host thread
 * against a simulated register block and production gate word. The production
 * retire path publishes EXIT, closes the window, and releases the gate.
 *
 * That wait has no timeout, so each case publishes the release before it joins
 * and uses non-fatal assertions, so a failed expectation can never skip the
 * cleanup and leave the worker spinning.
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
// inner_kernel.h needs RegId / reg_offset / sparse_reg_ptr from
// common/platform_config.h and sys_cnt_now_ticks from aicpu/device_time.h, so it
// is included after both rather than on its own.
#include "aicore/inner_kernel.h"

namespace {

// One sparse register block per core, laid out the way sparse_reg_ptr expects.
class CoreRegs {
public:
    CoreRegs() { block_.fill(0); }
    uint64_t addr() { return reinterpret_cast<uint64_t>(block_.data()); }
    volatile uint8_t *base() { return block_.data(); }
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

void publish_release(AicoreTeardownControl &gate) {
    __atomic_store_n(&gate.post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
}

}  // namespace

namespace {
thread_local volatile uint8_t *sim_reg_base = nullptr;
}

volatile uint8_t *sim_get_reg_base() { return sim_reg_base; }

// Only the AICPU's post-close publication ends the wait; the AICore's own exit
// write does not.
TEST(ReturnGateWait, RealSimWorkerReturnsOnlyAfterReleaseIsPublished) {
    AicoreTeardownControl gate{};
    std::atomic<bool> returned{false};
    std::thread worker([&] {
        wait_for_post_close_release(&gate.post_close_release);
        returned.store(true, std::memory_order_release);
    });

    EXPECT_FALSE(wait_until(
        [&] {
            return returned.load(std::memory_order_acquire);
        },
        std::chrono::milliseconds(50)
    ));

    publish_release(gate);
    EXPECT_TRUE(wait_until(
        [&] {
            return returned.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));

    // Unconditional second publish and join: the wait has no timeout, so this is
    // what keeps a failure above from leaving the thread spinning.
    publish_release(gate);
    worker.join();
}

// A stale release from the prior run cannot let a resident-startup failure ACK
// before AICPU publishes EXIT and resets this run's gate.
TEST(ReturnGateWait, StartupFailureWaitsForExitBeforeAckAndPostCloseReturn) {
    CoreRegs core;
    AicoreTeardownControl gate{};
    gate.post_close_release = AICORE_POST_CLOSE_RELEASE;
    AicoreExitTarget target{core.addr(), &gate};
    std::atomic<bool> entered_exit_wait{false};
    std::atomic<bool> acked{false};
    std::atomic<bool> returned{false};
    std::atomic<bool> saw_closed_window{false};

    std::thread worker([&] {
        sim_reg_base = core.base();
        entered_exit_wait.store(true, std::memory_order_release);
        wait_for_aicpu_exit_signal();
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
        acked.store(true, std::memory_order_release);
        wait_for_post_close_release(&gate.post_close_release);
        saw_closed_window.store(core.closed(), std::memory_order_release);
        returned.store(true, std::memory_order_release);
    });

    EXPECT_TRUE(wait_until(
        [&] {
            return entered_exit_wait.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(acked.load(std::memory_order_acquire));
    EXPECT_FALSE(returned.load(std::memory_order_acquire));
    EXPECT_EQ(__atomic_load_n(&gate.post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);

    // Model pre_handshake_init: reset the reused descriptor before the AICPU
    // retirement path can publish this run's EXIT register signal.
    __atomic_store_n(&gate.post_close_release, 0U, __ATOMIC_RELEASE);
    wmb();
    int32_t rc = -1;
    std::thread caller([&] {
        rc = platform_retire_aicore_group(&target, 1, platform_aicore_exit_deadline());
    });
    caller.join();
    if (rc != 0 && !returned.load(std::memory_order_acquire)) {
        platform_signal_aicore_exit(core.addr());
        wmb();
        core.ack();
        publish_release(gate);
    }
    EXPECT_EQ(rc, 0);
    EXPECT_TRUE(acked.load(std::memory_order_acquire));
    EXPECT_TRUE(core.closed());
    EXPECT_TRUE(wait_until(
        [&] {
            return returned.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));

    publish_release(gate);
    worker.join();
    EXPECT_TRUE(saw_closed_window.load(std::memory_order_acquire));
}

// The production retire path publishes the gate, and it must do so only after
// the whole group has acknowledged and every window has closed. The real sim
// wait is the consumer of that publication, so this pins the two halves
// together.
TEST(ReturnGateWait, RealSimWorkerIsHeldUntilTheGroupAcknowledges) {
    std::array<CoreRegs, 2> cores;
    std::array<AicoreTeardownControl, 2> gates{};
    AicoreExitTarget targets[] = {{cores[0].addr(), &gates[0]}, {cores[1].addr(), &gates[1]}};
    std::atomic<bool> worker_returned{false};
    std::atomic<bool> worker_saw_closed_windows{false};

    std::thread worker([&] {
        wait_for_post_close_release(&gates[0].post_close_release);
        const bool both_windows_closed = cores[0].closed() && cores[1].closed();
        worker_saw_closed_windows.store(both_windows_closed, std::memory_order_release);
        worker_returned.store(true, std::memory_order_release);
    });

    // HBG's resident shutdown has already broadcast EXIT before retirement
    // starts collecting acknowledgements and closing windows.
    platform_signal_aicore_exit(cores[0].addr());
    platform_signal_aicore_exit(cores[1].addr());
    wmb();
    cores[0].ack();
    int32_t rc = -1;
    std::thread caller([&] {
        rc = platform_retire_aicore_group(targets, 2, platform_aicore_exit_deadline());
    });

    // One pre-signalled core is still awaiting ACK. No target may publish its
    // return gate while the group remains in the collection phase.
    EXPECT_TRUE(wait_until(
        [&] {
            return cores[1].signalled();
        },
        std::chrono::milliseconds(500)
    ));
    // One core has not acknowledged, so no gate may be published yet and the
    // worker must still be waiting.
    EXPECT_FALSE(worker_returned.load(std::memory_order_acquire));
    EXPECT_EQ(__atomic_load_n(&gates[0].post_close_release, __ATOMIC_ACQUIRE), 0U);
    EXPECT_EQ(__atomic_load_n(&gates[1].post_close_release, __ATOMIC_ACQUIRE), 0U);

    cores[1].ack();
    caller.join();
    EXPECT_EQ(rc, 0);
    EXPECT_TRUE(cores[0].closed());
    EXPECT_TRUE(cores[1].closed());
    EXPECT_EQ(__atomic_load_n(&gates[0].post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
    EXPECT_EQ(__atomic_load_n(&gates[1].post_close_release, __ATOMIC_ACQUIRE), AICORE_POST_CLOSE_RELEASE);
    EXPECT_TRUE(wait_until(
        [&] {
            return worker_returned.load(std::memory_order_acquire);
        },
        std::chrono::seconds(2)
    ));

    publish_release(gates[0]);
    publish_release(gates[1]);
    worker.join();
    EXPECT_TRUE(worker_saw_closed_windows.load(std::memory_order_acquire));
}
