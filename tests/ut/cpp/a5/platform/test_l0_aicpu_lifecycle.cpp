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
#include <cstdint>
#include <functional>
#include <future>
#include <thread>

#include "common/l0_profiling.h"
#include "aicpu/platform_regs.h"

static std::array<std::array<uint64_t, 64>, 2> regs{};
static uint32_t register_accesses = 0;
static bool ignore_global_en_disable = false;
static bool advance_counter_on_armed_wptr_read = false;
static bool always_advance_counter_on_armed_wptr_read = false;
static int unavailable_core = -1;
static std::function<void(RegId)> register_read_hook;

struct L0CaptureFixture {
    static constexpr size_t kDfxStorageWords =
        (sizeof(L0DataHeader) + 2 * sizeof(L0BufferState) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    std::array<uint64_t, 2> register_addresses{1, 2};
    std::array<uint64_t, 2> ring_addresses{0x10000, 0x20000};
    std::array<L0CoreRegSnapshot, 2> starts{};
    std::array<L0CoreEndSnapshot, 2> ends{};
    // L0DataHeader and L0BufferState are alignas(64); align the backing store to
    // match so the reinterpret_cast to L0DataHeader* is well-aligned.
    alignas(64) std::array<uint64_t, kDfxStorageWords> shared_storage{};
    L0DataHeader *dfx;

    L0CaptureFixture() :
        dfx(reinterpret_cast<L0DataHeader *>(shared_storage.data())) {
        regs = {};
        register_accesses = 0;
        ignore_global_en_disable = false;
        advance_counter_on_armed_wptr_read = false;
        always_advance_counter_on_armed_wptr_read = false;
        unavailable_core = -1;
        register_read_hook = {};
        dfx->num_cores = 2;
    }

    void configure(uint32_t ring_len = 64, uint32_t ring_capacity = 0) {
        L0ConfigArgs args{};
        args.reg_addr_table = reinterpret_cast<uint64_t>(register_addresses.data());
        args.gm_addr_table = reinterpret_cast<uint64_t>(ring_addresses.data());
        args.snapshots = reinterpret_cast<uint64_t>(starts.data());
        args.end_snapshots = reinterpret_cast<uint64_t>(ends.data());
        args.dfx_shm = reinterpret_cast<uint64_t>(dfx);
        args.run_epoch = 42;
        args.core_count = 2;
        args.buffer_len = ring_len;
        args.buffer_capacity = ring_capacity == 0 ? ring_len : ring_capacity;
        ASSERT_EQ(configure_l0_capture(args), 0);
        set_l0_swimlane_enabled(true, args.run_epoch);
    }
};

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    if (register_read_hook) register_read_hook(reg);
    ++register_accesses;
    if (reg_base_addr - 1 == static_cast<uint64_t>(unavailable_core)) return UINT64_MAX;
    auto &core = regs[reg_base_addr - 1];
    const uint64_t value = core[static_cast<uint8_t>(reg)];
    if ((advance_counter_on_armed_wptr_read || always_advance_counter_on_armed_wptr_read) && reg == RegId::L0_WPTR &&
        core[static_cast<uint8_t>(RegId::L0_GLOBAL_EN)] == 1) {
        core[static_cast<uint8_t>(RegId::L0_WPTR)] += 8;
        core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] += 8;
        advance_counter_on_armed_wptr_read = false;
    }
    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    ++register_accesses;
    if (reg_base_addr - 1 == static_cast<uint64_t>(unavailable_core)) return;
    if (ignore_global_en_disable && reg == RegId::L0_GLOBAL_EN && value == 0) return;
    regs[reg_base_addr - 1][static_cast<uint8_t>(reg)] = value;
}

TEST(L0AicpuLifecycle, FailedGateDisableDoesNotClaimProducerStopped) {
    L0CaptureFixture capture;
    capture.configure();
    ignore_global_en_disable = true;

    simpler_aicpu_l0_snapshot();

    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 0u);
    EXPECT_EQ(capture.ends[0].reserved, L0_DFX_STATUS_UNTRUSTWORTHY);
}

TEST(L0AicpuLifecycle, TerminalSnapshotCoversAllPhysicalSlotsAfterProducerRetirement) {
    L0CaptureFixture capture;
    capture.configure();
    for (auto &core_regs : regs) {
        core_regs[static_cast<uint8_t>(RegId::L0_WPTR)] = 16;
        core_regs[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 16;
    }

    simpler_aicpu_l0_snapshot();

    for (uint32_t core = 0; core < 2; ++core) {
        EXPECT_EQ(l0_dfx_state(capture.dfx, core)->producer_stopped, 1u);
        EXPECT_EQ(capture.ends[core].reserved, L0_DFX_STATUS_STAGED);
    }
    EXPECT_FALSE(is_l0_swimlane_enabled());
}

TEST(L0AicpuLifecycle, TerminalSnapshotMarksUnavailablePhysicalSlotsComplete) {
    L0CaptureFixture capture;
    unavailable_core = 1;
    capture.configure();
    ASSERT_TRUE(l0_slot_unavailable(capture.starts[1]));
    regs[0][static_cast<uint8_t>(RegId::L0_WPTR)] = 16;
    regs[0][static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 16;

    simpler_aicpu_l0_snapshot();

    EXPECT_EQ(capture.ends[0].reserved, L0_DFX_STATUS_STAGED);
    EXPECT_EQ(capture.ends[1].reserved, L0_DFX_STATUS_UNAVAILABLE);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->producer_stopped, 1u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->capture_error, L0_DFX_CAPTURE_OK);
}

TEST(L0AicpuLifecycle, ConcurrentTerminalRequestDoesNotTouchAnOwnedSnapshot) {
    L0CaptureFixture capture;
    capture.configure();
    std::promise<void> reader_entered;
    auto entered = reader_entered.get_future();
    std::promise<void> resume_reader;
    auto resume = resume_reader.get_future();
    std::atomic<bool> announced{false};
    register_read_hook = [&](RegId reg) {
        if (reg == RegId::L0_EN && !announced.exchange(true)) {
            reader_entered.set_value();
            resume.wait();
        }
    };
    std::thread owner([] {
        simpler_aicpu_l0_snapshot();
    });
    const bool reached_read = entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    if (reached_read) {
        const uint32_t accesses_before = register_accesses;
        simpler_aicpu_l0_snapshot();
        EXPECT_EQ(register_accesses, accesses_before);
    }
    resume_reader.set_value();
    owner.join();
    register_read_hook = {};
    ASSERT_TRUE(reached_read);
    EXPECT_EQ(capture.ends[0].reserved, L0_DFX_STATUS_STAGED);
    EXPECT_EQ(capture.ends[1].reserved, L0_DFX_STATUS_STAGED);
}

TEST(L0AicpuLifecycle, DisabledNextRunDoesNotTouchPreviousCapture) {
    L0CaptureFixture capture;
    capture.configure();
    EXPECT_TRUE(is_l0_swimlane_enabled());
    set_l0_swimlane_enabled(false, 0);
    EXPECT_FALSE(is_l0_swimlane_enabled());
    const uint32_t accesses_before = register_accesses;

    simpler_aicpu_l0_snapshot();

    EXPECT_EQ(register_accesses, accesses_before);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 0u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->producer_stopped, 0u);
}

TEST(L0AicpuLifecycle, MatchingRunBindingsPreserveLiveCapture) {
    L0CaptureFixture capture;
    capture.configure(2u << 20);

    for (int i = 0; i < 3; ++i) {
        set_l0_swimlane_enabled(true, 42);
        EXPECT_TRUE(is_l0_swimlane_enabled());
    }

    set_l0_swimlane_enabled(false, 0);
}

TEST(L0AicpuLifecycle, MismatchedRunBindingDoesNotTouchPreviousCapture) {
    L0CaptureFixture capture;
    capture.configure(2u << 20);
    const uint32_t accesses_before = register_accesses;

    set_l0_swimlane_enabled(true, 43);
    simpler_aicpu_l0_snapshot();

    EXPECT_FALSE(is_l0_swimlane_enabled());
    EXPECT_EQ(register_accesses, accesses_before);
}

TEST(L0AicpuLifecycle, DisabledCaptureCannotRearmWithoutConfiguration) {
    L0CaptureFixture capture;
    capture.configure();
    set_l0_swimlane_enabled(false, 0);
    const uint32_t accesses_before = register_accesses;

    set_l0_swimlane_enabled(true, 42);
    simpler_aicpu_l0_snapshot();

    EXPECT_FALSE(is_l0_swimlane_enabled());
    EXPECT_EQ(register_accesses, accesses_before);
}

TEST(L0AicpuLifecycle, CompletedCaptureCannotRearmWithoutConfiguration) {
    L0CaptureFixture capture;
    capture.configure();
    simpler_aicpu_l0_snapshot();
    const uint32_t accesses_before = register_accesses;

    set_l0_swimlane_enabled(true, 42);
    simpler_aicpu_l0_snapshot();

    EXPECT_FALSE(is_l0_swimlane_enabled());
    EXPECT_EQ(register_accesses, accesses_before);
}

TEST(L0AicpuLifecycle, TerminalSnapshotIsIdempotentAcrossRepeatedRequests) {
    L0CaptureFixture capture;
    capture.configure();
    for (auto &core_regs : regs) {
        core_regs[static_cast<uint8_t>(RegId::L0_WPTR)] = 16;
        core_regs[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 16;
    }

    simpler_aicpu_l0_snapshot();
    EXPECT_FALSE(is_l0_swimlane_enabled());
    const uint32_t accesses_after_terminal_snapshots = register_accesses;
    simpler_aicpu_l0_snapshot();
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 1u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->producer_stopped, 1u);
    EXPECT_EQ(register_accesses, accesses_after_terminal_snapshots);
}

TEST(L0AicpuLifecycle, NewConfigurationRearmsTerminalSnapshot) {
    L0CaptureFixture first;
    first.configure();
    simpler_aicpu_l0_snapshot();

    L0CaptureFixture second;
    second.configure();
    const uint32_t accesses_before_terminal = register_accesses;
    simpler_aicpu_l0_snapshot();

    EXPECT_GT(register_accesses, accesses_before_terminal);
    EXPECT_EQ(l0_dfx_state(second.dfx, 0)->producer_stopped, 1u);
    EXPECT_EQ(l0_dfx_state(second.dfx, 1)->producer_stopped, 1u);
}

TEST(L0AicpuLifecycle, ConfigRetriesTornCursorAnchor) {
    L0CaptureFixture capture;
    advance_counter_on_armed_wptr_read = true;
    capture.configure(2u << 20, (2u << 20) + L0_DFX_TAIL_HEADROOM_BYTES);

    EXPECT_EQ(capture.starts[0].initial_wptr, 8u);
    EXPECT_EQ(capture.starts[0].initial_samp_wrt, 8u);
    set_l0_swimlane_enabled(false, 0);
}

TEST(L0AicpuLifecycle, ConfigRejectsUnstableCursorAnchor) {
    L0CaptureFixture capture;
    always_advance_counter_on_armed_wptr_read = true;
    capture.configure(2u << 20, (2u << 20) + L0_DFX_TAIL_HEADROOM_BYTES);
    EXPECT_EQ(capture.starts[0].preflight_status, L0_DFX_PREFLIGHT_UNSTABLE_CURSOR);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->capture_error, L0_DFX_CAPTURE_RANGE_ERROR);

    set_l0_swimlane_enabled(false, 0);
}

TEST(L0RingRange, DetectsOverwriteAndSplitsWrappedCopy) {
    const auto first = l0_plan_ring_range(0, 0, 256, 256, 1024);
    EXPECT_TRUE(first.trustworthy());
    EXPECT_EQ(first.start, 0u);
    EXPECT_EQ(first.first_bytes, 256u);
    EXPECT_EQ(first.second_bytes, 0u);

    const auto nonzero_start = l0_plan_ring_range(128, 100, 356, 384, 1024);
    EXPECT_TRUE(nonzero_start.trustworthy());
    EXPECT_EQ(nonzero_start.start, 128u);
    EXPECT_EQ(nonzero_start.first_bytes, 256u);
    EXPECT_EQ(nonzero_start.second_bytes, 0u);

    const auto full = l0_plan_ring_range(128, 0, 1024, 128, 1024);
    EXPECT_TRUE(full.trustworthy());
    EXPECT_EQ(full.bytes, 1024u);
    EXPECT_EQ(full.start, 128u);

    const auto overwritten = l0_plan_ring_range(128, 0, 1280, 384, 1024);
    EXPECT_FALSE(overwritten.trustworthy());
    EXPECT_EQ(overwritten.lost_bytes, 256u);
    EXPECT_EQ(overwritten.start, 384u);
    EXPECT_EQ(overwritten.first_bytes, 640u);
    EXPECT_EQ(overwritten.second_bytes, 384u);

    const auto starts_near_end = l0_plan_ring_range(900, 100, 400, 176, 1024);
    EXPECT_TRUE(starts_near_end.trustworthy());
    EXPECT_EQ(starts_near_end.start, 900u);
    EXPECT_EQ(starts_near_end.first_bytes, 124u);
    EXPECT_EQ(starts_near_end.second_bytes, 176u);

    const auto short_tail = l0_plan_ring_range(900, 0, 200, 100, 1024);
    EXPECT_TRUE(short_tail.trustworthy());
    EXPECT_TRUE(short_tail.wrapped);
    EXPECT_EQ(short_tail.tail_unused_bytes, 24u);
    EXPECT_EQ(short_tail.start, 900u);
    EXPECT_EQ(short_tail.first_bytes, 100u);
    EXPECT_EQ(short_tail.second_bytes, 100u);

    // The counter can advance 32 bytes farther than the programmed ring length
    // when a record reaches the guard. Those bytes are still part of GM.
    const auto guarded_tail = l0_plan_ring_range(900, 0, 160, 4, 1024, 1088);
    EXPECT_TRUE(guarded_tail.trustworthy());
    EXPECT_EQ(guarded_tail.first_bytes, 156u);
    EXPECT_EQ(guarded_tail.second_bytes, 4u);
    EXPECT_EQ(guarded_tail.bytes, 160u);

    const auto early_overwrite = l0_plan_ring_range(900, 0, 970, 920, 1024);
    EXPECT_FALSE(early_overwrite.trustworthy());
    EXPECT_EQ(early_overwrite.tail_unused_bytes, 74u);
    EXPECT_EQ(early_overwrite.lost_bytes, 20u);
    EXPECT_EQ(early_overwrite.start, 920u);
    EXPECT_EQ(early_overwrite.first_bytes, 30u);
    EXPECT_EQ(early_overwrite.second_bytes, 920u);

    EXPECT_TRUE(l0_plan_ring_range(128, 0, 256, 300, 1024).pointer_mismatch);
    EXPECT_TRUE(l0_plan_ring_range(0, 0, 2500, 452, 1024).pointer_mismatch);
    EXPECT_TRUE(l0_plan_ring_range(0, UINT32_MAX - 3, 4, 8, 1024).counter_wrapped);
    EXPECT_TRUE(l0_plan_ring_range(0, 0, 256, 1025, 1024).invalid_pointer);
    EXPECT_TRUE(l0_plan_ring_range(1025, 0, 256, 256, 1024).invalid_pointer);
}
