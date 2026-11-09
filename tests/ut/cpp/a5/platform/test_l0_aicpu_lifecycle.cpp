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
#include <cstdint>

#include "common/l0_profiling.h"
#include "aicpu/platform_regs.h"

extern "C" int simpler_aicpu_l0_config(void *arg);

namespace {
std::array<std::array<uint64_t, 64>, 2> regs{};
uint32_t register_accesses = 0;
bool ignore_global_en_disable = false;
bool advance_counter_on_armed_wptr_read = false;
bool always_advance_counter_on_armed_wptr_read = false;

struct CaptureFixture {
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

    CaptureFixture() :
        dfx(reinterpret_cast<L0DataHeader *>(shared_storage.data())) {
        regs = {};
        register_accesses = 0;
        ignore_global_en_disable = false;
        advance_counter_on_armed_wptr_read = false;
        always_advance_counter_on_armed_wptr_read = false;
        dfx->num_cores = 2;
        dfx->num_threads = 2;
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
        ASSERT_EQ(simpler_aicpu_l0_config(&args), 0);
    }
};
}  // namespace

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    ++register_accesses;
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
    if (ignore_global_en_disable && reg == RegId::L0_GLOBAL_EN && value == 0) return;
    regs[reg_base_addr - 1][static_cast<uint8_t>(reg)] = value;
}

TEST(L0AicpuLifecycle, FailedGateDisableDoesNotClaimProducerStopped) {
    CaptureFixture capture;
    capture.configure();
    ignore_global_en_disable = true;

    simpler_aicpu_l0_snapshot(0, nullptr, 0);

    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 0u);
    EXPECT_EQ(capture.ends[0].reserved, L0_DFX_STATUS_UNTRUSTWORTHY);
}

TEST(L0AicpuLifecycle, DisabledNextRunDoesNotTouchPreviousCapture) {
    CaptureFixture capture;
    capture.configure();
    EXPECT_TRUE(simpler::dfx::l0::l0_live_capture_matches(42));
    simpler::dfx::l0::l0_live_capture_set_armed(false);
    EXPECT_FALSE(simpler::dfx::l0::l0_live_capture_matches(42));
    const uint32_t accesses_before = register_accesses;

    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    simpler_aicpu_l0_snapshot(1, nullptr, 0);

    EXPECT_EQ(register_accesses, accesses_before);
    EXPECT_EQ(capture.dfx->queue_tails[0], 0u);
    EXPECT_EQ(capture.dfx->queue_tails[1], 0u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 0u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->producer_stopped, 0u);
}

TEST(L0AicpuLifecycle, EveryProducerPublishesTerminalPartition) {
    CaptureFixture capture;
    capture.configure();
    for (auto &core_regs : regs) {
        core_regs[static_cast<uint8_t>(RegId::L0_WPTR)] = 16;
        core_regs[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 16;
    }

    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    EXPECT_TRUE(simpler::dfx::l0::l0_live_capture_matches(42));
    simpler_aicpu_l0_snapshot(1, nullptr, 0);
    EXPECT_FALSE(simpler::dfx::l0::l0_live_capture_matches(42));
    const uint32_t accesses_after_terminal_snapshots = register_accesses;
    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    simpler_aicpu_l0_snapshot(1, nullptr, 0);

    EXPECT_EQ(capture.dfx->queue_tails[0], 1u);
    EXPECT_EQ(capture.dfx->queue_tails[1], 1u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->producer_stopped, 1u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->producer_stopped, 1u);
    EXPECT_EQ(register_accesses, accesses_after_terminal_snapshots);
}

TEST(L0AicpuLifecycle, FirstTerminalPartitionLeavesOtherCheckpointLive) {
    CaptureFixture capture;
    capture.configure(2u << 20);

    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    EXPECT_TRUE(simpler::dfx::l0::l0_live_capture_matches(42));
    const uint32_t accesses_after_snapshot = register_accesses;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(register_accesses, accesses_after_snapshot);

    auto &other_core = regs[1];
    other_core[static_cast<uint8_t>(RegId::L0_WPTR)] = 128u << 10;
    other_core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 128u << 10;
    simpler_aicpu_l0_checkpoint(1);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 1)->sample_count, 1u);

    simpler_aicpu_l0_snapshot(1, nullptr, 0);
    EXPECT_FALSE(simpler::dfx::l0::l0_live_capture_matches(42));
}

TEST(L0AicpuLifecycle, NewConfigurationRearmsTerminalPartitions) {
    CaptureFixture first;
    first.configure();
    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    simpler_aicpu_l0_snapshot(1, nullptr, 0);

    CaptureFixture second;
    second.configure();
    const uint32_t accesses_before_terminal = register_accesses;
    simpler_aicpu_l0_snapshot(0, nullptr, 0);
    simpler_aicpu_l0_snapshot(1, nullptr, 0);

    EXPECT_GT(register_accesses, accesses_before_terminal);
    EXPECT_EQ(l0_dfx_state(second.dfx, 0)->producer_stopped, 1u);
    EXPECT_EQ(l0_dfx_state(second.dfx, 1)->producer_stopped, 1u);
}

TEST(L0AicpuLifecycle, CheckpointPublishesSealedLiveRange) {
    CaptureFixture capture;
    capture.configure(2u << 20);
    auto &core = regs[0];
    core[static_cast<uint8_t>(RegId::L0_WPTR)] = 300u << 10;
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 300u << 10;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(capture.dfx->queue_tails[0], 0u);

    core[static_cast<uint8_t>(RegId::L0_WPTR)] = 600u << 10;
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 600u << 10;
    simpler_aicpu_l0_checkpoint(0);
    ASSERT_EQ(capture.dfx->queue_tails[0], 1u);
    const auto &range = capture.dfx->queues[0][0];
    EXPECT_EQ(range.core_index, 0u);
    EXPECT_EQ(range.start_wptr, 0u);
    EXPECT_EQ(range.end_wptr, 300u << 10);
    EXPECT_EQ(range.flags, L0_DFX_FLAG_LIVE);
    const auto *state = l0_dfx_state(capture.dfx, 0);
    EXPECT_EQ(state->sample_count, 2u);
    EXPECT_EQ(state->live_batch_count, 1u);
    EXPECT_EQ(state->capture_error, L0_DFX_CAPTURE_OK);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}

TEST(L0AicpuLifecycle, CheckpointDerivesPointerFromOneCounterRead) {
    CaptureFixture capture;
    constexpr uint32_t ring_len = 2u << 20;
    capture.configure(ring_len, ring_len + L0_DFX_TAIL_HEADROOM_BYTES);
    auto &core = regs[0];
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 300u << 10;
    const uint32_t accesses_before = register_accesses;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(register_accesses, accesses_before + 1);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->sampled_wptr, 300u << 10);

    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 600u << 10;
    simpler_aicpu_l0_checkpoint(0);
    ASSERT_EQ(capture.dfx->queue_tails[0], 1u);
    EXPECT_EQ(capture.dfx->queues[0][0].end_wptr, 300u << 10);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->sampled_wptr, 600u << 10);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}

TEST(L0AicpuLifecycle, FullQueueDoesNotFreezeSampledCursor) {
    CaptureFixture capture;
    constexpr uint32_t ring_len = 2u << 20;
    capture.configure(ring_len, ring_len + L0_DFX_TAIL_HEADROOM_BYTES);
    auto &core = regs[0];
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 300u << 10;
    simpler_aicpu_l0_checkpoint(0);

    capture.dfx->queue_tails[0] = L0_DFX_READY_QUEUE_SIZE - 1;
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 600u << 10;
    simpler_aicpu_l0_checkpoint(0);
    const auto *state = l0_dfx_state(capture.dfx, 0);
    EXPECT_EQ(state->capture_error, L0_DFX_CAPTURE_OK);
    EXPECT_EQ(state->published_samp_wrt, 0u);
    EXPECT_EQ(state->sampled_samp_wrt, 600u << 10);
    EXPECT_EQ(state->sample_count, 2u);
    EXPECT_EQ(state->advance_count, 2u);

    capture.dfx->queue_heads[0] = 1;
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = 700u << 10;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(capture.dfx->queues[0][L0_DFX_READY_QUEUE_SIZE - 1].end_samp_wrt, 600u << 10);
    EXPECT_EQ(state->advance_count, 3u);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}

TEST(L0AicpuLifecycle, ConfigRetriesTornCursorAnchor) {
    CaptureFixture capture;
    advance_counter_on_armed_wptr_read = true;
    capture.configure(2u << 20, (2u << 20) + L0_DFX_TAIL_HEADROOM_BYTES);

    EXPECT_EQ(capture.starts[0].initial_wptr, 8u);
    EXPECT_EQ(capture.starts[0].initial_samp_wrt, 8u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->sampled_wptr, 8u);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->sampled_samp_wrt, 8u);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}

TEST(L0AicpuLifecycle, ConfigRejectsUnstableCursorAnchor) {
    CaptureFixture capture;
    always_advance_counter_on_armed_wptr_read = true;
    capture.configure(2u << 20, (2u << 20) + L0_DFX_TAIL_HEADROOM_BYTES);
    EXPECT_EQ(capture.starts[0].preflight_status, L0_DFX_PREFLIGHT_UNSTABLE_CURSOR);
    EXPECT_EQ(l0_dfx_state(capture.dfx, 0)->capture_error, L0_DFX_CAPTURE_RANGE_ERROR);

    const uint32_t accesses_before = register_accesses;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(register_accesses, accesses_before);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}

TEST(L0AicpuLifecycle, FailedFallbackDoesNotAdvanceCursor) {
    CaptureFixture capture;
    constexpr uint32_t ring_len = 2u << 20;
    const uint32_t anchor = ring_len - L0_DFX_TAIL_HEADROOM_BYTES - 8;
    regs[0][static_cast<uint8_t>(RegId::L0_WPTR)] = anchor;
    regs[0][static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = anchor;
    capture.configure(ring_len, ring_len + L0_DFX_TAIL_HEADROOM_BYTES);
    auto &core = regs[0];
    core[static_cast<uint8_t>(RegId::L0_WPTR)] = anchor + 8;
    core[static_cast<uint8_t>(RegId::L0_SAMP_WRT)] = anchor + 8;
    always_advance_counter_on_armed_wptr_read = true;

    simpler_aicpu_l0_checkpoint(0);
    const auto *state = l0_dfx_state(capture.dfx, 0);
    EXPECT_EQ(state->sample_count, 1u);
    EXPECT_EQ(state->derive_fallback_count, 1u);
    EXPECT_EQ(state->advance_count, 0u);
    EXPECT_EQ(state->sampled_samp_wrt, anchor);

    always_advance_counter_on_armed_wptr_read = false;
    simpler_aicpu_l0_checkpoint(0);
    EXPECT_EQ(state->sample_count, 2u);
    EXPECT_EQ(state->derive_fallback_count, 2u);
    EXPECT_EQ(state->advance_count, 1u);
    EXPECT_EQ(state->capture_error, L0_DFX_CAPTURE_OK);
    simpler::dfx::l0::l0_live_capture_set_armed(false);
}
