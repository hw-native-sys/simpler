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
 * @file l0_collector_aicpu.cpp
 * @brief AICPU-side L0 hardware-sampler capture (a5)
 *
 * Configures sampler registers and snapshots the final cursors after every
 * AICore has acknowledged retirement. Host copies follow the sampling stop.
 */

#include <array>
#include <atomic>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "common/l0_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"

struct L0LiveCapture {
    const uint64_t *regs{nullptr};
    const uint64_t *buffers{nullptr};
    const L0CoreRegSnapshot *initial{nullptr};
    L0CoreEndSnapshot *end_snapshots{nullptr};
    L0DataHeader *dfx{nullptr};
    uint64_t run_epoch{0};
    uint32_t core_count{0};
    uint32_t ring_len{0};
    uint32_t ring_capacity{0};
};
static L0LiveCapture g_l0_live;
static std::atomic<bool> g_l0_live_enabled{false};
// A disabled bind invalidates the previous run's device addresses.
static std::atomic<bool> g_l0_snapshot_valid{false};
static std::atomic<bool> g_l0_snapshot_claimed{false};

static bool sample_l0_cursor(uint64_t reg, uint32_t *wptr, uint32_t *samp_wrt) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        const uint32_t before = static_cast<uint32_t>(read_reg(reg, RegId::L0_SAMP_WRT));
        const uint32_t pointer = static_cast<uint32_t>(read_reg(reg, RegId::L0_WPTR));
        const uint32_t after = static_cast<uint32_t>(read_reg(reg, RegId::L0_SAMP_WRT));
        if (before == after) {
            *wptr = pointer;
            *samp_wrt = after;
            return true;
        }
    }
    return false;
}

extern "C" bool is_l0_swimlane_enabled() { return g_l0_live_enabled.load(std::memory_order_acquire); }

extern "C" void set_l0_swimlane_enabled(bool enable, uint64_t run_epoch) {
    const bool armed = enable && g_l0_snapshot_valid.load(std::memory_order_acquire) && is_l0_swimlane_enabled() &&
                       g_l0_live.dfx != nullptr && g_l0_live.run_epoch == run_epoch;
    if (!armed) g_l0_snapshot_valid.store(false, std::memory_order_release);
    g_l0_live_enabled.store(armed, std::memory_order_release);
}

void l0_live_capture_note_core_type(uint32_t physical_core_id, bool is_aic) {
    if (!g_l0_live_enabled.load(std::memory_order_acquire)) return;
    const auto &live = g_l0_live;
    if (live.dfx == nullptr || physical_core_id >= live.core_count) return;
    l0_dfx_state(live.dfx, physical_core_id)->core_type = is_aic ? L0_DFX_CORE_TYPE_AIC : L0_DFX_CORE_TYPE_AIV;
}

int configure_l0_capture(const L0ConfigArgs &config) {
    g_l0_live_enabled.store(false, std::memory_order_release);
    g_l0_snapshot_valid.store(false, std::memory_order_release);
    g_l0_snapshot_claimed.store(false, std::memory_order_relaxed);
    if (config.reg_addr_table == 0 || config.gm_addr_table == 0 || config.snapshots == 0 || config.end_snapshots == 0 ||
        config.core_count == 0 || config.core_count > PLATFORM_MAX_CORES || config.buffer_len == 0 ||
        config.dfx_shm == 0 || config.buffer_capacity < config.buffer_len) {
        return -1;
    }
    auto *dfx = reinterpret_cast<L0DataHeader *>(config.dfx_shm);
    if (dfx->num_cores != config.core_count) {
        return -1;
    }

    const auto *regs = reinterpret_cast<const uint64_t *>(config.reg_addr_table);
    const auto *buffers = reinterpret_cast<const uint64_t *>(config.gm_addr_table);
    auto *snapshots = reinterpret_cast<L0CoreRegSnapshot *>(config.snapshots);
    // Record inherited WPTR before configuring every addressable physical
    // sub-core. Values above buf_len are diagnostic, not a reason to omit the
    // base/len/global_en write; the host probes those GM buffers separately.
    // An invalid address table still rejects the whole configuration.
    bool invalid_address = false;
    for (uint32_t i = 0; i < config.core_count; ++i) {
        auto &snapshot = snapshots[i];
        if (regs[i] == 0 || buffers[i] == 0) {
            snapshot.preflight_status = L0_DFX_PREFLIGHT_MISSING_ADDRESS;
            invalid_address = true;
            continue;
        }
        snapshot.pre_config_wptr = static_cast<uint32_t>(read_reg(regs[i], RegId::L0_WPTR));
        if (snapshot.pre_config_wptr > config.buffer_capacity) {
            snapshot.preflight_status = L0_DFX_PREFLIGHT_WPTR_OUT_OF_RANGE;
        }
    }
    if (invalid_address) {
        wmb();
        return 0;
    }
    for (uint32_t i = 0; i < config.core_count; ++i) {
        const uint64_t reg = regs[i];
        const uint64_t buffer = buffers[i];
        write_reg(reg, RegId::L0_BASE_LO, static_cast<uint32_t>(buffer));
        write_reg(reg, RegId::L0_BASE_HI, static_cast<uint32_t>(buffer >> 32));
        write_reg(reg, RegId::L0_BUF_LEN, config.buffer_len);
        write_reg(reg, RegId::L0_GLITCH_FILTER, 2);
        write_reg(reg, RegId::L0_GLOBAL_EN, 1);

        L0CoreRegSnapshot &snapshot = snapshots[i];
        snapshot.requested_base = buffer;
        snapshot.observed_base = read_reg(reg, RegId::L0_BASE_LO) | (read_reg(reg, RegId::L0_BASE_HI) << 32);
        snapshot.observed_len = static_cast<uint32_t>(read_reg(reg, RegId::L0_BUF_LEN));
        snapshot.observed_global_en = static_cast<uint32_t>(read_reg(reg, RegId::L0_GLOBAL_EN));
        snapshot.observed_en = static_cast<uint32_t>(read_reg(reg, RegId::L0_EN));
        if (!sample_l0_cursor(reg, &snapshot.initial_wptr, &snapshot.initial_samp_wrt)) {
            snapshot.preflight_status = L0_DFX_PREFLIGHT_UNSTABLE_CURSOR;
        }
    }
    for (uint32_t core = 0; core < config.core_count; ++core) {
        auto *state = l0_dfx_state(dfx, core);
        state->capture_error = snapshots[core].preflight_status == L0_DFX_PREFLIGHT_UNSTABLE_CURSOR ?
                                   L0_DFX_CAPTURE_RANGE_ERROR :
                                   L0_DFX_CAPTURE_OK;
        state->producer_stopped = 0;
        state->core_type = L0_DFX_CORE_TYPE_UNKNOWN;
    }
    g_l0_live = L0LiveCapture{
        regs,
        buffers,
        snapshots,
        reinterpret_cast<L0CoreEndSnapshot *>(config.end_snapshots),
        dfx,
        config.run_epoch,
        config.core_count,
        config.buffer_len,
        config.buffer_capacity
    };
    g_l0_snapshot_valid.store(true, std::memory_order_release);
    g_l0_live_enabled.store(true, std::memory_order_release);
    wmb();
    return 0;
}

extern "C" void simpler_aicpu_l0_snapshot() {
    if (!g_l0_snapshot_valid.load(std::memory_order_acquire)) return;
    const auto live = g_l0_live;
    if (live.dfx == nullptr || live.end_snapshots == nullptr) return;
    if (g_l0_snapshot_claimed.exchange(true, std::memory_order_acq_rel)) return;
    g_l0_live_enabled.store(false, std::memory_order_release);
    const auto *regs = live.regs;
    auto *out = live.end_snapshots;

    std::array<bool, PLATFORM_MAX_CORES> producer_stopped{};
    for (uint32_t core = 0; core < live.core_count; ++core) {
        if (l0_slot_unavailable(live.initial[core])) continue;
        const uint64_t reg = regs[core];
        auto &snapshot = out[core];
        snapshot.en = static_cast<uint32_t>(read_reg(reg, RegId::L0_EN));
        snapshot.global_en = static_cast<uint32_t>(read_reg(reg, RegId::L0_GLOBAL_EN));
        write_reg(reg, RegId::L0_GLOBAL_EN, 0);
        const bool stopped = (read_reg(reg, RegId::L0_GLOBAL_EN) & 1u) == 0;
        producer_stopped[core] = stopped;
        snapshot.observed_base = read_reg(reg, RegId::L0_BASE_LO) | (read_reg(reg, RegId::L0_BASE_HI) << 32);
        if (!stopped || !sample_l0_cursor(reg, &snapshot.wptr, &snapshot.samp_wrt)) snapshot.global_en = 0;
        snapshot.buffer_len = static_cast<uint32_t>(read_reg(reg, RegId::L0_BUF_LEN));
        snapshot.reserved = L0_DFX_STATUS_SKIPPED;
    }
    wmb();

    auto *dfx = live.dfx;
    const auto *initial = live.initial;
    const auto *buffers = live.buffers;
    for (uint32_t core = 0; core < live.core_count; ++core) {
        const auto &start = initial[core];
        auto &end = out[core];
        auto *state = l0_dfx_state(dfx, core);
        state->producer_stopped = producer_stopped[core] || l0_slot_unavailable(start);
        if (l0_slot_unavailable(start)) {
            end.reserved = L0_DFX_STATUS_UNAVAILABLE;
            continue;
        }
        if (buffers[core] == 0) {
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        if ((end.global_en & 1u) == 0) {
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        if (state->capture_error != L0_DFX_CAPTURE_OK) {
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        if (start.preflight_status != L0_DFX_PREFLIGHT_OK || start.initial_wptr > live.ring_capacity ||
            end.wptr > live.ring_capacity) {
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        const auto range = l0_plan_ring_range(
            start.initial_wptr, start.initial_samp_wrt, end.samp_wrt, end.wptr, live.ring_len, live.ring_capacity
        );
        if (end.observed_base != start.requested_base || end.buffer_len != live.ring_len || !range.trustworthy() ||
            range.bytes >= live.ring_len) {
            state->capture_error = L0_DFX_CAPTURE_RANGE_ERROR;
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        end.reserved = L0_DFX_STATUS_STAGED;
    }
    wmb();
}
