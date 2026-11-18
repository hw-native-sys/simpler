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
 * Owns the L0 live-capture state and the three AICPU entries the host/scheduler
 * drive: config (arm + program sampler registers), checkpoint (periodic cursor
 * sampling + range publication), and snapshot (terminal tail publication).
 */

#include <array>
#include <atomic>
#include <cstdint>

#include "aicpu/l0_dfx_aicpu.h"
#include "aicpu/platform_regs.h"
#include "common/l0_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"

namespace {

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
L0LiveCapture g_l0_live;
std::atomic<bool> g_l0_live_enabled{false};
// A disabled bind invalidates the previous run's device addresses. Terminal
// snapshots keep this valid while each producer publishes its own partition.
std::atomic<bool> g_l0_snapshot_valid{false};
std::array<std::atomic<bool>, PLATFORM_MAX_AICPU_THREADS> g_l0_snapshot_claimed{};
std::atomic<uint32_t> g_l0_snapshots_completed{0};

bool sample_l0_cursor(uint64_t reg, uint32_t *wptr, uint32_t *samp_wrt) {
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
}  // namespace

namespace simpler::dfx::l0 {
bool l0_live_capture_armed() { return g_l0_live_enabled.load(std::memory_order_acquire); }

bool l0_live_capture_matches(uint64_t run_epoch) {
    return g_l0_snapshot_valid.load(std::memory_order_acquire) && g_l0_live_enabled.load(std::memory_order_acquire) &&
           g_l0_live.dfx != nullptr && g_l0_live.run_epoch == run_epoch;
}

void l0_live_capture_set_armed(bool armed) {
    if (!armed) g_l0_snapshot_valid.store(false, std::memory_order_release);
    g_l0_live_enabled.store(armed, std::memory_order_release);
}

void l0_live_capture_note_core_type(uint32_t physical_core_id, bool is_aic) {
    const auto live = g_l0_live;
    if (!g_l0_live_enabled.load(std::memory_order_acquire) || live.dfx == nullptr ||
        physical_core_id >= live.core_count)
        return;
    l0_dfx_state(live.dfx, physical_core_id)->core_type = is_aic ? L0_DFX_CORE_TYPE_AIC : L0_DFX_CORE_TYPE_AIV;
}
}  // namespace simpler::dfx::l0

extern "C" void simpler_aicpu_l0_checkpoint(int thread_index) {
    if (!g_l0_live_enabled.load(std::memory_order_acquire)) return;
    const auto live = g_l0_live;
    constexpr uint32_t kOverwriteGuard = L0_DFX_TAIL_HEADROOM_BYTES;
    if (live.dfx == nullptr || live.ring_len <= kOverwriteGuard || thread_index < 0 ||
        thread_index >= static_cast<int>(live.dfx->num_threads)) {
        return;
    }
    if (g_l0_snapshot_claimed[thread_index].load(std::memory_order_acquire)) return;
    for (uint32_t core = static_cast<uint32_t>(thread_index); core < live.core_count; core += live.dfx->num_threads) {
        if (l0_slot_unavailable(live.initial[core]) || live.initial[core].preflight_status != L0_DFX_PREFLIGHT_OK) {
            continue;
        }
        auto *state = l0_dfx_state(live.dfx, core);
        if (state->capture_error != L0_DFX_CAPTURE_OK) continue;
        const uint32_t verify_request = state->verify_request;
        uint32_t now_samp_wrt = static_cast<uint32_t>(read_reg(live.regs[core], RegId::L0_SAMP_WRT));
        ++state->sample_count;
        state->last_observed_samp_wrt = now_samp_wrt;
        if (verify_request != state->verify_response) {
            state->verified_samp_wrt = now_samp_wrt;
            wmb();
            state->verify_response = verify_request;
        }
        if (now_samp_wrt < state->consumed_samp_wrt && now_samp_wrt <= live.ring_len &&
            state->consumed_samp_wrt <= live.ring_len) {
            state->capture_error = L0_DFX_CAPTURE_COUNTER_RESET;
            continue;
        }
        const uint32_t pending = now_samp_wrt - state->consumed_samp_wrt;
        if (pending >= live.ring_len - kOverwriteGuard) {
            state->capture_error = L0_DFX_CAPTURE_OVERRUN;
            continue;
        }
        const uint32_t safe_bytes = state->sampled_samp_wrt - state->published_samp_wrt;
        if (safe_bytes >= live.ring_len / 8) {
            const bool wrap_unambiguous = l0_dfx_wrap_unambiguous(
                state->consumed_samp_wrt, state->published_samp_wrt, state->sampled_samp_wrt, live.ring_len
            );
            const bool published =
                l0_dfx_publish_range(
                    live.dfx, thread_index, core, state->published_wptr, state->published_samp_wrt, state->sampled_wptr,
                    state->sampled_samp_wrt, wrap_unambiguous, live.ring_len, live.ring_capacity, L0_DFX_FLAG_LIVE
                ) == 0;
            if (!published && state->capture_error != L0_DFX_CAPTURE_OK) continue;
            if (published) ++state->live_batch_count;
        }
        uint32_t now_wptr = 0;
        if (!simpler::dfx::l0::derive_linear_wptr(
                state->sampled_wptr, state->sampled_samp_wrt, now_samp_wrt, live.ring_len, live.ring_capacity, &now_wptr
            )) {
            ++state->derive_fallback_count;
            if (!sample_l0_cursor(live.regs[core], &now_wptr, &now_samp_wrt)) continue;
            if (now_wptr > live.ring_capacity ||
                now_samp_wrt - state->consumed_samp_wrt >= live.ring_len - kOverwriteGuard) {
                state->capture_error = L0_DFX_CAPTURE_OVERRUN;
                continue;
            }
        }
        state->last_observed_wptr = now_wptr;
        state->last_observed_samp_wrt = now_samp_wrt;
        state->sampled_wptr = now_wptr;
        state->sampled_samp_wrt = now_samp_wrt;
        ++state->advance_count;
    }
    wmb();
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l0_config(void *arg) {
    g_l0_live_enabled.store(false, std::memory_order_release);
    g_l0_snapshot_valid.store(false, std::memory_order_release);
    for (auto &claimed : g_l0_snapshot_claimed)
        claimed.store(false, std::memory_order_relaxed);
    g_l0_snapshots_completed.store(0, std::memory_order_relaxed);
    if (arg == nullptr) return -1;
    const auto &config = *static_cast<const L0ConfigArgs *>(arg);
    if (config.reg_addr_table == 0 || config.gm_addr_table == 0 || config.snapshots == 0 || config.end_snapshots == 0 ||
        config.core_count == 0 || config.core_count > PLATFORM_MAX_CORES || config.buffer_len == 0 ||
        config.dfx_shm == 0 || config.buffer_capacity < config.buffer_len) {
        return -1;
    }
    auto *dfx = reinterpret_cast<L0DataHeader *>(config.dfx_shm);
    if (dfx->num_cores != config.core_count || dfx->num_threads == 0 || dfx->num_threads > PLATFORM_MAX_AICPU_THREADS) {
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
        state->consumed_wptr = snapshots[core].initial_wptr;
        state->consumed_samp_wrt = snapshots[core].initial_samp_wrt;
        state->published_wptr = snapshots[core].initial_wptr;
        state->published_samp_wrt = snapshots[core].initial_samp_wrt;
        state->sampled_wptr = snapshots[core].initial_wptr;
        state->sampled_samp_wrt = snapshots[core].initial_samp_wrt;
        state->capture_error = snapshots[core].preflight_status == L0_DFX_PREFLIGHT_UNSTABLE_CURSOR ?
                                   L0_DFX_CAPTURE_RANGE_ERROR :
                                   L0_DFX_CAPTURE_OK;
        state->batch_count = 0;
        state->live_batch_count = 0;
        state->sample_count = 0;
        state->advance_count = 0;
        state->derive_fallback_count = 0;
        state->last_observed_wptr = snapshots[core].initial_wptr;
        state->last_observed_samp_wrt = snapshots[core].initial_samp_wrt;
        state->verify_request = 0;
        state->verified_samp_wrt = snapshots[core].initial_samp_wrt;
        state->verify_response = 0;
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

extern "C" void simpler_aicpu_l0_snapshot(int thread_index, const int32_t *core_ids, int32_t core_num) {
    if (!g_l0_snapshot_valid.load(std::memory_order_acquire)) return;
    const auto live = g_l0_live;
    if (live.dfx == nullptr || live.end_snapshots == nullptr || thread_index < 0 ||
        thread_index >= static_cast<int>(live.dfx->num_threads)) {
        return;
    }
    // The scheduler and its run-exit fallback may both reach this partition.
    // Claim it once before touching registers or publishing its terminal range.
    if (g_l0_snapshot_claimed[thread_index].exchange(true, std::memory_order_acq_rel)) return;
    const auto *regs = live.regs;
    auto *out = live.end_snapshots;
    const uint32_t num_threads = live.dfx->num_threads;

    // The cores this call snapshots: the caller's retired-core list when given
    // (TMR passes its own CoreTracker), otherwise the round-robin slice of the
    // full table (single-producer HBG, where num_threads == 1).
    uint32_t cores[PLATFORM_MAX_CORES];
    uint32_t core_n = 0;
    if (core_ids != nullptr && core_num > 0) {
        for (int32_t i = 0; i < core_num; ++i) {
            const uint32_t core = static_cast<uint32_t>(core_ids[i]);
            if (core < live.core_count && regs[core] != 0) cores[core_n++] = core;
        }
    } else {
        for (uint32_t core = static_cast<uint32_t>(thread_index); core < live.core_count; core += num_threads) {
            if (regs[core] != 0) cores[core_n++] = core;
        }
    }

    std::array<bool, PLATFORM_MAX_CORES> producer_stopped{};
    for (uint32_t i = 0; i < core_n; ++i) {
        const uint32_t core = cores[i];
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
    for (uint32_t i = 0; i < core_n; ++i) {
        const uint32_t core = cores[i];
        const auto &start = initial[core];
        auto &end = out[core];
        auto *state = l0_dfx_state(dfx, core);
        state->last_observed_wptr = end.wptr;
        state->last_observed_samp_wrt = end.samp_wrt;
        state->verified_samp_wrt = end.samp_wrt;
        wmb();
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
        if (end.samp_wrt < state->consumed_samp_wrt && end.samp_wrt <= live.ring_len &&
            state->consumed_samp_wrt <= live.ring_len) {
            state->capture_error = L0_DFX_CAPTURE_COUNTER_RESET;
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        if (end.observed_base != start.requested_base || end.buffer_len != live.ring_len ||
            end.samp_wrt - state->consumed_samp_wrt >= live.ring_len) {
            state->capture_error = L0_DFX_CAPTURE_OVERRUN;
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        // Publish to this core's owning thread queue so its terminal range stays
        // ordered after its live ranges, which the checkpoint publishes to the
        // same queue.
        const int owner_thread = static_cast<int>(core % num_threads);
        const bool wrap_unambiguous =
            l0_dfx_wrap_unambiguous(state->consumed_samp_wrt, state->published_samp_wrt, end.samp_wrt, live.ring_len);
        if (l0_dfx_publish_range(
                dfx, owner_thread, core, state->published_wptr, state->published_samp_wrt, end.wptr, end.samp_wrt,
                wrap_unambiguous, live.ring_len, live.ring_capacity, 0
            ) != 0) {
            end.reserved = L0_DFX_STATUS_UNTRUSTWORTHY;
            continue;
        }
        end.reserved = L0_DFX_STATUS_STAGED;
    }
    wmb();
    if (g_l0_snapshots_completed.fetch_add(1, std::memory_order_acq_rel) + 1 == num_threads)
        g_l0_live_enabled.store(false, std::memory_order_release);
}
