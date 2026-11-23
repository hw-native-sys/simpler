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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/platform_config.h"

struct L0ConfigArgs {
    uint64_t reg_addr_table{0};
    uint64_t gm_addr_table{0};
    uint64_t snapshots{0};
    uint64_t end_snapshots{0};
    uint64_t dfx_shm{0};
    uint64_t run_epoch{0};
    uint32_t core_count{0};
    uint32_t buffer_len{0};
    uint32_t buffer_capacity{0};
    uint32_t reserved{0};
};

constexpr uint32_t L0_DFX_PREFLIGHT_OK = 0;
constexpr uint32_t L0_DFX_PREFLIGHT_MISSING_ADDRESS = 1;
constexpr uint32_t L0_DFX_PREFLIGHT_WPTR_OUT_OF_RANGE = 2;
constexpr uint32_t L0_DFX_PREFLIGHT_UNSTABLE_CURSOR = 3;

struct L0CoreRegSnapshot {
    uint64_t requested_base{0};
    uint64_t observed_base{0};
    uint32_t observed_len{0};
    uint32_t observed_global_en{0};
    uint32_t observed_en{0};
    uint32_t pre_config_wptr{0};
    uint32_t initial_wptr{0};
    uint32_t initial_samp_wrt{0};
    uint32_t preflight_status{L0_DFX_PREFLIGHT_OK};
};

inline bool l0_slot_unavailable(const L0CoreRegSnapshot &snapshot) {
    const bool all_ones = snapshot.pre_config_wptr == UINT32_MAX && snapshot.initial_wptr == UINT32_MAX &&
                          snapshot.observed_base == UINT64_MAX && snapshot.observed_len == UINT32_MAX &&
                          snapshot.observed_global_en == UINT32_MAX;
    const bool unwritable_zero = snapshot.requested_base != 0 && snapshot.pre_config_wptr == 0 &&
                                 snapshot.initial_wptr == 0 && snapshot.initial_samp_wrt == 0 &&
                                 snapshot.observed_base == 0 && snapshot.observed_len == 0 &&
                                 snapshot.observed_global_en == 0 && snapshot.observed_en == 0;
    return all_ones || unwritable_zero;
}

constexpr uint32_t L0_DFX_STATUS_SKIPPED = 0;
constexpr uint32_t L0_DFX_STATUS_STAGED = 1;
constexpr uint32_t L0_DFX_STATUS_UNAVAILABLE = 2;
constexpr uint32_t L0_DFX_STATUS_UNTRUSTWORTHY = 3;
constexpr uint32_t L0_DFX_CORE_TYPE_UNKNOWN = 0;
constexpr uint32_t L0_DFX_CORE_TYPE_AIC = 1;
constexpr uint32_t L0_DFX_CORE_TYPE_AIV = 2;

struct L0CoreEndSnapshot {
    uint64_t observed_base{0};
    uint32_t wptr{0};
    uint32_t samp_wrt{0};
    uint32_t en{0};
    uint32_t global_en{0};
    uint32_t buffer_len{0};
    uint32_t reserved{0};
};

static_assert(std::is_trivially_copyable_v<L0ConfigArgs> && std::is_standard_layout_v<L0ConfigArgs>);
static_assert(std::is_trivially_copyable_v<L0CoreRegSnapshot> && std::is_standard_layout_v<L0CoreRegSnapshot>);
static_assert(std::is_trivially_copyable_v<L0CoreEndSnapshot> && std::is_standard_layout_v<L0CoreEndSnapshot>);
static_assert(sizeof(L0ConfigArgs) == 64 && sizeof(L0CoreRegSnapshot) == 48);
static_assert(sizeof(L0CoreEndSnapshot) == 32);

// Per-core metadata is read only after sampling has stopped.
constexpr uint32_t L0_DFX_TAIL_HEADROOM_BYTES = 64u << 10;

struct alignas(64) L0BufferState {
    uint32_t physical_core_id;
    uint32_t capture_error;
    volatile uint32_t producer_stopped;
    volatile uint32_t core_type;
};

constexpr uint32_t L0_DFX_CAPTURE_OK = 0;
constexpr uint32_t L0_DFX_CAPTURE_OVERRUN = 1;
constexpr uint32_t L0_DFX_CAPTURE_RANGE_ERROR = 2;
constexpr uint32_t L0_DFX_CAPTURE_COUNTER_RESET = 3;

struct alignas(32) L0RangeEntry {
    uint32_t core_index;
    uint32_t start_wptr;
    uint32_t end_wptr;
    uint32_t start_samp_wrt;
    uint32_t end_samp_wrt;
    uint32_t sequence;
    uint32_t flags;
};

struct alignas(64) L0DataHeader {
    uint32_t num_cores;
    uint32_t num_threads;
};

inline size_t l0_dfx_data_size(uint32_t num_cores) {
    return sizeof(L0DataHeader) + static_cast<size_t>(num_cores) * sizeof(L0BufferState);
}

inline L0BufferState *l0_dfx_state(void *base, uint32_t core_index) {
    return reinterpret_cast<L0BufferState *>(static_cast<uint8_t *>(base) + sizeof(L0DataHeader)) + core_index;
}

static_assert(std::is_standard_layout_v<L0BufferState> && std::is_trivially_copyable_v<L0BufferState>);
static_assert(sizeof(L0BufferState) == 64);
static_assert(std::is_standard_layout_v<L0RangeEntry> && std::is_trivially_copyable_v<L0RangeEntry>);
static_assert(std::is_standard_layout_v<L0DataHeader> && std::is_trivially_copyable_v<L0DataHeader>);
static_assert(sizeof(L0RangeEntry) == 32);

// Ring range calculation shared by AICPU and Host.

struct L0RingRange {
    uint32_t start{0};
    uint32_t bytes{0};
    uint32_t first_bytes{0};
    uint32_t second_bytes{0};
    uint32_t lost_bytes{0};
    uint32_t tail_unused_bytes{0};
    bool wrapped{false};
    bool counter_wrapped{false};
    bool invalid_pointer{false};
    bool pointer_mismatch{false};
    bool possible_overrun{false};

    bool trustworthy() const {
        return !counter_wrapped && !invalid_pointer && !pointer_mismatch && !possible_overrun && lost_bytes == 0;
    }
};

// samp_wrt counts written bytes, while a variable-size timestamp record may
// force WPTR to wrap before reaching ring_len or extend into the allocated
// guard after it. For one wrap, the written tail ends at initial_wptr +
// produced - end_wptr. The GM allocation, not the programmed ring length,
// bounds that final record. Multiple wraps cannot be reconstructed from two
// snapshots and must be rejected by the caller's occupancy check.
inline L0RingRange l0_plan_ring_range(
    uint32_t initial_wptr, uint32_t initial_samp_wrt, uint32_t end_samp_wrt, uint32_t end_wptr, uint32_t ring_len,
    uint32_t ring_capacity
) {
    L0RingRange result{};
    if (ring_len == 0 || ring_capacity < ring_len || initial_wptr > ring_capacity || end_wptr > ring_capacity) {
        result.invalid_pointer = true;
        return result;
    }
    result.counter_wrapped = end_samp_wrt < initial_samp_wrt;
    const uint32_t produced = end_samp_wrt - initial_samp_wrt;
    result.possible_overrun = produced > ring_len;
    const uint64_t linear_end = static_cast<uint64_t>(initial_wptr) + produced;
    if (linear_end == end_wptr) {
        result.start = initial_wptr;
        result.bytes = produced;
        result.first_bytes = produced;
        return result;
    }
    if (linear_end < end_wptr) {
        result.pointer_mismatch = true;
        return result;
    }
    const uint64_t wrap_end = linear_end - end_wptr;
    if (wrap_end < initial_wptr || wrap_end > ring_capacity) {
        result.pointer_mismatch = true;
        return result;
    }
    result.wrapped = true;
    result.tail_unused_bytes = wrap_end < ring_len ? ring_len - static_cast<uint32_t>(wrap_end) : 0;
    result.start = std::max(initial_wptr, end_wptr);
    result.first_bytes = result.start < wrap_end ? static_cast<uint32_t>(wrap_end - result.start) : 0;
    result.second_bytes = end_wptr;
    result.bytes = result.first_bytes + result.second_bytes;
    result.lost_bytes = produced - result.bytes;
    return result;
}

inline L0RingRange l0_plan_ring_range(
    uint32_t initial_wptr, uint32_t initial_samp_wrt, uint32_t end_samp_wrt, uint32_t end_wptr, uint32_t ring_len
) {
    return l0_plan_ring_range(initial_wptr, initial_samp_wrt, end_samp_wrt, end_wptr, ring_len, ring_len);
}

// Called after all AICores have acknowledged retirement.
extern "C" void simpler_aicpu_l0_snapshot();

// Bind the invocation epoch before its enable bit. Only a live capture
// configured for that epoch can remain enabled.
extern "C" void set_platform_l0_run_epoch(uint64_t run_epoch);
extern "C" void set_l0_swimlane_enabled(bool enable);
extern "C" bool is_l0_swimlane_enabled();

// Called when an AICore reports its physical ID and type during handshake.
void l0_live_capture_note_core_type(uint32_t physical_core_id, bool is_aic);
