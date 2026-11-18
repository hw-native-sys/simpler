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

constexpr uint32_t L0_DFX_FLAG_LIVE = 2u;
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

// DFX transport: the checkpoint publishes safe ring ranges to a per-thread
// ready queue; the host reads the ring directly and advances the per-core
// drained cursor. There is no intermediate device-side buffer.
// HBG publishes all physical sub-cores through one queue. One sweep must fit
// even when the host is still draining the preceding descriptors.
constexpr uint32_t L0_DFX_READY_QUEUE_SIZE = 256;
constexpr uint32_t L0_DFX_TAIL_HEADROOM_BYTES = 64u << 10;

struct alignas(64) L0BufferState {
    volatile uint64_t next_range_sequence;
    volatile uint64_t published_bytes;
    uint32_t physical_core_id;
    // Host-written cursor for software overrun detection. The hardware sampler
    // does not read it and can overwrite unread ring data.
    volatile uint32_t consumed_wptr;
    volatile uint32_t consumed_samp_wrt;
    // AICPU-written: the end of the last published range.
    volatile uint32_t published_wptr;
    volatile uint32_t published_samp_wrt;
    // AICPU-written: the ring cursor sampled on the previous checkpoint. A range
    // [published, sampled] is stable for the host because the sampler cannot reach
    // it again before writing another ring_len bytes.
    volatile uint32_t sampled_wptr;
    volatile uint32_t sampled_samp_wrt;
    uint32_t capture_error;
    uint32_t batch_count;
    uint32_t live_batch_count;
    uint32_t sample_count;
    uint32_t advance_count;
    uint32_t derive_fallback_count;
    uint32_t last_observed_wptr;
    uint32_t last_observed_samp_wrt;
    // Host requests a cursor sampled after its live GM copy. The AICPU writes
    // the cursor before acknowledging; terminal snapshot marks the producer
    // stopped so a late host copy can use its final cursor instead.
    volatile uint32_t verify_request;
    volatile uint32_t verified_samp_wrt;
    volatile uint32_t verify_response;
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
    L0RangeEntry queues[PLATFORM_MAX_AICPU_THREADS][L0_DFX_READY_QUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];
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
static_assert(sizeof(L0BufferState) == 128);
static_assert(std::is_standard_layout_v<L0RangeEntry> && std::is_trivially_copyable_v<L0RangeEntry>);
static_assert(std::is_standard_layout_v<L0DataHeader> && std::is_trivially_copyable_v<L0DataHeader>);
static_assert(sizeof(L0RangeEntry) == 32);

// Ring range calculation shared by AICPU and Host.
namespace simpler::dfx::l0 {

struct RingRange {
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
inline RingRange plan_ring_range(
    uint32_t initial_wptr, uint32_t initial_samp_wrt, uint32_t end_samp_wrt, uint32_t end_wptr, uint32_t ring_len,
    uint32_t ring_capacity
) {
    RingRange result{};
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

inline RingRange plan_ring_range(
    uint32_t initial_wptr, uint32_t initial_samp_wrt, uint32_t end_samp_wrt, uint32_t end_wptr, uint32_t ring_len
) {
    return plan_ring_range(initial_wptr, initial_samp_wrt, end_samp_wrt, end_wptr, ring_len, ring_len);
}

// A record can wrap early or extend into the allocation's tail headroom.
// Before the cursor enters that tail window, written bytes advance WPTR
// linearly from an exact anchor. Once it reaches the window, the hardware
// WPTR must be read to locate the wrap boundary.
inline bool derive_linear_wptr(
    uint32_t anchor_wptr, uint32_t anchor_samp_wrt, uint32_t now_samp_wrt, uint32_t ring_len, uint32_t ring_capacity,
    uint32_t *wptr
) {
    if (wptr == nullptr || ring_len == 0 || ring_capacity <= ring_len || anchor_wptr > ring_capacity) return false;
    const uint32_t produced = now_samp_wrt - anchor_samp_wrt;
    if (produced == 0) {
        *wptr = anchor_wptr;
        return true;
    }
    if (ring_capacity - ring_len < L0_DFX_TAIL_HEADROOM_BYTES || ring_len <= L0_DFX_TAIL_HEADROOM_BYTES ||
        anchor_wptr >= ring_len - L0_DFX_TAIL_HEADROOM_BYTES ||
        produced >= ring_len - L0_DFX_TAIL_HEADROOM_BYTES - anchor_wptr)
        return false;
    *wptr = anchor_wptr + produced;
    return true;
}

}  // namespace simpler::dfx::l0

// Runtime schedulers request checkpoints; the AICPU kernel owns register and
// DFX buffer state. Each producer thread handles a disjoint set of cores (a
// single AICPU producer in host_build_graph; one per scheduler thread in
// tensormap_and_ringbuffer). At run exit, the terminal snapshot reads each
// partition's final cursors and publishes its tail range.
// The caller passes its own producer index in [0, num_threads); the configured
// num_threads must equal the number of active L0 sampling producers.
// A partition's terminal snapshot is claimed once per configuration; a later
// fallback call for the same partition is a no-op.
extern "C" void simpler_aicpu_l0_checkpoint(int thread_index);
extern "C" void simpler_aicpu_l0_snapshot(int thread_index, const int32_t *core_ids, int32_t core_num);

namespace simpler::dfx::l0 {

// True while a run has an armed L0 live capture; the checkpoint itself
// re-checks the same flag.
bool l0_live_capture_armed();

// True while a configured capture for this epoch is still armed. A terminal
// snapshot disarms it; a later run cannot reuse the previous configuration.
// bind_platform_task uses this with the L0 flag to arm the per-run cadence.
bool l0_live_capture_matches(uint64_t run_epoch);

// Arm or disarm the live-capture cadence from the per-task bind path.
void l0_live_capture_set_armed(bool armed);
// Called when an AICore reports its physical ID and type during handshake.
void l0_live_capture_note_core_type(uint32_t physical_core_id, bool is_aic);

// A producer thread owns one physical-core partition (all cores, under
// host_build_graph's single AICPU producer). Call from each producer loop
// turn: the cadence gates on the armed flag and a time interval, so the
// checkpoint runs only when a capture is armed and due. Completions bring a
// check forward, but not for every short task: each check reads the cursors of
// all cores in the partition.
constexpr uint64_t kCheckpointIntervalCycles = PLATFORM_PROF_SYS_CNT_FREQ / 10000;            // 100 us
constexpr uint64_t kCompletionCheckpointIntervalCycles = PLATFORM_PROF_SYS_CNT_FREQ / 20000;  // 50 us

template <typename ReadClock>
inline void tick_checkpoint(int thread_index, uint64_t &last_check, bool completed, ReadClock read_clock) {
    if (!l0_live_capture_armed()) return;
    const uint64_t interval = completed ? kCompletionCheckpointIntervalCycles : kCheckpointIntervalCycles;
    if (read_clock() - last_check < interval) return;
    simpler_aicpu_l0_checkpoint(thread_index);
    last_check = read_clock();
}

}  // namespace simpler::dfx::l0
