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
 * The percounter scheduler's GM wire structs, shared by the host that builds
 * them, the AICPU that publishes the topology into them, and the AICore lanes
 * that read them.
 *
 * Host-only planning (the layout planner and the table builder) lives in
 * host/region_plan.h; this header is compiled by the AICore
 * target too, so it includes nothing host-only.
 *
 * Calendar table offsets, including notify_slots_offset, are relative to
 * the start of their calendar image. PercounterControl::calendar_offset
 * locates the range calendar from the scheduler control block. After binding,
 * commands and addresses identify device memory or SoC SRAM MMIO addresses.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "common/device_phase.h"
#include "common/platform_config.h"

namespace simpler::percounter {

inline constexpr uint32_t CALENDAR_NO_JOIN = UINT32_MAX;
inline constexpr uint32_t CALENDAR_NO_INPUT_OWNER = UINT32_MAX;
inline constexpr uint32_t CALENDAR_COUNT_MASK = 0x00ff'ffff;

// Offsets are relative to the start of this immutable image in GM.
struct CalendarHeader {
    uint32_t worker_count;
    uint32_t row_count;
    uint32_t join_count;
    uint32_t index_offset;
    uint32_t row_offset;
    uint32_t join_offset;
};

struct CalendarRow {
    uint64_t entry;
    uint32_t local_join;
    uint32_t reserved;
};

struct CalendarLocalJoin {
    uint32_t arrive[3];  // One-based ordinal on the corresponding cluster lane, zero if absent.
    uint32_t leader_lane;
};

// The producer writes the first line and the defer resolver writes the second.
struct alignas(64) CalendarTiming {
    uint64_t start;
    uint64_t end;
    uint64_t defer_register;
    alignas(64) uint64_t defer_completion;
};

static_assert(sizeof(CalendarHeader) == 24);
static_assert(sizeof(CalendarRow) == 16);
static_assert(sizeof(CalendarLocalJoin) == 16);
static_assert(sizeof(CalendarTiming) == 128);

enum class CalendarPmuPublication : uint32_t { Empty, Executed, Skipped };

struct alignas(64) CalendarPmuCell {
    uint64_t total_cycles;
    uint32_t counters[PMU_COUNTER_COUNT_A5];
    uint16_t kernel_id;  // Bound by Host, unchanged by the producer.
    CalendarPmuPublication publication;
};

static_assert(sizeof(CalendarPmuCell) == 64);

// Per-core ranges reference shared logical rows and per-invocation bindings.
struct CalendarRangeHeader {
    uint32_t worker_count;
    uint32_t worker_index_offset;
    uint32_t range_offset;
    uint32_t range_count;
    uint32_t patch_offset;
    uint32_t sample_index_offset;
    uint32_t notify_slots_offset;
    uint32_t notify_slots_count;
    uint32_t peer_endpoints_offset{0};  // One binding per logical worker when a peer edge exists.
    uint32_t peer_fanouts_offset{0};    // List of distinct SPMD multicast destination sets.
};

struct alignas(32) CalendarRange {
    uint32_t rows_offset;
    uint32_t row_count;
    uint32_t binding_offset;
    uint32_t entry_wait_offset;
    uint32_t entry_wait_count;
    uint32_t sequence_base;
    uint64_t tail_add_cmd;
    uint32_t exit_wait_offset;
    uint32_t exit_wait_count;
    uint32_t graph_shell_id;
    uint32_t phase_row_offset;   // Rows on this worker before this range, within this phase.
    uint32_t pmu_offset;         // Optional, per invocation, one cell per packed row.
    uint32_t progress_offset{};  // Per-row state for an ordinary URMA range with work behind a wait.
};

// Only the assigned lane reads or writes this state. Its array starts on a private cache line.
struct CalendarRowProgress {
    uint64_t defer_cursor;
    uint32_t next_input;
    uint32_t retired;
};

enum : uint16_t {
    CALENDAR_EXEC_ROLE_MASK = 3,
    CALENDAR_EXEC_SYNC_START = 1 << 2,
    CALENDAR_EXEC_CONTROL = 1 << 4,
    CALENDAR_EXEC_SINK = 1 << 5,
    CALENDAR_EXEC_PREDICATE = 1 << 6,
    CALENDAR_EXEC_FANIN = 1 << 7,
    CALENDAR_EXEC_INPUT_OWNER = 1 << 8,
    CALENDAR_EXEC_INPUT_FOLLOWER = 1 << 9,
    CALENDAR_EXEC_PEER = 1 << 10,
    CALENDAR_EXEC_SPMD_WAIT = 1 << 11,
    CALENDAR_EXEC_SPMD_SEND = 1 << 12,
    CALENDAR_EXEC_PACED_WAIT = 1 << 13,
};

inline constexpr uint32_t CALENDAR_TASK_TIMING_SHIFT = 10;
inline constexpr uint16_t CALENDAR_TASK_TIMING_MASK = 0x1f;
static_assert(NUM_TASK_TIMING_SLOTS <= CALENDAR_TASK_TIMING_MASK);

inline constexpr uint16_t calendar_task_timing_flag(int32_t slot) {
    return static_cast<uint16_t>((slot + 1) << CALENDAR_TASK_TIMING_SHIFT);
}

struct alignas(16) CalendarExecutionRow {
    uint32_t dispatch_offset;
    uint32_t mix_delta;  // Join delta, helper begin, or a follower's input owner ordinal.
    uint16_t block_index;
    uint16_t block_count;
    uint16_t wait_count;
    uint16_t flags;
};

struct alignas(32) CalendarDispatch {
    uint64_t kernel_entry;
    uint64_t args_address;
    uint64_t done_add_cmd;
    uint32_t waits_offset;
    uint32_t extra_offset;
};

struct CalendarExecutionExtra {
    uint32_t task_id;
    uint32_t local_task_id;
    int32_t shell_task_id;
    uint64_t shell_add_cmd;
    uint64_t predicate_address;
};

// Only a selected dispatch has this larger extra. Worker numbers and ordinals
// are static; startup binds the actual register and two launch-local tokens.
struct CalendarPeerExtra {
    CalendarExecutionExtra execution;
    uint32_t sends_offset;
    uint16_t send_count;
    uint16_t input_count;
};

struct CalendarPeerSend {
    uint32_t worker;
    uint32_t ordinal;
};

// SPMD endpoints receive change notifications. The completion counter supplies the proof.
inline constexpr uint32_t CALENDAR_PEER_SPMD_ENDPOINT = UINT32_MAX;

struct CalendarPeerEndpoint {
    uint64_t address;
    uint32_t token_base;
    uint32_t input_count;
};

// Each header is followed by three packed arrays: addresses (u64), token bases
// (u32), and immutable logical workers (u32). Only startup writes the first two.
struct CalendarPeerFanout {
    uint32_t previous_offset;
    uint16_t count;
    uint16_t burst_count;
    uint64_t common_token_base;  // UINT64_MAX when different destinations need different tokens.
};
// With a common token the launch binder can store a short burst list in the
// otherwise unused per-target token-base array.
struct CalendarPeerBurst {
    uint64_t start;
    uint32_t stride_bytes;
    uint32_t targets;
};
static_assert(sizeof(CalendarPeerBurst) == 16);
inline constexpr size_t CALENDAR_FANOUT_TARGET_BYTES = sizeof(uint64_t) + 2 * sizeof(uint32_t);

enum : uint64_t {
    CALENDAR_WAIT_FUSED_FANIN = 0,
    CALENDAR_WAIT_NOTIFY = 1,
    CALENDAR_WAIT_GM_TARGET = 2,
    CALENDAR_WAIT_DEFER = 3,
    CALENDAR_WAIT_TAG_MASK = 3,
};

inline constexpr uint32_t CALENDAR_FANIN_TASKS_PER_LANE = 4;
inline constexpr uint32_t CALENDAR_FUSED_FANIN_MAX_PRODUCERS = 4;

inline constexpr uint32_t CALENDAR_FOLLOWER_DONE_INTERVAL = 64;
inline constexpr uint32_t CALENDAR_FOLLOWER_FANIN_INTERVAL = 2 * CALENDAR_FOLLOWER_DONE_INTERVAL;

struct alignas(16) CalendarWait {
    uint64_t read_cmd;
    uint64_t target;
};

struct CalendarSpmdPeerExtra {
    CalendarExecutionExtra execution;
    CalendarWait completion;
    uint32_t single_worker;  // Used only by a single-target notification.
    uint32_t target_count;
    uint32_t token_begin;
    uint32_t fanout_offset;
};

struct CalendarDispatchPatch {
    uint32_t dispatch_offsets[3];
    uint32_t predicate_extra_offset;
};

static_assert(sizeof(CalendarRangeHeader) == 40 && sizeof(CalendarPeerFanout) == 16);
static_assert(sizeof(CalendarRange) == 64);
static_assert(sizeof(CalendarRowProgress) == 16);
static_assert(sizeof(CalendarExecutionRow) == 16);
static_assert(sizeof(CalendarDispatch) == 32);
static_assert(sizeof(CalendarExecutionExtra) == 32);
static_assert(sizeof(CalendarPeerExtra) == 40 && offsetof(CalendarPeerExtra, execution) == 0);
static_assert(sizeof(CalendarPeerEndpoint) == 16 && sizeof(CalendarPeerSend) == 8);
static_assert(sizeof(CalendarWait) == 16);
static_assert(sizeof(CalendarSpmdPeerExtra) == 64 && offsetof(CalendarSpmdPeerExtra, execution) == 0);

}  // namespace simpler::percounter
