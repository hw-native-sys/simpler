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

#include "common/lane_mask.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/platform_config.h"
#include "kernel_args.h"
#include "scheduler_layout.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

// AICPU supervision and Host run-control initialization share the platform budget.
constexpr int32_t SCHEDULER_TIMEOUT_MS = PLATFORM_SCHEDULER_TIMEOUT_MS;
constexpr uint64_t SCHEDULER_TIMEOUT_CYCLES =
    static_cast<uint64_t>(SCHEDULER_TIMEOUT_MS) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);

// =============================================================================
// Resident AICore scheduler wire contracts
//
// These types are shared by Host, AICPU lifecycle, and AICore workers.
// =============================================================================
// Wire subregions use 64B alignment; the allocation base must also satisfy
// the shared GM types' alignment, computed below.
inline constexpr uint64_t SCHEDULER_STATE_ALIGNMENT = 64;
inline constexpr uint64_t SCHEDULER_SCALAR_DCACHE_LINE_BYTES = 64;
// CANN's SIMT GM-atomic model uses 128B sectors within a 512B L2 line.
// AIC/AIV scalar atomicAdd/ld_dev probes support this separation for layout;
// it is not a scalar ordering guarantee. See the data-layout contract.
inline constexpr uint64_t SCHEDULER_GM_ATOMIC_SECTOR_BYTES = 128;
inline constexpr uint64_t SCHEDULER_WORKER_CAPACITY = 108;
inline constexpr uint32_t SCHEDULER_PENDING_SLOT_COUNT = 2;
enum class SchedulerExecutionEngine : uint32_t { AIC = 0, AIV = 1 };
inline constexpr uint32_t SCHEDULER_EXECUTION_ENGINE_COUNT = 2;
inline constexpr uint32_t SCHEDULER_CLUSTER_CAPACITY = SCHEDULER_WORKER_CAPACITY / 3;
inline constexpr uint32_t SCHEDULER_ACTIVITY_CAPACITY = 1024;
inline constexpr uint64_t SCHEDULER_PROFILING_TASK_TIMING_LEVEL = 1;
inline constexpr uint64_t SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL = 2;
inline constexpr uint64_t SCHEDULER_PROFILING_SCHED_PHASES_LEVEL = 3;

inline __aicore__ bool scheduler_task_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_TASK_TIMING_LEVEL;
}

inline __aicore__ bool scheduler_schedule_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL;
}

inline __aicore__ bool scheduler_phase_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL;
}

inline constexpr bool scheduler_activity_record_count_valid(uint32_t committed) noexcept {
    return committed <= SCHEDULER_ACTIVITY_CAPACITY;
}
// A task ref is the index of an expanded top-level, child or control node.
// Public TaskId identity is recovered through the graph view.
inline constexpr int64_t SCHEDULER_TASK_REF_INVALID = -1;

enum class SchedulerTaskState : int64_t {
    BLOCKED = 0,
    DONE = 2,
};

enum class SchedulerReadySource : uint8_t {
    CATALOG = 0,
    COOPERATIVE = 3,
};

struct SchedulerIdleRecord {
    uint64_t start_time;
    uint64_t end_time;
    uint32_t loop_iter;
};
static_assert(sizeof(SchedulerIdleRecord) == 24, "Scheduler idle record layout changed");

inline constexpr uint32_t SCHEDULER_SCALAR_RECORD_CAPACITY = 262144;

struct SchedulerScalarRecord {
    uint64_t start_time;
    uint64_t end_time;
    int64_t task_ref;
    uint32_t kind;
    uint32_t count;
};
static_assert(sizeof(SchedulerScalarRecord) == 32);

// Compact rows keep an absolute 64-bit start and a 32-bit duration. Rows with
// a task, a non-unit count or a long duration use a second slot without loss.
inline constexpr uint32_t SCHEDULER_SCALAR_STORAGE_EXTENDED = UINT32_C(0x8000);
struct alignas(16) SchedulerScalarStorageSlot {
    uint64_t time;
    uint32_t kind;
    uint32_t value;
};
static_assert(sizeof(SchedulerScalarStorageSlot) == 16);

inline __aicore__ bool scheduler_decode_scalar_storage(
    const SchedulerScalarStorageSlot *slots, uint32_t size, uint32_t &cursor, SchedulerScalarRecord &record
) {
    if (cursor >= size) return false;
    const auto &head = slots[cursor++];
    record.start_time = head.time;
    record.kind = head.kind & ~SCHEDULER_SCALAR_STORAGE_EXTENDED;
    if ((head.kind & SCHEDULER_SCALAR_STORAGE_EXTENDED) != 0) {
        if (cursor >= size) return false;
        const auto &tail = slots[cursor++];
        record.end_time = tail.time;
        record.task_ref = static_cast<int64_t>(uint64_t{tail.kind} | (uint64_t{tail.value} << 32));
        record.count = head.value;
    } else {
        if (head.time > UINT64_MAX - head.value) return false;
        record.end_time = head.time + head.value;
        record.task_ref = -1;
        record.count = 1;
    }
    return record.end_time >= record.start_time;
}

struct alignas(64) SchedulerActivityBuffer {
    volatile uint32_t committed;
    volatile uint32_t dropped;
    SchedulerIdleRecord records[SCHEDULER_ACTIVITY_CAPACITY];
    alignas(64) uint32_t scalar_committed;
    uint32_t scalar_dropped;
    uint64_t scalar_write_cycles;
    uint64_t scalar_poll_loops;
    uint64_t scalar_records_offset;
    uint64_t scalar_short_intervals;
    uint64_t scalar_function_min_cycles;
    uint32_t scalar_storage_format;
    uint32_t scalar_storage_slots;
};
static_assert(sizeof(SchedulerActivityBuffer) % 64 == 0, "scheduler activity buffer must be cache aligned");
static_assert(
    offsetof(SchedulerActivityBuffer, scalar_storage_slots) + sizeof(uint32_t) <=
        offsetof(SchedulerActivityBuffer, scalar_committed) + 64,
    "Scalar metadata must fit its existing cache line"
);

enum class SchedulerDispatchSlotState : uint8_t {
    EMPTY = 0,
    FREE = 1,
    FILLING = 2,
    READY = 3,
    EXECUTING = 4,
};

enum SchedulerTaskMetadataFlags : uint8_t {
    SCHEDULER_TASK_EXECUTABLE = 1U << 0,
    SCHEDULER_TASK_HAS_FANIN = 1U << 1,
    SCHEDULER_TASK_MIX = 1U << 2,
    SCHEDULER_TASK_SPMD = 1U << 3,
    SCHEDULER_TASK_SYNC_START = 1U << 4,
    SCHEDULER_TASK_INLINE = 1U << 5,
    SCHEDULER_TASK_HAS_PREDICATE = 1U << 6,
};

#if !defined(__CCE_AICORE__)
#include "host_build_graph/submit_types.h"
// Derive the compact AICore metadata from the established HBG scheduling
// types. This is a host-only projection: LaneMask and TaskAttrs remain the
// source of truth, while SchedulerTaskMetadata stays a 16-byte GM read model.
inline uint8_t scheduler_task_metadata_flags_from_submit_state(
    LaneMask active_mask, TaskAttrs task_attrs, int32_t logical_block_num, bool has_fanin, bool inline_task
) {
    uint8_t flags = SCHEDULER_TASK_EXECUTABLE;
    const uint32_t active_subtasks = static_cast<uint32_t>(active_mask.count());
    if (has_fanin) flags |= SCHEDULER_TASK_HAS_FANIN;
    if (active_subtasks > 1) flags |= SCHEDULER_TASK_MIX;
    if (logical_block_num > 1) flags |= SCHEDULER_TASK_SPMD;
    if (task_attrs.requires_sync_start()) flags |= SCHEDULER_TASK_SYNC_START;
    if (inline_task) flags |= SCHEDULER_TASK_INLINE;
    if (task_attrs.has_predicate()) flags |= SCHEDULER_TASK_HAS_PREDICATE;
    return flags;
}
#endif

constexpr int32_t SCHEDULER_TASK_TIMING_SLOT_COUNT = 16;

// GM records have a 16-byte stride. Stack copies use scalar alignment: their
// callers pass local objects across outlined boundaries without realigning SP.
struct alignas(8) SchedulerTaskMetadata {
    uint16_t kernel_ids[3];
    LaneMask active_mask;
    uint8_t flags;
    uint32_t trace_index_base;
    int16_t timing_slot;
    uint16_t logical_block_num;
};

inline __aicore__ bool
scheduler_subtask_count_checked(uint64_t logical_block_num, uint32_t active_subtasks, uint32_t *out) {
    if (out == nullptr || logical_block_num == 0 || active_subtasks == 0 || active_subtasks > 3 ||
        logical_block_num > static_cast<uint64_t>(INT16_MAX) / active_subtasks)
        return false;
    *out = static_cast<uint32_t>(logical_block_num * active_subtasks);
    return true;
}

inline __aicore__ bool scheduler_trace_count_checked(uint64_t prefix, uint64_t added, uint64_t *out) {
    constexpr uint64_t capacity = static_cast<uint64_t>(UINT32_MAX) + 1;
    if (out == nullptr || added == 0 || prefix > UINT32_MAX || added > capacity - prefix) return false;
    *out = prefix + added;
    return true;
}

inline __aicore__ LaneMask scheduler_task_lanes(const SchedulerTaskMetadata &metadata) { return metadata.active_mask; }

// A lane set held in device memory. An AICore reaches those structs through
// `__gm__` references, which LaneMask's own members do not accept, so the byte
// moves through a local. The GM access is the one the caller needed anyway;
// `scheduler_task_lanes` pays the same copy on the dispatch path.
inline __aicore__ LaneMask scheduler_lane_mask_load(const LaneMask &lanes) { return lanes; }
inline __aicore__ void scheduler_lane_mask_store(LaneMask &field, LaneMask lanes) { field = lanes; }

#if defined(__CCE_AICORE__)
inline __aicore__ LaneMask scheduler_lane_mask_load(const __gm__ LaneMask &lanes) {
    LaneMask local;
    __builtin_memcpy(&local, &lanes, sizeof(local));
    return local;
}

inline __aicore__ void scheduler_lane_mask_store(__gm__ LaneMask &field, LaneMask lanes) {
    __builtin_memcpy(&field, &lanes, sizeof(lanes));
}

inline __aicore__ LaneMask scheduler_task_lanes(const __gm__ SchedulerTaskMetadata &metadata) {
    LaneMask lanes;
    // The one-byte POD copy transfers the field from GM into a local value.
    __builtin_memcpy(&lanes, &metadata.active_mask, sizeof(lanes));
    return lanes;
}
#endif

// Trace and PMU cells are ordered by task, then by block, then by active
// subtask, so a task owns `logical_block_num * active_mask.count()` of them
// starting at its `trace_index_base`.
inline constexpr uint8_t SCHEDULER_TASK_SUBTASK_SLOTS = 3;

inline __aicore__ uint32_t scheduler_task_trace_index(
    uint32_t trace_index_base, LaneMask active_mask, SubtaskSlot subtask, uint32_t block_idx = 0
) {
    const uint64_t offset =
        static_cast<uint64_t>(block_idx) * active_mask.count() + static_cast<uint64_t>(active_mask.rank_of(subtask));
    return static_cast<uint32_t>(static_cast<uint64_t>(trace_index_base) + offset);
}
inline __aicore__ uint32_t scheduler_task_execution_engine(uint8_t flags, LaneMask active_mask) {
    return static_cast<uint32_t>(
        (flags & SCHEDULER_TASK_MIX) != 0 || active_mask.has_aic() ? SchedulerExecutionEngine::AIC :
                                                                     SchedulerExecutionEngine::AIV
    );
}

inline __aicore__ bool scheduler_task_is_executable(uint8_t flags) { return (flags & SCHEDULER_TASK_EXECUTABLE) != 0; }

inline __aicore__ bool scheduler_task_has_fanin(uint8_t flags) { return (flags & SCHEDULER_TASK_HAS_FANIN) != 0; }

inline __aicore__ bool scheduler_task_is_mix(uint8_t flags) { return (flags & SCHEDULER_TASK_MIX) != 0; }

// The coordinator is a resource participant even for a Vector-only Mix.
inline __aicore__ LaneMask scheduler_mix_resource_mask(LaneMask active_mask) {
    return active_mask.merged_with(LaneMask::aic());
}

inline __aicore__ bool scheduler_task_is_spmd(uint8_t flags) { return (flags & SCHEDULER_TASK_SPMD) != 0; }

inline __aicore__ bool scheduler_task_requires_sync_start(uint8_t flags) {
    return (flags & SCHEDULER_TASK_SYNC_START) != 0;
}

// A lone AIV subtask can run on either of a cluster's two V lanes, so it has
// two claimable positions there; every other shape is pinned to one.
inline __aicore__ uint64_t scheduler_block_claim_positions(LaneMask active_mask, uint64_t clusters) {
    return clusters * (active_mask.to_shape() == ResourceShape::AIV ? 2 : 1);
}

inline __aicore__ bool
scheduler_task_uses_ticket_claim(uint8_t flags, LaneMask active_mask, uint64_t blocks, uint64_t clusters) {
    const uint64_t positions = scheduler_block_claim_positions(active_mask, clusters);
    return positions != 0 && blocks > 1 && scheduler_task_requires_sync_start(flags) && blocks < positions;
}

inline __aicore__ bool scheduler_task_is_inline(uint8_t flags) { return (flags & SCHEDULER_TASK_INLINE) != 0; }

inline __aicore__ bool scheduler_task_has_predicate(uint8_t flags) {
    return (flags & SCHEDULER_TASK_HAS_PREDICATE) != 0;
}

// Classification uses immutable flags; payload preparation still validates shape.
inline __aicore__ bool scheduler_task_requires_admission(uint8_t flags) {
    return (flags & (SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE)) == SCHEDULER_TASK_EXECUTABLE &&
           (flags & SCHEDULER_TASK_SYNC_START) != 0;
}

inline __aicore__ bool scheduler_task_requires_admission(const SchedulerTaskMetadata &metadata) {
    return scheduler_task_requires_admission(metadata.flags);
}

inline __aicore__ bool scheduler_task_uses_distributed_blocks(uint8_t flags, uint32_t blocks) {
    return scheduler_task_is_executable(flags) && !scheduler_task_is_inline(flags) && !scheduler_task_is_mix(flags) &&
           blocks > 1 && !scheduler_task_requires_admission(flags);
}

inline __aicore__ bool scheduler_task_uses_distributed_blocks(const SchedulerTaskMetadata &metadata) {
    return scheduler_task_uses_distributed_blocks(metadata.flags, metadata.logical_block_num);
}

enum class SchedulerAdmissionState : uint64_t {
    UNCLAIMED = 0,
    ADMITTED = 3,
    INITIALIZED = 4,
};

// The immutable total distinguishes a live completed gate from uninitialized
// storage. Remaining computation and async debt are the only readiness source.
inline constexpr uint64_t SCHEDULER_COMPLETION_RETURN_UNIT = (UINT64_C(1) << 32) + 1;
inline constexpr uint64_t SCHEDULER_COMPLETION_LIVE_MASK = (UINT64_C(1) << 48) - 1;
inline constexpr uint32_t SCHEDULER_COMPLETION_TOTAL_MAX = INT16_MAX;

inline __aicore__ constexpr uint64_t scheduler_completion_encode(uint32_t total, uint32_t remaining, uint32_t debt) {
    return (static_cast<uint64_t>(total) << 48) | (static_cast<uint64_t>(remaining) << 32) | debt;
}
inline __aicore__ constexpr uint64_t scheduler_completion_total(uint64_t state) { return state >> 48; }
inline __aicore__ constexpr uint64_t scheduler_completion_remaining(uint64_t state) {
    return (state >> 32) & UINT16_MAX;
}
inline __aicore__ constexpr uint64_t scheduler_completion_debt(uint64_t state) { return state & UINT32_MAX; }
inline __aicore__ constexpr bool scheduler_completion_valid(uint64_t state) {
    const uint64_t total = scheduler_completion_total(state);
    const uint64_t remaining = scheduler_completion_remaining(state);
    const uint64_t debt = scheduler_completion_debt(state);
    return total != 0 && total <= SCHEDULER_COMPLETION_TOTAL_MAX && remaining <= total && debt >= remaining &&
           debt <= remaining + total;
}
inline __aicore__ constexpr bool scheduler_completion_done(uint64_t state) {
    return scheduler_completion_valid(state) && (state & SCHEDULER_COMPLETION_LIVE_MASK) == 0;
}
inline __aicore__ constexpr uint64_t scheduler_completion_count(uint64_t state) {
    return scheduler_completion_total(state) - scheduler_completion_remaining(state);
}
inline __aicore__ constexpr uint64_t scheduler_completion_initial_state(uint32_t total) {
    return scheduler_completion_encode(total, total, total);
}
inline __aicore__ constexpr uint64_t scheduler_completion_completed_state(uint32_t total) {
    return scheduler_completion_encode(total, 0, 0);
}
inline __aicore__ constexpr int64_t scheduler_completion_task_state(uint64_t state) {
    return !scheduler_completion_valid(state) ? -1 :
           scheduler_completion_done(state)   ? static_cast<int64_t>(SchedulerTaskState::DONE) :
                                                static_cast<int64_t>(SchedulerTaskState::BLOCKED);
}

struct alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) SchedulerTaskControl {
    volatile uint64_t claim_tickets;

    // Block claimers and retiring workers update these counters concurrently.
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t next_block;
    // Bits [0,32) hold total debt D, [32,48) remaining contributions C and
    // [48,64) the immutable validated total N. Only C=D=0 is logically done.
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t completion_state;

    // Only the admission owner bypass-publishes this signal. No cached store
    // or atomic RMW may target its line.
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t admission_state;

    // The final block claimer publishes this immutable claim certificate.
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t claim_seal;
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t parked_count;
};

// Each task owns one cache-line record; membership follows its frozen resource role.
struct alignas(64) SchedulerAdmissionRecord {
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t mix_close_seals_offset;
    uint32_t logical_block_num;
    uint32_t member_count;
    LaneMask active_mask;
    uint8_t flags;
};

// Each cluster's retained close seal occupies an independent atomic region.
struct alignas(64) SchedulerMixCloseSeal {
    volatile uint64_t value;
};

struct alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) SchedulerAdmissionControl {
    // Only the lock uses atomics. Tail publishers are serialized by the lock
    // and complete their bypass store before releasing it.
    volatile uint64_t commit_lock;
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t commit_tail;
};

struct alignas(64) SchedulerGroupDescriptor {
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t group_seq;
    uint32_t block_idx;
    uint32_t cluster_index;
    uint64_t previous_lane_seq[3];
};

struct alignas(64) SchedulerGroupLaneAck {
    volatile uint64_t arrived_seq;
    volatile uint64_t released_seq;
    volatile uint64_t assigned_seq;
};

struct alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) SchedulerGroupMailbox {
    SchedulerGroupDescriptor descriptor;
    alignas(SCHEDULER_GM_ATOMIC_SECTOR_BYTES) volatile uint64_t published_seq;
    SchedulerGroupLaneAck lanes[3];
};

enum class SchedulerTopologyLayout : uint64_t { GENERAL = 0, COMPACT_LAUNCH = 1 };

// Facts frozen by READY: the host supplies graph facts, and AICPU discovery
// validates the topology. Workers only read this region during the run.
struct alignas(64) SchedulerGraphSummary {
    uint64_t mix_task_count;
    uint64_t sync_mix_task_count;
    uint64_t cooperative_task_count;
    uint64_t worker_count;
    uint64_t cluster_count;
    uint64_t admission_records_offset;
    uint64_t admission_record_stride;
    uint64_t admission_log_offset;
    uint64_t admission_control_offset;
    uint64_t group_mailboxes_offset;
    uint64_t mix_close_seals_offset;
    uint64_t run_epoch;
    uint64_t protocol_layout_offset;
    SchedulerTopologyLayout topology_layout;
    uint64_t task_catalog_offset;
    uint64_t async_layout_offset;
    uint64_t prepared_layout_offset;
};

// The executing core stages one own-slot trace and consolidates it into the
// task's final trace during retirement, before the slot can execute again.
struct SchedulerExecutorTaskTrace {
    uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t ready_observe_cycles;
};

// Stable device-side localization for the first scheduler failure. These values
// are diagnostic ABI: keep existing numbers stable when adding new sites.
enum class SchedulerErrorSite : uint64_t {
    UNKNOWN = 0,
    ROUTE_INVALID_STATE = 2,
    READY_POP_INVALID_HEAD = 40,
    READY_POP_INVALID_LINK = 41,
    DISPATCH_INVALID_SHAPE = 44,
    DISPATCH_INVALID_CALLABLE = 45,
    DISPATCH_MATERIALIZE_FAILED = 46,
    DISPATCH_INVALID_PREDICATE = 47,
    EXECUTOR_INVALID_DISPATCH_SLOT = 50,
    BOOTSTRAP_WAKE_INVALID_HEAD = 60,
    COMPLETION_TASK_NOT_DONE = 61,
    COMPLETION_WAKE_ALREADY_CLOSED = 62,
    COMPLETION_INVALID_WAITER = 63,
    COMPLETION_WAITER_NOT_EXECUTABLE = 64,
    COMPLETION_READY_APPEND_FAILED = 65,
    COMPLETION_READY_PUBLISH_FAILED = 66,
    COMPLETION_INVALID_SHAPE = 67,
    COMPLETION_RESOLVE_FAILED = 68,
    COMPLETION_REFILL_DISPATCH_FAILED = 70,
    COMPLETION_SLOT_BINDING_INVALID = 74,
    EXECUTOR_PREFERRED_SLOT_INVALID = 79,
    READY_INBOX_REFRESH_FAILED = 80,
    BOOTSTRAP_FAILED = 90,
    CONTEXT_READY_TIMEOUT = 91,
    BOOTSTRAP_SCAN_TIMEOUT = 92,
    BOOTSTRAP_COMPLETE_TIMEOUT = 93,
    REGISTER_RELEASE_TIMEOUT = 94,
    EXECUTION_PROGRESS_TIMEOUT = 95,
    EXIT_WAIT_TIMEOUT = 96,
    AICPU_HANDSHAKE_TIMEOUT = 100,
    AICPU_INIT_BARRIER_TIMEOUT = 101,
    AICPU_SHUTDOWN_BARRIER_TIMEOUT = 102,
    ADMISSION_INVALID_CONFIG = 110,
    ADMISSION_INVALID_STATE = 111,
    ADMISSION_INVALID_TOPOLOGY = 112,
    ADMISSION_INVALID_RECORD = 113,
    ADMISSION_LOG_CAPACITY = 114,
    ADMISSION_EPOCH_MISMATCH = 115,
    ADMISSION_INVALID_FRONTIER = 116,
    BLOCK_CURSOR_INVALID = 117,
    BLOCK_COMPLETION_INVALID = 118,
    ORDERED_CANDIDATE_INVALID = 119,
    GROUP_INVALID_STATE = 120,
    GROUP_INVALID_DESCRIPTOR = 121,
    GROUP_SEQUENCE_OVERFLOW = 122,
    GROUP_ACK_INVALID = 123,
    GROUP_CLOSE_INVALID = 124,
    ORDERED_EXIT_IN_FLIGHT = 125,
    ORDERED_EXIT_TIMEOUT = 126,
    GRAPH_BIND_NODE = 130,
    GRAPH_BIND_DEFINITION = 131,
    GRAPH_BIND_INSTANCE = 132,
    GRAPH_BIND_TENSOR = 133,
    GRAPH_BIND_SCALAR = 134,
    GRAPH_BIND_PREDICATE = 135,
    ASYNC_CONTEXT_INVALID = 140,
    ASYNC_SEAL_FAILED = 141,
    ASYNC_CHECK_FAILED = 142,
    ASYNC_FINISH_FAILED = 143,
    ASYNC_BLOCK_RELEASE_FAILED = 144,
    ARGS_DUMP_BOUNDARY_TIMEOUT = 150,
    ARGS_DUMP_REFUSED = 151,
    ARGS_DUMP_ABORT_TIMEOUT = 152,
};

enum SchedulerErrorIdentityField : uint32_t {
    SCHEDULER_ERROR_HAS_EPOCH = 1U << 0,
    SCHEDULER_ERROR_HAS_ORDER = 1U << 1,
    SCHEDULER_ERROR_HAS_BLOCK = 1U << 2,
    SCHEDULER_ERROR_HAS_GROUP = 1U << 3,
    SCHEDULER_ERROR_HAS_WORKER = 1U << 4,
    SCHEDULER_ERROR_HAS_LANE = 1U << 5,
};

struct alignas(64) SchedulerErrorSignal {
    volatile uint64_t status;
};
static_assert(sizeof(SchedulerErrorSignal) == 64);

struct alignas(64) SchedulerRunControl {
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t active_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t aic_worker_demand;
    uint64_t task_metadata_offset;
    uint64_t aiv_worker_demand;
    uint64_t graph_summary_offset;
    volatile uint64_t scheduler_timeout_cycles;
    volatile uint64_t chip_swimlane_level;
    volatile uint64_t sampled_task_timing_enabled;

    alignas(64) volatile uint64_t resolved_task_count;
    volatile uint64_t bootstrap_arrived_count;
    volatile uint64_t bootstrap_complete;
    volatile uint64_t bootstrap_scan_arrived_count;
    volatile uint64_t bootstrap_scan_complete;

    alignas(64) volatile uint64_t error_claimed;
    volatile uint64_t error_task_ref;
    volatile uint64_t error_core_id;
    volatile uint64_t error_core_type;
    volatile uint64_t error_graph_task_count;
    volatile uint64_t error_storage_address;
    volatile uint64_t error_task_window_last_index;
    volatile uint64_t error_site;
    volatile uint64_t error_run_epoch;
    volatile uint64_t error_order_key;
    volatile uint64_t error_block_idx;
    volatile uint64_t error_group_seq;
    volatile uint64_t error_worker_index;
    volatile uint32_t error_cluster_lane;
    volatile uint32_t error_identity_fields;
    // Only the elected error writer publishes this word. Its line has no
    // atomic RMW or cached AICore store, including readers that add zero.
    alignas(64) volatile uint64_t scheduler_error;
    // The first error writer publishes each reader's line before the canonical
    // status above. A worker polls only its own line during steady execution.
    alignas(64) SchedulerErrorSignal worker_errors[SCHEDULER_WORKER_CAPACITY];
};

struct alignas(64) AicpuThreadLifecycleTrace {
    uint64_t aicpu_thread_id;
    uint64_t handshake_start_cycles;
    uint64_t handshake_complete_cycles;
    uint64_t config_start_cycles;
    uint64_t topology_complete_cycles;
    uint64_t context_publish_start_cycles;
    uint64_t context_publish_complete_cycles;
    uint64_t bootstrap_wait_start_cycles;
    uint64_t bootstrap_complete_cycles;
    uint64_t register_release_start_cycles;
    uint64_t register_release_end_cycles;
    uint64_t exit_signal_start_cycles;
    uint64_t exit_signal_end_cycles;
    uint64_t exit_wait_start_cycles;
    uint64_t exit_wait_end_cycles;
};

struct alignas(64) SchedulerWorkerContext {
    // READY publishes the configuration prefix used by peer routing.
    int32_t core_type;
    int32_t physical_core_id;
    int32_t type_rank;
    uint64_t run_control_offset;
    uint64_t task_controls_offset;
    uint64_t scheduler_state_base_address;
    uint64_t prepared_layout_offset;
    uint64_t worker_index;
    uint64_t worker_contexts_offset;

    // The complete graph view shares one line, including its format and extent.
    uint64_t graph_storage_address;
    uint64_t graph_task_count;
    uint64_t task_window_last_index;
    volatile uint64_t native_header_offset;
    volatile uint64_t scheduler_state_bytes;
    volatile uint64_t original_task_count;
    volatile uint32_t graph_view_version;
    volatile uint32_t graph_view_bytes;
    uint64_t graph_summary_offset;

    uint64_t task_metadata_offset;
    // The callable's registration-owned entry table is outside this run's state.
    uint64_t callable_addresses_address;
    uint64_t callable_addresses_count;
    uint64_t runtime_worker_count;

    uint64_t cluster_count;
    uint64_t cluster_index;
    uint64_t cluster_worker_ids[3];
    uint64_t aicpu_lifecycle_traces_offset;

    uint64_t trace_cells_offset;
    uint64_t activity_buffers_offset;
    uint64_t pmu_cells_offset;
    uint64_t args_dump_address;

    // Only this worker writes its bootstrap diagnostic line.
    alignas(64) uint64_t bootstrap_task_count;
    uint64_t bootstrap_start_cycles;
    uint64_t target_bootstrap_end_cycles;
};

enum SchedulerExecutionIdentityFlags : uint32_t {
    SCHEDULER_TRACE_IDENTITY_CAPTURED = 1U << 0,
    SCHEDULER_TRACE_EXECUTED = 1U << 1,
    SCHEDULER_TRACE_ORDERED = 1U << 2,
    SCHEDULER_TRACE_GROUP = 1U << 3,
    SCHEDULER_TRACE_CLUSTER_GROUP = 1U << 4,
};

struct alignas(64) SchedulerTaskTrace {
    // Dispatch publishes this line before READY; completion publishes valid.
    volatile uint64_t valid;
    uint64_t worker_id;
    uint64_t task_ref;
    uint64_t state_probe_scheduler_worker_id;
    uint64_t state_probe_start_cycles;
    uint64_t state_probe_end_cycles;

    // This core owns execution timing through slot retirement.
    alignas(64) uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t ready_observe_cycles;

    alignas(64) uint64_t dispatch_start_cycles;
    uint64_t dispatch_end_cycles;
    uint64_t dispatch_scheduler_worker_id;
    uint64_t dispatch_loop_iter;
    uint64_t complete_start_cycles;
    uint64_t complete_end_cycles;
    uint64_t complete_scheduler_worker_id;
    uint64_t complete_loop_iter;

    alignas(64) uint64_t execution_run_epoch;
    uint64_t execution_order_key;
    uint64_t execution_group_seq;
    uint32_t execution_block_idx;
    uint32_t execution_block_num;
    uint32_t execution_subtask_slot;
    uint32_t execution_pending_slot;
    uint32_t execution_identity_flags;
    uint32_t execution_cluster_index;
    uint32_t execution_cluster_lane;
};

/**
 * One executed subtask's PMU snapshot, staged by the AICore that ran it.
 *
 * One cell per (task, subtask slot), written once by the core that executed it
 * and read once by the AICPU after every core has
 * retired. `published` carries the task id plus one and is written last, so a
 * zero cell is a subtask this run never executed rather than one whose counters
 * read zero.
 */
struct alignas(64) SchedulerTaskPmuCell {
    uint64_t total_cycles;
    uint32_t counters[PMU_COUNTER_COUNT_A5];
    uint32_t worker_id;
    volatile uint32_t published;
};

static_assert(sizeof(SchedulerTaskPmuCell) == 64, "one PMU cell per cache line");

static_assert(sizeof(SchedulerTaskMetadata) == 16, "task metadata layout changed");
static_assert(offsetof(SchedulerTaskMetadata, active_mask) == 6);
static_assert(std::is_trivially_copyable_v<SchedulerTaskMetadata> && std::is_standard_layout_v<SchedulerTaskMetadata>);
static_assert(alignof(SchedulerTaskMetadata) == 8, "metadata copies require scalar-aligned call frames");
static_assert(offsetof(SchedulerTaskMetadata, timing_slot) == 12, "timing tag offset changed");
static_assert(offsetof(SchedulerTaskMetadata, logical_block_num) == 14, "block count must occupy the metadata tail");
// Allocation alignment follows the shared GM types, not DCache maintenance.
inline constexpr uint64_t SCHEDULER_STATE_BASE_ALIGNMENT = alignof(SchedulerTaskControl);
static_assert(alignof(SchedulerAdmissionControl) <= SCHEDULER_STATE_BASE_ALIGNMENT);

static_assert(sizeof(SchedulerTaskControl) == 768, "task control layout changed");
static_assert(alignof(SchedulerTaskControl) == SCHEDULER_GM_ATOMIC_SECTOR_BYTES, "task control alignment changed");
static_assert(offsetof(SchedulerTaskControl, claim_tickets) == 0, "ticket cursor must use the atomic region");
static_assert(offsetof(SchedulerTaskControl, next_block) == 128, "block cursor requires an isolated atomic line");
static_assert(
    offsetof(SchedulerTaskControl, completion_state) == 256, "completion counter requires an isolated atomic line"
);
static_assert(
    offsetof(SchedulerTaskControl, admission_state) == 384, "admission state requires an isolated signal region"
);
static_assert(offsetof(SchedulerTaskControl, claim_seal) == 512, "claim seal requires an isolated atomic line");
static_assert(offsetof(SchedulerTaskControl, parked_count) == 640, "parked count requires an isolated atomic line");
static_assert(sizeof(SchedulerAdmissionRecord) == 64, "admission header must occupy one cache line");
static_assert(sizeof(SchedulerMixCloseSeal) == 64, "close seals require independent atomic regions");
static_assert(alignof(SchedulerMixCloseSeal) == 64, "close seals require isolated alignment");
static_assert(offsetof(SchedulerMixCloseSeal, value) == 0, "close seal value must lead its region");
static_assert(std::is_standard_layout_v<SchedulerMixCloseSeal> && std::is_trivially_copyable_v<SchedulerMixCloseSeal>);
static_assert(sizeof(SchedulerAdmissionControl) == 256, "admission control layout changed");
static_assert(
    alignof(SchedulerAdmissionControl) == SCHEDULER_GM_ATOMIC_SECTOR_BYTES,
    "admission control requires isolated atomic regions"
);
static_assert(offsetof(SchedulerAdmissionControl, commit_tail) == 128, "tail must not share the lock's atomic region");
static_assert(sizeof(SchedulerGroupDescriptor) == 64, "group descriptor layout changed");
static_assert(offsetof(SchedulerGroupDescriptor, run_epoch) == 0);
static_assert(offsetof(SchedulerGroupDescriptor, task_ref) == 8);
static_assert(offsetof(SchedulerGroupDescriptor, order_key) == 16);
static_assert(offsetof(SchedulerGroupDescriptor, group_seq) == 24);
static_assert(offsetof(SchedulerGroupDescriptor, block_idx) == 32);
static_assert(offsetof(SchedulerGroupDescriptor, cluster_index) == 36);
static_assert(offsetof(SchedulerGroupDescriptor, previous_lane_seq) == 40);
static_assert(sizeof(SchedulerGroupLaneAck) == 64, "lane acknowledgements require separate atomic regions");
static_assert(sizeof(SchedulerGroupMailbox) == 384, "group mailbox layout changed");
static_assert(offsetof(SchedulerGroupMailbox, published_seq) == 128, "descriptor and publication must be separate");
static_assert(offsetof(SchedulerGroupMailbox, lanes) == 192, "each lane must own its acknowledgement region");
static_assert(sizeof(SchedulerGraphSummary) == 192, "graph summary layout changed");
static_assert(offsetof(SchedulerGraphSummary, async_layout_offset) == 120);
static_assert(offsetof(SchedulerGraphSummary, prepared_layout_offset) == 128);
static_assert(alignof(SchedulerGraphSummary) == 64, "graph summary alignment changed");
static_assert(sizeof(SchedulerExecutorTaskTrace) == 24, "executor trace payload layout changed");
static_assert(alignof(SchedulerExecutorTaskTrace) == alignof(uint64_t), "executor trace must have natural alignment");
static_assert(offsetof(SchedulerExecutorTaskTrace, ready_observe_cycles) == 16, "executor trace layout changed");
static_assert(sizeof(SchedulerRunControl) == 384 + SCHEDULER_WORKER_CAPACITY * 64, "run control layout changed");
static_assert(alignof(SchedulerRunControl) == 64, "run control alignment changed");
static_assert(offsetof(SchedulerRunControl, resolved_task_count) == 128, "lifecycle atomics need their own line");
static_assert(offsetof(SchedulerRunControl, error_claimed) == 192, "error state needs its own line");
static_assert(offsetof(SchedulerRunControl, error_site) == 248, "error site ABI changed");
static_assert(offsetof(SchedulerRunControl, error_run_epoch) == 256, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_order_key) == 264, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_block_idx) == 272, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_group_seq) == 280, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_worker_index) == 288, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_cluster_lane) == 296, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, error_identity_fields) == 300, "error identity ABI changed");
static_assert(offsetof(SchedulerRunControl, scheduler_error) == 320, "error publication needs its own line");
static_assert(offsetof(SchedulerRunControl, worker_errors) == 384, "worker error signals need disjoint lines");
static_assert(sizeof(AicpuThreadLifecycleTrace) == 128, "AICPU lifecycle trace layout changed");
static_assert(sizeof(SchedulerWorkerContext) == 320, "worker context layout changed");
static_assert(alignof(SchedulerWorkerContext) == 64, "worker context alignment changed");
static_assert(offsetof(SchedulerWorkerContext, worker_contexts_offset) == 56, "peer routing must share the first line");
static_assert(offsetof(SchedulerWorkerContext, graph_storage_address) == 64, "graph view must share one line");
static_assert(offsetof(SchedulerWorkerContext, graph_summary_offset) == 120, "graph view exceeds its line");
static_assert(offsetof(SchedulerWorkerContext, task_metadata_offset) == 128, "dispatch configuration line changed");
static_assert(offsetof(SchedulerWorkerContext, cluster_count) == 160, "topology configuration offset changed");
static_assert(offsetof(SchedulerWorkerContext, trace_cells_offset) == 208, "diagnostic offsets changed");
static_assert(offsetof(SchedulerWorkerContext, args_dump_address) == 232, "argument capture address changed");
static_assert(offsetof(SchedulerWorkerContext, prepared_layout_offset) == 40, "prepared execution address changed");
static_assert(offsetof(SchedulerWorkerContext, bootstrap_task_count) == 256, "worker diagnostics need an owned line");
static_assert(sizeof(SchedulerTaskTrace) == 256, "task trace layout changed");
static_assert(offsetof(SchedulerTaskTrace, execution_run_epoch) == 192, "execution identity offset changed");
static_assert(offsetof(SchedulerTaskTrace, execution_cluster_lane) == 240, "execution identity exceeds trace tail");
static_assert(
    offsetof(SchedulerTaskTrace, dispatch_start_cycles) % 64 == 0,
    "Scheduler-owned dispatch and completion fields must begin on a cache-line boundary"
);
static_assert(
    offsetof(SchedulerTaskTrace, complete_loop_iter) / 64 == offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64,
    "dispatch and completion fields must share their Scheduler-owned cache line"
);

template <typename T>
inline __aicore__ __gm__ T *scheduler_state_at(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

// Worker i's context, from the base worker 0 sits at — the address the host
// publishes in DeviceRuntimeLaunchDesc::scheduler_bootstrap. The AICore derives
// its own pre-READY context with this and the AICPU republishes the same address
// onto each handshake at hand-off, so the stride has one definition.
inline __aicore__ uint64_t scheduler_worker_context_address(uint64_t worker_context_base, int32_t worker_index) {
    return worker_context_base + static_cast<uint64_t>(worker_index) * sizeof(SchedulerWorkerContext);
}

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<AicoreSchedulerLayout> && std::is_trivially_copyable_v<AicoreSchedulerLayout>);
static_assert(std::is_standard_layout_v<SchedulerIdleRecord> && std::is_trivially_copyable_v<SchedulerIdleRecord>);
static_assert(
    std::is_standard_layout_v<SchedulerActivityBuffer> && std::is_trivially_copyable_v<SchedulerActivityBuffer>
);
static_assert(std::is_standard_layout_v<SchedulerTaskMetadata> && std::is_trivially_copyable_v<SchedulerTaskMetadata>);
static_assert(std::is_standard_layout_v<SchedulerTaskControl> && std::is_trivially_copyable_v<SchedulerTaskControl>);
static_assert(
    std::is_standard_layout_v<SchedulerSpmdRequirements> && std::is_trivially_copyable_v<SchedulerSpmdRequirements>
);
static_assert(
    std::is_standard_layout_v<SchedulerAdmissionRecord> && std::is_trivially_copyable_v<SchedulerAdmissionRecord>
);
static_assert(
    std::is_standard_layout_v<SchedulerAdmissionControl> && std::is_trivially_copyable_v<SchedulerAdmissionControl>
);
static_assert(
    std::is_standard_layout_v<SchedulerGroupDescriptor> && std::is_trivially_copyable_v<SchedulerGroupDescriptor>
);
static_assert(std::is_standard_layout_v<SchedulerGroupLaneAck> && std::is_trivially_copyable_v<SchedulerGroupLaneAck>);
static_assert(std::is_standard_layout_v<SchedulerGroupMailbox> && std::is_trivially_copyable_v<SchedulerGroupMailbox>);
static_assert(std::is_standard_layout_v<SchedulerGraphSummary> && std::is_trivially_copyable_v<SchedulerGraphSummary>);
static_assert(std::is_standard_layout_v<SchedulerRunControl> && std::is_trivially_copyable_v<SchedulerRunControl>);
static_assert(
    std::is_standard_layout_v<AicpuThreadLifecycleTrace> && std::is_trivially_copyable_v<AicpuThreadLifecycleTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerWorkerContext> && std::is_trivially_copyable_v<SchedulerWorkerContext>
);
static_assert(std::is_standard_layout_v<SchedulerTaskTrace> && std::is_trivially_copyable_v<SchedulerTaskTrace>);

inline bool scheduler_layout_checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool scheduler_layout_checked_mul(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool scheduler_layout_checked_align(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!scheduler_layout_checked_add(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool scheduler_layout_reserve(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !scheduler_layout_checked_align(*cursor, alignment, &aligned))
        return false;
    uint64_t end = 0;
    if (!scheduler_layout_checked_add(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool scheduler_spmd_requirements_valid(const SchedulerSpmdRequirements &spmd, uint64_t task_count) {
    if (spmd.cooperative_task_count > task_count || spmd.cooperative_task_count > INT32_MAX ||
        spmd.mix_task_count > task_count || spmd.sync_mix_task_count > spmd.cooperative_task_count ||
        spmd.sync_mix_task_count > spmd.mix_task_count)
        return false;
    if (spmd.worker_count == 0)
        return spmd.cluster_count == 0 && spmd.cooperative_task_count == 0 && spmd.mix_task_count == 0;
    return spmd.worker_count <= SCHEDULER_WORKER_CAPACITY && spmd.worker_count % PLATFORM_CORES_PER_BLOCKDIM == 0 &&
           spmd.cluster_count == spmd.worker_count / PLATFORM_CORES_PER_BLOCKDIM;
}

inline bool scheduler_plan_spmd_storage(uint64_t *cursor, AicoreSchedulerLayout *layout) {
    if (cursor == nullptr || layout == nullptr) return false;
    if (layout->spmd.cooperative_task_count == 0) return true;
    layout->admission_record_stride = sizeof(SchedulerAdmissionRecord);
    uint64_t bytes = 0;
    if (!scheduler_layout_reserve(
            cursor, sizeof(SchedulerAdmissionControl), alignof(SchedulerAdmissionControl),
            &layout->admission_control_offset
        ) ||
        !scheduler_layout_checked_mul(layout->task_count, layout->admission_record_stride, &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->admission_records_offset) ||
        !scheduler_layout_checked_mul(layout->spmd.cooperative_task_count, sizeof(int64_t), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->admission_log_offset))
        return false;
    if (layout->spmd.sync_mix_task_count == 0) return true;
    if (!scheduler_layout_checked_mul(layout->spmd.cluster_count, sizeof(SchedulerGroupMailbox), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, alignof(SchedulerGroupMailbox), &layout->group_mailboxes_offset) ||
        !scheduler_layout_checked_mul(
            layout->spmd.sync_mix_task_count, layout->spmd.cluster_count, &layout->mix_close_seal_count
        ) ||
        !scheduler_layout_checked_mul(layout->mix_close_seal_count, sizeof(SchedulerMixCloseSeal), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->mix_close_seals_offset))
        return false;
    return true;
}

inline bool scheduler_plan_protocol_storage(uint64_t *cursor, AicoreSchedulerLayout *layout, uint64_t trace_count) {
    if (cursor == nullptr || layout == nullptr) return false;
    if (layout->spmd.worker_count == 0) return false;
    auto &protocol = layout->protocol;
    protocol.magic = SCHEDULER_PROTOCOL_MAGIC;
    protocol.task_count = layout->task_count;
    protocol.worker_count = layout->spmd.worker_count;
    protocol.trace_count = trace_count;
    protocol.cooperative_task_count = layout->spmd.cooperative_task_count;
    if (!scheduler_layout_checked_mul(protocol.cooperative_task_count, protocol.worker_count, &protocol.member_count))
        return false;
    uint64_t bytes = 0;
#define SCHEDULER_PROTOCOL_RESERVE(count, type, field)              \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(cursor, bytes, alignof(type), &(field)))
    const bool valid =
        SCHEDULER_PROTOCOL_RESERVE(1, SchedulerProtocolLayout, layout->protocol_layout_offset) &&
        SCHEDULER_PROTOCOL_RESERVE(protocol.task_count, SchedulerProtocolTaskTrace, protocol.tasks_offset) &&
        SCHEDULER_PROTOCOL_RESERVE(protocol.member_count, SchedulerProtocolMemberTrace, protocol.members_offset) &&
        SCHEDULER_PROTOCOL_RESERVE(protocol.trace_count, SchedulerProtocolBlockTrace, protocol.blocks_offset) &&
        SCHEDULER_PROTOCOL_RESERVE(protocol.trace_count, SchedulerProtocolExecutionTrace, protocol.executions_offset) &&
        SCHEDULER_PROTOCOL_RESERVE(protocol.worker_count, SchedulerProtocolWorkerTrace, protocol.workers_offset);
#undef SCHEDULER_PROTOCOL_RESERVE
    if (valid) protocol.storage_end_offset = *cursor;
    return valid;
}

inline bool scheduler_plan_prepared_storage(uint64_t *cursor, AicoreSchedulerLayout *layout, uint64_t execution_count) {
    if (cursor == nullptr || layout == nullptr) return false;
    if (execution_count == 0) return true;
    auto &prepared = layout->prepared;
    prepared.task_count = layout->task_count;
    prepared.execution_count = execution_count;
    uint64_t bytes = 0;
#define SCHEDULER_PREPARED_RESERVE(count, type, field)              \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(cursor, bytes, alignof(type), &(field)))
    const bool valid =
        SCHEDULER_PREPARED_RESERVE(1, SchedulerPreparedLayout, layout->prepared_layout_offset) &&
        SCHEDULER_PREPARED_RESERVE(layout->task_count, SchedulerExecutionPlan, prepared.plans_offset) &&
        SCHEDULER_PREPARED_RESERVE(execution_count, SchedulerPreparedControl, prepared.controls_offset) &&
        SCHEDULER_PREPARED_RESERVE(execution_count, SchedulerExecutionBinding, prepared.bindings_offset) &&
        SCHEDULER_PREPARED_RESERVE(execution_count, SchedulerPreparedCall, prepared.calls_offset);
#undef SCHEDULER_PREPARED_RESERVE
    return valid;
}

inline bool scheduler_plan_layout(
    uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, AicoreSchedulerLayout *layout,
    bool enable_activity_profiling = false, uint64_t trace_count = 0, const SchedulerSpmdRequirements &spmd = {},
    bool enable_protocol_profiling = false, bool enable_pmu = false, uint64_t native_readonly_end_offset = 0
) {
    // Expanded node references and scan cursors fit the validated task-count bound.
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count ||
        task_count > static_cast<uint64_t>(INT32_MAX) || !scheduler_spmd_requirements_valid(spmd, task_count))
        return false;
    if (native_readonly_end_offset != 0 &&
        (native_readonly_end_offset < sizeof(SchedulerRunControl) ||
         native_readonly_end_offset % SCHEDULER_STATE_ALIGNMENT != 0 || spmd.worker_count == 0))
        return false;
    AicoreSchedulerLayout next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    next.spmd = spmd;
    next.native_readonly_end_offset = native_readonly_end_offset;
    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define SCHEDULER_RESERVE_ARRAY(count, type, field)                 \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(&cursor, bytes, alignof(type), &next.field))
    if (!scheduler_layout_reserve(
            &cursor, sizeof(SchedulerRunControl), alignof(SchedulerRunControl), &next.run_control_offset
        ))
        return false;
    if (native_readonly_end_offset != 0) {
        cursor = native_readonly_end_offset;
    }
    if (!SCHEDULER_RESERVE_ARRAY(
            PLATFORM_MAX_AICPU_THREADS, AicpuThreadLifecycleTrace, aicpu_lifecycle_traces_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerWorkerContext, worker_contexts_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskMetadata, task_metadata_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskControl, task_controls_offset) ||
        !scheduler_layout_reserve(
            &cursor, sizeof(SchedulerGraphSummary), alignof(SchedulerGraphSummary), &next.graph_summary_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(trace_count == 0 ? task_count : trace_count, SchedulerTaskTrace, trace_cells_offset) ||
        (enable_pmu && !SCHEDULER_RESERVE_ARRAY(
                           trace_count == 0 ? task_count : trace_count, SchedulerTaskPmuCell, pmu_cells_offset
                       )) ||
        (enable_activity_profiling &&
         !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerActivityBuffer, activity_buffers_offset)) ||
        !scheduler_plan_spmd_storage(&cursor, &next) || !scheduler_plan_prepared_storage(&cursor, &next, trace_count) ||
        (enable_protocol_profiling &&
         !scheduler_plan_protocol_storage(&cursor, &next, trace_count == 0 ? task_count : trace_count)) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) {
#undef SCHEDULER_RESERVE_ARRAY
        return false;
    }
#undef SCHEDULER_RESERVE_ARRAY
    *layout = next;
    return true;
}

// Record payload is a write-only device tail. Only its metadata belongs to the
// initialized/uploaded prefix, including when async storage precedes this tail.
inline bool scheduler_plan_scalar_storage(AicoreSchedulerLayout *layout) {
    if (layout == nullptr) return false;
    if (layout->activity_buffers_offset == 0) return true;
    return scheduler_layout_reserve(
        &layout->total_size,
        uint64_t{SCHEDULER_WORKER_CAPACITY} * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord),
        SCHEDULER_STATE_ALIGNMENT, &layout->scalar_records_offset
    );
}

inline bool scheduler_init_data_from_layout(
    void *base, const AicoreSchedulerLayout &layout, uint64_t device_base_address, uint64_t initialized_bytes = 0
) {
    const uint64_t prefix_end = layout.scalar_records_offset == 0 ? layout.total_size : layout.scalar_records_offset;
    if (initialized_bytes == 0) initialized_bytes = prefix_end;
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_BASE_ALIGNMENT - 1)) != 0 ||
        device_base_address == 0 || (device_base_address & (SCHEDULER_STATE_BASE_ALIGNMENT - 1)) != 0 ||
        layout.total_size > UINT64_MAX - device_base_address || initialized_bytes > prefix_end ||
        (layout.async_layout_offset == 0 ? initialized_bytes != prefix_end :
                                           initialized_bytes < layout.async_layout_offset))
        return false;
    if (layout.prepared_layout_offset != 0 &&
        ((reinterpret_cast<uintptr_t>(base) | device_base_address) & (SCHEDULER_PREPARED_ALIGNMENT - 1)) != 0)
        return false;
    __builtin_memset(base, 0, static_cast<size_t>(initialized_bytes));
    if (layout.scalar_records_offset != 0) {
        auto *buffers = scheduler_state_at<SchedulerActivityBuffer>(base, layout.activity_buffers_offset);
        for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
            buffers[worker].scalar_storage_format = 1;
            buffers[worker].scalar_records_offset =
                layout.scalar_records_offset +
                worker * SCHEDULER_SCALAR_RECORD_CAPACITY * sizeof(SchedulerScalarRecord);
        }
    }
    if (layout.protocol_layout_offset != 0) {
        *scheduler_state_at<SchedulerProtocolLayout>(base, layout.protocol_layout_offset) = layout.protocol;
        auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(base, layout.protocol.workers_offset);
        for (uint64_t worker = 0; worker < layout.protocol.worker_count; ++worker)
            workers[worker].worker_index = worker;
    }
    auto *run = scheduler_state_at<SchedulerRunControl>(base, layout.run_control_offset);
    run->error_run_epoch = UINT64_MAX;
    run->error_order_key = UINT64_MAX;
    run->error_block_idx = UINT64_MAX;
    run->error_group_seq = UINT64_MAX;
    run->error_worker_index = UINT64_MAX;
    run->error_cluster_lane = UINT32_MAX;
    auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].completion_state = scheduler_completion_initial_state(1);
        controls[i].parked_count = 0;
    }
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(base, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        contexts[worker].physical_core_id = -1;
        contexts[worker].cluster_index = UINT64_MAX;
        contexts[worker].activity_buffers_offset = layout.activity_buffers_offset;
        contexts[worker].prepared_layout_offset = layout.prepared_layout_offset;
        contexts[worker].cluster_worker_ids[0] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[1] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[2] = UINT64_MAX;
    }
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, layout.graph_summary_offset);
    summary->mix_task_count = layout.spmd.mix_task_count;
    summary->sync_mix_task_count = layout.spmd.sync_mix_task_count;
    summary->cooperative_task_count = layout.spmd.cooperative_task_count;
    summary->worker_count = layout.spmd.worker_count;
    summary->cluster_count = layout.spmd.cluster_count;
    summary->admission_records_offset = layout.admission_records_offset;
    summary->admission_record_stride = layout.admission_record_stride;
    summary->admission_log_offset = layout.admission_log_offset;
    summary->admission_control_offset = layout.admission_control_offset;
    summary->group_mailboxes_offset = layout.group_mailboxes_offset;
    summary->mix_close_seals_offset = layout.mix_close_seals_offset;
    summary->protocol_layout_offset = layout.protocol_layout_offset;
    summary->async_layout_offset = layout.async_layout_offset;
    summary->prepared_layout_offset = layout.prepared_layout_offset;
    if (layout.prepared_layout_offset != 0)
        *scheduler_state_at<SchedulerPreparedLayout>(base, layout.prepared_layout_offset) = layout.prepared;
    if (layout.spmd.cooperative_task_count != 0) {
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const uint64_t offset = layout.admission_records_offset + task * layout.admission_record_stride;
            auto *record = scheduler_state_at<SchedulerAdmissionRecord>(base, offset);
            record->task_ref = SCHEDULER_TASK_REF_INVALID;
            record->order_key = UINT64_MAX;
        }
        // Log elements use bypass stores; isolated close seals use atomics.
        // Neither region permits device cached stores or dirty writeback.
        auto *log = scheduler_state_at<int64_t>(base, layout.admission_log_offset);
        for (uint64_t index = 0; index < layout.spmd.cooperative_task_count; ++index)
            log[index] = SCHEDULER_TASK_REF_INVALID;
    }
    if (layout.spmd.sync_mix_task_count != 0) {
        auto *mailboxes = scheduler_state_at<SchedulerGroupMailbox>(base, layout.group_mailboxes_offset);
        for (uint64_t cluster = 0; cluster < layout.spmd.cluster_count; ++cluster) {
            auto &descriptor = mailboxes[cluster].descriptor;
            descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
            descriptor.order_key = UINT64_MAX;
            descriptor.block_idx = UINT32_MAX;
            descriptor.cluster_index = UINT32_MAX;
        }
    }
    return true;
}

inline bool
scheduler_init_prepared_execution_records(void *base, const AicoreSchedulerLayout &layout, uint64_t run_epoch) {
    const auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
    auto *plans = layout.prepared_layout_offset == 0 ?
                      nullptr :
                      scheduler_state_at<SchedulerExecutionPlan>(base, layout.prepared.plans_offset);
    auto *bindings = layout.prepared_layout_offset == 0 ?
                         nullptr :
                         scheduler_state_at<SchedulerExecutionBinding>(base, layout.prepared.bindings_offset);
    uint64_t next_execution = 0;
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        if (plans != nullptr) plans[task].execution_begin = next_execution;
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        const uint32_t lanes = static_cast<uint32_t>(meta.active_mask.count());
        uint32_t count = 0;
        uint64_t end = 0;
        if (plans == nullptr || !scheduler_subtask_count_checked(meta.logical_block_num, lanes, &count) ||
            !scheduler_layout_checked_add(next_execution, count, &end) || end > layout.prepared.execution_count)
            return false;
        plans[task] = {
            next_execution, count, meta.logical_block_num, meta.active_mask.raw(), static_cast<uint8_t>(lanes)
        };
        for (uint32_t block = 0; block < meta.logical_block_num; ++block) {
            for (uint32_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
                if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                auto &binding = bindings[next_execution++];
                binding.local_context.block_idx = block;
                binding.local_context.block_num = meta.logical_block_num;
            }
        }
    }
    if (next_execution != layout.prepared.execution_count) return false;
    if (layout.prepared_layout_offset != 0)
        scheduler_state_at<SchedulerPreparedLayout>(base, layout.prepared_layout_offset)->run_epoch = run_epoch;
    return true;
}

inline bool scheduler_init_spmd_task_records(void *base, const AicoreSchedulerLayout &layout, uint64_t run_epoch) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_BASE_ALIGNMENT - 1)) != 0)
        return false;
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    uint64_t cooperative_count = 0;
    uint64_t mix_count = 0;
    uint64_t sync_mix_count = 0;
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        const uint32_t active_subtasks = static_cast<uint32_t>(meta.active_mask.count());
        uint32_t expected_subtasks = 0;
        if (!meta.active_mask.is_valid() ||
            !scheduler_subtask_count_checked(meta.logical_block_num, active_subtasks, &expected_subtasks) ||
            scheduler_task_is_mix(meta.flags) != (active_subtasks > 1) ||
            scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
            (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) ||
            (scheduler_task_has_predicate(meta.flags) && (meta.logical_block_num != 1 || active_subtasks != 1)) ||
            meta.timing_slot < -1 || meta.timing_slot >= SCHEDULER_TASK_TIMING_SLOT_COUNT)
            return false;
        if (scheduler_task_requires_admission(meta)) ++cooperative_count;
        if (scheduler_task_is_mix(meta.flags)) {
            ++mix_count;
            if (scheduler_task_requires_sync_start(meta.flags)) ++sync_mix_count;
        }
    }
    if (cooperative_count != layout.spmd.cooperative_task_count || mix_count != layout.spmd.mix_task_count ||
        sync_mix_count != layout.spmd.sync_mix_task_count)
        return false;
    if (!scheduler_init_prepared_execution_records(base, layout, run_epoch)) return false;

    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, layout.graph_summary_offset);
    summary->run_epoch = run_epoch;
    if (layout.protocol_layout_offset != 0) {
        auto *protocol = scheduler_state_at<SchedulerProtocolLayout>(base, layout.protocol_layout_offset);
        protocol->run_epoch = run_epoch;
        auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(base, protocol->workers_offset);
        for (uint64_t worker = 0; worker < protocol->worker_count; ++worker)
            workers[worker].run_epoch = run_epoch;
    }
    uint64_t mix_index = 0;
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto &meta = metadata[task];
        const uint32_t contributions =
            scheduler_task_is_executable(meta.flags) && !scheduler_task_is_inline(meta.flags) ?
                meta.logical_block_num * meta.active_mask.count() :
                1;
        controls[task].completion_state = scheduler_completion_initial_state(contributions);
        if (!scheduler_task_requires_admission(meta)) continue;
        auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base, layout.admission_records_offset + task * layout.admission_record_stride
        );
        record->task_ref = static_cast<int64_t>(task);
        record->logical_block_num = meta.logical_block_num;
        record->active_mask = meta.active_mask;
        record->flags = meta.flags;
        if (scheduler_task_is_mix(meta.flags)) {
            record->mix_close_seals_offset =
                layout.mix_close_seals_offset + mix_index * layout.spmd.cluster_count * sizeof(SchedulerMixCloseSeal);
            ++mix_index;
        }
    }
    return true;
}
#endif
