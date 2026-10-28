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

#include <cstddef>
#include <cstdint>
#include <type_traits>

inline constexpr uint64_t SCHEDULER_PROTOCOL_MAGIC = UINT64_C(0x53504d4454520001);

enum SchedulerProtocolError : uint64_t {
    SCHEDULER_PROTOCOL_DUPLICATE = UINT64_C(1),
    SCHEDULER_PROTOCOL_SEQUENCE_OVERFLOW = UINT64_C(2),
    SCHEDULER_PROTOCOL_INVALID_IDENTITY = UINT64_C(4),
    SCHEDULER_PROTOCOL_AFTER_FINISH = UINT64_C(8),
};

struct SchedulerProtocolStamp {
    uint64_t cycles;
    uint64_t sequence;
};

struct alignas(128) SchedulerProtocolLayout {
    uint64_t magic;
    uint64_t run_epoch;
    uint64_t storage_end_offset;
    uint64_t task_count;
    uint64_t worker_count;
    uint64_t trace_count;
    uint64_t cooperative_task_count;
    uint64_t member_count;
    uint64_t tasks_offset;
    uint64_t members_offset;
    uint64_t blocks_offset;
    uint64_t executions_offset;
    uint64_t workers_offset;
    uint64_t reserved[3];
};

// Sequence zero denotes an unrecorded point, not cycle zero or epoch zero.
struct alignas(64) SchedulerProtocolTaskActor {
    SchedulerProtocolStamp before;
    SchedulerProtocolStamp after;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint32_t worker_index;
    uint32_t flags;
};

struct alignas(128) SchedulerProtocolTaskTrace {
    SchedulerProtocolTaskActor admission;
    SchedulerProtocolTaskActor completion;
};

struct alignas(128) SchedulerProtocolMemberTrace {
    SchedulerProtocolStamp observed;
    SchedulerProtocolStamp closed;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t observed_tail;
    uint64_t closed_cursor;
    uint64_t close_seal;
    uint64_t last_group_seq;
    uint32_t worker_index;
    uint32_t flags;
    uint64_t pending_mask;
    uint64_t reserved[3];
};

struct alignas(64) SchedulerProtocolBlockOwner {
    SchedulerProtocolStamp claim_before;
    SchedulerProtocolStamp claim_after;
    SchedulerProtocolStamp publish_before;
    SchedulerProtocolStamp publish_after;
    SchedulerProtocolStamp completion_before;
    SchedulerProtocolStamp completion_after;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t group_seq;
    uint64_t completion_ticket;
    uint32_t worker_index;
    uint32_t block_idx;
    uint32_t cluster_index;
    uint16_t logical_block_num;
    LaneMask active_mask;
    uint8_t flags;
    SchedulerProtocolStamp releases_observed;
    uint64_t released_seq[3];
};

struct alignas(128) SchedulerProtocolLaneTrace {
    SchedulerProtocolStamp arrive_before;
    SchedulerProtocolStamp arrive_after;
    SchedulerProtocolStamp release_before;
    SchedulerProtocolStamp release_after;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t group_seq;
    uint32_t worker_index;
    uint32_t block_idx;
    uint32_t cluster_index;
    uint32_t lane;
    LaneMask active_mask;
    uint32_t flags;
    uint64_t arrived_seq[3];
    SchedulerProtocolStamp descriptor_ready;
    SchedulerProtocolStamp arrivals_observed;
    uint32_t logical_block_num;
    LaneMask resource_mask;
    uint8_t lane_subtasks[3];
    uint8_t reserved[8];
};

struct alignas(128) SchedulerProtocolBlockTrace {
    SchedulerProtocolBlockOwner owner;
    SchedulerProtocolLaneTrace lanes[3];
};

struct alignas(128) SchedulerProtocolExecutionTrace {
    SchedulerProtocolStamp entry;
    SchedulerProtocolStamp returned;
    SchedulerProtocolStamp visible;
    SchedulerProtocolStamp completion_consumed;
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t permit_tail;
    uint32_t worker_index;
    uint32_t block_idx;
    uint32_t generation;
    uint16_t pending_slot;
    uint16_t subtask_slot;
    uint32_t flags;
    uint32_t ready_source;
    uint64_t group_seq;
    SchedulerProtocolStamp permit;
    SchedulerProtocolStamp trace_published;
    uint32_t logical_block_num;
    uint32_t cluster_index;
    uint32_t cluster_lane;
    uint32_t publication_mode;
    // Ordinary payloads can be rebuilt before their one entry permit. Only the final
    // publication is retained; these snapshot times consume no event sequence.
    uint64_t publication_before_cycles;
    uint64_t publication_after_cycles;
    uint32_t publication_count;
    uint32_t publication_reserved;
};

enum SchedulerProtocolExecutionFlags : uint32_t {
    SCHEDULER_PROTOCOL_ENTRY = 1U << 0,
    SCHEDULER_PROTOCOL_EXECUTED = 1U << 1,
    SCHEDULER_PROTOCOL_COOPERATIVE = 1U << 2,
    SCHEDULER_PROTOCOL_GM_GROUP = 1U << 3,
    SCHEDULER_PROTOCOL_PERMIT_TAIL = 1U << 4,
    SCHEDULER_PROTOCOL_PERMIT_NO_LOG = 1U << 5,
    SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED = 1U << 6,
};

// One worker owns this record; no global event counter participates in scheduling.
struct alignas(128) SchedulerProtocolWorkerTrace {
    uint64_t run_epoch;
    uint64_t worker_index;
    volatile uint64_t next_sequence;
    volatile uint64_t error_flags;
    volatile uint64_t finished;
    uint64_t reserved[11];
};

static_assert(sizeof(SchedulerProtocolStamp) == 16);
static_assert(sizeof(SchedulerProtocolLayout) == 128);
static_assert(sizeof(SchedulerProtocolTaskActor) == 64);
static_assert(sizeof(SchedulerProtocolTaskTrace) == 128);
static_assert(offsetof(SchedulerProtocolTaskTrace, completion) == 64);
static_assert(sizeof(SchedulerProtocolMemberTrace) == 128);
static_assert(sizeof(SchedulerProtocolBlockOwner) == 192);
static_assert(sizeof(SchedulerProtocolLaneTrace) == 256);
static_assert(offsetof(SchedulerProtocolBlockTrace, lanes) == 256);
static_assert(sizeof(SchedulerProtocolBlockTrace) == 1024);
static_assert(sizeof(SchedulerProtocolExecutionTrace) == 256);
static_assert(sizeof(SchedulerProtocolWorkerTrace) == 128);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolLayout> && std::is_trivially_copyable_v<SchedulerProtocolLayout>
);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolTaskTrace> && std::is_trivially_copyable_v<SchedulerProtocolTaskTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolMemberTrace> &&
    std::is_trivially_copyable_v<SchedulerProtocolMemberTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolBlockTrace> && std::is_trivially_copyable_v<SchedulerProtocolBlockTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolExecutionTrace> &&
    std::is_trivially_copyable_v<SchedulerProtocolExecutionTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerProtocolWorkerTrace> &&
    std::is_trivially_copyable_v<SchedulerProtocolWorkerTrace>
);
