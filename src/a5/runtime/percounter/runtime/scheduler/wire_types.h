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
 * Every offset here is from the region base, never a pointer, so the whole
 * region survives one memcpy to the device.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

// scheduler_protocol_types.h, which scheduler_layout.h pulls in, names LaneMask
// without including it; host_build_graph's own translation units happen to have
// it in scope by then, and this runtime's do not.
#include "async_completion_types.h"
#include "common/lane_mask.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/ssbuf.h"
#include "scheduler/scheduler_types.h"

// Continues the SCHEDULER_RUNTIME_MODE_* numbering scheduler_layout.h starts,
// which reaches 4. The value is published into SchedulerBootstrapInputs and read
// back by every tier of this runtime, so it has to agree with those four even
// though the host_build_graph runtime never produces it.
constexpr uint32_t SCHEDULER_RUNTIME_MODE_PERCOUNTER = 5;

namespace simpler::percounter {

inline constexpr uint32_t PERCOUNTER_CORE_TYPE_COUNT = 2;  // 0 = AIC, 1 = AIV

inline constexpr uint32_t PERCOUNTER_PENDING_CAPACITY = 1;
inline constexpr uint32_t PERCOUNTER_PAYLOAD_SLOT_COUNT = 1;
static_assert(PERCOUNTER_PENDING_CAPACITY == 1 || PERCOUNTER_PENDING_CAPACITY == 2 || PERCOUNTER_PENDING_CAPACITY == 4);
static_assert(PERCOUNTER_PAYLOAD_SLOT_COUNT == 1 || PERCOUNTER_PAYLOAD_SLOT_COUNT == 2);
inline constexpr uint32_t PERCOUNTER_SLOT_MASK = (1U << PERCOUNTER_PENDING_CAPACITY) - 1;
inline constexpr uint32_t PERCOUNTER_NO_SYNC = UINT32_MAX;

// Payload and publication generation have one writer: the cluster's AIV0.
// Each participant exclusively writes its own entry and completion token.
struct alignas(64) PercounterMixSignal {
    volatile uint64_t entry;
    volatile uint32_t global_order_id;
    volatile uint32_t active_mask;
    volatile uint32_t generation;
    volatile uint32_t fanin_ready;
    volatile uint32_t enter_ready[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t enter_idle[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t finished[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t completed;
    volatile uint32_t sequence;
    volatile uint32_t start_arrived;
    volatile uint32_t fanin_cursor;
    volatile uint32_t start_ready;  // Lowest active lane publishes global arrival through SSBUF.
    volatile uint64_t fanin_address;
    volatile int64_t fanin_body_base;
    volatile uint32_t fanin_count;
    volatile uint32_t fanin_body;
    volatile uint32_t metadata_generation;
    volatile uint32_t sync_index;
    volatile uint64_t fanin_pending;
    volatile uint64_t fanin_other;
};
static_assert(sizeof(PercounterMixSignal) == 128);
static_assert(std::is_trivially_copyable_v<PercounterMixSignal> && std::is_standard_layout_v<PercounterMixSignal>);

struct alignas(64) PercounterSsbufRegion {
    PercounterMixSignal mix[PERCOUNTER_PENDING_CAPACITY];
    volatile uint32_t mix_floor_rank;
    volatile uint32_t publish_sequence;
    volatile uint32_t publication_closed;
    uint8_t reserved[PERCOUNTER_SSBUF_REGION_SIZE - PERCOUNTER_PENDING_CAPACITY * sizeof(PercounterMixSignal) - 12];
};
static_assert(sizeof(PercounterSsbufRegion) == PERCOUNTER_SSBUF_REGION_SIZE);
static_assert(std::is_trivially_copyable_v<PercounterSsbufRegion> && std::is_standard_layout_v<PercounterSsbufRegion>);

inline __aicore__ PERCOUNTER_SSBUF PercounterSsbufRegion *percounter_ssbuf_region(uint64_t hardware_base) {
    return reinterpret_cast<PERCOUNTER_SSBUF PercounterSsbufRegion *>(hardware_base + PERCOUNTER_SSBUF_REGION_OFFSET);
}

inline __aicore__ void percounter_mix_signal_clear(PERCOUNTER_SSBUF PercounterMixSignal *signal) {
    percounter_ssbuf_store_field(&signal->fanin_ready, uint32_t{0});
    percounter_ssbuf_store_field(&signal->completed, uint32_t{0});
    percounter_ssbuf_store_field(&signal->start_arrived, uint32_t{0});
    percounter_ssbuf_store_field(&signal->start_ready, uint32_t{0});
    percounter_ssbuf_store_field(&signal->metadata_generation, uint32_t{0});
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        percounter_ssbuf_store_field(&signal->enter_ready[lane], uint32_t{0});
        percounter_ssbuf_store_field(&signal->enter_idle[lane], uint32_t{0});
        percounter_ssbuf_store_field(&signal->finished[lane], uint32_t{0});
    }
    percounter_ssbuf_store_release(&signal->generation, 0);
}

inline __aicore__ bool percounter_mix_armed(uint32_t generation) { return (generation & 1U) != 0; }

/** Independent atomic groups occupy distinct 128-byte GM sectors.
 * Scalar DCache maintenance still operates on 64-byte lines.
 */
struct alignas(128) PercounterCounter {
    // 64-bit to match percounter_gm_fetch_add, the runtime's only atomic add.
    uint64_t value;
    // Graph body identity and fanin are immutable after metadata_ready.
    uint64_t storage_address;  // the invocation's ChipTaskStorage array
    uint32_t body_index;       // this task's index in it, and in its body
    uint32_t flags;            // PERCOUNTER_BODY_*
    uint32_t shell_task_id;
    uint32_t fanin_count;
    // Region offset of uint64 body references: (producer index << 1) | partitioned.
    // A root observes its shell's fanin instead.
    uint64_t fanin_offset;
    uint32_t invocation;
    // Host prepares task-table arguments; AICPU prepares Graph body arguments.
    uint32_t argument_counts;
    uint64_t function_addresses[PLATFORM_CORES_PER_BLOCKDIM];
    uint64_t tensors_address;
    uint64_t scalars_address;
    uint8_t reserved[40];
};
static_assert(sizeof(PercounterCounter) == 128, "a counter owns one GM sector");
static_assert(std::is_trivially_copyable_v<PercounterCounter> && std::is_standard_layout_v<PercounterCounter>);
static_assert(alignof(PercounterCounter) == 128, "counters must not straddle lines");

/**
 * A completion component is published when it reads positive as a signed value.
 *
 * It starts at `1 - component_subtasks` and every subtask adds one, so the
 * value walks `1 - total .. 0` and lands on exactly 1. Nothing has to detect the
 * last arriver. Multi-block tasks partition blocks across independent
 * device-only sectors; every component must be positive.
 *
 * Signed `> 0` rather than `== 1`: an equality test would turn a counter that
 * overshot into a consumer that waits forever. With `> 0`, a defect that ran a
 * subtask twice shows up as a wrong result rather than as an op timeout.
 *
 * A task completed inline at bind is preset to 1 directly, the value it would
 * have reached.
 */
inline constexpr uint64_t PERCOUNTER_COUNTER_PUBLISHED = 1;

inline __aicore__ bool percounter_counter_published(uint64_t value) { return static_cast<int64_t>(value) > 0; }

inline __aicore__ constexpr uint64_t percounter_counter_preset(int64_t total_subtasks) {
    return static_cast<uint64_t>(static_cast<int64_t>(PERCOUNTER_COUNTER_PUBLISHED) - total_subtasks);
}

// Bit zero tags a producer with partitioned completion components. GM counter
// references have an aligned address; Graph body references have a shifted id.
inline constexpr uint64_t PERCOUNTER_COMPLETION_PARTITIONED = 1;

inline constexpr uint32_t PERCOUNTER_ASYNC_CAPACITY = 4;
static_assert(PERCOUNTER_ASYNC_CAPACITY <= MAX_COMPLETIONS_PER_TASK);

// One physical invocation can append at most one completion record.
struct alignas(64) PercounterAsyncCompletion {
    uint64_t next;
    uint64_t pending_mask;
    uint64_t completion_address;
    int64_t shell;
    uint64_t sink;
    uint64_t reserved[3];
    DeferredCompletionEntry entries[PERCOUNTER_ASYNC_CAPACITY];
};
static_assert(sizeof(PercounterAsyncCompletion) == 192);
static_assert(
    std::is_trivially_copyable_v<PercounterAsyncCompletion> && std::is_standard_layout_v<PercounterAsyncCompletion>
);

/** A ticket owns an atomic sector independently of other groups. */
struct alignas(128) PercounterTicket {
    uint64_t next;
    uint8_t reserved[120];
};
static_assert(sizeof(PercounterTicket) == 128, "a ticket owns one GM sector");
static_assert(std::is_trivially_copyable_v<PercounterTicket> && std::is_standard_layout_v<PercounterTicket>);

// Arrival counts use the logical block index, independent of its claiming core.
inline constexpr uint32_t PERCOUNTER_START_SECTOR_COUNT = 4;
struct alignas(128) PercounterStartBarrier {
    PercounterTicket sectors[PERCOUNTER_START_SECTOR_COUNT];
};
static_assert(sizeof(PercounterStartBarrier) == 128 * PERCOUNTER_START_SECTOR_COUNT);
static_assert(
    std::is_trivially_copyable_v<PercounterStartBarrier> && std::is_standard_layout_v<PercounterStartBarrier>
);

inline constexpr uint32_t PERCOUNTER_CLAIM_SECTOR_COUNT = 4;
static_assert(
    PERCOUNTER_CLAIM_SECTOR_COUNT >= 2 && (PERCOUNTER_CLAIM_SECTOR_COUNT & (PERCOUNTER_CLAIM_SECTOR_COUNT - 1)) == 0
);
struct alignas(128) PercounterTaskClaim {
    PercounterTicket sectors[PERCOUNTER_CLAIM_SECTOR_COUNT];
    uint64_t closed_sectors;
    uint8_t reserved[120];
};
static_assert(sizeof(PercounterTaskClaim) == 128 * (PERCOUNTER_CLAIM_SECTOR_COUNT + 1));
static_assert(offsetof(PercounterTaskClaim, closed_sectors) % 128 == 0);
static_assert(std::is_trivially_copyable_v<PercounterTaskClaim> && std::is_standard_layout_v<PercounterTaskClaim>);

inline constexpr uint32_t PERCOUNTER_COMPLETION_SECTOR_COUNT = 4;
static_assert(
    PERCOUNTER_COMPLETION_SECTOR_COUNT >= 2 && PERCOUNTER_COMPLETION_SECTOR_COUNT <= 64 &&
    (PERCOUNTER_COMPLETION_SECTOR_COUNT & (PERCOUNTER_COMPLETION_SECTOR_COUNT - 1)) == 0
);
struct alignas(128) PercounterCompletionSector {
    uint64_t value;
    uint8_t reserved[120];
};
static_assert(sizeof(PercounterCompletionSector) == 128);
static_assert(
    std::is_trivially_copyable_v<PercounterCompletionSector> && std::is_standard_layout_v<PercounterCompletionSector>
);

struct alignas(128) PercounterClaimSummary {
    uint64_t closed;
    uint8_t reserved[120];
};
static_assert(sizeof(PercounterClaimSummary) == 128);
static_assert(
    std::is_trivially_copyable_v<PercounterClaimSummary> && std::is_standard_layout_v<PercounterClaimSummary>
);

inline constexpr uint32_t PERCOUNTER_NO_BOUNDARY = UINT32_MAX;

struct alignas(8) PercounterTaskMetadata {
    uint32_t global_order_id;
    uint16_t active_mask;
    uint16_t block_num;
    uint32_t sync_index;
    uint32_t next_boundary_segment;
};
static_assert(sizeof(PercounterTaskMetadata) == 16);
static_assert(
    std::is_trivially_copyable_v<PercounterTaskMetadata> && std::is_standard_layout_v<PercounterTaskMetadata>
);

// [begin, end) contains ordinary single-block tasks. Each following MIX or
// SPMD task uses a private counter; no ticket is reset during a run.
struct PercounterSegment {
    uint64_t task_shape;
    uint32_t begin;
    uint32_t end;
    uint64_t ticket_begin;
};
static_assert(sizeof(PercounterSegment) == 24);
static_assert(std::is_trivially_copyable_v<PercounterSegment> && std::is_standard_layout_v<PercounterSegment>);

// Zero denotes a tail interval. Every boundary task has at least one block.
inline constexpr uint64_t percounter_claim_shape(uint32_t task, uint32_t blocks, uint32_t mask, bool sync) {
    return uint64_t{task} | (uint64_t{mask | (sync ? 8U : 0U)} << 32) | (uint64_t{blocks} << 48);
}

inline bool percounter_sync_widths_fit(const uint64_t *widths, uint64_t clusters, uint64_t aivs_per_cluster) {
    return widths[0] <= clusters && widths[1] <= clusters * aivs_per_cluster && widths[2] <= clusters;
}

/**
 * AIC and vector-domain lists. The vector list contains ordinary AIV and MIX.
 * Indexes the per-list arrays of the two records below.
 */
inline constexpr uint32_t PERCOUNTER_LIST_COUNT = PERCOUNTER_CORE_TYPE_COUNT;
inline constexpr uint32_t PERCOUNTER_SYNC_DOMAIN_COUNT = 3;  // AIC, ordinary AIV, MIX

/** One body task of a Graph template. */
struct PercounterBodyTask {
    // Subtasks the task runs in all, `block_num * popcount(active_mask)`: what
    // its counter is preset against.
    uint32_t subtask_total;
    uint32_t flags;  // PERCOUNTER_BODY_*
};
static_assert(sizeof(PercounterBodyTask) == 8, "percounter body task is a wire struct");
static_assert(std::is_trivially_copyable_v<PercounterBodyTask> && std::is_standard_layout_v<PercounterBodyTask>);

// No producer inside the body: it waits on the shell's own fanin instead.
inline constexpr uint32_t PERCOUNTER_BODY_ROOT = 1U << 0;
// No consumer inside the body: each of its subtasks also counts toward the shell.
inline constexpr uint32_t PERCOUNTER_BODY_SINK = 1U << 1;
inline constexpr uint32_t PERCOUNTER_BODY_PREDICATE = 1U << 2;

/**
 * What percounter keeps of one Graph Definition, built once per Definition per
 * bind and shared by every invocation of it.
 *
 * The body's tasks are numbered 0..task_count-1 as in the Definition. The order
 * entries carry those body indices as their task id; an invocation's entries are
 * the same words with the invocation's first percounter id added.
 *
 * Every `*_offset` is from the region base.
 */
struct alignas(64) PercounterGraphTemplate {
    uint64_t task_count;
    uint64_t edge_count;
    uint64_t tasks_offset;
    uint64_t fanin_offsets_offset;
    // Each reference is (body producer index << 1) | completion-pair bit.
    uint64_t fanin_indices_offset;
};
static_assert(sizeof(PercounterGraphTemplate) == 64, "percounter graph template is a wire struct");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphTemplate> && std::is_standard_layout_v<PercounterGraphTemplate>
);

/**
 * One Graph invocation: an outer task of kind GRAPH and the body it runs.
 *
 * The host writes everything but `storage_address`, which exists only once the
 * AICPU has localized the invocation: the body's ChipTaskStorage array sits in
 * the shell's heap, and the host never sees that heap's real address.
 */
struct alignas(64) PercounterGraphInvocation {
    uint64_t template_offset;
    // The shell's percounter id, which is also its task-table id. Its fanin is
    // what the body's roots wait on, and its counter is what the sinks publish.
    uint64_t shell_task_id;
    // Percounter id of body task 0; body task i is `body_base + i`.
    uint64_t body_base;
    uint64_t storage_address;
};
static_assert(sizeof(PercounterGraphInvocation) == 64, "a graph invocation owns exactly one cache line");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphInvocation> && std::is_standard_layout_v<PercounterGraphInvocation>
);

/**
 * The AICPU threads' shared state while they materialize the invocations.
 *
 * Only AICPU threads touch it, and their caches are coherent with each other, so
 * its fields take plain atomics. The invocation records are in priority order,
 * so drawing `next` hands out the most urgent body first.
 */
struct alignas(64) PercounterGraphPrepare {
    uint64_t next;  // invocations claimed
    uint8_t reserved0[56];
    alignas(64) uint64_t prepared;  // invocations whose every body task is published
    uint64_t first_publish_cycles;  // when the first body tasks were published; 0 = not yet
    uint64_t prepared_cycles;       // when the last invocation was finished
    uint64_t error;                 // first failure, a SchedulerGraphResult
    uint64_t error_invocation;
    uint8_t reserved1[24];
};
static_assert(sizeof(PercounterGraphPrepare) == 128, "the prepare record is two cache lines");
static_assert(
    std::is_trivially_copyable_v<PercounterGraphPrepare> && std::is_standard_layout_v<PercounterGraphPrepare>
);

/**
 * Body tasks one AICPU draw materializes before it publishes them. Smaller
 * gets the first body task to a lane sooner; larger flushes and publishes less
 * often.
 */
inline constexpr int32_t PERCOUNTER_GRAPH_SLICE_TASKS = 16;

/** Where a task's producers live in the flat fanin-address array. */
struct PercounterFanin {
    int32_t count;
    int32_t begin;
};
static_assert(sizeof(PercounterFanin) == 8, "percounter fanin record is a wire struct");
static_assert(std::is_trivially_copyable_v<PercounterFanin> && std::is_standard_layout_v<PercounterFanin>);

/**
 * An order-list entry names one task. An owned entry identifies its claimed
 * block using the same encoding:
 *
 *   bits  0..31  task id
 *   bits 32..47  block index
 *   bits 48..63  block count
 *
 * Each task has one order-list entry. An SPMD entry has a sentinel block index;
 * a successful claim replaces it with the actual block index in the owned slot.
 *
 * The block count is carried here because the descriptor a lane can reach holds
 * the task id and the kernel ids only, and `logical_block_num` lives on the
 * AICPU-side slot state, which an AICore should not read by offset. For a
 * single-slot task the block count is also the subtask total the counter is
 * preset against.
 *
 * The orchestrator refuses any task whose `block_num * popcount(active_mask)`
 * exceeds INT16_MAX, so both halves fit 16 bits unsigned, and a task id is
 * non-negative and bounded by the graph's task count.
 */
inline constexpr uint32_t PERCOUNTER_ORDER_BLOCK_IDX_SHIFT = 32;
inline constexpr uint32_t PERCOUNTER_ORDER_BLOCK_NUM_SHIFT = 48;
inline constexpr uint64_t PERCOUNTER_ORDER_TASK_MASK = 0xffffffffULL;
inline constexpr uint64_t PERCOUNTER_ORDER_BLOCK_MASK = 0xffffULL;

inline __aicore__ uint64_t percounter_order_entry(int64_t task_id, int32_t block_idx, int32_t block_num) {
    return ((static_cast<uint64_t>(block_num) & PERCOUNTER_ORDER_BLOCK_MASK) << PERCOUNTER_ORDER_BLOCK_NUM_SHIFT) |
           ((static_cast<uint64_t>(block_idx) & PERCOUNTER_ORDER_BLOCK_MASK) << PERCOUNTER_ORDER_BLOCK_IDX_SHIFT) |
           (static_cast<uint64_t>(task_id) & PERCOUNTER_ORDER_TASK_MASK);
}

inline __aicore__ int64_t percounter_order_task_id(uint64_t entry) {
    return static_cast<int64_t>(entry & PERCOUNTER_ORDER_TASK_MASK);
}

inline __aicore__ int32_t percounter_order_block_idx(uint64_t entry) {
    return static_cast<int32_t>((entry >> PERCOUNTER_ORDER_BLOCK_IDX_SHIFT) & PERCOUNTER_ORDER_BLOCK_MASK);
}

inline __aicore__ int32_t percounter_order_block_num(uint64_t entry) {
    return static_cast<int32_t>((entry >> PERCOUNTER_ORDER_BLOCK_NUM_SHIFT) & PERCOUNTER_ORDER_BLOCK_MASK);
}

// An SPMD boundary names a task; its blocks come from its private claim counter.
inline constexpr int32_t PERCOUNTER_ORDER_BLOCK_IDX_SPMD = 0xffff;
inline __aicore__ bool percounter_order_is_spmd(uint64_t entry) {
    return percounter_order_block_idx(entry) == PERCOUNTER_ORDER_BLOCK_IDX_SPMD;
}

/**
 * The device-visible header: where every other segment is, and how many entries
 * each order list holds.
 */
struct alignas(128) PercounterControl {
    uint64_t task_count;
    // Top-level sinks: the tasks no task in the table consumes. Every task
    // reaches one along consumer edges, so their completion covers the graph.
    uint64_t sink_count;
    uint64_t sinks_offset;
    uint64_t edge_count;
    uint64_t order_count[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t orders_offset[PERCOUNTER_CORE_TYPE_COUNT];
    uint64_t tickets_offset;
    uint64_t mix_task_count;
    uint64_t boundary_task_count;
    uint64_t task_claims_offset;
    uint64_t claim_summary_offset;
    uint64_t sync_cohort_count;
    uint64_t sync_arrivals_offset;
    uint64_t metadata_offset;
    uint64_t metadata_ready_offset;
    uint64_t segments_offset;
    uint64_t segment_begin[PERCOUNTER_LIST_COUNT];
    uint64_t segment_count[PERCOUNTER_LIST_COUNT];
    uint64_t max_sync_blocks[PERCOUNTER_SYNC_DOMAIN_COUNT];
    uint64_t has_predicates;
    uint64_t table_task_count;
    uint64_t graph_invocation_count;
    uint64_t graph_invocations_offset;
    uint64_t graph_prepare_offset;
    uint64_t mix_signal_ready_offset;
    uint64_t fanin_offset;
    uint64_t fanin_addr_offset;
    uint64_t counters_offset;
    uint64_t cluster_count;
    uint64_t aiv_per_cluster;
    uint64_t lane_error;
    uint64_t graph_storage_address;
    uint64_t graph_storage_count;
    uint64_t callable_addresses_address;
    uint64_t callable_addresses_count;
    uint64_t dispatch_payloads_offset;
    // One deferred-completion record per lane and the entry slots behind it,
    // indexed the way the payloads are and for the same reason. The condition
    // cap is one record's, and the epoch is what tells a record of this run
    // from one the retained storage still held.
    uint64_t async_groups_offset;
    uint64_t async_entries_offset;
    uint64_t async_capacity;
    uint64_t async_run_epoch;
    uint64_t async_completions_offset;
    uint64_t async_completion_capacity;
    uint64_t scheduler_timeout_cycles;
    uint64_t profiling_level;
    uint64_t profile_buffers_offset;
    uint64_t lane_error_claim;
    uint64_t lane_error_task;
    uint64_t lane_error_detail;
    uint32_t first_boundary_segment[PERCOUNTER_SYNC_DOMAIN_COUNT];
    uint64_t completion_sectors_offset;

    // Allocation precedes the corresponding block's completed publication.
    // Both counters are monotone; equality after all blocks ran permits exit.
    alignas(128) uint64_t async_registered;
    uint64_t async_settled;
    uint8_t async_reserved[112];
};
static_assert(sizeof(PercounterControl) == 640);
static_assert(
    offsetof(PercounterControl, async_registered) % 128 == 0,
    "the deferred-group counts are the atomics lanes write; they must start a cache line, or every reader of "
    "their neighbours pays for it"
);
static_assert(
    offsetof(PercounterControl, lane_error) / 128 != offsetof(PercounterControl, async_registered) / 128,
    "lane_error is read every task; it must not share a line with the counts lanes increment"
);
static_assert(std::is_trivially_copyable_v<PercounterControl> && std::is_standard_layout_v<PercounterControl>);

}  // namespace simpler::percounter
