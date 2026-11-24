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
#include "invocation_context.h"
#include "scheduler/scheduler_types.h"

// Continues the SCHEDULER_RUNTIME_MODE_* numbering scheduler_layout.h starts,
// which reaches 4. The value is published into SchedulerBootstrapInputs and read
// back by every tier of this runtime, so it has to agree with those four even
// though the host_build_graph runtime never produces it.
constexpr uint32_t SCHEDULER_RUNTIME_MODE_PERCOUNTER = 5;

namespace simpler::percounter {

inline constexpr uint32_t PERCOUNTER_NO_SYNC = UINT32_MAX;

struct PercounterMixSignal {
    volatile uint32_t arrive[PLATFORM_CORES_PER_BLOCKDIM];
    volatile uint32_t release[PLATFORM_CORES_PER_BLOCKDIM];
};

struct alignas(64) PercounterSsbufRegion {
    PercounterMixSignal mix;
    uint8_t reserved[PERCOUNTER_CONTEXT_SSBUF_OFFSET - PERCOUNTER_SSBUF_REGION_OFFSET - sizeof(PercounterMixSignal)];
    PercounterInvocationContextSlot contexts[PLATFORM_CORES_PER_BLOCKDIM];
};
static_assert(sizeof(PercounterSsbufRegion) <= PERCOUNTER_SSBUF_REGION_SIZE);
static_assert(
    offsetof(PercounterSsbufRegion, contexts) + PERCOUNTER_SSBUF_REGION_OFFSET == PERCOUNTER_CONTEXT_SSBUF_OFFSET
);

inline __aicore__ PERCOUNTER_SSBUF PercounterSsbufRegion *percounter_ssbuf_region(uint64_t hardware_base) {
    return reinterpret_cast<PERCOUNTER_SSBUF PercounterSsbufRegion *>(hardware_base + PERCOUNTER_SSBUF_REGION_OFFSET);
}

// Each defer head owns its own GM atomic sector.
struct alignas(128) PercounterCounter {
    uint64_t completion_notify_address;
    uint64_t calendar_defer_head;
    uint32_t completion_notify_target;
    uint32_t calendar_defer_bit;
    uint64_t function_addresses[PLATFORM_CORES_PER_BLOCKDIM];
    uint64_t arguments_address;
    uint64_t predicate_address;
};
static_assert(sizeof(PercounterCounter) == 128);

inline constexpr uint32_t PERCOUNTER_ASYNC_CAPACITY = 4;
static_assert(PERCOUNTER_ASYNC_CAPACITY <= MAX_COMPLETIONS_PER_TASK);

// One physical invocation can append at most one completion record.
struct alignas(128) PercounterAsyncCompletion {
    uint64_t calendar_defer_owner;
    uint64_t next;
    uint64_t pending_mask;
    uint64_t completion_add_cmd;
    uint64_t calendar_sample;
    uint64_t calendar_shell_add_cmd;
    alignas(64) DeferredCompletionEntry entries[PERCOUNTER_ASYNC_CAPACITY];
};
static_assert(sizeof(PercounterAsyncCompletion) == 256);
static_assert(offsetof(PercounterAsyncCompletion, calendar_defer_owner) == 0);
static_assert(
    std::is_trivially_copyable_v<PercounterAsyncCompletion> && std::is_standard_layout_v<PercounterAsyncCompletion>
);

// The CntNotify ADD alias is eight bytes above SET.
inline __aicore__ uint64_t percounter_notify_add_address(uint64_t read_address) {
    return read_address == 0 ? 0 : read_address + 8;
}

inline bool percounter_sync_widths_fit(const uint64_t *widths, uint64_t clusters, uint64_t aivs_per_cluster) {
    return widths[0] <= clusters && widths[1] <= clusters * aivs_per_cluster && widths[2] <= clusters;
}

struct alignas(64) PercounterGraphTemplate {
    uint64_t task_count;
};

// Each invocation owns its publication cache line.
struct alignas(64) PercounterGraphInvocation {
    uint64_t template_offset;
    // The body sinks contribute to this top-level task's counter.
    uint64_t shell_task_id;
    // Percounter id of body task 0; body task i is `body_base + i`.
    uint64_t body_base;
    uint64_t calendar_ready;
    uint64_t materialize_start_cycles;
    uint64_t ready_timing_offset;
    uint64_t materialize_end_cycles;
    uint64_t materializer_thread_id;
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

struct PercounterGraphReadyTiming {
    uint64_t cycles;
    uint64_t watermark;
};

struct alignas(64) PercounterAicpuTiming {
    uint64_t ready_publish_start_cycles;
    uint64_t ready_publish_end_cycles;
    uint64_t observe_start_cycles;
    uint64_t sinks_observed_cycles;
    uint64_t observe_end_cycles;
    uint64_t shutdown_start_cycles;
    uint64_t shutdown_end_cycles;
};

// Calendar plan and DFX key: [block count:16, block index:16, task ID:32].
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

struct alignas(128) PercounterControl {
    uint64_t task_count;
    // Top-level sinks: the tasks no task in the table consumes. Every task
    // reaches one along consumer edges, so their completion covers the graph.
    uint64_t sink_count;
    uint64_t sinks_offset;
    uint64_t mix_task_count;
    uint64_t sync_cohort_count;
    // One CntNotify SET address per explicit START cohort.
    uint64_t sync_notify_offset;
    uint64_t max_sync_blocks[3];
    uint64_t table_task_count;
    uint64_t graph_invocation_count;
    uint64_t graph_invocations_offset;
    uint64_t graph_prepare_offset;
    uint64_t mix_signal_ready_offset;
    uint64_t counters_offset;
    uint64_t cluster_count;
    uint64_t aiv_per_cluster;
    uint64_t lane_error;
    uint64_t graph_storage_address;
    uint64_t graph_storage_count;
    uint64_t callable_addresses_address;
    uint64_t callable_addresses_count;
    uint64_t calendar_offset;
    uint64_t image_bytes;
    // One deferred-completion registration scratch per physical lane. The condition
    // cap is one record's, and the epoch is what tells a record of this run
    // from one the retained storage still held.
    uint64_t async_groups_offset;
    uint64_t async_entries_offset;
    uint64_t async_run_epoch;
    uint64_t async_completions_offset;
    uint64_t async_completion_capacity;
    uint64_t scheduler_timeout_cycles;
    uint64_t profiling_level;
    uint64_t profile_buffers_offset;
    uint64_t lane_error_claim;
    uint64_t lane_error_task;
    uint64_t lane_error_detail;

    // Allocation precedes the corresponding block's completed publication.
    // Both counters are monotone; equality after all blocks ran permits exit.
    alignas(128) uint64_t async_registered;
    uint64_t async_settled;
    uint64_t aicpu_profile_offset;
};
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
