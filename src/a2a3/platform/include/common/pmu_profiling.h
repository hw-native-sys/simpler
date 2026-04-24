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
 * @file pmu_profiling.h
 * @brief DAV_2201 (a2a3) AICore Performance Monitoring Unit configuration
 *
 * PMU event ID tables (values from pypto's `pmu_common.cpp`, CANN Open
 * Software License 2.0). Register offsets live in platform_config.h and are
 * accessed via RegId / reg_index().
 *
 * Streaming buffer design (mirrors l2_perf_profiling.h):
 *   PmuFreeQueue    — SPSC queue: Host pushes free PmuBuffers, AICPU pops them.
 *   PmuBufferState  — Per-core state: current active buffer pointer + free_queue.
 *   PmuDataHeader   — Fixed shared-memory header: per-thread ready queues.
 *   PmuBuffer       — Fixed-capacity record buffer (PLATFORM_PMU_RECORDS_PER_BUFFER).
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_

#include <cstdint>
#include <cstddef>

#include "common/platform_config.h"

// DAV_2201 hardware counter count.
constexpr int PMU_COUNTER_COUNT_A2A3 = 8;

/**
 * PMU event type selector. Values match pypto's `PROF_PMU_EVENT_TYPE`.
 * Simpler exposes this through the SIMPLER_PMU_EVENT_TYPE environment variable.
 */
enum class PmuEventType : uint32_t {
    ARITHMETIC_UTILIZATION = 1,
    PIPE_UTILIZATION = 2,  // default
    MEMORY = 4,
    MEMORY_L0 = 5,
    RESOURCE_CONFLICT = 6,
    MEMORY_UB = 7,
    L2_CACHE = 8,
};

constexpr uint32_t PMU_EVENT_TYPE_DEFAULT = static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION);

/**
 * Event ID table for a single event type.
 * `event_ids[i]` programs PMU_CNTi_IDX; `counters[i]` in the L2PerfRecord is the
 * value of PMU_CNTi after the task completes.
 * `counter_names[i]` is the human-readable CSV column name for counter i.
 * Empty string ("") marks an unused slot.
 *
 * Names match pypto's `tilefwk_pmu_to_csv.py` so CSVs are comparable across projects.
 */
struct PmuEventConfig {
    uint32_t event_ids[PMU_COUNTER_COUNT_A2A3];
    const char *counter_names[PMU_COUNTER_COUNT_A2A3];
};

// DAV_2201 event tables (values from pypto pmu_common.cpp SetPmuEventTypeDAV2201).
// Counter names follow pypto's tilefwk_pmu_to_csv.py tables for cross-project consistency.
constexpr PmuEventConfig PMU_EVENTS_A2A3_ARITHMETIC = {
    {0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x0},
    {"cube_fp16_exec", "cube_int8_exec", "vec_fp32_exec", "vec_fp16_128lane_exec", "vec_fp16_64lane_exec",
     "vec_int32_exec", "vec_misc_exec", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_PIPE_UTIL = {
    {0x08, 0x0a, 0x09, 0x0b, 0x0c, 0x0d, 0x55, 0x54},
    {"vec_busy_cycles", "cube_busy_cycles", "scalar_busy_cycles", "mte1_busy_cycles", "mte2_busy_cycles",
     "mte3_busy_cycles", "icache_miss", "icache_req"},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_MEMORY = {
    {0x15, 0x16, 0x31, 0x32, 0x0f, 0x10, 0x12, 0x13},
    {"ub_read_req", "ub_write_req", "l1_read_req", "l1_write_req", "l2_read_req", "l2_write_req", "main_read_req",
     "main_write_req"},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_MEMORY_L0 = {
    {0x1b, 0x1c, 0x21, 0x22, 0x27, 0x28, 0x0, 0x0},
    {"l0a_read_req", "l0a_write_req", "l0b_read_req", "l0b_write_req", "l0c_read_req", "l0c_write_req", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_RESOURCE_CONFLICT = {
    {0x64, 0x65, 0x66, 0x0, 0x0, 0x0, 0x0, 0x0},
    {"bankgroup_stall_cycles", "bank_stall_cycles", "vec_resc_conflict_cycles", "", "", "", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_MEMORY_UB = {
    {0x3d, 0x10, 0x13, 0x3e, 0x43, 0x44, 0x37, 0x38},
    {"ub_read_bw_mte", "l2_write_bw", "main_mem_write_bw", "ub_write_bw_mte", "ub_read_bw_vector", "ub_write_bw_vector",
     "ub_read_bw_scalar", "ub_write_bw_scalar"},
};
constexpr PmuEventConfig PMU_EVENTS_A2A3_L2_CACHE = {
    {0x500, 0x502, 0x504, 0x506, 0x508, 0x50a, 0x0, 0x0},
    {"write_cache_hit", "write_cache_miss_allocate", "r0_read_cache_hit", "r0_read_cache_miss_allocate",
     "r1_read_cache_hit", "r1_read_cache_miss_allocate", "", ""},
};

/**
 * Resolve an event type to the DAV_2201 event table. Returns nullptr for
 * unknown values (caller falls back to PIPE_UTILIZATION).
 */
inline const PmuEventConfig *pmu_resolve_event_config_a2a3(uint32_t event_type) {
    switch (static_cast<PmuEventType>(event_type)) {
    case PmuEventType::ARITHMETIC_UTILIZATION:
        return &PMU_EVENTS_A2A3_ARITHMETIC;
    case PmuEventType::PIPE_UTILIZATION:
        return &PMU_EVENTS_A2A3_PIPE_UTIL;
    case PmuEventType::MEMORY:
        return &PMU_EVENTS_A2A3_MEMORY;
    case PmuEventType::MEMORY_L0:
        return &PMU_EVENTS_A2A3_MEMORY_L0;
    case PmuEventType::RESOURCE_CONFLICT:
        return &PMU_EVENTS_A2A3_RESOURCE_CONFLICT;
    case PmuEventType::MEMORY_UB:
        return &PMU_EVENTS_A2A3_MEMORY_UB;
    case PmuEventType::L2_CACHE:
        return &PMU_EVENTS_A2A3_L2_CACHE;
    }
    return nullptr;
}

// =============================================================================
// PMU Record
// =============================================================================

#include "common/core_type.h"

/**
 * Per-task PMU snapshot written by AICPU after each AICore task FIN.
 */
struct PmuRecord {
    uint64_t task_id;                               // Same encoding as L2PerfRecord.task_id
    uint32_t func_id;                               // Kernel function identifier
    CoreType core_type;                             // AIC or AIV
    uint64_t pmu_total_cycles;                      // PMU_CNT_TOTAL (64-bit combined)
    uint32_t pmu_counters[PMU_COUNTER_COUNT_A2A3];  // PMU_CNT0..CNT7
} __attribute__((aligned(64)));

// =============================================================================
// PMU Streaming Buffer Structures (mirrors l2_perf_profiling.h)
// =============================================================================

/**
 * Fixed-capacity PMU record buffer.
 * Allocated by Host, pushed into per-core free_queue.
 */
struct PmuBuffer {
    PmuRecord records[PLATFORM_PMU_RECORDS_PER_BUFFER];
    volatile uint32_t count;
} __attribute__((aligned(64)));

/**
 * SPSC lock-free queue for free PmuBuffer management.
 *
 * Producer: Host (PmuMemoryManager thread) pushes recycled/new buffers.
 * Consumer: Device (AICPU thread) pops buffers when switching.
 *
 * Memory ordering:
 *   Device pop:  rmb() → read tail → read buffer_ptrs[head % COUNT] → rmb() → write head → wmb()
 *   Host push:   write buffer_ptrs[tail % COUNT] → wmb() → write tail → wmb()
 */
struct PmuFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PMU_SLOT_COUNT];  // 4 * 8 = 32 bytes
    volatile uint32_t head;                                  // Consumer read position (Device increments)
    volatile uint32_t tail;                                  // Producer write position (Host increments)
    uint32_t pad[22];                                        // Pad 40 + 88 -> 128 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(PmuFreeQueue) == 128, "PmuFreeQueue must be 128 bytes");

/**
 * Per-core PMU buffer state.
 *
 * Writers:
 *   free_queue.tail:        Host writes (pushes new/recycled buffers)
 *   free_queue.head:        Device writes (pops buffers)
 *   current_buf_ptr:        Device writes (after pop), Host reads (for collect/drain)
 *   current_buf_seq:        Device writes (monotonic counter)
 *   dropped_record_count:   Device writes (tasks whose PmuRecord was never handed
 *                           to the host, e.g. free_queue empty, ready_queue full,
 *                           no active buffer)
 *   total_record_count:     Device writes — monotonic count of every task the
 *                           AICPU attempted to record (success + dropped)
 *
 * Host reads dropped / total at finalize time to cross-check:
 *   collected_on_host + sum(dropped) == sum(total)
 */
struct PmuBufferState {
    PmuFreeQueue free_queue;                 // SPSC queue of free PmuBuffer addresses
    volatile uint64_t current_buf_ptr;       // Current active PmuBuffer (0 = none)
    volatile uint32_t current_buf_seq;       // Sequence number for ordering
    volatile uint32_t dropped_record_count;  // Tasks whose record was dropped on device
    volatile uint32_t total_record_count;    // Total tasks the AICPU attempted to record
    uint32_t pad[11];                        // Pad to 192 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(PmuBufferState) == 192, "PmuBufferState must be 192 bytes");

/**
 * Ready queue entry.
 * When a PmuBuffer is full, AICPU adds this entry to the thread's ready queue.
 */
struct PmuReadyQueueEntry {
    uint32_t core_index;  // Core index (0 ~ num_cores-1)
    uint32_t pad0;
    uint64_t buffer_ptr;  // Device pointer to the full PmuBuffer
    uint32_t buffer_seq;  // Sequence number for ordering
    uint32_t pad1;
} __attribute__((aligned(32)));

static_assert(sizeof(PmuReadyQueueEntry) == 32, "PmuReadyQueueEntry must be 32 bytes");

/**
 * PMU data fixed header, located at the start of PMU shared memory.
 *
 * Per-thread ready queues (one per AICPU scheduling thread):
 *   Producer: AICPU thread (adds full PmuBuffers)
 *   Consumer: Host PmuMemoryManager thread
 */
struct PmuDataHeader {
    PmuReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PMU_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)
    uint32_t num_cores;
    uint32_t event_type;  // PmuEventType value, written by host at init
    uint32_t pad[2];
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions
// =============================================================================

inline size_t calc_pmu_data_size(int num_cores) {
    return sizeof(PmuDataHeader) + static_cast<size_t>(num_cores) * sizeof(PmuBufferState);
}

inline PmuDataHeader *get_pmu_header(void *base_ptr) { return reinterpret_cast<PmuDataHeader *>(base_ptr); }

inline PmuBufferState *get_pmu_buffer_state(void *base_ptr, int core_id) {
    return reinterpret_cast<PmuBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(PmuDataHeader)) + core_id;
}

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_
