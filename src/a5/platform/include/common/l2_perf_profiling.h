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
 * @file l2_perf_profiling.h
 * @brief Performance profiling data structures
 *
 * Architecture: per-core L2PerfBuffer + per-thread PhaseBuffer + a single
 * L2PerfSetupHeader, all allocated on device by Host.
 *
 * Layout (count-first + flexible array):
 *   L2PerfBuffer  = 64B header (count + padding) + records[capacity]
 *   PhaseBuffer = 64B header (count + padding) + records[capacity]
 *
 * Buffer device pointers + total_tasks + AicpuPhaseHeader (with orch_summary
 * and core_to_thread) are consolidated in L2PerfSetupHeader. Host copies
 * L2PerfSetupHeader to device once before execution; AICPU reads pointers in
 * its profiling init. After execution, Host copies L2PerfSetupHeader back, then
 * does a two-step memcpy (header → count → records) on each L2PerfBuffer /
 * PhaseBuffer to reduce transfer to "actual records" instead of full
 * capacity.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_
#define SRC_A5_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_

#include <cstdint>
#include <vector>

#include "common/core_type.h"
#include "common/platform_config.h"

// Maximum number of successor tasks per L2PerfRecord (matches Task::fanout)
#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 128
#endif

// =============================================================================
// L2PerfRecord - Single Task Execution Record
// =============================================================================

/**
 * Single task execution record
 */
struct L2PerfRecord {
    // Timing information (device clock timestamps)
    uint64_t start_time;  // Task start timestamp (get_sys_cnt)
    uint64_t end_time;    // Task end timestamp
    uint64_t duration;    // Execution duration (end - start)

    // AICPU-side timestamps (written by AICPU, not AICore)
    uint64_t dispatch_time;  // AICPU timestamp: when task was dispatched to AICore
    uint64_t finish_time;    // AICPU timestamp: when AICPU observed task completion

    // AICore writes the register dispatch token (low 32 bits only) zero-extended into task_id.
    // For multi-ring runtimes (tensormap_and_ringbuffer, aicpu_build_graph), AICPU overwrites
    // with the full PTO2 encoding (ring_id << 32) | local_id after FIN/perf row match.
    // For host_build_graph, task_id stays as the plain integer task index (ring_id = 0).
    uint64_t task_id;
    uint32_t func_id;    // Kernel function identifier
    CoreType core_type;  // Core type (AIC/AIV)

    // Dependency relationship (fanout only)
    uint64_t fanout[RUNTIME_MAX_FANOUT];  // Successor task task_id array
    int32_t fanout_count;                 // Number of successor tasks
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfRecord) % 64 == 0, "L2PerfRecord must be 64-byte aligned for optimal cache performance");

// =============================================================================
// L2PerfBuffer - Record Buffer with Count-First Layout and WIP Staging Slots
// =============================================================================

/**
 * Performance record buffer (count-first + flexible array)
 *
 * Layout: 64B header (count + padding), then wip[2] staging slots,
 * then records[].
 * Actual allocation: sizeof(L2PerfBuffer) + capacity * sizeof(L2PerfRecord).
 *
 * Count-first enables two-step collection: Host copies sizeof(L2PerfBuffer) to
 * read count, then copies only count * sizeof(L2PerfRecord) of actual data.
 *
 * WIP protocol: AICore writes timing to wip[reg_task_id & 1], AICPU copies
 * it into records[count] at completion. Dual-slot parity ensures no overlap.
 */
struct L2PerfBuffer {
    volatile uint32_t count;  // Current committed record count (at offset 0 for cache-line read)
    uint32_t pad[15];         // Pad to 64B cache line boundary
    L2PerfRecord wip[2];      // AICore WIP staging slots (index = reg_task_id & 1)
    L2PerfRecord records[];   // Flexible array member (not counted in sizeof)
} __attribute__((aligned(64)));

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================

/**
 * AICPU phase identifier
 *
 * Scheduler phases (0-3): four phases in each scheduler loop iteration.
 * Orchestrator phases (16-24): sub-steps within each pto2_submit_task() call.
 */
enum class AicpuPhaseId : uint32_t {
    // Scheduler phases (0-3)
    SCHED_COMPLETE = 0,     // Process completed tasks (fanout traversal)
    SCHED_DISPATCH = 1,     // Dispatch ready tasks to idle cores
    SCHED_SCAN = 2,         // Incremental scan for root tasks
    SCHED_IDLE_WAIT = 3,    // Idle/spinning (no progress)
    SCHED_PHASE_COUNT = 4,  // Sentinel: number of scheduler phases
    // Orchestrator phases (16-24)
    ORCH_SYNC = 16,      // tensormap sync
    ORCH_ALLOC = 17,     // task_ring_alloc
    ORCH_PARAMS = 18,    // param copy
    ORCH_LOOKUP = 19,    // tensormap lookup + dep
    ORCH_HEAP = 20,      // heap alloc
    ORCH_INSERT = 21,    // tensormap insert
    ORCH_FANIN = 22,     // fanin + early-ready
    ORCH_FINALIZE = 23,  // scheduler init + SM
    ORCH_SCOPE_END = 24  // scope_end
};

/**
 * Single AICPU scheduler phase record (32 bytes)
 *
 * Records one phase within one loop iteration of a scheduler thread.
 * No thread_id field: identity is derived from array index (position = identity).
 */
struct AicpuPhaseRecord {
    uint64_t start_time;    // Phase start timestamp
    uint64_t end_time;      // Phase end timestamp
    uint32_t loop_iter;     // Loop iteration number
    AicpuPhaseId phase_id;  // Phase type
    union {
        uint64_t task_id;          // Multi-ring runtimes (tensormap_and_ringbuffer, aicpu_build_graph):
                                   // full PTO2 encoding (ring_id << 32) | local_id for cross-view correlation.
        uint64_t tasks_processed;  // Scheduler phases: number of tasks processed in this batch
    };
};

/**
 * AICPU orchestrator cumulative summary
 *
 * Contains accumulated cycle counts from the orchestrator thread.
 * Written once after orchestration completes.
 */
struct AicpuOrchSummary {
    uint64_t start_time;       // Orchestrator start timestamp
    uint64_t end_time;         // Orchestrator end timestamp
    uint64_t sync_cycle;       // sync_tensormap phase
    uint64_t alloc_cycle;      // task_ring_alloc phase
    uint64_t args_cycle;       // param_copy phase
    uint64_t lookup_cycle;     // lookup+dep phase
    uint64_t heap_cycle;       // heap_alloc phase
    uint64_t insert_cycle;     // tensormap_insert phase
    uint64_t fanin_cycle;      // fanin+ready phase
    uint64_t scope_end_cycle;  // scope_end phase
    int64_t submit_count;      // Total tasks submitted
    uint32_t magic;            // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t padding;          // Alignment padding
} __attribute__((aligned(64)));

constexpr uint32_t AICPU_PHASE_MAGIC = 0x41435048;  // "ACPH"

/**
 * Phase record buffer (count-first + flexible array)
 *
 * Capacity: PLATFORM_PHASE_RECORDS_PER_THREAD
 * Actual allocation: 64 + capacity * sizeof(AicpuPhaseRecord).
 */
struct PhaseBuffer {
    volatile uint32_t count;     // Current record count (at offset 0 for cache-line read)
    uint32_t pad[15];            // Pad to 64B cache line boundary
    AicpuPhaseRecord records[];  // Flexible array member
} __attribute__((aligned(64)));

/**
 * AICPU phase profiling header
 *
 * Embedded in L2PerfSetupHeader. Contains the magic, per-thread metadata, the
 * core_id → scheduler thread mapping, and the orchestrator's cumulative
 * cycle summary.
 */
struct AicpuPhaseHeader {
    uint32_t magic;                             // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t num_sched_threads;                 // Number of scheduler threads
    uint32_t records_per_thread;                // Max records per PhaseBuffer
    uint32_t num_cores;                         // Total number of cores with valid assignments
    int8_t core_to_thread[PLATFORM_MAX_CORES];  // core_id → scheduler thread index (-1 = unassigned)
    AicpuOrchSummary orch_summary;              // Orchestrator cumulative data
} __attribute__((aligned(64)));

// =============================================================================
// L2PerfSetupHeader - Host/Device Metadata Exchange
// =============================================================================

/**
 * Performance setup header
 *
 * Allocated on device by Host. Host writes buffer pointers and metadata,
 * then copies this struct to device once before execution. AICPU reads
 * buffer pointers during init. After execution completes, Host copies
 * this struct back to read total_tasks and phase_header.
 *
 * Memory layout: kernel_args.l2_perf_data_base points to this struct on
 * device (read by AICPU via get_platform_l2_perf_base()).
 */
struct L2PerfSetupHeader {
    // Host writes, AICPU reads (init)
    uint32_t num_cores;          // Number of AICore instances
    uint32_t num_phase_threads;  // Number of phase profiling threads
    uint32_t pad0[14];           // Pad to 64B

    // Host writes, AICPU reads: per-core / per-thread buffer device pointers
    uint64_t core_buffer_ptrs[PLATFORM_MAX_CORES];           // L2PerfBuffer* on device
    uint64_t phase_buffer_ptrs[PLATFORM_MAX_AICPU_THREADS];  // PhaseBuffer* on device

    // AICPU writes, host reads back
    volatile uint32_t total_tasks;  // Updated by orchestrator
    uint32_t pad1[15];              // Pad to 64B

    // AICPU writes, host reads back (via collect_all)
    AicpuPhaseHeader phase_header;  // Contains orch_summary and core_to_thread
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get L2PerfSetupHeader pointer from base address
 *
 * @param base_ptr Device base address (kernel_args.l2_perf_data_base)
 * @return L2PerfSetupHeader pointer
 */
inline L2PerfSetupHeader *get_perf_setup_header(void *base_ptr) {
    return reinterpret_cast<L2PerfSetupHeader *>(base_ptr);
}

/**
 * Calculate L2PerfSetupHeader allocation size
 */
inline size_t calc_l2_perf_setup_size() { return sizeof(L2PerfSetupHeader); }

/**
 * Calculate total bytes for a L2PerfBuffer with the given capacity
 *
 * @param capacity Number of L2PerfRecord slots (e.g. PLATFORM_PROF_BUFFER_SIZE)
 * @return 64B header + capacity * sizeof(L2PerfRecord)
 */
inline size_t calc_l2_perf_buffer_size(int capacity) {
    return sizeof(L2PerfBuffer) + static_cast<size_t>(capacity) * sizeof(L2PerfRecord);
}

/**
 * Calculate total bytes for a PhaseBuffer with the given capacity
 *
 * @param capacity Number of AicpuPhaseRecord slots (e.g. PLATFORM_PHASE_RECORDS_PER_THREAD)
 * @return 64B header + capacity * sizeof(AicpuPhaseRecord)
 */
inline size_t calc_phase_buffer_size(int capacity) {
    return sizeof(PhaseBuffer) + static_cast<size_t>(capacity) * sizeof(AicpuPhaseRecord);
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_
