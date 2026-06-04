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
 * @file scope_stats.h
 * @brief scope_stats streaming shared-memory data structures.
 *
 * Two ScopeStatsRecords are produced per scope — one at scope_begin and one at
 * scope_end — each carrying the task/heap ring start/end and the tensormap
 * live-entry count sampled at that boundary, tagged with a phase flag. Records
 * stream off the device in
 * fixed-capacity buffers, mirroring PMU / dep_gen / tensor_dump / l2_swimlane (the
 * single source of mgmt-loop truth is
 * src/common/platform/include/host/profiler_base.h):
 *
 *   ScopeStatsFreeQueue   — SPSC: Host pushes free buffers, AICPU pops them.
 *   ScopeStatsBufferState — Per-instance state: free_queue + current buffer ptr
 *                           + drop/total counters.
 *   ScopeStatsDataHeader  — Fixed header: per-thread ready queues + static
 *                           capacity metadata + fatal latch.
 *   ScopeStatsBuffer      — Fixed-capacity ScopeStatsRecord buffer.
 *
 * Single-instance: the orchestrator is one AICPU thread, so the BufferState
 * array has length 1 — kept array-shaped for symmetry with the other
 * collectors and to match ProfilerBase<ScopeStatsModule>::for_each_instance.
 *
 * Concurrency contract (SPSC, both directions single-side on each end — the
 * orchestrator thread is the only AICPU writer):
 *   free_queue  — Host producer  / AICPU consumer.
 *   ready_queue — AICPU producer / Host consumer.
 * The host collector thread DOES read buffers concurrently with AICPU writing
 * other buffers; ownership of any single buffer is handed off atomically via
 * the queues.
 */

#ifndef PLATFORM_COMMON_SCOPE_STATS_H_
#define PLATFORM_COMMON_SCOPE_STATS_H_

#include <cstddef>
#include <cstdint>

#include "common/platform_config.h"

#define PTO2_SCOPE_STATS_MAX_RING_DEPTH 4
#define PTO2_SCOPE_STATS_MAX_SCOPE_DEPTH 64

// Max distinct PTO2_SCOPE line numbers the host-side site filter can hold.
#define PTO2_SCOPE_STATS_MAX_SITE_FILTERS 16

#ifdef __cplusplus
extern "C" {
#endif

// Phase tag: which sampling point a record was taken at. BEGIN/END are the
// per-scope boundaries; TASK is a per-task sample taken at submit_task when
// scope_stats_task is enabled (carries the submitted task_id).
#define SCOPE_STATS_PHASE_BEGIN 0
#define SCOPE_STATS_PHASE_END 1
#define SCOPE_STATS_PHASE_TASK 2

// One record per scope boundary (begin or end). Layout MUST stay in sync with
// the device-side writer in platform/shared/aicpu/scope_stats_collector_aicpu.cpp
// and the host reader in platform/shared/host/scope_stats_collector.cpp.
//
// Each boundary writes one record into host-visible (uncached) device memory,
// so record width is directly on the orchestrator hot path. A scope only ever
// touches its own ring (ring_id = min(scope_depth, PTO2_MAX_RING_DEPTH-1)), so
// a single ring's start/end is stored rather than a per-ring array.
//
// For both rings, end-start is the live span: task_end-task_start is in-flight
// tasks, heap_end-heap_start is bytes in use. tensormap is a free-list pool
// (not a ring), so its current live-entry count is stored directly.
//
// Field order keeps the uint64 heap pointers 8-byte aligned with no internal
// padding.
struct ScopeStatsRecord {
    char site_file_basename[32];  // NUL-terminated basename of the PTO2_SCOPE site,
                                  // captured at append time so the host JSON has a
                                  // human-readable path without dereferencing a
                                  // device pointer (the string table lives in the
                                  // orchestration .so, not in shared memory).
    uint64_t heap_start;          // Heap ring reclaim pointer (heap_tail_).
    uint64_t heap_end;            // Heap ring allocation pointer (heap_top_).
    uint64_t task_id;             // PTO2TaskId.raw for PHASE_TASK records; 0 for begin/end.
    int32_t site_line;
    int32_t task_start;      // Task ring tail (last_task_alive).
    int32_t task_end;        // Task ring head (next task id).
    int32_t tensormap_used;  // tensormap pool live-entry count (not a ring).
    int16_t depth;
    int16_t ring_id;  // Ring whose start/end this record carries.
    int16_t phase;    // SCOPE_STATS_PHASE_BEGIN / _END / _TASK.
    int16_t _pad;
};

// Fixed-capacity ScopeStatsRecord buffer. Allocated by Host, pushed into the
// orchestrator instance's free_queue. The orchestrator thread is the single
// producer; it commits directly into records[count].
struct ScopeStatsBuffer {
    // Header (first 64 bytes) — host copies this alone first to learn count.
    volatile uint32_t count;  // Number of valid records committed
    uint32_t _pad0[15];       // Pad count to 64 B; isolates count's cache line.

    ScopeStatsRecord records[PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER];
} __attribute__((aligned(64)));

static_assert(offsetof(ScopeStatsBuffer, records) == 64, "ScopeStatsBuffer header must be exactly 64 bytes");

// SPSC free queue: Host (producer) pushes recycled/new buffers, Device (AICPU
// consumer) pops them when switching the current buffer.
struct ScopeStatsFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_SCOPE_STATS_SLOT_COUNT];
    volatile uint32_t head;  // Consumer read position (Device increments)
    volatile uint32_t tail;  // Producer write position (Host increments)
    uint32_t _pad[14];
} __attribute__((aligned(64)));

// Per-instance buffer state. current_buf_ptr / current_buf_seq / the counters
// are all Device-written, Host-read (except free_queue.tail which Host writes).
struct ScopeStatsBufferState {
    ScopeStatsFreeQueue free_queue;
    volatile uint64_t current_buf_ptr;       // Device's in-progress buffer (0 = none)
    volatile uint32_t current_buf_seq;       // Device monotonic counter
    volatile uint32_t dropped_record_count;  // Device: records dropped (free_queue empty / ready_queue full)
    volatile uint32_t total_record_count;    // Device: monotonic count of every record appended (success + dropped)
    uint32_t _pad[11];
} __attribute__((aligned(64)));

// Ready queue entry — when a ScopeStatsBuffer fills, AICPU pushes one of these
// onto the per-thread ready queue for host pickup.
struct ScopeStatsReadyQueueEntry {
    uint32_t instance_index;  // Always 0 (single instance)
    uint32_t _pad0;
    uint64_t buffer_ptr;  // Device pointer to the full ScopeStatsBuffer
    uint32_t buffer_seq;
    uint32_t _pad1;
} __attribute__((aligned(32)));

// scope_stats data fixed header, located at the start of the shared region.
// Per-thread ready queues match the ProfilerBase contract (poll over
// header->queues[q] for q in [0, num_threads)); the orchestrator writes into
// queue[orch_thread_idx].
//
// Static capacity metadata + fatal latch live here (not in the streamed
// records) because they are run-constants written once by the AICPU capacity
// setters at orchestrator init and read by the host at finalize.
struct ScopeStatsDataHeader {
    ScopeStatsReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_SCOPE_STATS_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)
    uint32_t num_instances;                                     // Always 1 for now

    // Per-ring static capacities — written once by AICPU at orchestrator init
    // (scope_stats_set_ring_capacity / scope_stats_set_tensormap_capacity).
    // Host needs them to render the "used/cap" ratio without a separate query.
    int32_t task_window_cap[PTO2_SCOPE_STATS_MAX_RING_DEPTH];
    uint64_t heap_cap[PTO2_SCOPE_STATS_MAX_RING_DEPTH];
    int32_t tensormap_cap;
    volatile uint32_t fatal_latched;  // AICPU sets to 1 on first fatal.

    // Scope site filter — host-written run-constant (count == 0 = no filter,
    // the default: every scope is collected). The orchestration is a single
    // source file, so a PTO2_SCOPE site is uniquely identified by its line
    // number; the filter is just a list of line numbers. When count > 0 the
    // AICPU collector emits records only for scopes whose site_line appears in
    // site_filter_lines[0..count). Depth/site bookkeeping is unaffected so
    // nesting stays correct; filtered-out scopes are not appended and do not
    // count toward total_record_count.
    int32_t site_filter_lines[PTO2_SCOPE_STATS_MAX_SITE_FILTERS];
    int32_t site_filter_count;

    // Per-task sampling (scope_stats_task). Host-written run-constant: when
    // non-zero, the AICPU collector also emits a PHASE_TASK record at each
    // submit_task (subject to the same site filter). 0 = scope boundaries only.
    int32_t task_enabled;
    uint32_t _pad[1];
} __attribute__((aligned(64)));

// =============================================================================
// Memory layout helpers
// =============================================================================

// Total bytes for the scope_stats shared-mem region (header + buffer states).
// Actual ScopeStatsBuffers are dynamically allocated and tracked by the host.
inline size_t calc_scope_stats_shm_size(int num_instances) {
    return sizeof(ScopeStatsDataHeader) + static_cast<size_t>(num_instances) * sizeof(ScopeStatsBufferState);
}

inline ScopeStatsDataHeader *get_scope_stats_header(void *base_ptr) {
    return reinterpret_cast<ScopeStatsDataHeader *>(base_ptr);
}

inline ScopeStatsBufferState *get_scope_stats_buffer_state(void *base_ptr, int instance_index) {
    return reinterpret_cast<ScopeStatsBufferState *>(
               reinterpret_cast<char *>(base_ptr) + sizeof(ScopeStatsDataHeader)
           ) +
           instance_index;
}

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_SCOPE_STATS_H_
