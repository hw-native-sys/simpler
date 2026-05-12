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
 * @file dep_gen.h
 * @brief dep_gen (SubmitTrace) shared-memory data structures
 *
 * Captures the inputs to every Orchestrator::submit_task call into a streaming
 * ring of DepGenRecord. The host side replays these records offline to
 * reconstruct the full task dependency graph (deps.json), bypassing the race
 * window in L2PerfRecord::fanout[] (where an early-finishing producer would
 * have its record sealed before later-submitted consumers can register).
 *
 * Streaming buffer design mirrors PMU / L2Perf / TensorDump (single source of
 * algorithmic truth in src/a2a3/platform/include/host/profiling_common/profiler_base.h):
 *
 *   DepGenFreeQueue    — SPSC: Host pushes free DepGenBuffers, AICPU pops them.
 *   DepGenBufferState  — Per-instance state: free_queue + current buffer ptr.
 *   DepGenDataHeader   — Fixed shared-mem header: per-thread ready queues.
 *   DepGenBuffer       — Fixed-capacity record buffer.
 *
 * Single-instance: the orchestrator is one AICPU thread, so the BufferState
 * array has length 1. Kept array-shaped (vs scalar) for symmetry with PMU /
 * L2Perf and to match ProfilerBase<DepGenModule>::for_each_instance.
 *
 * Tensor data is captured as opaque 128-byte blobs (`DEP_GEN_TENSOR_SIZE`)
 * matching the runtime Tensor struct size. The AICPU writer
 * (dep_gen_collector_aicpu.cpp) static_asserts sizeof(Tensor) == 128 against
 * the runtime headers it imports; the platform shared-memory header stays
 * runtime-agnostic.
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_DEP_GEN_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_DEP_GEN_H_

#include <cstddef>
#include <cstdint>

#include "arg_direction.h"  // CORE_MAX_TENSOR_ARGS
#include "common/platform_config.h"

// =============================================================================
// dep_gen-local capacity constants
// =============================================================================

/**
 * Bytes per captured Tensor slot — matches runtime sizeof(Tensor). Verified
 * in dep_gen_collector_aicpu.cpp via static_assert against the runtime header.
 * Two cache lines: cache line 1 (lookup hot path) + cache line 2 (offsets).
 */
constexpr int DEP_GEN_TENSOR_SIZE = 128;

/**
 * Max explicit_dep entries captured per submit by dep_gen replay. This is a
 * diagnostic-side cap only — the runtime's Arg::set_dependencies has no hard
 * limit on dep count. Submits exceeding this cap are logged and truncated by
 * dep_gen_aicpu_record_submit(); runtime correctness is unaffected.
 */
constexpr int DEP_GEN_MAX_EXPLICIT_DEPS = 16;

// =============================================================================
// DepGenRecord — one captured submit_task call
// =============================================================================

/**
 * Bitmask flags for DepGenRecord.flags.
 */
enum DepGenRecordFlags : uint32_t {
    DEP_GEN_FLAG_IN_MANUAL_SCOPE = 1u << 0,  // submit happened inside a manual scope
};

/**
 * Per-submit_task capture. Replay reads these to reconstruct the dep graph.
 *
 * Layout:
 *   - task_id, flags, counts, explicit_deps, arg_types in the first cache lines
 *   - tensors[] (16 × 128 B opaque blobs) at the tail; covers ~64% of the entry
 *
 * Total size: 8 + 4 + 4 + 16*8 + 16 + 32 (pad) + 16*128 = 2240 bytes.
 * Aligned to 64 B → 2240 B (already a multiple of 64). The 32-byte _pad0
 * pushes the tensors[] array to offset 192 = 3 * 64 so each 128-byte tensor
 * blob covers exactly two cache lines instead of straddling three.
 */
struct DepGenRecord {
    uint64_t task_id;                                            // PTO2 encoding (ring_id << 32) | local_id
    uint32_t flags;                                              // DepGenRecordFlags bitmask
    uint16_t tensor_count;                                       // number of valid Tensor slots
    uint16_t explicit_dep_count;                                 // number of valid explicit_dep slots
    uint64_t explicit_deps[DEP_GEN_MAX_EXPLICIT_DEPS];           // PTO2TaskId::raw, length = explicit_dep_count
    uint8_t arg_types[CORE_MAX_TENSOR_ARGS];                     // TensorArgType, length = tensor_count
    uint8_t _pad0[32];                                           // align tensors[] to 64 B (offset 192)
    uint8_t tensors[CORE_MAX_TENSOR_ARGS][DEP_GEN_TENSOR_SIZE];  // opaque Tensor blobs
} __attribute__((aligned(64)));

static_assert(sizeof(DepGenRecord) % 64 == 0, "DepGenRecord must be cache-line aligned");
static_assert(offsetof(DepGenRecord, tensors) % 64 == 0, "DepGenRecord::tensors[] must start on a cache-line boundary");

// =============================================================================
// DepGenBuffer — fixed-capacity record container
// =============================================================================

/**
 * Fixed-capacity DepGenRecord buffer.
 * Allocated by Host, pushed into the orchestrator instance's free_queue.
 *
 * AICPU writer is the orchestrator thread (single producer); it commits
 * directly into records[count] without a dual_issue staging slot (PMU's
 * staging slots exist because AICore reads MMIO and writes them, not us).
 */
struct DepGenBuffer {
    // Header (first 64 bytes) — host copies this alone first to learn count.
    volatile uint32_t count;  // Number of valid records committed
    uint32_t _pad0[15];       // Pad count to 64 B; isolates count's cache line.

    // Records (flexible-size, up to PLATFORM_DEP_GEN_RECORDS_PER_BUFFER)
    DepGenRecord records[PLATFORM_DEP_GEN_RECORDS_PER_BUFFER];
} __attribute__((aligned(64)));

static_assert(offsetof(DepGenBuffer, records) == 64, "DepGenBuffer header must be exactly 64 bytes");

// =============================================================================
// SPSC free queue
// =============================================================================

/**
 * SPSC lock-free queue for free DepGenBuffer management.
 *
 * Producer: Host (DepGenCollector mgmt thread) pushes recycled/new buffers.
 * Consumer: Device (AICPU orchestrator thread) pops buffers when switching.
 */
struct DepGenFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_DEP_GEN_SLOT_COUNT];
    volatile uint32_t head;  // Consumer read position (Device increments)
    volatile uint32_t tail;  // Producer write position (Host increments)
    uint32_t _pad[22];       // Pad to 128 B
} __attribute__((aligned(64)));

static_assert(sizeof(DepGenFreeQueue) == 128, "DepGenFreeQueue must be 128 bytes");

// =============================================================================
// Per-instance buffer state
// =============================================================================

/**
 * Per-instance state for dep_gen.
 *
 * Writers:
 *   free_queue.tail:        Host writes (pushes new/recycled buffers)
 *   free_queue.head:        Device writes (pops buffers)
 *   current_buf_ptr:        Device writes (after pop), Host reads
 *   current_buf_seq:        Device writes (monotonic counter)
 *   dropped_record_count:   Device writes — submits dropped because free_queue
 *                           was empty / ready_queue was full / no active buf
 *   total_record_count:     Device writes — monotonic count of every submit
 *                           the orchestrator attempted to record (success +
 *                           dropped)
 *
 * Host reads dropped / total at finalize to cross-check:
 *   collected_on_host + sum(dropped) == sum(total)
 */
struct DepGenBufferState {
    DepGenFreeQueue free_queue;
    volatile uint64_t current_buf_ptr;
    volatile uint32_t current_buf_seq;
    volatile uint32_t dropped_record_count;
    volatile uint32_t total_record_count;
    uint32_t _pad[11];
} __attribute__((aligned(64)));

static_assert(sizeof(DepGenBufferState) == 192, "DepGenBufferState must be 192 bytes");

// =============================================================================
// Ready queue entry
// =============================================================================

/**
 * Ready queue entry — when a DepGenBuffer fills, AICPU pushes one of these
 * onto the per-thread ready queue for host pickup.
 */
struct DepGenReadyQueueEntry {
    uint32_t instance_index;  // Always 0 for dep_gen (single instance)
    uint32_t _pad0;
    uint64_t buffer_ptr;  // Device pointer to the full DepGenBuffer
    uint32_t buffer_seq;
    uint32_t _pad1;
} __attribute__((aligned(32)));

static_assert(sizeof(DepGenReadyQueueEntry) == 32, "DepGenReadyQueueEntry must be 32 bytes");

// =============================================================================
// Top-level shared-memory header
// =============================================================================

/**
 * dep_gen data fixed header. Located at the start of dep_gen shared memory.
 *
 * Per-thread ready queues (one per AICPU scheduling thread):
 *   Producer: AICPU thread (adds full DepGenBuffers)
 *   Consumer: Host DepGenCollector mgmt thread
 *
 * Even though dep_gen is single-instance, ready queues are per-thread to
 * match the ProfilerBase contract (which polls header->queues[q] for q in
 * [0, num_threads)). The orchestrator currently writes into queue[0].
 */
struct DepGenDataHeader {
    DepGenReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_DEP_GEN_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)
    uint32_t num_instances;                                     // Always 1 for now
    uint32_t _pad[3];
} __attribute__((aligned(64)));

// =============================================================================
// Memory layout helpers
// =============================================================================

/**
 * Total bytes for the dep_gen shared-mem region (header + buffer states).
 * Actual DepGenBuffers are dynamically allocated and tracked by the host.
 */
inline size_t calc_dep_gen_shm_size(int num_instances) {
    return sizeof(DepGenDataHeader) + static_cast<size_t>(num_instances) * sizeof(DepGenBufferState);
}

inline DepGenDataHeader *get_dep_gen_header(void *base_ptr) { return reinterpret_cast<DepGenDataHeader *>(base_ptr); }

inline DepGenBufferState *get_dep_gen_buffer_state(void *base_ptr, int instance_index) {
    return reinterpret_cast<DepGenBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(DepGenDataHeader)) +
           instance_index;
}

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_DEP_GEN_H_
