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
 * Architecture: Fixed header + per-core/thread buffer states + optional phase profiling region
 *
 * Memory layout (shared memory between Host and Device):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ L2PerfDataHeader (fixed header)                               │
 * │  - ReadyQueue (FIFO, capacity=PLATFORM_PROF_READYQUEUE_SIZE)│
 * │  - Metadata (num_cores, flags)                              │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2PerfBufferState[0] (Core 0)                                 │
 * │  - free_queue: SPSC queue of available buffer pointers      │
 * │  - current_buf_ptr, current_buf_seq                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2PerfBufferState[1] (Core 1)                                 │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2PerfBufferState[num_cores-1]                                │
 * ├─────────────────────────────────────────────────────────────┤
 * │ AicpuPhaseHeader (optional, present when phase profiling)   │
 * │  - magic, num_sched_threads, records_per_thread             │
 * │  - orch_summary                                             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PhaseBufferState[thread0]                                   │
 * │  - free_queue: SPSC queue of available buffer pointers      │
 * │  - current_buf_ptr, current_buf_seq                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PhaseBufferState[thread1]                                   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Actual L2PerfBuffer / PhaseBuffer are allocated dynamically by Host
 * and pushed into the per-core/thread free_queue.
 *
 * Base size = sizeof(L2PerfDataHeader) + num_cores * sizeof(L2PerfBufferState)
 * With phases = Base + sizeof(AicpuPhaseHeader) + num_threads * sizeof(PhaseBufferState)
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_

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
    // For tensormap_and_ringbuffer, AICPU overwrites with the full PTO2 encoding
    // (ring_id << 32) | local_id after FIN/perf row match.
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
// L2PerfAicoreRing - Stable AICore→AICPU Staging Ring (per core, never rotated)
// =============================================================================

/**
 * Per-core staging ring written exclusively by AICore.
 *
 * AICore stores each task's timing in `dual_issue_slots[reg_task_id %
 * PLATFORM_L2_AICORE_RING_SIZE]` and never touches any other L2Perf memory.
 * The ring is allocated once by the host, addressed through
 * `L2PerfBufferState[block_idx].aicore_ring_ptr` (also published into the
 * `KernelArgs::aicore_ring_addr` table the AICore kernel entry forwards
 * into `set_aicore_l2_perf_ring()`), and lives for the entire run — its
 * address is never reassigned, decoupling AICore writes from the AICPU's
 * records-buffer rotation.
 *
 * Sizing: PLATFORM_L2_AICORE_RING_SIZE must be ≥ in-flight issue depth on a
 * single core (see runtime "completion-before-dispatch" invariant). The
 * default 2 covers today's dual-issue dispatch.
 */
struct L2PerfAicoreRing {
    L2PerfRecord dual_issue_slots[PLATFORM_L2_AICORE_RING_SIZE];
} __attribute__((aligned(64)));

// =============================================================================
// L2PerfBuffer - Fixed-Size Record Buffer (AICPU-only)
// =============================================================================

/**
 * Fixed-size performance record buffer
 *
 * Capacity: PLATFORM_PROF_BUFFER_SIZE (defined in platform_config.h)
 * Allocated dynamically by Host, pushed into per-core free_queue, rotated
 * by AICPU when full.
 *
 * Owned and written exclusively by AICPU: AICore never touches this memory.
 * AICPU reads timing from L2PerfAicoreRing::dual_issue_slots, fills in the
 * AICPU-side fields, then commits into records[count++].
 */
struct L2PerfBuffer {
    L2PerfRecord records[PLATFORM_PROF_BUFFER_SIZE];  // Committed records (AICPU writes)
    volatile uint32_t count;                          // Current committed record count
} __attribute__((aligned(64)));

// =============================================================================
// L2PerfFreeQueue - SPSC Lock-Free Queue for Free Buffers
// =============================================================================

/**
 * Single Producer Single Consumer (SPSC) lock-free queue for free buffer management
 *
 * Producer: Host (ProfMemoryManager thread) pushes newly allocated buffers
 * Consumer: Device (AICPU thread) pops buffers when switching
 *
 * Queue semantics:
 * - Empty: head == tail
 * - Full: (tail - head) >= PLATFORM_PROF_SLOT_COUNT
 * - Capacity: PLATFORM_PROF_SLOT_COUNT buffers
 *
 * Memory ordering:
 * - Device pop: rmb() → read tail → read buffer_ptrs[head % COUNT] → rmb() → write head → wmb()
 * - Host push: write buffer_ptrs[tail % COUNT] → wmb() → write tail → wmb()
 */
struct L2PerfFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PROF_SLOT_COUNT];  // Free buffer addresses
    volatile uint32_t head;                                   // Consumer read position (Device increments)
    volatile uint32_t tail;                                   // Producer write position (Host increments)
    uint32_t pad[13];                                         // Pad to 128 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfFreeQueue) == 128, "L2PerfFreeQueue must be 128 bytes for cache alignment");

// =============================================================================
// L2PerfBufferState - Per-Core/Thread Buffer State (Unified for L2PerfRecord and Phase)
// =============================================================================

/**
 * Per-core or per-thread buffer state for dynamic profiling
 *
 * Contains:
 * - free_queue: SPSC queue of available buffer addresses
 * - current_buf_ptr: Currently active buffer being written (0 = no active buffer)
 * - current_buf_seq: Monotonic sequence number for ordering
 * - aicore_ring_ptr: Stable per-core L2PerfAicoreRing address (L2PerfRecord
 *   profiling only; unused by Phase profiling). Set by host at init, read by
 *   AICPU in `l2_perf_aicpu_complete_record` to read the AICore-published
 *   timing slots. Never reassigned during the run.
 * - total_record_count / dropped_record_count / mismatch_record_count:
 *   per-core/-thread tallies AICPU keeps so the host can cross-check
 *   `collected + dropped + mismatch == device_total` at end-of-run. Replaces
 *   the previous L2PerfDataHeader::total_tasks signal — the host no longer
 *   needs to know task count up front. `mismatch_record_count` accounts for
 *   ring slot/task_id invariant violations (a hard error class, distinct
 *   from capacity drops).
 *
 * Used in two contexts:
 * - Per-core L2PerfRecord profiling (current_buf_ptr → L2PerfBuffer,
 *   aicore_ring_ptr → L2PerfAicoreRing)
 * - Per-thread Phase profiling (current_buf_ptr → PhaseBuffer,
 *   aicore_ring_ptr / mismatch_record_count unused)
 *
 * Writers:
 * - free_queue.tail: Host writes (pushes new buffers)
 * - free_queue.head: Device writes (pops buffers)
 * - current_buf_ptr: Device writes (after pop), Host reads (for flush/collect)
 * - current_buf_seq: Device writes (monotonic counter)
 * - aicore_ring_ptr: Host writes once at init, AICPU reads
 * - total_record_count / dropped_record_count / mismatch_record_count:
 *   Device writes, Host reads at drain time (no concurrency on a per-state
 *   basis since each state belongs to a single core/thread)
 */
struct L2PerfBufferState {
    L2PerfFreeQueue free_queue;               // SPSC queue of free buffer addresses
    volatile uint64_t current_buf_ptr;        // Current active buffer (0 = none)
    volatile uint64_t aicore_ring_ptr;        // Stable AICore staging ring (L2Perf only; 0 for Phase)
    volatile uint32_t current_buf_seq;        // Sequence number for ordering
    volatile uint32_t total_record_count;     // Records the AICPU attempted to write to this state
    volatile uint32_t dropped_record_count;   // Records dropped (queue full / overwrite / no buffer)
    volatile uint32_t mismatch_record_count;  // Records lost to ring/task_id invariant violation (hard errors)
    uint32_t pad[8];                          // Pad to 192 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfBufferState) == 192, "L2PerfBufferState must be 192 bytes for cache alignment");

// Type alias for semantic clarity in Phase profiling context
using PhaseBufferState = L2PerfBufferState;  // Per-thread Phase profiling

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, AICPU adds this entry to the queue.
 * Host memory manager retrieves entries from the queue.
 *
 * Entry types (distinguished by is_phase flag):
 * - L2PerfRecord entry: core_index = core ID, is_phase = 0
 * - Phase entry:      core_index = thread_idx, is_phase = 1
 */
struct ReadyQueueEntry {
    uint32_t core_index;  // Core index (0 ~ num_cores-1), or thread_idx for phase entries
    uint32_t is_phase;    // 0 = L2PerfRecord, 1 = Phase
    uint64_t buffer_ptr;  // Device pointer to the full buffer
    uint32_t buffer_seq;  // Sequence number for ordering
    uint32_t pad;         // Alignment padding
} __attribute__((aligned(32)));

// =============================================================================
// L2PerfDataHeader - Fixed Header
// =============================================================================

/**
 * Performance data fixed header
 *
 * Located at the start of shared memory, contains:
 * 1. Per-thread ready queues (FIFO Circular Buffers)
 * 2. Metadata (core count)
 *
 * Ready queue design:
 * - Per-thread queues: Avoid lock contention between AICPU threads
 * - Capacity per queue: PLATFORM_PROF_READYQUEUE_SIZE (full capacity for each thread)
 * - Implementation: Circular Buffer
 * - Producer: AICPU thread (adds full buffers to its own queue)
 * - Consumer: Host memory manager thread (reads from all queues)
 * - Queue empty: head == tail
 * - Queue full: (tail + 1) % capacity == head
 */
struct L2PerfDataHeader {
    // Per-thread ready queues (FIFO Circular Buffers)
    // Each AICPU thread has its own queue to avoid lock contention
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PROF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Consumer read positions (Host modifies)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // Producer write positions (AICPU modifies)

    // Metadata (Host initializes, Device read-only)
    uint32_t num_cores;  // Actual number of cores launched
} __attribute__((aligned(64)));

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================

/**
 * AICPU phase identifier
 *
 * Scheduler phases (0-3): four phases in each scheduler loop iteration.
 * Orchestrator phases (16-24): sub-steps within each submit_task() call.
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
        uint64_t task_id;          // tensormap_and_ringbuffer: full PTO2 encoding
                                   // (ring_id << 32) | local_id for cross-view correlation.
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

constexpr uint32_t AICPU_PHASE_MAGIC = 0x41435048;        // "ACPH"
constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;  // ~512KB per thread

/**
 * Fixed-size phase record buffer (analogous to L2PerfBuffer)
 *
 * Capacity: PLATFORM_PHASE_RECORDS_PER_THREAD
 * Allocated dynamically by Host, pushed into per-thread free_queue.
 */
struct PhaseBuffer {
    AicpuPhaseRecord records[PLATFORM_PHASE_RECORDS_PER_THREAD];
    volatile uint32_t count;
} __attribute__((aligned(64)));

/**
 * AICPU phase profiling header
 *
 * Located after the L2PerfBufferState array in shared memory.
 * Contains metadata and per-thread tracking.
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
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate total memory size for performance data (buffer states only, no buffers)
 *
 * Formula: Total size = Fixed header + Dynamic tail
 *                     = sizeof(L2PerfDataHeader) + num_cores × sizeof(L2PerfBufferState)
 *
 * @param num_cores Number of cores (block_dim × PLATFORM_CORES_PER_BLOCKDIM)
 * @return Total bytes for header + buffer states
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(L2PerfDataHeader) + num_cores * sizeof(L2PerfBufferState);
}

/**
 * Get header pointer
 *
 * @param base_ptr Shared memory base address (device_ptr or host_ptr)
 * @return L2PerfDataHeader pointer
 */
inline L2PerfDataHeader *get_l2_perf_header(void *base_ptr) { return reinterpret_cast<L2PerfDataHeader *>(base_ptr); }

/**
 * Get L2PerfBufferState array start address
 *
 * @param base_ptr Shared memory base address
 * @return L2PerfBufferState array pointer
 */
inline L2PerfBufferState *get_perf_buffer_states(void *base_ptr) {
    return reinterpret_cast<L2PerfBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(L2PerfDataHeader));
}

/**
 * Get L2PerfBufferState for specified core
 *
 * @param base_ptr Shared memory base address
 * @param core_index Core index (0 ~ num_cores-1)
 * @return L2PerfBufferState pointer
 */
inline L2PerfBufferState *get_perf_buffer_state(void *base_ptr, int core_index) {
    return &get_perf_buffer_states(base_ptr)[core_index];
}

/**
 * Calculate total memory size including phase profiling region (buffer states only)
 *
 * @param num_cores Number of AICore instances
 * @param num_sched_threads Number of phase profiling threads (scheduler + orchestrator)
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_threads) {
    return calc_perf_data_size(num_cores) + sizeof(AicpuPhaseHeader) + num_sched_threads * sizeof(PhaseBufferState);
}

/**
 * Get AicpuPhaseHeader pointer (located after L2PerfBufferState array)
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @return AicpuPhaseHeader pointer
 */
inline AicpuPhaseHeader *get_phase_header(void *base_ptr, int num_cores) {
    return reinterpret_cast<AicpuPhaseHeader *>(reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores));
}

/**
 * Get PhaseBufferState array start address (located after AicpuPhaseHeader)
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @return PhaseBufferState array pointer
 */
inline PhaseBufferState *get_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<PhaseBufferState *>(
        reinterpret_cast<char *>(get_phase_header(base_ptr, num_cores)) + sizeof(AicpuPhaseHeader)
    );
}

/**
 * Get PhaseBufferState for specified thread
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @param thread_idx Thread index
 * @return PhaseBufferState pointer
 */
inline PhaseBufferState *get_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_
