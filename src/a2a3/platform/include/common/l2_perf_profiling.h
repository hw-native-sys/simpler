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
 * │  - core_to_thread mapping                                   │
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

// =============================================================================
// L2 perf_level — granularity ladder for the L2 swimlane profiler.
//
// Each level is a strict superset of the previous: higher levels add the data
// described by their name on top of all lower-level data. Naming describes
// what is NEWLY captured at that level (incremental view), so gate sites read
// naturally — e.g. `if (level >= SCHED_PHASES)` means "this section runs when
// scheduler phase records are being collected (or any higher tier)".
//
// Transported via `L2PerfDataHeader::l2_perf_level` (host → AICPU,
// shared memory) and `CallConfig::enable_l2_swimlane` (Python → C). The wire
// representation stays integer (uint32_t / int32_t) for ABI stability; this
// enum is the canonical in-code type used for comparisons.
// =============================================================================
enum class L2PerfLevel : uint32_t {
    DISABLED = 0,       // No collection at all
    AICORE_TIMING = 1,  // AICore per-task start/end timestamps + task record buffer
    AICPU_TIMING = 2,   // + AICPU dispatch/finish timestamps
    SCHED_PHASES = 3,   // + scheduler main-loop phase records (SCHED_COMPLETE/DISPATCH/IDLE_WAIT)
    ORCH_PHASES = 4,    // + orchestrator phase records
};

// =============================================================================
// L2PerfRecord - Single Task Execution Record
// =============================================================================

/**
 * Single task execution record.
 *
 * Fanout edges live in the static DAG (deps.json from dep_gen) — not in
 * this record. Keeping fanout out of the hot AICPU commit path avoids a
 * per-task ~1 KB GM store + a linked-list walk on the scheduler's
 * critical fanin tail. The host swimlane export emits empty fanout
 * fields; `swimlane_converter.py` joins deps.json at post-process time.
 */
struct L2PerfRecord {
    // Timing information (device clock timestamps)
    uint64_t start_time;  // Task start timestamp (get_sys_cnt) — host-filled at flush from AICore buffer
    uint64_t end_time;    // Task end timestamp — host-filled at flush from AICore buffer
    uint64_t duration;    // Execution duration (end - start) — host-filled at flush

    // AICPU-side timestamps (written by AICPU directly)
    uint64_t dispatch_time;  // AICPU timestamp: when task was dispatched to AICore
    uint64_t finish_time;    // AICPU timestamp: when AICPU observed task completion

    // Full PTO2 task id (host-visible identity, what swimlane export and
    // dep_gen join keys use). For tensormap_and_ringbuffer this is
    // (ring_id << 32) | local_id; for host_build_graph it is the plain
    // integer task index.
    uint64_t task_id;
    uint32_t func_id;      // Kernel function identifier
    CoreType core_type;    // Core type (AIC/AIV)
    uint32_t reg_task_id;  // Register dispatch token (monotonic per core).
                           // Used by the host as the join key against
                           // L2PerfAicoreRecord.task_id, which is what
                           // AICore writes into the slim record.
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfRecord) % 64 == 0, "L2PerfRecord must be 64-byte aligned for optimal cache performance");

// =============================================================================
// L2PerfAicoreRecord - Slim AICore-Only Record (written by AICore, read by Host)
// =============================================================================

/**
 * Slim per-task record written by AICore directly into its own per-core
 * output buffer (no staging slot, no AICPU read). AICPU never touches this
 * record. The host post-processor joins it against the AICPU-side
 * L2PerfRecord on `task_id` at flush time.
 *
 * Layout: 24B payload + 8B pad → 32B (half a cache line). Two records pack
 * into one cache line so AICore's per-task store is at most a single line
 * commit + dcci.
 */
struct L2PerfAicoreRecord {
    uint64_t start_time;  // Task start timestamp (get_sys_cnt)
    uint64_t end_time;    // Task end timestamp
    uint32_t task_id;     // Register dispatch token (low 32 bits)
    uint32_t _pad;
} __attribute__((aligned(32)));

static_assert(sizeof(L2PerfAicoreRecord) == 32, "L2PerfAicoreRecord must be 32B");

// =============================================================================
// TypedBuffer<Record, N> - Templated Fixed-Size Profiling Buffer
// =============================================================================

/**
 * Generic fixed-capacity profiling buffer: contiguous record array followed
 * by a producer-written count. Layout matches the legacy L2PerfBuffer so the
 * host allocator and the AICPU consumer can treat all concrete instances
 * uniformly.
 *
 * Concrete instantiations live below as `using` aliases.
 *   - L2PerfBuffer        — AICPU-written, rotated, ready-queue tagged is_phase=0
 *   - L2PerfAicoreBuffer  — AICore-written, NOT rotated (sized for the full
 *                           session), read by host at flush time
 */
template <typename Record, size_t N>
struct TypedBuffer {
    Record records[N];
    volatile uint32_t count;
} __attribute__((aligned(64)));

using L2PerfBuffer = TypedBuffer<L2PerfRecord, PLATFORM_PROF_BUFFER_SIZE>;

// AICore buffer is rotated like L2PerfBuffer: a small fixed capacity per
// buffer plus a per-core pool, so an arbitrarily long session never wraps.
// Per-buffer capacity is a power of two so the AICore-local
// `slot_within_buf` increment lowers to a bitwise AND for boundary checks.
constexpr int PLATFORM_AICORE_BUFFER_SIZE = 1024;
static_assert(
    (PLATFORM_AICORE_BUFFER_SIZE & (PLATFORM_AICORE_BUFFER_SIZE - 1)) == 0,
    "PLATFORM_AICORE_BUFFER_SIZE must be a power of two"
);

// PLATFORM_AICORE_BUFFERS_PER_CORE is declared in platform_config.h so the
// ready-queue capacity formula there can include the AICore pool's worst-case
// burst depth alongside the AICPU and Phase pools.

using L2PerfAicoreBuffer = TypedBuffer<L2PerfAicoreRecord, PLATFORM_AICORE_BUFFER_SIZE>;

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
 * Per-core or per-thread buffer state for dynamic profiling.
 *
 * Contains:
 * - free_queue: SPSC queue of available buffer addresses
 * - current_buf_ptr: Currently active buffer being written (0 = no active
 *   buffer). AICPU writes L2PerfRecord into it; rotated by AICPU when full.
 * - current_buf_seq: Monotonic sequence number for ordering
 * - aicore_ring_ptr: Per-core L2PerfAicoreBuffer device address (L2PerfRecord
 *   profiling only; 0 for Phase). Allocated once by the host, addressed
 *   through this field; AICore reads timing-publication target via
 *   KernelArgs::aicore_ring_addr (a flat uint64_t[num_cores] table the host
 *   builds from this field). Never reassigned during the run. AICPU does
 *   NOT touch it on the hot path post-AICore-as-producer; the host reads
 *   it at flush time and joins the slim AICore records into the L2PerfRecord
 *   stream by reg_task_id.
 * - total_record_count / dropped_record_count: per-core/-thread tallies
 *   AICPU keeps so the host can cross-check `collected + dropped ==
 *   device_total` at end-of-run.
 * - mismatch_record_count: legacy field for the pre-AICore-as-producer
 *   ring-slot mismatch class. No longer written; kept for ABI continuity.
 *
 * Used in two contexts:
 * - Per-core L2PerfRecord profiling (current_buf_ptr → L2PerfBuffer,
 *   aicore_ring_ptr → L2PerfAicoreBuffer)
 * - Per-thread Phase profiling (current_buf_ptr → PhaseBuffer,
 *   aicore_ring_ptr unused)
 *
 * Writers:
 * - free_queue.tail: Host writes (pushes new buffers)
 * - free_queue.head: Device writes (pops buffers)
 * - current_buf_ptr: Device writes (after pop), Host reads (for flush/collect)
 * - current_buf_seq: Device writes (monotonic counter)
 * - aicore_ring_ptr: Host writes once at init; AICPU never reads on the
 *   hot path; host reads at flush time to do the AICore-side join.
 * - total_record_count / dropped_record_count: Device writes, Host reads
 *   at drain time (no concurrency on a per-state basis since each state
 *   belongs to a single core/thread).
 */
struct L2PerfBufferState {
    L2PerfFreeQueue free_queue;               // SPSC queue of free buffer addresses
    volatile uint64_t current_buf_ptr;        // Current active L2PerfBuffer (0 = none)
    volatile uint64_t aicore_ring_ptr;        // Per-core L2PerfAicoreBuffer (L2Perf only; 0 for Phase)
    volatile uint32_t current_buf_seq;        // Sequence number for ordering
    volatile uint32_t total_record_count;     // Records the AICPU attempted to write to this state
    volatile uint32_t dropped_record_count;   // Records dropped (queue full / overwrite / no buffer)
    volatile uint32_t mismatch_record_count;  // Legacy: ring/task_id mismatches (no longer written, kept for ABI)
    uint32_t pad[8];                          // Pad to 192 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfBufferState) == 192, "L2PerfBufferState must be 192 bytes for cache alignment");

// Type alias for semantic clarity in Phase profiling context
using PhaseBufferState = L2PerfBufferState;  // Per-thread Phase profiling

// =============================================================================
// AicoreRotation - Per-Core AICore Buffer Rotation Channel
// =============================================================================

/**
 * Single cache-line struct AICore reads on every task to decide which
 * L2PerfAicoreBuffer to write into. AICPU updates it when rotating; AICore
 * detects the change via the generation counter and resets its local slot.
 *
 *   Writer: AICPU (host writes initial values at init)
 *   Reader: AICore (dcci's this line per task — cheap relative to baseline
 *           dcci(payload, ENTIRE_DATA_CACHE))
 *
 * Race avoidance: AICPU rotates strictly at dispatch boundaries (immediately
 * before write_reg(DATA_MAIN_BASE) for task K when K % BUFFER_SIZE == 0). The
 * runtime's completion-before-dispatch invariant guarantees all tasks < K
 * have FIN'd, so AICore has already finished writing their records into the
 * old buffer before AICPU enqueues it to ready_queue.
 */
struct AicoreRotation {
    volatile uint64_t current_buf_ptr;  // Device address of the active L2PerfAicoreBuffer
    volatile uint32_t generation;       // Bumps on each rotation; AICore compares to detect changes
    uint32_t _pad_a;
    uint32_t _pad_b[12];
} __attribute__((aligned(64)));

static_assert(sizeof(AicoreRotation) == 64, "AicoreRotation must be one cache line");

// =============================================================================
// L2PerfAicoreBufferState - Per-Core AICore Pool State
// =============================================================================

/**
 * Per-core AICore-side rotation state. Owns:
 *   - rotation: the cache line AICore polls
 *   - free_queue: SPSC queue of recycled L2PerfAicoreBuffer*; host pushes,
 *                 AICPU pops when rotating
 *   - total_record_count / dropped_record_count: AICPU-maintained tallies
 *
 * Note that AICore records flow through the existing per-thread ready_queue
 * in L2PerfDataHeader (with ReadyQueueEntry::is_phase=2). This keeps the
 * mgmt-thread drain path uniform with the L2PerfBuffer / PhaseBuffer paths.
 */
struct L2PerfAicoreBufferState {
    AicoreRotation rotation;                 // 64B — cache-line independent
    L2PerfFreeQueue free_queue;              // 128B
    volatile uint32_t total_record_count;    // AICPU dispatches that should have been recorded
    volatile uint32_t dropped_record_count;  // Buffers dropped (free_queue empty at rotation time)
    volatile uint32_t current_buf_seq;       // Monotonic per-core rotation counter
    uint32_t pad[13];                        // → 256B total
} __attribute__((aligned(64)));

static_assert(sizeof(L2PerfAicoreBufferState) == 256, "L2PerfAicoreBufferState must be 256 bytes");

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, the producer (AICPU for kinds 0/1,
 * AICPU on behalf of AICore for kind 2) pushes this entry. Host memory
 * manager retrieves entries from the queue.
 *
 * Entry kinds (carried in the is_phase field — name is historical, semantic
 * is "buffer kind"):
 *   0: L2PerfBuffer       (per-core,   AICPU writes)
 *   1: PhaseBuffer        (per-thread, AICPU writes)
 *   2: L2PerfAicoreBuffer (per-core,   AICore writes, AICPU enqueues at
 *                          rotation time)
 */
struct ReadyQueueEntry {
    uint32_t core_index;  // Core index (0 ~ num_cores-1), or thread_idx for phase entries
    uint32_t is_phase;    // Buffer kind: 0=L2PerfBuffer, 1=PhaseBuffer, 2=L2PerfAicoreBuffer
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
    uint32_t num_cores;      // Actual number of cores launched
    uint32_t l2_perf_level;  // 0=off, 1=AICore timing, 2=+dispatch/fanout,
                             // 3=+sched phases, 4=+orch phases. Host writes
                             // at init; AICPU reads in l2_perf_aicpu_init.
} __attribute__((aligned(64)));

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================

/**
 * AICPU phase identifier
 *
 * Scheduler phases (0-1): the two work-emitting phases per scheduler loop
 * iteration. Idle iterations no longer emit a record — host tooling recovers
 * idle spans from the gap between consecutive sched records on the same
 * thread (see swimlane_converter.py / sched_overhead_analysis.py).
 *
 * Orchestrator phase (25): one record per submit_task() / alloc_tensors()
 * call captures the entire submit's [start, end] wall-clock window.
 * Per-sub-step cycle splits (ALLOC / SYNC / LOOKUP / INSERT / PARAMS /
 * FANIN) still live in the device cold-path log as cumulative counters
 * (`g_orch_*_cycle`) — they are the right tool for "which sub-step
 * dominates overall", while the per-submit record covers "which submit
 * was slow".
 *
 * ORCH_SUBMIT is intentionally numbered above the legacy range so older
 * captures' per-sub-step records do not get re-interpreted as full-submit
 * envelopes by the new host parser (in particular: id 16 used to be
 * ORCH_SYNC — picking 16 for ORCH_SUBMIT would silently relabel every
 * legacy sync record as a submit envelope, breaking backward decoding).
 *
 * Legacy IDs:
 *   - 2, 3: SCHED_SCAN (never emitted) / SCHED_IDLE_WAIT — host parser
 *           silently drops them on old captures (idle reconstructed from
 *           gaps between work records).
 *   - 16-24: pre-fold per-sub-step orch phases (ORCH_SYNC..ORCH_SCOPE_END).
 *           Old captures may carry them; host parser maps to "unknown"
 *           and tools drop them.
 */
enum class AicpuPhaseId : uint32_t {
    // Scheduler phases (per scheduler loop iter)
    SCHED_COMPLETE = 0,  // Process completed tasks (fanin traversal)
    SCHED_DISPATCH = 1,  // Dispatch ready tasks to idle cores
    // Orchestrator phase (per submit_task() call)
    ORCH_SUBMIT = 25,  // Entire submit_task() span (placed above legacy 16-24 to avoid collision)
};

/**
 * Single AICPU scheduler phase record (40 bytes)
 *
 * Records one phase within one loop iteration of a scheduler thread.
 * No thread_id field: identity is derived from array index (position = identity).
 *
 * extra1 / extra2 carry phase-specific stats; meaning is keyed by phase_id:
 *   SCHED_DISPATCH: extra1 = pop_hit delta since last emit
 *                   extra2 = pop_miss delta since last emit
 *   SCHED_COMPLETE: extras are 0.
 *   Orchestrator phases: extras are 0 (reserved for future per-phase metrics).
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
    uint32_t extra1;  // Phase-specific delta (e.g. SCHED_DISPATCH = pop_hit)
    uint32_t extra2;  // Phase-specific delta (e.g. SCHED_DISPATCH = pop_miss)
};
static_assert(sizeof(AicpuPhaseRecord) == 40, "AicpuPhaseRecord layout drift");

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
 * Calculate total memory size including AICore states and phase profiling
 * region (buffer states only, not the record payloads themselves).
 *
 * Layout (after the fixed L2PerfDataHeader):
 *   [L2PerfBufferState × num_cores]
 *   [L2PerfAicoreBufferState × num_cores]
 *   [AicpuPhaseHeader]
 *   [PhaseBufferState × num_sched_threads]
 *
 * @param num_cores Number of AICore instances
 * @param num_sched_threads Number of phase profiling threads (scheduler + orchestrator)
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_threads) {
    return calc_perf_data_size(num_cores) + num_cores * sizeof(L2PerfAicoreBufferState) + sizeof(AicpuPhaseHeader) +
           num_sched_threads * sizeof(PhaseBufferState);
}

/**
 * Get L2PerfAicoreBufferState array start address (located immediately
 * after the L2PerfBufferState array, before the AicpuPhaseHeader).
 */
inline L2PerfAicoreBufferState *get_aicore_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2PerfAicoreBufferState *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores)
    );
}

inline L2PerfAicoreBufferState *get_aicore_buffer_state(void *base_ptr, int num_cores, int core_index) {
    return &get_aicore_buffer_states(base_ptr, num_cores)[core_index];
}

/**
 * Get AicpuPhaseHeader pointer (located after the L2PerfAicoreBufferState array).
 */
inline AicpuPhaseHeader *get_phase_header(void *base_ptr, int num_cores) {
    return reinterpret_cast<AicpuPhaseHeader *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores) +
        num_cores * sizeof(L2PerfAicoreBufferState)
    );
}

/**
 * Get PhaseBufferState array start address (located after AicpuPhaseHeader)
 */
inline PhaseBufferState *get_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<PhaseBufferState *>(
        reinterpret_cast<char *>(get_phase_header(base_ptr, num_cores)) + sizeof(AicpuPhaseHeader)
    );
}

inline PhaseBufferState *get_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_PERF_PROFILING_H_
