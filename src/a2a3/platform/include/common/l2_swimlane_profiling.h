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
 * @file l2_swimlane_profiling.h
 * @brief Performance profiling data structures
 *
 * Architecture: Fixed header + per-core/thread buffer states + optional phase profiling region
 *
 * Memory layout (shared memory between Host and Device):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ L2SwimlaneDataHeader (fixed header)                         │
 * │  - ReadyQueue (FIFO, capacity=PLATFORM_PROF_READYQUEUE_SIZE)│
 * │  - num_cores, l2_swimlane_level                             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuTaskPool[0..num_cores-1]                     │
 * │  - head:       active L2SwimlaneAicpuTaskBuffer + counters  │
 * │  - free_queue: SPSC ring of recycled buffers                │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicoreTaskPool[0..num_cores-1]                    │
 * │  - head:       active L2SwimlaneAicoreTaskBuffer (AICore    │
 * │                dcci-polls; AICPU rotates at dispatch        │
 * │                boundaries by bumping current_buf_seq)       │
 * │  - free_queue: SPSC ring of recycled AICore buffers         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuPhaseHeader (optional, present when phase    │
 * │                             profiling is enabled)           │
 * │  - magic, num_sched_threads, records_per_thread             │
 * │  - core_to_thread mapping                                   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuPhasePool[0..num_threads-1]                  │
 * │  - head, free_queue (same shape as AicpuTaskPool)           │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Actual L2SwimlaneAicpuTaskBuffer / L2SwimlaneAicoreTaskBuffer /
 * L2SwimlaneAicpuPhaseBuffer are allocated dynamically by Host and pushed
 * into the per-core/thread free_queue.
 *
 * Base size = sizeof(L2SwimlaneDataHeader) + num_cores * sizeof(L2SwimlaneAicpuTaskPool)
 * With phases = Base + num_cores * sizeof(L2SwimlaneAicoreTaskPool)
 *                    + sizeof(L2SwimlaneAicpuPhaseHeader)
 *                    + num_threads * sizeof(L2SwimlaneAicpuPhasePool)
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_SWIMLANE_PROFILING_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_SWIMLANE_PROFILING_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/core_type.h"
#include "common/platform_config.h"

// =============================================================================
// L2 swimlane_level — granularity ladder for the L2 swimlane profiler.
//
// Each level is a strict superset of the previous: higher levels add the data
// described by their name on top of all lower-level data. Naming describes
// what is NEWLY captured at that level (incremental view), so gate sites read
// naturally — e.g. `if (level >= SCHED_PHASES)` means "this section runs when
// scheduler phase records are being collected (or any higher tier)".
//
// Transported via `L2SwimlaneDataHeader::l2_swimlane_level` (host → AICPU,
// shared memory) and `CallConfig::enable_l2_swimlane` (Python → C). The wire
// representation stays integer (uint32_t / int32_t) for ABI stability; this
// enum is the canonical in-code type used for comparisons.
// =============================================================================
enum class L2SwimlaneLevel : uint32_t {
    DISABLED = 0,       // No collection at all
    AICORE_TIMING = 1,  // AICore per-task start/end timestamps + task record buffer
    AICPU_TIMING = 2,   // + AICPU dispatch/finish timestamps
    SCHED_PHASES = 3,   // + scheduler main-loop phase records (SCHED_COMPLETE/DISPATCH/IDLE_WAIT)
    ORCH_PHASES = 4,    // + orchestrator phase records
};

// =============================================================================
// L2SwimlaneAicpuTaskRecord - Single Task Execution Record
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
struct L2SwimlaneAicpuTaskRecord {
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
                           // L2SwimlaneAicoreTaskRecord.task_id, which is what
                           // AICore writes into the slim record.
} __attribute__((aligned(64)));

static_assert(
    sizeof(L2SwimlaneAicpuTaskRecord) % 64 == 0,
    "L2SwimlaneAicpuTaskRecord must be 64-byte aligned for optimal cache performance"
);

// =============================================================================
// L2SwimlaneAicoreTaskRecord - Slim AICore-Only Record (written by AICore, read by Host)
// =============================================================================

/**
 * Slim per-task record written by AICore directly into its own per-core
 * output buffer (no staging slot, no AICPU read). AICPU never touches this
 * record. The host post-processor joins it against the AICPU-side
 * L2SwimlaneAicpuTaskRecord on `task_id` at flush time.
 *
 * Layout: 24B payload + 8B pad → 32B (half a cache line). Two records pack
 * into one cache line so AICore's per-task store is at most a single line
 * commit + dcci.
 */
struct L2SwimlaneAicoreTaskRecord {
    uint64_t start_time;  // Task start timestamp (get_sys_cnt)
    uint64_t end_time;    // Task end timestamp
    uint32_t task_id;     // Register dispatch token (low 32 bits)
    uint32_t _pad;
} __attribute__((aligned(32)));

static_assert(sizeof(L2SwimlaneAicoreTaskRecord) == 32, "L2SwimlaneAicoreTaskRecord must be 32B");

// =============================================================================
// TypedBuffer<Record, N> - Templated Fixed-Size Profiling Buffer
// =============================================================================

/**
 * Generic fixed-capacity profiling buffer: contiguous record array followed
 * by a producer-written count. Layout matches the legacy L2SwimlaneAicpuTaskBuffer so the
 * host allocator and the AICPU consumer can treat all concrete instances
 * uniformly.
 *
 * Concrete instantiations live below as `using` aliases.
 *   - L2SwimlaneAicpuTaskBuffer        — AICPU-written, rotated, ready-queue tagged kind=AicpuTask
 *   - L2SwimlaneAicoreTaskBuffer  — AICore-written, NOT rotated (sized for the full
 *                           session), read by host at flush time
 */
template <typename Record, size_t N>
struct TypedBuffer {
    Record records[N];
    volatile uint32_t count;
} __attribute__((aligned(64)));

using L2SwimlaneAicpuTaskBuffer = TypedBuffer<L2SwimlaneAicpuTaskRecord, PLATFORM_PROF_BUFFER_SIZE>;

// AICore buffer is rotated like L2SwimlaneAicpuTaskBuffer: a small fixed capacity per
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

using L2SwimlaneAicoreTaskBuffer = TypedBuffer<L2SwimlaneAicoreTaskRecord, PLATFORM_AICORE_BUFFER_SIZE>;

// =============================================================================
// L2SwimlaneFreeQueue - SPSC Lock-Free Queue for Free Buffers
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
struct L2SwimlaneFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PROF_SLOT_COUNT];  // Free buffer addresses
    volatile uint32_t head;                                   // Consumer read position (Device increments)
    volatile uint32_t tail;                                   // Producer write position (Host increments)
    uint32_t pad[13];                                         // Pad to 128 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(L2SwimlaneFreeQueue) == 128, "L2SwimlaneFreeQueue must be 128 bytes for cache alignment");

// =============================================================================
// L2SwimlaneActiveHead - Shared "Active Buffer" Cache Line
// =============================================================================

/**
 * Single cache-line head describing the per-pool active buffer.
 *
 * Shared by all three pool kinds (AicpuTask / AicpuPhase / AicoreTask). The
 * field set is intentionally uniform — every pool needs:
 *   - current_buf_ptr      : device address of the buffer the producer is
 *                            currently writing into (0 = no active buffer)
 *   - current_buf_seq      : monotonic sequence number; bumped on every
 *                            rotation. For AICore this doubles as the
 *                            "generation" the per-core local state compares
 *                            against to detect a rotation.
 *   - total_record_count   : producer-maintained tally; host cross-checks at
 *                            end-of-run that `collected + dropped == total`.
 *   - dropped_record_count : producer-maintained tally; counts records lost
 *                            (free_queue empty / overwrite / no buffer).
 *
 * Single-writer (AICPU) for every pool; readers are either AICore (via dcci
 * SINGLE_CACHE_LINE, AicoreTask pool only) or the host at drain time. Because
 * AICore only reads current_buf_ptr/current_buf_seq and invalidates the whole
 * line, the cohabiting counter fields are harmless — AICore never reads them.
 *
 * Race avoidance for AicoreTask pools: AICPU rotates strictly before
 * `write_reg(DATA_MAIN_BASE)` for the first task of a new BUFFER_SIZE batch.
 * The runtime's completion-before-dispatch invariant guarantees all prior
 * tasks have FIN'd, so AICore has already finished writing their records into
 * the old buffer before AICPU enqueues it to ready_queue.
 */
struct L2SwimlaneActiveHead {
    volatile uint64_t current_buf_ptr;       // 8 — active buffer device address (0 = none)
    volatile uint32_t current_buf_seq;       // 4 — monotonic seq / AICore rotation generation
    volatile uint32_t total_record_count;    // 4 — producer-attempted writes
    volatile uint32_t dropped_record_count;  // 4 — producer-dropped writes
    uint32_t pad[11];                        // 44 → 64B
} __attribute__((aligned(64)));

static_assert(sizeof(L2SwimlaneActiveHead) == 64, "L2SwimlaneActiveHead must be one cache line");

// =============================================================================
// Pool layouts: every pool = ActiveHead (64B) + L2SwimlaneFreeQueue (128B) = 192B
// =============================================================================

/**
 * Per-core or per-thread AICPU-written pool (Task profiling or Phase profiling).
 *
 *   head:       cache line AICPU writes when rotating buffers
 *   free_queue: SPSC ring; host pushes recycled buffers, AICPU pops
 *
 * Used in two contexts:
 *   - Per-core L2SwimlaneAicpuTaskBuffer  (kind = AicpuTask)
 *   - Per-thread L2SwimlaneAicpuPhaseBuffer (kind = AicpuPhase)
 */
struct L2SwimlaneAicpuTaskPool {
    L2SwimlaneActiveHead head;       // 64B
    L2SwimlaneFreeQueue free_queue;  // 128B
} __attribute__((aligned(64)));

static_assert(sizeof(L2SwimlaneAicpuTaskPool) == 192, "L2SwimlaneAicpuTaskPool must be 192 bytes");
// Lock the head@0 / free_queue@64 ABI: AICPU publishes `&pool.head` device
// addresses into the AICore rotation table, and host/device drain paths rely
// on this layout being byte-stable across builds. Drift here is the kind of
// silent corruption that doesn't trip any test.
static_assert(offsetof(L2SwimlaneAicpuTaskPool, head) == 0, "L2SwimlaneAicpuTaskPool::head must be at offset 0");
static_assert(
    offsetof(L2SwimlaneAicpuTaskPool, free_queue) == 64, "L2SwimlaneAicpuTaskPool::free_queue must be at offset 64"
);

// Type alias for semantic clarity in Phase profiling context
using L2SwimlaneAicpuPhasePool = L2SwimlaneAicpuTaskPool;  // Per-thread Phase profiling

/**
 * Per-core AICore-written pool.
 *
 *   head:       cache line AICPU writes when rotating; AICore dcci-polls per
 *               task to detect a current_buf_seq bump (= "generation" change).
 *   free_queue: SPSC ring of recycled L2SwimlaneAicoreTaskBuffer*; host pushes,
 *               AICPU pops when rotating.
 *
 * AICore records flow through the existing per-thread ready_queue in
 * L2SwimlaneDataHeader (with ReadyQueueEntry::kind = AicoreTask). This keeps
 * the mgmt-thread drain path uniform with the AICPU buffer paths.
 *
 * The AICore-readable rotation channel that AICore's per-task dcci targets is
 * exactly `&pool.head` — AICPU publishes that address into
 * `KernelArgs::l2_swimlane_aicore_rotation_table[block_idx]` during
 * `l2_swimlane_aicpu_init`, and AICore lazy-resolves it via
 * `get_l2_swimlane_aicore_head()`.
 */
struct L2SwimlaneAicoreTaskPool {
    L2SwimlaneActiveHead head;       // 64B
    L2SwimlaneFreeQueue free_queue;  // 128B
} __attribute__((aligned(64)));

static_assert(sizeof(L2SwimlaneAicoreTaskPool) == 192, "L2SwimlaneAicoreTaskPool must be 192 bytes");
// Same ABI lock as AicpuTaskPool: head must be the line AICore dcci's, so
// `&pool.head` (= base + 0) and `&pool.free_queue` (= base + 64) must stay
// at these exact offsets across builds.
static_assert(offsetof(L2SwimlaneAicoreTaskPool, head) == 0, "L2SwimlaneAicoreTaskPool::head must be at offset 0");
static_assert(
    offsetof(L2SwimlaneAicoreTaskPool, free_queue) == 64, "L2SwimlaneAicoreTaskPool::free_queue must be at offset 64"
);

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * Buffer kind for ReadyQueueEntry::kind. Wire-stable uint32_t underlying so the
 * struct layout matches the prior `is_phase` field byte-for-byte. The AicpuTask
 * and Phase values match the historical 0/1; AicoreTask was 2.
 */
enum class L2SwimlaneBufferKind : uint32_t {
    AicpuTask = 0,   // Per-core L2SwimlaneAicpuTaskBuffer, AICPU writes
    AicpuPhase = 1,  // Per-thread L2SwimlaneAicpuPhaseBuffer, AICPU writes
    AicoreTask = 2,  // Per-core L2SwimlaneAicoreTaskBuffer, AICore writes, AICPU enqueues at rotation
};

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, the producer (AICPU for
 * AicpuTask/AicpuPhase, AICPU on behalf of AICore for AicoreTask) pushes this
 * entry. Host memory manager retrieves entries from the queue.
 */
struct ReadyQueueEntry {
    uint32_t core_index;        // Core index (0 ~ num_cores-1), or thread_idx for phase entries
    L2SwimlaneBufferKind kind;  // Buffer kind discriminator (uint32_t underlying)
    uint64_t buffer_ptr;        // Device pointer to the full buffer
    uint32_t buffer_seq;        // Sequence number for ordering
    uint32_t pad;               // Alignment padding
} __attribute__((aligned(32)));

// =============================================================================
// L2SwimlaneDataHeader - Fixed Header
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
struct L2SwimlaneDataHeader {
    // Per-thread ready queues (FIFO Circular Buffers)
    // Each AICPU thread has its own queue to avoid lock contention
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PROF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Consumer read positions (Host modifies)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // Producer write positions (AICPU modifies)

    // Metadata (Host initializes, Device read-only)
    uint32_t num_cores;          // Actual number of cores launched
    uint32_t l2_swimlane_level;  // 0=off, 1=AICore timing, 2=+dispatch/fanout,
                                 // 3=+sched phases, 4=+orch phases. Host writes
                                 // at init; AICPU reads in l2_swimlane_aicpu_init.
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
enum class L2SwimlaneAicpuPhaseId : uint32_t {
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
struct L2SwimlaneAicpuPhaseRecord {
    uint64_t start_time;              // Phase start timestamp
    uint64_t end_time;                // Phase end timestamp
    uint32_t loop_iter;               // Loop iteration number
    L2SwimlaneAicpuPhaseId phase_id;  // Phase type
    union {
        uint64_t task_id;          // tensormap_and_ringbuffer: full PTO2 encoding
                                   // (ring_id << 32) | local_id for cross-view correlation.
        uint64_t tasks_processed;  // Scheduler phases: number of tasks processed in this batch
    };
    uint32_t extra1;  // Phase-specific delta (e.g. SCHED_DISPATCH = pop_hit)
    uint32_t extra2;  // Phase-specific delta (e.g. SCHED_DISPATCH = pop_miss)
};
static_assert(sizeof(L2SwimlaneAicpuPhaseRecord) == 40, "L2SwimlaneAicpuPhaseRecord layout drift");

constexpr uint32_t L2_SWIMLANE_AICPU_PHASE_MAGIC = 0x41435048;  // "ACPH"
constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;        // ~512KB per thread

// Fixed-size phase record buffer. Same TypedBuffer template as L2SwimlaneAicpuTaskBuffer
// and L2SwimlaneAicoreTaskBuffer — keeps the drain machinery uniform.
using L2SwimlaneAicpuPhaseBuffer = TypedBuffer<L2SwimlaneAicpuPhaseRecord, PLATFORM_PHASE_RECORDS_PER_THREAD>;

/**
 * AICPU phase profiling header
 *
 * Located after the L2SwimlaneAicpuTaskPool array in shared memory.
 * Contains metadata and per-thread tracking.
 */
struct L2SwimlaneAicpuPhaseHeader {
    uint32_t magic;                             // Validation magic (L2_SWIMLANE_AICPU_PHASE_MAGIC)
    uint32_t num_sched_threads;                 // Number of scheduler threads
    uint32_t records_per_thread;                // Max records per L2SwimlaneAicpuPhaseBuffer
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
 *                     = sizeof(L2SwimlaneDataHeader) + num_cores × sizeof(L2SwimlaneAicpuTaskPool)
 *
 * @param num_cores Number of cores (block_dim × PLATFORM_CORES_PER_BLOCKDIM)
 * @return Total bytes for header + buffer states
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(L2SwimlaneDataHeader) + num_cores * sizeof(L2SwimlaneAicpuTaskPool);
}

/**
 * Get header pointer
 *
 * @param base_ptr Shared memory base address (device_ptr or host_ptr)
 * @return L2SwimlaneDataHeader pointer
 */
inline L2SwimlaneDataHeader *get_l2_swimlane_header(void *base_ptr) {
    return reinterpret_cast<L2SwimlaneDataHeader *>(base_ptr);
}

/**
 * Get L2SwimlaneAicpuTaskPool array start address
 *
 * @param base_ptr Shared memory base address
 * @return L2SwimlaneAicpuTaskPool array pointer
 */
inline L2SwimlaneAicpuTaskPool *get_perf_buffer_states(void *base_ptr) {
    return reinterpret_cast<L2SwimlaneAicpuTaskPool *>(
        reinterpret_cast<char *>(base_ptr) + sizeof(L2SwimlaneDataHeader)
    );
}

/**
 * Get L2SwimlaneAicpuTaskPool for specified core
 *
 * @param base_ptr Shared memory base address
 * @param core_index Core index (0 ~ num_cores-1)
 * @return L2SwimlaneAicpuTaskPool pointer
 */
inline L2SwimlaneAicpuTaskPool *get_perf_buffer_state(void *base_ptr, int core_index) {
    return &get_perf_buffer_states(base_ptr)[core_index];
}

/**
 * Calculate total memory size including AICore states and phase profiling
 * region (buffer states only, not the record payloads themselves).
 *
 * Layout (after the fixed L2SwimlaneDataHeader):
 *   [L2SwimlaneAicpuTaskPool × num_cores]
 *   [L2SwimlaneAicoreTaskPool × num_cores]
 *   [L2SwimlaneAicpuPhaseHeader]
 *   [L2SwimlaneAicpuPhasePool × num_sched_threads]
 *
 * @param num_cores Number of AICore instances
 * @param num_sched_threads Number of phase profiling threads (scheduler + orchestrator)
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_threads) {
    return calc_perf_data_size(num_cores) + num_cores * sizeof(L2SwimlaneAicoreTaskPool) +
           sizeof(L2SwimlaneAicpuPhaseHeader) + num_sched_threads * sizeof(L2SwimlaneAicpuPhasePool);
}

/**
 * Get L2SwimlaneAicoreTaskPool array start address (located immediately
 * after the L2SwimlaneAicpuTaskPool array, before the L2SwimlaneAicpuPhaseHeader).
 */
inline L2SwimlaneAicoreTaskPool *get_aicore_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2SwimlaneAicoreTaskPool *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores)
    );
}

inline L2SwimlaneAicoreTaskPool *get_aicore_buffer_state(void *base_ptr, int num_cores, int core_index) {
    return &get_aicore_buffer_states(base_ptr, num_cores)[core_index];
}

/**
 * Get L2SwimlaneAicpuPhaseHeader pointer (located after the L2SwimlaneAicoreTaskPool array).
 */
inline L2SwimlaneAicpuPhaseHeader *get_phase_header(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2SwimlaneAicpuPhaseHeader *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores) +
        num_cores * sizeof(L2SwimlaneAicoreTaskPool)
    );
}

/**
 * Get L2SwimlaneAicpuPhasePool array start address (located after L2SwimlaneAicpuPhaseHeader)
 */
inline L2SwimlaneAicpuPhasePool *get_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2SwimlaneAicpuPhasePool *>(
        reinterpret_cast<char *>(get_phase_header(base_ptr, num_cores)) + sizeof(L2SwimlaneAicpuPhaseHeader)
    );
}

inline L2SwimlaneAicpuPhasePool *get_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_SWIMLANE_PROFILING_H_
