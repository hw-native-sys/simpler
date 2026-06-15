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
 * │  - num_sched_phase_threads, num_orch_phase_threads,         │
 * │    num_phase_cores, core_to_thread[]                        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuTaskPool[0..num_cores-1]                     │
 * │  - head:       active L2SwimlaneAicpuTaskBuffer + counters  │
 * │  - free_queue: SPSC ring of recycled buffers                │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicoreTaskPool[0..num_cores-1]                    │
 * │  - head:       active L2SwimlaneAicoreTaskBuffer (AICore    │
 * │                dcci-polls; AICPU rotates at dispatch        │
 * │                boundaries by counting per-core dispatches   │
 * │                and bumping current_buf_seq when the count   │
 * │                crosses PLATFORM_AICORE_BUFFER_SIZE)         │
 * │  - free_queue: SPSC ring of recycled AICore buffers         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuSchedPhasePool[0..num_sched_phase_threads-1] │
 * │  - head, free_queue (same shape as AicpuTaskPool)           │
 * ├─────────────────────────────────────────────────────────────┤
 * │ L2SwimlaneAicpuOrchPhasePool[0..num_orch_phase_threads-1]   │
 * │  - head, free_queue                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Actual L2SwimlaneAicpuTaskBuffer / L2SwimlaneAicpuSchedPhaseBuffer /
 * L2SwimlaneAicpuOrchPhaseBuffer / L2SwimlaneAicoreTaskBuffer are allocated
 * dynamically by Host and pushed into the per-core/thread free_queue.
 *
 * Base size = sizeof(L2SwimlaneDataHeader) + num_cores * sizeof(L2SwimlaneAicpuTaskPool)
 * With phases = Base + num_cores * sizeof(L2SwimlaneAicoreTaskPool)
 *                    + num_sched_phase_threads * sizeof(L2SwimlaneAicpuSchedPhasePool)
 *                    + num_orch_phase_threads  * sizeof(L2SwimlaneAicpuOrchPhasePool)
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
// L2SwimlaneAicpuTaskRecord - AICPU-side timing record
// =============================================================================

/**
 * AICPU-side timing record. The minimal AICPU-only payload after the
 * AICore-as-producer split: identity (task_token_raw, core_type) and
 * AICore-side timing (start/end) all live in L2SwimlaneAicoreTaskRecord; the
 * AICPU record only carries the two timestamps the AICore side cannot
 * produce, plus the host-side join key against the AICore stream.
 *
 *   - dispatch_time : AICPU timestamp when DATA_MAIN_BASE was written.
 *   - finish_time   : AICPU timestamp when AICPU observed FIN.
 *   - reg_task_id   : per-core monotonic dispatch token; join key against
 *                     L2SwimlaneAicoreTaskRecord.reg_task_id.
 *
 * Host post-processing pulls task_token_raw + start_time + end_time from
 * the matched AICore record, derives core_type from the per-core static
 * table published via L2SwimlaneCollector::set_core_types, and emits
 * func_id = -1 (resolved post-process by `swimlane_converter.py` from
 * deps.json's `kernel_ids[]`). Same path AICORE_TIMING (level=1) uses.
 *
 * Fanout edges live in the static DAG (deps.json from dep_gen) — not in
 * this record. Keeping fanout out of the hot AICPU commit path avoids a
 * per-task ~1 KB GM store + a linked-list walk on the scheduler's
 * critical fanin tail. `swimlane_converter.py` joins deps.json at
 * post-process time.
 *
 * Layout: 16B timing + 4B reg_task_id → 20B logical; `aligned(32)` rounds
 * the struct size up to 32B (compiler-inserted trailing pad) and forces
 * 32B placement alignment so each record sits in one half of a 64B cache
 * line. Two records per cache line.
 */
struct L2SwimlaneAicpuTaskRecord {
    uint64_t dispatch_time;  // AICPU timestamp: when task was dispatched to AICore
    uint64_t finish_time;    // AICPU timestamp: when AICPU observed task completion
    uint32_t reg_task_id;    // Per-core dispatch token; host join key vs AICore record
} __attribute__((aligned(32)));

static_assert(sizeof(L2SwimlaneAicpuTaskRecord) == 32, "L2SwimlaneAicpuTaskRecord must be 32B");

// =============================================================================
// L2SwimlaneAicoreTaskRecord - Slim AICore-Only Record (written by AICore, read by Host)
// =============================================================================

/**
 * Slim per-task record written by AICore directly into its own per-core
 * output buffer (no staging slot, no AICPU read). AICPU never touches this
 * record at AICORE_TIMING (level=1); at AICPU_TIMING+ the host joins it
 * against the AICPU record stream on `reg_task_id` (NOT `task_token_raw`).
 *
 * Two identity fields with different roles:
 *
 * - `task_token_raw` — the PTO2 task identity `(ring_id << 32) | local_id`.
 *   Per-task unique. AICore reads it from
 *   `LocalContext.async_ctx.task_token.raw` (already in the dispatch
 *   payload's cache line). The host pulls it from here as the canonical
 *   task id + ring decoder at ALL levels — the AICPU record carries no
 *   identity after the slim-down (only dispatch/finish timestamps and the
 *   reg_task_id join key), so AICore is the single source of truth for
 *   task identity. NOT a join key on its own: SPMD `block_num > num_cores`,
 *   MIX cluster spread, and pipeline dual-issue all dispatch the same
 *   `task_token_raw` multiple times to the same core, each producing one
 *   AICore execution record sharing the same token. The host disambiguates
 *   by `reg_task_id` below.
 *
 * - `reg_task_id` — the per-core dispatch token (low 32 bits of the
 *   per-core monotonic `dispatch_seq`). Per-dispatch unique within a core.
 *   At level≥2 the host uses this as the join key against the AICPU
 *   record stream's `L2SwimlaneAicpuTaskRecord.reg_task_id`. Each dispatch
 *   produces one AICore record + one AICPU record sharing the same
 *   reg_task_id, giving a clean 1:1 join even when multiple dispatches
 *   of the same task land on the same core.
 *
 * Layout: 24B identity/timing + 4B reg_task_id + 4B pad → 32B (half a
 * cache line). Two records pack into one cache line so AICore's per-task
 * store is at most a single line commit + dcci.
 */
struct L2SwimlaneAicoreTaskRecord {
    uint64_t start_time;      // Task start timestamp (get_sys_cnt)
    uint64_t end_time;        // Task end timestamp
    uint64_t task_token_raw;  // PTO2TaskId::raw — identity (NOT join key)
    uint32_t reg_task_id;     // Per-core dispatch token — host join key vs AICPU stream
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
 * Shared by all four pool kinds (AicpuTask / AicpuSchedPhase / AicpuOrchPhase / AicoreTask). The
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
 * The runtime's completion-before-dispatch invariant (AICore is single-
 * threaded per core and AICPU does not dispatch task K+1 until K FIN'd)
 * guarantees all prior tasks have FIN'd at rotation time, so AICore has
 * already finished writing their records (and dcci'd them out) into the
 * old buffer before AICPU enqueues it to ready_queue.
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
 * Per-core or per-thread AICPU-written pool (task profiling, sched-phase
 * profiling, or orch-phase profiling — all three share the same head +
 * free_queue plumbing; only the buffer payload type differs).
 *
 *   head:       cache line AICPU writes when rotating buffers
 *   free_queue: SPSC ring; host pushes recycled buffers, AICPU pops
 *
 * Buffer-type aliases (see further below) attach the payload type:
 *   - L2SwimlaneAicpuTaskPool      → L2SwimlaneAicpuTaskBuffer       (kind = AicpuTask)
 *   - L2SwimlaneAicpuSchedPhasePool → L2SwimlaneAicpuSchedPhaseBuffer (kind = AicpuSchedPhase)
 *   - L2SwimlaneAicpuOrchPhasePool  → L2SwimlaneAicpuOrchPhaseBuffer  (kind = AicpuOrchPhase)
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

/**
 * Per-core AICore-written pool.
 *
 *   head:        cache line AICPU writes when rotating; AICore dcci-polls per
 *                task to detect a current_buf_seq bump (= "generation" change).
 *   free_queue:  SPSC ring of recycled L2SwimlaneAicoreTaskBuffer*; host pushes,
 *                AICPU pops when rotating.
 *
 * AICore records flow through the existing per-thread ready_queue in
 * L2SwimlaneDataHeader (with ReadyQueueEntry::kind = AicoreTask). This keeps
 * the mgmt-thread drain path uniform with the AICPU buffer paths.
 *
 * Rotation trigger: AICPU counts dispatches per core in the scheduler dispatch
 * path; when a core's count crosses a PLATFORM_AICORE_BUFFER_SIZE boundary,
 * AICPU rotates BEFORE writing the next DATA_MAIN_BASE. The completion-before-
 * dispatch invariant guarantees AICore has FIN'd (and dcci'd out) every record
 * in the old buffer by then. No AICore-side signal is needed — AICPU has full
 * dispatch-count visibility on its own.
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
// ABI lock: `&pool.head` is what AICPU publishes into the rotation_table for
// AICore to dcci. Must stay at offset 0 so AICore can index from KernelArgs.
static_assert(offsetof(L2SwimlaneAicoreTaskPool, head) == 0, "L2SwimlaneAicoreTaskPool::head must be at offset 0");
static_assert(
    offsetof(L2SwimlaneAicoreTaskPool, free_queue) == 64, "L2SwimlaneAicoreTaskPool::free_queue must be at offset 64"
);

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/** Buffer kind for ReadyQueueEntry::kind. uint32_t underlying. */
enum class L2SwimlaneBufferKind : uint32_t {
    AicpuTask = 0,        // Per-core L2SwimlaneAicpuTaskBuffer, AICPU writes
    AicpuSchedPhase = 1,  // Per-thread L2SwimlaneAicpuSchedPhaseBuffer, AICPU writes
    AicpuOrchPhase = 2,   // Per-thread L2SwimlaneAicpuOrchPhaseBuffer, AICPU writes
    AicoreTask = 3,       // Per-core L2SwimlaneAicoreTaskBuffer, AICore writes, AICPU enqueues at rotation
};

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, the producer (AICPU for AICPU
 * task / sched-phase / orch-phase, AICPU on behalf of AICore for AicoreTask)
 * pushes this entry. Host memory manager retrieves entries from the queue.
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

    // Phase profiling metadata (AICPU writes in l2_swimlane_aicpu_init_phase;
    // Host reads at drain time). Both counts == 0 means phase profiling was not
    // initialized. Gated by l2_swimlane_level >= SCHED_PHASES at write time.
    // num_sched_phase_threads counts the active scheduler threads (sched-phase
    // pools are per scheduler thread, indexed by thread id). Orchestration is
    // single-threaded, so orch-phase is a single instance: num_orch_phase_threads
    // == 1 and records land in orch pool ordinal 0 (dep_gen / scope_stats style),
    // regardless of which AICPU thread the orchestrator runs on.
    uint32_t num_sched_phase_threads;           // Number of sched-phase pools the AICPU initialized
    uint32_t num_orch_phase_threads;            // Number of orch-phase pools the AICPU initialized
    uint32_t num_phase_cores;                   // Number of valid entries in core_to_thread (0 = unset)
    int8_t core_to_thread[PLATFORM_MAX_CORES];  // core_id → scheduler thread index (-1 = unassigned)
} __attribute__((aligned(64)));

// ABI lock for the merged header. The phase metadata fields and the
// core_to_thread[] array are read by both host and AICPU .so's; silent
// layout drift between them is undetectable at runtime (no magic gate
// anymore). Mirrors the pool-layout asserts in #939.
static_assert(
    offsetof(L2SwimlaneDataHeader, num_sched_phase_threads) ==
        offsetof(L2SwimlaneDataHeader, l2_swimlane_level) + sizeof(uint32_t),
    "L2SwimlaneDataHeader: num_sched_phase_threads must follow l2_swimlane_level"
);
static_assert(
    offsetof(L2SwimlaneDataHeader, num_orch_phase_threads) ==
        offsetof(L2SwimlaneDataHeader, num_sched_phase_threads) + sizeof(uint32_t),
    "L2SwimlaneDataHeader: num_orch_phase_threads must follow num_sched_phase_threads"
);
static_assert(
    offsetof(L2SwimlaneDataHeader, core_to_thread) ==
        offsetof(L2SwimlaneDataHeader, num_phase_cores) + sizeof(uint32_t),
    "L2SwimlaneDataHeader: core_to_thread[] must follow num_phase_cores"
);
static_assert(sizeof(L2SwimlaneDataHeader) % 64 == 0, "L2SwimlaneDataHeader must be 64-byte aligned");

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================
//
// Two record types route through two distinct BufferKinds (AicpuSchedPhase /
// AicpuOrchPhase) into two distinct per-thread pool arrays. Each side carries
// only the fields it actually uses — no union, no magic phase_id range gate
// at parse time.
//
// Sched records: one per work-emitting phase per scheduler loop iteration.
//   Idle iterations do not emit; host tooling reconstructs idle spans from
//   gaps between consecutive sched records on the same thread
//   (see swimlane_converter.py / sched_overhead_analysis.py).
//
// Orch records: one per submit_task() / alloc_tensors() call captures the
//   entire submit's [start, end] wall-clock window. Per-sub-step cycle
//   splits live in the device cold-path log as cumulative counters
//   (`g_orch_*_cycle`) — they answer "which sub-step dominates overall";
//   the per-submit envelope answers "which submit was slow".

/** Discriminator for the SCHED phase records (the orch side has no kind). */
enum class L2SwimlaneSchedPhaseKind : uint32_t {
    Complete = 0,  // Process completed tasks (fanin traversal)
    Dispatch = 1,  // Dispatch ready tasks to idle cores
    // Surfaced for loop iterations that produce no Complete/Dispatch bar but
    // still do real work — otherwise hidden inside the "idle" stretch.
    Poll = 2,     // Completion poll scanned running cores but retired nothing
    Release = 3,  // Deferred-release drain (on_task_release fanout/refcount work)
    // Nested child kinds — rendered inside their parent Complete/Dispatch bar
    // (Perfetto stacks time-contained events on the same lane).
    Fanout = 4,  // on_mixed_task_complete: fanin release + consumer doorbells
    Scan = 5,    // one check_running_cores pass — tasks_processed = cores' MMIO COND read
    // Speculative early-dispatch (Hook 1): building + publishing a flagged
    // producer's consumer's gated blocks. Its own bar so a long staging pass is
    // NOT mislabeled Poll — it runs on the producer's owner thread and the
    // payload-build cost is real (~us/subtask), so it must be visible directly.
    Prestage = 6,  // try_speculative_prestage — tasks_processed = blocks staged
};

/** Index layout of the queue-depth snapshot arrays below: AIC=0, AIV=1, MIX=2.
 *  Must match PTO2ResourceShape's first three values (see pto_submit_types.h).
 *  Hardcoded here rather than included to keep this header runtime-independent. */
constexpr int L2SWIMLANE_NUM_QUEUE_SHAPES = 3;

/**
 * AICPU scheduler phase record (64 bytes).
 *
 * Position in the per-thread buffer is the identity — no thread_id field.
 *
 * pop_hit / pop_miss carry SCHED_DISPATCH delta counters since the last emit
 * (zero for Complete). Kept named, not "extra1"/"extra2", so the device-side
 * commit and the host-side JSON emit don't drift on which extra means which.
 *
 * Queue-depth snapshots (local_depth_*, shared_depth_*) record the per-shape
 * scheduler queue occupancy at phase boundaries. They surface the
 * dep-release-then-discovery latency that head OH alone can't distinguish from
 * register-write latency: a phase whose start sees `local_depth=N, shared=0`
 * and end sees `local_depth=N-K` shows that K tasks were popped from this
 * thread's private buffer (invisible to peer threads) — peers must spin until
 * those tasks overflow into shared. Filled with 0 below SCHED_PHASES.
 */
struct L2SwimlaneAicpuSchedPhaseRecord {
    uint64_t start_time;                                        // Phase start timestamp
    uint64_t end_time;                                          // Phase end timestamp
    uint32_t loop_iter;                                         // Scheduler-loop iteration number on this thread
    L2SwimlaneSchedPhaseKind kind;                              // Complete or Dispatch
    uint32_t tasks_processed;                                   // Tasks processed in this phase batch
    uint32_t pop_hit;                                           // SCHED_DISPATCH delta since last emit (0 for Complete)
    uint32_t pop_miss;                                          // SCHED_DISPATCH delta since last emit (0 for Complete)
    int16_t local_depth_at_start[L2SWIMLANE_NUM_QUEUE_SHAPES];  // this thread's PTO2LocalReadyBuffer.count
    int16_t local_depth_at_end[L2SWIMLANE_NUM_QUEUE_SHAPES];
    int16_t shared_depth_at_start[L2SWIMLANE_NUM_QUEUE_SHAPES];  // sched->ready_queues[shape].size()
    int16_t shared_depth_at_end[L2SWIMLANE_NUM_QUEUE_SHAPES];
    uint32_t _pad;  // 64B alignment padding
};
static_assert(sizeof(L2SwimlaneAicpuSchedPhaseRecord) == 64, "L2SwimlaneAicpuSchedPhaseRecord layout drift");

/**
 * AICPU orchestrator phase record (32 bytes).
 *
 * One record per submit. No kind field — only one orch event type exists
 * (per-submit envelope); if a second is ever needed, add a kind field then.
 */
struct L2SwimlaneAicpuOrchPhaseRecord {
    uint64_t start_time;  // Submit start timestamp
    uint64_t end_time;    // Submit end timestamp
    uint64_t task_id;     // Full PTO2 encoding (ring_id << 32) | local_id
    uint32_t submit_idx;  // Monotonic submit counter
    uint32_t _pad;        // 32B alignment padding
};
static_assert(sizeof(L2SwimlaneAicpuOrchPhaseRecord) == 32, "L2SwimlaneAicpuOrchPhaseRecord layout drift");

constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;  // ~512KB per sched thread, ~512KB per orch thread

// Fixed-size phase record buffers. Same TypedBuffer template as the task
// buffers — keeps the drain machinery uniform.
using L2SwimlaneAicpuSchedPhaseBuffer = TypedBuffer<L2SwimlaneAicpuSchedPhaseRecord, PLATFORM_PHASE_RECORDS_PER_THREAD>;
using L2SwimlaneAicpuOrchPhaseBuffer = TypedBuffer<L2SwimlaneAicpuOrchPhaseRecord, PLATFORM_PHASE_RECORDS_PER_THREAD>;

// Sched and orch phase pools share the same head+free_queue layout as
// AicpuTaskPool; the buffer payload type differs but the pool plumbing is
// identical. Aliasing keeps the drain machinery polymorphic.
using L2SwimlaneAicpuSchedPhasePool = L2SwimlaneAicpuTaskPool;
using L2SwimlaneAicpuOrchPhasePool = L2SwimlaneAicpuTaskPool;

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
 * Calculate total memory size including AICore states and both phase
 * profiling regions (buffer states only, not the record payloads themselves).
 *
 * Layout (after the fixed L2SwimlaneDataHeader, which carries the phase
 * metadata fields):
 *   [L2SwimlaneAicpuTaskPool       × num_cores]
 *   [L2SwimlaneAicoreTaskPool      × num_cores]
 *   [L2SwimlaneAicpuSchedPhasePool × num_sched_phase_threads]
 *   [L2SwimlaneAicpuOrchPhasePool  × num_orch_phase_threads]
 *
 * @param num_cores               Number of AICore instances
 * @param num_sched_phase_threads Number of scheduler-phase pools
 * @param num_orch_phase_threads  Number of orchestrator-phase pools
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_phase_threads, int num_orch_phase_threads) {
    return calc_perf_data_size(num_cores) + num_cores * sizeof(L2SwimlaneAicoreTaskPool) +
           num_sched_phase_threads * sizeof(L2SwimlaneAicpuSchedPhasePool) +
           num_orch_phase_threads * sizeof(L2SwimlaneAicpuOrchPhasePool);
}

/**
 * Get L2SwimlaneAicoreTaskPool array start address (located immediately
 * after the L2SwimlaneAicpuTaskPool array).
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
 * Get L2SwimlaneAicpuSchedPhasePool array start address (located immediately
 * after the L2SwimlaneAicoreTaskPool array).
 */
inline L2SwimlaneAicpuSchedPhasePool *get_sched_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2SwimlaneAicpuSchedPhasePool *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores) +
        num_cores * sizeof(L2SwimlaneAicoreTaskPool)
    );
}

inline L2SwimlaneAicpuSchedPhasePool *get_sched_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_sched_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

/**
 * Get L2SwimlaneAicpuOrchPhasePool array start address (located immediately
 * after the L2SwimlaneAicpuSchedPhasePool array).
 *
 * Layout stride is fixed at `PLATFORM_MAX_AICPU_THREADS`, NOT the runtime
 * `num_sched_phase_threads` count. The host pre-allocates both phase pool
 * arrays at the platform max because it doesn't know the actual AICPU
 * thread count at shm-alloc time; AICPU later picks a smaller count for
 * iteration. The OFFSET must match the host's alloc layout (max), not
 * AICPU's iteration count (actual) — otherwise AICPU reads the orch array
 * from inside the (still allocated) sched array tail, corrupting both.
 */
inline L2SwimlaneAicpuOrchPhasePool *get_orch_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<L2SwimlaneAicpuOrchPhasePool *>(
        reinterpret_cast<char *>(get_sched_phase_buffer_states(base_ptr, num_cores)) +
        PLATFORM_MAX_AICPU_THREADS * sizeof(L2SwimlaneAicpuSchedPhasePool)
    );
}

inline L2SwimlaneAicpuOrchPhasePool *get_orch_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_orch_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_L2_SWIMLANE_PROFILING_H_
