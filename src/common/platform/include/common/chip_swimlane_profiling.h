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
 * @file chip_swimlane_profiling.h
 * @brief Performance profiling data structures
 *
 * Architecture: Fixed header + per-core/thread buffer states + optional phase profiling region
 *
 * Memory layout (shared memory between Host and Device):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ ChipSwimlaneDataHeader (fixed header)                         │
 * │  - ReadyQueue (FIFO, capacity=PLATFORM_PROF_READYQUEUE_SIZE)│
 * │  - num_cores, chip_swimlane_level                             │
 * │  - num_sched_phase_threads, num_orch_phase_threads,         │
 * │    num_phase_cores, core_to_thread[]                        │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ChipSwimlaneAicpuTaskPool[0..num_cores-1]                     │
 * │  - head:       active ChipSwimlaneAicpuTaskBuffer + counters  │
 * │  - free_queue: SPSC ring of recycled buffers                │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ChipSwimlaneAicoreTaskPool[0..num_cores-1]                    │
 * │  - head:       active ChipSwimlaneAicoreTaskBuffer (AICore    │
 * │                dcci-polls; AICPU rotates at dispatch        │
 * │                boundaries by counting per-core dispatches   │
 * │                and bumping current_buf_seq when the count   │
 * │                crosses PLATFORM_AICORE_BUFFER_SIZE)         │
 * │  - free_queue: SPSC ring of recycled AICore buffers         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ChipSwimlaneAicpuSchedPhasePool[0..MAX_AICPU_THREADS-1]       │
 * │  - head, free_queue; header records the active thread count │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ChipSwimlaneAicpuOrchPhasePool[0..num_orch_phase_threads-1]   │
 * │  - head, free_queue                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Actual ChipSwimlaneAicpuTaskBuffer / ChipSwimlaneAicpuSchedPhaseBuffer /
 * ChipSwimlaneAicpuOrchPhaseBuffer / ChipSwimlaneAicoreTaskBuffer are allocated
 * dynamically by Host and pushed into the per-core/thread free_queue.
 *
 * Base size = sizeof(ChipSwimlaneDataHeader) + num_cores * sizeof(ChipSwimlaneAicpuTaskPool)
 * With phases = Base + num_cores * sizeof(ChipSwimlaneAicoreTaskPool)
 *                    + PLATFORM_MAX_AICPU_THREADS * sizeof(ChipSwimlaneAicpuSchedPhasePool)
 *                    + num_orch_phase_threads  * sizeof(ChipSwimlaneAicpuOrchPhasePool)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/core_type.h"
#include "common/dfx_backpressure_device.h"
#include "common/host_phase_kind.h"
#include "common/platform_config.h"
#include "common/scheduler_profiling.h"

// =============================================================================
// chip swimlane_level — granularity ladder for the chip swimlane profiler.
//
// Each level is a strict superset of the previous: higher levels add the data
// described by their name on top of all lower-level data. Naming describes
// what is NEWLY captured at that level (incremental view), so gate sites read
// naturally — e.g. `if (level >= SCHED_PHASES)` means "this section runs when
// scheduler phase records are being collected (or any higher tier)".
//
// Transported via `ChipSwimlaneDataHeader::chip_swimlane_level` (host → AICPU,
// shared memory) and `CallConfig::enable_chip_swimlane` (Python → C). The wire
// representation stays integer (uint32_t / int32_t) for ABI stability; this
// enum is the canonical in-code type used for comparisons.
// =============================================================================
enum class ChipSwimlaneLevel : uint32_t {
    DISABLED = 0,         // No collection at all
    TASK_TIMING = 1,      // AICore per-task start/end timestamps + task record buffer
    SCHEDULE_TIMING = 2,  // + Scheduler per-task dispatch/finish timestamps
    SCHED_PHASES = 3,     // + scheduler main-loop phase records (SCHED_COMPLETE/DISPATCH/IDLE_WAIT)
    ORCH_PHASES = 4,      // + orchestrator phase records
};

// =============================================================================
// ChipSwimlaneAicpuTaskRecord - AICPU-side timing record
// =============================================================================

/**
 * AICPU Scheduler timing record. The minimal AICPU-only payload after the
 * AICore-as-producer split: identity (task_token_raw, core_type) and
 * AICore-side timing (start/end) all live in ChipSwimlaneAicoreTaskRecord; the
 * AICPU record carries only the two timestamps the AICore side cannot produce
 * (the scheduler's dispatch/finish), plus the host-side join key against the
 * AICore stream.
 *
 *   - dispatch_time : AICPU timestamp when DATA_MAIN_BASE was written.
 *   - finish_time   : AICPU timestamp when AICPU observed FIN.
 *   - reg_task_id   : per-core monotonic dispatch token; join key against
 *                     ChipSwimlaneAicoreTaskRecord.reg_task_id.
 *
 * Host post-processing pulls task_token_raw + start_time + end_time from
 * the matched AICore record, derives core_type from the per-core static
 * table published via ChipSwimlaneCollector::set_core_types, and emits
 * func_id = -1 (resolved post-process by `swimlane_converter.py` from
 * deps.json's `kernel_ids[]`). Same path TASK_TIMING (level=1) uses.
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
struct ChipSwimlaneAicpuTaskRecord {
    uint64_t dispatch_time;  // AICPU timestamp: when task was dispatched to AICore
    uint64_t finish_time;    // AICPU timestamp: when AICPU observed task completion
    uint32_t reg_task_id;    // Per-core dispatch token; host join key vs AICore record
} __attribute__((aligned(32)));

static_assert(sizeof(ChipSwimlaneAicpuTaskRecord) == 32, "ChipSwimlaneAicpuTaskRecord must be 32B");

// =============================================================================
// ChipSwimlaneAicoreTaskRecord - Slim AICore-Only Record (written by AICore, read by Host)
// =============================================================================

/**
 * Slim per-task record written by AICore directly into its own per-core
 * output buffer (no staging slot, no AICPU read). AICPU never touches this
 * record at TASK_TIMING (level=1); at SCHEDULE_TIMING+ the host joins it
 * against the active Scheduler producer's record stream on `reg_task_id`
 * (NOT `task_token_raw`).
 *
 * Two identity fields with different roles:
 *
 * - `task_token_raw` — the task identity, a `TaskId::raw` in whatever layout
 *   the minting runtime uses. Per-task unique. AICore reads it from
 *   `LocalContext.async_ctx.task_token.raw` (already in the dispatch
 *   payload's cache line). The host pulls it from here as the canonical
 *   task id at ALL levels — the AICPU record carries no
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
 *   record stream's `ChipSwimlaneAicpuTaskRecord.reg_task_id`. Each dispatch
 *   produces one AICore record + one AICPU record sharing the same
 *   reg_task_id, giving a clean 1:1 join even when multiple dispatches
 *   of the same task land on the same core.
 *
 * Layout: 24B identity/timing + 4B reg_task_id + 4B receive_to_start delta →
 * 32B (half a cache line). Two records pack into one cache line so AICore's
 * per-task store is at most a single line commit + dcci.
 *
 * receive_to_start_cycles isolates the AICore-side dcci+ack cost from the
 * AICPU→AICore NoC propagation. AICore captures receive_time right after
 * `read_reg(DATA_MAIN_BASE)` returns the new task_id (before dcci+ack), and
 * start_time after them. Host derives:
 *   - receive_time  = start_time - receive_to_start_cycles
 *   - propagation   = receive_time - dispatch_ts (AICPU view)
 *   - local_setup   = receive_to_start_cycles    (dcci + ack)
 * Delta fits in 32 bits on all supported platforms (minimum wrap window
 * is approximately 4.3 s at the fastest 1 GHz counter).
 */
struct ChipSwimlaneAicoreTaskRecord {
    uint64_t start_time;               // Post-dcci+ack timestamp (kernel begins next)
    uint64_t end_time;                 // Post-kernel timestamp
    uint64_t task_token_raw;           // TaskId::raw — identity (NOT join key)
    uint32_t reg_task_id;              // Per-core dispatch token — host join key vs AICPU stream
    uint32_t receive_to_start_cycles;  // start_time - receive_time (AICore-local dcci + ack cost)
} __attribute__((aligned(32)));

static_assert(sizeof(ChipSwimlaneAicoreTaskRecord) == 32, "ChipSwimlaneAicoreTaskRecord must be 32B");

// =============================================================================
// TypedBuffer<Record, N> - Templated Fixed-Size Profiling Buffer
// =============================================================================

/**
 * Generic fixed-capacity profiling buffer: contiguous record array followed
 * by a producer-written count. Layout matches the legacy ChipSwimlaneAicpuTaskBuffer so the
 * host allocator and the AICPU consumer can treat all concrete instances
 * uniformly.
 *
 * Concrete instantiations live below as `using` aliases.
 *   - ChipSwimlaneAicpuTaskBuffer        — AICPU-written, rotated, ready-queue tagged kind=AicpuTask
 *   - ChipSwimlaneAicoreTaskBuffer  — AICore-written, NOT rotated (sized for the full
 *                           session), read by host at flush time
 */
template <typename Record, size_t N>
struct TypedBuffer {
    Record records[N];
    volatile uint32_t count;
} __attribute__((aligned(64)));

using ChipSwimlaneAicpuTaskBuffer = TypedBuffer<ChipSwimlaneAicpuTaskRecord, PLATFORM_PROF_BUFFER_SIZE>;

// AICore buffer is rotated like ChipSwimlaneAicpuTaskBuffer: a small fixed capacity per
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

using ChipSwimlaneAicoreTaskBuffer = TypedBuffer<ChipSwimlaneAicoreTaskRecord, PLATFORM_AICORE_BUFFER_SIZE>;

// =============================================================================
// ChipSwimlaneFreeQueue - SPSC Lock-Free Queue for Free Buffers
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
struct ChipSwimlaneFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PROF_SLOT_COUNT];  // Free buffer addresses
    volatile uint32_t head;                                   // Consumer read position (Device increments)
    volatile uint32_t tail;                                   // Producer write position (Host increments)
    uint32_t pad[13];                                         // Pad to 128 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(ChipSwimlaneFreeQueue) == 128, "ChipSwimlaneFreeQueue must be 128 bytes for cache alignment");

// =============================================================================
// ChipSwimlaneActiveHead - Shared "Active Buffer" Cache Line
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
struct ChipSwimlaneActiveHead {
    volatile uint64_t current_buf_ptr;       // 8 — active buffer device address (0 = none)
    volatile uint32_t current_buf_seq;       // 4 — monotonic seq / AICore rotation generation
    volatile uint32_t total_record_count;    // 4 — producer-attempted writes
    volatile uint32_t dropped_record_count;  // 4 — producer-dropped writes
    uint32_t pad[11];                        // 44 → 64B
} __attribute__((aligned(64)));

static_assert(sizeof(ChipSwimlaneActiveHead) == 64, "ChipSwimlaneActiveHead must be one cache line");

// =============================================================================
// Pool layouts: every pool = ActiveHead (64B) + ChipSwimlaneFreeQueue (128B) = 192B
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
 *   - ChipSwimlaneAicpuTaskPool      → ChipSwimlaneAicpuTaskBuffer       (kind = AicpuTask)
 *   - ChipSwimlaneAicpuSchedPhasePool → ChipSwimlaneAicpuSchedPhaseBuffer (kind = AicpuSchedPhase)
 *   - ChipSwimlaneAicpuOrchPhasePool  → ChipSwimlaneAicpuOrchPhaseBuffer  (kind = AicpuOrchPhase)
 */
struct ChipSwimlaneAicpuTaskPool {
    ChipSwimlaneActiveHead head;       // 64B
    ChipSwimlaneFreeQueue free_queue;  // 128B
} __attribute__((aligned(64)));

static_assert(sizeof(ChipSwimlaneAicpuTaskPool) == 192, "ChipSwimlaneAicpuTaskPool must be 192 bytes");
// Lock the head@0 / free_queue@64 ABI: AICPU publishes `&pool.head` device
// addresses into the AICore rotation table, and host/device drain paths rely
// on this layout being byte-stable across builds. Drift here is the kind of
// silent corruption that doesn't trip any test.
static_assert(offsetof(ChipSwimlaneAicpuTaskPool, head) == 0, "ChipSwimlaneAicpuTaskPool::head must be at offset 0");
static_assert(
    offsetof(ChipSwimlaneAicpuTaskPool, free_queue) == 64, "ChipSwimlaneAicpuTaskPool::free_queue must be at offset 64"
);

/**
 * Per-core AICore-written pool.
 *
 *   head:        cache line AICPU writes when rotating; AICore dcci-polls per
 *                task to detect a current_buf_seq bump (= "generation" change).
 *   free_queue:  SPSC ring of recycled ChipSwimlaneAicoreTaskBuffer*; host pushes,
 *                AICPU pops when rotating.
 *
 * AICore records flow through the existing per-thread ready_queue in
 * ChipSwimlaneDataHeader (with ReadyQueueEntry::kind = AicoreTask). This keeps
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
 * `KernelArgs::chip_swimlane_aicore_rotation_table[block_idx]` during
 * `chip_swimlane_aicpu_init`, and AICore lazy-resolves it via
 * `get_chip_swimlane_aicore_head()`.
 */
struct ChipSwimlaneAicoreTaskPool {
    ChipSwimlaneActiveHead head;       // 64B
    ChipSwimlaneFreeQueue free_queue;  // 128B
} __attribute__((aligned(64)));

static_assert(sizeof(ChipSwimlaneAicoreTaskPool) == 192, "ChipSwimlaneAicoreTaskPool must be 192 bytes");
// ABI lock: `&pool.head` is what AICPU publishes into the rotation_table for
// AICore to dcci. Must stay at offset 0 so AICore can index from KernelArgs.
static_assert(offsetof(ChipSwimlaneAicoreTaskPool, head) == 0, "ChipSwimlaneAicoreTaskPool::head must be at offset 0");
static_assert(
    offsetof(ChipSwimlaneAicoreTaskPool, free_queue) == 64,
    "ChipSwimlaneAicoreTaskPool::free_queue must be at offset 64"
);

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/** Buffer kind for ReadyQueueEntry::kind. uint32_t underlying. */
enum class ChipSwimlaneBufferKind : uint32_t {
    AicpuTask = 0,        // Per-core ChipSwimlaneAicpuTaskBuffer, AICPU writes
    AicpuSchedPhase = 1,  // Per-thread ChipSwimlaneAicpuSchedPhaseBuffer, AICPU writes
    AicpuOrchPhase = 2,   // Per-thread ChipSwimlaneAicpuOrchPhaseBuffer, AICPU writes
    AicoreTask = 3,       // Per-core ChipSwimlaneAicoreTaskBuffer, AICore writes, AICPU enqueues at rotation
};

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, the producer (AICPU for AICPU
 * task / sched-phase / orch-phase, AICPU on behalf of AICore for AicoreTask)
 * pushes this entry. Host memory manager retrieves entries from the queue.
 */
struct ReadyQueueEntry {
    uint32_t core_index;          // Core index (0 ~ num_cores-1), or thread_idx for phase entries
    ChipSwimlaneBufferKind kind;  // Buffer kind discriminator (uint32_t underlying)
    uint64_t buffer_ptr;          // Device pointer to the full buffer
    uint32_t buffer_seq;          // Sequence number for ordering
    uint32_t pad;                 // Alignment padding
} __attribute__((aligned(32)));

// =============================================================================
// ChipSwimlaneDataHeader - Fixed Header
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
struct ChipSwimlaneDataHeader {
    // Per-thread ready queues (FIFO Circular Buffers)
    // Each AICPU thread has its own queue to avoid lock contention
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PROF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Consumer read positions (Host modifies)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // Producer write positions (AICPU modifies)

    // Metadata (Host initializes, Device read-only)
    uint32_t num_cores;            // Actual number of cores launched
    uint32_t chip_swimlane_level;  // 0=off, 1=AICore timing, 2=+Scheduler task timing,
                                   // 3=+sched phases, 4=+orch phases. Host writes
                                   // at init; AICPU reads in chip_swimlane_aicpu_init.

    // Phase profiling metadata (AICPU writes in chip_swimlane_aicpu_init_phase;
    // Host reads at drain time). Both counts == 0 means phase profiling was not
    // initialized. Gated by chip_swimlane_level >= SCHED_PHASES at write time.
    // num_sched_phase_threads counts the active scheduler threads (sched-phase
    // pools are per scheduler thread, indexed by thread id). Orchestration is
    // single-threaded, so orch-phase is a single instance: num_orch_phase_threads
    // == 1 and records land in orch pool ordinal 0 (dep_gen / scope_stats style),
    // regardless of which AICPU thread the orchestrator runs on.
    uint32_t num_sched_phase_threads;           // Number of sched-phase pools the AICPU initialized
    uint32_t num_orch_phase_threads;            // Number of orch-phase pools the AICPU initialized
    uint32_t num_phase_cores;                   // Number of valid entries in core_to_thread (0 = unset)
    int8_t core_to_thread[PLATFORM_MAX_CORES];  // core_id → scheduler thread index (-1 = unassigned)

    // DFX backpressure coordination (unified across all DFX subsystems).
    DfxBackpressureHeader backpressure;
} __attribute__((aligned(64)));

// ABI lock for the merged header. The phase metadata fields and the
// core_to_thread[] array are read by both host and AICPU .so's; silent
// layout drift between them is undetectable at runtime (no magic gate
// anymore). Mirrors the pool-layout asserts in #939.
static_assert(
    offsetof(ChipSwimlaneDataHeader, num_sched_phase_threads) ==
        offsetof(ChipSwimlaneDataHeader, chip_swimlane_level) + sizeof(uint32_t),
    "ChipSwimlaneDataHeader: num_sched_phase_threads must follow chip_swimlane_level"
);
static_assert(
    offsetof(ChipSwimlaneDataHeader, num_orch_phase_threads) ==
        offsetof(ChipSwimlaneDataHeader, num_sched_phase_threads) + sizeof(uint32_t),
    "ChipSwimlaneDataHeader: num_orch_phase_threads must follow num_sched_phase_threads"
);
static_assert(
    offsetof(ChipSwimlaneDataHeader, core_to_thread) ==
        offsetof(ChipSwimlaneDataHeader, num_phase_cores) + sizeof(uint32_t),
    "ChipSwimlaneDataHeader: core_to_thread[] must follow num_phase_cores"
);
static_assert(sizeof(ChipSwimlaneDataHeader) % 64 == 0, "ChipSwimlaneDataHeader must be 64-byte aligned");

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

/**
 * AICPU orchestrator phase record (32 bytes).
 *
 * One record per submit. No kind field — only one orch event type exists
 * (per-submit envelope); if a second is ever needed, add a kind field then.
 */
struct ChipSwimlaneAicpuOrchPhaseRecord {
    uint64_t start_time;  // Submit start timestamp
    uint64_t end_time;    // Submit end timestamp
    uint64_t task_id;     // TaskId::raw, in the minting runtime's layout
    uint32_t submit_idx;  // Monotonic submit counter
    uint32_t _pad;        // 32B alignment padding
};
static_assert(sizeof(ChipSwimlaneAicpuOrchPhaseRecord) == 32, "ChipSwimlaneAicpuOrchPhaseRecord layout drift");

/**
 * Host phase record (40 bytes).
 *
 * One record per timed host operation on host_build_graph's prepare path —
 * both the bind stage's segments and the host orchestrator's submit-level
 * operations, distinguished by `kind`. Timestamps are host monotonic
 * nanoseconds, the clock the `[STRACE]` host spans use, so records and spans
 * read against each other with no alignment step.
 *
 * `payload` is kind-discriminated: a task id for the kinds that submit a task
 * (see host_phase_kind_submits_task), otherwise a per-kind detail count such as
 * a byte or in-graph task count. Readers must consult `kind` before interpreting it.
 */
struct HostPhaseRecord {
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t payload;
    uint32_t kind;       // HostPhaseKind
    uint32_t index;      // submit_idx for orchestrator operations, 0 for bind segments
    uint32_t thread_id;  // OS tid when available, stable producer id otherwise
    uint32_t _pad;
};
static_assert(sizeof(HostPhaseRecord) == 40, "HostPhaseRecord layout drift");

/**
 * What one HostPhaseRecord measured — the kinds themselves live in
 * common/host_phase_kind.h, so the AICPU build and the orchestration .so can name
 * them without taking this header's dependencies. The predicates below are
 * host-only and stay here.
 */

/**
 * Whether this kind ends with a task submitted to the runtime, i.e. whether its
 * record's `payload` is a task id and whether it counts towards the bind's
 * total_tasks. The three that do are what a per-submit consumer sees; the rest
 * are sub-operations of a submit, or bind work with no task at all.
 */
inline bool host_phase_kind_submits_task(HostPhaseKind kind) {
    return kind == HostPhaseKind::OrchSubmitTask || kind == HostPhaseKind::OrchAllocTensors ||
           kind == HostPhaseKind::OrchGraphSubmit;
}

/**
 * Whether this kind is a host-to-device transfer, i.e. host work that has to
 * finish before the device can start on what it produced.
 *
 * These are the bind segments a device timeline can say something about: drawn
 * beside the device lanes they show the handover, where the rest of the bind
 * stage is host-only setup with no device counterpart.
 */
inline bool host_phase_kind_is_device_upload(HostPhaseKind kind) {
    return kind == HostPhaseKind::BindGraphUpload || kind == HostPhaseKind::BindArenaH2d;
}

/**
 * Whether a kind's `detail` is a quantity, i.e. whether summing it across a bind
 * means anything.
 *
 * Most kinds put in `detail` the identity of what their interval measured — a
 * task id, a Graph key, the submission index — and a sum over identities is a
 * number no reader can act on. Only the kinds that carry a count get the summed
 * column in the breakdown; the rest print their count and total alone.
 */
inline bool host_phase_kind_detail_is_quantity(HostPhaseKind kind) {
    return kind == HostPhaseKind::OrchBuildDefinition || kind == HostPhaseKind::OrchRecordingWait;
}

inline const char *host_phase_kind_name(HostPhaseKind kind) {
    switch (kind) {
    case HostPhaseKind::BindArgs:
        return "args";
    case HostPhaseKind::BindArenaBuild:
        return "arena_build";
    case HostPhaseKind::BindStaticArena:
        return "static_arena";
    case HostPhaseKind::BindGmHeap:
        return "gm_heap";
    case HostPhaseKind::BindSharedMem:
        return "shared_mem";
    case HostPhaseKind::BindRuntimeInit:
        return "runtime_init";
    case HostPhaseKind::BindHostOrch:
        return "host_orch";
    case HostPhaseKind::BindGraphUpload:
        return "graph_upload";
    case HostPhaseKind::BindArenaH2d:
        return "arena_h2d";
    case HostPhaseKind::BindHostViewClose:
        return "host_view_close";
    case HostPhaseKind::OrchSubmitTask:
        return "submit_task";
    case HostPhaseKind::OrchAllocTensors:
        return "alloc_tensors";
    case HostPhaseKind::OrchRecordInGraphTask:
        return "record_in_graph_task";
    case HostPhaseKind::OrchGraphSubmit:
        return "graph_submit";
    case HostPhaseKind::OrchBuildDefinition:
        return "build_definition";
    case HostPhaseKind::OrchGraphBegin:
        return "graph_begin";
    case HostPhaseKind::OrchRecordingWait:
        return "recording_wait";
    case HostPhaseKind::OrchGraphCommit:
        return "graph_commit";
    case HostPhaseKind::OrchSubmitAdmit:
        return "submit_admit";
    case HostPhaseKind::OrchRecordHandoff:
        return "record_handoff";
    case HostPhaseKind::OrchGeneratedArgs:
        return "generated_args";
    case HostPhaseKind::Count:
        break;
    }
    return "unknown";
}

constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;  // 1 MiB per sched thread, 512 KiB per orch thread

// Fixed-size phase record buffers. Same TypedBuffer template as the task
// buffers — keeps the drain machinery uniform.
using ChipSwimlaneAicpuSchedPhaseBuffer =
    TypedBuffer<ChipSwimlaneAicpuSchedPhaseRecord, PLATFORM_PHASE_RECORDS_PER_THREAD>;
using ChipSwimlaneAicpuOrchPhaseBuffer =
    TypedBuffer<ChipSwimlaneAicpuOrchPhaseRecord, PLATFORM_PHASE_RECORDS_PER_THREAD>;

// Sched and orch phase pools share the same head+free_queue layout as
// AicpuTaskPool; the buffer payload type differs but the pool plumbing is
// identical. Aliasing keeps the drain machinery polymorphic.
using ChipSwimlaneAicpuSchedPhasePool = ChipSwimlaneAicpuTaskPool;
using ChipSwimlaneAicpuOrchPhasePool = ChipSwimlaneAicpuTaskPool;

// The host phase pool holds the same fixed-size record buffers over host DDR
// instead of device shared memory, but not the head + free_queue plumbing: those
// exist so a single device producer can hand filled buffers to a host reader
// mid-run, and the host pool has many concurrent producers and one reader that
// only ever looks after the bind is closed. It is defined in
// host/host_phase_records.h, which owns that host-only shape; the buffers are
// smaller and fewer than a device thread's because a producer emits hundreds of
// records per bind rather than tens of thousands (see
// PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER).
using HostPhaseRecordBuffer = TypedBuffer<HostPhaseRecord, PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER>;

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
 *                     = sizeof(ChipSwimlaneDataHeader) + num_cores × sizeof(ChipSwimlaneAicpuTaskPool)
 *
 * @param num_cores Number of cores (block_dim × PLATFORM_CORES_PER_BLOCKDIM)
 * @return Total bytes for header + buffer states
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(ChipSwimlaneDataHeader) + num_cores * sizeof(ChipSwimlaneAicpuTaskPool);
}

/**
 * Get header pointer
 *
 * @param base_ptr Shared memory base address (device_ptr or host_ptr)
 * @return ChipSwimlaneDataHeader pointer
 */
inline ChipSwimlaneDataHeader *get_chip_swimlane_header(void *base_ptr) {
    return reinterpret_cast<ChipSwimlaneDataHeader *>(base_ptr);
}

/**
 * Get ChipSwimlaneAicpuTaskPool array start address
 *
 * @param base_ptr Shared memory base address
 * @return ChipSwimlaneAicpuTaskPool array pointer
 */
inline ChipSwimlaneAicpuTaskPool *get_perf_buffer_states(void *base_ptr) {
    return reinterpret_cast<ChipSwimlaneAicpuTaskPool *>(
        reinterpret_cast<char *>(base_ptr) + sizeof(ChipSwimlaneDataHeader)
    );
}

/**
 * Get ChipSwimlaneAicpuTaskPool for specified core
 *
 * @param base_ptr Shared memory base address
 * @param core_index Core index (0 ~ num_cores-1)
 * @return ChipSwimlaneAicpuTaskPool pointer
 */
inline ChipSwimlaneAicpuTaskPool *get_perf_buffer_state(void *base_ptr, int core_index) {
    return &get_perf_buffer_states(base_ptr)[core_index];
}

/**
 * Calculate total memory size including AICore states and both phase
 * profiling regions (buffer states only, not the record payloads themselves).
 *
 * Layout (after the fixed ChipSwimlaneDataHeader, which carries the phase
 * metadata fields):
 *   [ChipSwimlaneAicpuTaskPool       × PLATFORM_MAX_CORES]
 *   [ChipSwimlaneAicoreTaskPool      × PLATFORM_MAX_CORES]
 *   [ChipSwimlaneAicpuSchedPhasePool × PLATFORM_MAX_AICPU_THREADS]
 *   [ChipSwimlaneAicpuOrchPhasePool  × PLATFORM_MAX_AICPU_THREADS]
 *
 * Every array is dimensioned by a platform maximum, not by the run. The host
 * and the AICPU both address this region, and each used to supply its own core
 * count to locate the arrays — the host the run's, the device its worker_count.
 * A single basis they cannot disagree about is the point: the pool *states* are
 * a fixed grid, and only the record buffers hanging off them are allocated per
 * run. header->num_cores therefore still means "cores this run uses" and is not
 * an addressing input.
 *
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases() {
    return calc_perf_data_size(PLATFORM_MAX_CORES) + PLATFORM_MAX_CORES * sizeof(ChipSwimlaneAicoreTaskPool) +
           PLATFORM_MAX_AICPU_THREADS * sizeof(ChipSwimlaneAicpuSchedPhasePool) +
           PLATFORM_MAX_AICPU_THREADS * sizeof(ChipSwimlaneAicpuOrchPhasePool);
}

/**
 * Get ChipSwimlaneAicoreTaskPool array start address (located immediately
 * after the ChipSwimlaneAicpuTaskPool array).
 */
inline ChipSwimlaneAicoreTaskPool *get_aicore_buffer_states(void *base_ptr) {
    return reinterpret_cast<ChipSwimlaneAicoreTaskPool *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(PLATFORM_MAX_CORES)
    );
}

inline ChipSwimlaneAicoreTaskPool *get_aicore_buffer_state(void *base_ptr, int core_index) {
    return &get_aicore_buffer_states(base_ptr)[core_index];
}

/**
 * Get ChipSwimlaneAicpuSchedPhasePool array start address (located immediately
 * after the ChipSwimlaneAicoreTaskPool array).
 */
inline ChipSwimlaneAicpuSchedPhasePool *get_sched_phase_buffer_states(void *base_ptr) {
    return reinterpret_cast<ChipSwimlaneAicpuSchedPhasePool *>(
        reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(PLATFORM_MAX_CORES) +
        PLATFORM_MAX_CORES * sizeof(ChipSwimlaneAicoreTaskPool)
    );
}

inline ChipSwimlaneAicpuSchedPhasePool *get_sched_phase_buffer_state(void *base_ptr, int thread_idx) {
    return &get_sched_phase_buffer_states(base_ptr)[thread_idx];
}

/**
 * Get ChipSwimlaneAicpuOrchPhasePool array start address (located immediately
 * after the ChipSwimlaneAicpuSchedPhasePool array).
 *
 * Layout stride is fixed at `PLATFORM_MAX_AICPU_THREADS`, NOT the runtime
 * `num_sched_phase_threads` count. The host pre-allocates both phase pool
 * arrays at the platform max because it doesn't know the actual AICPU
 * thread count at shm-alloc time; AICPU later picks a smaller count for
 * iteration. The OFFSET must match the host's alloc layout (max), not
 * AICPU's iteration count (actual) — otherwise AICPU reads the orch array
 * from inside the (still allocated) sched array tail, corrupting both.
 */
inline ChipSwimlaneAicpuOrchPhasePool *get_orch_phase_buffer_states(void *base_ptr) {
    return reinterpret_cast<ChipSwimlaneAicpuOrchPhasePool *>(
        reinterpret_cast<char *>(get_sched_phase_buffer_states(base_ptr)) +
        PLATFORM_MAX_AICPU_THREADS * sizeof(ChipSwimlaneAicpuSchedPhasePool)
    );
}

inline ChipSwimlaneAicpuOrchPhasePool *get_orch_phase_buffer_state(void *base_ptr, int thread_idx) {
    return &get_orch_phase_buffer_states(base_ptr)[thread_idx];
}

#ifdef __cplusplus
}
#endif
