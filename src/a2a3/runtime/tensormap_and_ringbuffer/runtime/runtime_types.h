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
 * tensormap_and_ringbuffer Runtime - Core Type Definitions
 *
 * This header defines the fundamental tensormap_and_ringbuffer runtime types:
 * - Configuration constants
 * - Worker types and task states
 * - simpler::tmr::Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "profiling_config.h"
#include "constants.h"
#include "runtime_status.h"
#include "dispatch_payload.h"
#include "aicore_completion_mailbox.h"
#include "submit_types.h"
#include "task_id.h"
#include "types.h"

// Spin-wait hint for AICPU threads.  On real hardware the AICPU has dedicated
// ARM A55 cores — no OS yield is needed, so the hint is a no-op.  In simulation
// all threads share host CPU cores, so we yield to prevent starvation.
// This header is also compiled into the Host .so (for struct definitions only),
// where the hint is never called — the fallback no-op keeps Host builds clean.
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
#include "aicpu/device_time.h"
#endif

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
// NOTE: CHIP_TASK_WINDOW_SIZE is now a per-ring default value.
// Actual window size is passed at runtime to runtime_reserve_layout().
// The slot is local_id masked by the window size, which is why the window is a power of two.
#define CHIP_TASK_WINDOW_SIZE 16384  // Default per-ring task window size (power of 2)

// Multi-ring: number of independent ring layers (HeapRing + TaskRing + DepPool per layer)
// Scope depth maps to ring index via: min(scope_depth, CHIP_MAX_RING_DEPTH - 1)
#define CHIP_MAX_RING_DEPTH 4

// Memory pools (per-ring defaults; total = value × CHIP_MAX_RING_DEPTH)
#define CHIP_HEAP_SIZE (256 * 1024 * 1024)  // 256MB per ring (1GB total)
#define CHIP_DEP_LIST_POOL_SIZE 16384       // Per-ring dependency list pool entries
#define CHIP_TENSORMAP_POOL_SIZE (65536)    // TensorMap entry pool
#define CHIP_TENSORMAP_NUM_BUCKETS 4096     // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Scope management
#define CHIP_MAX_SCOPE_DEPTH 64  // Maximum nesting depth
// Hard cap for the scope_tasks buffer. Equals the total in-flight ring slot
// budget (CHIP_TASK_WINDOW_SIZE × CHIP_MAX_RING_DEPTH): once every ring slot
// is in flight, no more tasks can ever be pushed regardless of buffer size.
// scope_tasks_push fatals on overflow rather than growing the arena-owned
// buffer (which would be UB on the arena's malloc'd backing).
#define CHIP_SCOPE_TASKS_CAP (CHIP_TASK_WINDOW_SIZE * CHIP_MAX_RING_DEPTH)

// Ready queue
#define CHIP_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Cross-thread early-dispatch candidate queue (power of two). A single wide
// producer can publish more candidates than there are physical cores, so the
// capacity tracks the publication burst rather than the simultaneously
// stageable cohort: a 128-token EP8 routed-expert layer bursts 128 gate/up MM
// candidates onto one shape, and this holds 2x that.
#define CHIP_EARLY_DISPATCH_QUEUE_SIZE 256

// Fanin storage
#define CHIP_FANIN_INLINE_CAP 64

// Bounded WAIT-ancestor reachability window, in tasks. One native word; the
// shift-merge in reduce_wait_edges relies on WAIT_REACH_WINDOW == 64 (a
// d == WAIT_REACH_WINDOW shift would be undefined behavior, and every ancestor
// of such a producer is already outside the window anyway).
inline constexpr int WAIT_REACH_WINDOW = 64;

// Dependency-degree diagnostic: log once at debug level when a task's fanin or
// a producer's fanout first exceeds this degree, so dense dependency graphs can
// be inspected without adding noise to normal runtime logs.
#define CHIP_DEP_DEGREE_DEBUG_THRESHOLD 16

// TensorMap cleanup interval
#define CHIP_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks
#define CHIP_DEP_POOL_CLEANUP_INTERVAL 64   // Cleanup every N retired tasks

// get_tensor_data/set_tensor_data spin-wait timeout, expressed in time. The cycle
// count (TENSOR_DATA_TIMEOUT_CYCLES) is derived from this in runtime_core.cpp
// — its only user — by scaling with the platform counter frequency, like
// SCHEDULER_TIMEOUT_CYCLES, so it reaps at the same wall-clock on every arch (a
// fixed raw cycle count would be 15 s on a5 at 1 GHz but 300 s on a2a3 at 50 MHz).
// PLATFORM_PROF_SYS_CNT_FREQ is deliberately NOT pulled into this header: it is
// included by orchestrations that define that constant locally, so doing so caused
// a redefinition conflict. See issue #1189.
constexpr uint64_t TENSOR_DATA_TIMEOUT_MS = 15000;  // 15 s

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 *
 * State transitions:
 *   PENDING -> COMPLETED -> CONSUMED
 *
 * The slot stays in PENDING from submit through "ready in queue" and "running
 * on a worker"; readiness and running-vs-idle are derived from fanin_refcount
 * and per-core running_slot_state respectively, not from task_state itself.
 *
 * Conditions:
 *   PENDING->COMPLETED:   all subtasks finish (set by scheduler) or task is a
 *                         hidden alloc completed inline by the orchestrator
 *   COMPLETED->CONSUMED:  fanout_refcount == fanout_count && state == COMPLETED
 */
typedef enum {
    CHIP_TASK_PENDING = 0,    // Submitted; awaiting fanin, queued, or dispatched
    CHIP_TASK_COMPLETED = 1,  // Execution finished, output may still be in use
    CHIP_TASK_CONSUMED = 2    // Output fully consumed, buffers can be released
} ChipTaskState;

/**
 * Result of a unified task allocation.
 */
struct TaskAllocResult {
    int32_t task_id;    // Absolute task ID (not wrapped)
    int32_t slot;       // task_id & (window_size - 1)
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const { return task_id < 0; }
};

struct OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

// =============================================================================
// Dependency List Entry
// =============================================================================

/**
 * Fanin spill entry
 * Stored in the dedicated fanin spill ring buffer.
 */
struct ChipTaskSlotState;  // Forward declaration
struct FaninPool;          // Forward declaration

// One fanin edge: a producer slot pointer with the per-edge DepFlags packed into
// bits 0..1 of the pointer. ChipTaskSlotState is alignas(64), so bits 0..5 are
// always zero in a real pointer. Used for both the inline fanin array and the
// spill pool, keeping sizeof == sizeof(uintptr_t) so neither footprint grows.
struct FaninSpillEntry {
    static constexpr uintptr_t FLAG_MASK = 0x3;
    // No in-class initializer: the type stays trivially default-constructible so
    // the payload's fanin_inline_edges[64] and the orchestrator's per-submit
    // builder array are default-initialized at zero cost (only entries [0, count)
    // are written via set()). Value-init (`FaninSpillEntry{}`) still zeroes it.
    uintptr_t packed;

    ChipTaskSlotState *slot_state() const { return reinterpret_cast<ChipTaskSlotState *>(packed & ~FLAG_MASK); }
    DepFlags flags() const { return static_cast<DepFlags>(packed & FLAG_MASK); }
    // Only bits within FLAG_MASK are stored; any bit outside it (a malformed
    // DepFlags value) is masked off so it can never corrupt the slot pointer.
    void set(ChipTaskSlotState *s, DepFlags f) {
        packed = reinterpret_cast<uintptr_t>(s) | (static_cast<uintptr_t>(f) & FLAG_MASK);
    }
    void add_flags(DepFlags f) { packed |= (static_cast<uintptr_t>(f) & FLAG_MASK); }
    void clear() { packed = 0; }
};
static_assert(sizeof(FaninSpillEntry) == sizeof(uintptr_t));

// Per-slot frozen WAIT-ancestor reachability. `ancestors` bit i set means the
// task with submit_seq == seq - i - 1 has a WAIT path to this slot's task;
// published once at the owning submit, immutable for the slot's task
// generation. `seq` is the global submission sequence of that task. Both
// fields are written by the orchestrator thread only.
struct WaitReachEntry {
    uint64_t ancestors;
    uint64_t seq;
};
static_assert(sizeof(WaitReachEntry) == 16, "WaitReachEntry is one 16B stride");

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer.
 */
struct DepListEntry {
    ChipTaskSlotState *slot_state;  // Consumer slot state (direct pointer)
    DepListEntry *next;             // next entry
};

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in ChipTaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct TaskDescriptor {
    // Mixed-task identification (encodes ring_id in upper 32 bits)
    TaskId task_id;  // raw: (ring_id << 32) | local_id, see task_id_encoding.h

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// A 4-byte alignment pad follows kernel_id[3]; the scheduler and shared-memory
// ABI depend on the descriptor size and packed_buffer_base offset staying fixed.
static_assert(sizeof(TaskDescriptor) == 40, "TaskDescriptor size is part of the shared-memory ABI");
static_assert(offsetof(TaskDescriptor, packed_buffer_base) == 24, "packed_buffer_base offset must be unchanged");

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

// Early-dispatch claim states for TaskPayload::early_dispatch_state.
enum EarlyDispatchState : uint8_t {
    EARLY_DISPATCH_NONE = 0,       // not pre-staged
    EARLY_DISPATCH_STAGING = 1,    // Hook 1 claimed it; staging in progress
    EARLY_DISPATCH_STAGED = 2,     // reserved
    EARLY_DISPATCH_DISPATCHED = 3  // producers released; staged blocks may still be gated
};

enum EarlyDispatchLaunchState : uint8_t {
    EARLY_DISPATCH_LAUNCH_NONE = 0,
    EARLY_DISPATCH_LAUNCH_RINGING = 1,
    EARLY_DISPATCH_LAUNCH_COMPLETE = 2,
};

enum EarlySyncDrainState : uint8_t {
    EARLY_SYNC_DRAIN_NONE = 0,
    EARLY_SYNC_DRAIN_OWNER = 1 << 0,
    EARLY_SYNC_DRAIN_ARMED = 1 << 1,
    EARLY_SYNC_DRAIN_READY = 1 << 2,
    EARLY_SYNC_DRAIN_COMPLETE = 1 << 3,
};

// A pre-staged consumer occupies one core per gated subtask block. WHICH cores
// it occupies is recorded as a bitmask (staged_core_mask, 1 bit per global
// core_id); the completion-path release iterates the set bits and rings each
// core's doorbell from the scheduler's per-core doorbell table. Bounded by the
// chip's core count (RUNTIME_MAX_WORKER = 72; no two-level pre-dispatch means
// gated cores in flight <= core count), NOT by block_num — so a wide SPMD
// consumer can pre-stage all its idle cores. 2 words = 128 bits >= 72.
inline constexpr int EARLY_DISPATCH_CORE_MASK_WORDS = 2;

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + inline fanin packed in the first 9 cache lines, followed
 * by bulk tensor and scalar data. Small fanins stay fully inline; larger
 * fanins spill into a per-ring ring buffer slice.
 */
struct TaskPayload {
    // === Cache lines 0-8 (576B) — metadata + inline fanin ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_actual_count{0};  // Actual fanin count (without the +1 redundance)
    int32_t fanin_spill_start{0};   // Linear start index in fanin spill pool (0 = no spill)
    FaninPool *fanin_spill_pool{nullptr};
    // Inline fanin edges (producer slot + packed DepFlags). Spill beyond
    // CHIP_FANIN_INLINE_CAP goes to fanin_spill_pool. Same packed layout as the
    // spill entries, so the array footprint is unchanged.
    FaninSpillEntry fanin_inline_edges[CHIP_FANIN_INLINE_CAP];
    // Early-dispatch metadata (AICPU-side only). Ordered by descending
    // alignment so the block packs without internal padding. Cache line 8
    // contains the rarely-touched fanin tail rather than the hot tensor/scalar
    // counts, and the block fits between the fanin array and the 64B-aligned
    // predicate without changing the payload layout.
    //
    // Concurrent stagers publish global core-id bits with fetch_or. A regular
    // consumer destructively splits those bits between release and late-stager
    // owners; a sync_start cohort keeps the completed mask stable for one launch
    // owner across both local staging and the global-drain fallback.
    std::atomic<uint64_t> staged_core_mask[EARLY_DISPATCH_CORE_MASK_WORDS]{};
    // Candidate detection is the event-driven dual of fanin_refcount. Wiring
    // seeds producers already complete, and flagged producers increment the
    // count only after all logical blocks are launch-visible. Equality with
    // fanin_wait_count (only DEP_WAIT producers link onto fanout_head and bump
    // it) makes the consumer eligible for early dispatch.
    std::atomic<int32_t> dispatch_fanin{0};  // CONSUMER side: fully-published + pre-completed producers
    // Claimed-but-unpublished blocks are not launch-visible. Seq_cst updates
    // pair with early_dispatch_state so final publication cannot be lost when
    // producer release races a stager.
    std::atomic<int16_t> published_block_count{0};
    // Shared stager/release claim state. STAGING remains stable while multiple
    // AICPU threads claim blocks and publish mask bits; release changes it to
    // DISPATCHED. Regular consumers split mask ownership, while sync_start
    // consumers preserve the mask for rendezvous counting and one launch pass.
    std::atomic<uint8_t> early_dispatch_state{0};
    // COMPLETE is published only after every owned doorbell is visible, keeping
    // fanout private until all gated blocks have launched.
    std::atomic<uint8_t> early_dispatch_launch_state{EARLY_DISPATCH_LAUNCH_NONE};
    // Number of this sync_start task's gated cores occupying RUNNING slots.
    // Counting cores is shape-agnostic for MIX, whose cluster cores promote
    // independently. Doorbells launch only after this count matches the staged
    // mask and producer release is visible. Unused for non-sync_start tasks.
    std::atomic<int16_t> running_slot_count{0};
    // OWNER persists through ARMED and COMPLETE for one task lifetime. READY
    // records that producer release observed the owner; only cancellation clears
    // ownership before payload reinitialization.
    std::atomic<uint8_t> early_sync_drain_state{EARLY_SYNC_DRAIN_NONE};
    // Set when transitive reduction cleared DEP_WAIT on an edge whose producer
    // does not allow early resolve. Such a producer never propagates
    // dispatch_fanin, so while it was in fanin_wait_count the count was
    // permanently short and this task could not early-dispatch. Reduction takes
    // it out of that count, so the unreachable unit is carried here instead --
    // see early_dispatch_target(). Occupies alignment padding ahead of
    // fanin_wait_count, so the payload layout is unchanged.
    uint8_t early_dispatch_blocked{0};
    // Number of DEP_WAIT fanin edges (readiness-bearing), <= fanin_actual_count.
    // fanin_actual_count is the TOTAL edge count that for_each iteration and
    // on_task_release walk (including RETAIN-only and reduction-dropped edges);
    // this is the WAIT-edge count the early-dispatch threshold compares
    // dispatch_fanin against. They differ once transitive reduction produces
    // RETAIN-only or dropped edges that carry no DEP_WAIT. Occupies the padding
    // between the early-dispatch block and the 64B-aligned predicate, so the
    // payload layout is unchanged.
    int32_t fanin_wait_count{0};
    // The count dispatch_fanin must hit for this task to be an early-dispatch
    // candidate. Only producers linked onto a fanout_head ever bump
    // dispatch_fanin, so it can reach at most fanin_wait_count: a nonzero
    // early_dispatch_blocked therefore makes this target permanently
    // unreachable, which is exactly what the reduced-away producer did while it
    // was still counted.
    int32_t early_dispatch_target() const { return fanin_wait_count + early_dispatch_blocked; }
    // === Cache line 9 (byte 576) — dispatch predicate (AICPU-only) ===
    // Offset is a fixed 576, independent of MAX_TENSOR_ARGS / MAX_SCALAR_ARGS.
    // AICore never reads it — args are materialized from the tensor_count / tensors
    // / scalars offsets only. Resolved at submit; evaluated by the scheduler at
    // dispatch.
    alignas(64) DispatchPredicate predicate;
    // === Cache lines 10-73 (4096B) — tensors (alignas(64) forces alignment) ===
    simpler::tmr::Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 74-75 (128B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    // Layout verification (size checks that don't need offsetof).
    static_assert(sizeof(simpler::tmr::Tensor) == 128, "simpler::tmr::Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == 128, "scalar region must be 128B (2 cache lines)");

    // Argument region access. Shared AICPU code (args_dump_aicpu.h) reads a payload's
    // arguments through these, so both runtimes offer them whether the arguments sit
    // inline as they do here or in a pool.
    simpler::tmr::Tensor *tensor_data() { return tensors; }
    const simpler::tmr::Tensor *tensor_data() const { return tensors; }
    uint64_t *scalar_data() { return scalars; }
    const uint64_t *scalar_data() const { return scalars; }

    /**
     * Prefetch for write the regions init() fills. tensor_count/scalar_count
     * come from the argument because the payload counts are not initialized
     * yet. The touched range also warms the early-dispatch metadata line.
     */
    void prefetch(int32_t tensor_count, int32_t scalar_count) const {
        for (int32_t i = 0; i < tensor_count; i++) {
            __builtin_prefetch(&tensors[i], 1, 3);
            __builtin_prefetch(reinterpret_cast<const char *>(&tensors[i]) + 64, 1, 3);
        }
        for (int32_t i = 0; i < scalar_count; i += 8) {
            __builtin_prefetch(&scalars[i], 1, 3);
        }
        __builtin_prefetch(this, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 64, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 128, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 512, 1, 3);  // early-dispatch fields (cache line 8)
    }

    /**
     * Initialize payload: copy tensors, store scalars.
     *
     * For each param slot, the tensor source is determined by TensorArgType:
     * - OUTPUT -> use materialized_outputs.output_ptr(out_idx++)
     * - INPUT / INOUT -> use refs[i].tensor
     *
     * @param args                Task arguments (tensors + scalars)
     * @param result  Materialized output tensors (from TensorCreateInfo path)
     */
    void
    init(const CoreTaskArgs &args, TaskOutputTensors &result, TaskAllocResult &alloc_result, OutputLayout &layout) {
        tensor_count = args.tensor_count();
        scalar_count = args.scalar_count();

        // int32_t out_idx = 0;
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) {
                tensors[i].copy(args.tensor(i).ref());
            } else {
                init_tensor_from_create_info(
                    tensors[i], args.tensor(i).create_info(),
                    reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]),
                    layout.buffer_sizes[i]
                );
                tensors[i].owner_task_id = result.task_id();
                result.materialize_output(tensors[i]);
            }
        }
        // Round up to cache line boundary. Both arrays are 128B so no overrun.
        // Eliminates branches; extra bytes within the same CL have zero additional cost.
        memcpy(scalars, args.scalars(), CHIP_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));

        // reset_for_reuse deliberately skips this cold payload. These fields
        // are consumer-side state and must be initialized on every submit even
        // when the consumer itself does not allow early resolve, because one of
        // its flagged producers may still update them.
        early_dispatch_state.store(EARLY_DISPATCH_NONE, std::memory_order_relaxed);
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_core_mask[w].store(0, std::memory_order_relaxed);
        dispatch_fanin.store(0, std::memory_order_relaxed);
        published_block_count.store(0, std::memory_order_relaxed);
        early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        running_slot_count.store(0, std::memory_order_relaxed);
        early_sync_drain_state.store(EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
    }
};

// TaskPayload layout verification (offsetof requires complete type).
static_assert(offsetof(TaskPayload, fanin_spill_pool) == 16, "spill pool pointer layout drift");
static_assert(offsetof(TaskPayload, fanin_inline_edges) == 24, "inline fanin array must follow spill metadata");
static_assert(
    offsetof(TaskPayload, fanin_wait_count) >=
            offsetof(TaskPayload, early_sync_drain_state) + sizeof(TaskPayload::early_sync_drain_state) &&
        offsetof(TaskPayload, fanin_wait_count) + sizeof(TaskPayload::fanin_wait_count) <=
            offsetof(TaskPayload, predicate),
    "fanin_wait_count occupies the padding between the early-dispatch block and the 64B-aligned predicate"
);
static_assert(
    offsetof(TaskPayload, predicate) == 576,
    "dispatch predicate occupies cache line 9 at fixed byte 576 (before tensors, never moves)"
);
static_assert(offsetof(TaskPayload, tensors) == 640, "tensors must start at byte 640 (cache line 10, after predicate)");
static_assert(
    offsetof(TaskPayload, scalars) == 640 + MAX_TENSOR_ARGS * sizeof(simpler::tmr::Tensor),
    "scalars must immediately follow tensors"
);
static_assert(
    sizeof(TaskPayload) == 640 + MAX_TENSOR_ARGS * sizeof(simpler::tmr::Tensor) + MAX_SCALAR_ARGS * sizeof(uint64_t),
    "TaskPayload size = metadata(576) + predicate cache line(64) + tensors + scalars"
);

/**
 * Per-task slot scheduling state (scheduler-private, NOT in shared memory)
 *
 * Consolidates all hot-path scheduling fields into a single cache-friendly
 * structure (64 bytes = one cache line). Accessing any field of a task's
 * slot state brings all related fields into the same cache line.
 *
 * Concurrency notes:
 * - fanout_head, fanout_count protected by fanout_lock (per-task spinlock)
 * - fanin_count set once at submission, read-only after (hot path for ready check)
 * - task_state, fanin_refcount, fanout_refcount updated atomically
 */

// fanout_count / fanout_refcount bit encoding (both uint32):
//   bits [30:0] = consumer references (count: # consumers; refcount: # released)
//   bit  [31]   = the owning scope's reference (FANOUT_SCOPE_BIT)
// fanout_count is seeded to FANOUT_SCOPE_BIT and ++'d per consumer, so it
// ends as (SCOPE_BIT | num_consumers). release adds 1 (consumer completion) or
// SCOPE_BIT (scope_end). CONSUMED iff fanout_refcount == fanout_count (every
// consumer released AND scope bit set). Keeping the scope ref in a distinct bit
// (rather than folding scope + consumers into one count) lets a consumer reach
// fanout_refcount == (fanout_count & ~FANOUT_SCOPE_BIT) while the scope bit
// is still unset -- i.e. "all consumers done but scope still open" stays
// distinguishable from "fully consumed". Structural allocation deadlock is
// detected separately by checking whether the blocking ring head is the oldest
// task pinned by any open scope on that ring.
static constexpr uint32_t FANOUT_SCOPE_BIT = 0x80000000u;

enum TaskLifecycleFlag : uint8_t {
    LIFECYCLE_FLAGS_NONE = 0,
    READY_CLAIMED = 1U << 0,
    COMPLETION_DONE = 1U << 1,
    // Deferred-completion discriminator packed into lifecycle_flags so the
    // slot remains 64B. Call sites use mark_any_subtask_deferred() and
    // has_any_subtask_deferred().
    SUBTASK_DEFERRED = 1U << 2,
    DISPATCH_PROPAGATED = 1U << 3,
};

static_assert((DISPATCH_PROPAGATED & (READY_CLAIMED | COMPLETION_DONE | SUBTASK_DEFERRED)) == 0);

struct alignas(64) ChipTaskSlotState {
    // Fanout lock + list (accessed together under lock in on_task_complete)
    std::atomic<int32_t> fanout_lock;  // Per-task spinlock (0=unlocked, 1=locked)
    uint32_t fanout_count;             // SCOPE_BIT (owning scope) | number of consumers

    DepListEntry *fanout_head;  // Pointer to first fanout entry (nullptr = empty)

    // Task state (completion, consumed check, ready check)
    std::atomic<ChipTaskState> task_state;  // PENDING/COMPLETED/CONSUMED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;  // Dynamic: counts completed producers
    int32_t fanin_count;                  // Number of producer dependencies (set once by wiring)

    // Fanout refcount (accessed with fanout_count in check_and_handle_consumed)
    std::atomic<uint32_t> fanout_refcount;  // Dynamic: low bits = released consumers, bit31 = scope released

    // --- Per-slot constant, re-bound by orch::prepare_task each submit ---
    // Value is the same on every reuse (&task_payloads[slot] / &task_descriptors[slot]),
    // but written here per-submit instead of in an O(window_size) init loop —
    // these are the only "scale-dependent" pointers in this struct, so moving
    // them out of init makes startup cost independent of task_window_size.
    TaskPayload *payload;
    TaskDescriptor *task;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Ring layer (immutable after init)
    // Single per-task attributes byte (early-dispatch hint, sync_start,
    // has_predicate, selective timing tag). Lives on slot_state (not payload) so
    // fanin walks and the completion path read them off the already-hot producer
    // slot_state cache line. Packed into the padding before dep_pool_mark to keep
    // ChipTaskSlotState at 64 bytes. Plain-write (set once at submit, before the
    // slot is scheduler-visible), so it MUST NOT share a byte with the atomically
    // mutated lifecycle_flags.
    TaskAttrs task_attrs{};
    // Concurrent lifecycle updates preserve unrelated bits. Slot reuse clears
    // the byte only after the previous task lifetime is quiescent.
    std::atomic<uint8_t> lifecycle_flags{LIFECYCLE_FLAGS_NONE};
    int32_t dep_pool_mark{0};  // Dep pool top after Orch-side wiring

    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};                // Total logical blocks (set by orchestrator)
    // Next block to dispatch. Normal dispatch and late early-dispatch stagers
    // can run concurrently after a partial staged release. All paths claim
    // ranges through claim_block_range().
    std::atomic<int16_t> next_block_idx{0};

    int32_t claim_block_range(int32_t block_limit, int32_t max_count, int32_t &start) {
        int16_t current = next_block_idx.load(std::memory_order_relaxed);
        while (current < block_limit && max_count > 0) {
            int32_t count = block_limit - current;
            if (count > max_count) count = max_count;
            int16_t desired = static_cast<int16_t>(current + count);
            if (next_block_idx.compare_exchange_weak(
                    current, desired, std::memory_order_seq_cst, std::memory_order_relaxed
                )) {
                start = current;
                return count;
            }
        }
        start = current;
        return 0;
    }

    /**
     * Bind the slot-invariant ring id. Called once per slot during
     * RingSchedState::init(); ring_id never changes across reuses.
     */
    void bind_ring(uint8_t rid) { ring_id = rid; }

    /**
     * Re-bind the per-slot payload/task pointers. Called by
     * orch::prepare_task on every submit. Value is constant for a given
     * slot, but we pay the cheap re-write each submit (both fields land on
     * the same 64B slot_state cache line that prepare_task is already
     * dirtying) to avoid the init-time per-slot loop.
     */
    void bind_buffers(TaskPayload *p, TaskDescriptor *t) {
        payload = p;
        task = t;
    }

    // Lock-free callers use this as a fast-path hint. A false result is
    // rechecked by try_mark_dispatch_propagated() while holding fanout_lock.
    bool has_dispatch_propagated() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & DISPATCH_PROPAGATED) != 0;
    }

    // The propagation owner holds fanout_lock through its fanout snapshot, so
    // wiring can classify late edges exactly once.
    bool try_mark_dispatch_propagated() {
        return (lifecycle_flags.fetch_or(DISPATCH_PROPAGATED, std::memory_order_acq_rel) & DISPATCH_PROPAGATED) == 0;
    }

    void mark_completed() {
        task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
        lifecycle_flags.fetch_or(COMPLETION_DONE, std::memory_order_release);
    }

    bool is_completion_flag_set() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & COMPLETION_DONE) != 0;
    }

    // Set by any subtask FIN that pushed deferred-completion CONDITIONs to the
    // runtime mailbox; read by the last subtask FIN to decide whether the task
    // needs MPSC-deferred completion or can complete inline on this thread. The
    // release write is sequenced before on_subtask_complete's acq_rel fetch_add;
    // the acquire read by the last subtask observes all earlier writes.
    void mark_any_subtask_deferred() { lifecycle_flags.fetch_or(SUBTASK_DEFERRED, std::memory_order_release); }

    bool has_any_subtask_deferred() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & SUBTASK_DEFERRED) != 0;
    }

    /**
     * Reset dynamic scheduling fields for slot reuse.
     * Called by advance_ring_pointers() after a slot transitions to CONSUMED
     * and last_task_alive advances past it, but before sync_to_sm() publishes
     * the new last_task_alive to the orchestrator.
     *
     * Skips payload, task, ring_id (immutable, bound once at init).
     * Skips task_state: left as CONSUMED so that wait_for_tensor_ready()
     * callers holding stale owner_task_id still observe a completed state.
     * task_state is set to PENDING by the orchestrator when it reuses the slot.
     */
    void reset_for_reuse() {
        fanout_lock.store(0, std::memory_order_relaxed);
        fanout_count = FANOUT_SCOPE_BIT;  // bit31 = owning-scope ref; consumers ++ into low bits
        fanout_head = nullptr;
        fanin_refcount.store(0, std::memory_order_relaxed);
        fanout_refcount.store(0, std::memory_order_relaxed);
        completed_subtasks.store(0, std::memory_order_relaxed);
        next_block_idx.store(0, std::memory_order_relaxed);
        lifecycle_flags.store(LIFECYCLE_FLAGS_NONE, std::memory_order_relaxed);
        // Note: active_mask and task_attrs are per-submit-constant fields
        // rewritten in prepare_task on every reuse, so they are not reset here.
        // Payload early-dispatch fields are initialized by TaskPayload::init
        // on every submit before the slot becomes scheduler-visible.
    }

    // === Per-task fanout spinlock ===
    //
    // Used by BOTH the orchestrator and the scheduler. The fanout_lock MUST
    // be held whenever reading or writing fanout_head / fanout_count, because
    // the orchestrator adds consumers concurrently with the scheduler
    // traversing the list after task completion.

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    void lock_fanout(uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;

        for (;;) {
            while (fanout_lock.load(std::memory_order_acquire) != 0) {
                contended = true;
                atomic_ops++;
                SPIN_WAIT_HINT();
            }
            int32_t expected = 0;
            if (fanout_lock.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                atomic_ops++;
                atomic_count += atomic_ops;
                if (contended) {
                    wait_cycle += (get_sys_cnt_aicpu() - t0);
                }
                return;
            }
            contended = true;
            atomic_ops++;
        }
    }
#endif

    void lock_fanout() {
        for (;;) {
            while (fanout_lock.load(std::memory_order_acquire) != 0) {
                SPIN_WAIT_HINT();
            }
            int32_t expected = 0;
            if (fanout_lock.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
        }
    }

    void unlock_fanout() { fanout_lock.store(0, std::memory_order_release); }
};

static_assert(sizeof(ChipTaskSlotState) == 64);
// FaninSpillEntry packs DepFlags into the low bits of a ChipTaskSlotState*.
// That is only lossless while the type is aligned past those tag bits.
static_assert(
    alignof(ChipTaskSlotState) > FaninSpillEntry::FLAG_MASK,
    "ChipTaskSlotState alignment must exceed FaninSpillEntry::FLAG_MASK so the packed DepFlags bits are free"
);
