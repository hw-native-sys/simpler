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
 * PTO Runtime2 - Core Type Definitions
 *
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <atomic>

#include "profiling_config.h"
#include "pto_constants.h"
#include "pto_runtime_status.h"
#include "pto2_dispatch_payload.h"
#include "aicore_completion_mailbox.h"
#include "pto_submit_types.h"
#include "pto_task_id.h"
#include "pto_types.h"

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
// NOTE: PTO2_TASK_WINDOW_SIZE is now a per-ring default value.
// Actual window size is passed at runtime to runtime_create_from_sm().
// Use pto2_task_slot(sched, task_id) for slot calculation.
#define PTO2_TASK_WINDOW_SIZE 16384  // Default per-ring task window size (power of 2)

// Multi-ring: number of independent ring layers (HeapRing + TaskRing + DepPool per layer)
// Scope depth maps to ring index via: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
#define PTO2_MAX_RING_DEPTH 4

// Memory pools (per-ring defaults; total = value × PTO2_MAX_RING_DEPTH)
#define PTO2_HEAP_SIZE (256 * 1024 * 1024)  // 256MB per ring (1GB total)
#define PTO2_DEP_LIST_POOL_SIZE 16384       // Per-ring dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE (65536)    // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096     // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64  // Maximum nesting depth
// Hard cap for the scope_tasks buffer. Equals the total in-flight ring slot
// budget (PTO2_TASK_WINDOW_SIZE × PTO2_MAX_RING_DEPTH): once every ring slot
// is in flight, no more tasks can ever be pushed regardless of buffer size.
// scope_tasks_push fatals on overflow rather than growing the arena-owned
// buffer (which would be UB on the arena's malloc'd backing).
#define PTO2_SCOPE_TASKS_CAP (PTO2_TASK_WINDOW_SIZE * PTO2_MAX_RING_DEPTH)

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Cross-thread early-dispatch work queue (power of two)
#define PTO2_EARLY_DISPATCH_QUEUE_SIZE 64

// Fanin storage
#define PTO2_FANIN_INLINE_CAP 64

// Dependency-degree diagnostic: log once at debug level when a task's fanin or
// a producer's fanout first exceeds this degree, so dense dependency graphs can
// be inspected without adding noise to normal runtime logs.
#define PTO2_DEP_DEGREE_DEBUG_THRESHOLD 16

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks
#define PTO2_DEP_POOL_CLEANUP_INTERVAL 64   // Cleanup every N retired tasks

// get_tensor_data/set_tensor_data spin-wait timeout, expressed in time. The cycle
// count (PTO2_TENSOR_DATA_TIMEOUT_CYCLES) is derived from this in pto_runtime2.cpp
// — its only user — by scaling with the platform counter frequency, like
// SCHEDULER_TIMEOUT_CYCLES, so it reaps at the same wall-clock on every arch (a
// fixed raw cycle count would be 15 s on a5 at 1 GHz but 300 s on a2a3 at 50 MHz).
// PLATFORM_PROF_SYS_CNT_FREQ is deliberately NOT pulled into this header: it is
// included by orchestrations that define that constant locally, so doing so caused
// a redefinition conflict. See issue #1189.
constexpr uint64_t PTO2_TENSOR_DATA_TIMEOUT_MS = 15000;  // 15 s

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
    PTO2_TASK_PENDING = 0,    // Submitted; awaiting fanin, queued, or dispatched
    PTO2_TASK_COMPLETED = 1,  // Execution finished, output may still be in use
    PTO2_TASK_CONSUMED = 2    // Output fully consumed, buffers can be released
} PTO2TaskState;

/**
 * Result of a unified task allocation.
 */
struct PTO2TaskAllocResult {
    int32_t task_id;    // Absolute task ID (not wrapped)
    int32_t slot;       // task_id & (window_size - 1)
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const { return task_id < 0; }
};

struct PTO2OutputLayout {
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
struct PTO2TaskSlotState;  // Forward declaration
struct PTO2FaninPool;      // Forward declaration
struct PTO2FaninSpillEntry {
    PTO2TaskSlotState *slot_state;
};
static_assert(sizeof(PTO2FaninSpillEntry) == sizeof(uintptr_t));

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer.
 */
struct PTO2DepListEntry {
    PTO2TaskSlotState *slot_state;  // Consumer slot state (direct pointer)
    PTO2DepListEntry *next;         // next entry
};

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in PTO2TaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct PTO2TaskDescriptor {
    // Mixed-task identification (encodes ring_id in upper 32 bits)
    PTO2TaskId task_id;  // raw: (ring_id << 32) | local_id

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[PTO2_SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)
};

// A 4-byte alignment pad follows kernel_id[3]; the scheduler and shared-memory
// ABI depend on the descriptor size and packed_buffer_base offset staying fixed.
static_assert(sizeof(PTO2TaskDescriptor) == 40, "PTO2TaskDescriptor size is part of the shared-memory ABI");
static_assert(offsetof(PTO2TaskDescriptor, packed_buffer_base) == 24, "packed_buffer_base offset must be unchanged");

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + inline fanin packed in the first 9 cache lines, followed
 * by bulk tensor and scalar data. Small fanins stay fully inline; larger
 * fanins spill into a per-ring ring buffer slice.
 */
// Early-dispatch claim states for PTO2TaskPayload::early_dispatch_state.
enum PTO2EarlyDispatchState : uint8_t {
    PTO2_EARLY_DISPATCH_NONE = 0,       // not pre-staged
    PTO2_EARLY_DISPATCH_STAGING = 1,    // Hook 1 claimed it; staging in progress
    PTO2_EARLY_DISPATCH_STAGED = 2,     // reserved
    PTO2_EARLY_DISPATCH_DISPATCHED = 3  // producers released; staged blocks may still be gated
};

enum PTO2EarlyDispatchLaunchState : uint8_t {
    PTO2_EARLY_DISPATCH_LAUNCH_NONE = 0,
    PTO2_EARLY_DISPATCH_LAUNCH_RINGING = 1,
    PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE = 2,
};

enum PTO2EarlySyncDrainState : uint8_t {
    PTO2_EARLY_SYNC_DRAIN_NONE = 0,
    PTO2_EARLY_SYNC_DRAIN_OWNER = 1 << 0,
    PTO2_EARLY_SYNC_DRAIN_ARMED = 1 << 1,
    PTO2_EARLY_SYNC_DRAIN_READY = 1 << 2,
    PTO2_EARLY_SYNC_DRAIN_COMPLETE = 1 << 3,
};

// A pre-staged consumer occupies one core per gated subtask block. WHICH cores
// it occupies is recorded as a bitmask (staged_core_mask, 1 bit per global
// core_id); the completion-path release iterates the set bits and rings each
// core's doorbell from the scheduler's per-core doorbell table. Bounded by the
// chip's core count (RUNTIME_MAX_WORKER = 72; no two-level pre-dispatch means
// gated cores in flight <= core count), NOT by block_num — so a wide SPMD
// consumer can pre-stage all its idle cores. 2 words = 128 bits >= 72.
inline constexpr int PTO2_EARLY_DISPATCH_CORE_MASK_WORDS = 2;

struct PTO2TaskPayload {
    // === Cache lines 0-8 (576B) — metadata + inline fanin ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_actual_count{0};  // Actual fanin count (without the +1 redundance)
    int32_t fanin_spill_start{0};   // Linear start index in fanin spill pool (0 = no spill)
    PTO2FaninPool *fanin_spill_pool{nullptr};
    PTO2TaskSlotState *fanin_inline_slot_states[PTO2_FANIN_INLINE_CAP];
    // Early-dispatch metadata (AICPU-side only). Ordered by descending
    // alignment (8B mask, 4B fanin, then 2B/1B counters and flags) so the block packs with no
    // internal padding. Kept here after the fanin array (not moved up front): on
    // cache line 8 it shares only with the rarely-touched fanin tail, whereas in
    // line 0 the early-dispatch atomics (written during staging) would false-share with
    // tensor_count/scalar_count (read by build_payload at dispatch). Fits in the 40B
    // between the fanin array (offset 536) and the 64B-aligned tensors[] (offset
    // 576), so sizeof and tensors[] are unchanged.
    //
    // Bitmask of global core_ids this consumer is pre-staged (gated) on. Concurrent
    // stagers publish bits with atomic fetch_or. A regular consumer destructively
    // splits them between release and late-stager owners; a sync_start drain keeps
    // the completed mask stable for its single cohort launch owner.
    std::atomic<uint64_t> staged_core_mask[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS]{};
    // Early-dispatch CANDIDATE detection (event-driven, dual of fanin_refcount):
    // seeded at wiring with producers already complete, then a flagged producer
    // bumps each consumer after all of its logical blocks are published.
    // dispatch_fanin == fanin_actual_count  <=>  every producer is
    // flagged-and-fully-published or was
    // pre-completed  =>  this task is an early-dispatch candidate (push early_dispatch_queues[shape]).
    std::atomic<int32_t> dispatch_fanin{0};  // CONSUMER side: fully-published + pre-completed producers
    // Number of logical blocks whose payloads and MMIO tokens are published.
    // Claimed-but-unpublished blocks do not make a producer launch-visible. Its
    // seq_cst updates pair with early_dispatch_state to avoid losing the final
    // publish vs. release wakeup for a pre-staged producer.
    std::atomic<int16_t> published_block_count{0};
    // Lock-free claim state shared by the stagers (Hook 1, possibly several AICPU
    // threads concurrently) and the completion-path release: 0=NONE, 1=STAGING,
    // 3=DISPATCHED (2=STAGED is unused now). STAGING is the STABLE gated state —
    // many threads stage blocks concurrently while it holds, each claiming a block
    // via the atomic next_block_idx and OR-ing its cores into staged_core_mask.
    // Release does STAGING->DISPATCHED. For a regular consumer it claims the current
    // mask and a late stager rings only its remaining bits. A sync_start consumer
    // preserves the mask for rendezvous counting and its single launch pass.
    std::atomic<uint8_t> early_dispatch_state{0};
    // The launch owner publishes COMPLETE only after all owned doorbells are
    // visible, keeping fanout private until every gated block has launched.
    std::atomic<uint8_t> early_dispatch_launch_state{PTO2_EARLY_DISPATCH_LAUNCH_NONE};
    // sync_start early-dispatch rendezvous: count of this task's gated CORES currently
    // occupying a RUNNING slot (staged directly to an idle core, or promoted from a
    // gated pending slot). Counted per-core (not per-block) so it is shape-agnostic: a
    // MIX block spans a cluster whose cores promote independently. A sync_start task's
    // doorbells are rung only once this reaches popcount(staged_core_mask) AND the
    // producer released, so all cores launch atomically. Unused (0) for non-sync_start.
    std::atomic<int16_t> running_slot_count{0};
    // Ownership handshake between the early sync queue and final ready routing.
    // A successful OWNER persists through ARMED and COMPLETE until payload
    // reinitialization. READY records that producer release observed OWNER;
    // only cancellation clears OWNER during the current task lifetime.
    std::atomic<uint8_t> early_sync_drain_state{PTO2_EARLY_SYNC_DRAIN_NONE};
    // === Cache line 9 (byte 576) — dispatch predicate (AICPU-only) ===
    // Offset is a fixed 576, independent of MAX_TENSOR_ARGS / MAX_SCALAR_ARGS.
    // AICore never reads it — args are materialized from the tensor_count / tensors
    // / scalars offsets only. Resolved at submit; evaluated by the scheduler at
    // dispatch.
    alignas(64) DispatchPredicate predicate;
    // === Cache lines 10-73 (4096B) — tensors (alignas(64) forces alignment) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 74-75 (128B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    // Layout verification (size checks that don't need offsetof).
    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == 128, "scalar region must be 128B (2 cache lines)");

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). Warms the early-dispatch block at
     * offset 536 (cache line 8) too. A member fn lowers to the same prefetch
     * instructions as a free function (`this` is just a register), no cache impact.
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
    void init(
        const L0TaskArgs &args, TaskOutputTensors &result, PTO2TaskAllocResult &alloc_result, PTO2OutputLayout &layout
    ) {
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
        memcpy(scalars, args.scalars(), PTO2_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));

        // Early-dispatch metadata — the single init point for these
        // fields. reset_for_reuse MUST NOT touch the payload (it runs on the
        // scheduler's advance-ring path and would pull this cold cache line across
        // structures); prepare_task only allocates/binds. prefetch() warms this
        // line (offset 512) so these writes land in warm cache.
        //
        // early_dispatch_state / staged_core_mask / dispatch_fanin are all CONSUMER-side: a
        // task whose own allow_early_resolve is false still has them touched when
        // one of ITS producers is flagged (propagate_dispatch_fanin bumps
        // dispatch_fanin and may CAS early_dispatch_state on any consumer, independent of the
        // consumer's own hint). So they MUST be zeroed here unconditionally.
        // Publication and launch fields share this same per-submit lifetime
        // and are reset here too.
        early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_core_mask[w].store(0, std::memory_order_relaxed);
        dispatch_fanin.store(0, std::memory_order_relaxed);
        published_block_count.store(0, std::memory_order_relaxed);
        early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        running_slot_count.store(0, std::memory_order_relaxed);
        early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
static_assert(offsetof(PTO2TaskPayload, fanin_spill_pool) == 16, "spill pool pointer layout drift");
static_assert(
    offsetof(PTO2TaskPayload, fanin_inline_slot_states) == 24, "inline fanin array must follow spill metadata"
);
static_assert(
    offsetof(PTO2TaskPayload, predicate) == 576,
    "dispatch predicate occupies cache line 9 at fixed byte 576 (before tensors, never moves)"
);
static_assert(
    offsetof(PTO2TaskPayload, tensors) == 640, "tensors must start at byte 640 (cache line 10, after predicate)"
);
static_assert(
    offsetof(PTO2TaskPayload, scalars) == 640 + MAX_TENSOR_ARGS * sizeof(Tensor),
    "scalars must immediately follow tensors"
);
static_assert(
    sizeof(PTO2TaskPayload) == 640 + MAX_TENSOR_ARGS * sizeof(Tensor) + MAX_SCALAR_ARGS * sizeof(uint64_t),
    "PTO2TaskPayload size = metadata(576) + predicate cache line(64) + tensors + scalars"
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
//   bit  [31]   = the owning scope's reference (PTO2_FANOUT_SCOPE_BIT)
// fanout_count is seeded to PTO2_FANOUT_SCOPE_BIT and ++'d per consumer, so it
// ends as (SCOPE_BIT | num_consumers). release adds 1 (consumer completion) or
// SCOPE_BIT (scope_end). CONSUMED iff fanout_refcount == fanout_count (every
// consumer released AND scope bit set). Keeping the scope ref in a distinct bit
// (rather than folding scope + consumers into one count) lets a consumer reach
// fanout_refcount == (fanout_count & ~PTO2_FANOUT_SCOPE_BIT) while the scope bit
// is still unset -- i.e. "all consumers done but scope still open" stays
// distinguishable from "fully consumed". The heap/task deadlock detector keys
// off exactly that complement: that condition with state==COMPLETED means the
// head can only be released by scope_end, which a blocked orchestrator can
// never reach -> provable deadlock.
static constexpr uint32_t PTO2_FANOUT_SCOPE_BIT = 0x80000000u;

enum PTO2TaskLifecycleFlag : uint8_t {
    PTO2_LIFECYCLE_FLAGS_NONE = 0,
    PTO2_READY_CLAIMED = 1U << 0,
    PTO2_COMPLETION_DONE = 1U << 1,
    PTO2_SUBTASK_DEFERRED = 1U << 2,
    PTO2_DISPATCH_PROPAGATED = 1U << 3,
};

static_assert((PTO2_DISPATCH_PROPAGATED & (PTO2_READY_CLAIMED | PTO2_COMPLETION_DONE | PTO2_SUBTASK_DEFERRED)) == 0);

struct alignas(64) PTO2TaskSlotState {
    // Fanout lock + list (accessed together under lock in on_task_complete)
    std::atomic<int32_t> fanout_lock;  // Per-task spinlock (0=unlocked, 1=locked)
    uint32_t fanout_count;             // SCOPE_BIT (owning scope) | number of consumers

    PTO2DepListEntry *fanout_head;  // Pointer to first fanout entry (nullptr = empty)

    // Task state (completion, consumed check, ready check)
    std::atomic<PTO2TaskState> task_state;  // PENDING/COMPLETED/CONSUMED

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
    PTO2TaskPayload *payload;
    PTO2TaskDescriptor *task;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Ring layer (immutable after init)
    // Single per-task attributes byte (early-dispatch hint, sync_start,
    // has_predicate, selective timing tag). Lives on slot_state (not payload) so
    // fanin walks and the completion path read them off the already-hot producer
    // slot_state cache line. Packed into the padding before dep_pool_mark to keep
    // PTO2TaskSlotState at 64 bytes. Plain-write (set once at submit, before the
    // slot is scheduler-visible), so it MUST NOT share a byte with the atomically
    // mutated lifecycle_flags.
    TaskAttrs task_attrs{};
    // Concurrent lifecycle updates preserve unrelated bits. Slot reuse clears
    // the byte only after the previous task lifetime is quiescent.
    std::atomic<uint8_t> lifecycle_flags{PTO2_LIFECYCLE_FLAGS_NONE};
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
    void bind_buffers(PTO2TaskPayload *p, PTO2TaskDescriptor *t) {
        payload = p;
        task = t;
    }

    // Lock-free callers use this only as a fast-path hint. A false result is
    // rechecked by try_mark_dispatch_propagated() while holding fanout_lock.
    bool has_dispatch_propagated() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & PTO2_DISPATCH_PROPAGATED) != 0;
    }

    // The propagation owner holds fanout_lock from this claim through its
    // fanout snapshot so wiring can classify late edges exactly once.
    bool try_mark_dispatch_propagated() {
        return (lifecycle_flags.fetch_or(PTO2_DISPATCH_PROPAGATED, std::memory_order_acq_rel) &
                PTO2_DISPATCH_PROPAGATED) == 0;
    }

    void mark_completed() {
        task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        lifecycle_flags.fetch_or(PTO2_COMPLETION_DONE, std::memory_order_release);
    }

    bool is_completion_flag_set() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & PTO2_COMPLETION_DONE) != 0;
    }

    // Set by any subtask FIN that pushed deferred-completion CONDITIONs to the
    // runtime mailbox; read by the last subtask FIN to decide whether the task
    // needs MPSC-deferred completion or can complete inline on this thread. The
    // release write is sequenced before on_subtask_complete's acq_rel fetch_add
    // and the acquire read after, so all earlier subtasks' writes are visible to
    // the last subtask.
    void mark_any_subtask_deferred() { lifecycle_flags.fetch_or(PTO2_SUBTASK_DEFERRED, std::memory_order_release); }

    bool has_any_subtask_deferred() const {
        return (lifecycle_flags.load(std::memory_order_acquire) & PTO2_SUBTASK_DEFERRED) != 0;
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
        fanout_count = PTO2_FANOUT_SCOPE_BIT;  // bit31 = owning-scope ref; consumers ++ into low bits
        fanout_head = nullptr;
        fanin_refcount.store(0, std::memory_order_relaxed);
        fanout_refcount.store(0, std::memory_order_relaxed);
        completed_subtasks.store(0, std::memory_order_relaxed);
        next_block_idx.store(0, std::memory_order_relaxed);
        lifecycle_flags.store(PTO2_LIFECYCLE_FLAGS_NONE, std::memory_order_relaxed);
        // Note: active_mask and task_attrs are per-submit-constant fields
        // rewritten in prepare_task on every reuse, so they are not reset here.
        // Note: payload early-dispatch fields (state, masks, fanin, publication count)
        // are NOT reset here — this method skips the payload by contract. They are
        // (re)initialized in PTO2TaskPayload::init on every submit, before the slot
        // becomes visible to the scheduler.
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

static_assert(sizeof(PTO2TaskSlotState) == 64);

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_RUNTIME2_TYPES_H_
