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

#ifndef SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_

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

// Two physical arenas let O build graph N+1 while S executes graph N.
inline constexpr int32_t PTO2_REPLAY_GRAPH_BUFFER_COUNT = 2;

enum class PTO2ReplayGraphBufferState : int32_t {
    FREE = 0,
    BUILDING = 1,
    RUNNING = 2,
    DONE = 3,
};

struct PTO2ReplayGraphBufferControl {
    std::atomic<PTO2ReplayGraphBufferState> state;
    std::atomic<int32_t> exec_done;
    std::atomic<int32_t> dep_closed;
    std::atomic<int32_t> completed_count;
    int32_t buffer_id;
    uint64_t graph_epoch;
    int32_t task_begin;
    int32_t task_count;
};

struct PTO2ReplayGraphPipelineState {
    std::atomic<int32_t> all_done;
    std::atomic<int32_t> published_task_count;
    int32_t active_buffer;
    int32_t graph_count;
    uint64_t current_graph_epoch;
    PTO2ReplayGraphBufferControl buffers[PTO2_REPLAY_GRAPH_BUFFER_COUNT];
};

inline bool pto2_try_release_graph_buffer(PTO2ReplayGraphBufferControl &buffer) {
    if (buffer.exec_done.load(std::memory_order_acquire) == 0 ||
        buffer.dep_closed.load(std::memory_order_acquire) == 0) {
        return false;
    }
    buffer.state.store(PTO2ReplayGraphBufferState::FREE, std::memory_order_release);
    return true;
}

// Task management. Each graph must fit in one half of the task window.
#define PTO2_TASK_WINDOW_SIZE 16384

// Invocation-scoped heap, dependency, and TensorMap capacities.
#define PTO2_HEAP_SIZE (256 * 1024 * 1024)
#define PTO2_DEP_LIST_POOL_SIZE 16384
#define PTO2_TENSORMAP_POOL_SIZE (65536)  // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096   // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Scope management
#define PTO2_MAX_SCOPE_DEPTH 64  // Maximum nesting depth
// Maximum logical tasks retained by diagnostics and completion accounting.
#define PTO2_SCOPE_TASKS_CAP PTO2_TASK_WINDOW_SIZE

// Ready queue
#define PTO2_READY_QUEUE_SIZE 65536  // Per-shape queue size

// Cross-thread early-dispatch work queue (power of two)
#define PTO2_EARLY_DISPATCH_QUEUE_SIZE 64

// Dependency-degree diagnostic: warn once when a task's fanin or a producer's
// fanout first exceeds this degree, so dense dependency graphs surface without
// flooding the AICPU hot-path device log.
#define PTO2_DEP_DEGREE_WARN_THRESHOLD 16

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
 *   PENDING -> COMPLETED
 *
 * The slot stays in PENDING from submit through "ready in queue" and "running
 * on a worker"; readiness and running-vs-idle are derived from fanin_refcount
 * and per-core running_slot_state respectively, not from task_state itself.
 *
 * Conditions:
 *   PENDING->COMPLETED: all subtasks finish (set by scheduler) or task is a
 *                       hidden alloc completed inline by the orchestrator
 */
typedef enum { PTO2_TASK_PENDING = 0, PTO2_TASK_COMPLETED = 1 } PTO2TaskState;

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

enum class PTO2TaskKind : uint8_t {
    KERNEL = 0,
    DUMMY = 1,
    GRAPH = 2,
    GRAPH_NODE = 3,
};

struct PTO2OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

// =============================================================================
// Dependency List Entry
// =============================================================================

struct PTO2TaskSlotState;  // Forward declaration

/**
 * Dependency list entry (singly-linked list node)
 * Stored in the whole-graph dependency bump pool.
 */
struct PTO2DepListEntry {
    PTO2TaskSlotState *slot_state;  // Consumer slot state (direct pointer)
    PTO2DepListEntry *next;         // next entry
};

inline PTO2DepListEntry *pto2_fanout_closed_sentinel() {
    return reinterpret_cast<PTO2DepListEntry *>(static_cast<uintptr_t>(1));
}

inline bool pto2_is_fanout_closed(PTO2DepListEntry *ptr) { return ptr == pto2_fanout_closed_sentinel(); }

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stream-task descriptors are stored in the Task Window. Graph Execution
 * nodes use the same layout in scheduler-owned dynamic storage.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state is in PTO2TaskSlotState; Graph-node topology stays
 * in the immutable Graph Definition.
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
    void *packed_buffer_end;   // End of this task's packed output range

    // Graph nodes live in a GraphExecution allocation instead of the task
    // window. The outer Graph task and all of its nodes point at the same
    // execution; ordinary tasks keep this null.
    void *graph_execution;
    int32_t graph_node_index;
    PTO2TaskKind kind;
};

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * The fanin producer list is not retained: submit_task builds fanout edges
 * directly, so payload only carries dispatch-time data and early-dispatch
 * metadata.
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
    // === Cache line 0 — metadata + early-dispatch state ===
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    // Early-dispatch metadata (AICPU-side only). Ordered by descending
    // alignment (8B mask, 4B fanin, then 2B/1B counters and flags) so the block packs with no
    // internal padding. It shares cache line 0 with tensor_count/scalar_count
    // and fits before the aligned tensors[] array.
    //
    // Bitmask of global core_ids this consumer is pre-staged (gated) on. Concurrent
    // stagers publish bits with atomic fetch_or. A regular consumer destructively
    // splits them between release and late-stager owners; a sync_start drain keeps
    // the completed mask stable for its single cohort launch owner.
    std::atomic<uint64_t> staged_core_mask[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS]{};
    // Early-dispatch CANDIDATE detection (event-driven, dual of fanin_refcount):
    // Starts at zero. Scheduler propagation bumps each consumer after a
    // producer has published all of its logical blocks.
    // dispatch_fanin == slot_state.fanin_count  <=>  every producer is
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
    std::atomic<uint8_t> dispatch_propagated{0};  // PRODUCER side: once-guard for fanout propagation
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
    // === Cache lines 1-64 (4096B) — tensors (alignas(64) forces alignment) ===
    Tensor tensors[MAX_TENSOR_ARGS];
    // === Cache lines 65-66 (128B) — scalars ===
    uint64_t scalars[MAX_SCALAR_ARGS];

    // Layout verification (size checks that don't need offsetof).
    static_assert(sizeof(Tensor) == 128, "Tensor must be 2 cache lines");
    static_assert(MAX_SCALAR_ARGS * sizeof(uint64_t) == 128, "scalar region must be 128B (2 cache lines)");

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). A member fn lowers to the same prefetch
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
        // fields. Between-replay reset initializes shared slot state; payload
        // state is initialized here once for each submitted task. prepare_task
        // only allocates/binds, and prefetch() warms this metadata line.
        //
        // early_dispatch_state / staged_core_mask / dispatch_fanin are all CONSUMER-side: a
        // task whose own allow_early_resolve is false still has them touched when
        // one of ITS producers is flagged (propagate_dispatch_fanin bumps
        // dispatch_fanin and may CAS early_dispatch_state on any consumer, independent of the
        // consumer's own hint). So they MUST be zeroed here unconditionally.
        // Publication, propagation, and launch fields share this same
        // per-submit lifetime and are reset here too.
        early_dispatch_state.store(PTO2_EARLY_DISPATCH_NONE, std::memory_order_relaxed);
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_core_mask[w].store(0, std::memory_order_relaxed);
        dispatch_fanin.store(0, std::memory_order_relaxed);
        dispatch_propagated.store(0, std::memory_order_relaxed);
        published_block_count.store(0, std::memory_order_relaxed);
        early_dispatch_launch_state.store(PTO2_EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        running_slot_count.store(0, std::memory_order_relaxed);
        early_sync_drain_state.store(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
    }
};

// PTO2TaskPayload layout verification (offsetof requires complete type).
static_assert(offsetof(PTO2TaskPayload, tensors) == 64, "tensors must start at byte 64 (cache line 1)");
static_assert(
    offsetof(PTO2TaskPayload, scalars) == 64 + MAX_TENSOR_ARGS * sizeof(Tensor),
    "scalars must immediately follow tensors"
);
static_assert(
    sizeof(PTO2TaskPayload) == 64 + MAX_TENSOR_ARGS * sizeof(Tensor) + MAX_SCALAR_ARGS * sizeof(uint64_t),
    "PTO2TaskPayload size must stay on the baseline cache-line footprint"
);

/**
 * Per-task slot scheduling state (scheduler-private, NOT in shared memory)
 *
 * Consolidates all hot-path scheduling fields into a single cache-friendly
 * structure (32 bytes = half a cache line). Accessing any field of a task's
 * slot state brings all related fields into the same cache line.
 *
 * Stream-task fanout_head is an atomic intrusive stack. Completion closes it
 * with a sentinel, so a concurrent orchestrator append either joins the list
 * before completion or observes that the producer is already complete. Graph
 * nodes leave fanout_head empty and traverse their Definition's CSR fanout.
 */

enum PTO2ReadyState : uint8_t {
    PTO2_READY_UNCLAIMED = 0,
    PTO2_READY_CLAIMED = 1,
};

enum PTO2CompletionFlag : uint8_t {
    PTO2_COMPLETION_DONE = 2,
};

enum PTO2DeferredCompletionFlag : uint8_t {
    PTO2_SUBTASK_DEFERRED = 4,
};

struct alignas(64) PTO2TaskSlotState {
    std::atomic<PTO2DepListEntry *> fanout_head;  // nullptr = empty, CLOSED = producer completed

    // Task state (completion and ready checks)
    std::atomic<PTO2TaskState> task_state;  // PENDING/COMPLETED

    // Fanin (accessed together in release_fanin_and_check_ready)
    std::atomic<int32_t> fanin_refcount;  // Dynamic: counts completed producers
    int32_t fanin_count;                  // Exact producer count (set once during submit)

    // --- Per-slot constant, bound by orch::prepare_task on first use ---
    // The value is fixed for a slot (&task_payloads[slot] / &task_descriptors[slot])
    // and written at submit instead of in an O(window_size) init loop;
    // these are the only "scale-dependent" pointers in this struct, so moving
    // them out of init makes startup cost independent of task_window_size.
    PTO2TaskPayload *payload;
    PTO2TaskDescriptor *task;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    uint8_t ring_id;         // Compatibility field; replay_graph always binds ring 0
    // These compact flags keep PTO2TaskSlotState within one cache line.
    // Codegen early-dispatch hint, copied from Arg at submit. Lives on
    // slot_state (not payload) so fanin walks read the already-hot producer
    // slot_state cache line.
    bool allow_early_resolve{false};
    std::atomic<uint8_t> ready_state{PTO2_READY_UNCLAIMED};
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
     * Bind the compatibility ring id. replay_graph always uses ring 0.
     */
    void bind_ring(uint8_t rid) { ring_id = rid; }

    /**
     * Bind the per-slot payload/task pointers during the slot's only submit in
     * this replay. Both fields land on the slot_state cache line already being
     * initialized, avoiding a separate startup pass.
     */
    void bind_buffers(PTO2TaskPayload *p, PTO2TaskDescriptor *t) {
        payload = p;
        task = t;
    }

    void mark_completed() {
        task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        ready_state.fetch_or(PTO2_COMPLETION_DONE, std::memory_order_release);
    }

    bool is_completion_flag_set() const {
        return (ready_state.load(std::memory_order_acquire) & PTO2_COMPLETION_DONE) != 0;
    }

    // Set by any subtask FIN that pushed deferred-completion CONDITIONs to the
    // runtime mailbox; read by the last subtask FIN to decide whether the task
    // needs MPSC-deferred completion or can complete inline on this thread. The
    // release write is sequenced before on_subtask_complete's acq_rel fetch_add
    // and the acquire read after, so all earlier subtasks' writes are visible to
    // the last subtask.
    void mark_any_subtask_deferred() { ready_state.fetch_or(PTO2_SUBTASK_DEFERRED, std::memory_order_release); }

    bool has_any_subtask_deferred() const {
        return (ready_state.load(std::memory_order_acquire) & PTO2_SUBTASK_DEFERRED) != 0;
    }

    void set_allow_early_resolve(bool v) { allow_early_resolve = v; }
};

static_assert(sizeof(PTO2TaskSlotState) == 64);

#endif  // SRC_A2A3_RUNTIME_REPLAY_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
