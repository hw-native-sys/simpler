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
 * host_build_graph core type definitions
 *
 * This header defines all fundamental types used by the host_build_graph runtime:
 * - Configuration constants
 * - Worker types and task states
 * - simpler::hbg::Tensor regions and task parameters
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
#include <cstddef>
#include <type_traits>

#include "profiling_config.h"
#include "host_build_graph/constants.h"
#include "host_build_graph/runtime_status.h"
// NOTE (host_build_graph divergence from tensormap_and_ringbuffer): the
// dispatch_payload.h include is intentionally dropped here. This header is
// reached by a path-qualified include, and dispatch_payload.h uses #pragma once
// (path-keyed), so leaving it in double-defines DispatchPayload against
// tensormap_and_ringbuffer's copy inside the shared host-dispatcher TU.
// runtime_types.h never references DispatchPayload itself; consumers that
// need it include it via runtime.h directly.
#include "aicore_completion_mailbox.h"
#include "common/args_dump_task_metadata.h"
#include "host_build_graph/self_relative_ptr.h"
#include "host_build_graph/submit_types.h"
#include "task_id.h"
#include "host_build_graph/types.h"

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
//
// The task table is a flat array of slots, indexed directly by local task id:
// ids start at 0, are never recycled, and alloc() caps them at the table's size,
// so there is no wrap and no slot mask. The size need not be a power of two —
// nothing masks with it.
//
// This is the default; `CallConfig.runtime_env.ring_task_window` overrides it per
// task. The host mirror is allocated at whatever size is in effect and committed
// by first touch, so a run pays only for the slots and argument-pool bytes it
// actually writes. Raising it costs virtual address space, bounded by the int32
// reach of a payload's self-relative region deltas (checked in
// SharedMemoryHandle::init).
#define CHIP_DEFAULT_GRAPH_TASKS 16384

// host_build_graph carries no per-scope-depth task partition: host-orch builds
// the whole graph on the host and the device runs it once without reclaim. tmr's
// multi-ring design existed only to let inner scopes reclaim independently under
// small rings; with no reclaim and a whole-graph-resident task table, per-depth
// isolation is moot. The RuntimeEnv ABI still carries RUNTIME_ENV_RING_COUNT
// slots because it is shared with tmr; this runtime reads none of them and warns
// when one is set (see bind_callable_to_runtime_impl).

// Memory pools (total = value)
#define CHIP_TENSORMAP_POOL_SIZE (65536)  // TensorMap entry pool
#define CHIP_TENSORMAP_NUM_BUCKETS 4096   // Power of 2 for fast hash (4096×8B=32KB fits L1)

// Three address classes coexist during orchestration, in windows the two constants
// below keep disjoint: real device addresses stay below HEAP_VIRTUAL_BASE, since
// Ascend VA is 48-bit and the asserts named further down hold caller-owned ones
// there; the graph heap spans HEAP_VIRTUAL_BASE up to GRAPH_RECORD_VIRTUAL_BASE;
// Graph recording takes everything above.
//
// Base of the window the graph heap is allocated out of during orchestration.
// The heap's device region is committed only once orchestration has run and its
// exact size is known, so the addresses handed out while the graph is being
// built cannot be the device ones; compact_live_image rewrites them to the real
// base before the image travels. Nothing dereferences an address in this window.
inline constexpr uint64_t HEAP_VIRTUAL_BASE = 1ULL << 62;

// Base of the address range Graph recording hands to an in-graph task's packed
// outputs. Recorded addresses are never dereferenced: they exist so
// graph_classify_tensor can tell an internal producer's output from a boundary
// tensor by address-range containment alone, and the Definition stores them as
// offsets. That classification is only sound while the range is disjoint from
// every graph-heap address, which TaskAllocator::init() asserts, and from
// every real device address, which the two asserts in the host's bind path
// (the acquired heap base, and each caller tensor as it enters device_args)
// keep below HEAP_VIRTUAL_BASE.
inline constexpr uint64_t GRAPH_RECORD_VIRTUAL_BASE = 1ULL << 63;

// Span of the graph-heap window: everything between the two virtual bases. This
// is the bound orchestration allocates against, so a graph is limited by what
// the device can commit afterwards rather than by a configured heap size.
inline constexpr uint64_t HEAP_VIRTUAL_CAPACITY = GRAPH_RECORD_VIRTUAL_BASE - HEAP_VIRTUAL_BASE;

// Scope management
#define CHIP_MAX_SCOPE_DEPTH 64  // Maximum nesting depth

// Per-queue arena reservation ceiling. Bind configures each queue to the next
// power of two covering the tasks that can reach it and rejects a larger graph.
inline constexpr uint64_t READY_QUEUE_CAPACITY_LIMIT = 32768;

// Cross-thread early-dispatch work queue (power of two)
#define CHIP_EARLY_DISPATCH_QUEUE_SIZE 64

// Fanin storage
#define CHIP_FANIN_INLINE_CAP 64

// Polling-scheduler fanin cap. The polling model stores producer dependencies as
// flat position-independent local-id integers in the fanin pool (no dep-pool
// spill), so a task's fanin degree is hard-capped here. Must cover the worst-case
// fanin of any workload (paged_attention is the densest).
#define CHIP_MAX_FANIN 128

// How far back transitive reduction can prove one fanin edge redundant, measured
// in task local ids. One native word, so a task's whole ancestor set is a single
// uint64 and the shift-merge in reduce_redundant_fanin is one instruction; the
// pass relies on that width, since a shift by FANIN_REACH_WINDOW would be
// undefined and every ancestor of a producer that far back is out of the window
// anyway. Raising it means widening the entry to several words and shifting
// across them, not editing this number.
inline constexpr int32_t FANIN_REACH_WINDOW = 64;

// Alignment of every per-task region inside an argument pool. Each region starts
// and ends on a cache line so TaskPayload::init's round-up scalar memcpy stays
// inside the task's own region — see its comment. simpler::hbg::Tensor is already 2 cache
// lines, so only the fanin and scalar regions need the round-up.
inline constexpr int32_t ARG_POOL_ALIGN = 64;

// Dependency-degree diagnostic: warn once when a task's fanin or a producer's
// fanout first exceeds this degree, so dense dependency graphs surface without
// flooding the AICPU hot-path device log.
#define CHIP_DEP_DEGREE_WARN_THRESHOLD 16

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
 * on a worker": readiness comes from the producers' completion state, and
 * running-vs-idle from the per-core running_slot_state -- neither from this
 * field. Which completion state carries readiness depends on the task's id
 * space; see ChipTaskSlotState below.
 *
 * Conditions:
 *   PENDING->COMPLETED:   all subtasks finish (set by scheduler) or task is a
 *                         hidden alloc completed inline by the orchestrator
 *
 * COMPLETED is terminal: no slot is recycled before the run ends, so nothing
 * advances a task past it.
 */
typedef enum : uint8_t {
    CHIP_TASK_PENDING = 0,   // Submitted; awaiting fanin, queued, or dispatched
    CHIP_TASK_COMPLETED = 1  // Execution finished, output may still be in use
} ChipTaskState;

/**
 * Result of a unified task allocation.
 *
 * There is no separate slot: a task id indexes the task table directly.
 */
struct TaskAllocResult {
    int32_t task_id;    // Task id, which is also its task-table index
    void *packed_base;  // Heap allocation result (nullptr if failure)
    void *packed_end;   // packed_base + aligned output_size

    bool failed() const { return task_id < 0; }
};

/**
 * What a task is, independent of where it belongs.
 *
 * KERNEL and DUMMY are leaves: KERNEL dispatches to cores, DUMMY carries only
 * dependencies. GRAPH is a container — a shell that expands into its own body of
 * tasks and completes when they all have.
 *
 * Membership is not a kind: a task inside a Graph body is an ordinary KERNEL or
 * DUMMY, and `graph_context` names the Graph it belongs to.
 */
enum class TaskKind : uint8_t {
    KERNEL = 0,
    DUMMY = 1,
    GRAPH = 2,
};

struct OutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    int32_t total_output_size = 0;
};

// =============================================================================
// Dependency List Entry
// =============================================================================

struct ChipTaskSlotState;  // Forward declaration (defined below)
struct TaskPayload;        // Forward declaration (defined below)

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure (shared memory)
 *
 * Stored in the TaskDescriptor table in shared memory.
 * Contains static identification and buffer pointers only.
 * Dynamic scheduling state (fanin/fanout/task_state) is in ChipTaskSlotState.
 *
 * Fields set by Orchestrator at submission, read by Scheduler for dispatch.
 */
struct alignas(64) TaskDescriptor {
    // Task identity. See src/common/host_build_graph/task_id_encoding.h: the
    // upper 32 bits are this runtime's id space, not a ring index.
    TaskId task_id;

    // Per-slot kernel IDs (INVALID_KERNEL_ID = inactive)
    int32_t kernel_id[SUBTASK_SLOT_COUNT];

    // Packed output buffer (all outputs packed into single contiguous buffer)
    void *packed_buffer_base;  // Start of packed buffer in GM Heap
    void *packed_buffer_end;   // End of packed buffer (for heap reclamation)

    // Pads the descriptor to the cache line ChipTaskStorage places the slot state
    // on, which is what makes that container's slot offset equal to this size.
    uint8_t reserved[24];

    // This task's other two records, defined below once ChipTaskStorage is complete.
    ChipTaskSlotState &to_slot();
    const ChipTaskSlotState &to_slot() const;
    TaskPayload &to_payload();
    const TaskPayload &to_payload() const;
};

// A 4-byte alignment pad follows kernel_id[3]; the scheduler and shared-memory
// ABI depend on the descriptor size and packed_buffer_base offset staying fixed.
static_assert(sizeof(TaskDescriptor) == 64, "TaskDescriptor size is part of the shared-memory ABI");
static_assert(offsetof(TaskDescriptor, packed_buffer_base) == 24, "packed_buffer_base offset must be unchanged");

// =============================================================================
// Per-Slot Scheduling State
// =============================================================================

/**
 * Task payload data (cold path - only accessed during orchestration and dispatch)
 *
 * Layout: metadata + inline fanin packed in the first 9 cache lines, followed
 * by bulk tensor and scalar data. Fanin is always inline: it is hard-capped at
 * CHIP_MAX_FANIN and there is no spill pool.
 */
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
// chip's core count (RUNTIME_MAX_WORKER; no two-level pre-dispatch means
// gated cores in flight <= core count), NOT by block_num — so a wide SPMD
// consumer can pre-stage all its idle cores. Two words cover every
// architecture's core count, which scheduler_dispatch.cpp static_asserts.
inline constexpr int EARLY_DISPATCH_CORE_MASK_WORDS = 2;

struct TaskPayload {
    // === Cache line 0 (64B) — the dispatch path's own line ===
    // sizeof is independent of CHIP_MAX_FANIN / MAX_TENSOR_ARGS / MAX_SCALAR_ARGS:
    // widening a cap costs pool bytes for the tasks that need them, not a control
    // block on every task.
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    int32_t fanin_count{0};  // Producer dependency count (raw, no +1 redundance)

    // This task's three argument regions, each in a pool outside this struct and
    // named by a delta from the naming field's own address. A delta holds only for
    // the layout it was taken in, so every one is bound twice on the host: against
    // the mirror when the slot is claimed, and against the image in
    // compact_live_image, which re-pitches the segments.
    //
    // fanin holds flat position-independent producer local task ids. A producer is
    // named by its local id alone, so no per-edge indirection is stored. Scanned by
    // classify_fanin_state against the shared-memory completion_flags. Hard-capped at
    // CHIP_MAX_FANIN (no dep-pool spill). Unbound on an in-graph task, whose
    // dependencies live in the Definition's fanin CSR instead.
    simpler::hbg::SelfRelativePtr<simpler::hbg::Tensor> tensors;
    simpler::hbg::SelfRelativePtr<uint64_t> scalars;
    simpler::hbg::SelfRelativePtr<int32_t> fanin;

    // === Cache line 1 — early-dispatch metadata (AICPU-side only) ===
    // Ordered by descending alignment (8B mask, 4B fanin, then 2B/1B counters and
    // flags) so the block packs with no internal padding. On its own line rather
    // than line 0: these atomics are written during staging while line 0's counts
    // and region deltas are read by build_payload at dispatch, so sharing a line
    // would false-share.
    //
    // Bitmask of global core_ids this consumer is pre-staged (gated) on. Concurrent
    // stagers publish bits with atomic fetch_or. A regular consumer destructively
    // splits them between release and late-stager owners; a sync_start cohort keeps
    // the completed mask stable for its single launch owner, whether staging is local
    // or uses the global drain fallback.
    alignas(64) std::atomic<uint64_t> staged_core_mask[EARLY_DISPATCH_CORE_MASK_WORDS]{};
    // Early-dispatch CANDIDATE detection, event-driven and counted rather than
    // polled: seeded to 0 at submit with the producers already complete, then a
    // flagged producer bumps each consumer after all of its logical blocks are
    // published (propagate_dispatch_fanin).
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
    std::atomic<uint8_t> dispatch_propagated{0};  // PRODUCER side: once-guard for fanout propagation
    // The launch owner publishes COMPLETE only after all owned doorbells are
    // visible, keeping fanout private until every gated block has launched.
    std::atomic<uint8_t> early_dispatch_launch_state{EARLY_DISPATCH_LAUNCH_NONE};
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
    std::atomic<uint8_t> early_sync_drain_state{EARLY_SYNC_DRAIN_NONE};
    // === Cache line 2 — dispatch predicate + dump metadata (AICPU-only) ===
    // AICore never reads either — args are materialized from line 0's counts and
    // region deltas. Resolved at submit; evaluated by the scheduler at dispatch.
    alignas(64) DispatchPredicate predicate;
    ArgsDumpTaskMetadata dump_metadata;

    // --- Argument region access ---
    // Each accessor resolves its delta once and hands back the region's first
    // element, so a caller indexes the region directly. Deliberately no per-element
    // accessor: one inside a loop would re-resolve the delta on every iteration, and
    // a store through an unrelated pointer in the loop body is enough to stop the
    // compiler hoisting that load — build_payload's args[] writes are exactly that.
    simpler::hbg::Tensor *tensor_data() { return tensors.get(); }
    const simpler::hbg::Tensor *tensor_data() const { return tensors.get(); }
    uint64_t *scalar_data() { return scalars.get(); }
    const uint64_t *scalar_data() const { return scalars.get(); }
    int32_t *fanin_data() { return fanin.get(); }
    const int32_t *fanin_data() const { return fanin.get(); }

    /**
     * Point this payload's three argument regions at pool-resident storage. Must run
     * before prefetch() and init(), which dereference them.
     *
     * An in-graph task passes nullptr for fanin: its dependencies come from the
     * Definition's CSR, so the region does not exist and fanin_count stays 0.
     */
    void bind_regions(simpler::hbg::Tensor *tensor_region, uint64_t *scalar_region, int32_t *fanin_region) {
        tensors.set(tensor_region);
        scalars.set(scalar_region);
        fanin.set(fanin_region);
    }

    /**
     * Prefetch (for write) the regions init() is about to fill so the stores land
     * in warm cache. tensor_count/scalar_count come from the Arg — the payload's
     * own counts are not set until init(). A member fn lowers to the same prefetch
     * instructions as a free function (`this` is just a register), no cache impact.
     */
    void prefetch(int32_t tensor_count, int32_t scalar_count) const {
        const simpler::hbg::Tensor *t = tensor_data();
        for (int32_t i = 0; i < tensor_count; i++) {
            __builtin_prefetch(&t[i], 1, 3);
            __builtin_prefetch(reinterpret_cast<const char *>(&t[i]) + 64, 1, 3);
        }
        const uint64_t *s = scalar_data();
        for (int32_t i = 0; i < scalar_count; i += 8) {
            __builtin_prefetch(&s[i], 1, 3);
        }
        __builtin_prefetch(this, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 64, 1, 3);
        __builtin_prefetch(reinterpret_cast<const char *>(this) + 128, 1, 3);
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

        // bind_regions must already have run: an unbound region reads back as null and
        // the stores below would go through it. A count of zero needs no region, so a
        // task with neither argument kind may leave both unbound.
        debug_assert(args.tensor_count() == 0 || tensor_data() != nullptr);
        debug_assert(args.scalar_count() == 0 || scalar_data() != nullptr);

        simpler::hbg::Tensor *dst = tensor_data();
        for (int32_t i = 0; i < args.tensor_count(); i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) {
                dst[i].copy(args.tensor(i).ref());
            } else {
                init_tensor_from_create_info(
                    dst[i], args.tensor(i).create_info(),
                    reinterpret_cast<void *>(reinterpret_cast<char *>(alloc_result.packed_base) + layout.offsets[i]),
                    layout.buffer_sizes[i]
                );
                dst[i].owner_task_id = result.task_id();
                result.materialize_output(dst[i]);
            }
        }
        // Round up to cache line boundary. Every scalar region is a whole number of
        // cache lines (ARG_POOL_ALIGN), so the rounded copy stays inside this
        // task's own region. Eliminates branches; extra bytes within the same CL have
        // zero additional cost.
        memcpy(scalar_data(), args.scalars(), CHIP_ALIGN_UP(args.scalar_count() * sizeof(uint64_t), 64));

        // The task table's payload storage is raw shared memory that no constructor
        // runs over, so an unset predicate reads back as whatever the slot last held —
        // and compact_live_image translates predicate.addr as a graph-heap address
        // for every submitted slot. An ordinary task overwrites this right
        // after init(); a hidden-alloc task has nothing following it, so this is
        // the only value its predicate ever gets.
        predicate = DispatchPredicate{};
        dump_metadata = {};
#if SIMPLER_DFX
        dump_metadata.dump_arg_mask = args.dump_arg_mask();
        dump_metadata.dump_arg_flags = args.dump_arg_index_ambiguous_mask();
        memcpy(dump_metadata.scalar_dtypes, args.scalar_dtypes(), args.scalar_count() * sizeof(uint8_t));
#endif

        // Early-dispatch metadata — the single init point for these
        // fields. reset_for_reuse MUST NOT touch the payload (it runs at slot
        // init and would pull this cold cache line across structures);
        // prepare_task only allocates/binds. prefetch() warms this
        // line (cache line 1) so these writes land in warm cache.
        //
        // early_dispatch_state / staged_core_mask / dispatch_fanin are all CONSUMER-side: a
        // task whose own allow_early_resolve is false still has them touched when
        // one of ITS producers is flagged (propagate_dispatch_fanin bumps
        // dispatch_fanin and may CAS early_dispatch_state on any consumer, independent of the
        // consumer's own hint). So they MUST be zeroed here unconditionally.
        // Publication, propagation, and launch fields share this same
        // per-submit lifetime and are reset here too.
        early_dispatch_state.store(EARLY_DISPATCH_NONE, std::memory_order_relaxed);
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_core_mask[w].store(0, std::memory_order_relaxed);
        dispatch_fanin.store(0, std::memory_order_relaxed);
        dispatch_propagated.store(0, std::memory_order_relaxed);
        published_block_count.store(0, std::memory_order_relaxed);
        early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_NONE, std::memory_order_relaxed);
        running_slot_count.store(0, std::memory_order_relaxed);
        early_sync_drain_state.store(EARLY_SYNC_DRAIN_NONE, std::memory_order_relaxed);
    }

    // This task's other two records, defined below once ChipTaskStorage is complete.
    TaskDescriptor &to_descriptor();
    const TaskDescriptor &to_descriptor() const;
    ChipTaskSlotState &to_slot();
    const ChipTaskSlotState &to_slot() const;
};

// TaskPayload layout verification (offsetof requires complete type). The counts
// and region deltas share the first cache line, the early-dispatch atomics own the
// second, and the AICPU-only predicate + dump metadata own the third.
static_assert(offsetof(TaskPayload, tensors) == 12, "region deltas must follow the three counts");
static_assert(
    offsetof(TaskPayload, fanin) + sizeof(simpler::hbg::SelfRelativePtr<int32_t>) <= 64,
    "counts + region deltas must fit the first cache line"
);
static_assert(
    offsetof(TaskPayload, staged_core_mask) == 64,
    "the early-dispatch atomics own cache line 1: they are written during staging while "
    "line 0's counts and deltas are read at dispatch, so sharing a line would false-share"
);
static_assert(offsetof(TaskPayload, predicate) == 128, "dispatch predicate owns cache line 2");
static_assert(
    offsetof(TaskPayload, dump_metadata) + sizeof(ArgsDumpTaskMetadata) <= 192,
    "dump metadata must fit the predicate's cache line"
);
static_assert(sizeof(TaskPayload) == 192, "TaskPayload is three cache lines and independent of every argument cap");
// compact_live_image restacks the payload segment with one memcpy and the device
// copy moves the whole image, so the payload has to be a POD wire struct. Deleting
// SelfRelativePtr's copy operations does not cost it that — a deleted special member
// is trivial — but only an assertion keeps a future member from doing so silently.
static_assert(
    std::is_trivially_copyable_v<TaskPayload> && std::is_standard_layout_v<TaskPayload>,
    "TaskPayload crosses to the device by memcpy"
);
static_assert(sizeof(simpler::hbg::Tensor) == 128, "simpler::hbg::Tensor must be 2 cache lines");

/**
 * Per-task slot scheduling state (scheduler-private, NOT in shared memory)
 *
 * 64 bytes = one cache line. Under the polling completion model a task's
 * readiness is derived from its producers' completion state; producer completion
 * is published by marking this task complete + draining its wake list. There is
 * no fanout adjacency, refcount, or per-task lock here.
 *
 * Which field carries that completion state depends on which task table the slot
 * belongs to, and both are load-bearing:
 *
 *   - A GLOBAL task holds a slot in the SM task table, so its readiness truth is
 *     `completion_flags[local_id]` — a byte-per-slot array, which is what lets a
 *     fanin scan read many producers out of one cache line. `task_state` is then
 *     a mirror, read only by the cold-path stall dump.
 *   - An IN_GRAPH task lives in its Graph's own storage and has no slot in that
 *     table, hence no flag byte. `task_state` IS its readiness truth, read on the
 *     device by graph_first_unmet_producer; only the outer Graph shell (a GLOBAL
 *     task) gets a flag when the body finishes.
 *
 * So a completion publishes both for a GLOBAL task and `task_state` alone for an
 * IN_GRAPH one.
 */
struct alignas(64) ChipTaskSlotState {
    // --- Wake list: last-fanin notification (intrusive, lock-free) ---
    // A pending consumer whose fanin scan finds an unmet producer registers on
    // that producer's wake list (CAS push through next_in_wake_list). On
    // completion the producer atomic-exchanges wake_list_head to
    // WAKE_LIST_SENTINEL and routes every waiter. Reset to nullptr at init.
    std::atomic<ChipTaskSlotState *> wake_list_head{nullptr};
    ChipTaskSlotState *next_in_wake_list{nullptr};

    // Graph membership, and which of the two Graph structs this points at is
    // decided by task_kind rather than by anything stored here:
    //
    //   nullptr                        an ordinary task, in no Graph
    //   != nullptr, task_kind == GRAPH the outer Graph task, pointing at the
    //                                  shared GraphDefinition until localize
    //                                  swaps in its GraphExecution
    //   != nullptr, task_kind != GRAPH an in-graph task, pointing at the
    //                                  GraphExecution it belongs to
    //
    // So every reader must test task_kind before casting, and complete_task
    // routes on exactly that pair: a null context or a GRAPH kind takes the
    // ordinary global fanout, anything else is counted against its Graph. A
    // localize that fails puts this back to nullptr (scheduler_cold_path.cpp),
    // so the outer task cannot be mistaken for an in-graph one.
    void *graph_context{nullptr};

    // Graph-only scheduling metadata, paired with graph_context above. Readiness
    // uses the shared intrusive wake-list fields; this index identifies the task in
    // the saved fanin CSR. Ordinary tasks leave both Graph fields -1/null.
    int32_t in_graph_task_index{-1};

    std::atomic<int16_t> completed_subtasks{0};  // Each core completion increments by 1
    int16_t total_required_subtasks{0};          // = logical_block_num * popcount(active_mask)
    int16_t logical_block_num{1};                // Total logical blocks (set by orchestrator)
    // Next block to dispatch. Normal dispatch and late early-dispatch stagers
    // can run concurrently after a partial staged release. All paths claim
    // ranges through claim_block_range().
    std::atomic<int16_t> next_block_idx{0};

    // Completion state. PENDING at submit; COMPLETED at whichever completion path
    // owns this slot. For an IN_GRAPH task this is the readiness truth the device
    // itself polls (graph_first_unmet_producer); for a GLOBAL task it mirrors
    // completion_flags[slot], which is what the device reads instead. Also read by
    // the cold-path stall dump.
    std::atomic<ChipTaskState> task_state;

    // --- Set per-submit (depend on task inputs) ---
    ActiveMask active_mask;  // Bitmask of active subtask slots (set once)
    // Single per-task attributes byte (early-dispatch hint, sync_start,
    // has_predicate, selective timing tag). Lives on slot_state (not payload) so
    // fanin walks and the completion path read them off the already-hot producer
    // slot_state cache line. Plain-write (set once at submit, before the slot is
    // scheduler-visible).
    TaskAttrs task_attrs{};
    // Set by any subtask FIN that pushed a deferred-completion CONDITION to the
    // runtime mailbox; read by the last subtask FIN to decide inline vs
    // MPSC-deferred completion. The release write is sequenced before
    // on_subtask_complete's acq_rel fetch_add and the acquire read after.
    std::atomic<bool> any_subtask_deferred{false};
    TaskKind task_kind{TaskKind::KERNEL};

    // Keeps the record at one cache line. Members run widest-first, so their sizes
    // sum to exactly the bytes this leaves and the record carries no padding.
    uint8_t reserved[23];

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

    // Publishes completion. For an IN_GRAPH task this store is the whole
    // publication — that task has no completion_flags byte. For a GLOBAL task it
    // accompanies the completion_flags[slot] store that on_mixed_task_complete
    // makes, and is the copy the cold-path stall dump reads.
    void mark_completed() { task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release); }

    void mark_any_subtask_deferred() { any_subtask_deferred.store(true, std::memory_order_release); }

    bool has_any_subtask_deferred() const { return any_subtask_deferred.load(std::memory_order_acquire); }

    /**
     * Reset dynamic scheduling fields to their pristine values. Called once per
     * slot as the orchestrator claims it in prepare_task, and again as an
     * in-graph task's storage is materialized — whole-graph-resident hbg has no
     * execution-time slot recycle. Skips task_state (the orchestrator sets PENDING
     * when it populates the slot).
     * wake_list_head starts nullptr (open for registration), NOT SENTINEL.
     */
    void reset_for_reuse() {
        wake_list_head.store(nullptr, std::memory_order_relaxed);
        next_in_wake_list = nullptr;
        any_subtask_deferred.store(false, std::memory_order_relaxed);
        completed_subtasks.store(0, std::memory_order_relaxed);
        next_block_idx.store(0, std::memory_order_relaxed);
        in_graph_task_index = -1;
        graph_context = nullptr;
        task_kind = TaskKind::KERNEL;
        // Note: active_mask and task_attrs are per-submit-constant fields
        // rewritten in prepare_task on every reuse, so they are not reset here.
        // Payload early-dispatch/fanin fields are (re)initialized in
        // TaskPayload::init on every submit, before the slot is visible.
    }

    // This task's other two records, defined below once ChipTaskStorage is complete.
    // They are that type's layout rather than stored state: a record is always a
    // member of one, so the distances are fixed and no delta is kept here.
    TaskPayload &to_payload();
    const TaskPayload &to_payload() const;
    TaskDescriptor &to_descriptor();
    const TaskDescriptor &to_descriptor() const;
};

static_assert(sizeof(ChipTaskSlotState) == 64);
// Pins the widest-first order that leaves the record padding-free: every member
// before `reserved` is naturally aligned where the one before it ends.
static_assert(offsetof(ChipTaskSlotState, reserved) == 41, "ChipTaskSlotState grew interior padding");

// =============================================================================
// Per-Task Storage
// =============================================================================

/**
 * One task's three shared-memory records, held together so their relative
 * positions are a property of this type rather than data any of them stores.
 *
 * Every hbg task's records live in one of these — that is the invariant the
 * accessors below rest on, and none of the three types may be instantiated on its
 * own. A GLOBAL task's storage sits in the shared-memory storage segment, indexed by
 * its local task id (SharedMemoryTaskHeader::storage_at); an IN_GRAPH task's sits in
 * its Graph's own heap tail, indexed by its position in the body
 * (GraphExecution::task_at). The two therefore share one addressing rule, which is
 * what lets the scheduler reach a task's descriptor and payload from its slot state
 * with neither record naming the other.
 *
 * Field order is part of the layout AICore reads: it resolves a descriptor and a
 * payload from this type's base by the byte offsets in scheduler_graph.h. That
 * header is A5-only and cannot be reached from here, so the reverse lookup lives
 * on the A5 side, in scheduler.h — the asserts below only pin this type's own
 * shape.
 */
struct alignas(64) ChipTaskStorage {
    TaskDescriptor task;
    ChipTaskSlotState slot;
    // The payload carries its argument regions as deltas into pools past the task
    // array, so its size is the same for every task and the storage array strides by
    // this type.
    TaskPayload payload;
};

// offsetof is defined for a standard-layout type, which is what the accessors need.
static_assert(std::is_standard_layout_v<ChipTaskStorage>);
static_assert(std::is_trivially_destructible_v<ChipTaskStorage>);
static_assert(sizeof(ChipTaskStorage) == 320);
static_assert(offsetof(ChipTaskStorage, slot) == sizeof(TaskDescriptor));
static_assert(offsetof(ChipTaskStorage, payload) == offsetof(ChipTaskStorage, slot) + sizeof(ChipTaskSlotState));

// A record's siblings, reached by ChipTaskStorage's own layout. Valid because none of
// the three types is ever instantiated outside one — that is the invariant the whole
// set rests on, and the only thing a caller has to keep true.
//
// Each distance is a function-local compile-time constant, so an accessor is one
// constant displacement and nothing about the layout leaks into a namespace. They are
// ptrdiff_t because half of them are negative.
#define CHIP_TASK_STORAGE_DELTA(from, to) \
    (static_cast<ptrdiff_t>(offsetof(ChipTaskStorage, to)) - static_cast<ptrdiff_t>(offsetof(ChipTaskStorage, from)))

inline ChipTaskSlotState &TaskDescriptor::to_slot() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(task, slot);
    return *reinterpret_cast<ChipTaskSlotState *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const ChipTaskSlotState &TaskDescriptor::to_slot() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(task, slot);
    return *reinterpret_cast<const ChipTaskSlotState *>(reinterpret_cast<const char *>(this) + kDelta);
}
inline TaskPayload &TaskDescriptor::to_payload() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(task, payload);
    return *reinterpret_cast<TaskPayload *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const TaskPayload &TaskDescriptor::to_payload() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(task, payload);
    return *reinterpret_cast<const TaskPayload *>(reinterpret_cast<const char *>(this) + kDelta);
}

inline TaskDescriptor &ChipTaskSlotState::to_descriptor() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(slot, task);
    return *reinterpret_cast<TaskDescriptor *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const TaskDescriptor &ChipTaskSlotState::to_descriptor() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(slot, task);
    return *reinterpret_cast<const TaskDescriptor *>(reinterpret_cast<const char *>(this) + kDelta);
}
inline TaskPayload &ChipTaskSlotState::to_payload() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(slot, payload);
    return *reinterpret_cast<TaskPayload *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const TaskPayload &ChipTaskSlotState::to_payload() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(slot, payload);
    return *reinterpret_cast<const TaskPayload *>(reinterpret_cast<const char *>(this) + kDelta);
}

inline TaskDescriptor &TaskPayload::to_descriptor() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(payload, task);
    return *reinterpret_cast<TaskDescriptor *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const TaskDescriptor &TaskPayload::to_descriptor() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(payload, task);
    return *reinterpret_cast<const TaskDescriptor *>(reinterpret_cast<const char *>(this) + kDelta);
}
inline ChipTaskSlotState &TaskPayload::to_slot() {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(payload, slot);
    return *reinterpret_cast<ChipTaskSlotState *>(reinterpret_cast<char *>(this) + kDelta);
}
inline const ChipTaskSlotState &TaskPayload::to_slot() const {
    constexpr ptrdiff_t kDelta = CHIP_TASK_STORAGE_DELTA(payload, slot);
    return *reinterpret_cast<const ChipTaskSlotState *>(reinterpret_cast<const char *>(this) + kDelta);
}

#undef CHIP_TASK_STORAGE_DELTA

// Sentinel marking a wake list as "owner already completed; no more
// registrations accepted". Distinct from any real slot_state pointer.
inline ChipTaskSlotState *const WAKE_LIST_SENTINEL = reinterpret_cast<ChipTaskSlotState *>(static_cast<uintptr_t>(0x1));
