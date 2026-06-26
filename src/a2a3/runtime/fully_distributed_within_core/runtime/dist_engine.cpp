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
 * fully_distributed_within_core engine.
 *
 * SPMD orchestration + scheduling + execution on the AI cores. See
 * docs/fully_distributed_within_core.md for the authoritative design and
 * src/.../docs/RUNTIME_LOGIC.md for the local overview.
 *
 * Each AICore worker thread runs dist_core_main(), which:
 *   1. replays the full orchestration submit stream (every core builds an
 *      identical per-core TensorMap and computes identical deterministic GM
 *      output-heap addresses; only ownership differs);
 *   2. on each rt_submit_*, races to claim the task on one of two global
 *      cursors (cube for AIC-anchored, vector for AIV-only). The winner is
 *      owner = builder = executor and builds the task into its private ring;
 *   3. runs an EXECUTE-FIRST run-ahead loop: on every submit point it first
 *      drains ready owned tasks (and pulls follower deposits), THEN claims at
 *      most this one new task. Because claim+build is fast but execute is slow,
 *      interleaving execution with claiming stops a fast core from greedily
 *      claiming a full ring of consecutive tasks: while it executes a long task
 *      other cores advance the cursor and claim subsequent tasks (load balance,
 *      see docs §6/§6.1). The ring (small, kPrivateSlots) only back-pressures
 *      when genuinely full of not-yet-ready tasks. After orchestration returns,
 *      a final loop drains the ring to completion. A task is ready once all its
 *      fan-in producers have set their entry in the global completion-flag
 *      ring; on completion the owner sets its own flag (release).
 *
 * This file is compiled into the AICPU .so (build_config aicore source_dirs do
 * not include runtime/), but dist_core_main runs ON the AICore worker threads
 * (invoked through a function pointer), so kernels execute on AICore threads
 * with their sim TLS in place.
 *
 * M2 scope: single-core tasks (1C / 1V) only — sufficient for benchmark_bgemm.
 * Multi-core co-ownership (MIX / 2V, block.won) is M3; GM heap reclamation is
 * M4. A MIX task encountered in M2 raises a fatal error.
 */

#include "dist_engine.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "callable.h"
#include "common/core_type.h"
#include "intrinsic.h"
#include "pto2_dispatch_payload.h"
#include "pto_constants.h"
#include "pto_runtime2.h"
#include "pto_submit_types.h"
#include "pto_types.h"
#include "runtime.h"
#include "spin_hint.h"
#include "tensor.h"
#include "tensor_create_info.h"

namespace {

// -----------------------------------------------------------------------------
// Tunables. The completion-flag ring is sized to hold an entire run without
// wrap (>= total tasks); the GM output heap is a BOUNDED RING reclaimed by the
// completion frontier (M4, §9.5/§11.4) rather than a run-sized bump.
// -----------------------------------------------------------------------------
// Kept deliberately SMALL: the out-of-order window is num_cores * kPrivateSlots,
// and this also caps how far a single core can run ahead of "ready-to-execute".
// A large ring lets one fast core greedily claim a long run of consecutive tasks
// and serialize them while other cores starve (load imbalance, docs §6.1). OoO
// capacity should come from the core-count dimension, not a deep per-core ring.
constexpr int32_t kPrivateSlots = 4;       // PRIVATE_TASK_SLOT_NUM (back-pressure cap)
// Ring slots a core reserves for draining block.won deposits addressed to its
// lane. Self-claimed tasks (consumers / single-core / own anchor subtask) may
// only occupy kPrivateSlots - kWonReserve slots, so a follower can ALWAYS pull
// and run an (immediately-ready) deposit even when its ring is otherwise full of
// not-yet-ready consumers — breaking the consumer<->deposit priority inversion.
constexpr int32_t kWonReserve = 2;
constexpr int32_t kMaxFanin = 16;          // max distinct producers a task waits on
constexpr int32_t kOutPoolSlots = 1024;    // per-core ring of materialized output Tensors
constexpr int32_t kMapCap = 16384;         // per-core producer-map capacity (distinct regions)
constexpr int32_t kFlagCap = 1 << 16;      // global completion-flag ring (>= total tasks)

// M4 GM-heap reclamation (§9.5/§11.4).
//   kHeapRingDefault — bounded physical heap ring (env PTO_DIST_HEAP_MB overrides,
//     in MiB). The deterministic virtual bump is unbounded; physical address is
//     (virtual_offset mod ring). A region is reused only after its previous
//     occupant's task id <= R (the reclaim frontier), enforced by back-pressure.
//   kHDefault — dependency-span bound H (env PTO_DIST_H overrides): every consumer
//     of task N has id <= N + H. R = F - H. Must be >= the graph's true heap span
//     or a producer region could be recycled while a late consumer still reads it
//     (run-time-checked → fatal "heap span exceeded").
constexpr size_t kHeapRingDefault = 64ull << 20;
constexpr int32_t kHDefault = 64;

// -----------------------------------------------------------------------------
// Per-core producer map (the "full per-core duplicate TensorMap").
//
// A faithful, compact stand-in for PTO2TensorMap: keyed by GM byte range, it
// records the most recent producer task id of each written region. INPUT/INOUT
// fan-in resolves to the producer(s) whose region overlaps. Exact-region writes
// (e.g. an INOUT accumulation chain) replace in place; new regions append.
// Every core builds an identical map by replaying the same submit stream.
// -----------------------------------------------------------------------------
// Intrusive entry, modeled on PTO2TensorMapEntry (tensormap_and_ringbuffer) but
// compact: it keys overlap on a byte range [lo, hi) instead of mirroring a full
// Tensor cache line, since the distributed map only needs producer lookup.
//   - bucket chain (doubly linked) — O(1) unlink during cleanup
//   - task chain (singly linked)   — cleanup frees a retired task's entries by
//                                     walking ITS chain, never scanning the pool
struct MapEntry {
    uint64_t buf_addr;       // Tensor.buffer.addr (GM buffer base, bytes) — hash key
    uint64_t lo;             // byte offset of view origin within buffer
    uint64_t hi;             // byte offset one-past the view extent
    int32_t producer;        // task id that wrote this region
    int32_t bucket;          // owning bucket index, or -1 when free
    int32_t next_in_bucket;  // bucket-chain links (entry indices, -1 = none)
    int32_t prev_in_bucket;
    int32_t next_in_task;    // task-chain link (entry index, -1 = none)
};

// Hash buckets (power of 2). Hashing by buffer BASE address groups every
// sub-region of one buffer into one chain; overlap is then tested per entry.
constexpr int32_t kMapBuckets = 1 << 13;  // 8192
constexpr int32_t kMapBucketShift = 13;   // log2(kMapBuckets)
// Per-task entry-head window (power of 2). Task `id` parks its entries under
// slot id & (kTaskWindow-1); the slot is recycled by id + kTaskWindow. cleanup
// retires a task once it leaves the H span, so kTaskWindow MUST exceed H (with
// margin) or a slot could be reused before its prior task is cleaned. Validated
// against g_dist.H at register time.
constexpr int32_t kTaskWindow = 1 << 10;  // 1024  (>> kHDefault = 64)
constexpr int32_t kTaskWindowMask = kTaskWindow - 1;

// Per-core producer map ("full per-core duplicate TensorMap"), a direct compact
// port of tensormap_and_ringbuffer's PTO2TensorMap (hash table + bucket chains +
// per-task entry tracking + free list + lazy invalidation + cleanup_retired).
//
// WHY (vs. the original O(count) linear array, which made submit O(N^2)):
// bgemm writes hundreds of disjoint tiles of ONE flattened output buffer, so the
// old `entries[count]` grew with the whole run and every lookup/insert rescanned
// it. Following the proven runtime, we instead:
//   * hash by buffer base + chain — distinct buffers cost O(1);
//   * RETIRE by H window — an entry whose producer is older than `alive_floor`
//     (= N - H) can never be a fan-in of any future task (a consumer of producer
//     p has id <= p + H, §9.5/§11.4, the same bound under which p's GM heap region
//     is recycled), so cleanup frees it. This bounds each chain to ~the live
//     H-window instead of the entire run → O(N*H) ~ O(N).
// Like the reference, insert ALWAYS links a fresh entry under its producer's task
// chain (no in-place replace), so cleanup_retired can free a task's entries via
// that chain without scanning; lookup returns the MAX (newest) overlapping
// producer, which subsumes the old replace-in-place semantics.
//
// `alive_floor` is N-derived (deterministic, identical on every core), never
// frontier-based (timing-dependent), so every per-core map — including the free
// list and cleanup progress — evolves identically. Determinism is preserved.
struct DistTensorMap {
    MapEntry entries[kMapCap];
    int32_t buckets[kMapBuckets];     // bucket head entry idx, or -1
    int32_t task_heads[kTaskWindow];  // per-task entry-chain head idx, or -1
    int32_t free_head;                // recycled-slot free list head, or -1
    int32_t high_water;               // next never-used slot in `entries`
    int32_t alive_floor;              // producer < alive_floor == retired
    int32_t cleaned_upto;             // tasks < cleaned_upto already freed

    void reset() {
        free_head = -1;
        high_water = 0;
        alive_floor = 0;
        cleaned_upto = 0;
        for (int32_t i = 0; i < kMapBuckets; i++) buckets[i] = -1;
        for (int32_t i = 0; i < kTaskWindow; i++) task_heads[i] = -1;
    }

    static uint32_t hash(uint64_t addr) {
        addr *= 0x9E3779B97F4A7C15ULL;  // golden-ratio multiplicative mix
        return static_cast<uint32_t>(addr >> (64 - kMapBucketShift));
    }

    static void byte_range(const Tensor &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
        const uint64_t esz = get_element_size(t.dtype);
        addr = t.buffer.addr;
        lo = t.start_offset * esz;
        hi = (t.start_offset + t.extent_elem()) * esz;
    }

    int32_t alloc_slot() {
        if (free_head >= 0) {
            const int32_t s = free_head;
            free_head = entries[s].next_in_bucket;
            return s;
        }
        if (high_water < kMapCap) return high_water++;
        return -1;  // pool exhausted (live H-window exceeds kMapCap)
    }

    // Unlink `idx` from its bucket chain (O(1) via prev) and push to the free list.
    void free_entry(int32_t idx) {
        MapEntry &e = entries[idx];
        if (e.prev_in_bucket < 0) buckets[e.bucket] = e.next_in_bucket;
        else entries[e.prev_in_bucket].next_in_bucket = e.next_in_bucket;
        if (e.next_in_bucket >= 0) entries[e.next_in_bucket].prev_in_bucket = e.prev_in_bucket;
        e.bucket = -1;
        e.next_in_bucket = free_head;
        free_head = idx;
    }

    // Free every entry produced by retired tasks [cleaned_upto, new_floor) by
    // walking each task's own chain (never the whole pool). Mirrors PTO2TensorMap
    // ::cleanup_retired. Advances alive_floor so lookups skip the freed window.
    void advance_retire(int32_t N, int32_t H) {
        const int32_t new_floor = N - H;
        if (new_floor <= cleaned_upto) {  // nothing newly retired
            if (new_floor > alive_floor) alive_floor = new_floor;
            return;
        }
        for (int32_t id = cleaned_upto; id < new_floor; id++) {
            int32_t cur = task_heads[id & kTaskWindowMask];
            while (cur >= 0) {
                const int32_t nxt = entries[cur].next_in_task;
                debug_assert(entries[cur].producer == id);
                free_entry(cur);
                cur = nxt;
            }
            task_heads[id & kTaskWindowMask] = -1;
        }
        cleaned_upto = new_floor;
        alive_floor = new_floor;
    }

    // Link a fresh entry for `producer`'s write of `t`'s region. Always a new
    // entry (no in-place replace) so it parks under producer's task chain.
    void insert(const Tensor &t, int32_t producer) {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        const int32_t s = alloc_slot();
        if (s < 0) return;  // pool full within the live window (should not happen)
        const uint32_t b = hash(addr);
        MapEntry &e = entries[s];
        e.buf_addr = addr;
        e.lo = lo;
        e.hi = hi;
        e.producer = producer;
        e.bucket = static_cast<int32_t>(b);
        // Insert at bucket head.
        e.prev_in_bucket = -1;
        e.next_in_bucket = buckets[b];
        if (buckets[b] >= 0) entries[buckets[b]].prev_in_bucket = s;
        buckets[b] = s;
        // Insert at task-chain head.
        const int32_t slot = producer & kTaskWindowMask;
        e.next_in_task = task_heads[slot];
        task_heads[slot] = s;
    }

    // Most-recent producer whose region overlaps `t`, or -1 if none. Entries
    // below alive_floor are treated as already retired (skipped — defensive,
    // since cleanup has usually freed them already).
    int32_t lookup(const Tensor &t) const {
        uint64_t addr, lo, hi;
        byte_range(t, addr, lo, hi);
        int32_t best = -1;
        for (int32_t cur = buckets[hash(addr)]; cur >= 0; cur = entries[cur].next_in_bucket) {
            const MapEntry &e = entries[cur];
            if (e.producer < alive_floor) continue;
            if (e.buf_addr == addr && lo < e.hi && e.lo < hi) {
                if (e.producer > best) best = e.producer;
            }
        }
        return best;
    }
};

// -----------------------------------------------------------------------------
// A private-ring slot: a fully materialized, self-contained task this core owns
// and will execute itself. Holds its own copy of the argument Tensors so it can
// be executed at any later point (deferred past further orchestration).
// -----------------------------------------------------------------------------
// One executed (sub)task, recorded only when swimlane tracing is on. Laid out
// in the exported Chrome trace by physical block (pid) and lane (tid).
struct TraceEvent {
    int32_t task_id;
    int32_t func_id;   // kernel id (e.g. 0=GEMM, 1=ADD); -1 if unknown
    int32_t lane;      // AIC=0 / AIV0=1 / AIV1=2
    uint8_t multicore;
    double ts_us;      // start, microseconds from g_trace_epoch
    double dur_us;     // execution duration, microseconds
};

struct RingSlot {
    bool occupied;
    // A slot can be reserved (occupied=true) before it is fully populated: the
    // submit winner grabs a slot up front so concurrent drains do not reuse it,
    // then may spin in block.won back-pressure (which itself drains Phase B)
    // before calling build_ring_slot. `built` gates execution so Phase B never
    // (re)runs a reserved-but-unbuilt slot still holding a prior occupant's
    // task_id/fanin/won linkage. build_ring_slot sets it; execute_slot clears it.
    bool built;
    int32_t task_id;
    int32_t func_id;  // kernel id of this slot's lane (swimlane label); -1 if none
    uint64_t function_bin_addr;

    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];

    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_ctx;
    GlobalContext global_ctx;

    int32_t fanin[kMaxFanin];
    int32_t fanin_count;

    // Multi-core (MIX / 2V) linkage. When is_multicore, the completion flag for
    // task_id is owned jointly: each co-owner decrements block.won[won_slot].remaining
    // after executing its own subtask, and the one driving it to zero publishes
    // the single global task_completed_flag. Single-core tasks set the flag directly.
    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
};

// -----------------------------------------------------------------------------
// block.won — the id-keyed anchor→follower deposit table (block-shared, §3.1).
// One BlockWon per physical block (1 AIC + 2 AIV). The anchor that wins a
// multi-core task builds its OWN physical-lane subtask into its private ring and
// deposits the remaining active-lane subtasks here; followers asynchronously
// drain the entry addressed to their physical lane (no blocking, no per-walk
// wait). Keyed by task id via per-slot task_id so concurrent multi-core tasks of
// one block never alias. `remaining` = popcount(active_mask) drives the single
// completion flag (§3.1). Lane index uses PTO2SubtaskSlot (AIC=0/AIV0=1/AIV1=2).
// -----------------------------------------------------------------------------
struct BuiltSubtask {
    bool present;
    int32_t func_id;  // kernel id of this lane's subtask (swimlane label); -1 if none
    uint64_t function_bin_addr;
    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t sub_block_id;
};

struct WonSlot {
    std::atomic<int32_t> state;       // 0=free, 1=published, 2=reserving
    int32_t task_id;
    std::atomic<int32_t> remaining;   // co-owners (incl. anchor) left to finish
    std::atomic<int32_t> drained[PTO2_SUBTASK_SLOT_COUNT];  // 0/1 per follower lane
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];             // deposited follower subtasks
};

struct BlockWon {
    WonSlot slots[kPrivateSlots];
};

enum LaneId : int32_t { LANE_AIC = 0, LANE_AIV0 = 1, LANE_AIV1 = 2, LANE_NONE = -1 };

struct CoreLayout {
    int32_t block_id;  // physical block index
    int32_t lane;      // LaneId of this core within its block
};

// -----------------------------------------------------------------------------
// Per-core engine state (the SPMD worker context).
// -----------------------------------------------------------------------------
struct DistCore {
    CoreType role;
    int32_t core_idx;      // index into g_dist.cores[] (for trace ownership)
    int32_t block_id;      // physical block this core belongs to
    int32_t lane;          // LaneId within the block (AIC / AIV0 / AIV1)
    int32_t sub_block_id;
    int32_t local_index;   // next task id this core will see (== tasks replayed)
    uint64_t heap_next;    // deterministic GM output-heap bump cursor (bytes)

    DistTensorMap map;

    RingSlot slots[kPrivateSlots];
    int32_t occupied_count;
    int32_t owned_total;  // tasks this core claimed+executed (debug)

    Tensor outpool[kOutPoolSlots];
    int32_t outpool_head;

    // Per-core swimlane events (only populated when tracing is on). Owned solely
    // by this core's worker thread, so push_back is lock-free.
    std::vector<TraceEvent> trace;

    void reset(CoreType r, int32_t block, int32_t lane_id) {
        role = r;
        block_id = block;
        lane = lane_id;
        sub_block_id = (lane_id == LANE_AIV1) ? 1 : 0;
        local_index = 0;
        heap_next = 0;
        map.reset();
        occupied_count = 0;
        owned_total = 0;
        outpool_head = 0;
        trace.clear();
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            slots[i].occupied = false;
            slots[i].built = false;
        }
    }
};

// -----------------------------------------------------------------------------
// Global engine state (shared by all worker threads in this process). Cursors +
// flags live here rather than in GM because in sim every core is a host thread
// in one address space; the GM output heap below is a real shared buffer.
// -----------------------------------------------------------------------------
struct DistGlobal {
    std::atomic<int32_t> cube_cursor;     // highest claimed AIC-anchored task id
    std::atomic<int32_t> vector_cursor;   // highest claimed AIV-only task id
    std::atomic<uint8_t> flags[kFlagCap]; // completion-flag ring (1 == task done)

    // M4 reclamation (§9.5/§11.4). `frontier` (F) is the global continuous
    // completion frontier — the largest prefix s.t. every task id <= F is done;
    // advanced cooperatively (CAS) by whichever core sets the flag that extends
    // the prefix. `R = frontier - H` is the reclaim frontier. `vend[N]` is the
    // cumulative virtual heap bytes through task N (deterministic & identical on
    // every core), so any core can compute the live byte window [vend[R], top).
    std::atomic<int32_t> frontier;
    int32_t H;
    std::atomic<uint64_t> vend[kFlagCap];

    uint8_t *heap_base;
    size_t heap_size;  // == bounded ring size

    DistOrchFunc orch_func;
    const L2TaskArgs *orch_args;
    PTO2Runtime *rt;
    Runtime *runtime;  // outer Runtime (for kernel-address resolution + done_count)

    std::atomic<int32_t> fatal;

    // Physical-block topology (1 AIC + 2 AIV per block), derived once at register
    // time from Runtime::workers[].core_type, identical to the centralized
    // scheduler's cluster discovery (AIC core b pairs with the 2b-th / (2b+1)-th
    // AIV cores in worker-index order).
    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[RUNTIME_MAX_WORKER];
    BlockWon blocks[RUNTIME_MAX_WORKER];  // indexed by block_id (<= num AIC)

    // Global "all cores finished orchestration replay" counter. A follower must
    // not conclude "no more pushes are coming for my lane" until every core has
    // finished replaying the submit stream (§7 tail-idle).
    std::atomic<int32_t> replay_done;

    DistCore cores[RUNTIME_MAX_WORKER];
};

DistGlobal g_dist;
thread_local DistCore *g_self = nullptr;

// Swimlane tracing (set PTO_DIST_SWIMLANE=<path> to enable). When off, the
// per-task timing capture is skipped entirely so a normal run pays nothing.
bool g_trace_on = false;
uint64_t g_trace_epoch_ns = 0;

// Orchestration/scheduling overhead isolation (set PTO_DIST_SKIP_EXEC=1). When
// on, execute_slot skips the actual incore kernel call — every (sub)task is
// treated as 0-cost and "completes" instantly — while ALL ownership/completion
// bookkeeping runs unchanged, so the loop terminates identically. This lets a
// benchmark measure the pure cost of on-core orchestration + claim race +
// scheduling, independent of kernel work. Outputs are NOT computed (run with
// golden checks disabled). See examples/.../runtime_overhead_test.
bool g_skip_exec = false;

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Opt-in per-core tracing (set PTO_DIST_TRACE=1). Off by default so a passing
// run is quiet; fatal/error/heap-exhaustion diagnostics are always emitted.
inline bool dist_trace() {
    static const bool on = (getenv("PTO_DIST_TRACE") != nullptr);
    return on;
}

// -----------------------------------------------------------------------------
// Fatal / claim / execution helpers
// -----------------------------------------------------------------------------
inline bool fatal_set() { return g_dist.fatal.load(std::memory_order_acquire) != 0; }
inline void set_fatal() { g_dist.fatal.store(1, std::memory_order_release); }

void dist_dump_state(int);  // defined below; dumps full engine state for hangs

// Env-gated stall watchdog (set PTO_DIST_WATCHDOG=<seconds>, default off). Called
// from inside the engine's spin loops on a worker thread (so fprintf is safe,
// unlike a signal handler). On the first call it records a start time; if a loop
// keeps spinning past the budget the engine is presumed deadlocked, so it dumps
// the full state once and sets fatal to unwind every core for a fast, diagnosed
// failure instead of an indefinite hang.
inline void watchdog(uint64_t &start_ns) {
    static const long budget_s = []() -> long {
        const char *e = getenv("PTO_DIST_WATCHDOG");
        return e ? atol(e) : 0;
    }();
    if (budget_s <= 0) return;
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
    if (start_ns == 0) {
        start_ns = now;
        return;
    }
    if (now - start_ns > static_cast<uint64_t>(budget_s) * 1000000000ull) {
        static std::atomic<int32_t> dumped{0};
        int32_t exp = 0;
        if (dumped.compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
            fprintf(stderr, "[dist_engine] WATCHDOG fired after %lds — presumed deadlock, dumping state\n", budget_s);
            dist_dump_state(0);
        }
        set_fatal();
    }
}

// CAS-loop fetch_max (§11.1): returns true (WON) iff this core advanced the
// cursor to N. No hardware fetch_max on the target, so this is the equivalent
// acq-rel CAS retry. Monotonic: each task id is claimed by exactly one core and
// no id is skipped within a cursor's subsequence.
bool claim(std::atomic<int32_t> &cursor, int32_t N) {
    int32_t c = cursor.load(std::memory_order_acquire);
    while (true) {
        if (N <= c) return false;
        if (cursor.compare_exchange_weak(c, N, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
    }
}

// Cooperatively advance the global completion frontier F (§11.4): after any core
// publishes flag(N), the contiguous-done prefix may have grown, so any core walks
// F forward while flag(F+1) is set. Lock-free; the CAS makes exactly one core win
// each step and the cost is amortized across all cores.
void advance_frontier() {
    int32_t f = g_dist.frontier.load(std::memory_order_acquire);
    while (true) {
        const int32_t next = f + 1;
        if (next >= kFlagCap) break;
        if (g_dist.flags[next & (kFlagCap - 1)].load(std::memory_order_acquire) == 0) break;
        if (g_dist.frontier.compare_exchange_weak(f, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            f = next;
        }
        // On CAS failure f was reloaded with the current value; retry.
    }
}

// Resolve a kernel id to its executable address (CoreCallable::resolved_addr()).
uint64_t resolve_kernel_addr(Runtime *runtime, int32_t kernel_id) {
    if (kernel_id == INVALID_KERNEL_ID) return 0;
    uint64_t callable_addr = runtime->get_function_bin_addr(kernel_id);
    if (callable_addr == 0) return 0;
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    return callable->resolved_addr();
}

// Execute one owned task, then publish its completion flag (release). In sim all
// cores share the address space, so the release/acquire pair is the visibility
// barrier between the kernel's output writes and a consumer's input reads.
void execute_slot(DistCore *self, RingSlot &s) {
    typedef void (*KernelFn)(int64_t *);
    // PTO_DIST_SKIP_EXEC: treat the incore task as 0-cost — skip the kernel call
    // but keep every flag/frontier/slot update below so termination is identical.
    if (s.function_bin_addr != 0 && !g_skip_exec) {
        if (g_trace_on) {
            const uint64_t t0 = now_ns();
            KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
            fn(reinterpret_cast<int64_t *>(s.args));
            const uint64_t t1 = now_ns();
            self->trace.push_back(TraceEvent{
                s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0),
                (t0 - g_trace_epoch_ns) / 1000.0, (t1 - t0) / 1000.0});
        } else {
            KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
            fn(reinterpret_cast<int64_t *>(s.args));
        }
    }
    if (s.is_multicore) {
        // Joint ownership: the co-owner that drives remaining to zero (the last
        // subtask to finish) publishes the single global completion flag (§3.1),
        // then frees the block.won entry for reuse.
        WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
        if (w.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            g_dist.flags[s.task_id & (kFlagCap - 1)].store(1, std::memory_order_release);
            w.state.store(0, std::memory_order_release);  // recycle the id-keyed slot
            advance_frontier();
        }
    } else {
        g_dist.flags[s.task_id & (kFlagCap - 1)].store(1, std::memory_order_release);
        advance_frontier();
    }
    s.built = false;
    s.occupied = false;
}

// Phase B: execute every ready owned task in the private ring. A task is ready
// once all its fan-in producers have set their completion flag (acquire).
// Returns the number of slots freed this pass.
int32_t drain_phase_b(DistCore *self) {
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;  // skip reserved-but-unbuilt slots
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (g_dist.flags[s.fanin[f] & (kFlagCap - 1)].load(std::memory_order_acquire) == 0) {
                ready = false;
                break;
            }
        }
        if (!ready) continue;
        execute_slot(self, s);
        self->occupied_count--;
        freed++;
    }
    return freed;
}

int32_t alloc_ring_slot(DistCore *self) {
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (!self->slots[i].occupied) return i;
    }
    return -1;
}

// Kernel id for a physical lane (AIC/AIV0/AIV1) of a MixedKernels.
inline int32_t kernel_id_for_lane(const MixedKernels &mixed, int32_t lane) {
    switch (lane) {
        case LANE_AIC: return mixed.aic_kernel_id;
        case LANE_AIV0: return mixed.aiv0_kernel_id;
        case LANE_AIV1: return mixed.aiv1_kernel_id;
        default: return INVALID_KERNEL_ID;
    }
}

inline bool lane_active(const ActiveMask &M, int32_t lane) {
    return M.subtask_active(static_cast<PTO2SubtaskSlot>(lane));
}

// Materialize a private-ring slot from already-resolved components (shared by the
// owner build path and the follower drain path). `tensors`/`scalars` are copied
// in; args[] is (re)built to point at this slot's own copies so the slot is
// self-contained and executable at any later time.
void build_ring_slot(
    RingSlot &s, int32_t task_id, int32_t func_id, uint64_t fn_addr, const Tensor *tensors, int32_t tc,
    const uint64_t *scalars, int32_t sc, const int32_t *fanin, int32_t fc, int32_t sub_block_id, bool is_multicore,
    int32_t won_block, int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = func_id;
    s.function_bin_addr = fn_addr;
    s.built = true;  // fully populated below — now safe for Phase B to execute
    s.tensor_count = tc;
    s.scalar_count = sc;
    for (int32_t i = 0; i < tc; i++) s.tensors[i].copy(tensors[i]);
    for (int32_t j = 0; j < sc; j++) s.scalars[j] = scalars[j];
    int32_t n = 0;
    for (int32_t i = 0; i < tc; i++) s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < sc; j++) s.args[n++] = s.scalars[j];
    s.local_ctx.block_idx = 0;
    s.local_ctx.block_num = 1;
    s.local_ctx.async_ctx = AsyncCtx{};
    s.global_ctx.sub_block_id = sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = fc;
    for (int32_t k = 0; k < fc; k++) s.fanin[k] = fanin[k];
    s.is_multicore = is_multicore;
    s.won_block = won_block;
    s.won_slot = won_slot;
}

// Reserve a free block.won slot in `block`. Returns slot index or -1 if full.
// 2V allows either AIV of the block to be an anchor, so allocation must be atomic.
int32_t alloc_won_slot(int32_t block) {
    BlockWon &bw = g_dist.blocks[block];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        int32_t exp = 0;
        if (bw.slots[i].state.compare_exchange_strong(exp, 2, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return i;
        }
    }
    return -1;
}

// True if a published block.won deposit for this core's lane has not yet been
// taken — used by the termination check to avoid finishing before draining.
bool has_pending_won(DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    BlockWon &bw = g_dist.blocks[self->block_id];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        WonSlot &w = bw.slots[i];
        if (w.state.load(std::memory_order_acquire) != 1) continue;
        if (w.lane[self->lane].present && w.drained[self->lane].load(std::memory_order_acquire) == 0) return true;
    }
    return false;
}

// Follower drain (§3.1, §6): pull every published block.won subtask addressed to
// this core's physical lane that we have not yet taken, building each into a free
// private-ring slot (back-pressure: stop when the ring is full). Non-blocking —
// if nothing is addressed to us we simply return.
void drain_block_won(DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return;  // AIC is never a follower
    BlockWon &bw = g_dist.blocks[self->block_id];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        WonSlot &w = bw.slots[i];
        if (w.state.load(std::memory_order_acquire) != 1) continue;
        if (!w.lane[self->lane].present) continue;
        int32_t exp = 0;
        if (!w.drained[self->lane].compare_exchange_strong(exp, 1, std::memory_order_acq_rel, std::memory_order_relaxed))
            continue;  // already taken by us on a prior pass
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            // Ring full: hand the deposit back and let Phase B free a slot first.
            w.drained[self->lane].store(0, std::memory_order_release);
            return;
        }
        const BuiltSubtask &b = w.lane[self->lane];
        build_ring_slot(
            self->slots[si], w.task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars,
            b.scalar_count, b.fanin, b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, i
        );
        self->occupied_count++;
        self->owned_total++;
    }
}

// -----------------------------------------------------------------------------
// Distributed submit op (replaces the centralized orchestrator submit).
//
// Every core runs this for every task (identical replay): materialize outputs
// at deterministic heap addresses, maintain the per-core producer map, then
// race to claim ownership. Only the winner builds the task into its private
// ring; losers return with map + outputs updated so downstream get_ref() and
// fan-in resolution stay consistent across cores.
// -----------------------------------------------------------------------------
TaskOutputTensors dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    DistCore *self = g_self;
    if (self == nullptr) return TaskOutputTensors{};
    Runtime *runtime = g_dist.runtime;

    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): before claiming this task, pull any
    // follower deposits and execute every ready owned task. This interleaves
    // execution with claiming so a fast core does not burst-claim a full ring of
    // consecutive tasks; while it executes a (long) task other cores advance the
    // cursor and claim subsequent ones. The deterministic replay below (id bump,
    // heap bump, map maintenance) is unaffected — draining only runs/flags tasks
    // this core already owns. Every core does this on every submit point.
    if (!fatal_set()) {
        drain_block_won(self);
        drain_phase_b(self);
    }

    const int32_t N = self->local_index++;
    const ActiveMask M = mixed.to_active_mask();
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {  // flag ring + vend[] are non-windowed; cap total tasks
        set_fatal();
        fprintf(stderr, "[dist_engine] task id %d exceeds kFlagCap %d (enlarge or window the flag/vend rings)\n", N,
                kFlagCap);
        return TaskOutputTensors{};
    }

    // (a) Deterministic GM output-heap allocation + materialization (§9.3, §11.4).
    // The virtual bump `heap_next` is unbounded and identical on every core; the
    // PHYSICAL address is (virtual mod ring). First sum this task's aligned output
    // bytes so we can keep the whole task within one ring lap: if it would straddle
    // the ring end, pad the virtual base up to the next ring boundary (deterministic
    // → every core agrees). A single task larger than the ring is unsatisfiable.
    const size_t ring = g_dist.heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
            fprintf(stderr, "[dist_engine] task %d outputs %llu B exceed heap ring %zu B (enlarge PTO_DIST_HEAP_MB)\n",
                    N, (unsigned long long)total, ring);
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) {
            task_base = ((task_base / ring) + 1) * ring;  // skip the ring tail; start next lap
        }
    }
    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = ci.buffer_size_bytes();
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            fprintf(stderr, "[dist_engine] GM output heap not allocated at task %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;  // straddle-pad guarantees phys+logical <= ring
        Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    // Publish cumulative virtual bytes through task N so any core can derive the
    // live window [vend[R], heap_next) for reclaim back-pressure. Deterministic, so
    // all cores store the same value (this core also reads its own writes for R<N).
    if (N >= 0 && N < kFlagCap) g_dist.vend[N].store(self->heap_next, std::memory_order_relaxed);

    // Once fatal, stop claiming/executing but keep replaying the deterministic
    // allocation above so this task's `result` carries valid (materialized) output
    // refs — the orchestration may still call get_ref() on them. This degrades a
    // fatal (e.g. heap-too-small) into a clean wrong-answer failure + diagnostic
    // rather than an assertion crash mid-replay.
    if (fatal_set()) return result;

    // Retire producer-map entries that have left the H span (deterministic,
    // N-derived) before this task's lookups/inserts. Bounds chain length so
    // submit stays ~O(N) instead of O(N^2). See DistTensorMap.
    self->map.advance_retire(N, g_dist.H);

    // (b) Fan-in resolution: look up producers of INPUT/INOUT regions BEFORE
    // this task registers its own writes.
    int32_t fanin[kMaxFanin];
    int32_t fc = 0;
    for (int32_t i = 0; i < tc; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
        const Tensor &t = args.tensor(i).ref();
        if (t.manual_dep) continue;
        const int32_t p = self->map.lookup(t);
        if (p < 0) continue;
        bool dup = false;
        for (int32_t k = 0; k < fc; k++)
            if (fanin[k] == p) {
                dup = true;
                break;
            }
        if (!dup && fc < kMaxFanin) fanin[fc++] = p;
    }

    // (c) Register this task as the producer of its OUTPUT / INOUT / existing
    // outputs (every core, so all maps stay identical).
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < tc; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) {
            self->map.insert(result.get_ref(out_idx), N);
            out_idx++;
        } else if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
            self->map.insert(args.tensor(i).ref(), N);
        }
    }

    // (d) Anchor type + claim race. Competition is by anchor TYPE (§2/§3.1): cube
    // tasks (any task with an AIC subtask) are contested by AIC cores; vector tasks
    // (AIV-only, incl. 2V) by AIV cores (AIV0 and AIV1 equally). Resolved from the
    // mask alone — no per-task Tensor copies — so losers bail out cheaply here.
    const uint8_t cmask = M.core_mask();
    const int32_t pc = __builtin_popcount(cmask);
    const bool has_aic = (cmask & PTO2_SUBTASK_MASK_AIC) != 0;
    const bool anchor_is_cube = has_aic;
    const bool type_match =
        anchor_is_cube ? (self->role == CoreType::AIC) : (self->role == CoreType::AIV);
    if (!type_match) return result;  // wrong type for this task: only TensorMap was updated

    std::atomic<int32_t> &cursor = anchor_is_cube ? g_dist.cube_cursor : g_dist.vector_cursor;
    if (!claim(cursor, N)) return result;  // lost the race (another core of this type owns N)

    // (e) Winner only: assemble the shared argument Tensors (identical for every
    // active lane of a multi-core task — they share the task tensors, each lane
    // writing its designated output per the kernels). Inputs are copied from the
    // args; outputs are the materialized heap-addressed descriptors. Done AFTER
    // the claim so the ~2/3 of cores that fail type_match / lose the race never
    // pay these tc x sizeof(Tensor) copies.
    const uint64_t *scalars = args.scalars();
    const int32_t sc = args.scalar_count();
    Tensor built[MAX_TENSOR_ARGS];
    {
        uint32_t bo = 0;
        for (int32_t i = 0; i < tc; i++) {
            if (args.tag(i) == TensorArgType::OUTPUT) {
                built[i].copy(result.get_ref(bo));
                bo++;
            } else {
                built[i].copy(args.tensor(i).ref());
            }
        }
    }

    // ---- Winner = owner (single-core) / anchor (multi-core). ----
    // Back-pressure for self-claimed work: wait until the ring has a non-reserved
    // slot free, draining block.won deposits + ready tasks meanwhile. The reserve
    // guarantees a follower can still pull its (ready) deposits when the rest of
    // the ring is full of not-yet-ready consumers (no priority inversion).
    uint64_t wd_self = 0;
    while (self->occupied_count >= kPrivateSlots - kWonReserve && !fatal_set()) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd_self);
        }
    }
    if (fatal_set()) return result;

    // Heap reclaim back-pressure (§9.5/§11.4): this owner is about to build (and
    // later write) task N's outputs at deterministic physical offsets. Recycling a
    // ring region is safe only once its previous occupant's task id <= R = F - H
    // (all that occupant's consumers, which have id <= occupant+H, are done). The
    // equivalent global-derivable test is: the live virtual window (heap_next minus
    // vend[R]) must fit in the ring. Spin (draining + advancing F) until it does.
    if (g_dist.heap_base != nullptr) {
        const size_t ring = g_dist.heap_size;
        uint64_t wd_heap = 0;
        while (!fatal_set()) {
            const int32_t f = g_dist.frontier.load(std::memory_order_acquire);
            const int32_t R = f - g_dist.H;
            const uint64_t vstart_live =
                (R < 0) ? 0 : g_dist.vend[R].load(std::memory_order_relaxed);
            if (self->heap_next - vstart_live <= ring) break;  // window fits — region free
            if (f >= N - 1) {  // every predecessor done yet H-window still overflows the ring
                set_fatal();
                fprintf(stderr,
                        "[dist_engine] heap ring %zu B too small for H=%d window at task %d (live=%llu B); "
                        "enlarge PTO_DIST_HEAP_MB or reduce PTO_DIST_H\n",
                        ring, g_dist.H, N, (unsigned long long)(self->heap_next - vstart_live));
                return result;
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_heap);
            }
        }
        if (fatal_set()) return result;
    }

    int32_t si = alloc_ring_slot(self);
    if (si < 0) {  // should not happen given the back-pressure gate above
        set_fatal();
        fprintf(stderr, "[dist_engine] no free private-ring slot after back-pressure at task %d\n", N);
        return result;
    }
    // Reserve so concurrent drains (including the block.won back-pressure loop
    // below, which calls drain_phase_b) do not reuse this slot. Mark it unbuilt
    // so Phase B skips it until build_ring_slot populates it (avoids re-executing
    // the prior occupant's stale task_id/fanin/won linkage).
    self->slots[si].occupied = true;
    self->slots[si].built = false;

    int32_t own_lane;
    int32_t won_block = -1;
    int32_t won_slot = -1;
    bool is_multicore = (pc > 1);

    if (!is_multicore) {
        // Single core (1C / 1V): the one active lane is the only subtask. For 1V
        // the winner may be physically AIV0 or AIV1, but the active lane/kernel is
        // AIV0 (rt_submit_aiv fills aiv0). Find the single active lane.
        own_lane = has_aic ? LANE_AIC : LANE_AIV0;
    } else {
        // Multi-core (MIX / 2V): we are the anchor. Our own physical lane subtask
        // goes to our private ring; the remaining active lanes are deposited into
        // block.won for our same-block followers to drain (§3.1).
        own_lane = self->lane;
        won_block = self->block_id;
        won_slot = alloc_won_slot(won_block);
        uint64_t wd_won = 0;
        while (won_slot < 0 && !fatal_set()) {  // block.won full → back-pressure (drain, then retry)
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_won);
            }
            won_slot = alloc_won_slot(won_block);
        }
        if (fatal_set()) return result;
        WonSlot &w = g_dist.blocks[won_block].slots[won_slot];
        w.task_id = N;
        w.remaining.store(pc, std::memory_order_relaxed);
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            w.drained[L].store(0, std::memory_order_relaxed);
            w.lane[L].present = false;
        }
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            if (L == own_lane || !lane_active(M, L)) continue;
            BuiltSubtask &b = w.lane[L];
            b.present = true;
            b.func_id = kernel_id_for_lane(mixed, L);
            b.function_bin_addr = resolve_kernel_addr(runtime, kernel_id_for_lane(mixed, L));
            b.tensor_count = tc;
            b.scalar_count = sc;
            for (int32_t i = 0; i < tc; i++) b.tensors[i].copy(built[i]);
            for (int32_t j = 0; j < sc; j++) b.scalars[j] = scalars[j];
            b.fanin_count = fc;
            for (int32_t k = 0; k < fc; k++) b.fanin[k] = fanin[k];
            b.sub_block_id = (L == LANE_AIV1) ? 1 : 0;
        }
        std::atomic_thread_fence(std::memory_order_release);
        w.state.store(1, std::memory_order_release);  // publish the deposits to followers
    }

    const int32_t own_sub_block = (own_lane == LANE_AIV1) ? 1 : 0;
    const int32_t own_func_id = kernel_id_for_lane(mixed, own_lane);
    build_ring_slot(
        self->slots[si], N, own_func_id, resolve_kernel_addr(runtime, own_func_id), built, tc, scalars, sc, fanin, fc,
        own_sub_block, is_multicore, won_block, won_slot
    );
    self->occupied_count++;
    self->owned_total++;

    return result;
}

// -----------------------------------------------------------------------------
// Remaining ops — minimal stubs (bgemm exercises submit/scope/log only).
// -----------------------------------------------------------------------------
void dist_scope_begin(PTO2Runtime *) {}
void dist_scope_end(PTO2Runtime *) {}
void dist_orchestration_done(PTO2Runtime *) {}
bool dist_is_fatal(PTO2Runtime *) { return fatal_set(); }

void dist_report_fatal(PTO2Runtime *, int32_t code, const char *func, const char *fmt, ...) {
    set_fatal();
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][FATAL][%s] code=%d: ", func ? func : "?", code);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

void dist_log_error(const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][E][%s] ", func ? func : "?");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
void dist_log_warn(const char *, const char *, ...) {}
void dist_log_debug(const char *, const char *, ...) {}
void dist_log_info_v(const char *, int, const char *, ...) {}

// Orchestration-side tensor data access (get/set_tensor_data). Replay runs on the
// AICore worker and reads/writes real GM, so these are genuine memory accesses.
// The only subtlety is read-after-write across tasks: if the region has a producer
// in this core's map, wait until that producer's completion flag is set (draining
// this core's own ring meanwhile so an owned producer actually runs). External
// tensors (no producer) are accessed immediately. Consumer (WAR) tracking is not
// modeled, mirroring the centralized runtime's documented INPUT-reader limitation.
void wait_producer_ready(DistCore *self, const Tensor &t) {
    // Cold path (get/set_tensor_data); uses the map's current alive_floor.
    const int32_t p = self->map.lookup(t);
    if (p < 0) return;
    uint64_t wd = 0;
    while (!fatal_set()) {
        if (g_dist.flags[p & (kFlagCap - 1)].load(std::memory_order_acquire) != 0) break;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd);
        }
    }
}

uint64_t dist_get_tensor_data(PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t *indices) {
    if (tensor.buffer.addr == 0) return 0;
    DistCore *self = g_self;
    if (self != nullptr) wait_producer_ready(self, tensor);
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    uint64_t result = 0;
    memcpy(&result, reinterpret_cast<const void *>(tensor.buffer.addr + flat * esz), esz);
    return result;
}

void dist_set_tensor_data(PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t *indices, uint64_t value) {
    if (tensor.buffer.addr == 0) return;
    DistCore *self = g_self;
    if (self != nullptr) wait_producer_ready(self, tensor);
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    memcpy(reinterpret_cast<void *>(tensor.buffer.addr + flat * esz), &value, esz);
}

// alloc_tensors — a kernel-less "hidden task" that only reserves GM output
// buffers (no compute). It consumes one task id, allocates its outputs on the
// deterministic heap exactly like dist_submit_impl step (a), registers itself as
// their producer, and completes INLINE (sets its own flag immediately) since no
// kernel runs. A later writer (INOUT / OUTPUT_EXISTING) becomes the new producer
// of the region, so real consumers depend on the writer, not on this alloc. Every
// core replays it identically, keeping heap addresses + maps consistent.
TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistCore *self = g_self;
    if (self == nullptr) return TaskOutputTensors{};
    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): every submit point first seeks an
    // execution opportunity before advancing the deterministic replay below.
    if (!fatal_set()) {
        drain_block_won(self);
        drain_phase_b(self);
    }
    const int32_t N = self->local_index++;
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {
        set_fatal();
        fprintf(stderr, "[dist_engine] alloc task id %d exceeds kFlagCap %d\n", N, kFlagCap);
        return TaskOutputTensors{};
    }

    // Deterministic GM heap allocation + straddle-padding (identical to submit (a)).
    const size_t ring = g_dist.heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
            fprintf(stderr, "[dist_engine] alloc task %d outputs %llu B exceed heap ring %zu B\n", N,
                    (unsigned long long)total, ring);
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) task_base = ((task_base / ring) + 1) * ring;
    }

    // Heap reclaim back-pressure (same window test as submit). An alloc bumps the
    // heap like any output; drain this core's ring while the live window overflows.
    if (total > 0 && g_dist.heap_base != nullptr) {
        const uint64_t want_next = task_base + total;
        uint64_t wd_heap = 0;
        while (!fatal_set()) {
            const int32_t f = g_dist.frontier.load(std::memory_order_acquire);
            const int32_t R = f - g_dist.H;
            const uint64_t vstart_live = (R < 0) ? 0 : g_dist.vend[R].load(std::memory_order_relaxed);
            if (want_next - vstart_live <= ring) break;
            if (f >= N - 1) {
                set_fatal();
                fprintf(stderr,
                        "[dist_engine] heap ring %zu B too small for H=%d window at alloc %d (live=%llu B)\n",
                        ring, g_dist.H, N, (unsigned long long)(want_next - vstart_live));
                return TaskOutputTensors{};
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(wd_heap);
            }
        }
        if (fatal_set()) return TaskOutputTensors{};
    }

    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = ci.buffer_size_bytes();
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            fprintf(stderr, "[dist_engine] GM output heap not allocated at alloc %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;
        Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    if (N >= 0 && N < kFlagCap) g_dist.vend[N].store(self->heap_next, std::memory_order_relaxed);
    if (fatal_set()) return result;

    // Register producer for each allocated output, then complete inline (no kernel).
    self->map.advance_retire(N, g_dist.H);
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        self->map.insert(result.get_ref(out_idx), N);
        out_idx++;
    }
    g_dist.flags[N & (kFlagCap - 1)].store(1, std::memory_order_release);
    advance_frontier();
    return result;
}

TaskOutputTensors dist_submit_dummy(PTO2Runtime *, const L0TaskArgs &) { return TaskOutputTensors{}; }
void dist_scope_set_site(const char *, int) {}

const PTO2RuntimeOps g_dist_ops = {
    dist_submit_impl,    dist_scope_begin,   dist_scope_end,      dist_orchestration_done,
    dist_is_fatal,       dist_report_fatal,  dist_log_error,      dist_log_warn,
    dist_log_debug,      dist_log_info_v,    dist_get_tensor_data, dist_set_tensor_data,
    dist_alloc_tensors,  dist_submit_dummy,  dist_scope_set_site,
};

// -----------------------------------------------------------------------------
// Deadlock diagnostics: dump the full engine state on SIGUSR1. Sim runs every
// core as a pthread in one process, so a single handler can walk g_dist. Used to
// debug hangs (kill -USR1 <pid>); compiled in but inert unless signalled.
// -----------------------------------------------------------------------------
void dist_dump_state(int) {
    fprintf(stderr, "\n===== DIST STATE DUMP =====\n");
    fprintf(stderr, "cube_cursor=%d vector_cursor=%d frontier=%d H=%d ring=%zuB replay_done=%d/%d num_blocks=%d fatal=%d\n",
            g_dist.cube_cursor.load(), g_dist.vector_cursor.load(), g_dist.frontier.load(), g_dist.H,
            g_dist.heap_size, g_dist.replay_done.load(), g_dist.num_workers, g_dist.num_blocks, g_dist.fatal.load());
    for (int32_t c = 0; c < g_dist.num_workers && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        fprintf(stderr, "core %d role=%d blk=%d lane=%d replayed=%d occ=%d owned=%d\n", c,
                static_cast<int>(co.role), co.block_id, co.lane, co.local_index, co.occupied_count, co.owned_total);
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            RingSlot &s = co.slots[i];
            if (!s.occupied) continue;
            int32_t unmet = -1;
            for (int32_t f = 0; f < s.fanin_count; f++)
                if (g_dist.flags[s.fanin[f] & (kFlagCap - 1)].load() == 0) { unmet = s.fanin[f]; break; }
            fprintf(stderr, "    slot%d tid=%d built=%d mc=%d won=(%d,%d) fanin=%d unmet=%d\n", i, s.task_id,
                    s.built, s.is_multicore, s.won_block, s.won_slot, s.fanin_count, unmet);
        }
    }
    for (int32_t b = 0; b < g_dist.num_blocks; b++) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            WonSlot &w = g_dist.blocks[b].slots[i];
            int32_t st = w.state.load();
            if (st == 0) continue;
            fprintf(stderr, "  won blk%d slot%d state=%d tid=%d remaining=%d drained=[%d,%d,%d] present=[%d,%d,%d]\n",
                    b, i, st, w.task_id, w.remaining.load(), w.drained[0].load(), w.drained[1].load(),
                    w.drained[2].load(), w.lane[0].present, w.lane[1].present, w.lane[2].present);
        }
    }
    fprintf(stderr, "===== END DUMP =====\n");
}

// -----------------------------------------------------------------------------
// Per-core entry point invoked by each AICore worker thread.
// -----------------------------------------------------------------------------
void dist_core_main(void *runtime_v, int core_idx, int core_type_int) {
    if (core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return;
    Runtime *runtime = reinterpret_cast<Runtime *>(runtime_v);
    DistCore *self = &g_dist.cores[core_idx];
    const CoreType role = static_cast<CoreType>(core_type_int);

    // sub_block lane: only meaningful for AIV in MIX tasks (M3). bgemm's 1V add
    // ignores it, so 0 is correct for the M2 single-core scope.
    const CoreLayout lay = g_dist.layout[core_idx];
    self->reset(role, lay.block_id, lay.lane);
    self->core_idx = core_idx;
    g_self = self;
    if (dist_trace()) fprintf(stderr, "[dist] core %d role=%d block=%d lane=%d START\n", core_idx, core_type_int,
                              lay.block_id, lay.lane);

    // Replay the full orchestration submit stream: build the per-core map and
    // claim/build owned tasks into the private ring (back-pressure inline). MIX
    // anchors deposit follower subtasks into block.won during this replay.
    if (g_dist.orch_func != nullptr && g_dist.orch_args != nullptr && !fatal_set()) {
        g_dist.orch_func(*g_dist.orch_args);
    }

    // Publish "my replay is done" so followers can eventually conclude that no
    // further block.won deposits will arrive for them (§7 tail-idle).
    g_dist.replay_done.fetch_add(1, std::memory_order_acq_rel);

    // Drain to completion: pull any follower deposits addressed to my lane, run
    // ready tasks, and only finish once every core has finished replay (no more
    // pushes), my private ring is empty, and there is no undrained deposit left
    // for my lane.
    uint64_t wd_drain = 0;
    while (!fatal_set()) {
        drain_block_won(self);
        int32_t freed = drain_phase_b(self);
        const bool all_replayed = g_dist.replay_done.load(std::memory_order_acquire) >= g_dist.num_workers;
        const bool ring_empty = (self->occupied_count == 0);
        const bool pending = has_pending_won(self);
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd_drain);
        }
    }

    if (dist_trace() || fatal_set()) {
        fprintf(stderr, "[dist] core %d role=%d DONE replayed=%d owned=%d fatal=%d\n", core_idx, core_type_int,
                self->local_index, self->owned_total, fatal_set() ? 1 : 0);
    }
    g_self = nullptr;
    __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
}

}  // namespace

void *dist_engine_register(
    PTO2Runtime *rt, DistOrchFunc orch_func, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime
) {
    // GM output heap: a BOUNDED ring reclaimed by the completion frontier (M4).
    // Size from PTO_DIST_HEAP_MB (MiB) else kHeapRingDefault. Allocated once per
    // process; if a later run needs a different size, free + realloc.
    {
        size_t want = kHeapRingDefault;
        if (const char *e = getenv("PTO_DIST_HEAP_MB")) {
            const long mb = atol(e);
            if (mb > 0) want = static_cast<size_t>(mb) << 20;
        }
        if (g_dist.heap_base != nullptr && g_dist.heap_size != want) {
            free(g_dist.heap_base);
            g_dist.heap_base = nullptr;
        }
        if (g_dist.heap_base == nullptr) {
            g_dist.heap_base = static_cast<uint8_t *>(malloc(want));
            g_dist.heap_size = (g_dist.heap_base != nullptr) ? want : 0;
        }
        // Zero the heap each run so freshly-allocated output regions read as 0,
        // matching the centralized runtime's zero-initialized GM. Kernels that
        // read a padded tile (e.g. softmax/PV where valid_len < tile width) rely
        // on the unwritten remainder being zero; an uninitialized (malloc) or
        // recycled heap would otherwise yield nondeterministic results.
        if (g_dist.heap_base != nullptr) memset(g_dist.heap_base, 0, g_dist.heap_size);
    }
    // Dependency-span bound H (R = F - H). Env override for graphs with longer
    // heap spans; default kHDefault.
    g_dist.H = kHDefault;
    if (const char *e = getenv("PTO_DIST_H")) {
        const long h = atol(e);
        if (h >= 0) g_dist.H = static_cast<int32_t>(h);
    }
    // The producer map recycles a task's entry-head slot kTaskWindow tasks later;
    // cleanup retires a task once it leaves the H span, so H must stay below the
    // window (with margin) or a slot could be reused before its task is cleaned.
    always_assert(g_dist.H < kTaskWindow - 1);
    // Swimlane tracing gate. Capture the epoch now so every core's event ts is
    // relative to the same run start.
    g_trace_on = (getenv("PTO_DIST_SWIMLANE") != nullptr);
    g_trace_epoch_ns = now_ns();
    // Overhead-isolation gate (skip incore kernel calls, keep all bookkeeping).
    g_skip_exec = (getenv("PTO_DIST_SKIP_EXEC") != nullptr);

    g_dist.cube_cursor.store(-1, std::memory_order_relaxed);
    g_dist.vector_cursor.store(-1, std::memory_order_relaxed);
    g_dist.frontier.store(-1, std::memory_order_relaxed);
    for (int32_t i = 0; i < kFlagCap; i++) g_dist.flags[i].store(0, std::memory_order_relaxed);
    g_dist.fatal.store(0, std::memory_order_relaxed);
    g_dist.replay_done.store(0, std::memory_order_relaxed);
    g_dist.orch_func = orch_func;
    g_dist.orch_args = orch_args;
    g_dist.rt = rt;
    g_dist.runtime = runtime;

    // Derive the physical-block topology (1 AIC + 2 AIV per block) the same way
    // the centralized scheduler discovers clusters: AIC/AIV cores in worker-index
    // order, AIC[b] paired with AIV[2b] (AIV0) and AIV[2b+1] (AIV1). Followers and
    // anchors use this to address block.won deposits. See §3.1.
    g_dist.num_workers = num_workers;
    int32_t aic_ids[RUNTIME_MAX_WORKER];
    int32_t aiv_ids[RUNTIME_MAX_WORKER];
    int32_t naic = 0, naiv = 0;
    for (int32_t i = 0; i < num_workers && i < RUNTIME_MAX_WORKER; i++) {
        g_dist.layout[i].block_id = -1;
        g_dist.layout[i].lane = LANE_NONE;
        if (runtime->workers[i].core_type == CoreType::AIC) {
            aic_ids[naic++] = i;
        } else {
            aiv_ids[naiv++] = i;
        }
    }
    g_dist.num_blocks = naic;
    for (int32_t b = 0; b < naic; b++) {
        g_dist.layout[aic_ids[b]] = CoreLayout{b, LANE_AIC};
        if (2 * b < naiv) g_dist.layout[aiv_ids[2 * b]] = CoreLayout{b, LANE_AIV0};
        if (2 * b + 1 < naiv) g_dist.layout[aiv_ids[2 * b + 1]] = CoreLayout{b, LANE_AIV1};
        for (int32_t s = 0; s < kPrivateSlots; s++) {
            g_dist.blocks[b].slots[s].state.store(0, std::memory_order_relaxed);
        }
    }

    if (dist_trace()) {
        fprintf(stderr, "[dist] register: num_workers=%d heap_base=%p heap_size=%zu\n", num_workers,
                (void *)g_dist.heap_base, g_dist.heap_size);
    }

    // Install the SIGUSR1 deadlock dumper once, but only when diagnostics are
    // opted in (PTO_DIST_WATCHDOG set) — default runs install no signal handler.
    static bool handler_installed = false;
    if (!handler_installed && getenv("PTO_DIST_WATCHDOG") != nullptr) {
        signal(SIGUSR1, dist_dump_state);
        handler_installed = true;
    }

    // Publish all of the above before any worker observes Runtime::dist.go.
    std::atomic_thread_fence(std::memory_order_release);
    rt->ops = &g_dist_ops;
    return reinterpret_cast<void *>(&dist_core_main);
}

void dist_engine_dump_trace() {
    if (!g_trace_on) return;
    const char *path = getenv("PTO_DIST_SWIMLANE");
    if (path == nullptr || path[0] == '\0') return;
    FILE *f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "[dist_engine] cannot open swimlane file %s for write\n", path);
        return;
    }

    auto lane_name = [](int32_t lane) -> const char * {
        switch (lane) {
            case LANE_AIC: return "AIC";
            case LANE_AIV0: return "AIV0";
            case LANE_AIV1: return "AIV1";
            default: return "?";
        }
    };

    // Chrome Trace Event Format (https://ui.perfetto.dev / chrome://tracing).
    // pid = physical block, tid = lane → each block is a 3-lane swimlane.
    fprintf(f, "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n");
    bool first = true;
    const int32_t nw = g_dist.num_workers;

    // Lane/process name metadata first (so idle lanes still appear).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        if (!first) fprintf(f, ",\n");
        first = false;
        fprintf(f,
                "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d\"}}",
                co.block_id, co.block_id);
        fprintf(f, ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
                   "\"args\":{\"name\":\"%s (core%d)\"}}",
                co.block_id, co.lane, lane_name(co.lane), c);
    }

    // Duration events, one per executed (sub)task.
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            char name[48];
            if (e.func_id >= 0) {
                snprintf(name, sizeof(name), "f%d#%d", e.func_id, e.task_id);
            } else {
                snprintf(name, sizeof(name), "task#%d", e.task_id);
            }
            if (!first) fprintf(f, ",\n");
            first = false;
            fprintf(f,
                    "    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                    "\"args\":{\"task_id\":%d,\"func_id\":%d,\"core\":%d,\"mc\":%d}}",
                    name, co.block_id, co.lane, e.ts_us, e.dur_us, e.task_id, e.func_id, c, e.multicore);
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[dist_engine] swimlane trace written to %s\n", path);
}
