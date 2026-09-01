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
 * tensormap_and_ringbuffer orchestrator implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "dep_compute.h"
#include "runtime_types.h"
#include "shared_memory.h"
#include "tensormap_and_ringbuffer/task_id_encoding.h"
#include "tensormap.h"
#include "types.h"
#include "tensor.h"

#if SIMPLER_DFX
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#endif

// Verify the captured simpler::tmr::Tensor blob size in DepGenRecord matches the runtime
// simpler::tmr::Tensor layout. The platform header defines DEP_GEN_TENSOR_SIZE without
// including runtime/tensor.h, so this check lives at the orch callsite.
static_assert(
    sizeof(simpler::tmr::Tensor) == DEP_GEN_TENSOR_SIZE,
    "DepGenRecord::tensors slot size out of sync with sizeof(simpler::tmr::Tensor)"
);
// DEP_GEN_MAX_EXPLICIT_DEPS is a diagnostic-side capture cap only; the runtime
// imposes no hard cap on explicit dep count. If a submit exceeds this cap,
// dep_gen_aicpu_record_submit() logs and truncates — runtime correctness is
// unaffected, only the captured replay record is truncated.

// Weak fallbacks: dep_gen_collector_aicpu.cpp provides the strong symbols in
// AICPU builds. Host builds (host_build_graph runtime, future dep_gen replay)
// link these no-op stubs so the runtime translation unit is self-contained.
// Visibility is hidden so the HOST .so doesn't export them into the global
// dynamic symbol table where they'd shadow the AICPU .so's strong symbols
// (same pattern as get_sys_cnt_aicpu / chip_swimlane_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(
    uint64_t, bool, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, const uint8_t *, uint8_t,
    int, const int32_t[3]
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// is_dep_gen_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// Heap-ring wrap report, called from the allocator (ring_buffer.h) on each
// wrap. Strong definition lives in the AICPU collector; host builds fall back to
// this weak no-op so the runtime translation unit stays self-contained.
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if SIMPLER_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link chip_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
chip_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                            \
    bool _prof_active = (orch->chip_swimlane_level >= ChipSwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                           \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                         \
    do {                                                                                            \
        if (_prof_active) {                                                                         \
            chip_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                           \
    } while (0)
#elif SIMPLER_DFX
#include "aicpu/device_time.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
chip_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                            \
    bool _prof_active = (orch->chip_swimlane_level >= ChipSwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                    \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                         \
    do {                                                                                            \
        if (_prof_active) {                                                                         \
            _t1 = get_sys_cnt_aicpu();                                                              \
            chip_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                           \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
#endif

static int32_t orch_mark_fatal(OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == SIMPLER_ERROR_NONE || orch->sm_header == nullptr) {
        return SIMPLER_ERROR_NONE;
    }

    int32_t expected = SIMPLER_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if SIMPLER_DFX
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != SIMPLER_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != SIMPLER_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

static uint32_t next_fanin_seen_epoch(OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
            memset(
                orch->fanin_seen_epoch[r], 0,
                static_cast<size_t>(orch->sm_header->rings[r].task_window_size) * sizeof(uint32_t)
            );
        }
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

struct FaninBuilder {
    FaninBuilder(OrchestratorState *orch, FaninPool &spill_pool, uint32_t seen_epoch) :
        count(0),
        wait_count(0),
        spill_start(0),
        orch(orch),
        seen_epoch(seen_epoch),
        spill_pool(spill_pool) {}
    int32_t count{0};       // total fanin edges (all flag combinations)
    int32_t wait_count{0};  // edges carrying DEP_WAIT — sizes readiness accounting
    // Set when reduction cleared DEP_WAIT on an edge to a producer that does
    // not allow early resolve; flushed to TaskPayload::early_dispatch_blocked.
    bool reduced_unflagged_producer{false};
    int32_t spill_start{0};
    OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    FaninPool &spill_pool;
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP];

    template <typename Fn>
    FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    // Mutating walk over the same edges for_each visits, handing the callback
    // the entry itself so flags can be rewritten (used by the reduction pass).
    template <typename Fn>
    void for_each_entry(Fn &&fn) {
        for (int32_t i = 0; i < count; i++) {
            fn(entry_at(i));
        }
    }

    bool mark_seen(uint8_t prod_ring, int32_t prod_slot) {
        if (prod_ring >= CHIP_MAX_RING_DEPTH || prod_slot < 0) {
            return false;
        }
        uint32_t *seen = orch->fanin_seen_epoch[prod_ring];
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }

    // Append a new edge (caller has already claimed the producer's fanout pin).
    void push_edge(FaninSpillEntry &entry, ChipTaskSlotState *prod_state, DepFlags kind) {
        entry.set(prod_state, kind);
        count++;
        if (dep_has_wait(kind)) {
            wait_count++;
        }
    }

    // Dedup path: a producer already recorded this submission is reached again
    // (e.g. as both a creator and a modifier). OR the new flags into the existing
    // edge instead of adding a second one — no extra fanout pin is claimed.
    void or_flags_into_existing(ChipTaskSlotState *prod_state, DepFlags kind) {
        for (int32_t i = 0; i < count; i++) {
            FaninSpillEntry &entry = entry_at(i);
            if (entry.slot_state() == prod_state) {
                accumulate_flags(entry, kind);
                return;
            }
        }
        // mark_seen reported this producer already owns an edge this submission,
        // so it must be present. A miss means the seen-set and the builder
        // disagree — a logic error, not a runtime condition.
        always_assert(false && "or_flags_into_existing: deduped producer missing from the fanin builder");
    }

private:
    // The i-th appended fanin edge: inline for i < CHIP_FANIN_INLINE_CAP, else in
    // the spill ring at the same linear->physical position for_each_fanin_storage
    // walks. Single source of the wrap arithmetic.
    FaninSpillEntry &entry_at(int32_t i) {
        if (i < CHIP_FANIN_INLINE_CAP) {
            return inline_slots[i];
        }
        int32_t spill_idx = i - CHIP_FANIN_INLINE_CAP;
        return spill_pool.base[(spill_start % spill_pool.capacity + spill_idx) % spill_pool.capacity];
    }

    void accumulate_flags(FaninSpillEntry &entry, DepFlags kind) {
        bool had_wait = dep_has_wait(entry.flags());
        entry.add_flags(kind);
        if (!had_wait && dep_has_wait(kind)) {
            wait_count++;
        }
    }
};

static bool append_fanin_or_fail(
    OrchestratorState *orch, uint8_t prod_ring, int32_t prod_slot, ChipTaskSlotState *prod_state,
    TaskId producer_task_id, FaninBuilder *fanin_builder, uint8_t ring_id, DepFlags kind
) {
    // Decide-and-claim under the producer's fanout_lock. Two conditions make this
    // resolved slot a non-dependency, and both must be checked together with the
    // fanout_count++ so the producer cannot slip from live to consumed/reused in
    // between:
    //   (1) Generation mismatch — the producer was CONSUMED, its slot
    //       reset_for_reuse'd and rebound to a newer task. The cached
    //       owner_task_id still resolves to this slot, but it no longer holds our
    //       producer; ++'ing it would corrupt an unrelated task.
    //   (2) Already CONSUMED in place — finished, output ready, no real edge.
    // In either case, adding it to the fanin and bumping fanout_count would leave
    // a stale ++/release pair that desyncs the slot's refcount (rc != fc) and
    // wedges in-order reclaim. Every edge (regardless of DepFlags) claims one
    // fanout_count++ here: it pins the producer's slot across the submit->wire
    // window so it cannot be CONSUMED + reused before wire_fanin_task links the
    // consumer. The pin's release differs by kind — an ordering-only (DEP_WAIT
    // without DEP_RETAIN) edge releases it at wiring; a DEP_RETAIN edge holds it
    // until the consumer's on_task_release. check_and_handle_consumed flips
    // COMPLETED->CONSUMED under the same lock, so the check and the ++ are atomic
    // against the consume. fanout_count is lock-protected per the
    // ChipTaskSlotState contract.
    //
    // Dedup (mark_seen) happens HERE, gated on a live producer — NOT before the
    // gone check. mark_seen keys only on (ring, slot); a stale owner that resolves
    // to a reused slot must not record it as seen, or a later dependency on the
    // live generation in the same submission would hit mark_seen and be skipped
    // without claiming it (dropped edge). A duplicate live producer claims no new
    // pin and adds no new edge; its flags are OR-accumulated into the existing
    // edge so the stronger of several discovery reasons (creator vs modifier vs
    // explicit) always wins, independent of the order they are reached in.
    prod_state->lock_fanout();
    ChipTaskState pstate = prod_state->task_state.load(std::memory_order_acquire);
    bool gone =
        prod_state->task == nullptr ||
        simpler::tmr::task_local_id(prod_state->task->task_id) != simpler::tmr::task_local_id(producer_task_id) ||
        pstate == CHIP_TASK_CONSUMED;
    bool already_seen = !gone && fanin_builder->mark_seen(prod_ring, prod_slot);
    bool claim = !gone && !already_seen;
    int32_t fanout_now = -1;
    if (claim) {
        // Low bits hold the consumer count; bit31 is the scope ref. The consumer
        // count must never carry into bit31 (would corrupt the scope-release
        // flag) — true for any sane fanout (<< 2^31).
        assert(
            (prod_state->fanout_count & ~FANOUT_SCOPE_BIT) < (FANOUT_SCOPE_BIT - 1) &&
            "fanout consumer count overflow into scope bit"
        );
        prod_state->fanout_count++;
        fanout_now = static_cast<int32_t>(prod_state->fanout_count & ~FANOUT_SCOPE_BIT);
    }
    prod_state->unlock_fanout();
#if SIMPLER_ORCH_PROFILING
    // lock + unlock always; one fanout_count store when we actually claim.
    g_orch_args_atomic_count += claim ? 3 : 2;
#endif
    // Dense-fanout diagnostic, emitted outside fanout_lock (no logging under
    // the spinlock). The monotonic fanout_count++ crosses THRESHOLD+1 exactly
    // once, so each dense producer emits one debug message when enabled.
    if (fanout_now == CHIP_DEP_DEGREE_DEBUG_THRESHOLD + 1) {
        LOG_DEBUG(
            "dense dependency: task ring=%u id=%u fanout>%d [orch submit]",
            static_cast<unsigned>(simpler::tmr::task_ring(producer_task_id)),
            simpler::tmr::task_local_id(producer_task_id), CHIP_DEP_DEGREE_DEBUG_THRESHOLD
        );
    }
    // Stale/consumed producer: no edge at all.
    if (gone) {
        return true;
    }
    // Duplicate live producer: fold the flags into the edge already recorded.
    if (already_seen) {
        fanin_builder->or_flags_into_existing(prod_state, kind);
        return true;
    }

    if (fanin_builder->count < CHIP_FANIN_INLINE_CAP) {
        fanin_builder->push_edge(fanin_builder->inline_slots[fanin_builder->count], prod_state, kind);
        return true;
    }

    FaninPool &fanin_pool = fanin_builder->spill_pool;
    if (!fanin_pool.ensure_space(orch->sm_header->rings[ring_id], 1)) {
        orch_mark_fatal(orch, SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
        return false;
    }
    int32_t spill_idx = fanin_pool.top;
    FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        orch_mark_fatal(orch, SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
        return false;
    }
    if (fanin_builder->count == CHIP_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    fanin_builder->push_edge(*entry, prod_state, kind);
    return true;
}

static bool all_claimed_fanin_completed(const FaninBuilder &fanin_builder) {
    if (fanin_builder.count == 0) return true;
    // Only DEP_WAIT edges gate readiness; a retention-only edge never blocks
    // dispatch, so it is treated as satisfied here.
    return fanin_builder.for_each([](ChipTaskSlotState *producer, DepFlags flags) -> bool {
        if (!dep_has_wait(flags)) return true;
        return producer != nullptr && producer->task_state.load(std::memory_order_acquire) >= CHIP_TASK_COMPLETED;
    });
}

static bool all_claimed_fanin_allow_early_resolve(const FaninBuilder &fanin_builder) {
    if (fanin_builder.count == 0) return true;
    return fanin_builder.for_each([](ChipTaskSlotState *producer, DepFlags flags) -> bool {
        if (!dep_has_wait(flags)) return true;
        return producer != nullptr && producer->task_attrs.allow_early_resolve();
    });
}

// Publish this task's frozen WAIT-ancestor bitmap. Runs once per submit, before
// the slot becomes a producer of any later submit (single orchestrator thread),
// so a reader always observes the entry of the task generation that owns the
// slot.
static void publish_wait_reach(OrchestratorState *orch, uint8_t ring, int32_t slot, uint64_t ancestors) {
    orch->wait_reach[ring][slot].ancestors = ancestors;
}

// Resolve a producer slot pointer to its (ring, slot) index. ring_id is
// per-slot invariant; pointer subtraction from the ring's slot_states base
// never dereferences a possibly-reused slot.
static bool slot_index_of(const OrchestratorState *orch, ChipTaskSlotState *p, uint8_t &ring, int32_t &slot) {
    ring = p->ring_id;
    if (ring >= CHIP_MAX_RING_DEPTH) return false;
    const SharedMemoryRingHeader &r = orch->sm_header->rings[ring];
    slot = static_cast<int32_t>(p - r.slot_states);
    return slot >= 0 && static_cast<uint64_t>(slot) < r.task_window_size;
}

// Two-pass bounded transitive reduction of the WAIT graph, run once per submit
// on the fully-built fanin, before it is flushed to the payload and wired.
//
// Pass 1 folds `direct` (this task's WAIT producers, one bit each at distance
// d-1) and `via` (ancestors reachable through another direct WAIT producer q:
// R[q] << d, distance addition), then publishes R[t] = direct | via. Pass 2
// clears DEP_WAIT on any tracked candidate whose bit is set in `via` — some
// direct producer proves a WAIT path p -> ... -> q -> t — demoting
// WAIT|RETAIN to RETAIN and WAIT-only to DEP_NONE (the entry stays in storage
// so its submit-claim pin is released by the !DEP_RETAIN paths). Both passes
// recompute d from the same seqs and fold with OR, so candidate enumeration
// order (creator vs tensormap vs explicit) cannot change the result.
//
// Only DEP_WAIT edges participate in reachability: a RETAIN-only candidate
// sets no bit and merges no ancestors, because the last hop to t must be a
// WAIT edge for the path to carry ordering. A candidate with
// d > WAIT_REACH_WINDOW keeps its WAIT and contributes nothing to R[t] — it
// and all its ancestors are unrepresentable in the window. d == WAIT_REACH_
// WINDOW sets only the direct bit: the shift would be undefined and every
// ancestor of that producer already lies outside t's window.
//
// Every producer in the builder is pinned for this submit: append_fanin_or_fail
// claimed it with fanout_count++ under the producer's fanout_lock, atomically
// with the consumed/generation check, for every edge regardless of DepFlags.
// Slot rebind requires the CONSUMED flip (which requires
// fanout_refcount == fanout_count) plus allocator reclaim, so a pinned
// producer's slot — and therefore its wait_reach entry and seq — still belongs
// to the claimed task for the whole reduction pass. The bitmap is a frozen
// value published at the producer's own submit, so unlike a fanin-pointer
// walk there is no staleness dimension beyond slot identity, and slot identity
// is settled by the pin.
static void reduce_wait_edges(OrchestratorState *orch, FaninBuilder *b, uint8_t ring, int32_t slot, uint64_t seq_t) {
    uint64_t direct = 0;
    uint64_t via = 0;
    b->for_each([&](ChipTaskSlotState *p, DepFlags flags) {
        if (!dep_has_wait(flags)) return;
        uint8_t pring;
        int32_t pslot;
        if (!slot_index_of(orch, p, pring, pslot)) return;
        const WaitReachEntry &entry = orch->wait_reach[pring][pslot];
        uint64_t d = seq_t - entry.seq;  // global seq: ring-independent, unsigned-exact
        if (d == 0 || d > static_cast<uint64_t>(WAIT_REACH_WINDOW)) return;
        direct |= 1ull << (d - 1);
        if (d < static_cast<uint64_t>(WAIT_REACH_WINDOW)) {
            via |= entry.ancestors << d;
        }
    });
    // A via bit lands at index (i + d) for ancestor bit i >= 0 of a producer at
    // distance d >= 1, so index 0 is unreachable: the nearest direct producer
    // can never be proven redundant. A set bit 0 means the shift-merge has
    // drifted (the classic form is shifting by d - 1) and reduction is about to
    // drop an edge nothing covers.
    always_assert((via & 1ull) == 0 && "via bit 0 set: distance-1 producers are not reducible");
    publish_wait_reach(orch, ring, slot, direct | via);
    if (via == 0) return;
    b->for_each_entry([&](FaninSpillEntry &entry) {
        DepFlags f = entry.flags();
        if (!dep_has_wait(f)) return;
        uint8_t pring;
        int32_t pslot;
        if (!slot_index_of(orch, entry.slot_state(), pring, pslot)) return;
        uint64_t d = seq_t - orch->wait_reach[pring][pslot].seq;
        if (d == 0 || d > static_cast<uint64_t>(WAIT_REACH_WINDOW)) return;
        if ((via & (1ull << (d - 1))) == 0) return;
        // A producer that does not allow early resolve never propagates
        // dispatch_fanin. Record that before the edge leaves wait_count, so the
        // early-dispatch target keeps the unit this producer used to hold.
        if (!entry.slot_state()->task_attrs.allow_early_resolve()) {
            b->reduced_unflagged_producer = true;
        }
        entry.set(entry.slot_state(), static_cast<DepFlags>(f & ~DEP_WAIT));
        b->wait_count--;
    });
}

void OrchestratorState::mark_dep_pool_position(ChipTaskSlotState &slot_state) {
    SchedulerState *sched = scheduler;
    auto &rss = sched->ring_sched_states[slot_state.ring_id];
    slot_state.dep_pool_mark = rss.dep_pool.top;
#if SIMPLER_DFX
    if (is_scope_stats_enabled()) {
        rss.publish_dep_pool_snapshot();
    }
#endif
}

void OrchestratorState::wire_fanin_task(ChipTaskSlotState &slot_state, int32_t wfanin) {
    SchedulerState *sched = scheduler;
    auto &rss = sched->ring_sched_states[slot_state.ring_id];
    TaskPayload *payload = slot_state.payload;
    slot_state.fanin_count = wfanin + 1;

    int32_t completed_fanin = 0;
    int32_t early_propagated = 0;
    bool early_disqualified = false;
    for_each_fanin_slot_state(*payload, [&](ChipTaskSlotState *producer, DepFlags flags) {
        // Only DEP_WAIT edges contribute to readiness: they gate fanin and are
        // linked onto the producer's fanout_head for completion notification. A
        // retention-only edge carries no ordering, so it is skipped here.
        if (dep_has_wait(flags)) {
            producer->lock_fanout();
            int32_t pstate = producer->task_state.load(std::memory_order_acquire);
            // Only WAIT producers reach this check. An unflagged producer whose
            // edge reduction demoted is not one of them, and is accounted for by
            // early_dispatch_blocked raising early_dispatch_target instead.
            if (!early_disqualified && !producer->task_attrs.allow_early_resolve()) {
                early_disqualified = true;
            }
            if (pstate >= CHIP_TASK_COMPLETED) {
                completed_fanin++;
            } else {
                producer->fanout_head = rss.dep_pool.prepend(producer->fanout_head, &slot_state);
                // The marker shares fanout_lock with propagation's snapshot. A set
                // marker means this edge is outside that snapshot and needs a seed.
                if (!early_disqualified && producer->has_dispatch_propagated()) {
                    early_propagated++;
                }
            }
            producer->unlock_fanout();
        }
        // The submit->wire pin protects an ordering-only edge only until the
        // consumer is linked. With the consumer now on fanout_head (or already
        // seen as completed), release it so the producer can be CONSUMED without
        // waiting for this consumer. A DEP_RETAIN edge keeps the pin until the
        // consumer's on_task_release. A reduction-dropped (DEP_NONE) edge is
        // released here too: it never links, so wiring is its only release point.
        if (!dep_has_retain(flags)) {
            // Wiring-phase atomics (this release, plus the lock_fanout / dep_pool.prepend /
            // fanin_refcount ops around it) are not bucketed: g_orch_args_atomic_count
            // covers the submit/dep-claim phase only, whose g_orch_args_cycle window has
            // already closed by the time wiring runs.
            sched->release_producer(*producer);
        }
    });

    // Completed producers and edges outside a producer's one-shot propagation
    // snapshot both need wiring-time contributions. Seed only when every
    // producer is codegen-flagged; one unflagged producer disqualifies the task.
    int32_t early_seed = completed_fanin + early_propagated;
    if (!early_disqualified && early_seed != 0) {
        int32_t dispatch_fanin = payload->dispatch_fanin.fetch_add(early_seed, std::memory_order_acq_rel) + early_seed;
        // A fully pre-completed fanin routes normally. If any producer was live,
        // the exact-full increment must enqueue the early candidate.
        int32_t target = payload->early_dispatch_target();
        if (completed_fanin != target && dispatch_fanin == target) {
            sched->try_enqueue_early_dispatch_candidate(slot_state);
        }
    }

    int32_t init_rc = completed_fanin + 1;
    int32_t new_rc = slot_state.fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
    mark_dep_pool_position(slot_state);
    if (new_rc >= slot_state.fanin_count) {
        sched->route_ready_once(slot_state);
    }
}

static ChipTaskSlotState *oldest_open_task_on_current_ring(const OrchestratorState *orch) {
    // Scope depth maps directly to a ring until the deepest ring, where all
    // further nested scopes share that ring. Its shallowest open scope begin
    // therefore marks the oldest task pinned by any open scope on this ring.
    int32_t begin = orch->scope_begins[orch->current_ring_id()];
    return begin < orch->scope_tasks_size ? orch->scope_tasks[begin] : nullptr;
}

static bool orch_wire_live_fanin_task(OrchestratorState *orch, ChipTaskSlotState &slot_state, int32_t wfanin) {
    SchedulerState *sched = orch->scheduler;
    auto &rss = sched->ring_sched_states[slot_state.ring_id];

    // dep_pool is orchestrator-exclusive (no lock). ensure_space waits for the
    // scheduler to advance last_task_alive and, on a wedged reclaim watermark,
    // detects the deadlock with the same structural + wall-clock logic the
    // heap/task-window allocator uses (all three share last_task_alive), latches
    // SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED, and emits the structured report. A false
    // return also covers a fatal already latched elsewhere.
    if (!rss.dep_pool.ensure_space(*rss.ring, wfanin, oldest_open_task_on_current_ring(orch))) {
        orch->fatal = true;
        return false;
    }

    orch->wire_fanin_task(slot_state, wfanin);
    return true;
}

static void scope_tasks_push(OrchestratorState *orch, ChipTaskSlotState *task_slot_state);

struct PreparedTask {
    TaskId task_id = TaskId::invalid();
    TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    TaskDescriptor *task = nullptr;
    TaskPayload *payload = nullptr;
    ChipTaskSlotState *slot_state = nullptr;
    uint64_t seq = 0;  // global submission sequence assigned in prepare_task
};

static OutputLayout calculate_output_layout(const CoreTaskArgs &args) {
    OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] = CHIP_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool check_scope_can_accept_task(OrchestratorState *orch, TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
    LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
    LOG_ERROR("  no slots can be reclaimed -> deadlock.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: CHIP_TASK_WINDOW_SIZE in runtime_types.h");
    LOG_ERROR("     Per task:     CallConfig.runtime_env.ring_task_window=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, SIMPLER_ERROR_SCOPE_DEADLOCK);
    return false;
}

static bool prepare_task(
    OrchestratorState *orch, const CoreTaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    TaskAttrs task_attrs, PreparedTask *out
) {
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size, oldest_open_task_on_current_ring(orch));
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, SIMPLER_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = simpler::tmr::make_task_id(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];

    // Reset the fanout/fanin bookkeeping for this reuse. The allocator only
    // returns a slot whose previous occupant is CONSUMED and quiescent (alloc
    // spins until last_task_alive passes it; in-order reclaim + acquire load),
    // and the slot is not published to any scheduler thread until the
    // Orch-side wiring publish at the end of submit_task_common — so this reset
    // is race-free. Doing it here (not relying on the scheduler's eager
    // reset-after-CONSUMED, which only covers the contiguously-reclaimed tail)
    // makes every reused slot self-clean, which lets the per-boot SM init skip
    // its O(window) per-slot loop. bind_ring is slot-invariant but cheap to
    // re-assert on the already-dirtied cache line.
    out->slot_state->bind_ring(ring_id);
    out->slot_state->reset_for_reuse();
    out->slot_state->fanin_count = 0;

    // Assign the global submission sequence and stamp the slot's side entry
    // before any consumer can observe the slot: every task-consuming path runs
    // on this single thread, and a later submit reaches this slot as a
    // producer only after this submit returns. alloc_tensors shares this
    // path, so hidden alloc tasks get a seq without entering
    // submit_task_common.
    //
    // Both fields are written together so the entry always describes one task
    // generation. An empty bitmap contributes no `via` bit, so a slot whose
    // submit fails between here and reduce_wait_edges' publication carries a
    // conservative entry rather than the previous generation's ancestors.
    out->seq = orch->submit_seq++;
    orch->wait_reach[ring_id][out->alloc_result.slot] = WaitReachEntry{0, out->seq};

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(window_size) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the Orch-side wiring publish at the end of
    // submit_task_common — that publish is the first read of slot_state->task /
    // slot_state->payload by scheduler threads.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch fields) is initialized in TaskPayload::init, the
    // single payload-init point, which runs before Orch-side wiring publish.

    // Fields already reset by advance_ring_pointers (eager reset after CONSUMED):
    //   fanout_lock=0, fanout_count=FANOUT_SCOPE_BIT, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state left as CONSUMED by eager reset (safe for stale wait_for_tensor
    // observers); set to PENDING here when orchestrator actually reuses the slot.
    out->slot_state->task_state.store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    out->slot_state->task_attrs = task_attrs;
    // fanin_count is set during Orch-side wiring
    scope_tasks_push(orch, out->slot_state);

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(OrchestratorState *orch, ChipTaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        // scope_tasks lives in the per-Worker arena (single backing allocation),
        // so realloc is not legal. Capacity is the total in-flight slot budget
        // (sum of the per-ring task windows; see reserve_layout) — hitting it means
        // every ring is saturated, so no further push could succeed regardless of
        // buffer growth.
        orch->report_fatal(
            SIMPLER_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void OrchestratorState::begin_scope(ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported"
        );
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = orch->current_ring_id();
        auto &alloc = orch->rings[ring_id].task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_states[ring_id].read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_begin(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif
}

void OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if SIMPLER_DFX
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = orch->current_ring_id();
        auto &alloc = orch->rings[ring_id].task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_states[ring_id].read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_end(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif

#if SIMPLER_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;
    if (ending_manual_scope) {
        orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
    }

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if SIMPLER_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

// Ensure the tensormap entry pool has room for `needed` inserts before STEP 4
// registers this task's outputs. The pool is watermark-reclaimed like the
// task/heap/fanin pools — retired tasks' entries free once last_task_alive
// advances — so an exhausted pool is back-pressure, not a hard error. Reclaim
// across all rings (entries from every ring share one pool); if still short,
// spin until reclaim actually frees entries, with the same 500 ms wall-clock
// backstop as the task allocator and fanin spill pool. A pool that stays full
// (no entry freed) is a genuine deadlock: latch SIMPLER_ERROR_TENSORMAP_OVERFLOW
// and bail. Returns false on deadlock or on a fatal already latched by another
// party. Cold path — the fast path returns immediately when the pool has room.
static bool ensure_tensormap_capacity(OrchestratorState *orch, int32_t needed) {
    ChipTensorMap &tm = orch->tensor_map;
    if (tm.free_entries() >= needed) {
        return true;
    }

    int32_t alive[CHIP_MAX_RING_DEPTH];
    auto read_alive = [&]() {
        for (int32_t r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
            // Relaxed: a self-correcting poll re-read every reclaim tick, so a stale
            // watermark only defers reclaim one tick and never over-frees.
            alive[r] = orch->sm_header->rings[r].fc.last_task_alive.load(std::memory_order_relaxed);
        }
    };

    read_alive();
    int64_t cur_alive_sum = tm.reclaim_retired_all(alive);  // kept for the deadlock diagnostic
    int32_t prev_free = tm.free_entries();
    if (prev_free >= needed) {
        return true;
    }

    int spin_count = 0;
    uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
    bool block_timing = false;  // false until the first no-reclaim-progress tick
    while (tm.free_entries() < needed) {
        spin_count++;

        // Reclaim (and the all-ring watermark reads it needs) is the costly part of
        // this spin and the only path that frees entries; gate it to a periodic tick.
        // Cold path, but the spin itself is tight.
        if ((spin_count & 31) == 0) {
            read_alive();
            cur_alive_sum = tm.reclaim_retired_all(alive);
            int32_t cur_free = tm.free_entries();
            if (cur_free >= needed) {
                return true;
            }
            // Progress is entries actually freed, NOT watermark movement: a ring can
            // retire zero-output tasks (count_registrable_outputs == 0), advancing
            // last_task_alive without freeing any entry. Gating the backstop on
            // free_entries() keeps a wedged pool from dodging the timeout while some
            // unrelated ring keeps draining.
            if (cur_free > prev_free) {
                spin_count = 0;
                prev_free = cur_free;
                block_timing = false;
            }
        }

        if ((spin_count & 1023) == 0) {
            // A fatal latched elsewhere breaks this otherwise-unbounded spin.
            if (orch->sm_header->orch_error_code.load(std::memory_order_acquire) != SIMPLER_ERROR_NONE) {
                return false;
            }
            // Absolute-time backstop, matching the task allocator: stable across
            // chips/contention, unlike a fixed spin count. get_sys_cnt_aicpu()
            // is a cheap cntvct_el0 read; the 1024-spin gate keeps fatal-flag
            // polling and timeout bookkeeping out of every entry-pool wait spin.
            uint64_t now = get_sys_cnt_aicpu();
            if (!block_timing) {
                block_cycle0 = now;
                block_timing = true;
            } else if (now - block_cycle0 >= ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: TensorMap Entry Pool Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("TensorMap entry pool freed no entries for ~500 ms while a task waits.");
                LOG_ERROR("  - Pool used:   %d / %d", tm.current_used(), tm.pool_capacity());
                LOG_ERROR("  - Needed:      %d entries", needed);
                LOG_ERROR("  - last_task_alive (sum across rings): %" PRId64, cur_alive_sum);
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  No retiring task is freeing tensormap entries (last_task_alive may");
                LOG_ERROR("  still move on rings with no registered outputs). Check TaskRing");
                LOG_ERROR("  diagnostics for the stalled producer.");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Increase CHIP_TENSORMAP_POOL_SIZE (current: %d).", tm.pool_capacity());
                LOG_ERROR("========================================");
                orch_mark_fatal(orch, SIMPLER_ERROR_TENSORMAP_OVERFLOW);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and
// Orch-side wiring/ready publication.
static TaskOutputTensors submit_task_common(
    OrchestratorState *orch, const CoreTaskArgs &args, ActiveMask active_mask, TaskAttrs task_attrs,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, task_attrs, &prepared)) {
        return result;
    }
    uint8_t ring_id = simpler::tmr::task_ring(prepared.task_id);
    SchedulerState *sched = orch->scheduler;
    ChipRingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
    TaskId task_id = prepared.task_id;
    ChipTaskSlotState &cur_slot_state = *prepared.slot_state;
    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph — the sole
    // source of truth for fanout now that the swimlane hot path no longer
    // records it.
#if SIMPLER_DFX
    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        // TensorArgType is `enum class : int32_t` (4 bytes); the on-disk record
        // packs arg_types as uint8_t[16] (5-value enum fits in a byte). Narrow
        // each tag here rather than letting the AICPU writer reinterpret a
        // 4×-wider array as bytes — that path silently lost two of every three
        // tags on little-endian and synthesized phantom self-edges in replay.
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        // Clamp to MAX_TENSOR_ARGS even though the Arg builder caps adds at
        // MAX_TENSOR_ARGS: defensive against any future builder bypass /
        // shared-memory bit-flip that could otherwise overrun the two
        // MAX_TENSOR_ARGS-sized stack buffers above.
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            // OUTPUT slots carry create_info (not yet a simpler::tmr::Tensor); skip them —
            // they have no producer to look up and replay's per-tensor loop
            // also skips OUTPUT.
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), args.allow_early_resolve(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            reinterpret_cast<const uint8_t *>(args.explicit_dep_kinds_data()),
            static_cast<uint8_t>(DEP_WAIT | DEP_RETAIN), args.launch_spec.block_num(), kernel_ids_capture
        );
    }
#endif

    FaninBuilder fanin_builder(orch, orch->rings[ring_id].fanin_pool, next_fanin_seen_epoch(orch));

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    CYCLE_COUNT_LAP(g_orch_sync_cycle);

    for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
        TaskId dep_task_id = args.explicit_dep(i);
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return result;
        }
        uint8_t dep_ring_id = simpler::tmr::task_ring(dep_task_id);
        SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_ring_id];
        int32_t dep_local_task_id = static_cast<int32_t>(simpler::tmr::task_local_id(dep_task_id));
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
        ChipTaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
        if (!append_fanin_or_fail(
                orch, dep_ring_id, dep_slot, producer_slot_state, dep_task_id, &fanin_builder, ring_id,
                args.explicit_dep_kind(i)
            )) {
            return result;
        }
    }

    // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
    DepInputs dep_inputs{
        args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
        args.explicit_deps_data(),
    };

    auto runtime_emit = [&](TaskId producer_task_id, DepFlags kind) -> bool {
        uint8_t prod_ring = simpler::tmr::task_ring(producer_task_id);
        SharedMemoryRingHeader &producer_ring = orch->sm_header->rings[prod_ring];
        int32_t prod_slot =
            producer_ring.get_slot_by_task_id(static_cast<int32_t>(simpler::tmr::task_local_id(producer_task_id)));
        ChipTaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
        return append_fanin_or_fail(
            orch, prod_ring, prod_slot, prod_state, producer_task_id, &fanin_builder, ring_id, kind
        );
    };

    if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
        return result;
    }

    CYCLE_COUNT_LAP(g_orch_lookup_cycle);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    // Reserve pool capacity for this task's inserts before registering. The pool
    // is shared across rings and reclaimed as last_task_alive advances; an
    // exhausted pool back-pressures here (and detects a wedged watermark) rather
    // than tripping new_entry()'s hard assert mid-registration.
    int32_t tensormap_needed = count_registrable_outputs(dep_inputs, orch->in_manual_scope());
    if (tensormap_needed > 0 && !ensure_tensormap_capacity(orch, tensormap_needed)) {
        return result;
    }
    register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());

    CYCLE_COUNT_LAP(g_orch_insert_cycle);

    // === STEP 5: Batch-write to GM (single cache line burst) ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // fanout_count was already incremented per live producer inside
    // append_fanin_or_fail, atomically with the consumed/generation check under
    // the producer's fanout_lock. Doing it there (rather than a separate pass
    // here) is what prevents a producer from transitioning to CONSUMED between
    // the dependency decision and the claim.
    //
    // Bounded transitive reduction runs on the fully-built fanin, before it is
    // flushed to the payload and wired, so RETAIN-only and dropped edges are
    // reflected in every downstream count. Publication of R[t] happens inside,
    // so every slot that can later be read as a producer carries its own
    // bitmap.
    reduce_wait_edges(orch, &fanin_builder, ring_id, prepared.alloc_result.slot, prepared.seq);
    int32_t inline_count = std::min(fanin_builder.count, CHIP_FANIN_INLINE_CAP);
    // fanin_actual_count is the TOTAL edge count (for_each iteration and
    // on_task_release walk every edge, including RETAIN-only and
    // reduction-dropped ones); fanin_wait_count is the readiness WAIT-edge count
    // after reduction, always <= fanin_actual_count. The early-dispatch
    // denominator is early_dispatch_target(), which re-adds the unit a
    // reduced-away unflagged producer used to hold.
    always_assert(
        fanin_builder.wait_count <= fanin_builder.count &&
        "fanin_wait_count is the readiness WAIT denominator and never exceeds the edge total"
    );
    // Store fanin metadata in payload for scheduler to iterate
    payload.fanin_actual_count = fanin_builder.count;
    payload.fanin_wait_count = fanin_builder.wait_count;
    payload.early_dispatch_blocked = fanin_builder.reduced_unflagged_producer ? 1 : 0;
    // fanin_builder.count is finalized here and submit runs once per task, so
    // each dense consumer emits one debug message when enabled. This checks >
    // THRESHOLD because the count lands at its final total here.
    if (fanin_builder.count > CHIP_DEP_DEGREE_DEBUG_THRESHOLD) {
        LOG_DEBUG(
            "dense dependency: task ring=%u id=%u fanin>%d [orch submit]",
            static_cast<unsigned>(simpler::tmr::task_ring(task_id)), simpler::tmr::task_local_id(task_id),
            CHIP_DEP_DEGREE_DEBUG_THRESHOLD
        );
    }
    payload.fanin_spill_start = fanin_builder.spill_start;
    payload.fanin_spill_pool = &fanin_builder.spill_pool;
    for (int i = 0; i < inline_count; i++) {
        payload.fanin_inline_edges[i] = fanin_builder.inline_slots[i];
    }

    payload.init(args, result, prepared.alloc_result, layout);

    // Dispatch predicate: resolve the (tensor, indices) to an absolute GM address
    // now so the scheduler can read it at the dispatch point with a single load,
    // no Arg/simpler::tmr::Tensor access. Both branches write predicate.op explicitly because
    // payload slots are ring-reused; op == NONE means "always dispatch".
    {
        const CoreTaskPredicate &pred = args.predicate();
        if (pred.op != PredicateOp::NONE && pred.operand.tensor != nullptr && pred.operand.tensor->buffer.addr != 0) {
            uint64_t elem_size = get_element_size(pred.operand.tensor->dtype);
            uint64_t flat_offset = pred.operand.tensor->compute_flat_offset(pred.operand.indices, pred.operand.ndims);
            payload.predicate.addr = pred.operand.tensor->buffer.addr + flat_offset * elem_size;
            payload.predicate.target = pred.target;
            payload.predicate.elem_size = static_cast<uint8_t>(elem_size);
            payload.predicate.op = pred.op;
        } else {
            payload.predicate.addr = 0;
            payload.predicate.op = PredicateOp::NONE;
        }
    }
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Preserve the existing Level-1 task/arg mask whenever dump is enabled.
        // Level 1 uses it to select records; hybrid Level 3 reuses the same mask only
        // to decide which tensors contribute payload alongside full metadata.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);

    // === STEP 6: wire on the orchestrator side and publish readiness ===
    // Zero-fanin tasks and tasks whose claimed producers are already completed
    // do not need fanout links or dep_pool entries. Tasks with live producers
    // allocate fanout links here before any scheduler thread can dispatch them.
    if (fanin_builder.count == 0) {
        cur_slot_state.fanin_count = 1;
        cur_slot_state.fanin_refcount.store(1, std::memory_order_release);
        orch->mark_dep_pool_position(cur_slot_state);
        sched->push_ready_routed(&cur_slot_state);
    } else if (all_claimed_fanin_completed(fanin_builder)) {
        int32_t ready_seed = fanin_builder.wait_count + 1;
        cur_slot_state.fanin_count = ready_seed;
        if (all_claimed_fanin_allow_early_resolve(fanin_builder)) {
            payload.dispatch_fanin.store(fanin_builder.wait_count, std::memory_order_release);
        }
        cur_slot_state.fanin_refcount.store(ready_seed, std::memory_order_release);
        // wire_fanin_task is skipped here, so its ordering-only pin release runs
        // on this path too: an edge without retention — including a
        // reduction-dropped DEP_NONE edge — drops its submit->wire pin so the
        // (already completed) producer can be CONSUMED.
        for_each_fanin_slot_state(payload, [&](ChipTaskSlotState *producer, DepFlags flags) {
            if (!dep_has_retain(flags)) {
                sched->release_producer(*producer);  // wiring-phase atomic, not bucketed (see wire_fanin_task)
            }
        });
        orch->mark_dep_pool_position(cur_slot_state);
        sched->push_ready_routed(&cur_slot_state);
    } else {
        if (!orch_wire_live_fanin_task(orch, cur_slot_state, fanin_builder.wait_count)) {
            return result;
        }
    }

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    // sync_start is only meaningful for tasks with block_num > 1.
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                SIMPLER_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        task_attrs.set_sync_start();
    }

    if (args.predicate().op != PredicateOp::NONE) {
        task_attrs.set_predicate();
    }

    return submit_task_common(
        orch, args, active_mask, task_attrs, normalized.aic_kernel_id, normalized.aiv0_kernel_id,
        normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors OrchestratorState::submit_dummy_task(const CoreTaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, SIMPLER_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    // Dummy tasks never dispatch to an AICore, so sync_start / has_predicate do
    // not apply; only the early-dispatch hint and timing tag carry over.
    TaskAttrs task_attrs;
    task_attrs.set_early_resolve(args.allow_early_resolve());
    task_attrs.set_timing_slot(args.task_timing_slot());

    return submit_task_common(
        orch, args, ActiveMask{}, task_attrs, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID
    );
}

TaskOutputTensors OrchestratorState::alloc_tensors(const CoreTaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
        );
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    OutputLayout layout = calculate_output_layout(args);
    PreparedTask prepared;
    // Kernel-less alloc task: no active subtasks, no dispatch-time markers. The
    // early-dispatch hint is force-set below (see the flag-the-creator note).
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, TaskAttrs{}, &prepared)) {
        return TaskOutputTensors{};
    }

    TaskDescriptor &task = *prepared.task;
    TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if SIMPLER_DFX
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_actual_count = 0;
    payload.fanin_wait_count = 0;
    payload.early_dispatch_blocked = 0;
    payload.fanin_spill_start = 0;
    payload.fanin_spill_pool = &orch->rings[simpler::tmr::task_ring(prepared.task_id)].fanin_pool;
    // A hidden alloc task has no WAIT predecessors: its reachability bitmap is
    // empty and its seq was stamped by prepare_task, so a consumer reading it
    // as a creator producer sees a valid, conservative entry rather than a
    // stale slot generation.
    publish_wait_reach(orch, simpler::tmr::task_ring(prepared.task_id), prepared.alloc_result.slot, 0);
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        //
        // Flag the creator so it does NOT suppress its consumers' early-dispatch.
        // Under the direct-only model an unflagged producer disqualifies its
        // consumer, and a pre-completed producer only seeds dispatch_fanin when
        // flagged. A buffer allocation is pure memory whose output is ready at
        // creation — it should always be transparent, never a barrier. Unlike a
        // codegen task there is no Arg-driven hint to honor here, so mark it
        // unconditionally.
        prepared.slot_state->task_attrs.set_early_resolve(true);
        prepared.slot_state->mark_completed();
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if SIMPLER_DFX
    orch->tasks_submitted++;
#if SIMPLER_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void OrchestratorState::mark_done() {
    auto *orch = this;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_DEBUG("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &fanin_pool = orch->rings[r].fanin_pool;
        if (fanin_pool.top > 1) {
            LOG_DEBUG(
                "=== [FaninPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, fanin_pool.top,
                fanin_pool.tail, fanin_pool.top - fanin_pool.tail, fanin_pool.high_water, fanin_pool.capacity
            );
        }
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = CHIP_MAX_SCOPE_DEPTH;
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_ORCH_PROFILING
OrchProfilingData orchestrator_get_profiling() {
    OrchProfilingData d;
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
