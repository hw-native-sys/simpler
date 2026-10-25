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
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "assert_compat.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "scheduler_layout.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#if !defined(__CCE_AICORE__)
#include "host_build_graph/runtime_types.h"
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#endif

// =============================================================================
// Profiling macros (compile-time gated)
// =============================================================================

#if SIMPLER_DFX
#include "aicpu/device_time.h"
// Accumulated nanoseconds per sub-step
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#endif

// =============================================================================
// Scheduler constants
// =============================================================================

constexpr int32_t MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;

// Periodic cadence (in idle iterations) for emitting the per-thread STALL
// diagnostic while no progress is being made. Purely an observability knob,
// independent of the wall-clock timeout below: small enough to fire a few times
// before the budget expires, large enough not to flood device_log.
constexpr int32_t STALL_LOG_INTERVAL = 480000;
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check for a latched scheduler error every N idle iters

// Wall-clock budget for declaring "no progress = scheduler timeout". Replaces
// the per-thread iteration-count cap that once lived here as MAX_IDLE_ITERATIONS
// for the fatal-latch decision; STALL_LOG_INTERVAL above keeps the per-thread
// diagnostic cadence.
//
// Using wall-clock here is load-bearing for distributed runs: with per-thread
// iteration counts, a pure-idle thread spinning ~115 ns/iter hits the cap in
// ~92 ms while a sibling thread polling a RUNNING task takes ~200 ms for the
// same iteration count. The fast spinner racing ahead and latching fatal
// kills the slower-but-correct poller mid-poll — see the distributed
// startup-skew scenario in issue #897.
//
// The budget is platform-defined (PLATFORM_SCHEDULER_TIMEOUT_MS in
// platform_config.h), one value across every platform variant. Onboard keeps it
// below the STARS op-execute and host stream-sync budgets so the AICPU can flush
// diagnostics before the host-visible timeout chain fires. Sim has no STARS or
// ACL stream-sync timeout, but runs the same no-progress watchdog.
constexpr int32_t SCHEDULER_TIMEOUT_MS = PLATFORM_SCHEDULER_TIMEOUT_MS;
constexpr uint64_t SCHEDULER_TIMEOUT_CYCLES =
    static_cast<uint64_t>(SCHEDULER_TIMEOUT_MS) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;

// Which report a stall dump belongs to, and therefore what log level its lines
// take. A periodic round fires every STALL_LOG_INTERVAL idle iterations on a run
// that may still be making progress elsewhere, so it stays at INFO and is read by
// raising the device log level. A shutdown snapshot fires once, on the run the
// scheduler is about to kill, and carries the only record of what was still
// pending — so it takes the level of the SHUTDOWN_SNAPSHOT line that announces it,
// which the default device log level keeps.
enum class StallDumpReport : int32_t { Periodic, Shutdown };
constexpr int32_t PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int32_t PROGRESS_LOG_INTERVAL = 250;      // log every N completions after threshold

// =============================================================================
// Control flow signal from cold-path helpers back to the main dispatch loop.
// =============================================================================

enum class LoopAction : int8_t {
    NONE,        // cold path did not trigger; proceed normally
    BREAK_LOOP,  // equivalent to 'break' from the while(true) loop
};

// =============================================================================
// Per-core state: one cache line per core to eliminate false sharing
// and co-locate all hot-path fields for minimal cache misses.
// Dual-slot layout: running (currently executing) + pending (pre-loaded, awaiting hardware pickup).
// =============================================================================

struct alignas(64) CoreExecState {
    // --- Hot fields (completion + dispatch, every iteration) ---
    uint64_t reg_addr;                      // offset  0: register base address (set once in handshake)
    ChipTaskSlotState *running_slot_state;  // offset  8: slot state for running task (nullptr = empty)
    ChipTaskSlotState *pending_slot_state;  // offset 16: slot state for pending task (nullptr = empty)
    int32_t running_reg_task_id;            // offset 24: register task ID (AICPU_TASK_INVALID = idle)
    int32_t pending_reg_task_id;            // offset 28: pending register task ID (AICPU_TASK_INVALID = none)
    uint32_t dispatch_seq;                  // offset 32: monotonic dispatch counter
    SubtaskSlot running_subslot;            // offset 36: which subtask slot is running
    SubtaskSlot pending_subslot;            // offset 37: which subtask slot is pending
    uint8_t pad0_[2];                       // offset 38: alignment padding
    // Precomputed COND register pointer; resolved once in handshake so the
    // hot completion poll does a single volatile load instead of recomputing
    // reg_base + reg_offset(COND) on every iteration.
    volatile uint32_t *cond_ptr;  // offset 40: precomputed pointer to COND register
#if SIMPLER_DFX
    // --- Profiling fields (dispatch path, compile-time gated) ---
    uint64_t running_dispatch_timestamp;  // offset 48: AICPU dispatch timestamp for running task
    uint64_t pending_dispatch_timestamp;  // offset 56: AICPU dispatch timestamp for pending task
#else
    // --- Cold fields (init/diagnostics only, never in hot path) ---
    int32_t worker_id;          // offset 48: index in runtime.workers[]
    uint32_t physical_core_id;  // offset 52: hardware physical core ID
    CoreType core_type;         // offset 56: AIC or AIV (enum class : int32_t)
    uint8_t pad2_[4];           // offset 60: pad to 64 bytes
#endif
};
static_assert(sizeof(CoreExecState) == 64, "CoreExecState must occupy exactly one cache line");

// =============================================================================
// CoreTracker: cluster-based bitmask tracker for idle/running core state.
//
// core_states_ encodes per-cluster core idle/running in 3 bits per cluster:
//   bit i*3   = AIC of cluster i   (1 = idle, 0 = running)
//   bit i*3+1 = AIV0 of cluster i
//   bit i*3+2 = AIV1 of cluster i
// Max 21 clusters per tracker (63 bits in uint64_t).
// =============================================================================

class alignas(64) CoreTracker {
public:
    // The ceiling is the platform's cluster count, not the width of the backing
    // word: a run takes the whole device, and aicpu_thread_num can leave a single
    // scheduler thread owning all of it.
    static constexpr int32_t MAX_CLUSTERS = PLATFORM_MAX_BLOCKDIM;
    static constexpr int32_t MAX_CORE_PER_THREAD = MAX_CLUSTERS * PLATFORM_CORES_PER_BLOCKDIM;

public:
    CoreTracker() = default;

    class BitStates {
    public:
        // 3 state bits per cluster, so the backing word must hold
        // MAX_CLUSTERS * 3 bits. 128 covers both arches (a2a3 72, a5 108) with
        // room; a single uint64_t held only 21 clusters, fewer than either
        // device has.
        using Storage = unsigned __int128;

        BitStates() = default;

        explicit BitStates(Storage states) :
            states_(states) {}
        void init() { states_ = 0; }

        // A mask with exactly `offset` set. The shift lives here so a caller
        // cannot silently narrow it by writing `1ULL << offset`, which is
        // undefined once offset reaches 64.
        static BitStates bit(int32_t offset) { return BitStates(static_cast<Storage>(1) << offset); }

        BitStates operator~() const { return BitStates(~states_); }
        BitStates operator&(const BitStates &other) const { return BitStates(states_ & other.states_); }
        BitStates operator|(const BitStates &other) const { return BitStates(states_ | other.states_); }
        BitStates operator^(const BitStates &other) const { return BitStates(states_ ^ other.states_); }
        BitStates operator>>(int32_t offset) const { return BitStates(states_ >> offset); }
        BitStates operator<<(int32_t offset) const { return BitStates(states_ << offset); }
        void operator&=(const BitStates &other) { states_ &= other.states_; }
        void operator|=(const BitStates &other) { states_ |= other.states_; }
        void operator^=(const BitStates &other) { states_ ^= other.states_; }

        bool has_value() const { return states_ > 0; }
        int32_t count() const {
            return __builtin_popcountll(static_cast<uint64_t>(states_)) +
                   __builtin_popcountll(static_cast<uint64_t>(states_ >> 64));
        }
        void clear_bit(int32_t offset) { states_ &= ~(static_cast<Storage>(1) << offset); }

        // Extract the lowest set bit from mask, clear it, and return its position.
        // Returns -1 if mask is empty.
        int32_t pop_first() {
            if (states_ == 0) return -1;
            const uint64_t low = static_cast<uint64_t>(states_);
            const int32_t pos =
                (low != 0) ? __builtin_ctzll(low) : 64 + __builtin_ctzll(static_cast<uint64_t>(states_ >> 64));
            states_ &= states_ - 1;
            return pos;
        }

    private:
        Storage states_{0};
    };

public:
    static_assert(MAX_CORE_PER_THREAD <= 128, "CoreTracker state bits must fit BitStates::Storage");

    void init(int32_t cluster_count) {
        always_assert(
            cluster_count >= 0 && cluster_count <= MAX_CLUSTERS && "cluster_count outside CoreTracker capacity"
        );
        cluster_count_ = cluster_count;
        aic_mask_.init();
        aiv_mask_.init();
        pending_occupied_.init();
        for (int32_t i = 0; i < cluster_count; i++) {
            aic_mask_ |= BitStates::bit(i * 3);
            aiv_mask_ |= (BitStates::bit(i * 3 + 1) | BitStates::bit(i * 3 + 2));
        }
        core_states_ = aic_mask_ | aiv_mask_;
    }

    void set_cluster(int32_t cluster_idx, int32_t aic_wid, int32_t aiv0_wid, int32_t aiv1_wid) {
        always_assert(cluster_idx >= 0 && cluster_idx < MAX_CLUSTERS && "cluster_idx outside CoreTracker capacity");
        core_id_map_[cluster_idx * 3] = aic_wid;
        core_id_map_[cluster_idx * 3 + 1] = aiv0_wid;
        core_id_map_[cluster_idx * 3 + 2] = aiv1_wid;
    }

    int32_t get_cluster_count() const { return cluster_count_; }

    // --- Running core queries ---

    template <CoreType CT>
    bool has_running_cores() const {
        if constexpr (CT == CoreType::AIC) {
            return ((~core_states_) & aic_mask_).has_value();
        } else {
            return ((~core_states_) & aiv_mask_).has_value();
        }
    }

    bool has_any_running_cores() const { return ((~core_states_) & (aic_mask_ | aiv_mask_)).has_value(); }

    // True if any core on this thread still has a free slot to stage onto — an
    // idle core (running slot) or a running core with a free pending slot. A
    // core is unavailable only when running AND its pending slot is occupied;
    // idle cores keep pending_occupied_ clear by invariant, so
    // ~pending_occupied_ over all cores is exactly "has a free slot". Purely
    // local (no shared/atomic access) — used to skip early dispatch, and its
    // shared-queue pop, when this thread has no capacity at all.
    // BitStates of every core on this thread with a free slot to stage onto: a
    // core is unavailable only when running AND its pending slot is occupied.
    // Idle cores keep pending_occupied_ clear by invariant, so ~pending_occupied_
    // over aic|aiv is exactly "has a free slot". Spans AIC+AIV, so its .count() is
    // an upper bound on the early-dispatch drain's per-shape pop (never exceeds the
    // thread's total free cores), and .has_value() is the has_any_free_slot()
    // predicate that gates the Phase-4b early-dispatch pass. Purely local (no
    // shared/atomic access).
    BitStates get_free_slot_states() const { return (~pending_occupied_) & (aic_mask_ | aiv_mask_); }

    bool has_any_free_slot() const { return get_free_slot_states().has_value(); }

    template <CoreType CT>
    int32_t get_running_count() const {
        if constexpr (CT == CoreType::AIC) {
            return ((~core_states_) & aic_mask_).count();
        } else {
            return ((~core_states_) & aiv_mask_).count();
        }
    }

    // Return an opaque bitmask for iterating running cores of a given type.
    // Use pop_first() to extract core bit offsets one at a time.
    template <CoreType CT>
    BitStates get_running_cores() const {
        if constexpr (CT == CoreType::AIC) {
            return (~core_states_) & aic_mask_;
        } else {
            return (~core_states_) & aiv_mask_;
        }
    }

    BitStates get_all_running_cores() const { return (~core_states_) & (aic_mask_ | aiv_mask_); }
    BitStates get_cluster_offset_states() const { return aic_mask_; }

    // --- Cluster matching ---

    BitStates get_valid_cluster_offset_states(ResourceShape shape) const {
        switch (shape) {
        case ResourceShape::AIC:
            return core_states_ & aic_mask_;
        case ResourceShape::AIV:
            return ((core_states_ >> 1) | (core_states_ >> 2)) & aic_mask_;
        case ResourceShape::MIX:
            return (core_states_ >> 1) & (core_states_ >> 2) & core_states_ & aic_mask_;
        case ResourceShape::DUMMY:
            // DUMMY tasks never reach the core-tracker dispatch path; they are
            // completed inline by resolve_and_dispatch via dummy_ready_queue.
            return {};
        }
        return {};
    }

    int32_t get_aic_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset]; }
    int32_t get_aiv0_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 1]; }
    int32_t get_aiv1_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 2]; }

    int32_t get_aic_core_offset(int32_t cluster_offset) const { return cluster_offset; }
    int32_t get_aiv0_core_offset(int32_t cluster_offset) const { return cluster_offset + 1; }
    int32_t get_aiv1_core_offset(int32_t cluster_offset) const { return cluster_offset + 2; }

    bool is_aic_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> cluster_offset) & BitStates::bit(0)).has_value();
    }
    bool is_aiv0_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 1)) & BitStates::bit(0)).has_value();
    }
    bool is_aiv1_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 2)) & BitStates::bit(0)).has_value();
    }

    // --- State mutation ---

    // Toggle bit at the given bit offset (running <-> idle)
    void change_core_state(int32_t bit_offset) { core_states_ ^= BitStates::bit(bit_offset); }

    // --- Pending-occupied tracking ---
    // Tracks whether a core's pending payload slot is occupied (awaiting hardware ACK).
    // SET on dispatch (both running-first and pending), CLEAR on idle or pending_freed.

    void set_pending_occupied(int32_t bit_offset) { pending_occupied_ |= BitStates::bit(bit_offset); }
    void clear_pending_occupied(int32_t bit_offset) {
        pending_occupied_ ^= (pending_occupied_ & BitStates::bit(bit_offset));
    }

    // --- Two-phase dispatch queries ---

    // Idle dispatch: returns bit offsets of idle cores for the given shape.
    // For AIC: 1 bit per cluster (core offset == cluster offset).
    // For AIV: 1 bit per AIV core (2 bits per cluster at aiv_mask_ positions).
    // Only AIC needs pending_occupied filtering: by invariant, idle cores (core_states_ bit=1)
    // always have pending_occupied=0, so AIV/MIX need no extra filtering.
    // Skipping the AIC-centric filter also fixes a latent bug where a running+pending AIC core
    // would incorrectly block AIV idle dispatch on the same cluster.
    BitStates get_idle_core_offset_states(ResourceShape shape) const {
        if (shape == ResourceShape::AIC) {
            return get_valid_cluster_offset_states(shape) & ~(pending_occupied_ & aic_mask_);
        }
        if (shape == ResourceShape::AIV) {
            return core_states_ & aiv_mask_;
        }
        return get_valid_cluster_offset_states(shape);  // MIX: cluster-level
    }

    // Pending dispatch: returns bit offsets of cores eligible for pending-slot dispatch.
    // AIC: 1 bit per cluster (aic_mask_ positions). AIV: 1 bit per AIV core (aiv_mask_ positions).
    // Runtime MIX dispatch uses classify_mix_cluster() so the decision follows the task's active_mask.
    enum class MixPlacement : uint8_t { RUNNING, PENDING, REJECT };

    // Placement for the cores named by active_mask, ignoring cores this task does
    // not use. All used cores idle -> RUNNING placement (each to its running slot).
    // Otherwise -> PENDING placement: at dispatch each used core is filled per its
    // own state -- an idle core takes its running slot (and is marked running, so
    // the completion poller, which scans only running cores, tracks its FIN), an
    // already-running core takes its pending slot and executes after its in-flight
    // task. REJECT only when a used core's pending slot is already occupied (no free
    // slot) or the mask is empty.
    MixPlacement classify_mix_cluster(int32_t cluster_offset, uint8_t core_mask) const {
        BitStates used;
        if (core_mask & SUBTASK_MASK_AIC) {
            used |= BitStates::bit(cluster_offset);
        }
        if (core_mask & SUBTASK_MASK_AIV0) {
            used |= BitStates::bit(cluster_offset + 1);
        }
        if (core_mask & SUBTASK_MASK_AIV1) {
            used |= BitStates::bit(cluster_offset + 2);
        }
        if (!used.has_value() || (pending_occupied_ & used).has_value()) {
            return MixPlacement::REJECT;
        }

        BitStates idle = core_states_ & used;
        if (idle.count() == used.count()) {
            return MixPlacement::RUNNING;
        }
        return MixPlacement::PENDING;
    }

    BitStates get_mix_cluster_offset_states(uint8_t core_mask, MixPlacement placement) const {
        BitStates result;
        BitStates candidates = get_cluster_offset_states();
        while (candidates.has_value()) {
            int32_t cluster_offset = candidates.pop_first();
            if (classify_mix_cluster(cluster_offset, core_mask) == placement) {
                result |= BitStates::bit(cluster_offset);
            }
        }
        return result;
    }

    // --- Gated MIX split placement ---
    // A gated MIX block can place each of its cores INDEPENDENTLY (idle core -> gated
    // running slot; busy core with a free pending slot -> gated pending slot), because
    // every core waits on the doorbell and nothing executes until the rendezvous rings.
    // This is unsafe for immediate (non-gated) dispatch — hence separate from
    // classify_mix_cluster, which forces a single placement for the whole cluster.

    // Cores of `cluster_offset` named by core_mask.
    BitStates mix_used_cores(int32_t cluster_offset, uint8_t core_mask) const {
        BitStates used;
        if (core_mask & SUBTASK_MASK_AIC) used |= BitStates::bit(cluster_offset);
        if (core_mask & SUBTASK_MASK_AIV0) used |= BitStates::bit(cluster_offset + 1);
        if (core_mask & SUBTASK_MASK_AIV1) used |= BitStates::bit(cluster_offset + 2);
        return used;
    }

    // Every used core has SOME free slot: a core lacks one only when it is running AND
    // its pending slot is occupied (both slots taken).
    bool mix_cluster_all_slots(int32_t cluster_offset, uint8_t core_mask) const {
        BitStates used = mix_used_cores(cluster_offset, core_mask);
        if (!used.has_value()) return false;
        BitStates no_slot = (~core_states_) & pending_occupied_;  // running AND pending taken
        return !(used & no_slot).has_value();
    }

    // Used cores that are idle -> will take a running slot (rendezvous seed count).
    int32_t mix_cluster_idle_core_count(int32_t cluster_offset, uint8_t core_mask) const {
        return (mix_used_cores(cluster_offset, core_mask) & core_states_).count();
    }

    // Clusters where every used core has a free slot (gated MIX split gate/iteration).
    BitStates get_mix_split_cluster_offset_states(uint8_t core_mask) const {
        BitStates result;
        BitStates candidates = get_cluster_offset_states();
        while (candidates.has_value()) {
            int32_t off = candidates.pop_first();
            if (mix_cluster_all_slots(off, core_mask)) {
                result |= BitStates::bit(off);
            }
        }
        return result;
    }

    int32_t count_mix_split_clusters(uint8_t core_mask) const {
        return get_mix_split_cluster_offset_states(core_mask).count();
    }

    BitStates get_mix_running_cluster_offset_states(uint8_t core_mask) const {
        return get_mix_cluster_offset_states(core_mask, MixPlacement::RUNNING);
    }

    int32_t count_mix_running_clusters(uint8_t core_mask) const {
        return get_mix_running_cluster_offset_states(core_mask).count();
    }

    // Number of whole logical blocks this tracker can accept for sync_start
    // staging. AIC/AIV count cores; MIX counts clusters because one logical MIX
    // block may occupy multiple cores in the same cluster. Gated early staging
    // includes pending slots, while ready staging is restricted to idle slots.
    int32_t count_available_blocks(ResourceShape shape, uint8_t core_mask, bool include_pending) const {
        if (shape == ResourceShape::MIX) {
            return include_pending ? count_mix_split_clusters(core_mask) : count_mix_running_clusters(core_mask);
        }
        if (shape == ResourceShape::DUMMY) return 0;

        int32_t available = get_idle_core_offset_states(shape).count();
        if (include_pending) {
            available += get_pending_core_offset_states(shape).count();
        }
        return available;
    }

    BitStates get_pending_core_offset_states(ResourceShape shape) const {
        if (shape == ResourceShape::MIX) {
            // Shape-level query kept conservative for legacy callers/tests.
            // The real MIX dispatch path applies active_mask in classify_mix_cluster().
            // Any core without a pending payload can accept a dispatch (idle or running).
            BitStates available = ~pending_occupied_;
            BitStates mix_available =
                (available & aic_mask_) & ((available >> 1) & aic_mask_) & ((available >> 2) & aic_mask_);
            // Pending MIX can only reuse a fully-running cluster. Partially-running clusters
            // could split one MIX block across immediate and pending placement.
            BitStates running = ~core_states_;
            BitStates cluster_all_running =
                (running & aic_mask_) & ((running >> 1) & aic_mask_) & ((running >> 2) & aic_mask_);
            return mix_available & cluster_all_running;
        }
        if (shape == ResourceShape::AIC) {
            return (~core_states_) & aic_mask_ & ~(pending_occupied_ & aic_mask_);
        }
        // AIV
        return (~core_states_) & aiv_mask_ & ~pending_occupied_;
    }

    // --- Two-phase dispatch unified query ---

    enum class DispatchPhase : uint8_t { IDLE, PENDING };

    BitStates get_dispatchable_cores(ResourceShape shape, DispatchPhase phase) const {
        return (phase == DispatchPhase::IDLE) ? get_idle_core_offset_states(shape) :
                                                get_pending_core_offset_states(shape);
    }

    // --- Bit offset <-> worker_id mapping ---

    int32_t get_core_id_by_offset(int32_t offset) const { return core_id_map_[offset]; }

    const int32_t *core_ids() const { return core_id_map_; }
    int32_t core_num() const { return cluster_count_ * 3; }

private:
    int32_t cluster_count_{0};
    BitStates aic_mask_;
    BitStates aiv_mask_;
    BitStates core_states_;
    BitStates pending_occupied_;
    int32_t core_id_map_[MAX_CORE_PER_THREAD];  // bit position -> worker id
};

// =============================================================================
// SlotTransition: pure event signals from a single register poll.
// true = event occurred, false = no-op (maintain current state).
// =============================================================================

struct SlotTransition {
    bool running_done = false;   // running task completed
    bool pending_done = false;   // pending task completed
    bool running_freed = false;  // running slot data should be released
    bool pending_freed = false;  // pending_occupied can be cleared
    bool matched = false;        // some case was hit (otherwise skip apply)
};

// =============================================================================
// Profiling counters (compile-time gated)
// =============================================================================

#if SIMPLER_DFX
struct alignas(64) SchedChipSwimlaneCounters {
    bool chip_swimlane_enabled{false};
    uint64_t sched_start_ts{0};
    uint64_t sched_complete_cycle{0};
    uint64_t sched_dispatch_cycle{0};
    uint64_t sched_idle_cycle{0};
    uint64_t sched_loop_count{0};
    uint32_t phase_complete_count{0};
    // Sub-block retires that did NOT finish a slot (SPMD blocks of a multi-block
    // task retiring one at a time). Counted separately so the Complete-phase
    // emit can fire on poll iterations that only retired sub-blocks — otherwise
    // the serial-harvest tail of an SPMD slot is invisible (no slot completes
    // until the last block, leaving the scheduler lane blank for that window).
    uint32_t phase_subretire_count{0};
    uint32_t phase_dispatch_count{0};
    // Per-emit delta is (current - *_at_last_emit). Accumulated only when
    // chip_swimlane_level_ >= SCHED_PHASES.
    uint64_t pop_hit{0};
    uint64_t pop_miss{0};
    uint64_t pop_hit_at_last_emit{0};
    uint64_t pop_miss_at_last_emit{0};
#if SIMPLER_SCHED_PROFILING
    uint64_t complete_probe_count{0};
    uint64_t complete_hit_count{0};
    uint64_t sched_complete_perf_cycle{0};
    uint64_t sched_dispatch_pop_cycle{0};
    uint64_t sched_dispatch_setup_cycle{0};
#endif
    void reset() { *this = SchedChipSwimlaneCounters{}; }
};
#endif

// =============================================================================
// sync_start drain coordination
// =============================================================================

// When sync_start_pending != 0, all scheduler threads skip dispatch
// (only process completions) until the fixed coordinator finishes launching all blocks.
struct alignas(64) SyncStartDrainState {
    std::atomic<int32_t> sync_start_pending{0};              // 0=normal; -1=initializing; >0=active (value=block_num)
    std::atomic<ChipTaskSlotState *> pending_task{nullptr};  // held task (not re-queued)
    std::atomic<uint64_t> drain_attempt{0};                  // incremented whenever an ack round is reset
    // Parallel staging: after the coordinator confirms global availability it sets
    // stage_go, releasing every thread to stage its OWN cores concurrently (vs the old
    // single-thread serial fill). Each thread ORs its bit into stage_done_mask when it
    // finishes and accumulates its running-slot cores into running_staged; the coordinator
    // thread waits for all bits, seeds the rendezvous, and reopens the gate.
    std::atomic<int32_t> drain_stage_go{0};          // 0=hold; 1=coordinator released parallel staging
    std::atomic<uint32_t> drain_stage_done_mask{0};  // bit per thread; all-set = all threads done staging
    std::atomic<int32_t> drain_running_staged{0};    // sum of running-slot cores staged (rendezvous seed)
    int32_t _pad[7];
};
static_assert(sizeof(SyncStartDrainState) == 64);
static_assert(offsetof(SyncStartDrainState, pending_task) == 8);
static_assert(offsetof(SyncStartDrainState, drain_attempt) == 16);
static_assert(offsetof(SyncStartDrainState, drain_stage_go) == 24);

constexpr uint64_t SYNC_START_DRAIN_ACK_SUBTREE_READY = uint64_t{1} << 63;
constexpr uint64_t SYNC_START_DRAIN_ATTEMPT_MASK = ~SYNC_START_DRAIN_ACK_SUBTREE_READY;

inline uint64_t sync_start_drain_next_attempt(uint64_t attempt) {
    uint64_t next = (attempt + 1) & SYNC_START_DRAIN_ATTEMPT_MASK;
    return next == 0 ? 1 : next;
}

inline uint64_t sync_start_drain_ack_subtree_token(uint64_t attempt) {
    return attempt | SYNC_START_DRAIN_ACK_SUBTREE_READY;
}

#endif

// =============================================================================
// Resident AICore scheduler wire contracts
//
// These types are shared by Host, AICPU lifecycle, and AICore workers.
// =============================================================================
inline constexpr uint64_t SCHEDULER_STATE_ALIGNMENT = 128;
inline constexpr uint64_t SCHEDULER_WORKER_CAPACITY = 108;
inline constexpr uint32_t SCHEDULER_PENDING_SLOT_COUNT = 2;
// A Mix task's AIV subtasks land on slots whose only writer is the cluster's
// AIC, held separately from each AIV's own two slots. Sole ownership is what
// makes the AIC's capacity probe exact: it sees only its own history there, so
// a placement it plans cannot be taken away between the probe and the claim,
// and no back-pressure protocol is needed.
//
// Deliberately a second constant rather than widening SCHEDULER_PENDING_SLOT_COUNT:
// five places pack two 32-bit generations into one 64-bit word (scheduler_ssbuf.h,
// scheduler_ready.h, scheduler_completion.h, scheduler_dispatch.h), and a count
// of four would overflow every one of them.
inline constexpr uint32_t SCHEDULER_MIX_SLOT_COUNT = 2;
inline constexpr uint32_t SCHEDULER_CORE_TYPE_COUNT = 2;
inline constexpr uint32_t SCHEDULER_READY_QUEUE_COUNT = 3;
inline constexpr uint32_t SCHEDULER_MIX_QUEUE = 2;
inline constexpr uint32_t SCHEDULER_CLUSTER_CAPACITY = SCHEDULER_WORKER_CAPACITY / 3;
inline constexpr uint32_t SCHEDULER_ACTIVITY_CAPACITY = 1024;
inline constexpr uint64_t SCHEDULER_PROFILING_TASK_TIMING_LEVEL = 1;
inline constexpr uint64_t SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL = 2;
inline constexpr uint64_t SCHEDULER_PROFILING_SCHED_PHASES_LEVEL = 3;

inline __aicore__ bool scheduler_task_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_TASK_TIMING_LEVEL;
}

inline __aicore__ bool scheduler_schedule_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL;
}

inline __aicore__ bool scheduler_phase_timing_enabled(uint64_t level) {
    return level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL;
}

inline constexpr bool scheduler_activity_record_count_valid(uint32_t committed) noexcept {
    return committed <= SCHEDULER_ACTIVITY_CAPACITY;
}
inline constexpr uint32_t SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD = 7;

// A ready queue is consumed only by cores of one type, so it needs one inbox per
// such core rather than one per worker: the AIC and Mix queues are held by the
// 36 AIC cores, the AIV queue by the 72 AIV cores. Each is addressed by the
// core's type_rank, which the AICPU already assigns during core discovery, so
// 3 x 108 slots collapse to 36 + 72 + 36 with no new numbering.
//
// Addressing by rank also makes a directory shard type-homogeneous. Addressed by
// worker id the 7 owners of a shard interleave AIC and AIV cores, so a thief
// scanning its shard of the AIC queue finds at most a third of those bits ever
// set; by rank all 7 are cores that can hold the work it is looking for.
//
// It narrows how far one publisher's work can travel, which is the other half
// of the same property. A cluster's two AIV lanes take adjacent ranks, so both
// land in one shard, where their worker ids had straddled a shard boundary and
// reached two. A single-root fanout is therefore confined to one AIC shard plus
// one AIV shard; tests/st/a5/host_build_graph/fanout_imbalance measures it.
inline __aicore__ constexpr uint32_t scheduler_ready_queue_capacity(uint32_t core_type_index) {
    return core_type_index == 1U ? SCHEDULER_CLUSTER_CAPACITY * PLATFORM_AIV_CORES_PER_BLOCKDIM :
                                   SCHEDULER_CLUSTER_CAPACITY;
}

inline __aicore__ constexpr uint32_t scheduler_ready_queue_base(uint32_t core_type_index) {
    return core_type_index == 0U ? 0U :
           core_type_index == 1U ? SCHEDULER_CLUSTER_CAPACITY :
                                   SCHEDULER_CLUSTER_CAPACITY * (1U + PLATFORM_AIV_CORES_PER_BLOCKDIM);
}

inline constexpr uint32_t SCHEDULER_READY_INBOX_CAPACITY =
    SCHEDULER_CLUSTER_CAPACITY * (2U + PLATFORM_AIV_CORES_PER_BLOCKDIM);

// Mix slots exist on AIV cores, so their payloads and executor traces are
// addressed by the holder's AIV rank -- the same rank that addresses the AIV
// ready queue. The traces sit in global memory rather than the cluster's SSBUF
// because the SSBUF's remaining room goes to the dispatch and completion
// tokens, which are on the dispatch path where the traces are not.
inline constexpr uint32_t SCHEDULER_MIX_SLOT_CAPACITY =
    SCHEDULER_CLUSTER_CAPACITY * PLATFORM_AIV_CORES_PER_BLOCKDIM * SCHEDULER_MIX_SLOT_COUNT;

// Shards are allocated per queue at the widest queue's count so the directory
// stays a rectangular array; the unused tail of a narrower queue is never set,
// because scheduler_ready_queue_capacity bounds what can be addressed.
inline constexpr uint32_t SCHEDULER_READY_DIRECTORY_SHARD_COUNT =
    (SCHEDULER_CLUSTER_CAPACITY * PLATFORM_AIV_CORES_PER_BLOCKDIM + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD - 1) /
    SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;

// A task ref names one task to the resident Scheduler. Its encoding today is a
// GLOBAL task's index into the whole-graph-resident tables, which is why it is
// compared against graph.task_count and used as a subscript directly.
//
// It is deliberately not called a task id: TaskId (src/common/host_build_graph/
// task_id.h) is a 64-bit handle carrying a space and a parent, and a SUB_TASK --
// a task inside a graph task's body -- has no table slot at all and is located
// through its parent instead. TaskKind::GRAPH still sends the whole graph to the
// legacy scheduler, so no sub-task reaches this path; supporting one means
// widening this encoding to carry a parent. Every place that depends on the
// encoding being an index goes through the accessors below, so that widening has
// one set of call sites rather than a search for bare subscripts.
inline constexpr int64_t SCHEDULER_TASK_REF_INVALID = -1;
inline constexpr int64_t SCHEDULER_WAKE_LIST_OPEN = -1;
inline constexpr int64_t SCHEDULER_WAKE_LIST_CLOSED = -2;
inline constexpr int64_t SCHEDULER_INBOX_EMPTY = -1;
// Both endpoints empty, which is the only state where either one may be.
inline constexpr uint64_t SCHEDULER_READY_ENDPOINTS_EMPTY = UINT64_MAX;

enum class SchedulerTaskState : int64_t {
    BLOCKED = 0,
    DONE = 2,
};

enum class SchedulerReadySource : uint8_t {
    LOCAL = 0,
    STOLEN = 1,
    DIRECT_RESOLVE = 2,
    COOPERATIVE = 3,
};

enum class SchedulerPublicationMode : uint8_t {
    DISPATCH = 0,
    REFILL = 1,
};

struct SchedulerIdleRecord {
    uint64_t start_time;
    uint64_t end_time;
    uint32_t loop_iter;
    uint32_t reserved;
};
static_assert(sizeof(SchedulerIdleRecord) == 24, "Scheduler idle record layout changed");

struct alignas(64) SchedulerActivityBuffer {
    volatile uint32_t committed;
    volatile uint32_t dropped;
    uint32_t reserved[2];
    SchedulerIdleRecord records[SCHEDULER_ACTIVITY_CAPACITY];
};
static_assert(sizeof(SchedulerActivityBuffer) % 64 == 0, "scheduler activity buffer must be cache aligned");

enum class SchedulerDispatchSlotState : uint8_t {
    EMPTY = 0,
    FREE = 1,
    FILLING = 2,
    READY = 3,
};

enum SchedulerTaskMetadataFlags : uint8_t {
    SCHEDULER_TASK_EXECUTABLE = 1U << 0,
    SCHEDULER_TASK_HAS_FANIN = 1U << 1,
    SCHEDULER_TASK_MIX = 1U << 2,
    SCHEDULER_TASK_SPMD = 1U << 3,
    SCHEDULER_TASK_SYNC_START = 1U << 4,
    SCHEDULER_TASK_INLINE = 1U << 5,
    SCHEDULER_TASK_HAS_PREDICATE = 1U << 6,
};

#if !defined(__CCE_AICORE__)
// Derive the compact AICore metadata from the established HBG scheduling
// types. This is a host-only projection: ActiveMask and TaskAttrs remain the
// source of truth, while SchedulerTaskMetadata stays a 16-byte GM read model.
inline uint8_t scheduler_task_metadata_flags_from_submit_state(
    ActiveMask active_mask, TaskAttrs task_attrs, int32_t logical_block_num, bool has_fanin, bool inline_task
) {
    uint8_t flags = SCHEDULER_TASK_EXECUTABLE;
    const uint32_t active_subtasks = static_cast<uint32_t>(__builtin_popcount(active_mask.core_mask()));
    if (has_fanin) flags |= SCHEDULER_TASK_HAS_FANIN;
    if (active_subtasks > 1) flags |= SCHEDULER_TASK_MIX;
    if (logical_block_num > 1) flags |= SCHEDULER_TASK_SPMD;
    if (task_attrs.requires_sync_start()) flags |= SCHEDULER_TASK_SYNC_START;
    if (inline_task) flags |= SCHEDULER_TASK_INLINE;
    if (task_attrs.has_predicate()) flags |= SCHEDULER_TASK_HAS_PREDICATE;
    return flags;
}
#endif

constexpr int32_t SCHEDULER_TASK_TIMING_SLOT_COUNT = 16;

struct alignas(16) SchedulerTaskMetadata {
    uint16_t kernel_ids[3];
    uint8_t active_mask;
    uint8_t flags;
    uint32_t trace_index_base;
    int16_t timing_slot;
    uint16_t logical_block_num;
};

inline __aicore__ bool
scheduler_subtask_count_checked(uint64_t logical_block_num, uint32_t active_subtasks, uint32_t *out) {
    if (out == nullptr || logical_block_num == 0 || active_subtasks == 0 || active_subtasks > 3 ||
        logical_block_num > static_cast<uint64_t>(INT16_MAX) / active_subtasks)
        return false;
    *out = static_cast<uint32_t>(logical_block_num * active_subtasks);
    return true;
}

inline __aicore__ bool scheduler_trace_count_checked(uint64_t prefix, uint64_t added, uint64_t *out) {
    constexpr uint64_t capacity = static_cast<uint64_t>(UINT32_MAX) + 1;
    if (out == nullptr || added == 0 || prefix > UINT32_MAX || added > capacity - prefix) return false;
    *out = prefix + added;
    return true;
}

inline __aicore__ uint32_t
scheduler_task_trace_index(uint32_t trace_index_base, uint8_t active_mask, uint8_t subtask, uint32_t block_idx = 0) {
    const uint64_t offset = static_cast<uint64_t>(block_idx) * __builtin_popcount(active_mask) +
                            __builtin_popcount(active_mask & ((1U << subtask) - 1U));
    return static_cast<uint32_t>(static_cast<uint64_t>(trace_index_base) + offset);
}
inline __aicore__ uint32_t scheduler_task_ready_queue(uint8_t flags, uint8_t active_mask) {
    return (flags & SCHEDULER_TASK_MIX) != 0 ? SCHEDULER_MIX_QUEUE : ((active_mask & 1U) != 0 ? 0U : 1U);
}

inline __aicore__ bool scheduler_task_is_executable(uint8_t flags) { return (flags & SCHEDULER_TASK_EXECUTABLE) != 0; }

inline __aicore__ bool scheduler_task_has_fanin(uint8_t flags) { return (flags & SCHEDULER_TASK_HAS_FANIN) != 0; }

inline __aicore__ bool scheduler_task_is_mix(uint8_t flags) { return (flags & SCHEDULER_TASK_MIX) != 0; }

inline __aicore__ bool scheduler_task_is_spmd(uint8_t flags) { return (flags & SCHEDULER_TASK_SPMD) != 0; }

inline __aicore__ bool scheduler_task_requires_sync_start(uint8_t flags) {
    return (flags & SCHEDULER_TASK_SYNC_START) != 0;
}

inline __aicore__ bool scheduler_task_is_inline(uint8_t flags) { return (flags & SCHEDULER_TASK_INLINE) != 0; }

inline __aicore__ bool scheduler_task_has_predicate(uint8_t flags) {
    return (flags & SCHEDULER_TASK_HAS_PREDICATE) != 0;
}

inline __aicore__ bool scheduler_task_requires_admission(const SchedulerTaskMetadata &metadata) {
    return scheduler_task_is_executable(metadata.flags) && !scheduler_task_is_inline(metadata.flags) &&
           (metadata.logical_block_num > 1 || scheduler_task_is_mix(metadata.flags) ||
            scheduler_task_requires_sync_start(metadata.flags));
}

enum class SchedulerAdmissionState : uint64_t {
    UNCLAIMED = 0,
    CLAIMED = 1,
    PREPARED = 2,
    ADMITTED = 3,
};

struct alignas(128) SchedulerTaskControl {
    // Cross-core state publication, the wake-list RMWs, and the chain link share
    // the first line. Every field here is reached only through the atomic
    // wrappers, which is what the link needs on two counts: a Ready inbox has
    // several producers now, so the link transfers ownership of a node and must
    // be written as an atomic rather than observed-and-published; and nothing
    // publishes this line as a whole, so no producer's line writeback can
    // restore its stale copy of a link another producer just wrote.
    volatile int64_t state;
    volatile int64_t wake_list_head;
    // Links the task while it belongs to a dependency wake list, to a Ready
    // inbox, or to the local batch staged for one. A task is in at most one of
    // them: a resolver detaches a whole wake list before routing its waiters,
    // and a batch is spliced into an inbox without rewriting its links.
    volatile int64_t next_waiter;
    volatile uint64_t next_block;
    volatile uint64_t completed_blocks;
    volatile uint64_t admission_state;
    uint8_t atomic_line_padding[16];

    int32_t next_fanin_index;
    int32_t waiting_producer;
    uint64_t completion_resolve_start_cycles;
    uint64_t completion_resolve_end_cycles;
    uint64_t scheduler_worker_id;
    uint64_t completion_resolve_loop_iter;
    uint8_t scheduler_line_padding[24];
};

// Each record's bitmap follows its header within an independently aligned
// stride. Preparation owns that whole range; published records are immutable.
struct alignas(64) SchedulerAdmissionRecord {
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t member_words_offset;
    uint64_t mix_close_seals_offset;
    uint32_t logical_block_num;
    uint32_t member_count;
    uint8_t active_mask;
    uint8_t flags;
    uint8_t reserved[14];
};

struct alignas(128) SchedulerAdmissionControl {
    volatile uint64_t commit_lock;
    uint8_t lock_padding[120];
    alignas(128) volatile uint64_t commit_tail;
    uint8_t tail_padding[120];
};

struct alignas(128) SchedulerGroupDescriptor {
    uint64_t run_epoch;
    int64_t task_ref;
    uint64_t order_key;
    uint64_t group_seq;
    uint32_t block_idx;
    uint32_t logical_block_num;
    uint32_t cluster_index;
    uint8_t active_mask;
    uint8_t resource_mask;
    uint8_t lane_subtasks[3];
    uint8_t flags;
    uint8_t reserved[78];
};

struct alignas(128) SchedulerGroupLaneAck {
    volatile uint64_t arrived_seq;
    volatile uint64_t released_seq;
    uint8_t reserved[112];
};

struct alignas(128) SchedulerGroupMailbox {
    SchedulerGroupDescriptor descriptor;
    volatile uint64_t published_seq;
    uint8_t publication_padding[120];
    SchedulerGroupLaneAck lanes[3];
};

// A FIFO ready queue, addressed by both endpoints packed into one word.
//
// Sharing a word makes the endpoint pair atomic, which is what lets a consumer
// take the last task: head and tail have to be cleared together, and across two
// words an append landing between the two stores is overwritten and its whole
// batch lost. The alternative that keeps two words has to carry a sentinel node
// per queue so the queue is never structurally empty and that transition never
// arises.
//
// The cost is that one word is the single contention point for both ends: a
// producer appending and a consumer taking contend on every operation, even
// though they touch opposite ends of a non-empty queue. Separate head and tail
// on their own cache lines decouple that. Which one wins here is not measured.
struct alignas(128) SchedulerReadyInbox {
    volatile uint64_t endpoints;
    uint8_t atomic_line_padding[120];
};

// Graph-level facts the host measures once while building the graph. Every
// Scheduler reads this at bootstrap and nothing writes it afterwards, so it
// occupies its own cache line away from the state Schedulers mutate.
struct alignas(128) SchedulerGraphSummary {
    uint64_t mix_task_count;
    uint64_t cooperative_task_count;
    uint64_t worker_count;
    uint64_t cluster_count;
    uint64_t admission_member_word_count;
    uint64_t admission_records_offset;
    uint64_t admission_record_stride;
    uint64_t admission_log_offset;
    uint64_t admission_control_offset;
    uint64_t group_mailboxes_offset;
    uint64_t mix_close_seals_offset;
    uint64_t run_epoch;
    uint8_t cache_line_padding[32];
};

struct alignas(64) SchedulerReadyDirectoryShard {
    volatile uint64_t bits;
    uint8_t cache_line_padding[64 - sizeof(uint64_t)];
};

struct alignas(128) SchedulerReadyDirectory {
    SchedulerReadyDirectoryShard core_types[SCHEDULER_READY_QUEUE_COUNT][SCHEDULER_READY_DIRECTORY_SHARD_COUNT];
};

// The Executor publishes this per-slot payload before the Completion Inbox
// generation. Neither side writes the final per-task trace concurrently.
// Local copies have natural alignment; SchedulerSsbufLane supplies the
// 64-byte alignment and stride of shared trace storage.
struct SchedulerExecutorTaskTrace {
    uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t ready_scan_start_cycles;
    uint64_t ready_observe_cycles;
    uint64_t completion_id;
    uint64_t completion_inbox_index;
};

// Stable device-side localization for the first scheduler failure. These values
// are diagnostic ABI: keep existing numbers stable when adding new sites.
enum class SchedulerErrorSite : uint64_t {
    UNKNOWN = 0,
    ROUTE_INVALID_STATE = 2,
    READY_POP_INVALID_HEAD = 40,
    READY_POP_INVALID_LINK = 41,
    DISPATCH_INVALID_SHAPE = 44,
    DISPATCH_INVALID_CALLABLE = 45,
    DISPATCH_MATERIALIZE_FAILED = 46,
    DISPATCH_INVALID_PREDICATE = 47,
    EXECUTOR_INVALID_DISPATCH_SLOT = 50,
    BOOTSTRAP_WAKE_INVALID_HEAD = 60,
    COMPLETION_TASK_NOT_DONE = 61,
    COMPLETION_WAKE_ALREADY_CLOSED = 62,
    COMPLETION_INVALID_WAITER = 63,
    COMPLETION_WAITER_NOT_EXECUTABLE = 64,
    COMPLETION_READY_APPEND_FAILED = 65,
    COMPLETION_READY_PUBLISH_FAILED = 66,
    COMPLETION_INVALID_SHAPE = 67,
    COMPLETION_RESOLVE_FAILED = 68,
    COMPLETION_REFILL_DISPATCH_FAILED = 70,
    COMPLETION_GENERATION_MISMATCH = 74,
    EXECUTOR_PREFERRED_SLOT_INVALID = 79,
    READY_INBOX_REFRESH_FAILED = 80,
    BOOTSTRAP_FAILED = 90,
    CONTEXT_READY_TIMEOUT = 91,
    BOOTSTRAP_SCAN_TIMEOUT = 92,
    BOOTSTRAP_COMPLETE_TIMEOUT = 93,
    REGISTER_RELEASE_TIMEOUT = 94,
    EXECUTION_PROGRESS_TIMEOUT = 95,
    EXIT_WAIT_TIMEOUT = 96,
    AICPU_HANDSHAKE_TIMEOUT = 100,
    AICPU_INIT_BARRIER_TIMEOUT = 101,
    AICPU_SHUTDOWN_BARRIER_TIMEOUT = 102,
    ADMISSION_INVALID_CONFIG = 110,
    ADMISSION_INVALID_STATE = 111,
    ADMISSION_INVALID_TOPOLOGY = 112,
    ADMISSION_INVALID_RECORD = 113,
    ADMISSION_LOG_CAPACITY = 114,
    ADMISSION_EPOCH_MISMATCH = 115,
    ADMISSION_INVALID_FRONTIER = 116,
    BLOCK_CURSOR_INVALID = 117,
    BLOCK_COMPLETION_INVALID = 118,
    ORDERED_CANDIDATE_INVALID = 119,
};

struct alignas(128) SchedulerRunControl {
    uint64_t aic_task_count;
    uint64_t aiv_task_count;
    uint64_t active_worker_count;
    uint64_t expected_task_count;
    uint64_t inline_completed_count;
    uint64_t aic_active_worker_count;
    uint64_t aiv_active_worker_count;
    uint64_t aic_worker_demand;
    volatile uint64_t dispatch_payloads_offset;
    uint64_t task_metadata_offset;
    uint64_t ready_inboxes_offset;
    uint64_t ready_directory_offset;
    uint64_t aiv_worker_demand;
    uint64_t graph_summary_offset;
    uint64_t scheduler_count;
    // Pads the host-written configuration to one full cache line, which is what
    // keeps the counters below on a line of their own. The static_assert on
    // executed_task_count's offset is the check that says so.
    uint64_t configuration_line_padding;

    volatile uint64_t executed_task_count;
    volatile uint64_t resolved_task_count;
    volatile uint64_t bootstrap_arrived_count;
    volatile uint64_t bootstrap_complete;
    uint64_t completion_wait_start_cycles;
    uint64_t bootstrap_complete_cycles;
    uint64_t all_tasks_resolved_cycles;
    uint64_t shutdown_ready_cycles;
    uint64_t completion_poll_count;
    uint64_t completion_poll_cycles;
    uint64_t error_poll_count;
    volatile uint64_t bootstrap_scan_arrived_count;
    volatile uint64_t bootstrap_scan_complete;
    volatile uint64_t scheduler_timeout_cycles;
    volatile uint64_t chip_swimlane_level;
    volatile uint64_t sampled_task_timing_enabled;

    volatile uint64_t error_claimed;
    volatile uint64_t scheduler_error;
    volatile uint64_t error_task_ref;
    volatile uint64_t error_core_id;
    volatile uint64_t error_core_type;
    volatile uint64_t error_graph_task_count;
    volatile uint64_t error_storage_address;
    volatile uint64_t error_reserved_address;
    volatile uint64_t error_task_window_last_index;
    volatile uint64_t error_site;
    uint64_t error_reserved[6];
};

struct alignas(128) AicpuThreadLifecycleTrace {
    uint64_t aicpu_thread_id;
    uint64_t handshake_start_cycles;
    uint64_t handshake_complete_cycles;
    uint64_t config_start_cycles;
    uint64_t topology_complete_cycles;
    uint64_t context_publish_start_cycles;
    uint64_t context_publish_complete_cycles;
    uint64_t bootstrap_wait_start_cycles;
    uint64_t bootstrap_complete_cycles;
    uint64_t register_release_start_cycles;
    uint64_t register_release_end_cycles;
    uint64_t exit_signal_start_cycles;
    uint64_t exit_signal_end_cycles;
    uint64_t exit_wait_start_cycles;
    uint64_t exit_wait_end_cycles;
    uint64_t reserved;
};

struct alignas(128) SchedulerWorkerContext {
    volatile int32_t core_type;
    int32_t physical_core_id;
    volatile int32_t type_rank;
    uint32_t worker_reserved;
    volatile uint64_t run_control_offset;
    volatile uint64_t task_controls_offset;
    // Base of this run's ChipTaskStorage array. One address: a task's descriptor
    // and payload are members of one entry, so AICore strides by the storage.
    volatile uint64_t graph_storage_address;
    // Held so the wire layout survived the collapse from two addresses to one.
    // Whoever wires this context up must write 0 here and reject a non-zero read,
    // or a later field can be reintroduced over a value someone else is using.
    volatile uint64_t graph_reserved_address;
    volatile uint64_t scheduler_state_base_address;
    volatile uint64_t dispatch_payload_offset;
    volatile uint64_t trace_cells_offset;
    volatile uint64_t task_window_last_index;
    volatile uint64_t graph_task_count;
    volatile uint64_t worker_index;
    volatile uint64_t scheduler_ssbuf_reserved0;
    volatile uint64_t scheduler_ssbuf_reserved1;
    volatile uint64_t aicpu_lifecycle_traces_offset;

    alignas(128) volatile uint64_t task_metadata_offset;
    volatile uint64_t ready_inboxes_offset;
    volatile uint64_t ready_directory_offset;
    volatile uint64_t activity_buffers_offset;
    volatile uint64_t worker_contexts_offset;
    volatile uint64_t scheduler_ssbuf_reserved2;
    // The active callable's registration-owned kernel-entry table: an absolute
    // device address, not a scheduler-state offset, because the table lives in
    // that callable's registration block and outlives every run bound to it.
    // `callable_addresses_count` is the table's own length, and the only
    // bound available here: the scheduler state reserves no callable array.
    volatile uint64_t callable_addresses_address;
    volatile uint64_t runtime_worker_count;
    uint64_t bootstrap_scan_end_cycles;
    uint64_t target_bootstrap_start_cycles;
    uint64_t target_bootstrap_end_cycles;
    uint64_t bootstrap_target_aic_cycles;
    uint64_t bootstrap_target_aiv_cycles;
    volatile uint64_t callable_addresses_count;
    uint64_t bootstrap_timing_reserved;

    alignas(128) volatile uint64_t graph_summary_offset;
    volatile uint64_t scheduler_count;
    volatile uint64_t cluster_count;
    volatile uint64_t cluster_index;
    volatile uint64_t scheduler_index;
    volatile uint64_t is_scheduler;
    volatile uint64_t cluster_worker_ids[3];
    // Mix slots' payloads and executor traces, both addressed by the holding
    // AIV's rank within its type rather than by worker id. They sit in the
    // configuration prefix because that is the span a core observes before
    // reading any of it; a config offset outside it reads as whatever the line
    // last held.
    volatile uint64_t mix_payloads_offset;
    volatile uint64_t mix_traces_offset;
    uint64_t topology_reserved[1];

    alignas(128) uint64_t bootstrap_task_count;
    uint64_t ready_enqueue_count;
    uint64_t ready_batch_count;
    uint64_t ready_pop_count;
    uint64_t ready_steal_count;
    uint64_t ready_cas_retry_count;
    uint64_t ready_contention_giveup_count;
    uint64_t ready_stats_reserved[3];
    uint64_t executed_task_count;
    uint64_t task_state_poll_count;
    uint64_t fanin_state_load_count;
    uint64_t wake_register_count;
    uint64_t idle_iteration_count;
    uint64_t backoff_cycles;

    uint64_t wake_cas_retry_count;
    uint64_t wake_closed_retry_count;
    uint64_t wake_migrate_count;
    uint64_t wake_close_count;
    uint64_t completion_enqueue_count;
    uint64_t completion_resolve_count;
    uint64_t trace_aicore_entry_cycles;
    uint64_t trace_handshake_publish_cycles;
    uint64_t trace_register_release_cycles;
    uint64_t trace_descriptor_cache_observed_cycles;
    uint64_t completion_stats_reserved[2];
    uint64_t reserved_ready_to_kernel_cycles;
    uint64_t reserved_ready_to_kernel_max_cycles;
    uint64_t payload_cycles;
    uint64_t kernel_cycles;

    uint64_t completion_enqueue_cycles;
    uint64_t bootstrap_start_cycles;
    uint64_t bootstrap_end_reserved;
    uint64_t drain_start_cycles;
    uint64_t drain_end_cycles;
    uint64_t exit_wait_start_cycles;
    uint64_t exit_observed_cycles;
    uint64_t final_stats_publish_start_cycles;
    uint64_t final_stats_publish_end_cycles;
    uint64_t exit_ack_publish_cycles;
    uint64_t bootstrap_slot_fill_aic_cycles;
    uint64_t bootstrap_slot_fill_aiv_cycles;
    uint64_t termination_reserved[4];

    // Preserve the shared wire layout after retiring the unconsumed
    // Scheduler-tail profiling payload.
    uint64_t scheduler_tail_reserved[32];
};

struct alignas(128) SchedulerTaskTrace {
    // Dispatch publishes this line before READY; completion publishes valid.
    volatile uint64_t valid;
    uint64_t ready_source;
    uint64_t worker_id;
    uint64_t task_ref;
    uint64_t state_probe_scheduler_worker_id;
    uint64_t state_probe_start_cycles;
    uint64_t state_probe_end_cycles;
    uint64_t publication_mode;

    uint64_t kernel_start_cycles;
    uint64_t kernel_end_cycles;
    uint64_t ready_scan_start_cycles;
    uint64_t ready_observe_cycles;
    uint64_t completion_id;
    uint64_t completion_inbox_index;

    alignas(64) uint64_t ready_transition_cycles;
    uint64_t fanin_timing_reserved[4];
    uint64_t aicore_entry_cycles;
    uint64_t handshake_publish_cycles;
    uint64_t register_release_cycles;

    // The dispatching Scheduler owns this cache line through completion.
    uint64_t dispatch_start_cycles;
    uint64_t dispatch_end_cycles;
    uint64_t dispatch_scheduler_worker_id;
    uint64_t dispatch_loop_iter;
    uint64_t complete_start_cycles;
    uint64_t complete_end_cycles;
    uint64_t complete_scheduler_worker_id;
    uint64_t complete_loop_iter;
    uint64_t refill_scheduler_worker_id;
    uint64_t refill_start_cycles;
    uint64_t refill_end_cycles;
    uint64_t refill_task_ref;
    uint64_t refill_loop_iter;
    uint64_t completion_reserved[3];

    // The completion Scheduler consolidates the Executor's staged timing here.
    uint64_t descriptor_cache_observed_cycles;
    uint64_t executor_reserved[3];
};

static_assert(sizeof(SchedulerTaskMetadata) == 16, "task metadata layout changed");
static_assert(alignof(SchedulerTaskMetadata) == 16, "task metadata alignment changed");
static_assert(offsetof(SchedulerTaskMetadata, timing_slot) == 12, "timing tag offset changed");
static_assert(offsetof(SchedulerTaskMetadata, logical_block_num) == 14, "block count must occupy the metadata tail");
static_assert(sizeof(SchedulerTaskControl) == 128, "task control layout changed");
static_assert(alignof(SchedulerTaskControl) == 128, "task control alignment changed");
static_assert(offsetof(SchedulerTaskControl, next_block) == 24, "block cursor must use the atomic line");
static_assert(offsetof(SchedulerTaskControl, completed_blocks) == 32, "completion counter must use the atomic line");
static_assert(offsetof(SchedulerTaskControl, admission_state) == 40, "admission state must use the atomic line");
static_assert(sizeof(SchedulerAdmissionRecord) == 64, "admission header must occupy one cache line");
static_assert(sizeof(SchedulerAdmissionControl) == 256, "admission control layout changed");
static_assert(alignof(SchedulerAdmissionControl) == 128, "admission control requires isolated atomic regions");
static_assert(offsetof(SchedulerAdmissionControl, commit_tail) == 128, "tail must not share the lock's atomic region");
static_assert(sizeof(SchedulerGroupDescriptor) == 128, "group descriptor layout changed");
static_assert(sizeof(SchedulerGroupLaneAck) == 128, "lane acknowledgements require separate atomic regions");
static_assert(sizeof(SchedulerGroupMailbox) == 640, "group mailbox layout changed");
static_assert(offsetof(SchedulerGroupMailbox, published_seq) == 128, "descriptor and publication must be separate");
static_assert(offsetof(SchedulerGroupMailbox, lanes) == 256, "each lane must own its acknowledgement region");
// Both chain links share the atomic line. A link has several producers, so it is
// written as a single device word; keeping it off the line-published half is what
// stops one producer's writeback from restoring its stale copy of the link.
static_assert(offsetof(SchedulerTaskControl, next_waiter) < 64, "wake link must sit on the atomic line");
static_assert(
    offsetof(SchedulerTaskControl, next_fanin_index) == 64, "line-published waiter metadata needs its own line"
);
static_assert(sizeof(SchedulerReadyInbox) == 128, "ready inbox layout changed");
static_assert(alignof(SchedulerReadyInbox) == 128, "ready inbox alignment changed");
static_assert(sizeof(SchedulerGraphSummary) == 128, "graph summary layout changed");
static_assert(alignof(SchedulerGraphSummary) == 128, "graph summary alignment changed");
static_assert(sizeof(SchedulerReadyDirectoryShard) == 64, "ready directory shard must occupy one cache line");
static_assert(alignof(SchedulerReadyDirectoryShard) == 64, "ready directory shard alignment changed");
static_assert(
    SCHEDULER_READY_DIRECTORY_SHARD_COUNT * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD >=
        scheduler_ready_queue_capacity(1),
    "ready directory does not cover the widest ready queue"
);
static_assert(
    scheduler_ready_queue_base(SCHEDULER_READY_QUEUE_COUNT - 1) +
            scheduler_ready_queue_capacity(SCHEDULER_READY_QUEUE_COUNT - 1) ==
        SCHEDULER_READY_INBOX_CAPACITY,
    "per-type inbox segments must tile the inbox array exactly"
);
static_assert(
    scheduler_ready_queue_base(1) == scheduler_ready_queue_capacity(0) &&
        scheduler_ready_queue_base(2) == scheduler_ready_queue_base(1) + scheduler_ready_queue_capacity(1),
    "per-type inbox segments must be contiguous"
);
static_assert(
    sizeof(SchedulerReadyDirectory) ==
        (SCHEDULER_READY_QUEUE_COUNT * SCHEDULER_READY_DIRECTORY_SHARD_COUNT * 64 + 127) / 128 * 128,
    "ready directory layout changed"
);
static_assert(sizeof(SchedulerExecutorTaskTrace) == 48, "executor trace payload layout changed");
static_assert(alignof(SchedulerExecutorTaskTrace) == alignof(uint64_t), "executor trace must have natural alignment");
static_assert(offsetof(SchedulerExecutorTaskTrace, completion_inbox_index) == 40, "executor trace layout changed");
static_assert(sizeof(SchedulerRunControl) == 384, "run control layout changed");
static_assert(alignof(SchedulerRunControl) == 128, "run control alignment changed");
static_assert(offsetof(SchedulerRunControl, executed_task_count) == 128, "lifecycle atomics need their own line");
static_assert(offsetof(SchedulerRunControl, error_claimed) == 256, "error state needs its own line");
static_assert(offsetof(SchedulerRunControl, error_site) == 328, "error site ABI changed");
static_assert(sizeof(AicpuThreadLifecycleTrace) == 128, "AICPU lifecycle trace layout changed");
static_assert(sizeof(SchedulerWorkerContext) == 1024, "worker context layout changed");
static_assert(alignof(SchedulerWorkerContext) == 128, "worker context alignment changed");
static_assert(offsetof(SchedulerWorkerContext, task_metadata_offset) == 128, "runtime offsets changed");
static_assert(offsetof(SchedulerWorkerContext, graph_summary_offset) == 256, "topology offsets changed");
static_assert(offsetof(SchedulerWorkerContext, bootstrap_task_count) == 384, "worker stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, wake_cas_retry_count) == 512, "wake stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, completion_enqueue_cycles) == 640, "termination stats offset changed");
static_assert(offsetof(SchedulerWorkerContext, scheduler_tail_reserved) == 768, "worker reserved tail offset changed");
static_assert(sizeof(SchedulerTaskTrace) == 384, "task trace layout changed");
static_assert(
    offsetof(SchedulerTaskTrace, dispatch_start_cycles) % 64 == 0,
    "Scheduler-owned dispatch and completion fields must begin on a cache-line boundary"
);
static_assert(
    offsetof(SchedulerTaskTrace, complete_loop_iter) / 64 == offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64,
    "dispatch and completion fields must share their Scheduler-owned cache line"
);
static_assert(
    offsetof(SchedulerTaskTrace, refill_scheduler_worker_id) % 64 == 0,
    "Scheduler-owned refill fields must begin on a cache-line boundary"
);
static_assert(
    offsetof(SchedulerTaskTrace, descriptor_cache_observed_cycles) % 64 == 0,
    "consolidated descriptor timing must begin on a cache-line boundary"
);

template <typename T>
inline __aicore__ __gm__ T *scheduler_state_at(__gm__ void *base, uint64_t offset) {
    return reinterpret_cast<__gm__ T *>(reinterpret_cast<__gm__ uint8_t *>(base) + offset);
}

// Worker i's context, from the base worker 0 sits at — the address the host
// publishes in DeviceRuntimeLaunchDesc::scheduler_bootstrap. The AICore derives
// its own pre-READY context with this and the AICPU republishes the same address
// onto each handshake at hand-off, so the stride has one definition.
inline __aicore__ uint64_t scheduler_worker_context_address(uint64_t worker_context_base, int32_t worker_index) {
    return worker_context_base + static_cast<uint64_t>(worker_index) * sizeof(SchedulerWorkerContext);
}

#if !defined(__CCE_AICORE__)
#include <type_traits>
static_assert(std::is_standard_layout_v<AicoreSchedulerLayout> && std::is_trivially_copyable_v<AicoreSchedulerLayout>);
static_assert(std::is_standard_layout_v<SchedulerIdleRecord> && std::is_trivially_copyable_v<SchedulerIdleRecord>);
static_assert(
    std::is_standard_layout_v<SchedulerActivityBuffer> && std::is_trivially_copyable_v<SchedulerActivityBuffer>
);
static_assert(std::is_standard_layout_v<SchedulerTaskMetadata> && std::is_trivially_copyable_v<SchedulerTaskMetadata>);
static_assert(std::is_standard_layout_v<SchedulerTaskControl> && std::is_trivially_copyable_v<SchedulerTaskControl>);
static_assert(
    std::is_standard_layout_v<SchedulerSpmdRequirements> && std::is_trivially_copyable_v<SchedulerSpmdRequirements>
);
static_assert(
    std::is_standard_layout_v<SchedulerAdmissionRecord> && std::is_trivially_copyable_v<SchedulerAdmissionRecord>
);
static_assert(
    std::is_standard_layout_v<SchedulerAdmissionControl> && std::is_trivially_copyable_v<SchedulerAdmissionControl>
);
static_assert(
    std::is_standard_layout_v<SchedulerGroupDescriptor> && std::is_trivially_copyable_v<SchedulerGroupDescriptor>
);
static_assert(std::is_standard_layout_v<SchedulerGroupLaneAck> && std::is_trivially_copyable_v<SchedulerGroupLaneAck>);
static_assert(std::is_standard_layout_v<SchedulerGroupMailbox> && std::is_trivially_copyable_v<SchedulerGroupMailbox>);
static_assert(std::is_standard_layout_v<SchedulerReadyInbox> && std::is_trivially_copyable_v<SchedulerReadyInbox>);
static_assert(std::is_standard_layout_v<SchedulerGraphSummary> && std::is_trivially_copyable_v<SchedulerGraphSummary>);
static_assert(
    std::is_standard_layout_v<SchedulerReadyDirectory> && std::is_trivially_copyable_v<SchedulerReadyDirectory>
);
static_assert(std::is_standard_layout_v<SchedulerRunControl> && std::is_trivially_copyable_v<SchedulerRunControl>);
static_assert(
    std::is_standard_layout_v<AicpuThreadLifecycleTrace> && std::is_trivially_copyable_v<AicpuThreadLifecycleTrace>
);
static_assert(
    std::is_standard_layout_v<SchedulerWorkerContext> && std::is_trivially_copyable_v<SchedulerWorkerContext>
);
static_assert(std::is_standard_layout_v<SchedulerTaskTrace> && std::is_trivially_copyable_v<SchedulerTaskTrace>);

inline bool scheduler_layout_checked_add(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || rhs > UINT64_MAX - lhs) return false;
    *out = lhs + rhs;
    return true;
}

inline bool scheduler_layout_checked_mul(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (out == nullptr || (lhs != 0 && rhs > UINT64_MAX / lhs)) return false;
    *out = lhs * rhs;
    return true;
}

inline bool scheduler_layout_checked_align(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    uint64_t added = 0;
    if (!scheduler_layout_checked_add(value, alignment - 1, &added)) return false;
    *out = added & ~(alignment - 1);
    return true;
}

inline bool scheduler_layout_reserve(uint64_t *cursor, uint64_t size, uint64_t alignment, uint64_t *offset) {
    uint64_t aligned = 0;
    if (cursor == nullptr || offset == nullptr || !scheduler_layout_checked_align(*cursor, alignment, &aligned))
        return false;
    uint64_t end = 0;
    if (!scheduler_layout_checked_add(aligned, size, &end)) return false;
    *offset = aligned;
    *cursor = end;
    return true;
}

inline bool scheduler_spmd_requirements_valid(const SchedulerSpmdRequirements &spmd, uint64_t task_count) {
    if (spmd.cooperative_task_count > task_count || spmd.mix_task_count > spmd.cooperative_task_count) return false;
    if (spmd.worker_count == 0)
        return spmd.cluster_count == 0 && spmd.cooperative_task_count == 0 && spmd.mix_task_count == 0;
    return spmd.worker_count <= SCHEDULER_WORKER_CAPACITY && spmd.worker_count % PLATFORM_CORES_PER_BLOCKDIM == 0 &&
           spmd.cluster_count == spmd.worker_count / PLATFORM_CORES_PER_BLOCKDIM;
}

inline bool scheduler_plan_spmd_storage(uint64_t *cursor, AicoreSchedulerLayout *layout) {
    if (cursor == nullptr || layout == nullptr) return false;
    if (layout->spmd.cooperative_task_count == 0) return true;
    layout->admission_member_word_count = (layout->spmd.worker_count + 63) / 64;
    uint64_t bytes = 0;
    uint64_t record_bytes = 0;
    if (!scheduler_layout_checked_mul(layout->admission_member_word_count, sizeof(uint64_t), &bytes) ||
        !scheduler_layout_checked_add(sizeof(SchedulerAdmissionRecord), bytes, &record_bytes) ||
        !scheduler_layout_checked_align(record_bytes, SCHEDULER_STATE_ALIGNMENT, &layout->admission_record_stride) ||
        !scheduler_layout_reserve(
            cursor, sizeof(SchedulerAdmissionControl), alignof(SchedulerAdmissionControl),
            &layout->admission_control_offset
        ) ||
        !scheduler_layout_checked_mul(layout->task_count, layout->admission_record_stride, &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->admission_records_offset) ||
        !scheduler_layout_checked_mul(layout->spmd.cooperative_task_count, sizeof(int64_t), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->admission_log_offset))
        return false;
    if (layout->spmd.mix_task_count == 0) return true;
    if (!scheduler_layout_checked_mul(layout->spmd.cluster_count, sizeof(SchedulerGroupMailbox), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, alignof(SchedulerGroupMailbox), &layout->group_mailboxes_offset) ||
        !scheduler_layout_checked_mul(
            layout->spmd.mix_task_count, layout->spmd.cluster_count, &layout->mix_close_seal_count
        ) ||
        !scheduler_layout_checked_mul(layout->mix_close_seal_count, sizeof(uint64_t), &bytes) ||
        !scheduler_layout_reserve(cursor, bytes, SCHEDULER_STATE_ALIGNMENT, &layout->mix_close_seals_offset))
        return false;
    return true;
}

inline bool scheduler_plan_layout(
    uint64_t task_count, uint64_t aic_task_count, uint64_t aiv_task_count, AicoreSchedulerLayout *layout,
    bool enable_activity_profiling = false, uint64_t trace_count = 0, const SchedulerSpmdRequirements &spmd = {}
) {
    // A ready inbox holds its endpoints as int32 task refs, so a graph whose
    // refs would not fit that encoding cannot be scheduled at all.
    if (layout == nullptr || aic_task_count > task_count || aiv_task_count > task_count ||
        task_count > static_cast<uint64_t>(INT32_MAX) || !scheduler_spmd_requirements_valid(spmd, task_count))
        return false;
    AicoreSchedulerLayout next{};
    next.task_count = task_count;
    next.aic_task_count = aic_task_count;
    next.aiv_task_count = aiv_task_count;
    next.spmd = spmd;
    uint64_t cursor = 0;
    uint64_t bytes = 0;
#define SCHEDULER_RESERVE_ARRAY(count, type, field)                 \
    (scheduler_layout_checked_mul((count), sizeof(type), &bytes) && \
     scheduler_layout_reserve(&cursor, bytes, alignof(type), &next.field))
    if (!scheduler_layout_reserve(
            &cursor, sizeof(SchedulerRunControl), alignof(SchedulerRunControl), &next.run_control_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(
            PLATFORM_MAX_AICPU_THREADS, AicpuThreadLifecycleTrace, aicpu_lifecycle_traces_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerWorkerContext, worker_contexts_offset) ||
        !SCHEDULER_RESERVE_ARRAY(
            SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT, DispatchPayload, dispatch_payloads_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskMetadata, task_metadata_offset) ||
        !SCHEDULER_RESERVE_ARRAY(task_count, SchedulerTaskControl, task_controls_offset) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_READY_INBOX_CAPACITY, SchedulerReadyInbox, ready_inboxes_offset) ||
        !scheduler_layout_reserve(
            &cursor, sizeof(SchedulerReadyDirectory), alignof(SchedulerReadyDirectory), &next.ready_directory_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_MIX_SLOT_CAPACITY, DispatchPayload, mix_payloads_offset) ||
        !SCHEDULER_RESERVE_ARRAY(SCHEDULER_MIX_SLOT_CAPACITY, SchedulerExecutorTaskTrace, mix_traces_offset) ||
        !scheduler_layout_reserve(
            &cursor, sizeof(SchedulerGraphSummary), alignof(SchedulerGraphSummary), &next.graph_summary_offset
        ) ||
        !SCHEDULER_RESERVE_ARRAY(trace_count == 0 ? task_count : trace_count, SchedulerTaskTrace, trace_cells_offset) ||
        (enable_activity_profiling &&
         !SCHEDULER_RESERVE_ARRAY(SCHEDULER_WORKER_CAPACITY, SchedulerActivityBuffer, activity_buffers_offset)) ||
        !scheduler_plan_spmd_storage(&cursor, &next) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) {
#undef SCHEDULER_RESERVE_ARRAY
        return false;
    }
#undef SCHEDULER_RESERVE_ARRAY
    *layout = next;
    return true;
}

inline bool
scheduler_init_data_from_layout(void *base, const AicoreSchedulerLayout &layout, uint64_t device_base_address) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 ||
        device_base_address == 0 || (device_base_address & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0 ||
        layout.total_size > UINT64_MAX - device_base_address)
        return false;
    __builtin_memset(base, 0, static_cast<size_t>(layout.total_size));
    auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    for (uint64_t i = 0; i < layout.task_count; ++i) {
        controls[i].state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        controls[i].wake_list_head = SCHEDULER_WAKE_LIST_OPEN;
        controls[i].next_waiter = SCHEDULER_TASK_REF_INVALID;
        controls[i].waiting_producer = static_cast<int32_t>(SCHEDULER_TASK_REF_INVALID);
    }
    auto *ready = scheduler_state_at<SchedulerReadyInbox>(base, layout.ready_inboxes_offset);
    for (uint64_t i = 0; i < SCHEDULER_READY_INBOX_CAPACITY; ++i)
        ready[i].endpoints = SCHEDULER_READY_ENDPOINTS_EMPTY;
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(base, layout.worker_contexts_offset);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        contexts[worker].physical_core_id = -1;
        contexts[worker].cluster_index = UINT64_MAX;
        contexts[worker].scheduler_index = UINT64_MAX;
        contexts[worker].activity_buffers_offset = layout.activity_buffers_offset;
        contexts[worker].cluster_worker_ids[0] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[1] = UINT64_MAX;
        contexts[worker].cluster_worker_ids[2] = UINT64_MAX;
    }
    const auto initialize_payloads = [base, device_base_address](uint64_t offset, uint64_t count) {
        for (uint64_t index = 0; index < count; ++index) {
            const uint64_t payload_offset = offset + index * sizeof(DispatchPayload);
            auto *payload = scheduler_state_at<DispatchPayload>(base, payload_offset);
            const uint64_t device_payload = device_base_address + payload_offset;
            payload->args[PAYLOAD_LOCAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, local_context);
            payload->args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = device_payload + offsetof(DispatchPayload, global_context);
            payload->local_context.block_num = 1;
            payload->local_context.async_ctx.task_token = TaskId::invalid();
        }
    };
    initialize_payloads(layout.dispatch_payloads_offset, SCHEDULER_WORKER_CAPACITY * SCHEDULER_PENDING_SLOT_COUNT);
    initialize_payloads(layout.mix_payloads_offset, SCHEDULER_MIX_SLOT_CAPACITY);
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, layout.graph_summary_offset);
    summary->mix_task_count = layout.spmd.mix_task_count;
    summary->cooperative_task_count = layout.spmd.cooperative_task_count;
    summary->worker_count = layout.spmd.worker_count;
    summary->cluster_count = layout.spmd.cluster_count;
    summary->admission_member_word_count = layout.admission_member_word_count;
    summary->admission_records_offset = layout.admission_records_offset;
    summary->admission_record_stride = layout.admission_record_stride;
    summary->admission_log_offset = layout.admission_log_offset;
    summary->admission_control_offset = layout.admission_control_offset;
    summary->group_mailboxes_offset = layout.group_mailboxes_offset;
    summary->mix_close_seals_offset = layout.mix_close_seals_offset;
    if (layout.spmd.cooperative_task_count != 0) {
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const uint64_t offset = layout.admission_records_offset + task * layout.admission_record_stride;
            auto *record = scheduler_state_at<SchedulerAdmissionRecord>(base, offset);
            record->task_ref = SCHEDULER_TASK_REF_INVALID;
            record->order_key = UINT64_MAX;
            record->member_words_offset = offset + sizeof(SchedulerAdmissionRecord);
        }
        // Log elements and packed close seals are atomic-only device words;
        // adjacent publishers must never write back a cached whole line.
        auto *log = scheduler_state_at<int64_t>(base, layout.admission_log_offset);
        for (uint64_t index = 0; index < layout.spmd.cooperative_task_count; ++index)
            log[index] = SCHEDULER_TASK_REF_INVALID;
    }
    if (layout.spmd.mix_task_count != 0) {
        auto *mailboxes = scheduler_state_at<SchedulerGroupMailbox>(base, layout.group_mailboxes_offset);
        for (uint64_t cluster = 0; cluster < layout.spmd.cluster_count; ++cluster) {
            auto &descriptor = mailboxes[cluster].descriptor;
            descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
            descriptor.order_key = UINT64_MAX;
            descriptor.block_idx = UINT32_MAX;
            descriptor.cluster_index = UINT32_MAX;
            for (uint8_t &subtask : descriptor.lane_subtasks)
                subtask = UINT8_MAX;
        }
    }
    return true;
}

inline bool scheduler_init_spmd_task_records(void *base, const AicoreSchedulerLayout &layout, uint64_t run_epoch) {
    if (base == nullptr || (reinterpret_cast<uintptr_t>(base) & (SCHEDULER_STATE_ALIGNMENT - 1)) != 0) return false;
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
    uint64_t cooperative_count = 0;
    uint64_t mix_count = 0;
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        const uint32_t active_subtasks = static_cast<uint32_t>(__builtin_popcount(meta.active_mask));
        uint32_t expected_subtasks = 0;
        if ((meta.active_mask & ~UINT8_C(7)) != 0 ||
            !scheduler_subtask_count_checked(meta.logical_block_num, active_subtasks, &expected_subtasks) ||
            scheduler_task_is_mix(meta.flags) != (active_subtasks > 1) ||
            scheduler_task_is_spmd(meta.flags) != (meta.logical_block_num > 1) ||
            (scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num == 1) ||
            (scheduler_task_has_predicate(meta.flags) && (meta.logical_block_num != 1 || active_subtasks != 1)) ||
            meta.timing_slot < -1 || meta.timing_slot >= SCHEDULER_TASK_TIMING_SLOT_COUNT)
            return false;
        if (scheduler_task_requires_admission(meta)) ++cooperative_count;
        if (scheduler_task_is_mix(meta.flags)) ++mix_count;
    }
    if (cooperative_count != layout.spmd.cooperative_task_count || mix_count != layout.spmd.mix_task_count)
        return false;

    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, layout.graph_summary_offset);
    summary->run_epoch = run_epoch;
    uint64_t mix_index = 0;
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_requires_admission(meta)) continue;
        auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base, layout.admission_records_offset + task * layout.admission_record_stride
        );
        record->task_ref = static_cast<int64_t>(task);
        record->logical_block_num = meta.logical_block_num;
        record->active_mask = meta.active_mask;
        record->flags = meta.flags;
        if (scheduler_task_is_mix(meta.flags)) {
            record->mix_close_seals_offset =
                layout.mix_close_seals_offset + mix_index * layout.spmd.cluster_count * sizeof(uint64_t);
            ++mix_index;
        }
    }
    return true;
}
#endif
