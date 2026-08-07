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
#ifndef SCHEDULER_TYPES_H
#define SCHEDULER_TYPES_H

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "common/core_type.h"
#include "common/platform_config.h"
#include "pto_runtime2_types.h"
#include "spin_hint.h"

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
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check orchestrator error every N idle iters

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
// The budget is platform-defined (PLATFORM_SCHEDULER_TIMEOUT_MS in spin_hint.h).
// Onboard keeps it below the STARS op-execute and host stream-sync budgets so
// the AICPU can flush diagnostics before the host-visible timeout chain fires.
// Sim has no STARS or ACL stream-sync timeout, but uses the same no-progress
// watchdog shape. See spin_hint.h for the per-variant rationale.
constexpr int32_t SCHEDULER_TIMEOUT_MS = PLATFORM_SCHEDULER_TIMEOUT_MS;
constexpr uint64_t SCHEDULER_TIMEOUT_CYCLES =
    static_cast<uint64_t>(SCHEDULER_TIMEOUT_MS) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;
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
    PTO2TaskSlotState *running_slot_state;  // offset  8: slot state for running task (nullptr = empty)
    PTO2TaskSlotState *pending_slot_state;  // offset 16: slot state for pending task (nullptr = empty)
    int32_t running_reg_task_id;            // offset 24: register task ID (AICPU_TASK_INVALID = idle)
    int32_t pending_reg_task_id;            // offset 28: pending register task ID (AICPU_TASK_INVALID = none)
    uint32_t dispatch_seq;                  // offset 32: monotonic dispatch counter
    PTO2SubtaskSlot running_subslot;        // offset 36: which subtask slot is running
    PTO2SubtaskSlot pending_subslot;        // offset 37: which subtask slot is pending
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
    int32_t worker_id;          // offset 48: index in runtime.dev.workers[]
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
        static BitStates bit(int32_t offset) { return BitStates(Storage(1) << offset); }

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
        void clear_bit(int32_t offset) { states_ &= ~(Storage(1) << offset); }

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

    BitStates get_valid_cluster_offset_states(PTO2ResourceShape shape) const {
        switch (shape) {
        case PTO2ResourceShape::AIC:
            return core_states_ & aic_mask_;
        case PTO2ResourceShape::AIV:
            return ((core_states_ >> 1) | (core_states_ >> 2)) & aic_mask_;
        case PTO2ResourceShape::MIX:
            return (core_states_ >> 1) & (core_states_ >> 2) & core_states_ & aic_mask_;
        case PTO2ResourceShape::DUMMY:
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
    BitStates get_idle_core_offset_states(PTO2ResourceShape shape) const {
        if (shape == PTO2ResourceShape::AIC) {
            return get_valid_cluster_offset_states(shape) & ~(pending_occupied_ & aic_mask_);
        }
        if (shape == PTO2ResourceShape::AIV) {
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
        if (core_mask & PTO2_SUBTASK_MASK_AIC) {
            used |= BitStates::bit(cluster_offset);
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV0) {
            used |= BitStates::bit(cluster_offset + 1);
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV1) {
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

    // --- Gated MIX split placement ---
    // A gated MIX block can place each of its cores INDEPENDENTLY (idle core -> gated
    // running slot; busy core with a free pending slot -> gated pending slot), because
    // every core waits on the doorbell and nothing executes until the rendezvous rings.
    // This is unsafe for immediate (non-gated) dispatch — hence separate from
    // classify_mix_cluster, which forces a single placement for the whole cluster.

    // Cores of `cluster_offset` named by core_mask.
    BitStates mix_used_cores(int32_t cluster_offset, uint8_t core_mask) const {
        BitStates used;
        if (core_mask & PTO2_SUBTASK_MASK_AIC) used |= BitStates::bit(cluster_offset);
        if (core_mask & PTO2_SUBTASK_MASK_AIV0) used |= BitStates::bit(cluster_offset + 1);
        if (core_mask & PTO2_SUBTASK_MASK_AIV1) used |= BitStates::bit(cluster_offset + 2);
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
        BitStates result;
        BitStates candidates = get_cluster_offset_states();
        while (candidates.has_value()) {
            int32_t cluster_offset = candidates.pop_first();
            if (classify_mix_cluster(cluster_offset, core_mask) == MixPlacement::RUNNING) {
                result |= BitStates::bit(cluster_offset);
            }
        }
        return result;
    }

    int32_t count_mix_running_clusters(uint8_t core_mask) const {
        return get_mix_running_cluster_offset_states(core_mask).count();
    }

    // Number of whole logical blocks this tracker can accept for sync_start
    // staging. AIC/AIV count cores; MIX counts clusters because one logical MIX
    // block may occupy multiple cores in the same cluster. Gated early staging
    // includes pending slots, while ready staging is restricted to idle slots.
    int32_t count_available_blocks(PTO2ResourceShape shape, uint8_t core_mask, bool include_pending) const {
        if (shape == PTO2ResourceShape::MIX) {
            return include_pending ? count_mix_split_clusters(core_mask) : count_mix_running_clusters(core_mask);
        }
        if (shape == PTO2ResourceShape::DUMMY) return 0;

        int32_t available = get_idle_core_offset_states(shape).count();
        if (include_pending) {
            available += get_pending_core_offset_states(shape).count();
        }
        return available;
    }

    BitStates get_pending_core_offset_states(PTO2ResourceShape shape) const {
        if (shape == PTO2ResourceShape::MIX) {
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
        if (shape == PTO2ResourceShape::AIC) {
            return (~core_states_) & aic_mask_ & ~(pending_occupied_ & aic_mask_);
        }
        // AIV
        return (~core_states_) & aiv_mask_ & ~pending_occupied_;
    }

    // --- Two-phase dispatch unified query ---

    enum class DispatchPhase : uint8_t { IDLE, PENDING };

    BitStates get_dispatchable_cores(PTO2ResourceShape shape, DispatchPhase phase) const {
        return (phase == DispatchPhase::IDLE) ? get_idle_core_offset_states(shape) :
                                                get_pending_core_offset_states(shape);
    }

    // --- Bit offset <-> worker_id mapping ---

    int32_t get_core_id_by_offset(int32_t offset) const { return core_id_map_[offset]; }

    const int32_t *core_ids() const { return core_id_map_; }
    int32_t core_num() const { return cluster_count_ * PLATFORM_CORES_PER_BLOCKDIM; }

private:
    int32_t cluster_count_;
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
    uint64_t sched_async_cycle{0};
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
    std::atomic<PTO2TaskSlotState *> pending_task{nullptr};  // held task (not re-queued)
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

#endif  // SCHEDULER_TYPES_H
