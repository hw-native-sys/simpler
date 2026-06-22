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
#include <cstdint>

#include "common/core_type.h"
#include "common/platform_config.h"
#include "pto_runtime2_types.h"
#include "spin_hint.h"

// =============================================================================
// Profiling macros (compile-time gated)
// =============================================================================

#if PTO2_PROFILING
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
// The budget is platform-defined (PLATFORM_SCHEDULER_TIMEOUT_MS in spin_hint.h)
// because the safe value differs per variant: onboard trims it to 2 s so the
// AICPU detects a hang and flushes its diagnostics (tensor dump, in-flight
// partial output) before STARS reaps the op and poisons the context (chain:
// this < op-exec < host stream-sync, platform_config.h); sim has no STARS to
// race and keeps the full 5 s #897 headroom. See spin_hint.h for the per-variant
// rationale.
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
#if PTO2_PROFILING
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
    static inline int32_t MAX_CORE_PER_THREAD = 63;
    static constexpr int32_t MAX_CLUSTERS = 63 / 3;

public:
    CoreTracker() = default;

    class BitStates {
    public:
        BitStates() = default;

        explicit BitStates(uint64_t states) :
            states_(states) {}
        void init() { states_ = 0; }

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
        int32_t count() const { return __builtin_popcountll(states_); }
        void clear_bit(int32_t offset) { states_ &= ~(1ULL << offset); }

        // Extract the lowest set bit from mask, clear it, and return its position.
        // Returns -1 if mask is empty.
        int32_t pop_first() {
            if (states_ == 0) return -1;
            int32_t pos = __builtin_ctzll(states_);
            states_ &= states_ - 1;
            return pos;
        }

    private:
        uint64_t states_{0};
    };

public:
    void init(int32_t cluster_count) {
        cluster_count_ = cluster_count;
        aic_mask_.init();
        aiv_mask_.init();
        pending_occupied_.init();
        for (int32_t i = 0; i < cluster_count; i++) {
            aic_mask_ |= BitStates(1ULL << (i * 3));
            aiv_mask_ |= BitStates(6ULL << (i * 3));
        }
        core_states_ = aic_mask_ | aiv_mask_;
    }

    void set_cluster(int32_t cluster_idx, int32_t aic_wid, int32_t aiv0_wid, int32_t aiv1_wid) {
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
            return BitStates(0ULL);
        }
        return BitStates(0ULL);
    }

    int32_t get_aic_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset]; }
    int32_t get_aiv0_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 1]; }
    int32_t get_aiv1_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 2]; }

    int32_t get_aic_core_offset(int32_t cluster_offset) const { return cluster_offset; }
    int32_t get_aiv0_core_offset(int32_t cluster_offset) const { return cluster_offset + 1; }
    int32_t get_aiv1_core_offset(int32_t cluster_offset) const { return cluster_offset + 2; }

    bool is_aic_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> cluster_offset) & BitStates(1ULL)).has_value();
    }
    bool is_aiv0_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 1)) & BitStates(1ULL)).has_value();
    }
    bool is_aiv1_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 2)) & BitStates(1ULL)).has_value();
    }

    // --- State mutation ---

    // Toggle bit at the given bit offset (running <-> idle)
    void change_core_state(int32_t bit_offset) { core_states_ ^= BitStates(1ULL << bit_offset); }

    // --- Pending-occupied tracking ---
    // Tracks whether a core's pending payload slot is occupied (awaiting hardware ACK).
    // SET on dispatch (both running-first and pending), CLEAR on idle or pending_freed.

    void set_pending_occupied(int32_t bit_offset) { pending_occupied_ |= BitStates(1ULL << bit_offset); }
    void clear_pending_occupied(int32_t bit_offset) {
        pending_occupied_ ^= (pending_occupied_ & BitStates(1ULL << bit_offset));
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

    // A MIX block must place all cores named by active_mask the same way:
    // all idle means running placement, all running means pending placement,
    // and any mixed state is retried later.
    MixPlacement classify_mix_cluster(int32_t cluster_offset, uint8_t core_mask) const {
        BitStates used(0ULL);
        if (core_mask & PTO2_SUBTASK_MASK_AIC) {
            used |= BitStates(1ULL << cluster_offset);
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV0) {
            used |= BitStates(1ULL << (cluster_offset + 1));
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV1) {
            used |= BitStates(1ULL << (cluster_offset + 2));
        }
        if (!used.has_value() || (pending_occupied_ & used).has_value()) {
            return MixPlacement::REJECT;
        }

        BitStates idle = core_states_ & used;
        if (idle.count() == used.count()) {
            return MixPlacement::RUNNING;
        }
        if (!idle.has_value()) {
            return MixPlacement::PENDING;
        }
        return MixPlacement::REJECT;
    }

    BitStates get_mix_running_cluster_offset_states(uint8_t core_mask) const {
        BitStates result(0ULL);
        BitStates candidates = get_cluster_offset_states();
        while (candidates.has_value()) {
            int32_t cluster_offset = candidates.pop_first();
            if (classify_mix_cluster(cluster_offset, core_mask) == MixPlacement::RUNNING) {
                result |= BitStates(1ULL << cluster_offset);
            }
        }
        return result;
    }

    int32_t count_mix_running_clusters(uint8_t core_mask) const {
        return get_mix_running_cluster_offset_states(core_mask).count();
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
    int32_t core_num() const { return cluster_count_ * 3; }

private:
    int32_t cluster_count_;
    BitStates aic_mask_;
    BitStates aiv_mask_;
    BitStates core_states_;
    BitStates pending_occupied_;
    int32_t core_id_map_[63];  // bit_position -> worker_id, max 21 clusters * 3
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

#if PTO2_PROFILING
struct alignas(64) SchedL2SwimlaneCounters {
    bool l2_swimlane_enabled{false};
    uint64_t sched_start_ts{0};
    uint64_t sched_complete_cycle{0};
    uint64_t sched_dispatch_cycle{0};
    uint64_t sched_wiring_cycle{0};
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
    // l2_swimlane_level_ >= SCHED_PHASES.
    uint64_t pop_hit{0};
    uint64_t pop_miss{0};
    uint64_t pop_hit_at_last_emit{0};
    uint64_t pop_miss_at_last_emit{0};
    // Activity-fill: coalesced segment covering iterations that emit no
    // Complete/Dispatch bar, so the scheduler lane tiles with no blanks.
    // fill_kind: 0 = none open, else L2SwimlaneSchedPhaseKind value + 1.
    uint32_t fill_kind{0};
    uint64_t fill_start{0};
    uint64_t fill_end{0};
#if PTO2_SCHED_PROFILING
    uint32_t phase_wiring_count{0};
    uint64_t complete_probe_count{0};
    uint64_t complete_hit_count{0};
    uint64_t sched_complete_perf_cycle{0};
    uint64_t sched_dispatch_pop_cycle{0};
    uint64_t sched_dispatch_setup_cycle{0};
#endif
    void reset() { *this = SchedL2SwimlaneCounters{}; }
};
#endif

// =============================================================================
// sync_start drain coordination
// =============================================================================

// When sync_start_pending != 0, all scheduler threads skip dispatch
// (only process completions) until the drain worker finishes launching all blocks.
struct alignas(64) SyncStartDrainState {
    std::atomic<int32_t> sync_start_pending{0};    // 0=normal; -1=initializing; >0=active (value=block_num)
    std::atomic<int32_t> drain_worker_elected{0};  // 0=none; >0: elected thread's (thread_idx+1)
    std::atomic<uint32_t> drain_ack_mask{0};       // bit per thread; all-set = all threads reached ack barrier
    std::atomic<PTO2TaskSlotState *> pending_task{nullptr};  // held task (not re-queued)
    int32_t _pad[10];
};
static_assert(sizeof(SyncStartDrainState) == 64);

#endif  // SCHEDULER_TYPES_H
