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

constexpr int32_t MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;

// PLATFORM_MAX_IDLE_ITERATIONS was removed upstream; fixed cadence matches a5's
// equivalent (used only for per-thread diagnostic logging, not for the fatal-
// timeout path which uses wall-clock).
constexpr int32_t STALL_LOG_INTERVAL = 480000;
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check orchestrator error every N idle iters

constexpr int32_t SCHEDULER_TIMEOUT_MS = PLATFORM_SCHEDULER_TIMEOUT_MS;
constexpr uint64_t SCHEDULER_TIMEOUT_CYCLES = static_cast<uint64_t>(SCHEDULER_TIMEOUT_MS) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000);
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;
constexpr int32_t PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int32_t PROGRESS_LOG_INTERVAL = 250;      // log every N completions after threshold

enum class LoopAction : int8_t
{
    NONE,        // cold path did not trigger; proceed normally
    BREAK_LOOP,  // equivalent to 'break' from the while(true) loop
};

// Per-thread phase profiling. Accumulates cumulative cycle counts and entry
// counts for each phase of resolve_and_dispatch's main loop. Dumped once at
// loop exit via LOG_INFO_V9 — the hot path only does cycle counter math.
struct alignas(64) SchedulerThreadProfile
{
    uint64_t total_cycles{0};
    uint64_t completion_cycles{0};
    // Sub-phase of completion: time spent INSIDE complete_slot_task, and
    // count of times it ran (one per subtask completion observed).
    uint64_t complete_task_cycles{0};
    uint64_t complete_task_calls{0};
    // Sub-phase of completion: count of cores scanned per iter (proxy for
    // cond_ptr read cost; aggregate / completion_iters = avg cores/iter).
    uint64_t cores_scanned{0};
    uint64_t async_wait_cycles{0};
    uint64_t drain_wiring_cycles{0};
    uint64_t spsc_drain_cycles{0};    // sub-phase of drain_wiring: SPSC → pending FIFO
    uint64_t pending_poll_cycles{0};  // sub-phase of drain_wiring: pending FIFO → ready
    uint64_t dummy_drain_cycles{0};
    uint64_t dispatch_cycles{0};
    uint64_t idle_spin_cycles{0};
    uint64_t completion_iters{0};
    uint64_t async_wait_iters{0};
    uint64_t drain_wiring_iters{0};
    uint64_t spsc_drain_iters{0};
    uint64_t pending_poll_iters{0};
    uint64_t pending_poll_skipped{0};  // (a) gate hits: poll calls skipped due to no new completions
    uint64_t dummy_drain_iters{0};
    uint64_t dispatch_iters{0};
    uint64_t idle_iters{0};
    uint64_t total_iters{0};

    void reset() { *this = SchedulerThreadProfile{}; }
};

struct alignas(64) CoreExecState
{
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
    volatile uint32_t *cond_ptr;            // offset 40: precomputed pointer to COND register
    // --- Cold fields (init/diagnostics only, never in hot path) ---
    int32_t worker_id;          // offset 48: index in runtime.workers[]
    uint32_t physical_core_id;  // offset 52: hardware physical core ID
    CoreType core_type;         // offset 56: AIC or AIV (enum class : int32_t)
    uint8_t pad2_[4];           // offset 60: pad to 64 bytes
};
static_assert(sizeof(CoreExecState) == 64, "CoreExecState must occupy exactly one cache line");

class alignas(64) CoreTracker
{
public:
    static inline int32_t MAX_CORE_PER_THREAD = 63;
    static constexpr int32_t MAX_CLUSTERS = 63 / 3;

public:
    CoreTracker() = default;

    class BitStates
    {
    public:
        BitStates() = default;

        explicit BitStates(uint64_t states) :
            states_(states)
        {}
        void init()
        {
            states_ = 0;
        }

        BitStates operator~() const
        {
            return BitStates(~states_);
        }
        BitStates operator&(const BitStates &other) const
        {
            return BitStates(states_ & other.states_);
        }
        BitStates operator|(const BitStates &other) const
        {
            return BitStates(states_ | other.states_);
        }
        BitStates operator^(const BitStates &other) const
        {
            return BitStates(states_ ^ other.states_);
        }
        BitStates operator>>(int32_t offset) const
        {
            return BitStates(states_ >> offset);
        }
        BitStates operator<<(int32_t offset) const
        {
            return BitStates(states_ << offset);
        }
        void operator&=(const BitStates &other)
        {
            states_ &= other.states_;
        }
        void operator|=(const BitStates &other)
        {
            states_ |= other.states_;
        }
        void operator^=(const BitStates &other)
        {
            states_ ^= other.states_;
        }

        bool has_value() const
        {
            return states_ > 0;
        }
        int32_t count() const
        {
            return __builtin_popcountll(states_);
        }

        // Extract the lowest set bit from mask, clear it, and return its position.
        // Returns -1 if mask is empty.
        int32_t pop_first()
        {
            if (states_ == 0) return -1;
            int32_t pos = __builtin_ctzll(states_);
            states_ &= states_ - 1;
            return pos;
        }

    private:
        uint64_t states_{0};
    };

public:
    void init(int32_t cluster_count)
    {
        cluster_count_ = cluster_count;
        aic_mask_.init();
        aiv_mask_.init();
        pending_occupied_.init();
        for (int32_t i = 0; i < cluster_count; i++)
        {
            aic_mask_ |= BitStates(1ULL << (i * 3));
            aiv_mask_ |= BitStates(6ULL << (i * 3));
        }
        core_states_ = aic_mask_ | aiv_mask_;
    }

    void set_cluster(int32_t cluster_idx, int32_t aic_wid, int32_t aiv0_wid, int32_t aiv1_wid)
    {
        core_id_map_[cluster_idx * 3] = aic_wid;
        core_id_map_[cluster_idx * 3 + 1] = aiv0_wid;
        core_id_map_[cluster_idx * 3 + 2] = aiv1_wid;
    }

    int32_t get_cluster_count() const
    {
        return cluster_count_;
    }

    // --- Running core queries ---

    template <CoreType CT>
    bool has_running_cores() const
    {
        if constexpr (CT == CoreType::AIC) return ((~core_states_) & aic_mask_).has_value();
        else return ((~core_states_) & aiv_mask_).has_value();
    }

    bool has_any_running_cores() const
    {
        return ((~core_states_) & (aic_mask_ | aiv_mask_)).has_value();
    }

    template <CoreType CT>
    int32_t get_running_count() const
    {
        if constexpr (CT == CoreType::AIC) return ((~core_states_) & aic_mask_).count();
        else return ((~core_states_) & aiv_mask_).count();
    }

    // Return an opaque bitmask for iterating running cores of a given type.
    // Use pop_first() to extract core bit offsets one at a time.
    template <CoreType CT>
    BitStates get_running_cores() const
    {
        if constexpr (CT == CoreType::AIC) return (~core_states_) & aic_mask_;
        else return (~core_states_) & aiv_mask_;
    }

    BitStates get_all_running_cores() const
    {
        return (~core_states_) & (aic_mask_ | aiv_mask_);
    }

    // --- Cluster matching ---

    BitStates get_valid_cluster_offset_states(PTO2ResourceShape shape) const
    {
        switch (shape)
        {
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

    int32_t get_aic_core_id(int32_t cluster_offset) const
    {
        return core_id_map_[cluster_offset];
    }
    int32_t get_aiv0_core_id(int32_t cluster_offset) const
    {
        return core_id_map_[cluster_offset + 1];
    }
    int32_t get_aiv1_core_id(int32_t cluster_offset) const
    {
        return core_id_map_[cluster_offset + 2];
    }

    int32_t get_aic_core_offset(int32_t cluster_offset) const
    {
        return cluster_offset;
    }
    int32_t get_aiv0_core_offset(int32_t cluster_offset) const
    {
        return cluster_offset + 1;
    }
    int32_t get_aiv1_core_offset(int32_t cluster_offset) const
    {
        return cluster_offset + 2;
    }

    bool is_aic_core_idle(int32_t cluster_offset) const
    {
        return ((core_states_ >> cluster_offset) & BitStates(1ULL)).has_value();
    }
    bool is_aiv0_core_idle(int32_t cluster_offset) const
    {
        return ((core_states_ >> (cluster_offset + 1)) & BitStates(1ULL)).has_value();
    }
    bool is_aiv1_core_idle(int32_t cluster_offset) const
    {
        return ((core_states_ >> (cluster_offset + 2)) & BitStates(1ULL)).has_value();
    }

    // --- State mutation ---

    // Toggle bit at the given bit offset (running <-> idle)
    void change_core_state(int32_t bit_offset)
    {
        core_states_ ^= BitStates(1ULL << bit_offset);
    }

    void set_pending_occupied(int32_t bit_offset)
    {
        pending_occupied_ |= BitStates(1ULL << bit_offset);
    }
    void clear_pending_occupied(int32_t bit_offset)
    {
        pending_occupied_ ^= (pending_occupied_ & BitStates(1ULL << bit_offset));
    }

    // --- Two-phase dispatch queries ---

    BitStates get_idle_core_offset_states(PTO2ResourceShape shape) const
    {
        if (shape == PTO2ResourceShape::AIC) return get_valid_cluster_offset_states(shape) & ~(pending_occupied_ & aic_mask_);
        if (shape == PTO2ResourceShape::AIV) return core_states_ & aiv_mask_;
        return get_valid_cluster_offset_states(shape);  // MIX: cluster-level
    }

    BitStates get_pending_core_offset_states(PTO2ResourceShape shape) const
    {
        if (shape == PTO2ResourceShape::MIX)
        {
            // Any core without a pending payload can accept a dispatch (idle or running).
            BitStates available = ~pending_occupied_;
            BitStates mix_available = (available & aic_mask_) & ((available >> 1) & aic_mask_) & ((available >> 2) & aic_mask_);
            // Exclude fully-idle clusters (handled by IDLE phase) to prevent double-dispatch.
            BitStates running = ~core_states_;
            BitStates cluster_has_running = (running & aic_mask_) | ((running >> 1) & aic_mask_) | ((running >> 2) & aic_mask_);
            return mix_available & cluster_has_running;
        }
        if (shape == PTO2ResourceShape::AIC) return (~core_states_) & aic_mask_ & ~(pending_occupied_ & aic_mask_);
        // AIV
        return (~core_states_) & aiv_mask_ & ~pending_occupied_;
    }

    // --- Two-phase dispatch unified query ---

    enum class DispatchPhase : uint8_t
    {
        IDLE,
        PENDING
    };

    BitStates get_dispatchable_cores(PTO2ResourceShape shape, DispatchPhase phase) const
    {
        return (phase == DispatchPhase::IDLE) ? get_idle_core_offset_states(shape) : get_pending_core_offset_states(shape);
    }

    // --- Bit offset <-> worker_id mapping ---

    int32_t get_core_id_by_offset(int32_t offset) const
    {
        return core_id_map_[offset];
    }

    const int32_t *core_ids() const
    {
        return core_id_map_;
    }
    int32_t core_num() const
    {
        return cluster_count_ * 3;
    }

private:
    int32_t cluster_count_;
    BitStates aic_mask_;
    BitStates aiv_mask_;
    BitStates core_states_;
    BitStates pending_occupied_;
    int32_t core_id_map_[63];  // bit_position -> worker_id, max 21 clusters * 3
};

struct SlotTransition
{
    bool running_done = false;   // running task completed
    bool pending_done = false;   // pending task completed
    bool running_freed = false;  // running slot data should be released
    bool pending_freed = false;  // pending_occupied can be cleared
    bool matched = false;        // some case was hit (otherwise skip apply)
};

// When sync_start_pending != 0, all scheduler threads skip dispatch
// (only process completions) until the drain worker finishes launching all blocks.
struct alignas(64) SyncStartDrainState
{
    std::atomic<int32_t> sync_start_pending{0};              // 0=normal; -1=initializing; >0=active (value=block_num)
    std::atomic<int32_t> drain_worker_elected{0};            // 0=none; >0: elected thread's (thread_idx+1)
    std::atomic<uint32_t> drain_ack_mask{0};                 // bit per thread; all-set = all threads reached ack barrier
    std::atomic<PTO2TaskSlotState *> pending_task{nullptr};  // held task (not re-queued)
    int32_t _pad[10];
};
static_assert(sizeof(SyncStartDrainState) == 64);

#endif  // SCHEDULER_TYPES_H
