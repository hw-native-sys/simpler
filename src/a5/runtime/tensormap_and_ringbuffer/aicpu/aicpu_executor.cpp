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
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/orch_so_file.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/performance_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/perf_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"

#if PTO2_PROFILING
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

// Device orchestration function signature (loaded via dlopen).
// The executor binds the current thread's PTO2Runtime into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(const ChipStorageTaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime *rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipStorageTaskArgs &orch_args);

// From orchestration/common.cpp linked into this DSO — updates g_pto2_current_runtime here (distinct from
// pto2_framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void pto2_framework_bind_runtime(PTO2Runtime *rt);

constexpr int32_t MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int32_t MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

constexpr int32_t MAX_IDLE_ITERATIONS = 800000;       // ~20s idle then scheduler gives up (avoid long hang)
constexpr int32_t STALL_LOG_INTERVAL = 50000;         // DEV_ALWAYS every N idle iters to debug hang
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check orchestrator error every N idle iters
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;
constexpr int32_t PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int32_t PROGRESS_LOG_INTERVAL = 250;      // log every N completions after threshold

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    void *sm = runtime->get_pto2_gm_sm_ptr();
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return pto2_runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

// Per-core dispatch payload storage: dual-buffer to allow pipelining.
// buf_idx = reg_task_id & 1; adjacent dispatches use different slots,
// so AICPU can write pending payload while AICore reads running payload.
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER][2];

// Per-core state: one cache line per core to eliminate false sharing
// and co-locate all hot-path fields for minimal cache misses.
// Dual-slot layout: running (currently executing) + pending (pre-loaded, awaiting hardware pickup).
struct alignas(64) CoreExecState {
    // --- Hot fields (completion + dispatch, every iteration) ---
    uint64_t reg_addr;                      // offset  0: register address (set once in handshake)
    PTO2TaskSlotState *running_slot_state;  // offset  8: slot state for running task (nullptr = empty)
    PTO2TaskSlotState *pending_slot_state;  // offset 16: slot state for pending task (nullptr = empty)
    int32_t running_reg_task_id;            // offset 24: register task ID (AICPU_TASK_INVALID = idle)
    int32_t pending_reg_task_id;            // offset 28: pending register task ID (AICPU_TASK_INVALID = none)
    uint32_t dispatch_seq;                  // offset 32: monotonic dispatch counter
    PTO2SubtaskSlot running_subslot;        // offset 36: which subtask slot is running
    PTO2SubtaskSlot pending_subslot;        // offset 37: which subtask slot is pending
    uint8_t pad0_[2];                       // offset 38: alignment padding
#if PTO2_PROFILING
    // --- Profiling fields (dispatch path, compile-time gated) ---
    uint64_t running_dispatch_timestamp;  // offset 40: AICPU dispatch timestamp for running task
    uint64_t pending_dispatch_timestamp;  // offset 48: AICPU dispatch timestamp for pending task
    uint8_t pad1_[8];                     // offset 56: pad to 64 bytes
#else
    // --- Cold fields (init/diagnostics only, never in hot path) ---
    int32_t worker_id;          // offset 40: index in runtime.workers[]
    uint32_t physical_core_id;  // offset 44: hardware physical core ID
    CoreType core_type;         // offset 48: AIC or AIV
    uint8_t pad2_[12];          // offset 52: pad to 64 bytes
#endif
};
static_assert(sizeof(CoreExecState) == 64, "CoreExecState must occupy exactly one cache line");

// core_states_ encodes per-cluster core idle/running in 3 bits per cluster:
//   bit i*3   = AIC of cluster i   (1 = idle, 0 = running)
//   bit i*3+1 = AIV0 of cluster i
//   bit i*3+2 = AIV1 of cluster i
// Max 21 clusters per tracker (63 bits in uint64_t).
class alignas(64) CoreTracker {
public:
    static inline int32_t MAX_CORE_PER_THREAD = 63;
    static constexpr int32_t MAX_CLUSTERS = 63 / 3;

public:
    CoreTracker() = default;

    class BitStates {
    public:  // NOLINT(whitespace/indent)
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

        // Extract the lowest set bit from mask, clear it, and return its position.
        // Returns -1 if mask is empty.
        int32_t pop_first() {
            if (states_ == 0) return -1;
            int32_t pos = __builtin_ctzll(states_);
            states_ &= states_ - 1;
            return pos;
        }

    private:  // NOLINT(whitespace/indent)
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

    // --- Cluster matching ---

    BitStates get_valid_cluster_offset_states(PTO2ResourceShape shape) const {
        switch (shape) {
        case PTO2ResourceShape::AIC:
            return core_states_ & aic_mask_;
        case PTO2ResourceShape::AIV:
            return ((core_states_ >> 1) | (core_states_ >> 2)) & aic_mask_;
        case PTO2ResourceShape::MIX:
            return (core_states_ >> 1) & (core_states_ >> 2) & core_states_ & aic_mask_;
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

    // Count total idle AIV cores (AIV0 + AIV1) across all clusters.
    // Unlike get_valid_cluster_offset_states(AIV).count() which counts clusters with
    // at least one idle AIV, this counts individual idle cores — a cluster with both
    // AIV0 and AIV1 idle contributes 2, not 1.
    int32_t count_idle_aiv_cores() const {
        return ((core_states_ >> 1) & aic_mask_).count() + ((core_states_ >> 2) & aic_mask_).count();
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

    // Idle dispatch: clusters where core is idle AND pending slot is free (both slots empty).
    BitStates get_idle_cluster_offset_states(PTO2ResourceShape shape) const {
        BitStates idle = get_valid_cluster_offset_states(shape);
        // Mask out clusters whose AIC bit has pending_occupied set
        return idle & ~(pending_occupied_ & aic_mask_);
    }

    // Pending dispatch: clusters where core is running AND pending slot is free.
    // For AIC: core running (bit=0) but pending_occupied not set.
    BitStates get_pending_only_cluster_offset_states(PTO2ResourceShape shape) const {
        if (shape != PTO2ResourceShape::AIC) return BitStates(0ULL);  // Only AIC supports pending dispatch
        BitStates running_aic = (~core_states_) & aic_mask_;
        return running_aic & ~(pending_occupied_ & aic_mask_);
    }

    // --- Bit offset <-> worker_id mapping ---

    int32_t get_core_id_by_offset(int32_t offset) const { return core_id_map_[offset]; }

private:
    int32_t cluster_count_;
    BitStates aic_mask_;
    BitStates aiv_mask_;
    BitStates core_states_;
    BitStates pending_occupied_;
    int32_t core_id_map_[63];  // bit_position -> worker_id, max 21 clusters * 3
};

struct AicpuExecutor {
    int32_t sched_thread_num_;
    int32_t active_sched_threads_{0};  // Threads currently in dispatch loop (initially sched_thread_num_, becomes
                                       // thread_num_ after orch→sched transition)
    bool orch_to_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t thread_num_{0};
    int32_t cores_total_num_{0};
    int32_t thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int32_t core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int32_t core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered worker_id lists for core assignment (init-only)
    int32_t aic_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aiv_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    CoreTracker core_trackers_[MAX_AICPU_THREADS];

    // ===== sync_start drain coordination =====

    // When sync_start_pending != 0, all scheduler threads skip Phase 2 dispatch
    // (only process completions) until the drain worker finishes launching all blocks.
    struct alignas(64) SyncStartDrainState {
        std::atomic<int32_t> sync_start_pending{0};    // 0=normal; -1=initializing; >0=active (value=block_num)
        std::atomic<int32_t> drain_worker_elected{0};  // 0=none; >0: elected thread's (thread_idx+1)
        std::atomic<uint32_t> drain_ack_mask{0};       // bit per thread; all-set = all threads finished dispatch
        PTO2TaskSlotState *pending_task{nullptr};      // held task (not re-queued)
        int32_t _pad[10];
    };
    static_assert(sizeof(SyncStartDrainState) == 64);
    SyncStartDrainState drain_state_;

    // ===== Task queue state (managed by scheduler ready queues) =====

    // Task execution tracking
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    std::atomic<int32_t> finished_count_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> runtime_init_ready_{false};
    std::atomic<bool> pto2_init_complete_{false};  // init block finished; others wait for this

    // ===== Dynamic core transition state =====
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};
    std::atomic<bool> completed_{false};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void *orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // Shared orchestration function pointer (loaded by first orch thread, used by all)
    DeviceOrchestrationFunc orch_func_{nullptr};
    DeviceOrchestrationBindRuntimeFunc orch_bind_runtime_{nullptr};
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    uint64_t *func_id_to_addr_;
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t handshake_all_cores(Runtime *runtime);
    bool assign_cores_to_threads();
    void reassign_cores_for_all_threads();
    int32_t resolve_and_dispatch_pto2(Runtime *runtime, int32_t thread_idx);
    int32_t shutdown_aicore(Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    void emergency_shutdown(Runtime *runtime);
    void diagnose_stuck_state(
        Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num, Handshake *hank
    );

    // --- Dual-slot state machine helpers ---

    // SlotTransition: pure event signals from a single register poll.
    // true = event occurred, false = no-op (maintain current state).
    struct SlotTransition {
        bool running_done = false;   // running task completed
        bool pending_done = false;   // pending task completed
        bool running_freed = false;  // running slot data should be released
        bool pending_freed = false;  // pending_occupied can be cleared
        bool matched = false;        // some case was hit (otherwise skip apply)
    };

    // Pure function: read register result → SlotTransition (no side effects).
    static SlotTransition
    decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id) {
        SlotTransition t;
        if (pending_id != AICPU_TASK_INVALID && reg_task_id == pending_id) {
            t.matched = true;
            t.running_done = true;  // Serial execution: pending event implies running done
            t.running_freed = true;
            t.pending_freed = true;
            if (reg_state == TASK_FIN_STATE) {
                t.pending_done = true;  // Case 1: pending FIN
            }
            // else: Case 2: pending ACK (pending_done stays false)
        } else if (reg_task_id == running_id) {
            if (reg_state == TASK_FIN_STATE) {
                if (pending_id == AICPU_TASK_INVALID) {
                    // Case 3.2: running FIN, no pending → core goes idle
                    t.matched = true;
                    t.running_done = true;
                    t.running_freed = true;
                }
                // Case 3.1: running FIN, pending exists → skip (transient state).
                // Case 1/2 (pending ACK/FIN) will complete running implicitly via running_done=true.
            } else {
                // Case 4: running ACK — only pending_freed (slot now hardware-latched)
                t.matched = true;
                t.pending_freed = true;
            }
        }
        return t;
    }

    // Complete one slot's task: subtask counting, mixed completion, deferred release, profiling.
    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, PTO2SubtaskSlot subslot, int32_t thread_idx,
        int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
        PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs, CoreType ct
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_complete_count, uint64_t dispatch_ts
#endif
#if PTO2_SCHED_PROFILING
        ,
        uint64_t &notify_edges_total, int32_t &notify_max_degree, uint64_t &notify_tasks_enqueued,
        uint64_t &fanin_edges_total, int32_t &fanin_max_degree, uint64_t &sched_complete_perf_cycle
#endif
    ) {
#if !PTO2_PROFILING
        (void)hank;  // NOLINT(readability/casting)
        (void)ct;    // NOLINT(readability/casting)
#endif
        bool mixed_complete = rt->scheduler.on_subtask_complete(slot_state);
        if (mixed_complete) {
#if PTO2_SCHED_PROFILING
            PTO2CompletionStats cstats = rt->scheduler.on_mixed_task_complete(slot_state, thread_idx, local_bufs);
            notify_edges_total += cstats.fanout_edges;
            if (cstats.fanout_edges > notify_max_degree) notify_max_degree = cstats.fanout_edges;
            notify_tasks_enqueued += cstats.tasks_enqueued;
            phase_complete_count++;
#else
            rt->scheduler.on_mixed_task_complete(slot_state, local_bufs);
#if PTO2_PROFILING
            phase_complete_count++;
#endif
#endif
            if (deferred_release_count < 256) {
                deferred_release_slot_states[deferred_release_count++] = &slot_state;
            } else {
                DEV_ALWAYS("Thread %d: release", thread_idx);
                while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                    int32_t fe = rt->scheduler.on_task_release(
                        *deferred_release_slot_states[--deferred_release_count], thread_idx
                    );
#else
                    int32_t fe = rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                    (void)fe;  // NOLINT(readability/casting)
#if PTO2_SCHED_PROFILING
                    fanin_edges_total += fe;
                    if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
                }
                deferred_release_slot_states[deferred_release_count++] = &slot_state;
            }
            completed_this_turn++;
        }

#if PTO2_PROFILING
        if (profiling_enabled) {
#if PTO2_SCHED_PROFILING
            uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif
            Handshake *h = &hank[core_id];
            uint64_t finish_ts = get_sys_cnt_aicpu();
            PerfBuffer *perf_buf = reinterpret_cast<PerfBuffer *>(h->perf_records_addr);

            uint64_t fanout_arr[RUNTIME_MAX_FANOUT];
            int32_t fanout_n = 0;
            PTO2DepListEntry *cur = slot_state.fanout_head;
            while (cur != nullptr && fanout_n < RUNTIME_MAX_FANOUT) {
                fanout_arr[fanout_n++] = cur->slot_state->task->task_id.raw;
                cur = cur->next;
            }

            int32_t perf_slot_idx = static_cast<int32_t>(subslot);
            if (perf_aicpu_complete_record(
                    perf_buf, static_cast<uint32_t>(expected_reg_task_id), slot_state.task->task_id.raw,
                    slot_state.task->kernel_id[perf_slot_idx], ct, dispatch_ts, finish_ts, fanout_arr, fanout_n
                ) != 0) {
                DEV_ERROR(
                    "Core %d: perf_aicpu_complete_record failed for task 0x%" PRIx64, core_id,
                    static_cast<uint64_t>(slot_state.task->task_id.raw)
                );
            }
#if PTO2_SCHED_PROFILING
            sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
        }
#endif
    }

    // Promote pending slot data to running slot. Clears pending fields.
    static void promote_pending_to_running(CoreExecState &core) {
        core.running_slot_state = core.pending_slot_state;
        core.running_reg_task_id = core.pending_reg_task_id;
        core.running_subslot = core.pending_subslot;
#if PTO2_PROFILING
        core.running_dispatch_timestamp = core.pending_dispatch_timestamp;
#endif
        core.pending_slot_state = nullptr;
        core.pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Clear running slot (core becomes idle).
    static void clear_running_slot(CoreExecState &core) {
        core.running_slot_state = nullptr;
        core.running_reg_task_id = AICPU_TASK_INVALID;
    }

    template <CoreType CT>
    void check_running_cores_for_completion(
        int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
        bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
        ,
        uint64_t &complete_probe_count, uint64_t &complete_hit_count, uint64_t &notify_edges_total,
        int32_t &notify_max_degree, uint64_t &notify_tasks_enqueued, uint64_t &fanin_edges_total,
        int32_t &fanin_max_degree, uint64_t &sched_complete_perf_cycle
#endif
    ) {
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto running_core_states = tracker.get_running_cores<CT>();
        while (running_core_states.has_value()) {
            int32_t bit_pos = running_core_states.pop_first();
            int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
            CoreExecState &core = core_exec_states_[core_id];

            // --- Judgment phase: read register, derive transition ---
            uint64_t reg_val = read_reg(core.reg_addr, RegId::COND);
            int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
            int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

#if PTO2_SCHED_PROFILING
            if (profiling_enabled) {
                complete_probe_count++;
            }
#endif

            SlotTransition t =
                decide_slot_transition(reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id);
            if (!t.matched) continue;

#if PTO2_SCHED_PROFILING
            if (profiling_enabled && (t.running_done || t.pending_done)) {
                complete_hit_count++;
            }
#endif

            // --- Apply phase: execute actions based on transition ---

            // 1. Complete finished tasks (capture pointers before modifying core state)
            if (t.pending_done) {
                complete_slot_task(
                    *core.pending_slot_state, core.pending_reg_task_id, core.pending_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs, CT
#if PTO2_PROFILING
                    ,
                    profiling_enabled, phase_complete_count, core.pending_dispatch_timestamp
#endif
#if PTO2_SCHED_PROFILING
                    ,
                    notify_edges_total, notify_max_degree, notify_tasks_enqueued, fanin_edges_total, fanin_max_degree,
                    sched_complete_perf_cycle
#endif
                );
                cur_thread_completed++;
            }
            if (t.running_done) {
                complete_slot_task(
                    *core.running_slot_state, core.running_reg_task_id, core.running_subslot, thread_idx, core_id, hank,
                    completed_this_turn, deferred_release_slot_states, deferred_release_count, local_bufs, CT
#if PTO2_PROFILING
                    ,
                    profiling_enabled, phase_complete_count, core.running_dispatch_timestamp
#endif
#if PTO2_SCHED_PROFILING
                    ,
                    notify_edges_total, notify_max_degree, notify_tasks_enqueued, fanin_edges_total, fanin_max_degree,
                    sched_complete_perf_cycle
#endif
                );
                cur_thread_completed++;
            }

            // 2. Update slot data
            if (t.running_freed) {
                if (core.pending_slot_state != nullptr && !t.pending_done) {
                    promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
                } else {
                    clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                    if (t.pending_done) {
                        // Case 1: pending FIN observed directly — clear stale pending fields.
                        // Without this, pending_reg_task_id retains a stale value that blocks
                        // clear_pending_occupied (line 657) and permanently degrades pipelining.
                        core.pending_slot_state = nullptr;
                        core.pending_reg_task_id = AICPU_TASK_INVALID;
                    }
                }
            }

            // 3. Update tracker bitmap
            bool is_idle = (core.running_reg_task_id == AICPU_TASK_INVALID);
            if (is_idle) {
                tracker.change_core_state(bit_pos);       // Mark idle
                tracker.clear_pending_occupied(bit_pos);  // Idle safeguard: no payload to protect
            } else if (t.pending_freed && core.pending_reg_task_id == AICPU_TASK_INVALID) {
                // Case 4 (running ACK) or Case 2 (pending ACK): clear pending_occupied only
                // when no pending task is currently held. Otherwise pending slot is occupied
                // by a pre-loaded task and must stay protected.
                tracker.clear_pending_occupied(bit_pos);
            }

            // 4. Progress signal (only when running task completes)
            if (t.running_done) {
                made_progress = true;
            }
        }
    }

    static const char *shape_name(PTO2ResourceShape shape) {
        switch (shape) {
        case PTO2ResourceShape::AIC:
            return "AIC";
        case PTO2ResourceShape::AIV:
            return "AIV";
        case PTO2ResourceShape::MIX:
            return "MIX";
        }
        return "UNKNOWN";
    }

    /**
     * Returns the dispatch probe order for a given scheduler thread.
     * Widest shapes first to avoid consuming cluster resources with narrow tasks.
     * Even/odd threads use different fallback orders (AIC-first vs AIV-first)
     * to reduce contention on the same ready queue across adjacent threads.
     */
    static const PTO2ResourceShape *get_dispatch_order(int32_t thread_idx) {
        // Even threads: AIC-first fallback after widest
        static constexpr PTO2ResourceShape kEvenOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::MIX,
            PTO2ResourceShape::AIC,
            PTO2ResourceShape::AIV,
        };
        // Odd threads: AIV-first fallback after widest
        static constexpr PTO2ResourceShape kOddOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::MIX,
            PTO2ResourceShape::AIV,
            PTO2ResourceShape::AIC,
        };
        return (thread_idx % 2 == 0) ? kEvenOrder : kOddOrder;
    }

    int pop_ready_tasks_batch(
        PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out,
        int max_count
#if PTO2_SCHED_PROFILING
        ,
        uint64_t &pop_hit, uint64_t &pop_miss, uint64_t &local_dispatch_count, uint64_t &sched_dispatch_pop_cycle
#endif
    ) {
        (void)thread_idx;  // NOLINT(readability/casting)
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
        uint64_t t_pop_start = get_sys_cnt_aicpu();
        int count = rt->scheduler.get_ready_tasks_batch(
            shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx],
            local_dispatch_count
        );
        sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
        if (count > 0) {
            pop_hit += count;
        } else {
            pop_miss++;
        }
#else
        int count = rt->scheduler.get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
        return count;
    }

    /**
     * Build per-core dispatch payload: copy tensor pointers and scalars into
     * the per-core args[] array, then populate SPMD local context at the tail.
     *
     * Reads next_block_idx and block_num directly from the task descriptor
     * to populate LocalContext.  The caller is responsible for incrementing
     * next_block_idx AFTER dispatch.
     *
     * GlobalContext (sub_block_id) is NOT written here — it is initialized once
     * at runtime startup by init_global_context().
     */
    void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot) {
        int32_t slot_idx = static_cast<int32_t>(subslot);
        uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
        const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
        dispatch_payload.function_bin_addr = callable->resolved_addr();
        auto &payload = *slot_state.payload;
        int n = 0;
        for (int32_t i = 0; i < payload.tensor_count; i++) {
            dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
        }
        for (int32_t i = 0; i < payload.scalar_count; i++) {
            dispatch_payload.args[n++] = payload.scalars[i];
        }
        // Per-dispatch local context: read block_idx/block_num directly from slot_state.
        dispatch_payload.local_context.s_block_idx = slot_state.next_block_idx;
        dispatch_payload.local_context.s_block_num = slot_state.block_num;
        // Store context pointers at fixed suffix positions in args[]
        // (GlobalContext content is already set by init_global_context, but the
        //  pointer must be written each dispatch since args[] is rebuilt entirely)
        dispatch_payload.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
        dispatch_payload.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
    }

    void dispatch_subtask_to_core(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, bool to_pending
#if PTO2_PROFILING
        ,
        bool profiling_enabled
#endif
    ) {
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto core_id = tracker.get_core_id_by_offset(core_offset);
        CoreExecState &core_exec_state = core_exec_states_[core_id];
        // Per-core monotonic counter for register protocol uniqueness (32-bit).
        // PTO2 task_id encodes (ring_id << 32 | local_id); truncation to uint32 loses ring_id,
        // so tasks from different rings with the same local_id would write identical DATA_MAIN_BASE
        // values. The AICore uses last_reg_val to detect new dispatches and would skip the
        // duplicate, while the stale COND register from the previous task (same local_id) would
        // cause a false-positive completion.
        // PerfRecord.task_id: register token (low 32) until AICPU overwrites with full (ring_id << 32 | local_id).
        core_exec_state.dispatch_seq++;
        uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        // Skip reserved sentinel range [AICORE_EXIT_SIGNAL, 0x7FFFFFFF].
        // The skip distance must be even to preserve reg_task_id & 1 parity for dual-buffer.
        static_assert(
            (TASK_ID_MASK - AICORE_EXIT_SIGNAL + 1) % 2 == 0,
            "Sentinel skip must be even to preserve dual-buffer parity"
        );
        if (reg_task_id >= AICORE_EXIT_SIGNAL) {
            core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
            reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        }

        // Select dual-buffer slot: adjacent dispatches alternate automatically
        uint32_t buf_idx = reg_task_id & 1u;
        PTO2DispatchPayload &payload = s_pto2_payload_per_core[core_id][buf_idx];
        build_payload(payload, slot_state, subslot);

        // to_pending is determined by the caller (idle dispatch = false, pending dispatch = true).
        if (to_pending) {
            core_exec_state.pending_subslot = subslot;
            core_exec_state.pending_slot_state = &slot_state;
            core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
            if (profiling_enabled) {
                core_exec_state.pending_dispatch_timestamp = get_sys_cnt_aicpu();
            }
#endif
        } else {
            core_exec_state.running_subslot = subslot;
            core_exec_state.running_slot_state = &slot_state;
            core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
            if (profiling_enabled) {
                core_exec_state.running_dispatch_timestamp = get_sys_cnt_aicpu();
            }
#endif
            // Mark core as running (was idle)
            tracker.change_core_state(core_offset);
        }

        write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));

        // SET pending_occupied: serves as pre-reservation even on first dispatch
        // (guarantees next dispatch to this core uses pending-slot path until hardware ACKs)
        tracker.set_pending_occupied(core_offset);
    }

    // Dispatch one SPMD block of a MIX task to the cluster at cluster_offset.
    // Reads slot_state.next_block_idx as block_idx; caller increments it afterwards.
    void dispatch_mix_block_to_cluster(
        int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state
#if PTO2_PROFILING
        ,
        bool profiling_enabled
#endif
    ) {
        CoreTracker &tracker = core_trackers_[thread_idx];
        uint8_t core_mask = pto2_core_mask(slot_state.active_mask);
        if (core_mask & PTO2_SUBTASK_MASK_AIC) {
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC, false
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV0) {
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aiv0_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV0, false
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        }
        if (core_mask & PTO2_SUBTASK_MASK_AIV1) {
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aiv1_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV1, false
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        }
    }

    // ===== sync_start drain helpers =====

    // Take ownership of slot_state and signal all threads to enter drain mode.
    // Returns true if this thread won the CAS and owns the drain slot.
    // Returns false if another thread already holds drain; caller must re-push slot_state.
    //
    // Two-phase protocol: CAS 0 → -1 (sentinel) to claim ownership, store task and
    // reset election flag, then release-store block_num.  Other threads acquire-load
    // sync_start_pending; seeing block_num > 0 ensures all relaxed stores are visible.
    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num) {
        int32_t expected = 0;
        if (!drain_state_.sync_start_pending.compare_exchange_strong(
                expected, -1, std::memory_order_relaxed, std::memory_order_relaxed
            )) {
            return false;  // Another thread already holds the drain slot.
        }
        // We own the drain slot.  Store the task and reset election flag before making it visible.
        drain_state_.pending_task = slot_state;
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        // Release store: all stores above are now visible to any thread that
        // acquire-loads sync_start_pending and sees block_num > 0.
        drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
        return true;
    }

    // Dispatch one SPMD block to the cluster at cluster_offset, routing to the correct core(s)
    // based on shape.  For AIV, picks whichever AIV core in the cluster is currently idle.
    // Caller is responsible for incrementing slot_state.next_block_idx after this returns.
    void dispatch_block_to_cluster(
        int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_dispatch_count
#endif
    ) {
        CoreTracker &tracker = core_trackers_[thread_idx];
        if (shape == PTO2ResourceShape::MIX) {
            dispatch_mix_block_to_cluster(
                thread_idx, cluster_offset, slot_state
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        } else if (shape == PTO2ResourceShape::AIC) {
            dispatch_subtask_to_core(
                thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC, false
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        } else {  // AIV
            auto core_offset = tracker.is_aiv0_core_idle(cluster_offset) ?
                                   tracker.get_aiv0_core_offset(cluster_offset) :
                                   tracker.get_aiv1_core_offset(cluster_offset);
            dispatch_subtask_to_core(
                thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, false
#if PTO2_PROFILING
                ,
                profiling_enabled
#endif
            );
        }
#if PTO2_PROFILING
        phase_dispatch_count += __builtin_popcount(pto2_core_mask(slot_state.active_mask));
#endif
    }

    // Count total available resources across all scheduler threads for a given shape.
    int32_t count_global_available(PTO2ResourceShape shape) {
        int32_t total = 0;
        for (int32_t t = 0; t < active_sched_threads_; t++) {
            if (shape == PTO2ResourceShape::AIV) {
                total += core_trackers_[t].count_idle_aiv_cores();
            } else {
                total += core_trackers_[t].get_valid_cluster_offset_states(shape).count();
            }
        }
        return total;
    }

    // Drain worker: dispatch all blocks in one pass across all threads' trackers.
    // Called only when global resources >= block_num, so one pass always suffices.
    // All other threads are spinning — the drain worker has exclusive tracker access.
    void drain_worker_dispatch(
        int32_t block_num
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_dispatch_count
#endif
    ) {
        PTO2TaskSlotState *slot_state = drain_state_.pending_task;
        if (!slot_state) {
            drain_state_.sync_start_pending.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state->active_mask);

        for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++) {
            auto valid = core_trackers_[t].get_valid_cluster_offset_states(shape);
            while (valid.has_value() && slot_state->next_block_idx < block_num) {
                dispatch_block_to_cluster(
                    t, valid.pop_first(), *slot_state, shape
#if PTO2_PROFILING
                    ,
                    profiling_enabled, phase_dispatch_count
#endif
                );
                slot_state->next_block_idx++;
                if (slot_state->next_block_idx < block_num)
                    valid = core_trackers_[t].get_valid_cluster_offset_states(shape);
            }
        }

        // All blocks dispatched — clear drain state.
        // Release fence ensures tracker mutations are visible to threads that
        // acquire-load sync_start_pending == 0 and resume normal operation.
        std::atomic_thread_fence(std::memory_order_release);
        drain_state_.pending_task = nullptr;
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
    }

    // Called by each scheduler thread when drain_state_.sync_start_pending != 0.
    //
    // Three-phase protocol:
    //   1. Ack barrier: all threads signal they've stopped Phase 2 dispatch.
    //      If not all acked yet, return to Phase 1 (completion polling).
    //   2. Resource check: elected thread verifies global idle resources >= block_num.
    //      If insufficient, reset election state and return — all threads resume
    //      Phase 1 to free running cores, then retry next iteration.
    //   3. Dispatch: elected thread dispatches all blocks (one pass, resources guaranteed).
    //      Non-elected threads spin-wait until sync_start_pending == 0.
    //      During dispatch the elected thread has exclusive tracker access.
    void handle_drain_mode(
        int32_t thread_idx
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_dispatch_count
#endif
    ) {
        // Spin until drain is fully initialized (sentinel -1 → block_num > 0).
        int32_t block_num;
        do {
            block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
        } while (block_num < 0);
        if (block_num == 0) return;

        // Phase 1: Ack barrier — signal this thread has stopped Phase 2 dispatch.
        uint32_t all_acked = (1u << active_sched_threads_) - 1;
        drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

        // If not all threads have acked, return to do Phase 1 (completion polling).
        if ((drain_state_.drain_ack_mask.load(std::memory_order_acquire) & all_acked) != all_acked) return;

        // Phase 2: Election — exactly one thread wins the CAS.
        int32_t expected = 0;
        drain_state_.drain_worker_elected.compare_exchange_strong(
            expected, thread_idx + 1, std::memory_order_acquire, std::memory_order_relaxed
        );

        if (drain_state_.drain_worker_elected.load(std::memory_order_relaxed) != thread_idx + 1) {
            // Non-elected: spin-wait for drain completion or resource-insufficient reset.
            while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
                if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
                SPIN_WAIT_HINT();
            }
            return;
        }

        // Elected: check if global resources are sufficient.
        PTO2TaskSlotState *slot_state = drain_state_.pending_task;
        PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state->active_mask);
        int32_t available = count_global_available(shape);

        if (available < block_num) {
            // Insufficient resources — reset election, let all threads do Phase 1.
            drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }

        // Phase 3: Dispatch — all other threads are spinning, exclusive tracker access.
        drain_worker_dispatch(
            block_num
#if PTO2_PROFILING
            ,
            profiling_enabled, phase_dispatch_count
#endif
        );
    }
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

/**
 * Handshake with all cores and discover their types
 * Sets up register addresses for fast dispatch.
 */
int32_t AicpuExecutor::handshake_all_cores(Runtime *runtime) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    cores_total_num_ = runtime->worker_count;

    // Validate cores_total_num_ before using as array index
    if (cores_total_num_ == 0 || cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Invalid cores_total_num %d (expected 1-%d)", cores_total_num_, MAX_CORES_PER_THREAD);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Handshaking with %d cores", cores_total_num_);

    // Step 1: Write per-core payload addresses and send handshake signal
    // OUT_OF_ORDER_STORE_BARRIER() ensures task is globally visible before
    // aicpu_ready=1, so AICore reads the correct payload pointer after waking up.
    for (int32_t i = 0; i < cores_total_num_; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&s_pto2_payload_per_core[i][0]);
        OUT_OF_ORDER_STORE_BARRIER();
        all_handshakes[i].aicpu_ready = 1;
    }
    OUT_OF_ORDER_STORE_BARRIER();

    // Get platform physical cores count for validation
    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: Wait for all cores to respond, collect core type and register addresses
    bool handshake_failed = false;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];

        while (hank->aicore_regs_ready == 0) {}

        uint32_t physical_core_id = hank->physical_core_id;

        // Validate physical_core_id before using as array index
        if (physical_core_id >= max_physical_cores_count) {
            DEV_ERROR(
                "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                max_physical_cores_count
            );
            handshake_failed = true;
            continue;
        }

        // Get register address using physical_core_id
        uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

        // Initialize AICore registers after discovery (first round)
        platform_init_aicore_regs(reg_addr);
        OUT_OF_ORDER_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;

        OUT_OF_ORDER_STORE_BARRIER();

        while (hank->aicore_done == 0) {}

        CoreType type = hank->core_type;

        core_exec_states_[i].reg_addr = reg_addr;
#if !PTO2_PROFILING
        core_exec_states_[i].worker_id = i;
        core_exec_states_[i].physical_core_id = physical_core_id;
        core_exec_states_[i].core_type = type;
#endif

        if (type == CoreType::AIC) {
            aic_worker_ids_[aic_count_++] = i;
            DEV_INFO("Core %d: AIC, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        } else {
            aiv_worker_ids_[aiv_count_++] = i;
            DEV_INFO("Core %d: AIV, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        }
    }

    if (handshake_failed) {
        emergency_shutdown(runtime);
        return -1;
    }

    DEV_INFO("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);
    return 0;
}

/**
 * Assign discovered cores to scheduler threads
 * (Aligned with host_build_graph mechanism)
 */
bool AicpuExecutor::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    active_sched_threads_ = (sched_thread_num_ > 0) ? sched_thread_num_ : thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
    int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
    thread_cores_num_ = max_clusters_per_thread * 3;

    if (thread_cores_num_ > CoreTracker::MAX_CORE_PER_THREAD) {
        DEV_ERROR("Can't assign more then 64 cores in per scheduler");
        return false;
    }

    DEV_INFO(
        "Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)", cluster_count,
        active_sched_threads_, aic_count_, aiv_count_
    );

    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Count clusters per thread first (round-robin may distribute unevenly)
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % active_sched_threads_]++;
    }
    for (int32_t i = 0; i < active_sched_threads_; i++) {
        core_trackers_[i].init(clusters_per_thread[i]);
        core_count_per_thread_[i] = 0;
    }

    // Per-sched-thread running core index used while filling core_assignments_.
    int32_t core_idx[MAX_AICPU_THREADS] = {};
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % active_sched_threads_;
        int32_t &idx = core_idx[t];

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);

        core_assignments_[t][idx++] = aic_wid;
        core_assignments_[t][idx++] = aiv0_wid;
        core_assignments_[t][idx++] = aiv1_wid;

        DEV_INFO("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < thread_num_; t++) {
        core_count_per_thread_[t] = core_idx[t];
        DEV_INFO("Thread %d: total %d cores (%d clusters)", t, core_idx[t], core_trackers_[t].get_cluster_count());
    }

    return true;
}

/**
 * Reassign all cores evenly across all threads (schedulers + orchestrators).
 * Called by the last orchestrator thread when orchestration completes.
 * Writes into new_core_assignments_ / new_core_count_per_thread_.
 */
void AicpuExecutor::reassign_cores_for_all_threads() {
    DEV_INFO("Reassigning cores (cluster-aligned) for %d threads: %d AIC, %d AIV", thread_num_, aic_count_, aiv_count_);

    // Collect running worker_ids from all current trackers
    bool running_cores[MAX_CORES_PER_THREAD] = {};
    for (int32_t i = 0; i < thread_num_; i++) {
        auto all_running = core_trackers_[i].get_all_running_cores();
        int32_t bp;
        while ((bp = all_running.pop_first()) >= 0) {
            running_cores[core_trackers_[i].get_core_id_by_offset(bp)] = true;
        }
    }

    // Count clusters per thread (round-robin across all threads)
    int32_t cluster_count = aic_count_;
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % thread_num_]++;
    }

    // Re-init all trackers and reset core counts
    for (int32_t i = 0; i < thread_num_; i++) {
        core_trackers_[i].init(clusters_per_thread[i]);
        core_count_per_thread_[i] = 0;
    }

    // Assign clusters round-robin and restore running state
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % thread_num_;

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        int32_t cl_idx = cluster_idx_per_thread[t]++;
        core_trackers_[t].set_cluster(cl_idx, aic_wid, aiv0_wid, aiv1_wid);

        // init() marks all idle; toggle cores that were running and restore pending_occupied
        if (running_cores[aic_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3);
            core_trackers_[t].set_pending_occupied(cl_idx * 3);
        }
        if (running_cores[aiv0_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3 + 1);
            core_trackers_[t].set_pending_occupied(cl_idx * 3 + 1);
        }
        if (running_cores[aiv1_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3 + 2);
            core_trackers_[t].set_pending_occupied(cl_idx * 3 + 2);
        }

        core_assignments_[t][core_count_per_thread_[t]++] = aic_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv0_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv1_wid;
    }

    // Log final distribution
    DEV_INFO("Core reassignment complete:");
    for (int32_t t = 0; t < thread_num_; t++) {
        int32_t aic_running = core_trackers_[t].get_running_count<CoreType::AIC>();
        int32_t aiv_running = core_trackers_[t].get_running_count<CoreType::AIV>();
        DEV_INFO(
            "  Thread %d: %d cores, %d clusters (AIC running=%d, AIV running=%d)", t, core_count_per_thread_[t],
            core_trackers_[t].get_cluster_count(), aic_running, aiv_running
        );
    }
    active_sched_threads_ = thread_num_;
}

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    func_id_to_addr_ = runtime->func_id_to_addr_;

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    sched_thread_num_ = thread_num_ - 1;
    orch_to_sched_ = runtime->orch_to_sched;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Zero all per-core execution state before handshake
    memset(core_exec_states_, 0, sizeof(core_exec_states_));

    // Use handshake mechanism to discover cores (aligned with host_build_graph)
    int32_t rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("handshake_all_cores failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Dynamically assign cores to threads
    if (!assign_cores_to_threads()) {
        return -1;
    }

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Initialize runtime execution state
    // Task count comes from PTO2 shared memory
    if (runtime->get_pto2_gm_sm_ptr()) {
        auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_pto2_gm_sm_ptr());
        int32_t pto2_count = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            pto2_count += header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
        }
        total_tasks_ = pto2_count > 0 ? pto2_count : 0;
    } else {
        total_tasks_ = 0;
    }
    completed_tasks_.store(0, std::memory_order_release);
    // Host orchestration: graph already built, no wait needed. Device orch: Thread 3 will set this.
    bool orch_on_host = runtime->get_orch_built_on_host();
    DEV_INFO("Init: orch_built_on_host=%d", orch_on_host ? 1 : 0);
    orchestrator_done_ = orch_on_host;

    // Initial ready tasks will be populated via scheduler ready queues

    // Clear per-core dispatch payloads
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));

    // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
    // This is done once at startup and never modified afterwards.
    for (int32_t t = 0; t < sched_thread_num_; t++) {
        CoreTracker &tracker = core_trackers_[t];
        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
            int32_t cluster_offset = c * 3;  // Each cluster = 1 AIC + 2 AIV
            auto aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
            auto aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
            s_pto2_payload_per_core[aiv0_id][0].global_context.sub_block_id = 0;
            s_pto2_payload_per_core[aiv0_id][1].global_context.sub_block_id = 0;
            s_pto2_payload_per_core[aiv1_id][0].global_context.sub_block_id = 1;
            s_pto2_payload_per_core[aiv1_id][1].global_context.sub_block_id = 1;
        }
    }

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::shutdown_aicore(
    Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num
) {
    (void)runtime;  // NOLINT(readability/casting)
    if (core_num == 0) return 0;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        if (reg_addr != 0) {
            platform_deinit_aicore_regs(reg_addr);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

int32_t AicpuExecutor::resolve_and_dispatch_pto2(Runtime *runtime, int32_t thread_idx) {
    int32_t &core_num = core_count_per_thread_[thread_idx];
    CoreTracker &tracker = core_trackers_[thread_idx];
    DEV_INFO("Thread %d: resolve_and_dispatch_pto2 entry", thread_idx);

    void *sm_base = runtime->get_pto2_gm_sm_ptr();
    if (!sm_base) {
        DEV_ERROR("PTO2 dispatch: sm_base is null");
        return -1;
    }
    DEV_INFO("Thread %d: sm_base=%p", thread_idx, sm_base);

    PTO2SharedMemoryHeader *header = static_cast<PTO2SharedMemoryHeader *>(sm_base);
    DEV_INFO(
        "Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu", thread_idx, static_cast<void *>(header),
        static_cast<uint64_t>(header->rings[0].task_descriptors_offset),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    Handshake *hank = static_cast<Handshake *>(runtime->workers);
    DEV_INFO(
        "Thread %d: hank=%p, window_size=%lu", thread_idx, static_cast<void *>(hank),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);

#if PTO2_PROFILING
        // Assign perf buffers to cores early so profiling captures all tasks
        // (total_tasks written to header later when orchestrator completes)
        if (runtime->enable_profiling) {
            perf_aicpu_init_profiling(runtime);
            // Initialize phase profiling for scheduler threads + orchestrator threads
            perf_aicpu_init_phase_profiling(runtime, sched_thread_num_);
            perf_aicpu_set_orch_thread_idx(sched_thread_num_);
        }
#endif

        DEV_INFO("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    DEV_INFO("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_num);
    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if PTO2_PROFILING
    bool profiling_enabled = runtime->enable_profiling;
#endif

    // Scheduler profiling counters
#if PTO2_PROFILING
    uint64_t sched_scan_cycle = 0;
    uint64_t sched_complete_cycle = 0;
    uint64_t sched_dispatch_cycle = 0;
    uint64_t sched_wiring_cycle = 0;
    uint64_t sched_idle_cycle = 0;
    uint64_t sched_loop_count = 0;
    uint32_t phase_complete_count = 0;
    uint32_t phase_dispatch_count = 0;
#if PTO2_SCHED_PROFILING
    uint32_t phase_wiring_count = 0;
    uint64_t complete_probe_count = 0;
    uint64_t complete_hit_count = 0;
    uint64_t notify_edges_total = 0;
    int32_t notify_max_degree = 0;
    uint64_t notify_tasks_enqueued = 0;
    uint64_t fanin_edges_total = 0;
    int32_t fanin_max_degree = 0;
    uint64_t pop_hit = 0;
    uint64_t pop_miss = 0;
    uint64_t local_dispatch_count = 0;
    uint64_t local_overflow_count = 0;
    uint64_t sched_complete_perf_cycle = 0;
    uint64_t sched_dispatch_pop_cycle = 0;
    uint64_t sched_dispatch_setup_cycle = 0;
#endif
#endif

    // Local-first dispatch buffers (stack-allocated, one per CoreType per scheduling thread).
    // Initialized once; must be empty at the start of each iteration.
    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);
    }
    PTO2TaskSlotState *deferred_release_slot_states[256];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

#if PTO2_PROFILING
    uint64_t sched_start_ts = get_sys_cnt_aicpu();
#endif

    while (true) {
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (!tracker.has_any_running_cores()) {
            bool orch_done = orchestrator_done_;
            if (orch_done) {
                // Check for orchestrator fatal error — exit immediately
                int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
                if (orch_err != PTO2_ERROR_NONE) {
                    DEV_ERROR(
                        "Thread %d: Fatal error (code=%d), sending EXIT_SIGNAL to all cores. "
                        "completed_tasks=%d, total_tasks=%d",
                        thread_idx, orch_err, completed_tasks_.load(std::memory_order_relaxed), total_tasks_
                    );
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }

                // Normal exit: all tasks complete
                task_count = total_tasks_;
                if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
                    completed_.store(true, std::memory_order_release);
                    DEV_INFO(
                        "Thread %d: PTO2 completed tasks %d/%d", thread_idx,
                        completed_tasks_.load(std::memory_order_relaxed), task_count
                    );
                    break;
                }
            }
        }

        // Check for core transition request (execute once per thread)
        if (!cores_released && orch_to_sched_ && transition_requested_.load(std::memory_order_acquire)) {
            if (!reassigned_.load(std::memory_order_acquire)) {
                wait_reassign_.fetch_add(1, std::memory_order_release);
                while (!reassigned_.load(std::memory_order_acquire)) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (completed_.load(std::memory_order_acquire)) {
                    break;
                }
            }
            cores_released = true;
        }

#if PTO2_PROFILING
        CYCLE_COUNT_LAP(sched_idle_cycle);
#endif

        // Process completed and dispatch FIRST to minimize Sched (dispatch→finish) latency.
        // Sched time = finish_ts - dispatch_ts; recording finish_ts here at loop start reduces
        // tail overhead (time from AICore done to AICPU recording finish).

        // Phase 1: Check running cores for completion, process and move to idle
        int32_t completed_this_turn = 0;

        // Check AIC running cores
        bool try_completed = false;
        if (tracker.has_running_cores<CoreType::AIC>()) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIC>(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled, phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count, complete_hit_count, notify_edges_total, notify_max_degree, notify_tasks_enqueued,
                fanin_edges_total, fanin_max_degree, sched_complete_perf_cycle
#endif
            );
        }

        // Check AIV running cores
        if (tracker.has_running_cores<CoreType::AIV>()) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIV>(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled, phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count, complete_hit_count, notify_edges_total, notify_max_degree, notify_tasks_enqueued,
                fanin_edges_total, fanin_max_degree, sched_complete_perf_cycle
#endif
            );
        }
        if (completed_this_turn > 0) {
#if PTO2_SCHED_PROFILING
            rt->scheduler.tasks_completed.fetch_add(completed_this_turn, std::memory_order_relaxed);
#endif
            int32_t prev = completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            int32_t new_total = prev + completed_this_turn;
            last_progress_count = new_total;
            if (thread_idx == 0 && task_count > 0) {
                if (new_total <= PROGRESS_VERBOSE_THRESHOLD ||
                    new_total / PROGRESS_LOG_INTERVAL != prev / PROGRESS_LOG_INTERVAL || new_total >= task_count) {
                    DEV_ALWAYS(
                        "PTO2 progress: completed=%d total=%d (%.1f%%)", new_total, task_count,
                        100.0 * new_total / task_count
                    );
                }
            }
        }

#if PTO2_PROFILING
        if (!try_completed) {
            CYCLE_COUNT_LAP(sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(sched_complete_cycle);
            if (profiling_enabled && phase_complete_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, sched_loop_count, phase_complete_count
                );
                _t0_phase = _t1;
                phase_complete_count = 0;
            }
        }
#endif

        bool try_pushed = false;

        // Phase 2 drain check: if a sync_start task is waiting for resources,
        // pause normal dispatch and let the drain protocol run.
        // relaxed load is enough — drain state only needs to be visible within
        // a few iterations; exact ordering is enforced inside handle_drain_mode.
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(
                thread_idx
#if PTO2_PROFILING
                ,
                profiling_enabled, phase_dispatch_count
#endif
            );
            continue;
        }

        // Phase 3: Drain wiring queue — wire fanout edges for newly submitted tasks.
        // Only thread 0 does wiring to keep dep_pool single-threaded.
        if (thread_idx == 0) {
            int wired = rt->scheduler.drain_wiring_queue();
            if (wired > 0) {
                made_progress = true;
#if PTO2_SCHED_PROFILING
                phase_wiring_count += wired;
#endif
            }
        }
#if PTO2_PROFILING
        CYCLE_COUNT_LAP(sched_wiring_cycle);
#endif

        // Phase 4: Dispatch
        const PTO2ResourceShape *dispatch_order = get_dispatch_order(thread_idx);
        bool entered_drain = false;

        // === Idle dispatch: assign tasks to cores with both slots free ===
        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES && !entered_drain; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            auto valid_cluster_states = tracker.get_idle_cluster_offset_states(shape);
            if (!valid_cluster_states.has_value()) {
                continue;
            }
            auto &local_buf = local_bufs[static_cast<int32_t>(shape)];

            while (valid_cluster_states.has_value() && !entered_drain) {
                int want = valid_cluster_states.count();
                PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS];
                int got = pop_ready_tasks_batch(
                    shape, thread_idx, local_buf, batch, want
#if PTO2_SCHED_PROFILING
                    ,
                    pop_hit, pop_miss, local_dispatch_count, sched_dispatch_pop_cycle
#endif
                );
                if (got == 0) break;

                for (int bi = 0; bi < got; bi++) {
                    PTO2TaskSlotState *slot_state = batch[bi];
                    try_pushed = true;
#if PTO2_SCHED_PROFILING
                    uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
                    // sync_start: all blocks must dispatch atomically.
                    // Fast path  — enough local slots: fall through to normal dispatch loop below.
                    // Slow path  — not enough: enter drain mode, then re-push all remaining
                    //              tasks in the batch so nothing is lost.
                    // For AIV, one cluster can serve 2 blocks (AIV0 + AIV1), so compare against
                    // idle AIV core count rather than cluster count.
                    if (pto2_requires_sync_start(slot_state->active_mask)) {
                        int32_t available = (shape == PTO2ResourceShape::AIV) ? tracker.count_idle_aiv_cores() :
                                                                                valid_cluster_states.count();
                        if (available < slot_state->block_num) {
                            if (!enter_drain_mode(slot_state, slot_state->block_num)) {
                                // CAS lost: drain already active for another task; re-push and wait.
                                rt->scheduler.ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                            }
                            // Re-push all unprocessed tasks remaining in this batch.
                            for (int rem = bi + 1; rem < got; rem++) {
                                rt->scheduler.ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                            }
                            entered_drain = true;
                            break;
                        }
                        // Fast path: enough local resources, fall through to normal dispatch.
                    }

                    // Dispatch as many blocks as possible for this task using available clusters.
                    // For block_num=1 the inner body executes exactly once (no overhead).
                    do {
                        auto current_valid_cluster_offset = valid_cluster_states.pop_first();
                        dispatch_block_to_cluster(
                            thread_idx, current_valid_cluster_offset, *slot_state, shape
#if PTO2_PROFILING
                            ,
                            profiling_enabled, phase_dispatch_count
#endif
                        );
                        slot_state->next_block_idx++;
                        // For AIV, refresh cluster states so the do-while can pick up the
                        // other AIV core in the same cluster on the next iteration.
                        if (shape == PTO2ResourceShape::AIV && slot_state->next_block_idx < slot_state->block_num) {
                            valid_cluster_states = tracker.get_idle_cluster_offset_states(shape);
                        }
                        DEV_DEBUG(
                            "Thread %d: Dispatched %s task %" PRId64 " block %d/%d to cluster_offset %d", thread_idx,
                            shape_name(shape), static_cast<int64_t>(slot_state->task->task_id.raw),
                            slot_state->next_block_idx - 1, slot_state->block_num, current_valid_cluster_offset
                        );
                    } while (slot_state->next_block_idx < slot_state->block_num && valid_cluster_states.has_value());

                    // Re-enqueue only if blocks remain after exhausting local clusters
                    if (slot_state->next_block_idx < slot_state->block_num) {
                        rt->scheduler.ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    }
                    made_progress = true;
#if PTO2_SCHED_PROFILING
                    sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
                }

                // lazy update valid_cluster_states
                if (!valid_cluster_states.has_value()) {
                    valid_cluster_states = tracker.get_idle_cluster_offset_states(shape);
                }
            }
        }

        // === Pending dispatch: assign AIC tasks to pending slots (core running, pending free) ===
        // Only AIC tasks support pending dispatch; sync_start tasks are excluded.
        if (!entered_drain) {
            auto pending_clusters = tracker.get_pending_only_cluster_offset_states(PTO2ResourceShape::AIC);
            if (pending_clusters.has_value()) {
                auto &local_buf = local_bufs[static_cast<int32_t>(PTO2ResourceShape::AIC)];
                while (pending_clusters.has_value()) {
                    int want = pending_clusters.count();
                    PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS];
                    int got = pop_ready_tasks_batch(
                        PTO2ResourceShape::AIC, thread_idx, local_buf, batch, want
#if PTO2_SCHED_PROFILING
                        ,
                        pop_hit, pop_miss, local_dispatch_count, sched_dispatch_pop_cycle
#endif
                    );
                    if (got == 0) break;

                    for (int bi = 0; bi < got; bi++) {
                        PTO2TaskSlotState *slot_state = batch[bi];
                        // Skip sync_start tasks for pending dispatch (need all-idle for atomic dispatch)
                        if (pto2_requires_sync_start(slot_state->active_mask)) {
                            rt->scheduler.ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC)].push(slot_state);
                            continue;
                        }
                        try_pushed = true;
#if PTO2_SCHED_PROFILING
                        uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
                        auto cluster_offset = pending_clusters.pop_first();
                        dispatch_subtask_to_core(
                            thread_idx, tracker.get_aic_core_offset(cluster_offset), *slot_state, PTO2SubtaskSlot::AIC,
                            true
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
                        );
                        slot_state->next_block_idx++;
                        if (slot_state->next_block_idx < slot_state->block_num) {
                            rt->scheduler.ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC)].push(slot_state);
                        }
                        made_progress = true;
#if PTO2_SCHED_PROFILING
                        sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
                    }
                    if (!pending_clusters.has_value()) {
                        pending_clusters = tracker.get_pending_only_cluster_offset_states(PTO2ResourceShape::AIC);
                    }
                }
            }
        }

        // requeue in global ready queue
        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            auto &local_buf = local_bufs[static_cast<int32_t>(shape)];
            auto &ready_queue = rt->scheduler.ready_queues[static_cast<int32_t>(shape)];
#if PTO2_SCHED_PROFILING
            local_overflow_count += local_buf.count;
#endif
            if (local_buf.count > 0) {
                ready_queue.push_batch(local_buf.slot_states, local_buf.count);
                local_buf.count = 0;
            }
        }

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(sched_dispatch_cycle);
            if (profiling_enabled && phase_dispatch_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, sched_loop_count, phase_dispatch_count
                );
                _t0_phase = _t1;
                phase_dispatch_count = 0;
            }
        }
#endif

#if !PTO2_PROFILING
        (void)try_completed;  // NOLINT(readability/casting)
        (void)try_pushed;     // NOLINT(readability/casting)
#endif

        if (made_progress) {
            idle_iterations = 0;
        } else {
            // Batch deferred fanin releases during idle.
            // Processing all pending releases at once advances the ring faster,
            // freeing heap space for the orchestrator without blocking completion polling.
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                int32_t fe =
                    rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                int32_t fe = rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                (void)fe;  // NOLINT(readability/casting)
#if PTO2_SCHED_PROFILING
                fanin_edges_total += fe;
                if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
            }
            idle_iterations++;

            // Check for orchestrator fatal error during idle (every 1024 iterations)
            // orch_error_code is set in shared memory by the orchestrator's spin loop
            // BEFORE orchestrator_done_ is set, so this catches errors earlier.
            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
                if (orch_err != PTO2_ERROR_NONE) {
                    DEV_ERROR(
                        "Thread %d: Fatal error detected (code=%d), sending EXIT_SIGNAL to all cores", thread_idx,
                        orch_err
                    );
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }
            }

            if (thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                DEV_ALWAYS(
                    "PTO2 stall: no progress for %d iterations, completed=%d total=%d (last progress at %d)",
                    idle_iterations, c, task_count, last_progress_count
                );
                // Scan all task slots to find truly stuck tasks using scheduler state
                PTO2SchedulerState *sched = &rt->scheduler;
                PTO2SharedMemoryHeader *sm_header_diag = static_cast<PTO2SharedMemoryHeader *>(sm_base);
                int32_t cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    int32_t ring_task_count =
                        sm_header_diag->rings[r].fc.current_task_index.load(std::memory_order_relaxed);
                    for (int32_t si = 0; si < ring_task_count; si++) {
                        PTO2TaskSlotState &slot_state = sched->get_slot_state(r, si);
                        PTO2TaskState st = slot_state.task_state.load(std::memory_order_relaxed);
                        int32_t rc = slot_state.fanin_refcount.load(std::memory_order_relaxed);
                        int32_t fi = slot_state.fanin_count;
                        int32_t kid = slot_state.task->kernel_id[0];
                        if (st >= PTO2_TASK_COMPLETED) continue;  // Already done
                        if (st == PTO2_TASK_READY || st == PTO2_TASK_RUNNING) {
                            cnt_inflight++;
                            continue;
                        }
                        // PENDING
                        if (rc >= fi) {
                            // Ready (all deps satisfied) but not enqueued — this is the real bug
                            cnt_ready++;
                            if (cnt_ready <= STALL_DUMP_READY_MAX) {
                                DEV_ALWAYS(
                                    "  STUCK-READY  ring=%d task_id=%" PRId64
                                    " kernel_id=%d refcount=%d fanin=%d state=%d",  // NOLINT(whitespace/line_length)
                                    r, static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi,
                                    static_cast<int32_t>(st)
                                );
                            }
                        } else {
                            cnt_waiting++;
                            if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                                DEV_ALWAYS(
                                    "  STUCK-WAIT   ring=%d task_id=%" PRId64
                                    " kernel_id=%d refcount=%d fanin=%d state=%d",  // NOLINT(whitespace/line_length)
                                    r, static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi,
                                    static_cast<int32_t>(st)
                                );
                            }
                        }
                    }
                }
                DEV_ALWAYS(
                    "  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d", cnt_ready, cnt_waiting, cnt_inflight
                );
                // Log this thread's dispatch state
                int32_t aic_running = tracker.get_running_count<CoreType::AIC>();
                int32_t aiv_running = tracker.get_running_count<CoreType::AIV>();
                int32_t total_running = aic_running + aiv_running;
                DEV_ALWAYS(
                    "  thread=%d running_cores=%d (AIC=%d AIV=%d) core_num=%d", thread_idx, total_running, aic_running,
                    aiv_running, core_num
                );
                // Dump running cores
                auto all_running = tracker.get_all_running_cores();
                int32_t dump_count = 0;
                int32_t bp;
                while (dump_count < STALL_DUMP_CORE_MAX && (bp = all_running.pop_first()) >= 0) {
                    dump_count++;
                    int32_t cid = tracker.get_core_id_by_offset(bp);
                    int32_t sw_tid = core_exec_states_[cid].running_reg_task_id;
                    int32_t hw_kernel = -1;
                    if (sw_tid >= 0 && core_exec_states_[cid].running_slot_state) {
                        int32_t diag_slot = static_cast<int32_t>(core_exec_states_[cid].running_subslot);
                        hw_kernel = core_exec_states_[cid].running_slot_state->task->kernel_id[diag_slot];
                    }
                    uint64_t cond_reg = read_reg(core_exec_states_[cid].reg_addr, RegId::COND);
                    DEV_ALWAYS(
                        "    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d kernel=%d", cid,
                        static_cast<unsigned>(cond_reg), EXTRACT_TASK_STATE(cond_reg), EXTRACT_TASK_ID(cond_reg),
                        sw_tid, hw_kernel
                    );
                }
                // Dump cluster state
                for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++) {
                    int32_t offset = cli * 3;
                    DEV_ALWAYS(
                        "    cluster[%d] aic=%d(%s) aiv0=%d(%s) aiv1=%d(%s)", cli, tracker.get_aic_core_id(offset),
                        tracker.is_aic_core_idle(offset) ? "idle" : "busy", tracker.get_aiv0_core_id(offset),
                        tracker.is_aiv0_core_idle(offset) ? "idle" : "busy", tracker.get_aiv1_core_id(offset),
                        tracker.is_aiv1_core_idle(offset) ? "idle" : "busy"
                    );
                }
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
#if PTO2_PROFILING
                // Benchmark: scheduler lifetime end timestamp on timeout path
                uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
                DEV_ALWAYS(
                    "Thread %d: sched_start=%" PRIu64 " sched_end(timeout)=%" PRIu64 " sched_cost=%.3fus", thread_idx,
                    static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts),
                    cycles_to_us(sched_timeout_ts - sched_start_ts)
                );
#endif
                return -1;
            } else {
                SPIN_WAIT_HINT();
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_idle_cycle);
            if (profiling_enabled) {
                perf_aicpu_record_phase(thread_idx, AicpuPhaseId::SCHED_IDLE_WAIT, _t0_phase, _t1, sched_loop_count, 0);
                _t0_phase = _t1;
            }
#endif
        }
    }

#if PTO2_PROFILING
    // Record sched_end before any DEV_ALWAYS to avoid init cost contamination
    uint64_t sched_end_ts = get_sys_cnt_aicpu();
    DEV_ALWAYS(
        "Thread %d: sched_start=%" PRIu64 " sched_end=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_end_ts),
        cycles_to_us(sched_end_ts - sched_start_ts)
    );

    // Scheduler summary logging (always print when PTO2_PROFILING=1)
    uint64_t sched_total =
        sched_wiring_cycle + sched_complete_cycle + sched_scan_cycle + sched_dispatch_cycle + sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;  // avoid div-by-zero

#if PTO2_SCHED_PROFILING
    // Two-level tree display: sub-phase breakdown within complete and dispatch
    {
        PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll = (sched_complete_cycle > otc_total + sched_complete_perf_cycle) ?
                                     (sched_complete_cycle - otc_total - sched_complete_perf_cycle) :
                                     0;
        uint64_t dispatch_poll = (sched_dispatch_cycle > sched_dispatch_pop_cycle + sched_dispatch_setup_cycle) ?
                                     (sched_dispatch_cycle - sched_dispatch_pop_cycle - sched_dispatch_setup_cycle) :
                                     0;

        DEV_ALWAYS(
            "Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===", thread_idx,
            cycles_to_us(sched_total), cur_thread_completed
        );

        // Level 1: complete
        double notify_avg =
            cur_thread_completed > 0 ? static_cast<double>(notify_edges_total) / cur_thread_completed : 0.0;
        double fanin_avg =
            cur_thread_completed > 0 ? static_cast<double>(fanin_edges_total) / cur_thread_completed : 0.0;
        DEV_ALWAYS(
            "Thread %d:   complete       : %.3fus (%.1f%%)  [fanout: edges=%" PRIu64
            ", max_degree=%d, avg=%.1f]  [fanin: "  // NOLINT(whitespace/line_length)
            "edges=%" PRIu64 ", max_degree=%d, avg=%.1f]",
            thread_idx, cycles_to_us(sched_complete_cycle), sched_complete_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(notify_edges_total), notify_max_degree, notify_avg,
            static_cast<uint64_t>(fanin_edges_total), fanin_max_degree, fanin_avg
        );

        // Level 2: complete sub-phases (percentage relative to complete)
        uint64_t c_parent = sched_complete_cycle > 0 ? sched_complete_cycle : 1;
        uint64_t complete_miss_count =
            (complete_probe_count > complete_hit_count) ? (complete_probe_count - complete_hit_count) : 0;
        double complete_hit_rate = complete_probe_count > 0 ? complete_hit_count * 100.0 / complete_probe_count : 0.0;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)  hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%",
            thread_idx, cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent,
            static_cast<uint64_t>(complete_hit_count), static_cast<uint64_t>(complete_miss_count), complete_hit_rate
        );
        DEV_ALWAYS(
            "Thread %d:     otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
            cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle), cycles_to_us(sp.lock_wait_cycle),
            static_cast<uint64_t>(sp.lock_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
            cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle), cycles_to_us(sp.push_wait_cycle),
            static_cast<uint64_t>(sp.fanout_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_fanin    : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.fanin_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_self     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.self_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     perf         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_complete_perf_cycle),
            sched_complete_perf_cycle * 100.0 / c_parent
        );

        // Level 1: dispatch
        uint64_t pop_total = pop_hit + pop_miss;
        double pop_hit_rate = pop_total > 0 ? pop_hit * 100.0 / pop_total : 0.0;
        DEV_ALWAYS(
            "Thread %d:   dispatch       : %.3fus (%.1f%%)  [pop: hit=%" PRIu64 ", miss=%" PRIu64
            ", hit_rate=%.1f%%]",  // NOLINT(whitespace/line_length)
            thread_idx, cycles_to_us(sched_dispatch_cycle), sched_dispatch_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(pop_hit), static_cast<uint64_t>(pop_miss), pop_hit_rate
        );
        uint64_t global_dispatch_count = pop_hit - local_dispatch_count;
        uint64_t total_dispatched = local_dispatch_count + global_dispatch_count;
        double local_hit_rate = total_dispatched > 0 ? local_dispatch_count * 100.0 / total_dispatched : 0.0;
        DEV_ALWAYS(
            "Thread %d:     local_disp   : local=%" PRIu64 ", global=%" PRIu64 ", overflow=%" PRIu64
            ", local_rate=%.1f%%",  // NOLINT(whitespace/line_length)
            thread_idx, static_cast<uint64_t>(local_dispatch_count), static_cast<uint64_t>(global_dispatch_count),
            static_cast<uint64_t>(local_overflow_count), local_hit_rate
        );

        // Level 2: dispatch sub-phases (percentage relative to dispatch)
        uint64_t d_parent = sched_dispatch_cycle > 0 ? sched_dispatch_cycle : 1;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent
        );
        DEV_ALWAYS(
            "Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sched_dispatch_pop_cycle), sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(sched_dispatch_pop_cycle - sp.pop_wait_cycle), cycles_to_us(sp.pop_wait_cycle),
            static_cast<uint64_t>(sp.pop_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     setup        : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_dispatch_setup_cycle),
            sched_dispatch_setup_cycle * 100.0 / d_parent
        );

        // Level 1: scan
        DEV_ALWAYS(
            "Thread %d:   scan           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_scan_cycle),
            sched_scan_cycle * 100.0 / sched_total
        );

        // Level 1: wiring
#if PTO2_SCHED_PROFILING
        DEV_ALWAYS(
            "Thread %d:   wiring         : %.3fus (%.1f%%)  tasks=%d", thread_idx, cycles_to_us(sched_wiring_cycle),
            sched_wiring_cycle * 100.0 / sched_total, phase_wiring_count
        );
#else
        DEV_ALWAYS(
            "Thread %d:   wiring         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_wiring_cycle),
            sched_wiring_cycle * 100.0 / sched_total
        );
#endif

        // Level 1: idle
        DEV_ALWAYS(
            "Thread %d:   idle           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_idle_cycle),
            sched_idle_cycle * 100.0 / sched_total
        );

        // Average per completion
        if (cur_thread_completed > 0) {
            DEV_ALWAYS(
                "Thread %d:   avg/complete   : %.3fus", thread_idx,
                cycles_to_us(sched_complete_cycle) / cur_thread_completed
            );
        }
    }
#endif
    // Summary line (always print when PTO2_PROFILING=1)
    DEV_ALWAYS(
        "Thread %d: Scheduler summary: total_time=%.3fus, loops=%" PRIu64 ", tasks_scheduled=%d", thread_idx,
        cycles_to_us(sched_total), static_cast<uint64_t>(sched_loop_count), cur_thread_completed
    );
#endif

    return cur_thread_completed;
}

int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t thread_idx = thread_idx_++;
    DEV_INFO("Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
        int32_t pto2_submitted_tasks = -1;
#endif
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO("Thread %d: Host orchestration mode, no-op", thread_idx);
        } else {
            DEV_INFO("Thread %d: Orchestrator, loading SO via dlopen", thread_idx);

            const void *so_data = runtime->get_device_orch_so_data();
            size_t so_size = runtime->get_device_orch_so_size();

            if (so_data == nullptr || so_size == 0) {
                DEV_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
                return -1;
            }

            // Try multiple paths that may allow execution on AICPU
            char so_path[256];
            bool file_created = false;
            const char *candidate_dirs[] = {
                "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"
            };
            const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

            for (int32_t i = 0; i < num_candidates && !file_created; i++) {
                int32_t fd = create_orch_so_file(candidate_dirs[i], so_path, sizeof(so_path));
                if (fd < 0) {
                    DEV_INFO(
                        "Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path, errno
                    );
                    continue;
                }
                ssize_t written = write(fd, so_data, so_size);
                close(fd);
                if (written != static_cast<ssize_t>(so_size)) {
                    DEV_INFO(
                        "Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path, errno
                    );
                    unlink(so_path);
                    continue;
                }
                file_created = true;
                DEV_INFO("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
            }

            if (!file_created) {
                DEV_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
                return -1;
            }

            dlerror();
            void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
            const char *dlopen_err = dlerror();
            if (handle == nullptr) {
                DEV_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
                unlink(so_path);
                return -1;
            }
            DEV_INFO("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

            dlerror();
            auto config_func =
                reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, "aicpu_orchestration_config"));

            dlerror();
            DeviceOrchestrationFunc orch_func =
                reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, "aicpu_orchestration_entry"));
            const char *dlsym_error = dlerror();
            if (dlsym_error != nullptr) {
                DEV_ERROR("Thread %d: dlsym failed: %s", thread_idx, dlsym_error);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }
            if (orch_func == nullptr) {
                DEV_ERROR("Thread %d: dlsym returned NULL for aicpu_orchestration_entry", thread_idx);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            dlerror();
            auto bind_runtime_func =
                reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "pto2_framework_bind_runtime"));
            const char *bind_runtime_error = dlerror();
            if (bind_runtime_error != nullptr) {
                DEV_INFO("Thread %d: Optional TLS runtime binder not found: %s", thread_idx, bind_runtime_error);
                bind_runtime_func = nullptr;
            }

            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            int32_t arg_count = args.tensor_count() + args.scalar_count();
            DEV_INFO("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_pto2_gm_sm_ptr(), arg_count);
            for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                const ContinuousTensor &t = args.tensor(i);
                DEV_INFO(
                    "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                    static_cast<uint64_t>(t.data), t.ndims, static_cast<unsigned>(t.dtype)
                );
            }
            for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                DEV_INFO(
                    "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
                    static_cast<uint64_t>(args.scalar(i))
                );
            }

            uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            uint64_t heap_size = PTO2_HEAP_SIZE;
            int32_t expected_arg_count = 0;
            if (config_func) {
                PTO2OrchestrationConfig cfg = config_func(args);
                expected_arg_count = cfg.expected_arg_count;
                DEV_INFO("Thread %d: Config: expected_args=%d", thread_idx, expected_arg_count);
            } else {
                DEV_INFO("Thread %d: No config function, using defaults", thread_idx);
            }

            if (expected_arg_count > 0 && arg_count < expected_arg_count) {
                DEV_ERROR("Thread %d: arg_count %d < expected %d", thread_idx, arg_count, expected_arg_count);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            if (runtime->pto2_task_window_size > 0) {
                task_window_size = runtime->pto2_task_window_size;
            }
            if (runtime->pto2_heap_size > 0) {
                heap_size = runtime->pto2_heap_size;
            }
            int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
            if (runtime->pto2_dep_pool_size > 0) {
                dep_pool_capacity = static_cast<int32_t>(runtime->pto2_dep_pool_size);
            }
            DEV_INFO(
                "Thread %d: Ring sizes: task_window=%lu, heap=%lu, dep_pool=%d", thread_idx,
                static_cast<uint64_t>(task_window_size), static_cast<uint64_t>(heap_size), dep_pool_capacity
            );

            void *sm_ptr = runtime->get_pto2_gm_sm_ptr();
            void *gm_heap = runtime->get_pto2_gm_heap_ptr();

            uint64_t sm_size = pto2_sm_calculate_size(task_window_size);
            PTO2SharedMemoryHandle *sm_handle =
                pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size);
            if (!sm_handle) {
                DEV_ERROR("Thread %d: Failed to create shared memory handle", thread_idx);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            rt = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE, sm_handle, gm_heap, heap_size, dep_pool_capacity);
            if (!rt) {
                DEV_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                pto2_sm_destroy(sm_handle);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

#if PTO2_PROFILING
            rt->orchestrator.enable_profiling = runtime->enable_profiling;
#endif

            // Total core counts = aic_count_ / aiv_count_ (set once at runtime init).
            rt->orchestrator.total_cluster_count = aic_count_;
            rt->orchestrator.total_aiv_count = aiv_count_;

            // With multi-ring, slot_states are per-ring inside the scheduler.
            runtime->set_pto2_slot_states_ptr(nullptr);

            orch_func_ = orch_func;
            orch_bind_runtime_ = bind_runtime_func;
            orch_args_cached_ = &args;
            orch_so_handle_ = handle;
            snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);

            runtime_init_ready_.store(true, std::memory_order_release);

            // Wait for scheduler's one-time init to complete
            while (!pto2_init_complete_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }

#if PTO2_PROFILING
            if (runtime->enable_profiling) {
                perf_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            pto2_framework_bind_runtime(rt);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(rt);
            }
            pto2_rt_scope_begin(rt);
            orch_func_(*orch_args_cached_);
            pto2_rt_scope_end(rt);
#if PTO2_PROFILING
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            (void)orch_cycle_end;  // NOLINT(readability/casting)
#endif

            // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
            PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
            uint64_t total =
                p.sync_cycle + p.alloc_cycle + p.args_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
            if (total == 0) total = 1;  // avoid div-by-zero
            DEV_ALWAYS(
                "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            DEV_ALWAYS(
                "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle),
                static_cast<uint64_t>(p.fanin_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   avg/task       : %.3fus", thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
            );

#if PTO2_TENSORMAP_PROFILING
            PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
            DEV_ALWAYS("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            DEV_ALWAYS(
                "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
            );
            DEV_ALWAYS(
                "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                static_cast<uint64_t>(tp.lookup_chain_total),
                tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                tp.lookup_chain_max
            );
            DEV_ALWAYS(
                "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
                static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
                tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
            );
#endif

#if PTO2_PROFILING
            // Write orchestrator summary to shared memory for host-side export (only if profiling enabled)
            if (runtime->enable_profiling) {
                AicpuOrchSummary orch_summary = {};
                orch_summary.start_time = orch_cycle_start;
                orch_summary.end_time = orch_cycle_end;
                orch_summary.sync_cycle = p.sync_cycle;
                orch_summary.alloc_cycle = p.alloc_cycle;
                orch_summary.args_cycle = p.args_cycle;
                orch_summary.lookup_cycle = p.lookup_cycle;
                orch_summary.heap_cycle = 0;  // Now included in alloc_cycle
                orch_summary.insert_cycle = p.insert_cycle;
                orch_summary.fanin_cycle = p.fanin_cycle;
                orch_summary.scope_end_cycle = p.scope_end_cycle;
                orch_summary.submit_count = p.submit_count;
                perf_aicpu_write_orch_summary(&orch_summary);
            }
#endif
#endif

#if PTO2_PROFILING
            // Write core-to-thread mapping (one-time, after orchestration)
            if (runtime->enable_profiling) {
                perf_aicpu_write_core_assignments(
                    core_assignments_, core_count_per_thread_, sched_thread_num_, cores_total_num_
                );
            }
#endif

            // Signal completion and trigger core transition
            pto2_rt_orchestration_done(rt);

            void *sm = runtime->get_pto2_gm_sm_ptr();
            PTO2SharedMemoryHeader *sm_header = static_cast<PTO2SharedMemoryHeader *>(sm);
            int32_t pto2_task_count = 0;
            if (sm_header) {
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    pto2_task_count += sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                }
            }
#if PTO2_PROFILING
            pto2_submitted_tasks = pto2_task_count;
#endif
            total_tasks_ = pto2_task_count;
            if (runtime->enable_profiling && pto2_task_count > 0) {
                perf_aicpu_update_total_tasks(runtime, static_cast<uint32_t>(pto2_task_count));
            }
            int32_t inline_completed = static_cast<int32_t>(rt->orchestrator.inline_completed_tasks);
            if (inline_completed > 0) {
                completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
#if PTO2_SCHED_PROFILING
                rt->scheduler.tasks_completed.fetch_add(inline_completed, std::memory_order_relaxed);
#endif
            }
            orchestrator_done_ = true;
            {
                int32_t orch_err = 0;
                void *sm = runtime->get_pto2_gm_sm_ptr();
                if (sm) {
                    orch_err =
                        static_cast<PTO2SharedMemoryHeader *>(sm)->orch_error_code.load(std::memory_order_relaxed);
                }

                // Fatal error: shutdown AICore immediately before core transition.
                if (orch_err != PTO2_ERROR_NONE) {
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                }
            }

#if PTO2_ORCH_PROFILING
            uint64_t reassign_cycle_start = get_sys_cnt_aicpu();
#endif

            // Skip core transition on fatal error — cores already shut down above
            if (completed_.load(std::memory_order_acquire)) {
                // Signal transition to unblock scheduler threads waiting at core transition
                transition_requested_.store(true, std::memory_order_release);
                reassigned_.store(true, std::memory_order_release);
            } else if (orch_to_sched_) {
                // Compute new core assignments for all threads and initialize donated slots
                DEV_INFO("Thread %d: Set orchestrator_done=true, requesting core transition", thread_idx);
#if PTO2_PROFILING
                uint64_t orch_stage_end_ts = get_sys_cnt_aicpu();
#endif
                transition_requested_.store(true, std::memory_order_release);
#if PTO2_PROFILING
                DEV_ALWAYS(
                    "Thread %d: orch_stage_end=%" PRIu64 "", thread_idx, static_cast<uint64_t>(orch_stage_end_ts)
                );
#endif

                // Wait for scheduler threads to acknowledge transition request
                while (wait_reassign_.load(std::memory_order_acquire) != sched_thread_num_) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (!completed_.load(std::memory_order_acquire)) {
                    reassign_cores_for_all_threads();
                    reassigned_.store(true, std::memory_order_release);
                }
            }

#if PTO2_ORCH_PROFILING
            uint64_t reassign_cycle_end = get_sys_cnt_aicpu();
            DEV_ALWAYS(
                "Thread %d: reassign, cost %.3fus", thread_idx, cycles_to_us(reassign_cycle_end - reassign_cycle_start)
            );
#endif
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        DEV_ALWAYS(
            "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
        if (pto2_submitted_tasks >= 0) {
            DEV_ALWAYS(
                "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
                completed_tasks_.load(std::memory_order_acquire)
            );
        }
#endif
        DEV_INFO("Thread %d: Orchestrator completed", thread_idx);
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!completed_.load(std::memory_order_acquire) && (thread_idx < sched_thread_num_ || orch_to_sched_)) {
        // Device orchestration: wait for primary orchestrator to initialize SM header
        if (!runtime->get_orch_built_on_host()) {
            while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }
        }
        always_assert(rt != nullptr);
        int32_t completed = resolve_and_dispatch_pto2(runtime, thread_idx);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
    }

    // Always shutdown AICore — even if completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_count_per_thread_ == 0 so they skip the loop harmlessly.
    {
        const int32_t *shutdown_cores = core_assignments_[thread_idx];
        int32_t shutdown_count = core_count_per_thread_[thread_idx];
        if (shutdown_count > 0) {
            auto rc = shutdown_aicore(runtime, thread_idx, shutdown_cores, shutdown_count);
            if (rc != 0) {
                return rc;
            }
        }
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime and close orchestration SO (moved from orchestrator path)
        if (!runtime->get_orch_built_on_host() && orch_so_handle_ != nullptr) {
            // Clear g_pto2_current_runtime in this DSO and in the orchestration SO before destroying rt.
            pto2_framework_bind_runtime(nullptr);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(nullptr);
            }
            pto2_runtime_destroy(rt);
        }
    }

    return 0;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i] = {};
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Clear per-core dispatch payloads
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_ = false;
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    // Reset core transition state
    transition_requested_.store(false, std::memory_order_release);
    wait_reassign_.store(0, std::memory_order_release);
    reassigned_.store(false, std::memory_order_release);
    completed_.store(false, std::memory_order_release);

    // Reset core discovery and assignment state
    aic_count_ = 0;
    aiv_count_ = 0;
    cores_total_num_ = 0;
    thread_num_ = 0;
    sched_thread_num_ = 0;
    thread_cores_num_ = 0;
    orch_to_sched_ = false;
    active_sched_threads_ = 0;
    memset(core_trackers_, 0, sizeof(core_trackers_));
    memset(core_assignments_, 0, sizeof(core_assignments_));
    memset(core_count_per_thread_, 0, sizeof(core_count_per_thread_));

    regs_ = 0;
    orch_func_ = nullptr;
    orch_bind_runtime_ = nullptr;
    orch_args_cached_ = nullptr;
    if (orch_so_handle_ != nullptr) {
        dlclose(orch_so_handle_);
    }
    if (orch_so_path_[0] != '\0') {
        unlink(orch_so_path_);
    }
    orch_so_handle_ = nullptr;
    orch_so_path_[0] = '\0';

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::emergency_shutdown(Runtime *runtime) {
    DEV_WARN("Emergency shutdown: sending exit signal to all initialized cores");
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];
        OUT_OF_ORDER_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;
        if (core_exec_states_[i].reg_addr != 0) {
            platform_deinit_aicore_regs(core_exec_states_[i].reg_addr);
        }
    }

    DEV_WARN("Emergency shutdown complete");
}

void AicpuExecutor::diagnose_stuck_state(
    Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num, Handshake *hank
) {
    (void)runtime;  // NOLINT(readability/casting)
    PTO2SchedulerState *sched = &rt->scheduler;
    DEV_ALWAYS("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int32_t completed = completed_tasks_.load(std::memory_order_acquire);
    int32_t total = total_tasks_;
    DEV_ALWAYS("Progress: %d/%d tasks (%.1f%%)", completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    uint64_t aic_ready = 0, aiv_ready = 0, mix_ready = 0;
    if (rt) {
        aic_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC)].size();
        aiv_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIV)].size();
        mix_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size();
    }
    DEV_ALWAYS("Ready Queues: AIC=%lu, AIV=%lu, MIX=%lu", aic_ready, aiv_ready, mix_ready);

    int32_t busy_cores = 0;
    int32_t idle_cores = 0;

    DEV_ALWAYS("Core Status:");
    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        Handshake *h = &hank[core_id];
        const char *core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);
        int32_t task_id = core_exec_states_[core_id].running_reg_task_id;

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                int32_t kernel_id = -1;
                if (rt && rt->sm_handle && core_exec_states_[core_id].running_slot_state) {
                    int32_t diag_slot = static_cast<int32_t>(core_exec_states_[core_id].running_subslot);
                    kernel_id = core_exec_states_[core_id].running_slot_state->task->kernel_id[diag_slot];
                }
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), running_reg_task_id=%d, "
                    "kernel_id=%d",
                    core_id, core_type_str, reg_val, reg_task_id, reg_state == TASK_FIN_STATE ? "FIN" : "ACK", task_id,
                    kernel_id
                );
            } else {
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s) but task_id not tracked", core_id,
                    core_type_str, reg_val, reg_task_id, reg_state == TASK_FIN_STATE ? "FIN" : "ACK"
                );
            }
        } else {
            idle_cores++;
        }
    }

    DEV_ALWAYS("Summary: %d busy, %d idle", busy_cores, idle_cores);

    // Diagnose deadlock vs livelock
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < total) {
        DEV_ALWAYS("*** DEADLOCK DETECTED ***");
        DEV_ALWAYS("All cores idle, no ready tasks, but %d tasks incomplete", total - completed);
        DEV_ALWAYS("Check PTO2 shared memory for task dependency state");
    } else if (busy_cores > 0) {
        DEV_ALWAYS("*** LIVELOCK / HUNG TASK ***");
        DEV_ALWAYS("%d cores executing but no progress", busy_cores);
    }

    DEV_ALWAYS("========== END DIAGNOSTIC ==========");
}

// ===== Public Entry Point =====

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // Get platform register addresses from platform-level global
    g_aicpu_executor.regs_ = get_platform_regs();

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        DEV_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    DEV_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
