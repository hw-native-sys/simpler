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
#ifndef SCHEDULER_CONTEXT_H
#define SCHEDULER_CONTEXT_H

#include "aicpu/platform_regs.h"
#include "common/l2_swimlane_profiling.h"
#include "common/unified_log.h"
#include "scheduler_types.h"

#include "scheduler/pto_scheduler.h"

#include "aicore_completion_mailbox.h"
#include "pto2_dispatch_payload.h"

// These macros are defined in runtime.h, but we cannot include it here
// (it pulls in Handshake which we only forward-declare).  Mirror the
// authoritative values so the class layout compiles standalone.
#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 1024
#endif

// Forward declarations — avoid pulling in full headers for pointer/reference params.
class Runtime;
struct Handshake;
struct PTO2Runtime;

/**
 * SchedulerContext: owns all scheduler-side state and methods.
 *
 * Held as a member of AicpuExecutor (sched_ctx_).  The single public entry
 * point is resolve_and_dispatch(), called once per scheduler thread.
 *
 * All dispatch/completion/drain/cold-path logic is implemented as private
 * member methods, split across three .cpp files by responsibility:
 *   - scheduler_completion.cpp  (completion polling, drain protocol)
 *   - scheduler_cold_path.cpp   (exit checks, stall diagnostics, profiling)
 *   - scheduler_dispatch.cpp    (task dispatch loop and helpers)
 */
class SchedulerContext {
public:
    // =========================================================================
    // Lifecycle
    // =========================================================================

    // Initialize scheduler state from the given runtime and thread layout. Split
    // into three parts so the per-core AICore handshake — a serial, MMIO-bound
    // loop that dominates preamble (~217 µs of ~283 µs for 72 cores) — can run in
    // parallel across all AICPU threads. Orchestrated by AicpuExecutor::init:
    // the leader runs pre_handshake_init, every thread handshakes a disjoint
    // slice of cores via handshake_partition, then the leader runs
    // post_handshake_init after a barrier.
    //
    // Leader-only: per-core state + config + swimlane buffers + core count. Must
    // be published before any thread enters handshake_partition. Returns 0 on
    // success, negative on failure.
    int32_t
    pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, uint64_t regs_base);
    // All threads: handshake this thread's contiguous slice [lo, hi) of cores
    // (partitioned by tidx/nthreads). Each core is touched by exactly one thread.
    void handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);
    // Leader-only, after the handshake barrier: build worker-id lists, assign
    // cores, init profiling subsystems, read task counts, init payloads.
    int32_t post_handshake_init(Runtime *runtime);

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit();

    // =========================================================================
    // Per-thread execution entry points (called by AicpuExecutor::run)
    // =========================================================================

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    // Shutdown AICore registers for this thread's assigned cores.
    // Also runs PMU finalize (SIMPLER_DFX) before deinit when enabled.
    // Orchestrator threads (core_trackers_[thread_idx].core_num() == 0) are a no-op.
    int32_t shutdown(int32_t thread_idx);

    // Run all post-orchestration scheduler bookkeeping:
    //  - publishes core assignments to the perf collector (SIMPLER_DFX)
    //  - latches submitted task count from PTO2 shared memory
    //  - folds inline_completed_tasks into completed_tasks_
    //  - flips orchestrator_done_ and triggers core transition
    //    (skipped on fatal error — emergency_shutdown runs instead)
    // Callers must invoke rt_orchestration_done(rt) before this — that
    // step belongs to the orchestrator lifecycle, not the scheduler.
    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t thread_idx, int32_t total_tasks);

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt);

    // Serial orch->sched mode pre-dispatch wait. No AICore dispatch happens
    // before orchestrator_done_.
    void wait_for_orchestration_done_before_dispatch(Runtime *runtime, int32_t thread_idx);

    // =========================================================================
    // State queries / external synchronization points
    // =========================================================================

    int32_t aic_count() const { return aic_count_; }
    int32_t aiv_count() const { return aiv_count_; }
    bool is_completed() const { return completed_.load(std::memory_order_acquire); }
    int32_t completed_tasks_count() const { return completed_tasks_.load(std::memory_order_acquire); }
    bool orchestration_done() const { return orchestrator_done_.load(std::memory_order_relaxed); }

private:
    // =========================================================================
    // State
    // =========================================================================

    // --- Scheduler binding & per-core runtime state ---
    alignas(64) PTO2SchedulerState *sched_{nullptr};
    PTO2Runtime *rt_{nullptr};

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered core trackers, one per scheduler thread
    CoreTracker core_trackers_[MAX_AICPU_THREADS];

    // Per-core dispatch payload storage: dual-buffer for pipelining.
    // buf_idx = reg_task_id & 1; adjacent dispatches alternate automatically.
    PTO2DispatchPayload payload_per_core_[RUNTIME_MAX_WORKER][2];

    // Per-core deferred-completion software registration storage.  This has
    // the same runtime lifetime as payload_per_core_, but is kept out of the
    // dispatch payload so normal task dispatch layout and cache footprint stay
    // unchanged.
    DeferredCompletionSlab deferred_slab_per_core_[RUNTIME_MAX_WORKER][2];

    // sync_start drain coordination
    SyncStartDrainState drain_state_;

#if SIMPLER_DFX
    SchedL2SwimlaneCounters sched_l2_swimlane_[MAX_AICPU_THREADS];
    // Cached once at init() from get_l2_swimlane_level(), AFTER
    // l2_swimlane_aicpu_init has promoted the level from the shared-memory header.
    L2SwimlaneLevel l2_swimlane_level_{L2SwimlaneLevel::DISABLED};
#endif

    // --- Task-execution tracking ---
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    std::atomic<bool> orchestrator_done_{false};
    std::atomic<bool> completed_{false};
    uint64_t *func_id_to_addr_{nullptr};

    // --- Thread/core configuration ---
    int32_t active_sched_threads_{0};
    int32_t sched_thread_num_{0};
    int32_t aicpu_thread_num_{0};
    int32_t cores_total_num_{0};

    // Cluster-ordered worker_id lists, populated by post_handshake_init().
    int32_t aic_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aiv_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Compact per-core CoreType, packed contiguously (~2 cache lines total) so
    // post_handshake_init's ordered discovery scan reads it instead of taking a
    // per-core volatile GM load from the 64B-aligned Handshake struct. Filled by
    // each handshake thread for its own [lo,hi) slice during the parallel sweep.
    uint8_t core_type_compact_[RUNTIME_MAX_WORKER]{};

    // Set by any thread whose slice hits an invalid physical_core_id in
    // handshake_partition; checked by the leader in post_handshake_init.
    std::atomic<bool> handshake_failed_{false};

    // Platform AICore-register base array (set by AicpuExecutor before init()).
    uint64_t regs_{0};

#if SIMPLER_DFX
    // PMU profiling: physical core IDs for PMU MMIO base resolution.
    // Separate storage because CoreExecState's 64-byte budget has no room for
    // physical_core_id when SIMPLER_DFX=1.
    uint32_t physical_core_ids_[RUNTIME_MAX_WORKER]{};
#endif

    // =========================================================================
    // Core management (scheduler_cold_path.cpp)
    // =========================================================================

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads();

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime);

    // =========================================================================
    // Dispatch (scheduler_dispatch.cpp)
    // =========================================================================

    static const char *shape_name(PTO2ResourceShape shape);

    // Lower-case rendering of PTO2SubtaskSlot, used by dispatch and stall logs.
    // Kept lower-case to match the `kernels=[aic:N aiv0:N aiv1:N]` field
    // convention already established in the stall log family.
    static inline const char *subslot_name(PTO2SubtaskSlot s) {
        switch (s) {
        case PTO2SubtaskSlot::AIC:
            return "aic";
        case PTO2SubtaskSlot::AIV0:
            return "aiv0";
        case PTO2SubtaskSlot::AIV1:
            return "aiv1";
        }
        return "?";
    }

    int pop_ready_tasks_batch(
        PTO2ReadyQueue *queues, PTO2ResourceShape shape, int32_t thread_idx, PTO2TaskSlotState **out, int max_count
    );

    void build_payload(
        PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        int32_t block_idx, bool force_gate
    );

    // Batched-dispatch primitives. prepare_* builds the payload and per-core
    // state; publish_* issues the MMIO register write. Callers must wmb()
    // between the prepare batch and the publish batch, then sample
    // get_sys_cnt_aicpu() once and pass it to publish_* for every handle.
    //
    // dispatch_timestamp_slot points to the CoreExecState slot
    // (pending_dispatch_timestamp / running_dispatch_timestamp) selected at
    // prepare time, or nullptr when L2 swimlane is below AICPU_TIMING and no
    // dispatch timestamp is being recorded.
    struct PublishHandle {
        uint64_t reg_addr;
        uint32_t reg_task_id;
        int32_t core_offset;
        uint64_t *dispatch_timestamp_slot;
    };

    PublishHandle prepare_subtask_to_core(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        bool to_pending, int32_t block_idx, bool force_gate
    );

    inline void publish_subtask_to_core(const PublishHandle &h, uint64_t dispatch_ts) {
        if (h.dispatch_timestamp_slot != nullptr) {
            *h.dispatch_timestamp_slot = dispatch_ts;
        }
        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));
    }

    // Prefetch the cold per-core structures the next block's prepare touches.
    // Ordering is load-bearing: issue the STALLING LOAD first — CoreExecState,
    // read by dispatch_seq++ whose value feeds reg_task_id -> buf_idx -> the whole
    // dispatch — for every core of the block, BEFORE the store-target prefetches.
    // MSHRs saturate (a MIX block warms 3 cores); issuing the read prefetches
    // first keeps them from being the ones dropped. The dispatch-buffer writes
    // still get prefetched (measured to help ~30% on this shallow-store-buffer
    // control core), just after the reads. rw=1 on CoreExecState (read AND
    // written) gives Exclusive, serving both without a Shared->Exclusive upgrade.
    inline void prefetch_block_dst(int32_t thread_idx, int32_t core_offset, bool is_mix) {
        CoreTracker &tracker = core_trackers_[thread_idx];
        int32_t cids[3];
        int32_t nc = 0;
        if (is_mix) {
            cids[nc++] = tracker.get_core_id_by_offset(tracker.get_aic_core_offset(core_offset));
            cids[nc++] = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(core_offset));
            cids[nc++] = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(core_offset));
        } else {
            cids[nc++] = tracker.get_core_id_by_offset(core_offset);
        }
        // Stalling loads first.
        for (int32_t i = 0; i < nc; i++)
            __builtin_prefetch(&core_exec_states_[cids[i]], 1, 3);
        // Store targets after (dispatch buffer CL0 control + CL1 args, both bufs).
        for (int32_t i = 0; i < nc; i++) {
            for (int32_t buf = 0; buf < 2; buf++) {
                const char *dp = reinterpret_cast<const char *>(&payload_per_core_[cids[i]][buf]);
                __builtin_prefetch(dp, 1, 3);
                __builtin_prefetch(dp + 64, 1, 3);
            }
        }
    }

    // Fan out one block's subtasks (1 for AIC/AIV, 1-3 for MIX) into the
    // caller-supplied handles buffer. Returns the number of handles written.
    int prepare_block_for_dispatch(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape,
        bool to_pending, int32_t block_idx, PublishHandle *out_handles, bool force_gate = false
    );

    void dispatch_shape(
        int32_t thread_idx, PTO2ReadyQueue *disp_queues, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
        CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
    );

    // Early-dispatch (Hook 1). Mirrors dispatch_ready_tasks: owns its
    // own gating (off-PMU, this thread has a spare slot, and no normal ready work
    // is queued) and sets made_progress / try_pushed when it stages, so the caller
    // is a single unconditional call like normal dispatch. After normal dispatch
    // leaves idle cores spare, pre-stage the consumers of any RUNNING flagged
    // producer onto those cores with a non-zero src_payload (gated). Touches no dependency
    // state — the task is released by the doorbell at its normal ready-pop (Hook 2).
    int32_t try_early_dispatch(
        int32_t thread_idx, CoreTracker &tracker, bool pmu_active, bool &made_progress, bool &try_pushed
    );

    // Stage the already-claimed range [start, start+count) of consumer `c` onto
    // thread_idx's idle (RUNNING slot) then pending (gated-pending, promote-on-FIN)
    // cores from the provided free-core sets. The caller advances next_block_idx and
    // re-pushes `c` BEFORE calling, so this expensive prepare+publish runs
    // concurrently with peers (mirrors the normal SPMD dispatch path). Returns the
    // number of blocks staged.
    int32_t stage_consumer_blocks(
        int32_t thread_idx, PTO2TaskSlotState *c, PTO2ResourceShape shape, int32_t start, int32_t count,
        CoreTracker::BitStates &idle, CoreTracker::BitStates &pend
    );

    // Early-dispatch analog of dispatch_shape: drain early_dispatch_queues[shape] and
    // pre-stage claimed block ranges onto this thread's free cores of `shape` for the
    // given phase (IDLE -> onto idle cores in the RUNNING slot; PENDING -> onto a
    // running core's gated pending slot). Pop is sized to the shape's capacity exactly
    // as dispatch_shape sizes normal dispatch. Returns the number of blocks staged.
    int32_t early_dispatch_shape(int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase);

    // One pass of "Phase 4" in the resolve_and_dispatch loop: IDLE-stage dispatch
    // for MIX then (if no mix residual) AIC/AIV; then PENDING-stage dispatch with
    // cross-thread idle gating. MIX is strictly prioritized — when mix residual is
    // detected after MIX-IDLE, AIC/AIV are skipped for the whole pass but
    // MIX-PENDING still runs.
    //
    // Forward-progress argument for AIC/AIV: skip_aic_aiv is sticky for the
    // current pass only. The next loop iteration re-evaluates after Phase 1
    // completion polling and the global MIX queue draining (here or on any
    // peer thread). AIC/AIV starvation is therefore bounded by MIX throughput,
    // not unbounded — once mix completes on at least one cluster, the next
    // pass either drains the residual or admits AIC/AIV.
    void dispatch_ready_tasks(
        int32_t thread_idx, CoreTracker &tracker, bool pmu_active, bool &made_progress, bool &try_pushed
    );

    // Shared staging order for both dispatch sources (normal ready + speculative early):
    // MIX strict priority, IDLE stage before PENDING stage, cross-thread idle gating
    // (MIX-IDLE ▶ c/v-IDLE ▶ MIX-PEND ▶ c/v-PEND). `stage(shape, phase)` stages that
    // shape+phase bucket for the source and returns true to STOP the pass (normal returns
    // true when it enters drain mode; early always returns false). `residual_mix()` reports
    // whether MIX work remains queued for the source (normal reads ready_queues[MIX], early
    // reads early_dispatch_queues[MIX]). IDLE runs under PMU; PENDING is withheld under PMU.
    template <typename StageFn, typename ResidualMixFn>
    void run_staging_order(int32_t thread_idx, bool pmu_active, StageFn &&stage, ResidualMixFn &&residual_mix);

    // Returns true if any *other* scheduler thread currently has an idle core
    // matching `shape`. Used as a scheduling hint on the PENDING dispatch path
    // — see the implementation in scheduler_dispatch.cpp for the hint-semantics
    // rationale and the safety argument against the drain worker.
    bool has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const;

    // True if mix tasks remain in the global MIX ready queue. Approximate —
    // PTO2ReadyQueue::size() (see pto_scheduler.h) snapshots its enqueue/dequeue
    // positions with std::memory_order_relaxed and may interleave with concurrent
    // push/pop. A stale read here causes at most one
    // extra/missed AIC/AIV skip and self-corrects on the next loop iteration.
    bool has_residual_mix() const {
        return sched_->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    // Tier-0 analog of has_residual_mix for the ready sync_start lane: true if MIX
    // sync_start cohorts remain queued, so the Tier-0 pass keeps MIX strict priority
    // over its own AIC/AIV sync work. Same relaxed-size snapshot caveat.
    bool has_residual_sync_mix() const {
        return sched_->ready_sync_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    // Early-dispatch analog of has_residual_mix: true if MIX early-dispatch candidates
    // remain queued. has_residual_mix reads the normal MIX ready queue, which is empty
    // whenever the Phase-4b early pass runs (it is gated on all ready_queues being
    // empty), so early-dispatch MIX priority needs its own residual check against
    // early_dispatch_queues[MIX]. Same relaxed-size snapshot caveat as has_residual_mix.
    bool has_residual_early_mix() const {
        return sched_->early_dispatch_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    // =========================================================================
    // Completion & drain (scheduler_completion.cpp)
    // =========================================================================

    static SlotTransition decide_slot_transition(
        int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id, bool pending_gated = false
    );

    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, PTO2SubtaskSlot subslot, int32_t thread_idx,
        int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
        PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
#if SIMPLER_DFX
        ,
        uint64_t dispatch_ts, uint64_t finish_ts
#endif
    );

    static void promote_pending_to_running(CoreExecState &core);
    static void clear_running_slot(CoreExecState &core);

    void check_running_cores_for_completion(
        int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
        bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count
    );

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num);
    int32_t count_global_available(PTO2ResourceShape shape, uint8_t core_mask);
    void drain_worker_dispatch(int32_t block_num);
    void handle_drain_mode(int32_t thread_idx);

    // =========================================================================
    // Cold path: exit checks, stall diagnostics, profiling (scheduler_cold_path.cpp)
    // =========================================================================

    __attribute__((noinline, cold)) LoopAction
    handle_orchestrator_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count);

    __attribute__((noinline, cold)) LoopAction
    check_idle_fatal_error(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime);

    __attribute__((noinline, cold)) void
    log_stall_diagnostics(int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count);

    __attribute__((noinline, cold)) void log_shutdown_stall_snapshot(
        int32_t trigger_thread_idx, int32_t trigger_idle_iterations, int32_t trigger_last_progress_count
    );

    // Reverse lookup: given a global core_id, find which scheduler thread's
    // tracker owns it. Returns -1 if not found. Linear scan — only used on
    // the cold diagnostic path.
    int32_t find_core_owner_thread(int32_t core_id) const;

    // Does this thread own any core with a RUNNING task (running_slot_state set)?
    // Gates the scheduler timeout fatal latch: a thread without an owned
    // RUNNING task has no first-hand evidence of a stuck dispatch and must
    // not declare global fatal on its own idle observation. The thread that
    // does own the stuck task will reach the budget on its own polls and
    // latch with valid evidence (or recover when the COND register flips).
    bool self_owns_running_task(int32_t thread_idx) const;

    // Does *any* scheduler thread own a RUNNING task? Used as the second
    // fatal-latch condition: if the wall-clock budget elapsed AND no thread
    // owns RUNNING work AND tasks remain incomplete, the system is in a
    // pre-dispatch / WAIT-only deadlock (e.g. dependency cycle) and the
    // ownerless idle threads are the only observers — let one of them latch.
    bool no_thread_owns_running_task() const;

    // One-glance classification of a no-progress timeout, derived from state the
    // scheduler already holds at the stall. Reduces the multi-state snapshot to a
    // dominant PTO2_STALL_DETAIL_* sub-class plus a few locator fields, which
    // handle_timeout_exit propagates to host alongside the unchanged code 100.
    struct StallClassification {
        int32_t detail;         // PTO2_STALL_DETAIL_*
        int32_t cnt_running;    // tasks observed RUNNING (on a core)
        int32_t cnt_ready;      // fanin-satisfied but not dispatched
        int32_t cnt_waiting;    // still waiting on fanin
        int32_t completed;      // completed_tasks_ snapshot
        int32_t total;          // total_tasks_ snapshot
        int32_t orch_done;      // orchestrator_done flag (0/1)
        int64_t stuck_task_id;  // S1: first RUNNING task's id (-1 if none)
        int32_t stuck_core;     // S1: core hosting it (-1 if none)
    };

    // Scan the rings once (same ground truth as log_stall_diagnostics: a slot is
    // RUNNING iff a core holds it as running_slot_state) and reduce to a
    // StallClassification. Pure reads — safe to call from any scheduler thread.
    __attribute__((noinline, cold)) StallClassification classify_stall_reason() const;

    __attribute__((noinline, cold)) int32_t handle_timeout_exit(
        int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations,
        int32_t last_progress_count
#if SIMPLER_DFX
        ,
        uint64_t sched_start_ts
#endif
    );

#if SIMPLER_DFX
    __attribute__((noinline, cold)) void log_l2_swimlane_summary(int32_t thread_idx, int32_t cur_thread_completed);
#endif

    // =========================================================================
    // Small inline helpers
    // =========================================================================

    uint64_t get_function_bin_addr(int func_id) const {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d) or map is null", func_id, RUNTIME_MAX_FUNC_ID);
            return 0;
        }
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
