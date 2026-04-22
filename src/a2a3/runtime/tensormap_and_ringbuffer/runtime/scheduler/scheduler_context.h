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

#include "scheduler_types.h"

#include "scheduler/pto_scheduler.h"

// These macros are defined in runtime.h, but we cannot include it here
// (it pulls in Handshake which we only forward-declare).  Mirror the
// authoritative values so the class layout compiles standalone.
#ifndef RUNTIME_MAX_WORKER
#define RUNTIME_MAX_WORKER 72
#endif
#ifndef RUNTIME_MAX_FUNC_ID
#define RUNTIME_MAX_FUNC_ID 32
#endif

// Forward declarations — avoid pulling in full headers for pointer/reference params.
class Runtime;
struct Handshake;

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
    // === Public entry point ===
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    // === State (public for AicpuExecutor init/handshake access during transition) ===

    PTO2SchedulerState *sched_{nullptr};

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered core trackers, one per scheduler thread
    CoreTracker core_trackers_[MAX_AICPU_THREADS];

    // Per-core dispatch payload storage: dual-buffer for pipelining.
    // buf_idx = reg_task_id & 1; adjacent dispatches alternate automatically.
    PTO2DispatchPayload payload_per_core_[RUNTIME_MAX_WORKER][2];

    // sync_start drain coordination
    SyncStartDrainState drain_state_;

#if PTO2_PROFILING
    SchedProfilingCounters sched_perf_[MAX_AICPU_THREADS];
#endif

    // Shared state pointers (set during init, point into AicpuExecutor)
    std::atomic<int32_t> *completed_tasks_ptr_{nullptr};
    int32_t *total_tasks_ptr_{nullptr};
    volatile bool *orchestrator_done_ptr_{nullptr};
    std::atomic<bool> *completed_ptr_{nullptr};
    uint64_t *func_id_to_addr_{nullptr};

    // Core transition state pointers (set during init, point into AicpuExecutor)
    std::atomic<bool> *transition_requested_ptr_{nullptr};
    std::atomic<int32_t> *wait_reassign_ptr_{nullptr};
    std::atomic<bool> *reassigned_ptr_{nullptr};

    // Thread/core configuration
    int32_t active_sched_threads_{0};
    int32_t sched_thread_num_{0};
    bool orch_to_sched_{false};
    int32_t thread_num_{0};
    int32_t *core_count_per_thread_{nullptr};

    // One-time init coordination
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};

    // Emergency shutdown callback (calls AicpuExecutor::emergency_shutdown)
    void (*emergency_shutdown_fn_)(Runtime *runtime){nullptr};

    uint64_t get_function_bin_addr(int func_id) const {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

private:
    // === Completion & drain (scheduler_completion.cpp) ===

    static SlotTransition
    decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id);

    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, PTO2SubtaskSlot subslot, int32_t thread_idx,
        int32_t core_id, Handshake *hank, int32_t &completed_this_turn,
        PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
        ,
        uint64_t dispatch_ts
#endif
    );

    static void promote_pending_to_running(CoreExecState &core);
    static void clear_running_slot(CoreExecState &core);

    void check_running_cores_for_completion(
        int32_t thread_idx, Handshake *hank, int32_t &completed_this_turn, int32_t &cur_thread_completed,
        bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[], int32_t &deferred_release_count,
        PTO2LocalReadyBuffer *local_bufs
    );

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num);
    int32_t count_global_available(PTO2ResourceShape shape);
    void drain_worker_dispatch(Runtime *runtime, int32_t block_num);
    void handle_drain_mode(Runtime *runtime, int32_t thread_idx);

    // === Cold path (scheduler_cold_path.cpp) ===

    __attribute__((noinline, cold)) LoopAction
    handle_orchestrator_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count);

    __attribute__((noinline, cold)) LoopAction handle_core_transition(bool &cores_released);

    __attribute__((noinline, cold)) LoopAction
    check_idle_fatal_error(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime);

    __attribute__((noinline, cold)) void
    log_stall_diagnostics(int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count);

    __attribute__((noinline, cold)) int32_t handle_timeout_exit(
        int32_t thread_idx, int32_t idle_iterations
#if PTO2_PROFILING
        ,
        uint64_t sched_start_ts
#endif
    );

#if PTO2_PROFILING
    __attribute__((noinline, cold)) void log_profiling_summary(int32_t thread_idx, int32_t cur_thread_completed);
#endif

    // === Dispatch (scheduler_dispatch.cpp) ===

    static const char *shape_name(PTO2ResourceShape shape);
    static const PTO2ResourceShape *get_dispatch_order(int32_t thread_idx);

    int pop_ready_tasks_batch(
        PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out,
        int max_count
    );

    void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot);

    void dispatch_subtask_to_core(
        Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state,
        PTO2SubtaskSlot subslot, bool to_pending
    );

    void dispatch_mix_block_to_cluster(
        Runtime *runtime, int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending
    );

    void dispatch_block(
        Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state,
        PTO2ResourceShape shape, bool to_pending
    );

    void dispatch_shape(
        Runtime *runtime, int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
        PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress,
        bool &try_pushed
    );
};

#endif  // SCHEDULER_CONTEXT_H
