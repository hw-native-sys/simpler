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
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "common/l2_swimlane_profiling.h"
#include "scheduler/scheduler_types.h"

#include "scheduler/pto_scheduler.h"

#include "aicore_completion_mailbox.h"
#include "pto2_dispatch_payload.h"

#include <cinttypes>
#include <cstdio>
#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "aicpu/device_time.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "spin_hint.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

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

class SchedulerContext {
public:
    // Init is split into three parts so the per-core AICore handshake — a
    // serial, MMIO-bound loop that dominates preamble (~217 µs of ~283 µs for
    // 72 cores) — can run in parallel across all AICPU threads. The leader
    // (exec_idx 0) runs pre_handshake_init, then every thread handshakes a
    // disjoint slice of cores via handshake_partition, then the leader runs
    // post_handshake_init after a barrier. See AicpuExecutor::init.
    //
    // Leader-only: per-core state + config + swimlane buffers + core count.
    // Must be published before any thread enters handshake_partition.
    int32_t
    pre_handshake_init(Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, uint64_t regs_base);

    // All threads: handshake this thread's contiguous slice [lo, hi) of cores.
    // Each core is touched by exactly one thread (contiguous, gap-free
    // partition), so core_exec_states_ writes are race-free. The AIC/AIV
    // worker-id lists are built serially in post_handshake_init to preserve the
    // core-index ordering that assign_cores_to_threads relies on.
    void handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads);

    // Barrier-free init (multi-thread): a scheduler thread handshakes only
    // the cores it will later dispatch to — clusters ci with ci % active_threads ==
    // tidx, cluster ci = {ci, aic_n+2ci, aic_n+2ci+1} in the blocked layout
    // ([0,aic_n) AIC, [aic_n,3*aic_n) AIV). Same per-core handshake as
    // handshake_partition, over the owned set instead of a contiguous slice, so
    // core_exec_states_ writes stay race-free (each core owned by one thread).
    void handshake_owned_clusters(Runtime *runtime, int32_t tidx, int32_t active_threads) {
        Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->dev.workers);
        const int32_t aic_n = cores_total_num_ / 3;

        int32_t owned[RUNTIME_MAX_WORKER];
        int32_t own_n = 0;
        for (int32_t ci = tidx; ci < aic_n; ci += active_threads) {
            owned[own_n++] = ci;                  // AIC
            owned[own_n++] = aic_n + 2 * ci;      // AIV0
            owned[own_n++] = aic_n + 2 * ci + 1;  // AIV1
        }

        // Batched 4-phase handshake (adopts #1345's batching, keeping polling's
        // aicpu_ready release since the AICore waits on it before reporting): tasks
        // / windows / states are each published in one pass so posted MMIO STRs and
        // GM stores don't serialize, and only ~4 barriers fire for the whole owned
        // set instead of ~2 per core.

        // Phase 1: publish every task pointer (one barrier so all are visible before
        // any release), then release every owned core (aicpu_ready=1). The AICore
        // reads its task only after observing aicpu_ready.
        for (int32_t k = 0; k < own_n; k++)
            all_handshakes[owned[k]].task = reinterpret_cast<uint64_t>(&payload_per_core_[owned[k]][0]);
        OUT_OF_ORDER_STORE_BARRIER();
        for (int32_t k = 0; k < own_n; k++)
            all_handshakes[owned[k]].aicpu_ready = 1;
        OUT_OF_ORDER_STORE_BARRIER();

        uint32_t max_physical_cores_count = platform_get_physical_cores_count();
        uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);

        struct ReadyCore {
            int32_t i;
            uint32_t pcid;
            uint64_t reg_addr;
            CoreType core_type;
        };
        ReadyCore ready[RUNTIME_MAX_WORKER];
        int32_t n_ready = 0;
        bool core_serviced[RUNTIME_MAX_WORKER] = {false};

        // Phase 2: collect every owned core's report (spin until all done),
        // prefetching each CoreExecState line for the write pass.
        for (int32_t remaining = own_n; remaining > 0;) {
            for (int32_t k = 0; k < own_n; k++) {
                int32_t i = owned[k];
                if (core_serviced[i]) continue;
                Handshake *hank = &all_handshakes[i];
                if (hank->aicore_done == 0) {
                    SPIN_WAIT_HINT();
                    continue;
                }
                uint32_t physical_core_id = hank->physical_core_id;
                if (physical_core_id >= max_physical_cores_count) {
                    handshake_failed_.store(true, std::memory_order_release);
                    core_serviced[i] = true;
                    remaining--;
                    continue;
                }
                __builtin_prefetch(&core_exec_states_[i], 1, 3);
                ready[n_ready++] = {i, physical_core_id, regs[physical_core_id], hank->core_type};
                core_serviced[i] = true;
                remaining--;
            }
        }

        // Phase 3: open every core's register window, then ONE barrier.
        for (int32_t r = 0; r < n_ready; r++)
            platform_init_aicore_regs(ready[r].reg_addr);
        OUT_OF_ORDER_STORE_BARRIER();

        // Phase 4: publish each CoreExecState (AICPU-private, may follow the
        // windows). running/pending_reg_task_id start INVALID: pre_handshake_init
        // memset them to 0, which is a valid task id (AICPU_TASK_INVALID, the idle
        // sentinel, is not 0), so without this reset the scheduler reads every core
        // as "running task 0", never dispatches, and trips SCHEDULER_TIMEOUT.
        for (int32_t r = 0; r < n_ready; r++) {
            int32_t i = ready[r].i;
            core_exec_states_[i].reg_addr = ready[r].reg_addr;
            core_exec_states_[i].cond_ptr = get_reg_ptr(ready[r].reg_addr, RegId::COND);
            core_exec_states_[i].worker_id = i;
            core_exec_states_[i].physical_core_id = ready[r].pcid;
            core_exec_states_[i].core_type = ready[r].core_type;
            core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }
        OUT_OF_ORDER_STORE_BARRIER();
    }

    // Barrier-free counterpart of post_handshake_init's assignment: thread tidx
    // populates its own CoreTracker + per-owned-core sub_block_id right after
    // handshaking its clusters — no all-thread barrier, no leader serialization.
    // The blocked layout gives the owned clusters' worker ids directly, so no
    // aic_worker_ids_ discovery is needed. AsyncCtx/slab pointers are set per
    // dispatch by build_payload (as on the barrier path), so only the tracker and
    // the one-time sub_block_id are set here.
    void assign_own_clusters(int32_t tidx) {
        const int32_t aic_n = cores_total_num_ / 3;
        const int32_t active = active_sched_threads_;

        CoreTracker &tracker = core_trackers_[tidx];
        int32_t own_n = 0;
        for (int32_t ci = tidx; ci < aic_n; ci += active)
            own_n++;
        tracker.init(own_n);

        int32_t local = 0;
        for (int32_t ci = tidx; ci < aic_n; ci += active)
            tracker.set_cluster(local++, ci, aic_n + 2 * ci, aic_n + 2 * ci + 1);

        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
            int32_t cluster_offset = c * 3;
            int32_t aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
            int32_t aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
            payload_per_core_[aiv0_id][0].global_context.sub_block_id = 0;
            payload_per_core_[aiv0_id][1].global_context.sub_block_id = 0;
            payload_per_core_[aiv1_id][0].global_context.sub_block_id = 1;
            payload_per_core_[aiv1_id][1].global_context.sub_block_id = 1;
        }
    }

    // Latch completion + broadcast exit on a handshake failure seen without the
    // all-thread barrier (barrier-free path). Idempotent.
    void abort_and_shutdown(Runtime *runtime) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
    }

    bool handshake_failed() const { return handshake_failed_.load(std::memory_order_acquire); }

    // Leader-only, after the handshake barrier: build worker-id lists, assign
    // cores to threads, read task counts, init dispatch payloads.
    int32_t post_handshake_init(Runtime *runtime);

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit();

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx);

    int32_t shutdown(int32_t thread_idx);

    // Upstream-compatible overload: signature is (runtime, rt, thread_idx, total_tasks).
    // thread_idx is ignored — polling scheduler's bookkeeping is thread-agnostic at
    // this point.
    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t /*thread_idx*/, int32_t total_tasks);

    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t total_tasks);

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt);

    // Serial orch->sched mode pre-dispatch gate. Spin until the orchestrator
    // marks itself done; thread 0 may drain the wiring SPSC in the meantime
    // so the orchestrator's submit_task pushes don't back-pressure. Other
    // threads idle on the orchestrator_done_ flag.
    void wait_for_orchestration_done_before_dispatch(Runtime * /*runtime*/, int32_t thread_idx);

    int32_t aic_count() const { return aic_count_; }
    int32_t aiv_count() const { return aiv_count_; }
    bool is_completed() const { return completed_.load(std::memory_order_acquire); }
    int32_t completed_tasks_count() const { return completed_tasks_.load(std::memory_order_acquire); }

    // Block until the first scheduler thread has finished one-time PTO2 init.
    // Called by the orchestrator thread in device-orch mode.
    void wait_pto2_init_complete() const {
        while (!pto2_init_complete_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();
    }

private:
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

    DeferredCompletionSlab deferred_slab_per_core_[RUNTIME_MAX_WORKER][2];

    // sync_start drain coordination
    SyncStartDrainState drain_state_;

    // --- Task-execution tracking ---
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
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

    // Platform AICore-register base array (set by AicpuExecutor before init()).
    uint64_t regs_{0};

    // --- One-time init coordination ---
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};

    // Set by any thread whose slice hits an invalid physical_core_id in
    // handshake_partition; checked by the leader in post_handshake_init.
    std::atomic<bool> handshake_failed_{false};

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads();

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime);

    static const char *shape_name(PTO2ResourceShape shape);

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
        PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
    );

    void build_payload(
        PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        const AsyncCtx &async_ctx, int32_t block_idx
    );

    struct PublishHandle {
        uint64_t reg_addr;
        uint32_t reg_task_id;
        int32_t core_offset;
        uint64_t *dispatch_timestamp_slot;
        int32_t task_timing_slot;  // TASK_TIMING_SLOT_NONE unless the task is tagged
    };

    SchedulerContext::PublishHandle prepare_subtask_to_core(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
        bool to_pending, int32_t block_idx
    );

    // `thread_idx` selects the publishing Scheduler thread's per-thread task-timing
    // record; every call site already has it in scope.
    inline void publish_subtask_to_core(const PublishHandle &h, uint64_t dispatch_ts, int32_t thread_idx) {
        if (h.dispatch_timestamp_slot != nullptr) *h.dispatch_timestamp_slot = dispatch_ts;
        // Task-timing dispatch: earliest DATA_MAIN_BASE publication for a tagged
        // task, folded as min. Untagged tasks pay only this cache-hot compare.
        if (h.task_timing_slot != TASK_TIMING_SLOT_NONE) aicpu_task_timing_dispatch(h.task_timing_slot, thread_idx);
        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));
    }

    // Fan out one block's subtasks (1 for AIC/AIV, 1-3 for MIX) into the
    // caller-supplied handles buffer. Returns the number of handles written.
    int prepare_block_for_dispatch(
        int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape,
        bool to_pending, int32_t block_idx, PublishHandle *out_handles
    );

    void dispatch_shape(
        int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf,
        CoreTracker &tracker, bool &entered_drain, bool &made_progress
    );

    void dispatch_ready_tasks(
        int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES],
        bool pmu_active, bool &made_progress
    );

    bool has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const;

    bool has_residual_mix(const PTO2LocalReadyBuffer &mix_local_buf) const {
        return mix_local_buf.count > 0 || sched_->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    static SlotTransition
    decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id);

    void complete_slot_task(
        PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, int32_t core_id, int32_t &completed_this_turn
    );

    static void promote_pending_to_running(CoreExecState &core);

    static void clear_running_slot(CoreExecState &core);

    void check_running_cores_for_completion(
        int32_t thread_idx, int32_t &completed_this_turn, int32_t &cur_thread_completed, bool &made_progress
    );

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num);

    int32_t count_global_available(PTO2ResourceShape shape);

    void drain_worker_dispatch(int32_t block_num);

    void handle_drain_mode(int32_t thread_idx);

    LoopAction handle_orchestrator_exit(PTO2SharedMemoryHeader *header, Runtime *runtime);

    LoopAction check_idle_fatal_error(PTO2SharedMemoryHeader *header, Runtime *runtime);

    void log_stall_diagnostics(int32_t thread_idx);

    void log_shutdown_stall_snapshot();

    int32_t find_core_owner_thread(int32_t core_id) const;

    bool self_owns_running_task(int32_t thread_idx) const;

    bool no_thread_owns_running_task() const;

    int32_t handle_timeout_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime);

    uint64_t get_function_bin_addr(int func_id) const {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
