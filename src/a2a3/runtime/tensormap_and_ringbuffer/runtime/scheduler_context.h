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
#include "scheduler_types.h"

#include "pto_scheduler.h"

#include "aicore_completion_mailbox.h"
#include "pto2_dispatch_payload.h"

#include <cinttypes>
#include <cstdio>
#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "aicpu/device_time.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "spin_hint.h"
// SchedulerThreadProfile is defined in scheduler_types.h (above) so the
// drain_wiring_queue method in pto_scheduler.h can take a pointer to it.

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

inline void latch_scheduler_error(PTO2SharedMemoryHeader *header, int32_t thread_idx, int32_t error_code)
{
    if (header == nullptr || error_code == PTO2_ERROR_NONE) return;
    int32_t expected = PTO2_ERROR_NONE;
    if (header->sched_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) header->sched_error_thread.store(thread_idx, std::memory_order_release);
    if (thread_idx >= 0 && thread_idx < 32) header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
}

inline void format_core_status(char *buf, size_t buf_size, int32_t core_id, bool idle, const CoreExecState *core_state, uint64_t reg_addr_for_cond)
{
    if (idle)
    {
        snprintf(buf, buf_size, "core%d(idle)", core_id);
        return;
    }
    int32_t kernel = -1;
    int64_t task_id_raw = -1;
    if (core_state && core_state->running_slot_state)
    {
        int32_t subslot = static_cast<int32_t>(core_state->running_subslot);
        kernel = core_state->running_slot_state->task->kernel_id[subslot];
        task_id_raw = static_cast<int64_t>(core_state->running_slot_state->task->task_id.raw);
    }
    uint64_t cond_reg = read_reg(reg_addr_for_cond, RegId::COND);
    int32_t hw_state = EXTRACT_TASK_STATE(cond_reg);
    const char *cond_reg_state_str = (hw_state == TASK_ACK_STATE) ? "ack" : "fin";
    if (hw_state == TASK_ACK_STATE) snprintf(buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s)", core_id, kernel, task_id_raw, cond_reg_state_str);
    else snprintf(buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s ANOMALY)", core_id, kernel, task_id_raw, cond_reg_state_str);
}

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

class SchedulerContext
{
public:
    int32_t init(Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, bool orch_to_sched, uint64_t regs_base)
    {
        always_assert(runtime != nullptr);

        // Zero all per-core execution state before handshake
        memset(core_exec_states_, 0, sizeof(core_exec_states_));

        // Wire thread/transition configuration that handshake/assign need to read.
        aicpu_thread_num_ = aicpu_thread_num;
        sched_thread_num_ = sched_thread_num;
        orch_to_sched_ = orch_to_sched;
        regs_ = regs_base;

        // Discover cores and assign to scheduler threads.
        int32_t rc = handshake_all_cores(runtime);
        if (rc != 0) return rc;
        if (!assign_cores_to_threads()) return -1;

        // Initialize task counters. Task count comes from PTO2 shared memory.
        if (runtime->get_gm_sm_ptr())
        {
            auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
            int64_t pto2_count = 0;
            for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            {
                int32_t ring_tasks = header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                if (ring_tasks > 0 && ring_tasks <= PTO2_SCOPE_TASKS_CAP) pto2_count += ring_tasks;
            }
            total_tasks_ = static_cast<int32_t>(pto2_count);
        }
        else
        {
            total_tasks_ = 0;
        }
        completed_tasks_.store(0, std::memory_order_release);

        // Device orchestration: the orchestrator thread flips this when the graph is built.
        orchestrator_done_ = false;

        // Clear per-core dispatch payloads
        memset(payload_per_core_, 0, sizeof(payload_per_core_));
        memset(deferred_slab_per_core_, 0, sizeof(deferred_slab_per_core_));

        // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
        // This is done once at startup and never modified afterwards.
        for (int32_t t = 0; t < sched_thread_num_; t++)
        {
            CoreTracker &tracker = core_trackers_[t];
            for (int32_t c = 0; c < tracker.get_cluster_count(); c++)
            {
                int32_t cluster_offset = c * 3;  // Each cluster = 1 AIC + 2 AIV
                auto aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
                auto aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
                payload_per_core_[aiv0_id][0].global_context.sub_block_id = 0;
                payload_per_core_[aiv0_id][1].global_context.sub_block_id = 0;
                payload_per_core_[aiv1_id][0].global_context.sub_block_id = 1;
                payload_per_core_[aiv1_id][1].global_context.sub_block_id = 1;
            }
        }

        func_id_to_addr_ = runtime->func_id_to_addr_;

        return 0;
    }

    // Reset all SchedulerContext-owned state to its post-construction defaults.
    // Called by AicpuExecutor::deinit() during per-run teardown.
    void deinit()
    {
        // Reset all per-core execution state
        for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++)
        {
            core_exec_states_[i] = {};
            core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }

        // Clear per-core dispatch payloads
        memset(payload_per_core_, 0, sizeof(payload_per_core_));
        memset(deferred_slab_per_core_, 0, sizeof(deferred_slab_per_core_));

        // Reset sync-start drain coordination — a previous run that aborted mid-drain
        // would otherwise leave dirty pending/elected/ack state for the next reuse.
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
        drain_state_.drain_worker_elected.store(0, std::memory_order_release);
        drain_state_.drain_ack_mask.store(0, std::memory_order_release);
        drain_state_.pending_task.store(nullptr, std::memory_order_release);

        // Reset task counters and orchestrator state
        completed_tasks_.store(0, std::memory_order_release);
        total_tasks_ = 0;
        orchestrator_done_ = false;
        pto2_init_done_.store(false, std::memory_order_release);
        pto2_init_complete_.store(false, std::memory_order_release);

        // Reset core transition state
        transition_requested_.store(false, std::memory_order_release);
        wait_reassign_.store(0, std::memory_order_release);
        reassigned_.store(false, std::memory_order_release);
        completed_.store(false, std::memory_order_release);

        // Reset core discovery and assignment state
        aic_count_ = 0;
        aiv_count_ = 0;
        cores_total_num_ = 0;
        aicpu_thread_num_ = 0;
        sched_thread_num_ = 0;
        orch_to_sched_ = false;
        active_sched_threads_ = 0;
        for (int32_t t = 0; t < MAX_AICPU_THREADS; t++) core_trackers_[t] = CoreTracker{};

        regs_ = 0;
        sched_ = nullptr;
        rt_ = nullptr;
        func_id_to_addr_ = nullptr;
    }

    // Main scheduler thread entry: poll completion + dispatch ready tasks.
    int32_t resolve_and_dispatch(Runtime *runtime, int32_t thread_idx)
    {
        always_assert(sched_ != nullptr);
        CoreTracker &tracker = core_trackers_[thread_idx];

        PTO2SharedMemoryHeader *header = sched_->sm_header;
        if (!header) return -1;

        // One-time init: assign perf buffers (one thread does it; others wait)
        if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) pto2_init_complete_.store(true, std::memory_order_release);
        else
            while (!pto2_init_complete_.load(std::memory_order_acquire)) SPIN_WAIT_HINT();

        int32_t cur_thread_completed = 0;
        int32_t idle_iterations = 0;

        constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
        PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
        PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
        for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);

        bool cores_released = false;

        const bool pmu_active = is_pmu_enabled();

        uint64_t last_progress_ts = get_sys_cnt_aicpu();

        // Profile reset + total-cycle start. Reset here so each
        // resolve_and_dispatch call (≈ one kernel launch) records its own
        // breakdown. The dump happens at loop exit, well outside the hot path.
        SchedulerThreadProfile &profile = thread_profiles_[thread_idx];
        profile.reset();
        const uint64_t profile_loop_start = get_sys_cnt_aicpu();

        while (true)
        {
            if (completed_.load(std::memory_order_acquire)) break;
            bool made_progress = false;
            profile.total_iters++;
            if (!tracker.has_any_running_cores())
            {
                LoopAction action = handle_orchestrator_exit(header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (!cores_released && orch_to_sched_)
            {
                LoopAction action = handle_core_transition(cores_released);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            // Phase 1: Check running cores for completion
            int32_t completed_this_turn = 0;

            if (tracker.has_any_running_cores())
            {
                uint64_t t0 = get_sys_cnt_aicpu();
                check_running_cores_for_completion(thread_idx, completed_this_turn, cur_thread_completed, made_progress);
                profile.completion_cycles += get_sys_cnt_aicpu() - t0;
                profile.completion_iters++;
            }
            if (completed_this_turn > 0)
            {
                completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            }

            uint64_t t0_async = 0;
            if (rt_ != nullptr && rt_->aicore_mailbox != nullptr && (sched_->async_wait_list.count > 0 || rt_->aicore_mailbox->has_pending()))
            {
                t0_async = get_sys_cnt_aicpu();
                AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(rt_->aicore_mailbox, sched_);
                if (poll_result.error_code != PTO2_ERROR_NONE)
                {
                    int32_t expected = PTO2_ERROR_NONE;
                    header->sched_error_code.compare_exchange_strong(expected, poll_result.error_code, std::memory_order_acq_rel, std::memory_order_acquire);
                    completed_.store(true, std::memory_order_release);
                    break;
                }
                if (poll_result.completed > 0)
                {
                    completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                    made_progress = true;
                }
                profile.async_wait_cycles += get_sys_cnt_aicpu() - t0_async;
                profile.async_wait_iters++;
            }

            // Phase 2 drain check
            if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0)
            {
                handle_drain_mode(thread_idx);
                continue;
            }

            // Phase 3: Drain wiring queue (thread 0 only). Pass cumulative
            // sub-phase counters (SPSC drain stage 1 / pending-FIFO poll
            // stage 2) so drain_wiring_queue accumulates into them.
            if (thread_idx == 0)
            {
                uint64_t t0 = get_sys_cnt_aicpu();
                int wired = sched_->drain_wiring_queue(orchestrator_done_,
                    &profile.spsc_drain_cycles, &profile.spsc_drain_iters,
                    &profile.pending_poll_cycles, &profile.pending_poll_iters);
                if (wired > 0) made_progress = true;
                profile.drain_wiring_cycles += get_sys_cnt_aicpu() - t0;
                profile.drain_wiring_iters++;
            }

            if (thread_idx == 0)
            {
                uint64_t t0 = get_sys_cnt_aicpu();
                constexpr int DUMMY_DRAIN_BATCH = 16;
                PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
                int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
                for (int di = 0; di < dummy_got; di++)
                {
                    PTO2TaskSlotState &dummy_slot = *dummy_batch[di];
                    sched_->on_mixed_task_complete(dummy_slot);
                    completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                    cur_thread_completed++;
                }
                if (dummy_got > 0) made_progress = true;
                profile.dummy_drain_cycles += get_sys_cnt_aicpu() - t0;
                profile.dummy_drain_iters++;
            }

            // Phase 4: MIX-strict-priority dispatch with phase-split and
            // cross-thread idle gating. See dispatch_ready_tasks for the policy.
            {
                uint64_t t0 = get_sys_cnt_aicpu();
                dispatch_ready_tasks(thread_idx, tracker, local_bufs, pmu_active, made_progress);
                profile.dispatch_cycles += get_sys_cnt_aicpu() - t0;
                profile.dispatch_iters++;
            }

            if (made_progress)
            {
                idle_iterations = 0;
                last_progress_ts = get_sys_cnt_aicpu();
            }
            else
            {
                uint64_t t0_idle = get_sys_cnt_aicpu();
                idle_iterations++;

                if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0)
                {
                    LoopAction action = check_idle_fatal_error(header, runtime);
                    if (action == LoopAction::BREAK_LOOP) break;
                }

                if (idle_iterations % STALL_LOG_INTERVAL == 0) log_stall_diagnostics(thread_idx);
                if (get_sys_cnt_aicpu() - last_progress_ts > SCHEDULER_TIMEOUT_CYCLES)
                {
                    bool self_owns = self_owns_running_task(thread_idx);
                    bool global_stuck = !self_owns && total_tasks_ > 0 && completed_tasks_.load(std::memory_order_relaxed) < total_tasks_ && no_thread_owns_running_task();
                    if (self_owns || global_stuck) return handle_timeout_exit(thread_idx, header, runtime);
                    last_progress_ts = get_sys_cnt_aicpu();
                }
                SPIN_WAIT_HINT();
                profile.idle_spin_cycles += get_sys_cnt_aicpu() - t0_idle;
                profile.idle_iters++;
            }
        }

        // Dump profile breakdown for this thread. Logged AFTER the hot loop
        // exits, so this adds no overhead to the measured phases.
        profile.total_cycles = get_sys_cnt_aicpu() - profile_loop_start;
        LOG_INFO_V9(
            "CLAUDE_PROFILING thread=%d total_cyc=%lu iters=%lu compl_cyc=%lu compl_n=%lu ctask_cyc=%lu ctask_n=%lu cores_scan=%lu async_cyc=%lu async_n=%lu drain_cyc=%lu drain_n=%lu spsc_cyc=%lu spsc_n=%lu poll_cyc=%lu poll_n=%lu poll_skipped=%lu dummy_cyc=%lu dummy_n=%lu dispatch_cyc=%lu dispatch_n=%lu idle_cyc=%lu idle_n=%lu",
            (int)thread_idx,
            (unsigned long)profile.total_cycles, (unsigned long)profile.total_iters,
            (unsigned long)profile.completion_cycles, (unsigned long)profile.completion_iters,
            (unsigned long)profile.complete_task_cycles, (unsigned long)profile.complete_task_calls,
            (unsigned long)profile.cores_scanned,
            (unsigned long)profile.async_wait_cycles, (unsigned long)profile.async_wait_iters,
            (unsigned long)profile.drain_wiring_cycles, (unsigned long)profile.drain_wiring_iters,
            (unsigned long)profile.spsc_drain_cycles, (unsigned long)profile.spsc_drain_iters,
            (unsigned long)profile.pending_poll_cycles, (unsigned long)profile.pending_poll_iters,
            (unsigned long)profile.pending_poll_skipped,
            (unsigned long)profile.dummy_drain_cycles, (unsigned long)profile.dummy_drain_iters,
            (unsigned long)profile.dispatch_cycles, (unsigned long)profile.dispatch_iters,
            (unsigned long)profile.idle_spin_cycles, (unsigned long)profile.idle_iters);

        return cur_thread_completed;
    }

    int32_t shutdown(int32_t thread_idx)
    {
        const int32_t *cores = core_trackers_[thread_idx].core_ids();
        int32_t core_num = core_trackers_[thread_idx].core_num();
        if (core_num == 0) return 0;

        int32_t rc = 0;
        for (int32_t i = 0; i < core_num; i++)
        {
            int32_t core_id = cores[i];
            uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
            if (reg_addr != 0)
            {
                // Timeout means AICore is unresponsive. Log and continue deiniting remaining cores.
                if (platform_deinit_aicore_regs(reg_addr) != 0) rc = -1;
            }
            else
            {}
        }
        return rc;
    }

    void on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t total_tasks)
    {
        total_tasks_ = total_tasks;

        // Fold tasks completed inline during orchestration
        int32_t inline_completed = static_cast<int32_t>(rt->orchestrator.inline_completed_tasks);
        if (inline_completed > 0) completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
        orchestrator_done_ = true;

        // Check for fatal error from orchestration; if so, shut down immediately.
        int32_t orch_err = 0;
        if (sched_->sm_header) orch_err = sched_->sm_header->orch_error_code.load(std::memory_order_relaxed);
        if (orch_err != PTO2_ERROR_NONE)
        {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
        }

        // Skip core transition on fatal error — cores already shut down above.
        if (completed_.load(std::memory_order_acquire))
        {
            // Signal transition to unblock scheduler threads waiting at core transition
            transition_requested_.store(true, std::memory_order_release);
            reassigned_.store(true, std::memory_order_release);
        }
        else if (orch_to_sched_)
        {
            transition_requested_.store(true, std::memory_order_release);

            // Wait for scheduler threads to acknowledge transition request
            while (wait_reassign_.load(std::memory_order_acquire) != sched_thread_num_)
            {
                if (completed_.load(std::memory_order_acquire)) break;
                SPIN_WAIT_HINT();
            }
            if (!completed_.load(std::memory_order_acquire))
            {
                reassign_cores_for_all_threads();
                reassigned_.store(true, std::memory_order_release);
            }
        }
    }

    // Bind the PTO2Runtime scheduler pointer. Required in device-orchestration
    // mode where rt is created by the orchestrator thread after init().
    void bind_runtime(PTO2Runtime *rt)
    {
        rt_ = rt;
        sched_ = &rt->scheduler;
    }

    int32_t aic_count() const
    {
        return aic_count_;
    }
    int32_t aiv_count() const
    {
        return aiv_count_;
    }
    bool is_completed() const
    {
        return completed_.load(std::memory_order_acquire);
    }
    int32_t completed_tasks_count() const
    {
        return completed_tasks_.load(std::memory_order_acquire);
    }

    // Block until the first scheduler thread has finished one-time PTO2 init.
    // Called by the orchestrator thread in device-orch mode.
    void wait_pto2_init_complete() const
    {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) SPIN_WAIT_HINT();
    }

private:
    // --- Scheduler binding & per-core runtime state ---
    alignas(64) PTO2SchedulerState *sched_{nullptr};
    PTO2Runtime *rt_{nullptr};

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered core trackers, one per scheduler thread
    CoreTracker core_trackers_[MAX_AICPU_THREADS];
    SchedulerThreadProfile thread_profiles_[MAX_AICPU_THREADS];

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

    // --- Core-transition coordination ---
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};

    // --- Thread/core configuration ---
    int32_t active_sched_threads_{0};
    int32_t sched_thread_num_{0};
    bool orch_to_sched_{false};
    int32_t aicpu_thread_num_{0};
    int32_t cores_total_num_{0};

    // Cluster-ordered worker_id lists, populated by handshake_all_cores().
    int32_t aic_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aiv_worker_ids_[RUNTIME_MAX_WORKER]{};
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform AICore-register base array (set by AicpuExecutor before init()).
    uint64_t regs_{0};

    // --- One-time init coordination ---
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};

    // Handshake with all AICore workers; populates core_exec_states_, worker id lists.
    int32_t handshake_all_cores(Runtime *runtime)
    {
        Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
        cores_total_num_ = runtime->worker_count;

        // Validate cores_total_num_ before using as array index
        if (cores_total_num_ == 0 || cores_total_num_ > RUNTIME_MAX_WORKER) return -1;

        aic_count_ = 0;
        aiv_count_ = 0;

        for (int32_t i = 0; i < cores_total_num_; i++)
        {
            all_handshakes[i].task = reinterpret_cast<uint64_t>(&payload_per_core_[i][0]);
            OUT_OF_ORDER_STORE_BARRIER();
            all_handshakes[i].aicpu_ready = 1;
        }
        OUT_OF_ORDER_STORE_BARRIER();

        // Get platform physical cores count for validation
        uint32_t max_physical_cores_count = platform_get_physical_cores_count();

        // Step 2: Wait for all cores to respond, collect core type and register addresses
        bool handshake_failed = false;
        for (int32_t i = 0; i < cores_total_num_; i++)
        {
            Handshake *hank = &all_handshakes[i];

            while (hank->aicore_regs_ready == 0) SPIN_WAIT_HINT();

            uint32_t physical_core_id = hank->physical_core_id;

            if (physical_core_id >= max_physical_cores_count)
            {
                handshake_failed = true;
                continue;
            }

            uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);
            uint64_t reg_addr = regs[physical_core_id];

            // Initialize AICore registers after discovery (first round)
            platform_init_aicore_regs(reg_addr);
            OUT_OF_ORDER_STORE_BARRIER();
            hank->aicpu_regs_ready = 1;

            OUT_OF_ORDER_STORE_BARRIER();

            while (hank->aicore_done == 0) SPIN_WAIT_HINT();

            CoreType type = hank->core_type;

            core_exec_states_[i].reg_addr = reg_addr;
            core_exec_states_[i].cond_ptr = get_reg_ptr(reg_addr, RegId::COND);

            core_exec_states_[i].worker_id = i;
            core_exec_states_[i].physical_core_id = physical_core_id;
            core_exec_states_[i].core_type = type;

            if (type == CoreType::AIC) aic_worker_ids_[aic_count_++] = i;
            else aiv_worker_ids_[aiv_count_++] = i;
        }

        if (handshake_failed)
        {
            emergency_shutdown(runtime);
            return -1;
        }

        return 0;
    }

    // Assign discovered cores (cluster = 1 AIC + 2 AIV) round-robin across scheduler threads.
    bool assign_cores_to_threads()
    {
        // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
        // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
        active_sched_threads_ = (sched_thread_num_ > 0) ? sched_thread_num_ : aicpu_thread_num_;
        int32_t cluster_count = aic_count_;

        // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
        int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
        int32_t thread_cores_num = max_clusters_per_thread * 3;

        if (thread_cores_num > CoreTracker::MAX_CORE_PER_THREAD) return false;

        for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++)
        {
            core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
            core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
        }

        // Count clusters per thread first (round-robin may distribute unevenly)
        int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
        for (int32_t ci = 0; ci < cluster_count; ci++) clusters_per_thread[ci % active_sched_threads_]++;
        for (int32_t i = 0; i < active_sched_threads_; i++) core_trackers_[i].init(clusters_per_thread[i]);

        int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

        for (int32_t ci = 0; ci < cluster_count; ci++)
        {
            int32_t t = ci % active_sched_threads_;

            int32_t aic_wid = aic_worker_ids_[ci];
            int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
            int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

            core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);
        }

        for (int32_t t = 0; t < aicpu_thread_num_; t++)
        {}

        return true;
    }

    // Re-distribute all cores across all threads after orchestration completes.
    void reassign_cores_for_all_threads()
    {
        // Collect running worker_ids from all current trackers
        bool running_cores[RUNTIME_MAX_WORKER] = {};
        for (int32_t i = 0; i < aicpu_thread_num_; i++)
        {
            auto all_running = core_trackers_[i].get_all_running_cores();
            int32_t bp;
            while ((bp = all_running.pop_first()) >= 0) running_cores[core_trackers_[i].get_core_id_by_offset(bp)] = true;
        }

        // Count clusters per thread (round-robin across all threads)
        int32_t cluster_count = aic_count_;
        int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
        for (int32_t ci = 0; ci < cluster_count; ci++) clusters_per_thread[ci % aicpu_thread_num_]++;

        // Re-init all trackers and reset core counts
        for (int32_t i = 0; i < aicpu_thread_num_; i++) core_trackers_[i].init(clusters_per_thread[i]);

        // Assign clusters round-robin and restore running state
        int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};
        for (int32_t ci = 0; ci < cluster_count; ci++)
        {
            int32_t t = ci % aicpu_thread_num_;

            int32_t aic_wid = aic_worker_ids_[ci];
            int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
            int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

            int32_t cl_idx = cluster_idx_per_thread[t]++;
            core_trackers_[t].set_cluster(cl_idx, aic_wid, aiv0_wid, aiv1_wid);

            // init() marks all idle; toggle cores that were running and restore pending_occupied
            if (running_cores[aic_wid])
            {
                core_trackers_[t].change_core_state(cl_idx * 3);
                core_trackers_[t].set_pending_occupied(cl_idx * 3);
            }
            if (running_cores[aiv0_wid])
            {
                core_trackers_[t].change_core_state(cl_idx * 3 + 1);
                core_trackers_[t].set_pending_occupied(cl_idx * 3 + 1);
            }
            if (running_cores[aiv1_wid])
            {
                core_trackers_[t].change_core_state(cl_idx * 3 + 2);
                core_trackers_[t].set_pending_occupied(cl_idx * 3 + 2);
            }
        }

        active_sched_threads_ = aicpu_thread_num_;
    }

    // Emergency shutdown: broadcast exit signal to every handshake'd core and
    // deinit their AICore register blocks. Idempotent.
    void emergency_shutdown(Runtime *runtime)
    {
        Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
        int32_t timeout_count = 0;
        for (int32_t i = 0; i < cores_total_num_; i++)
        {
            Handshake *hank = &all_handshakes[i];
            OUT_OF_ORDER_STORE_BARRIER();
            hank->aicpu_regs_ready = 1;
            if (core_exec_states_[i].reg_addr != 0)
            {
                if (platform_deinit_aicore_regs(core_exec_states_[i].reg_addr) != 0) timeout_count++;
            }
        }
        if (timeout_count > 0)
        {}
    }

    static const char *shape_name(PTO2ResourceShape shape)
    {
        switch (shape)
        {
        case PTO2ResourceShape::AIC:
            return "AIC";
        case PTO2ResourceShape::AIV:
            return "AIV";
        case PTO2ResourceShape::MIX:
            return "MIX";
        case PTO2ResourceShape::DUMMY:
            return "DUMMY";
        }
        return "UNKNOWN";
    }

    static inline const char *subslot_name(PTO2SubtaskSlot s)
    {
        switch (s)
        {
        case PTO2SubtaskSlot::AIC:
            return "aic";
        case PTO2SubtaskSlot::AIV0:
            return "aiv0";
        case PTO2SubtaskSlot::AIV1:
            return "aiv1";
        }
        return "?";
    }

    int pop_ready_tasks_batch(PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count)
    {
        return sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
    }

    void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, const AsyncCtx &async_ctx, int32_t block_idx)
    {
        int32_t slot_idx = static_cast<int32_t>(subslot);
        uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
        const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
        dispatch_payload.function_bin_addr = callable->resolved_addr();
        auto &payload = *slot_state.payload;
        int n = 0;
        for (int32_t i = 0; i < payload.tensor_count; i++) dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
        for (int32_t i = 0; i < payload.scalar_count; i++) dispatch_payload.args[n++] = payload.scalars[i];
        dispatch_payload.local_context.block_idx = block_idx;
        dispatch_payload.local_context.block_num = slot_state.logical_block_num;
        dispatch_payload.local_context.async_ctx = async_ctx;
        dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
        dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
    }

    struct PublishHandle
    {
        uint64_t reg_addr;
        uint32_t reg_task_id;
        int32_t core_offset;
        uint64_t *dispatch_timestamp_slot;
    };

    SchedulerContext::PublishHandle prepare_subtask_to_core(int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, bool to_pending, int32_t block_idx)
    {
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto core_id = tracker.get_core_id_by_offset(core_offset);
        CoreExecState &core_exec_state = core_exec_states_[core_id];

        core_exec_state.dispatch_seq++;
        uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        static_assert((TASK_ID_MASK - AICORE_EXIT_SIGNAL + 1) % 2 == 0, "Sentinel skip must be even to preserve dual-buffer parity");
        if (reg_task_id >= AICORE_EXIT_SIGNAL)
        {
            core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
            reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        }

        uint32_t buf_idx = reg_task_id & 1u;
        PTO2DispatchPayload &payload = payload_per_core_[core_id][buf_idx];
        DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][buf_idx];
        deferred_slab->count = 0;
        deferred_slab->error_code = PTO2_ERROR_NONE;
        AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_slab);
        build_payload(payload, slot_state, subslot, async_ctx, block_idx);

        if (to_pending)
        {
            core_exec_state.pending_subslot = subslot;
            core_exec_state.pending_slot_state = &slot_state;
            core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
        }
        else
        {
            core_exec_state.running_subslot = subslot;
            core_exec_state.running_slot_state = &slot_state;
            core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
            tracker.change_core_state(core_offset);
        }
        tracker.set_pending_occupied(core_offset);

        uint64_t *dispatch_timestamp_slot = nullptr;

        return PublishHandle{core_exec_state.reg_addr, reg_task_id, core_offset, dispatch_timestamp_slot};
    }

    inline void publish_subtask_to_core(const PublishHandle &h, uint64_t dispatch_ts)
    {
        if (h.dispatch_timestamp_slot != nullptr) *h.dispatch_timestamp_slot = dispatch_ts;
        write_reg(h.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(h.reg_task_id));
    }

    // Fan out one block's subtasks (1 for AIC/AIV, 1-3 for MIX) into the
    // caller-supplied handles buffer. Returns the number of handles written.
    int prepare_block_for_dispatch(int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending, int32_t block_idx, PublishHandle *out_handles)
    {
        CoreTracker &tracker = core_trackers_[thread_idx];
        if (shape == PTO2ResourceShape::MIX)
        {
            uint8_t cmask = slot_state.active_mask.core_mask();
            int n = 0;
            if (cmask & PTO2_SUBTASK_MASK_AIC)
            {
                bool p = to_pending && !tracker.is_aic_core_idle(core_offset);
                out_handles[n++] = prepare_subtask_to_core(thread_idx, tracker.get_aic_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIC, p, block_idx);
            }
            if (cmask & PTO2_SUBTASK_MASK_AIV0)
            {
                bool p = to_pending && !tracker.is_aiv0_core_idle(core_offset);
                out_handles[n++] = prepare_subtask_to_core(thread_idx, tracker.get_aiv0_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIV0, p, block_idx);
            }
            if (cmask & PTO2_SUBTASK_MASK_AIV1)
            {
                bool p = to_pending && !tracker.is_aiv1_core_idle(core_offset);
                out_handles[n++] = prepare_subtask_to_core(thread_idx, tracker.get_aiv1_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIV1, p, block_idx);
            }
            return n;
        }
        else if (shape == PTO2ResourceShape::AIC)
        {
            out_handles[0] = prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
            return 1;
        }
        else
        {
            out_handles[0] = prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
            return 1;
        }
    }

    void dispatch_shape(int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress)
    {
        if (entered_drain) return;

        bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
        auto cores = tracker.get_dispatchable_cores(shape, phase);
        if (!cores.has_value()) return;

        while (cores.has_value() && !entered_drain)
        {
            int want = cores.count();
            PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
            int got = pop_ready_tasks_batch(shape, local_buf, batch, want);
            if (got == 0) break;

            bool any_sync_start = false;
            for (int bi = 0; bi < got; bi++)
            {
                if (batch[bi]->active_mask.requires_sync_start())
                {
                    any_sync_start = true;
                    break;
                }
            }

            PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
            int handle_count = 0;
            bool dispatched_any = false;

            auto flush_publish = [&]() {
                if (handle_count == 0) return;
                wmb();
                uint64_t dispatch_ts = 0;
                for (int i = 0; i < handle_count; i++) publish_subtask_to_core(handles[i], dispatch_ts);
                handle_count = 0;
                made_progress = true;
            };

            for (int bi = 0; bi < got; bi++)
            {
                PTO2TaskSlotState *slot_state = batch[bi];

                if (slot_state->active_mask.requires_sync_start())
                {
                    if (is_pending)
                    {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                        continue;
                    }
                    int32_t available = cores.count();
                    if (available < slot_state->logical_block_num)
                    {
                        flush_publish();
                        if (!enter_drain_mode(slot_state, slot_state->logical_block_num)) sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                        for (int rem = bi + 1; rem < got; rem++) sched_->ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                        entered_drain = true;
                        break;
                    }
                }

                if (!cores.has_value())
                {
                    flush_publish();
                    sched_->ready_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                    break;
                }

                dispatched_any = true;
                int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
                int32_t claim = std::min(cores.count(), remaining);
                int32_t start = slot_state->next_block_idx;
                slot_state->next_block_idx += claim;

                if (slot_state->next_block_idx < slot_state->logical_block_num) sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);

                for (int32_t b = 0; b < claim; b++)
                {
                    auto core_offset = cores.pop_first();
                    handle_count += prepare_block_for_dispatch(thread_idx, core_offset, *slot_state, shape, is_pending, start + b, &handles[handle_count]);
                }

                if (any_sync_start) flush_publish();
            }

            flush_publish();

            if (!dispatched_any) break;

            if (!cores.has_value()) cores = tracker.get_dispatchable_cores(shape, phase);
        }
    }

    void dispatch_ready_tasks(int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES], bool pmu_active, bool &made_progress)
    {
        using Phase = CoreTracker::DispatchPhase;
        constexpr int32_t MIX_I = static_cast<int32_t>(PTO2ResourceShape::MIX);

        static constexpr PTO2ResourceShape kAicAivOrder[2][2] = {
            {PTO2ResourceShape::AIC, PTO2ResourceShape::AIV},
            {PTO2ResourceShape::AIV, PTO2ResourceShape::AIC},
        };
        const PTO2ResourceShape *aic_aiv = kAicAivOrder[thread_idx & 1];

        const int32_t bd_per_thread = PLATFORM_MAX_BLOCKDIM / active_sched_threads_;
        const int32_t thread_capacity[PTO2_NUM_RESOURCE_SHAPES] = {
            bd_per_thread * PLATFORM_AIC_CORES_PER_BLOCKDIM,
            bd_per_thread * PLATFORM_AIV_CORES_PER_BLOCKDIM,
            bd_per_thread,
        };
        for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++)
        {
            auto &lb = local_bufs[s];
            int32_t excess = lb.count - thread_capacity[s];
            if (excess <= 0) continue;
            if (!has_idle_in_other_threads(thread_idx, static_cast<PTO2ResourceShape>(s))) continue;
            sched_->ready_queues[s].push_batch(&lb.slot_states[lb.count - excess], excess);
            lb.count -= excess;
        }

        auto flush_local_bufs = [&]() {
            for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++)
            {
                auto &lb = local_bufs[s];
                if (lb.count > 0)
                {
                    sched_->ready_queues[s].push_batch(lb.slot_states, lb.count);
                    lb.count = 0;
                }
            }
        };
        struct FlushGuard
        {
            decltype(flush_local_bufs) &flush_fn;
            ~FlushGuard()
            {
                flush_fn();
            }
        } flush_guard{flush_local_bufs};

        bool entered_drain = false;

        // ===== IDLE stage =====
        dispatch_shape(thread_idx, PTO2ResourceShape::MIX, Phase::IDLE, local_bufs[MIX_I], tracker, entered_drain, made_progress);
        if (entered_drain) return;

        bool skip_aic_aiv = has_residual_mix(local_bufs[MIX_I]);

        if (!skip_aic_aiv)
        {
            for (int i = 0; i < 2; i++)
            {
                PTO2ResourceShape s = aic_aiv[i];
                dispatch_shape(thread_idx, s, Phase::IDLE, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress);
                if (entered_drain) return;
            }
        }

        // Flush between IDLE and PENDING so PENDING-stage queue-size checks and any
        // peer-thread reads see the IDLE-stage release_fanin output.
        flush_local_bufs();

        if (pmu_active) return;

        if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX))
        {
            dispatch_shape(thread_idx, PTO2ResourceShape::MIX, Phase::PENDING, local_bufs[MIX_I], tracker, entered_drain, made_progress);
            if (entered_drain) return;
        }

        // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
        // it set; otherwise, escalate iff PENDING-MIX left residual.
        if (!skip_aic_aiv && has_residual_mix(local_bufs[MIX_I])) skip_aic_aiv = true;

        if (skip_aic_aiv) return;

        // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
        // will pull from the global queue on its next IDLE pass.
        for (int i = 0; i < 2; i++)
        {
            PTO2ResourceShape s = aic_aiv[i];
            if (has_idle_in_other_threads(thread_idx, s)) continue;
            dispatch_shape(thread_idx, s, Phase::PENDING, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress);
            if (entered_drain) return;
        }
    }

    bool has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const
    {
        for (int32_t t = 0; t < active_sched_threads_; t++)
        {
            if (t == self_thread_idx) continue;
            if (core_trackers_[t].get_idle_core_offset_states(shape).has_value()) return true;
        }
        return false;
    }

    bool has_residual_mix(const PTO2LocalReadyBuffer &mix_local_buf) const
    {
        return mix_local_buf.count > 0 || sched_->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size() > 0;
    }

    static SlotTransition decide_slot_transition(int32_t reg_task_id, int32_t reg_state, int32_t running_id, int32_t pending_id)
    {
        SlotTransition t;
        if (pending_id != AICPU_TASK_INVALID && reg_task_id == pending_id)
        {
            t.matched = true;
            t.running_done = true;  // Serial execution: pending event implies running done
            t.running_freed = true;
            t.pending_freed = true;
            if (reg_state == TASK_FIN_STATE) t.pending_done = true;  // Case 1: pending FIN
            // else: Case 2: pending ACK (pending_done stays false)
        }
        else if (reg_task_id == running_id)
        {
            if (reg_state == TASK_FIN_STATE)
            {
                if (pending_id == AICPU_TASK_INVALID)
                {
                    // Case 3.2: running FIN, no pending -> core goes idle
                    t.matched = true;
                    t.running_done = true;
                    t.running_freed = true;
                }
                // Case 3.1: running FIN, pending exists -> skip (transient state).
                // Case 1/2 (pending ACK/FIN) will complete running implicitly via running_done=true.
            }
            else
            {
                // Case 4: running ACK -- only pending_freed (slot now hardware-latched)
                t.matched = true;
                t.pending_freed = true;
            }
        }
        return t;
    }

    void complete_slot_task(PTO2TaskSlotState &slot_state, int32_t expected_reg_task_id, int32_t core_id, int32_t &completed_this_turn)
    {
        AICoreCompletionMailbox *mailbox = rt_ != nullptr ? rt_->aicore_mailbox : nullptr;
        bool defer_completion_to_consumer = false;

        if (slot_state.payload != nullptr)
        {
            volatile DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][expected_reg_task_id & 1];
            // (q) Read count first. AICore only writes error_code as part of a
            // condition-registration attempt that also increments count, so
            // count == 0 ⇒ no error and no conditions to forward. This is the
            // common path for kernels that don't use async waits (paged
            // attention, GEMM, etc.) and saves an L1 load + branch per call.
            uint32_t cond_count = deferred_slab->count;
            if (cond_count != 0)
            {
                int32_t slab_err = deferred_slab->error_code;
                if (slab_err != PTO2_ERROR_NONE)
                {
                    int32_t expected = PTO2_ERROR_NONE;
                    sched_->sm_header->sched_error_code.compare_exchange_strong(expected, slab_err, std::memory_order_acq_rel, std::memory_order_acquire);
                    completed_.store(true, std::memory_order_release);
                    return;
                }
                if (cond_count > MAX_COMPLETIONS_PER_TASK)
                {
                    int32_t expected = PTO2_ERROR_NONE;
                    sched_->sm_header->sched_error_code.compare_exchange_strong(expected, PTO2_ERROR_ASYNC_REGISTRATION_FAILED, std::memory_order_acq_rel, std::memory_order_acquire);
                    completed_.store(true, std::memory_order_release);
                    return;
                }

                slot_state.any_subtask_deferred.store(true, std::memory_order_release);

                const PTO2TaskId token = slot_state.task->task_id;
                for (uint32_t i = 0; i < cond_count; ++i)
                {
                    volatile DeferredCompletionEntry *e = &deferred_slab->entries[i];
                    while (!mailbox->try_push_condition(token, e->addr, e->expected_value, e->engine, e->completion_type))
                    {
                        sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
                        SPIN_WAIT_HINT();
                    }
                }
            }
        }

        bool mixed_complete = sched_->on_subtask_complete(slot_state);

        if (mixed_complete && slot_state.payload != nullptr && slot_state.any_subtask_deferred.load(std::memory_order_acquire))
        {
            // Some subtask of this task registered conditions; finish the
            // registration by handing the slot_state off to the consumer.
            while (!mailbox->try_push_normal_done(slot_state.task->task_id, reinterpret_cast<uint64_t>(&slot_state)))
            {
                sched_->async_wait_list.mpsc_skipped_count.fetch_add(1, std::memory_order_relaxed);
                SPIN_WAIT_HINT();
            }
            defer_completion_to_consumer = true;
        }

        if (mixed_complete && !defer_completion_to_consumer)
        {
            sched_->on_mixed_task_complete(slot_state);
            completed_this_turn++;
        }
    }

    static void promote_pending_to_running(CoreExecState &core)
    {
        core.running_slot_state = core.pending_slot_state;
        core.running_reg_task_id = core.pending_reg_task_id;
        core.running_subslot = core.pending_subslot;
        core.pending_slot_state = nullptr;
        core.pending_reg_task_id = AICPU_TASK_INVALID;
    }
    static void clear_running_slot(CoreExecState &core)
    {
        core.running_slot_state = nullptr;
        core.running_reg_task_id = AICPU_TASK_INVALID;
    }

    void check_running_cores_for_completion(int32_t thread_idx, int32_t &completed_this_turn, int32_t &cur_thread_completed, bool &made_progress)
    {
        SchedulerThreadProfile &profile = thread_profiles_[thread_idx];
        CoreTracker &tracker = core_trackers_[thread_idx];
        auto running_core_states = tracker.get_all_running_cores();
        while (running_core_states.has_value())
        {
            int32_t bit_pos = running_core_states.pop_first();
            int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
            CoreExecState &core = core_exec_states_[core_id];
            profile.cores_scanned++;

            uint64_t reg_val = static_cast<uint64_t>(*core.cond_ptr);
            rmb();
            int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
            int32_t reg_state = EXTRACT_TASK_STATE(reg_val);

            SlotTransition t = decide_slot_transition(reg_task_id, reg_state, core.running_reg_task_id, core.pending_reg_task_id);
            if (!t.matched) continue;

            // --- Apply phase: execute actions based on transition ---

            // 1. Complete finished tasks (capture pointers before modifying core state)
            if (t.pending_done)
            {
                uint64_t tc0 = get_sys_cnt_aicpu();
                complete_slot_task(*core.pending_slot_state, core.pending_reg_task_id, core_id, completed_this_turn);
                profile.complete_task_cycles += get_sys_cnt_aicpu() - tc0;
                profile.complete_task_calls++;
                cur_thread_completed++;
            }
            if (t.running_done)
            {
                uint64_t tc0 = get_sys_cnt_aicpu();
                complete_slot_task(*core.running_slot_state, core.running_reg_task_id, core_id, completed_this_turn);
                profile.complete_task_cycles += get_sys_cnt_aicpu() - tc0;
                profile.complete_task_calls++;
                cur_thread_completed++;
            }

            // 2. Update slot data
            if (t.running_freed)
            {
                if (core.pending_slot_state != nullptr && !t.pending_done)
                {
                    promote_pending_to_running(core);  // Case 2 or Case 3 (with pending)
                }
                else
                {
                    clear_running_slot(core);  // Case 1 or Case 3 (no pending)
                    if (t.pending_done)
                    {
                        core.pending_slot_state = nullptr;
                        core.pending_reg_task_id = AICPU_TASK_INVALID;
                    }
                }
            }

            // 3. Update tracker bitmap
            bool is_idle = (core.running_reg_task_id == AICPU_TASK_INVALID);
            if (is_idle)
            {
                tracker.change_core_state(bit_pos);       // Mark idle
                tracker.clear_pending_occupied(bit_pos);  // Idle safeguard: no payload to protect
            }
            else if (t.pending_freed && core.pending_reg_task_id == AICPU_TASK_INVALID)
            {
                tracker.clear_pending_occupied(bit_pos);
            }

            // 4. Progress signal (only when running task completes)
            if (t.running_done) made_progress = true;
        }
    }

    bool enter_drain_mode(PTO2TaskSlotState *slot_state, int32_t block_num)
    {
        int32_t expected = 0;
        if (!drain_state_.sync_start_pending.compare_exchange_strong(expected, -1, std::memory_order_relaxed, std::memory_order_relaxed)) return false;  // Another thread already holds the drain slot.
        // We own the drain slot.  Store the task and reset election flag before making it visible.
        drain_state_.pending_task.store(slot_state, std::memory_order_release);
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        // Release store: all stores above are now visible to any thread that
        // acquire-loads sync_start_pending and sees block_num > 0.
        drain_state_.sync_start_pending.store(block_num, std::memory_order_release);
        return true;
    }
    int32_t count_global_available(PTO2ResourceShape shape)
    {
        int32_t total = 0;
        for (int32_t t = 0; t < active_sched_threads_; t++) total += core_trackers_[t].get_idle_core_offset_states(shape).count();
        return total;
    }
    void drain_worker_dispatch(int32_t block_num)
    {
        PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
        if (!slot_state)
        {
            drain_state_.sync_start_pending.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();

        for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++)
        {
            auto valid = core_trackers_[t].get_idle_core_offset_states(shape);
            int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
            int32_t claim = std::min(valid.count(), remaining);
            int32_t start = slot_state->next_block_idx;
            slot_state->next_block_idx += claim;
            PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
            int handle_count = 0;
            for (int32_t b = 0; b < claim; b++)
            {
                auto core_offset = valid.pop_first();
                handle_count += prepare_block_for_dispatch(t, core_offset, *slot_state, shape, false, start + b, &handles[handle_count]);
            }
            wmb();
            uint64_t dispatch_ts = 0;
            for (int i = 0; i < handle_count; i++) publish_subtask_to_core(handles[i], dispatch_ts);
        }

        std::atomic_thread_fence(std::memory_order_release);
        drain_state_.pending_task.store(nullptr, std::memory_order_release);
        drain_state_.drain_ack_mask.store(0, std::memory_order_relaxed);
        drain_state_.drain_worker_elected.store(0, std::memory_order_relaxed);
        drain_state_.sync_start_pending.store(0, std::memory_order_release);
    }
    void handle_drain_mode(int32_t thread_idx)
    {
        // Spin until drain is fully initialized (sentinel -1 -> block_num > 0).
        int32_t block_num;
        do {
            block_num = drain_state_.sync_start_pending.load(std::memory_order_acquire);
        } while (block_num < 0);
        if (block_num == 0) return;

        uint32_t all_acked = (1u << active_sched_threads_) - 1;

        // Ack barrier -- signal this thread has stopped dispatch.
        drain_state_.drain_ack_mask.fetch_or(1u << thread_idx, std::memory_order_release);

        // Spin until all threads have acked.
        // If our bit is cleared while waiting, elected reset due to insufficient resources.
        while (true)
        {
            uint32_t ack = drain_state_.drain_ack_mask.load(std::memory_order_acquire);
            if ((ack & all_acked) == all_acked) break;
            if ((ack & (1u << thread_idx)) == 0) return;
            SPIN_WAIT_HINT();
        }

        // Election -- exactly one thread wins the CAS.
        int32_t expected = 0;
        drain_state_.drain_worker_elected.compare_exchange_strong(expected, thread_idx + 1, std::memory_order_acquire, std::memory_order_relaxed);

        if (drain_state_.drain_worker_elected.load(std::memory_order_relaxed) != thread_idx + 1)
        {
            // Non-elected: spin-wait for drain completion or resource-insufficient reset.
            while (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0)
            {
                if (drain_state_.drain_worker_elected.load(std::memory_order_acquire) == 0) return;
                SPIN_WAIT_HINT();
            }
            return;
        }

        // Elected: check if global resources are sufficient.
        PTO2TaskSlotState *slot_state = drain_state_.pending_task.load(std::memory_order_acquire);
        if (slot_state == nullptr)
        {
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        int32_t available = count_global_available(shape);

        if (available < block_num)
        {
            // Insufficient resources -- reset drain fields so threads can resume
            // completion polling to free running cores, then retry.
            drain_state_.drain_ack_mask.store(0, std::memory_order_release);
            drain_state_.drain_worker_elected.store(0, std::memory_order_release);
            return;
        }

        // Dispatch -- all other threads are spinning, elected thread has exclusive tracker access.
        drain_worker_dispatch(block_num);
    }

    LoopAction handle_orchestrator_exit(PTO2SharedMemoryHeader *header, Runtime *runtime)
    {
        if (completed_.load(std::memory_order_acquire)) return LoopAction::BREAK_LOOP;
        int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
        if (orch_err != PTO2_ERROR_NONE)
        {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
            return LoopAction::BREAK_LOOP;
        }
        int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
        if (sched_err != PTO2_ERROR_NONE)
        {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
            return LoopAction::BREAK_LOOP;
        }

        if (!orchestrator_done_) return LoopAction::NONE;

        if (total_tasks_ > 0 && completed_tasks_.load(std::memory_order_relaxed) >= total_tasks_)
        {
            completed_.store(true, std::memory_order_release);
            return LoopAction::BREAK_LOOP;
        }
        return LoopAction::NONE;
    }

    LoopAction handle_core_transition(bool &cores_released)
    {
        if (!transition_requested_.load(std::memory_order_acquire)) return LoopAction::NONE;
        if (!reassigned_.load(std::memory_order_acquire))
        {
            wait_reassign_.fetch_add(1, std::memory_order_release);
            while (!reassigned_.load(std::memory_order_acquire))
            {
                if (completed_.load(std::memory_order_acquire)) return LoopAction::BREAK_LOOP;
                SPIN_WAIT_HINT();
            }
        }
        cores_released = true;
        return LoopAction::NONE;
    }

    LoopAction check_idle_fatal_error(PTO2SharedMemoryHeader *header, Runtime *runtime)
    {
        if (completed_.load(std::memory_order_acquire)) return LoopAction::BREAK_LOOP;
        int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
        if (orch_err != PTO2_ERROR_NONE)
        {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
            return LoopAction::BREAK_LOOP;
        }
        int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
        if (sched_err != PTO2_ERROR_NONE)
        {
            if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
            return LoopAction::BREAK_LOOP;
        }
        return LoopAction::NONE;
    }

    void log_stall_diagnostics(int32_t thread_idx)
    {
        CoreTracker &tracker = core_trackers_[thread_idx];

        // T0 owns the shared-ring scan; printing it from other threads would
        // produce identical TASK lines once per scheduler thread.
        if (thread_idx == 0)
        {
            int32_t cnt_ready = 0, cnt_waiting = 0, cnt_running = 0, submitted_in_ring = 0;
            for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
            {
                PTO2SharedMemoryRingHeader &ring = *sched_->ring_sched_states[r].ring;
                int32_t ring_task_count = ring.fc.current_task_index.load(std::memory_order_relaxed);
                submitted_in_ring += ring_task_count;
                for (int32_t si = 0; si < ring_task_count; si++)
                {
                    PTO2TaskSlotState &slot_state = ring.get_slot_state_by_task_id(si);
                    // (m) task_state retired; use completion_flags directly.
                    bool fanin_ready = sched_->fanin_satisfied(&slot_state);
                    if (ring.completion_flags[si & ring.task_window_mask].load(std::memory_order_relaxed) != 0) continue;
                    char running_on[192] = {0};
                    int32_t owner = -1;
                    int32_t pos = 0;
                    bool is_running = false;
                    for (int32_t cid = 0; cid < cores_total_num_ && pos + 32 < (int32_t)sizeof(running_on); cid++)
                    {
                        if (core_exec_states_[cid].running_slot_state != &slot_state) continue;
                        is_running = true;
                        if (owner < 0) owner = find_core_owner_thread(cid);
                        const char *sname = subslot_name(core_exec_states_[cid].running_subslot);
                        int32_t written = snprintf(running_on + pos, sizeof(running_on) - pos, "%score=%d(%s)", pos == 0 ? "" : " ", cid, sname);
                        if (written > 0) pos += written;
                    }

                    if (is_running)
                    {
                        cnt_running++;
                        if (cnt_running > STALL_DUMP_READY_MAX) continue;
                        continue;
                    }
                    if (fanin_ready)
                    {
                        cnt_ready++;
                        if (cnt_ready > STALL_DUMP_READY_MAX) continue;
                        continue;
                    }
                    cnt_waiting++;
                    if (cnt_waiting > STALL_DUMP_WAIT_MAX) continue;
                }
            }
        }

        for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++)
        {
            int32_t offset = cli * 3;
            int32_t aic_id = tracker.get_aic_core_id(offset);
            int32_t aiv0_id = tracker.get_aiv0_core_id(offset);
            int32_t aiv1_id = tracker.get_aiv1_core_id(offset);
            bool aic_idle = tracker.is_aic_core_idle(offset);
            bool aiv0_idle = tracker.is_aiv0_core_idle(offset);
            bool aiv1_idle = tracker.is_aiv1_core_idle(offset);
            char aic_buf[128], aiv0_buf[128], aiv1_buf[128];
            format_core_status(aic_buf, sizeof(aic_buf), aic_id, aic_idle, &core_exec_states_[aic_id], core_exec_states_[aic_id].reg_addr);
            format_core_status(aiv0_buf, sizeof(aiv0_buf), aiv0_id, aiv0_idle, &core_exec_states_[aiv0_id], core_exec_states_[aiv0_id].reg_addr);
            format_core_status(aiv1_buf, sizeof(aiv1_buf), aiv1_id, aiv1_idle, &core_exec_states_[aiv1_id], core_exec_states_[aiv1_id].reg_addr);
        }
    }

    void log_shutdown_stall_snapshot()
    {
        int32_t thread_count = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
        if (thread_count < 0 || thread_count > MAX_AICPU_THREADS) thread_count = thread_count < 0 ? 0 : MAX_AICPU_THREADS;
        for (int32_t t = 0; t < thread_count; t++) log_stall_diagnostics(t);
    }

    int32_t find_core_owner_thread(int32_t core_id) const
    {
        for (int32_t t = 0; t < aicpu_thread_num_; t++)
        {
            const int32_t *ids = core_trackers_[t].core_ids();
            int32_t n = core_trackers_[t].core_num();
            for (int32_t i = 0; i < n; i++)
                if (ids[i] == core_id) return t;
        }
        return -1;
    }

    bool self_owns_running_task(int32_t thread_idx) const
    {
        const int32_t *cores = core_trackers_[thread_idx].core_ids();
        int32_t core_num = core_trackers_[thread_idx].core_num();
        for (int32_t i = 0; i < core_num; i++)
            if (core_exec_states_[cores[i]].running_slot_state != nullptr) return true;
        return false;
    }

    bool no_thread_owns_running_task() const
    {
        for (int32_t t = 0; t < aicpu_thread_num_; t++)
            if (self_owns_running_task(t)) return false;
        return true;
    }

    int32_t handle_timeout_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime)
    {
        latch_scheduler_error(header, thread_idx, PTO2_ERROR_SCHEDULER_TIMEOUT);
        if (!completed_.exchange(true, std::memory_order_acq_rel))
        {
            log_shutdown_stall_snapshot();
            emergency_shutdown(runtime);
        }
        return -PTO2_ERROR_SCHEDULER_TIMEOUT;
    }

    uint64_t get_function_bin_addr(int func_id) const
    {
        if (!func_id_to_addr_ || func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }
};

#endif  // SCHEDULER_CONTEXT_H
