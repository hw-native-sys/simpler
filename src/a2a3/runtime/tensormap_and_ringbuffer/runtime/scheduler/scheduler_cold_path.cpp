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
#include "scheduler_context.h"

#include <cinttypes>

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "common/perf_profiling.h"
#include "common/platform_config.h"
#include "runtime.h"
#include "spin_hint.h"

// =============================================================================
// Cold-path helpers for the main dispatch loop (noinline to reduce hot-loop icache)
// =============================================================================

LoopAction SchedulerContext::handle_orchestrator_exit(
    int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count
) {
    bool orch_done = *orchestrator_done_ptr_;
    if (!orch_done) return LoopAction::NONE;

    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        DEV_ERROR(
            "Thread %d: Fatal error (code=%d), sending EXIT_SIGNAL to all cores. "
            "completed_tasks=%d, total_tasks=%d",
            thread_idx, orch_err, completed_tasks_ptr_->load(std::memory_order_relaxed), *total_tasks_ptr_
        );
        emergency_shutdown_fn_(runtime);
        completed_ptr_->store(true, std::memory_order_release);
        return LoopAction::BREAK_LOOP;
    }

    task_count = *total_tasks_ptr_;
    if (task_count > 0 && completed_tasks_ptr_->load(std::memory_order_relaxed) >= task_count) {
        completed_ptr_->store(true, std::memory_order_release);
        DEV_INFO(
            "Thread %d: PTO2 completed tasks %d/%d", thread_idx, completed_tasks_ptr_->load(std::memory_order_relaxed),
            task_count
        );
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::handle_core_transition(bool &cores_released) {
    if (!transition_requested_ptr_->load(std::memory_order_acquire)) return LoopAction::NONE;
    if (!reassigned_ptr_->load(std::memory_order_acquire)) {
        wait_reassign_ptr_->fetch_add(1, std::memory_order_release);
        while (!reassigned_ptr_->load(std::memory_order_acquire)) {
            if (completed_ptr_->load(std::memory_order_acquire)) {
                return LoopAction::BREAK_LOOP;
            }
            SPIN_WAIT_HINT();
        }
    }
    cores_released = true;
    return LoopAction::NONE;
}

LoopAction
SchedulerContext::check_idle_fatal_error(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime) {
    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        DEV_ERROR("Thread %d: Fatal error detected (code=%d), sending EXIT_SIGNAL to all cores", thread_idx, orch_err);
        emergency_shutdown_fn_(runtime);
        completed_ptr_->store(true, std::memory_order_release);
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

void SchedulerContext::log_stall_diagnostics(
    int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count
) {
    int32_t c = completed_tasks_ptr_->load(std::memory_order_relaxed);
    DEV_ALWAYS(
        "PTO2 stall: no progress for %d iterations, completed=%d total=%d (last progress at %d)", idle_iterations, c,
        task_count, last_progress_count
    );
    CoreTracker &tracker = core_trackers_[thread_idx];
    int32_t cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        PTO2SharedMemoryRingHeader &ring = *sched_->ring_sched_states[r].ring;
        int32_t ring_task_count = ring.fc.current_task_index.load(std::memory_order_relaxed);
        for (int32_t si = 0; si < ring_task_count; si++) {
            PTO2TaskSlotState &slot_state = ring.get_slot_state_by_task_id(si);
            PTO2TaskState st = slot_state.task_state.load(std::memory_order_relaxed);
            int32_t rc = slot_state.fanin_refcount.load(std::memory_order_relaxed);
            int32_t fi = slot_state.fanin_count;
            int32_t kid = slot_state.task->kernel_id[0];
            if (st >= PTO2_TASK_COMPLETED) continue;
            if (st == PTO2_TASK_READY || st == PTO2_TASK_RUNNING) {
                cnt_inflight++;
                continue;
            }
            if (rc >= fi) {
                cnt_ready++;
                if (cnt_ready <= STALL_DUMP_READY_MAX) {
                    DEV_ALWAYS(
                        "  STUCK-READY  ring=%d task_id=%" PRId64 " kernel_id=%d refcount=%d fanin=%d state=%d", r,
                        static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi, static_cast<int32_t>(st)
                    );
                }
            } else {
                cnt_waiting++;
                if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                    DEV_ALWAYS(
                        "  STUCK-WAIT   ring=%d task_id=%" PRId64 " kernel_id=%d refcount=%d fanin=%d state=%d", r,
                        static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi, static_cast<int32_t>(st)
                    );
                }
            }
        }
    }
    DEV_ALWAYS("  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d", cnt_ready, cnt_waiting, cnt_inflight);
    int32_t aic_running = tracker.get_running_count<CoreType::AIC>();
    int32_t aiv_running = tracker.get_running_count<CoreType::AIV>();
    int32_t total_running = aic_running + aiv_running;
    int32_t core_num = core_count_per_thread_[thread_idx];
    DEV_ALWAYS(
        "  thread=%d running_cores=%d (AIC=%d AIV=%d) core_num=%d", thread_idx, total_running, aic_running, aiv_running,
        core_num
    );
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
            "    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d kernel=%d", cid, static_cast<unsigned>(cond_reg),
            EXTRACT_TASK_STATE(cond_reg), EXTRACT_TASK_ID(cond_reg), sw_tid, hw_kernel
        );
    }
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

int32_t SchedulerContext::handle_timeout_exit(
    int32_t thread_idx, int32_t idle_iterations
#if PTO2_PROFILING
    ,
    uint64_t sched_start_ts
#endif
) {
    DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
#if PTO2_PROFILING
    uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
    DEV_ALWAYS(
        "Thread %d: sched_start=%" PRIu64 " sched_end(timeout)=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts),
        cycles_to_us(sched_timeout_ts - sched_start_ts)
    );
#endif
    return -1;
}

#if PTO2_PROFILING
void SchedulerContext::log_profiling_summary(int32_t thread_idx, int32_t cur_thread_completed) {
    auto &perf = sched_perf_[thread_idx];
    uint64_t sched_end_ts = get_sys_cnt_aicpu();
    DEV_ALWAYS(
        "Thread %d: sched_start=%" PRIu64 " sched_end=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(perf.sched_start_ts), static_cast<uint64_t>(sched_end_ts),
        cycles_to_us(sched_end_ts - perf.sched_start_ts)
    );

    uint64_t sched_total = perf.sched_wiring_cycle + perf.sched_complete_cycle + perf.sched_scan_cycle +
                           perf.sched_dispatch_cycle + perf.sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;

#if PTO2_SCHED_PROFILING
    {
        PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll = (perf.sched_complete_cycle > otc_total + perf.sched_complete_perf_cycle) ?
                                     (perf.sched_complete_cycle - otc_total - perf.sched_complete_perf_cycle) :
                                     0;
        uint64_t dispatch_poll =
            (perf.sched_dispatch_cycle > perf.sched_dispatch_pop_cycle + perf.sched_dispatch_setup_cycle) ?
                (perf.sched_dispatch_cycle - perf.sched_dispatch_pop_cycle - perf.sched_dispatch_setup_cycle) :
                0;

        DEV_ALWAYS(
            "Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===", thread_idx,
            cycles_to_us(sched_total), cur_thread_completed
        );

        double notify_avg =
            cur_thread_completed > 0 ? static_cast<double>(perf.notify_edges_total) / cur_thread_completed : 0.0;
        double fanin_avg =
            cur_thread_completed > 0 ? static_cast<double>(perf.fanin_edges_total) / cur_thread_completed : 0.0;
        DEV_ALWAYS(
            "Thread %d:   complete       : %.3fus (%.1f%%)  [fanout: edges=%" PRIu64
            ", max_degree=%d, avg=%.1f]  [fanin: "
            "edges=%" PRIu64 ", max_degree=%d, avg=%.1f]",
            thread_idx, cycles_to_us(perf.sched_complete_cycle), perf.sched_complete_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(perf.notify_edges_total), perf.notify_max_degree, notify_avg,
            static_cast<uint64_t>(perf.fanin_edges_total), perf.fanin_max_degree, fanin_avg
        );

        uint64_t c_parent = perf.sched_complete_cycle > 0 ? perf.sched_complete_cycle : 1;
        uint64_t complete_miss_count = (perf.complete_probe_count > perf.complete_hit_count) ?
                                           (perf.complete_probe_count - perf.complete_hit_count) :
                                           0;
        double complete_hit_rate =
            perf.complete_probe_count > 0 ? perf.complete_hit_count * 100.0 / perf.complete_probe_count : 0.0;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)  hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%",
            thread_idx, cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent,
            static_cast<uint64_t>(perf.complete_hit_count), static_cast<uint64_t>(complete_miss_count),
            complete_hit_rate
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
            "Thread %d:     perf         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(perf.sched_complete_perf_cycle),
            perf.sched_complete_perf_cycle * 100.0 / c_parent
        );

        uint64_t pop_total = perf.pop_hit + perf.pop_miss;
        double pop_hit_rate = pop_total > 0 ? perf.pop_hit * 100.0 / pop_total : 0.0;
        DEV_ALWAYS(
            "Thread %d:   dispatch       : %.3fus (%.1f%%)  [pop: hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%]",
            thread_idx, cycles_to_us(perf.sched_dispatch_cycle), perf.sched_dispatch_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(perf.pop_hit), static_cast<uint64_t>(perf.pop_miss), pop_hit_rate
        );
        uint64_t global_dispatch_count = perf.pop_hit - perf.local_dispatch_count;
        uint64_t total_dispatched = perf.local_dispatch_count + global_dispatch_count;
        double local_hit_rate = total_dispatched > 0 ? perf.local_dispatch_count * 100.0 / total_dispatched : 0.0;
        DEV_ALWAYS(
            "Thread %d:     local_disp   : local=%" PRIu64 ", global=%" PRIu64 ", overflow=%" PRIu64
            ", local_rate=%.1f%%",
            thread_idx, static_cast<uint64_t>(perf.local_dispatch_count), static_cast<uint64_t>(global_dispatch_count),
            static_cast<uint64_t>(perf.local_overflow_count), local_hit_rate
        );

        uint64_t d_parent = perf.sched_dispatch_cycle > 0 ? perf.sched_dispatch_cycle : 1;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent
        );
        DEV_ALWAYS(
            "Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(perf.sched_dispatch_pop_cycle), perf.sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(perf.sched_dispatch_pop_cycle - sp.pop_wait_cycle), cycles_to_us(sp.pop_wait_cycle),
            static_cast<uint64_t>(sp.pop_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     setup        : %.3fus (%.1f%%)", thread_idx, cycles_to_us(perf.sched_dispatch_setup_cycle),
            perf.sched_dispatch_setup_cycle * 100.0 / d_parent
        );

        DEV_ALWAYS(
            "Thread %d:   scan           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(perf.sched_scan_cycle),
            perf.sched_scan_cycle * 100.0 / sched_total
        );

#if PTO2_SCHED_PROFILING
        DEV_ALWAYS(
            "Thread %d:   wiring         : %.3fus (%.1f%%)  tasks=%d", thread_idx,
            cycles_to_us(perf.sched_wiring_cycle), perf.sched_wiring_cycle * 100.0 / sched_total,
            perf.phase_wiring_count
        );
#else
        DEV_ALWAYS(
            "Thread %d:   wiring         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(perf.sched_wiring_cycle),
            perf.sched_wiring_cycle * 100.0 / sched_total
        );
#endif

        DEV_ALWAYS(
            "Thread %d:   idle           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(perf.sched_idle_cycle),
            perf.sched_idle_cycle * 100.0 / sched_total
        );

        if (cur_thread_completed > 0) {
            DEV_ALWAYS(
                "Thread %d:   avg/complete   : %.3fus", thread_idx,
                cycles_to_us(perf.sched_complete_cycle) / cur_thread_completed
            );
        }
    }
#endif
    DEV_ALWAYS(
        "Thread %d: Scheduler summary: total_time=%.3fus, loops=%" PRIu64 ", tasks_scheduled=%d", thread_idx,
        cycles_to_us(sched_total), static_cast<uint64_t>(perf.sched_loop_count), cur_thread_completed
    );
}
#endif
