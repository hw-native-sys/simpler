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
#include <cstdio>

#include "common/unified_log.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "common/memory_barrier.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "runtime.h"
#include "spin_hint.h"

// =============================================================================
// Cold-path helpers for the main dispatch loop (noinline to reduce hot-loop icache)
// =============================================================================

// Returns true iff this call won the first-writer CAS for sched_error_code — the
// caller may then write companion fields (e.g. the stall detail) knowing they
// describe the same observation that owns the latched code.
static bool latch_scheduler_error(PTO2SharedMemoryHeader *header, int32_t thread_idx, int32_t error_code) {
    if (header == nullptr || error_code == PTO2_ERROR_NONE) {
        return false;
    }
    // The first error code/thread pair wins; the bitmap cumulatively records all reporting threads.
    int32_t expected = PTO2_ERROR_NONE;
    bool won = header->sched_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel);
    if (won) {
        header->sched_error_thread.store(thread_idx, std::memory_order_release);
    }
    if (thread_idx >= 0 && thread_idx < 32) {
        header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
    }
    return won;
}

LoopAction SchedulerContext::handle_orchestrator_exit(
    int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count
) {
    if (completed_.load(std::memory_order_acquire)) {
        return LoopAction::BREAK_LOOP;
    }
    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        LOG_ERROR(
            "Thread %d: Fatal error (code=%d), sending EXIT_SIGNAL to all cores. "
            "completed_tasks=%d, total_tasks=%d",
            thread_idx, orch_err, completed_tasks_.load(std::memory_order_relaxed), total_tasks_
        );
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
        return LoopAction::BREAK_LOOP;
    }
    int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != PTO2_ERROR_NONE) {
        LOG_ERROR("Thread %d: Scheduler fatal error detected (code=%d)", thread_idx, sched_err);
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
        return LoopAction::BREAK_LOOP;
    }

    bool orch_done = orchestrator_done_.load(std::memory_order_acquire);
    if (!orch_done) return LoopAction::NONE;

    task_count = total_tasks_;
    if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
        completed_.store(true, std::memory_order_release);
        LOG_INFO_V0(
            "Thread %d: PTO2 completed tasks %d/%d", thread_idx, completed_tasks_.load(std::memory_order_relaxed),
            task_count
        );
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::handle_core_transition(bool &cores_released) {
    if (!transition_requested_.load(std::memory_order_acquire)) return LoopAction::NONE;
    if (!reassigned_.load(std::memory_order_acquire)) {
        wait_reassign_.fetch_add(1, std::memory_order_release);
        while (!reassigned_.load(std::memory_order_acquire)) {
            if (completed_.load(std::memory_order_acquire)) {
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
    if (completed_.load(std::memory_order_acquire)) {
        return LoopAction::BREAK_LOOP;
    }
    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        LOG_ERROR("Thread %d: Fatal error detected (code=%d), sending EXIT_SIGNAL to all cores", thread_idx, orch_err);
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
        return LoopAction::BREAK_LOOP;
    }
    int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != PTO2_ERROR_NONE) {
        LOG_ERROR("Thread %d: Scheduler fatal error detected (code=%d)", thread_idx, sched_err);
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

// =============================================================================
// Stall diagnostic log format.
//
// Every line is self-contained — when scheduler threads emit concurrently and
// device_log interleaves their output, each line still carries enough context
// to identify which thread / iteration / object it belongs to.
//
// Prefix on every line:
//   [STALL thread=N idle_iterations=K] CATEGORY ...
//
// All scheduler threads spinning at the same idle rate hit STALL_LOG_INTERVAL
// together, so lines with the same idle_iterations belong to one diagnostic
// round; grep "idle_iterations=N" groups one round's output.
//
// Categories (and which thread emits them):
//   SUMMARY  — completed / total counts and scan totals               (thread 0 only)
//   TASK     — one per non-completed task scanned from shared rings   (thread 0 only)
//              - state=RUNNING: includes running_on=[...] cross-ref
//              - state=READY:   fanin satisfied but no idle core yet
//              - state=WAIT:    includes missing_deps=N
//   CLUSTER  — one per cluster owned by this thread                   (every thread)
//              - busy slot shows kernel + task_id + cond_reg_state;
//                ANOMALY suffix when COND register is fin while software
//                still has the slot marked busy.
//
// Reader workflow:
//   1. grep SUMMARY                          -> overall completion status
//   2. grep "idle_iterations=N TASK"         -> stuck RUNNING task and which
//                                               core/thread it is on
//   3. grep "idle_iterations=N CLUSTER.*task=<id>" -> cross-check via the
//                                                     cluster line (or just
//                                                     read running_on in step 2)
// =============================================================================

namespace {

// Format a core's idle/busy state into a fixed buffer. Used inside CLUSTER lines.
// Layout (idle):    coreN(idle)
// Layout (busy):    coreN(busy kernel=K task=T cond_reg_state=ack)
// Layout (anomaly): coreN(busy kernel=K task=T cond_reg_state=fin ANOMALY)
//
// Healthy busy: COND register reports ack (AICore still executing). fin means
// AICore wrote completion but AICPU hasn't recycled the running slot yet —
// either a completion-poll bug or the diagnostic raced the recycle.
void format_core_status(
    char *buf, size_t buf_size, int32_t core_id, bool idle, const CoreExecState *core_state, uint64_t reg_addr_for_cond
) {
    if (idle) {
        snprintf(buf, buf_size, "core%d(idle)", core_id);
        return;
    }
    int32_t kernel = -1;
    int64_t task_id_raw = -1;
    if (core_state && core_state->running_slot_state) {
        int32_t subslot = static_cast<int32_t>(core_state->running_subslot);
        kernel = core_state->running_slot_state->task->kernel_id[subslot];
        task_id_raw = static_cast<int64_t>(core_state->running_slot_state->task->task_id.raw);
    }
    uint64_t cond_reg = read_reg(reg_addr_for_cond, RegId::COND);
    int32_t hw_state = EXTRACT_TASK_STATE(cond_reg);
    const char *cond_reg_state_str = (hw_state == TASK_ACK_STATE) ? "ack" : "fin";
    if (hw_state == TASK_ACK_STATE) {
        snprintf(
            buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s)", core_id, kernel, task_id_raw,
            cond_reg_state_str
        );
    } else {
        snprintf(
            buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s ANOMALY)", core_id, kernel,
            task_id_raw, cond_reg_state_str
        );
    }
}

}  // namespace

int32_t SchedulerContext::find_core_owner_thread(int32_t core_id) const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        const int32_t *ids = core_trackers_[t].core_ids();
        int32_t n = core_trackers_[t].core_num();
        for (int32_t i = 0; i < n; i++) {
            if (ids[i] == core_id) return t;
        }
    }
    return -1;
}

bool SchedulerContext::self_owns_running_task(int32_t thread_idx) const {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    int32_t core_num = core_trackers_[thread_idx].core_num();
    for (int32_t i = 0; i < core_num; i++) {
        if (core_exec_states_[cores[i]].running_slot_state != nullptr) {
            return true;
        }
    }
    return false;
}

bool SchedulerContext::no_thread_owns_running_task() const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        if (self_owns_running_task(t)) return false;
    }
    return true;
}

void SchedulerContext::log_stall_diagnostics(
    int32_t thread_idx, int32_t task_count, int32_t idle_iterations, int32_t last_progress_count
) {
    CoreTracker &tracker = core_trackers_[thread_idx];

    // T0 owns the shared-ring scan; printing it from other threads would
    // produce identical TASK lines once per scheduler thread.
    if (thread_idx == 0) {
        int32_t cnt_ready = 0, cnt_waiting = 0, cnt_running = 0, submitted_in_ring = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            PTO2SharedMemoryRingHeader &ring = *sched_->ring_sched_states[r].ring;
            int32_t ring_task_count = ring.fc.current_task_index.load(std::memory_order_relaxed);
            submitted_in_ring += ring_task_count;
            // Scan only live task_ids [last_task_alive, current_task_index): slots
            // wrap (slot = task_id % window), so starting at 0 re-reads each live
            // slot once per earlier task_id and inflates the scan_* counts.
            int32_t ring_task_start = ring.fc.last_task_alive.load(std::memory_order_relaxed);
            for (int32_t si = ring_task_start; si < ring_task_count; si++) {
                PTO2TaskSlotState &slot_state = ring.get_slot_state_by_task_id(si);
                PTO2TaskState st = slot_state.task_state.load(std::memory_order_relaxed);
                int32_t rc = slot_state.fanin_refcount.load(std::memory_order_relaxed);
                int32_t fi = slot_state.fanin_count;
                int32_t kid_aic = slot_state.task->kernel_id[0];
                int32_t kid_aiv0 = slot_state.task->kernel_id[1];
                int32_t kid_aiv1 = slot_state.task->kernel_id[2];
                int64_t task_id = static_cast<int64_t>(slot_state.task->task_id.raw);
                if (st >= PTO2_TASK_COMPLETED) continue;
                // task_state has no intermediate ready/running value — it
                // stays PENDING until the worker stores COMPLETED. Classify
                // by the ground truth instead: a slot is RUNNING iff some
                // core has it as running_slot_state. A task occupies at most
                // 3 cores (one cluster), all under the same owner thread by
                // construction of assign_cores_to_threads.
                char running_on[192] = {0};
                int32_t owner = -1;
                int32_t pos = 0;
                bool is_running = false;
                for (int32_t cid = 0; cid < cores_total_num_ && pos + 32 < (int32_t)sizeof(running_on); cid++) {
                    if (core_exec_states_[cid].running_slot_state != &slot_state) continue;
                    is_running = true;
                    if (owner < 0) owner = find_core_owner_thread(cid);
                    const char *sname = subslot_name(core_exec_states_[cid].running_subslot);
                    int32_t written = snprintf(
                        running_on + pos, sizeof(running_on) - pos, "%score=%d(%s)", pos == 0 ? "" : " ", cid, sname
                    );
                    if (written > 0) pos += written;
                }

                if (is_running) {
                    cnt_running++;
                    if (cnt_running > STALL_DUMP_READY_MAX) continue;
                    LOG_INFO_V9(
                        "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                        " state=RUNNING fanin_refcount=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d] "
                        "running_on=[owner_thread=%d cores=[%s]]",
                        thread_idx, idle_iterations, r, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1, owner, running_on
                    );
                    continue;
                }
                if (rc >= fi) {
                    cnt_ready++;
                    if (cnt_ready > STALL_DUMP_READY_MAX) continue;
                    LOG_INFO_V9(
                        "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                        " state=READY   fanin_refcount=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d]",
                        thread_idx, idle_iterations, r, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1
                    );
                    continue;
                }
                cnt_waiting++;
                if (cnt_waiting > STALL_DUMP_WAIT_MAX) continue;
                LOG_INFO_V9(
                    "[STALL thread=%d idle_iterations=%d] TASK ring=%d task_id=%" PRId64
                    " state=WAIT    fanin_refcount=%d/%d kernels=[aic:%d aiv0:%d aiv1:%d] missing_deps=%d",
                    thread_idx, idle_iterations, r, task_id, rc, fi, kid_aic, kid_aiv0, kid_aiv1, fi - rc
                );
            }
        }
        int32_t effective_total = task_count > 0 ? task_count : submitted_in_ring;
        int32_t c = completed_tasks_.load(std::memory_order_relaxed);
        LOG_INFO_V9(
            "[STALL thread=%d idle_iterations=%d] SUMMARY completed=%d/%d last_progress_iteration=%d "
            "scan_ready=%d scan_waiting=%d scan_running=%d",
            thread_idx, idle_iterations, c, effective_total, last_progress_count, cnt_ready, cnt_waiting, cnt_running
        );
    }

    // CLUSTER lines: one per cluster this thread owns.
    // cluster_id = local_cluster_idx * active_sched_threads_ + thread_idx, matching the
    // round-robin assignment in assign_cores_to_threads / reassign_cores_for_all_threads.
    int32_t ast = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
    for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++) {
        int32_t offset = cli * 3;
        int32_t aic_id = tracker.get_aic_core_id(offset);
        int32_t aiv0_id = tracker.get_aiv0_core_id(offset);
        int32_t aiv1_id = tracker.get_aiv1_core_id(offset);
        bool aic_idle = tracker.is_aic_core_idle(offset);
        bool aiv0_idle = tracker.is_aiv0_core_idle(offset);
        bool aiv1_idle = tracker.is_aiv1_core_idle(offset);
        int32_t cluster_id = cli * ast + thread_idx;
        char aic_buf[128], aiv0_buf[128], aiv1_buf[128];
        format_core_status(
            aic_buf, sizeof(aic_buf), aic_id, aic_idle, &core_exec_states_[aic_id], core_exec_states_[aic_id].reg_addr
        );
        format_core_status(
            aiv0_buf, sizeof(aiv0_buf), aiv0_id, aiv0_idle, &core_exec_states_[aiv0_id],
            core_exec_states_[aiv0_id].reg_addr
        );
        format_core_status(
            aiv1_buf, sizeof(aiv1_buf), aiv1_id, aiv1_idle, &core_exec_states_[aiv1_id],
            core_exec_states_[aiv1_id].reg_addr
        );
        LOG_INFO_V9(
            "[STALL thread=%d idle_iterations=%d] CLUSTER cluster_id=%d aic=%s aiv0=%s aiv1=%s", thread_idx,
            idle_iterations, cluster_id, aic_buf, aiv0_buf, aiv1_buf
        );
    }
}

void SchedulerContext::log_shutdown_stall_snapshot(
    int32_t trigger_thread_idx, int32_t trigger_idle_iterations, int32_t trigger_last_progress_count
) {
    LOG_WARN(
        "[SHUTDOWN_SNAPSHOT trigger_thread=%d reason=scheduler_timeout idle_iterations=%d] "
        "dumping all scheduler threads before emergency shutdown",
        trigger_thread_idx, trigger_idle_iterations
    );
    int32_t thread_count = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
    if (thread_count < 0 || thread_count > MAX_AICPU_THREADS) {
        LOG_ERROR(
            "[SHUTDOWN_SNAPSHOT trigger_thread=%d] invalid thread_count=%d, clamping to [0,%d]", trigger_thread_idx,
            thread_count, MAX_AICPU_THREADS
        );
        thread_count = thread_count < 0 ? 0 : MAX_AICPU_THREADS;
    }
    for (int32_t t = 0; t < thread_count; t++) {
        log_stall_diagnostics(t, total_tasks_, trigger_idle_iterations, trigger_last_progress_count);
    }
}

SchedulerContext::StallClassification SchedulerContext::classify_stall_reason() const {
    StallClassification cls{};
    cls.stuck_task_id = -1;
    cls.stuck_core = -1;
    int32_t cnt_running = 0, cnt_ready = 0, cnt_waiting = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        PTO2SharedMemoryRingHeader &ring = *sched_->ring_sched_states[r].ring;
        int32_t ring_task_count = ring.fc.current_task_index.load(std::memory_order_relaxed);
        // Active task_ids live in [last_task_alive, current_task_index); slots wrap
        // (slot = task_id % window), so scanning from 0 re-reads each live slot once
        // per earlier task_id that mapped to it -- inflating the counts to O(history).
        // Start at the tail so each live slot is visited exactly once (O(window)).
        int32_t ring_task_start = ring.fc.last_task_alive.load(std::memory_order_relaxed);
        for (int32_t si = ring_task_start; si < ring_task_count; si++) {
            PTO2TaskSlotState &slot_state = ring.get_slot_state_by_task_id(si);
            PTO2TaskState st = slot_state.task_state.load(std::memory_order_relaxed);
            if (st >= PTO2_TASK_COMPLETED) continue;
            // Same ground truth as log_stall_diagnostics: task_state stays PENDING
            // until COMPLETED, so RUNNING is read from core ownership, not the slot.
            int32_t run_core = -1;
            for (int32_t cid = 0; cid < cores_total_num_; cid++) {
                if (core_exec_states_[cid].running_slot_state == &slot_state) {
                    run_core = cid;
                    break;
                }
            }
            if (run_core >= 0) {
                if (cnt_running == 0) {
                    // Snapshot the non-atomic task pointer once: it can be null on a
                    // torn slot, and a concurrent writer may flip it mid-read.
                    PTO2TaskDescriptor *task_ptr = slot_state.task;
                    cls.stuck_task_id = (task_ptr != nullptr) ? static_cast<int64_t>(task_ptr->task_id.raw) : -1;
                    cls.stuck_core = run_core;
                }
                cnt_running++;
                continue;
            }
            int32_t rc = slot_state.fanin_refcount.load(std::memory_order_relaxed);
            int32_t fi = slot_state.fanin_count;
            if (rc >= fi) {
                cnt_ready++;
                continue;
            }
            cnt_waiting++;
        }
    }
    cls.cnt_running = cnt_running;
    cls.cnt_ready = cnt_ready;
    cls.cnt_waiting = cnt_waiting;
    cls.completed = completed_tasks_.load(std::memory_order_relaxed);
    cls.total = total_tasks_;
    cls.orch_done = orchestrator_done_ ? 1 : 0;
    cls.detail = classify_stall_detail(cnt_running, cnt_ready, cnt_waiting, cls.orch_done);
    return cls;
}

int32_t SchedulerContext::handle_timeout_exit(
    int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations,
    int32_t last_progress_count
#if PTO2_PROFILING
    ,
    uint64_t sched_start_ts
#endif
) {
    StallClassification cls = classify_stall_reason();
    LOG_ERROR(
        "[STALL thread=%d idle_iterations=%d] TIMEOUT_EXIT after_idle_iterations=%d sub_class=%s "
        "completed=%d/%d running=%d ready=%d waiting=%d orch_done=%d stuck_task_id=%" PRId64 " stuck_core=%d",
        thread_idx, idle_iterations, idle_iterations, stall_detail_name(cls.detail), cls.completed, cls.total,
        cls.cnt_running, cls.cnt_ready, cls.cnt_waiting, cls.orch_done, cls.stuck_task_id, cls.stuck_core
    );
    // Only the thread that wins the code-100 latch publishes the detail/locators,
    // keeping the host-visible sub-class consistent with the latched code.
    if (latch_scheduler_error(header, thread_idx, PTO2_ERROR_SCHEDULER_TIMEOUT) && header != nullptr) {
        header->sched_stall_completed.store(cls.completed, std::memory_order_relaxed);
        header->sched_stall_total.store(cls.total, std::memory_order_relaxed);
        header->sched_stall_cnt_running.store(cls.cnt_running, std::memory_order_relaxed);
        header->sched_stall_cnt_ready.store(cls.cnt_ready, std::memory_order_relaxed);
        header->sched_stall_cnt_waiting.store(cls.cnt_waiting, std::memory_order_relaxed);
        header->sched_stall_orch_done.store(cls.orch_done, std::memory_order_relaxed);
        header->sched_stall_task_id.store(cls.stuck_task_id, std::memory_order_relaxed);
        header->sched_stall_core.store(cls.stuck_core, std::memory_order_relaxed);
        // detail published last (release) so a host reading a non-NONE detail
        // sees the locators above already settled.
        header->sched_stall_detail.store(cls.detail, std::memory_order_release);
    }
    if (!completed_.exchange(true, std::memory_order_acq_rel)) {
        log_shutdown_stall_snapshot(thread_idx, idle_iterations, last_progress_count);
#if PTO2_PROFILING
        // Capture the in-flight kernels' partial output before signalling the
        // cores to exit, so the dump reflects the live stuck state.
        if (is_dump_args_enabled()) {
            dump_running_task_outputs<PTO2_SUBTASK_SLOT_COUNT>(
                thread_idx, cores_total_num_,
                [this](int32_t cid) {
                    return core_exec_states_[cid].running_slot_state;
                },
                [](ActiveMask active_mask, int raw_subtask_id) {
                    return active_mask.subtask_active(static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                },
                [this](int32_t func_id) {
                    return get_function_bin_addr(func_id);
                }
            );
        }
#endif
        emergency_shutdown(runtime);
    }
#if PTO2_PROFILING
    uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
    aicpu_phase_set_window(
        AicpuPhase::SchedWindow, static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts)
    );
#if PTO2_SCHED_PROFILING
    LOG_INFO_V9(
        "Thread %d: sched_start=%" PRIu64 " sched_end(timeout)=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts),
        cycles_to_us(sched_timeout_ts - sched_start_ts)
    );
#endif
#endif
    return -PTO2_ERROR_SCHEDULER_TIMEOUT;
}

#if PTO2_PROFILING
void SchedulerContext::log_l2_swimlane_summary(int32_t thread_idx, int32_t cur_thread_completed) {
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
    uint64_t sched_end_ts = get_sys_cnt_aicpu();
    // Ride the sched window home to the host phase buffer (the host reduces
    // across sched threads → the `Sched` [STRACE] marker). The verbose
    // per-thread device-log line below is now opt-in deep-dive.
    aicpu_phase_set_window(
        AicpuPhase::SchedWindow, static_cast<uint64_t>(l2_swimlane.sched_start_ts), static_cast<uint64_t>(sched_end_ts)
    );
#if PTO2_SCHED_PROFILING
    LOG_INFO_V9(
        "Thread %d: sched_start=%" PRIu64 " sched_end=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(l2_swimlane.sched_start_ts), static_cast<uint64_t>(sched_end_ts),
        cycles_to_us(sched_end_ts - l2_swimlane.sched_start_ts)
    );
#endif

    uint64_t sched_total = l2_swimlane.sched_wiring_cycle + l2_swimlane.sched_complete_cycle +
                           l2_swimlane.sched_scan_cycle + l2_swimlane.sched_dispatch_cycle +
                           l2_swimlane.sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;

#if PTO2_SCHED_PROFILING
    {
        PTO2SchedProfilingData sp = scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll =
            (l2_swimlane.sched_complete_cycle > otc_total + l2_swimlane.sched_complete_perf_cycle) ?
                (l2_swimlane.sched_complete_cycle - otc_total - l2_swimlane.sched_complete_perf_cycle) :
                0;
        uint64_t dispatch_poll = (l2_swimlane.sched_dispatch_cycle >
                                  l2_swimlane.sched_dispatch_pop_cycle + l2_swimlane.sched_dispatch_setup_cycle) ?
                                     (l2_swimlane.sched_dispatch_cycle - l2_swimlane.sched_dispatch_pop_cycle -
                                      l2_swimlane.sched_dispatch_setup_cycle) :
                                     0;

        LOG_INFO_V9(
            "Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===", thread_idx,
            cycles_to_us(sched_total), cur_thread_completed
        );

        // fanout / fanin per-thread aggregates live in
        // sched_overhead_analysis.compute_dag_stats_from_deps (deps.json edges
        // × core_to_thread).
        LOG_INFO_V9(
            "Thread %d:   complete       : %.3fus (%.1f%%)", thread_idx, cycles_to_us(l2_swimlane.sched_complete_cycle),
            l2_swimlane.sched_complete_cycle * 100.0 / sched_total
        );

        uint64_t c_parent = l2_swimlane.sched_complete_cycle > 0 ? l2_swimlane.sched_complete_cycle : 1;
        uint64_t complete_miss_count = (l2_swimlane.complete_probe_count > l2_swimlane.complete_hit_count) ?
                                           (l2_swimlane.complete_probe_count - l2_swimlane.complete_hit_count) :
                                           0;
        double complete_hit_rate = l2_swimlane.complete_probe_count > 0 ?
                                       l2_swimlane.complete_hit_count * 100.0 / l2_swimlane.complete_probe_count :
                                       0.0;
        LOG_INFO_V9(
            "Thread %d:     poll         : %.3fus (%.1f%%)  hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%",
            thread_idx, cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent,
            static_cast<uint64_t>(l2_swimlane.complete_hit_count), static_cast<uint64_t>(complete_miss_count),
            complete_hit_rate
        );
        LOG_INFO_V9(
            "Thread %d:     otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
            cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle), cycles_to_us(sp.lock_wait_cycle),
            static_cast<uint64_t>(sp.lock_atomic_count)
        );
        LOG_INFO_V9(
            "Thread %d:     otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
            cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle), cycles_to_us(sp.push_wait_cycle),
            static_cast<uint64_t>(sp.fanout_atomic_count)
        );
        LOG_INFO_V9(
            "Thread %d:     otc_fanin    : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.fanin_atomic_count)
        );
        LOG_INFO_V9(
            "Thread %d:     otc_self     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.self_atomic_count)
        );
        LOG_INFO_V9(
            "Thread %d:     perf         : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(l2_swimlane.sched_complete_perf_cycle),
            l2_swimlane.sched_complete_perf_cycle * 100.0 / c_parent
        );

        LOG_INFO_V9(
            "Thread %d:   dispatch       : %.3fus (%.1f%%)", thread_idx, cycles_to_us(l2_swimlane.sched_dispatch_cycle),
            l2_swimlane.sched_dispatch_cycle * 100.0 / sched_total
        );

        uint64_t d_parent = l2_swimlane.sched_dispatch_cycle > 0 ? l2_swimlane.sched_dispatch_cycle : 1;
        LOG_INFO_V9(
            "Thread %d:     poll         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent
        );
        LOG_INFO_V9(
            "Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(l2_swimlane.sched_dispatch_pop_cycle), l2_swimlane.sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(l2_swimlane.sched_dispatch_pop_cycle - sp.pop_wait_cycle), cycles_to_us(sp.pop_wait_cycle),
            static_cast<uint64_t>(sp.pop_atomic_count)
        );
        LOG_INFO_V9(
            "Thread %d:     setup        : %.3fus (%.1f%%)", thread_idx,
            cycles_to_us(l2_swimlane.sched_dispatch_setup_cycle),
            l2_swimlane.sched_dispatch_setup_cycle * 100.0 / d_parent
        );

        LOG_INFO_V9(
            "Thread %d:   scan           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(l2_swimlane.sched_scan_cycle),
            l2_swimlane.sched_scan_cycle * 100.0 / sched_total
        );

#if PTO2_SCHED_PROFILING
        LOG_INFO_V9(
            "Thread %d:   wiring         : %.3fus (%.1f%%)  tasks=%d", thread_idx,
            cycles_to_us(l2_swimlane.sched_wiring_cycle), l2_swimlane.sched_wiring_cycle * 100.0 / sched_total,
            l2_swimlane.phase_wiring_count
        );
#else
        LOG_INFO_V9(
            "Thread %d:   wiring         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(l2_swimlane.sched_wiring_cycle),
            l2_swimlane.sched_wiring_cycle * 100.0 / sched_total
        );
#endif

        LOG_INFO_V9(
            "Thread %d:   idle           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(l2_swimlane.sched_idle_cycle),
            l2_swimlane.sched_idle_cycle * 100.0 / sched_total
        );

        if (cur_thread_completed > 0) {
            LOG_INFO_V9(
                "Thread %d:   avg/complete   : %.3fus", thread_idx,
                cycles_to_us(l2_swimlane.sched_complete_cycle) / cur_thread_completed
            );
        }
    }
#endif
    LOG_INFO_V9(
        "Thread %d: Scheduler summary: total_time=%.3fus, loops=%" PRIu64 ", tasks_scheduled=%d", thread_idx,
        cycles_to_us(sched_total), static_cast<uint64_t>(l2_swimlane.sched_loop_count), cur_thread_completed
    );
}
#endif

// =============================================================================
// Shutdown: deinit AICore regs for this thread's cores.
// Orchestrator threads have core_trackers_[thread_idx].core_num() == 0 -> no-op.
// platform_deinit_aicore_regs is idempotent; safe to call after early completion.
// =============================================================================
int32_t SchedulerContext::shutdown(int32_t thread_idx) {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    int32_t core_num = core_trackers_[thread_idx].core_num();
    if (core_num == 0) return 0;

#if PTO2_PROFILING
    // Restore PMU CTRL registers for this thread's cores before AICore shutdown
    if (is_pmu_enabled()) {
        pmu_aicpu_finalize(cores, core_num);
    }
#endif

    LOG_INFO_V0("Thread %d: Shutting down %d cores", thread_idx, core_num);
    int32_t rc = 0;
    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cores[i];
        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        if (reg_addr != 0) {
            // Timeout means AICore is unresponsive. Log and continue deiniting remaining cores.
            if (platform_deinit_aicore_regs(reg_addr) != 0) {
                LOG_ERROR("Thread %d: Core %d deinit timed out", thread_idx, core_id);
                rc = -1;
            }
        } else {
            LOG_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    LOG_INFO_V0("Thread %d: Shutdown complete", thread_idx);
    return rc;
}

// =============================================================================
// Handshake with all AICore workers; discover core type and reg address.
// =============================================================================
int32_t SchedulerContext::handshake_all_cores(Runtime *runtime) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    cores_total_num_ = runtime->worker_count;

    // Validate cores_total_num_ before using as array index
    if (cores_total_num_ == 0 || cores_total_num_ > RUNTIME_MAX_WORKER) {
        LOG_ERROR("Invalid cores_total_num %d (expected 1-%d)", cores_total_num_, RUNTIME_MAX_WORKER);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    LOG_INFO_V0("Handshaking with %d cores", cores_total_num_);

    // Step 1: Write per-core payload addresses and send handshake signal.
    // OUT_OF_ORDER_STORE_BARRIER() ensures task is globally visible before
    // aicpu_ready=1, so AICore reads the correct payload pointer after waking up.
    for (int32_t i = 0; i < cores_total_num_; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&payload_per_core_[i][0]);
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

        while (hank->aicore_regs_ready == 0) {
            SPIN_WAIT_HINT();
        }

        uint32_t physical_core_id = hank->physical_core_id;

        if (physical_core_id >= max_physical_cores_count) {
            LOG_ERROR(
                "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                max_physical_cores_count
            );
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

        while (hank->aicore_done == 0) {
            SPIN_WAIT_HINT();
        }

        CoreType type = hank->core_type;

        core_exec_states_[i].reg_addr = reg_addr;
        core_exec_states_[i].cond_ptr = get_reg_ptr(reg_addr, RegId::COND);

#if PTO2_PROFILING
        physical_core_ids_[i] = physical_core_id;
#endif

#if !PTO2_PROFILING
        core_exec_states_[i].worker_id = i;
        core_exec_states_[i].physical_core_id = physical_core_id;
        core_exec_states_[i].core_type = type;
#endif

        if (type == CoreType::AIC) {
            aic_worker_ids_[aic_count_++] = i;
            LOG_INFO_V0("Core %d: AIC, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        } else {
            aiv_worker_ids_[aiv_count_++] = i;
            LOG_INFO_V0("Core %d: AIV, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        }
    }

    if (handshake_failed) {
        emergency_shutdown(runtime);
        return -1;
    }

    LOG_INFO_V0("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);
    return 0;
}

// =============================================================================
// Assign discovered cores to scheduler threads (cluster-aligned round-robin).
// =============================================================================
bool SchedulerContext::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    active_sched_threads_ = (sched_thread_num_ > 0) ? sched_thread_num_ : aicpu_thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
    int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
    int32_t thread_cores_num = max_clusters_per_thread * 3;

    if (thread_cores_num > CoreTracker::MAX_CORE_PER_THREAD) {
        LOG_ERROR("Can't assign more then 64 cores in per scheduler");
        return false;
    }

    LOG_INFO_V0(
        "Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)", cluster_count,
        active_sched_threads_, aic_count_, aiv_count_
    );

    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
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
    }

    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % active_sched_threads_;

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);

        LOG_INFO_V0("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        LOG_INFO_V0(
            "Thread %d: total %d cores (%d clusters)", t, core_trackers_[t].core_num(),
            core_trackers_[t].get_cluster_count()
        );
    }

    LOG_INFO_V0(
        "Config: threads=%d, cores=%d, cores_per_thread=%d", aicpu_thread_num_, cores_total_num_, thread_cores_num
    );
    return true;
}

// =============================================================================
// Reassign all cores across all threads (sched + orchestrator) after orchestration.
// =============================================================================
void SchedulerContext::reassign_cores_for_all_threads() {
    LOG_INFO_V0(
        "Reassigning cores (cluster-aligned) for %d threads: %d AIC, %d AIV", aicpu_thread_num_, aic_count_, aiv_count_
    );

    // Collect running worker_ids from all current trackers
    bool running_cores[RUNTIME_MAX_WORKER] = {};
    for (int32_t i = 0; i < aicpu_thread_num_; i++) {
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
        clusters_per_thread[ci % aicpu_thread_num_]++;
    }

    // Re-init all trackers and reset core counts
    for (int32_t i = 0; i < aicpu_thread_num_; i++) {
        core_trackers_[i].init(clusters_per_thread[i]);
    }

    // Assign clusters round-robin and restore running state
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % aicpu_thread_num_;

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
    }

    // Log final distribution
    LOG_INFO_V0("Core reassignment complete:");
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        int32_t aic_running = core_trackers_[t].get_running_count<CoreType::AIC>();
        int32_t aiv_running = core_trackers_[t].get_running_count<CoreType::AIV>();
        LOG_INFO_V0(
            "  Thread %d: %d cores, %d clusters (AIC running=%d, AIV running=%d)", t, core_trackers_[t].core_num(),
            core_trackers_[t].get_cluster_count(), aic_running, aiv_running
        );
    }
    active_sched_threads_ = aicpu_thread_num_;
}

// =============================================================================
// Emergency shutdown: broadcast exit signal to every handshake'd core and
// deinit their AICore register blocks. Idempotent.
// =============================================================================
void SchedulerContext::emergency_shutdown(Runtime *runtime) {
    LOG_WARN("Emergency shutdown: sending exit signal to all initialized cores");
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    int32_t timeout_count = 0;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];
        OUT_OF_ORDER_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;
        if (core_exec_states_[i].reg_addr != 0) {
            if (platform_deinit_aicore_regs(core_exec_states_[i].reg_addr) != 0) {
                timeout_count++;
            }
        }
    }
    if (timeout_count > 0) {
        LOG_ERROR("Emergency shutdown: %d cores did not acknowledge exit", timeout_count);
    }
    LOG_WARN("Emergency shutdown complete");
}

// =============================================================================
// Lifecycle: init / deinit
// =============================================================================
int32_t SchedulerContext::init(
    Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, bool orch_to_sched, uint64_t regs_base
) {
    always_assert(runtime != nullptr);

    // Zero all per-core execution state before handshake
    memset(core_exec_states_, 0, sizeof(core_exec_states_));

    // Wire thread/transition configuration that handshake/assign need to read.
    aicpu_thread_num_ = aicpu_thread_num;
    sched_thread_num_ = sched_thread_num;
    orch_to_sched_ = orch_to_sched;
    regs_ = regs_base;

#if PTO2_PROFILING
    // l2_swimlane_aicpu_init promotes g_l2_swimlane_level from the shared-memory
    // header — must be called BEFORE the orchestrator thread caches the level
    // via rt->orchestrator.l2_swimlane_level = get_l2_swimlane_level() in
    // AicpuExecutor::run(). Otherwise the cached value would still be DISABLED
    // (only the binary enable bit has been seeded by kernel.cpp at this point),
    // and the CYCLE_COUNT_START() gate in pto_orchestrator.cpp would suppress
    // all ORCH_PHASES records. Reset the cached level on disabled runs so a
    // prior enabled launch's level can't leak into the phase-record gates in
    // scheduler_dispatch (`>= SCHED_PHASES`).
    if (is_l2_swimlane_enabled()) {
        l2_swimlane_aicpu_init(runtime->worker_count);
        l2_swimlane_level_ = get_l2_swimlane_level();
        if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
            // When orchestrator phases merge into scheduler threads
            // (PTO2_ORCH_TO_SCHED=1), phase records flow through
            // aicpu_thread_num_ pools — matches the same branch in the
            // dump_args_init call below.
            // Sched phase pool count = number of scheduler threads.
            // This block runs before assign_cores_to_threads, so the
            // active_sched_threads_ member isn't set yet — recompute the same
            // normalization locally: sched_thread_num_ <= 0 is the "use all
            // AICPU threads as scheduler threads" sentinel (see
            // assign_cores_to_threads' active_sched_threads_). Without this
            // normalization here, init_phase would prime zero sched pools
            // and all sched_phase emits would silently drop.
            const int active_sched = (sched_thread_num_ > 0) ? sched_thread_num_ : aicpu_thread_num_;
            const int sched_phase_threads = orch_to_sched_ ? aicpu_thread_num_ : active_sched;
            // Orch phase is a single instance (PR #971 design), so the orch
            // pool count is always 1 regardless of orch_to_sched mode.
            const int orch_phase_threads = 1;
            l2_swimlane_aicpu_init_phase(runtime->worker_count, sched_phase_threads, orch_phase_threads);
        }
    } else {
        l2_swimlane_level_ = L2SwimlaneLevel::DISABLED;
    }
#endif

    // Discover cores and assign to scheduler threads.
    int32_t rc = handshake_all_cores(runtime);
    if (rc != 0) {
        LOG_ERROR("handshake_all_cores failed");
        return rc;
    }
    if (!assign_cores_to_threads()) {
        return -1;
    }

    // Profiling-subsystem buffer/state init: single-threaded cold path, so the
    // "do it once" guarantee is structural (no CAS needed). Runs after
    // handshake_all_cores / assign_cores_to_threads because pmu_aicpu_init needs
    // physical_core_ids_ / cores_total_num_. Mirrors the l2_swimlane_aicpu_init
    // convention above; the per-thread *_set_orch_thread_idx setters stay on the
    // orchestrator thread (see aicpu_executor.cpp).
#if PTO2_PROFILING
    if (is_dump_args_enabled()) {
        dump_args_init(orch_to_sched_ ? aicpu_thread_num_ : active_sched_threads_);
    }
    if (is_pmu_enabled()) {
        pmu_aicpu_init(physical_core_ids_, cores_total_num_);
        LOG_INFO_V0("PMU profiling started on %d cores", cores_total_num_);
    }
    // dep_gen is host-driven (SubmitTrace) — runtime-gated by the host flag —
    // and compiles out with the other profiling subsystems at PTO2_PROFILING=0.
    // init() only pops the initial buffer from instance 0's free_queue; the
    // orchestrator thread still records its idx via
    // dep_gen_aicpu_set_orch_thread_idx() before the first record_submit.
    if (is_dep_gen_enabled()) {
        dep_gen_aicpu_init();
    }
#endif

    // Initialize task counters. Task count comes from PTO2 shared memory.
    if (runtime->get_gm_sm_ptr()) {
        auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
        // Read at one-time boot init, before the SM is reset for the run, so a
        // ring not yet written holds uninitialized memory (0xbe... under ASAN's
        // malloc-fill). Sum in int64 and only count rings whose value is a
        // plausible task count — (0, PTO2_SCOPE_TASKS_CAP]; a ring cannot hold
        // more than the scope cap. This rejects any garbage pattern (negative
        // or positive), so uninitialized rings contribute 0 (the correct boot
        // count) while valid counts still add up, with no signed overflow.
        int64_t task_count = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            int32_t ring_tasks = header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
            if (ring_tasks > 0 && ring_tasks <= PTO2_SCOPE_TASKS_CAP) task_count += ring_tasks;
        }
        total_tasks_ = static_cast<int32_t>(task_count);
    } else {
        total_tasks_ = 0;
    }
    completed_tasks_.store(0, std::memory_order_release);

    // Device orchestration: the orchestrator thread flips this when the graph is built.
    orchestrator_done_.store(false, std::memory_order_release);

    // Clear per-core dispatch payloads
    memset(payload_per_core_, 0, sizeof(payload_per_core_));
    memset(deferred_slab_per_core_, 0, sizeof(deferred_slab_per_core_));

    // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
    // This is done once at startup and never modified afterwards.
    for (int32_t t = 0; t < sched_thread_num_; t++) {
        CoreTracker &tracker = core_trackers_[t];
        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
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

void SchedulerContext::deinit() {
    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i] = {};
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // No per-core memset of payload_per_core_ / deferred_slab_per_core_ here
    // (~300 KB across all cores). Both are fully re-initialized at dispatch
    // before they can be read: dispatch_task sets deferred_slab->count = 0 /
    // error_code = NONE and build_payload() overwrites every payload field
    // (function addr, args[], contexts, not_ready) on the exact [core][buf_idx]
    // about to run. The consumer side cannot reach a stale slot either: the
    // drain only services a core's running_reg_task_id, and the loop above
    // already reset every core_exec_states_[].running/pending_reg_task_id to
    // AICPU_TASK_INVALID — so no FIN for an undispatched slot is processed, and
    // the count-gated consumer never reads entries[] past the fresh count.

    // Reset sync-start drain coordination — a previous run that aborted mid-drain
    // would otherwise leave dirty pending/elected/ack state for the next reuse.
    drain_state_.sync_start_pending.store(0, std::memory_order_release);
    drain_state_.drain_worker_elected.store(0, std::memory_order_release);
    drain_state_.drain_ack_mask.store(0, std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);

    // Reset task counters and orchestrator state
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    orchestrator_done_.store(false, std::memory_order_release);

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
    for (int32_t t = 0; t < MAX_AICPU_THREADS; t++) {
        core_trackers_[t] = CoreTracker{};
    }

    regs_ = 0;
    sched_ = nullptr;
    rt_ = nullptr;
    func_id_to_addr_ = nullptr;
}

void SchedulerContext::bind_runtime(PTO2Runtime *rt) {
    rt_ = rt;
    sched_ = &rt->scheduler;
}

void SchedulerContext::wait_for_orchestration_done_before_dispatch(Runtime *runtime, int32_t thread_idx) {
    while (!orchestration_done() && !completed_.load(std::memory_order_acquire)) {
        if (thread_idx == 0 && sched_ != nullptr) {
            // Use the wiring subsystem's normal batch/backoff policy while
            // waiting. This still honors orch_needs_drain/producer_blocked
            // signals without force-draining an empty queue every spin.
            int wired = sched_->drain_wiring_queue(/*force_drain=*/false);
            if (wired > 0) {
                continue;
            }
        }
        if (sched_ != nullptr && sched_->sm_header != nullptr &&
            check_idle_fatal_error(thread_idx, sched_->sm_header, runtime) == LoopAction::BREAK_LOOP) {
            break;
        }
        SPIN_WAIT_HINT();
    }
}

// =============================================================================
// Post-orchestration bookkeeping. Runs on the orchestrator thread once the
// build phase finishes; folds inline-completed tasks, flips orchestrator_done_,
// and drives the orchestrator → scheduler core transition (or fatal shutdown).
// =============================================================================
void SchedulerContext::on_orchestration_done(
    Runtime *runtime, PTO2Runtime *rt, int32_t thread_idx, int32_t total_tasks
) {
#if PTO2_PROFILING
    if (l2_swimlane_level_ >= L2SwimlaneLevel::ORCH_PHASES) {
        // Flush orchestrator's phase record buffer (orch pool, ordinal 0)
        l2_swimlane_aicpu_flush_orch_phase_buffer(thread_idx);
    }
#endif

    total_tasks_ = total_tasks;

    // Fold tasks completed inline during orchestration
    int32_t inline_completed = static_cast<int32_t>(rt->orchestrator.inline_completed_tasks);
    if (inline_completed > 0) {
        completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
#if PTO2_SCHED_PROFILING
        rt->scheduler.tasks_completed.fetch_add(inline_completed, std::memory_order_relaxed);
#endif
    }
    orchestrator_done_.store(true, std::memory_order_release);

    // Check for fatal error from orchestration; if so, shut down immediately.
    int32_t orch_err = 0;
    if (sched_->sm_header) {
        orch_err = sched_->sm_header->orch_error_code.load(std::memory_order_relaxed);
    }
    if (orch_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) {
            emergency_shutdown(runtime);
        }
    }

    // Skip core transition on fatal error — cores already shut down above.
    if (completed_.load(std::memory_order_acquire)) {
        // Signal transition to unblock scheduler threads waiting at core transition
        transition_requested_.store(true, std::memory_order_release);
        reassigned_.store(true, std::memory_order_release);
    } else if (orch_to_sched_) {
        LOG_INFO_V0("Thread %d: Set orchestrator_done=true, requesting core transition", thread_idx);
        transition_requested_.store(true, std::memory_order_release);

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

#if PTO2_PROFILING
    // Write core-to-thread mapping AFTER reassignment so the profiling data
    // reflects the final distribution (all active_sched_threads_, including
    // former orchestrator threads when orch_to_sched_ is enabled).
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        l2_swimlane_aicpu_init_core_assignments(cores_total_num_);
        for (int32_t t = 0; t < active_sched_threads_; t++) {
            l2_swimlane_aicpu_write_core_assignments_for_thread(
                t, core_trackers_[t].core_ids(), core_trackers_[t].core_num()
            );
        }
    }
#endif
}
