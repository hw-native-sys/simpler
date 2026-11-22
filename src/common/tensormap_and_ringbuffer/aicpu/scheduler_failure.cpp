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
#include "scheduler/scheduler_context.h"

#include <cinttypes>

#include "aicpu/args_dump_aicpu.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_time.h"
#include "runtime_core.h"

using simpler::tmr::ExplicitAbort;
using simpler::tmr::OrchestrationSetupFailure;
using simpler::tmr::ProgressTimeout;
using simpler::tmr::SharedFailure;

LoopAction SchedulerContext::handle_orchestrator_exit(
    int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime, int32_t &task_count
) {
    if (check_idle_fatal_error(thread_idx, header, runtime) == LoopAction::BREAK_LOOP) return LoopAction::BREAK_LOOP;

    bool orch_done = orchestrator_done_.load(std::memory_order_acquire);
    if (!orch_done) return LoopAction::NONE;

    // orchestrator_done_ publishes the final task count, including zero.
    task_count = total_tasks_;
    if (completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
        completed_.store(true, std::memory_order_release);
        LOG_INFO(
            "Thread %d: completed tasks %d/%d", thread_idx, completed_tasks_.load(std::memory_order_relaxed), task_count
        );
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::check_idle_fatal_error(
    int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations,
    int32_t last_progress_count
) {
    if (completed_.load(std::memory_order_acquire)) {
        return LoopAction::BREAK_LOOP;
    }
    const int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != SIMPLER_ERROR_NONE) {
        handle_failure(
            runtime, thread_idx,
            SharedFailure{SharedFailure::Source::Orchestrator, orch_err, idle_iterations, last_progress_count}
        );
        return LoopAction::BREAK_LOOP;
    }
    const int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != SIMPLER_ERROR_NONE) {
        handle_failure(
            runtime, thread_idx,
            SharedFailure{SharedFailure::Source::Scheduler, sched_err, idle_iterations, last_progress_count}
        );
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

void SchedulerContext::record_progress_timeout(
    int32_t thread_idx, SharedMemoryHeader *header, int32_t idle_iterations
) {
    StallClassification cls = classify_stall_reason();
    LOG_ERROR(
        "[STALL thread=%d idle_iterations=%d] TIMEOUT_EXIT after_idle_iterations=%d sub_class=%s "
        "completed=%d/%d running=%d ready=%d waiting=%d async_wait=%d orch_done=%d stuck_task_id=0x%" PRIx64
        " stuck_core=%d",
        thread_idx, idle_iterations, idle_iterations, stall_detail_name(cls.detail), cls.completed, cls.total,
        cls.cnt_running, cls.cnt_ready, cls.cnt_waiting, cls.cnt_async_wait, cls.orch_done,
        TaskId::to_uint64(cls.stuck_task_id), cls.stuck_core
    );
    report_scheduler_timeout(header, thread_idx, cls);
}

int32_t SchedulerContext::handle_timeout_exit(
    int32_t thread_idx, SharedMemoryHeader *header, Runtime *runtime, int32_t idle_iterations,
    int32_t last_progress_count
#if SIMPLER_DFX
    ,
    uint64_t sched_start_ts
#endif
) {
    handle_failure(runtime, thread_idx, ProgressTimeout{header, idle_iterations, last_progress_count});
#if SIMPLER_DFX
    uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
    aicpu_phase_set_window(
        AicpuPhase::SchedWindow, static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts)
    );
#if SIMPLER_SCHED_PROFILING
    LOG_INFO(
        "Thread %d: sched_start=%" PRIu64 " sched_end(timeout)=%" PRIu64 " sched_cost=%.3fus", thread_idx,
        static_cast<uint64_t>(sched_start_ts), static_cast<uint64_t>(sched_timeout_ts),
        cycles_to_us(sched_timeout_ts - sched_start_ts)
    );
#endif
#endif
    return -SIMPLER_ERROR_SCHEDULER_TIMEOUT;
}

void SchedulerContext::emergency_shutdown(Runtime *runtime) { handle_failure(runtime, -1, ExplicitAbort{}); }

void SchedulerContext::handle_failure(int32_t thread_idx, simpler::tmr::LocalSchedulerFailure failure) {
    // The detecting thread publishes the first code before waking the other schedulers to retire their own cores.
    latch_scheduler_error(failure.header, thread_idx, failure.code);
    completed_.store(true, std::memory_order_release);
}

void SchedulerContext::handle_failure(Runtime *runtime, int32_t thread_idx, const simpler::tmr::Failure &failure) {
    if (const auto *setup = std::get_if<OrchestrationSetupFailure>(&failure)) {
        // No bound scheduler or valid error header yet. Wake peers to skip dispatch,
        // retire their own cores and meet the reporting thread at the completion gate.
        setup->runtime_init_ready.store(true, std::memory_order_release);
        return;
    }

    const auto *shared = std::get_if<SharedFailure>(&failure);
    const bool from_orch = shared != nullptr && shared->source == SharedFailure::Source::Orchestrator;
    const bool tensor_wait = from_orch && shared->code == SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT;
    if (shared != nullptr && thread_idx >= 0) {
        LOG_ERROR(
            "Thread %d: %s fatal error detected (code=%d)", thread_idx, from_orch ? "Orchestrator" : "Scheduler",
            shared->code
        );
        if (tensor_wait) log_owned_dispatch_records(thread_idx, StallDumpReport::Shutdown);
    }

    // Publishing a timeout snapshot and electing the thread that retires the
    // cores are independent. A peer can observe the error and win shutdown.
    const auto *timeout = std::get_if<ProgressTimeout>(&failure);
    if (timeout != nullptr) record_progress_timeout(thread_idx, timeout->header, timeout->idle_iterations);
    if (!begin_emergency_shutdown()) return;
    if (timeout != nullptr) {
        log_owned_dispatch_records(thread_idx, StallDumpReport::Shutdown);
        log_shutdown_stall_snapshot(
            thread_idx, timeout->idle_iterations, timeout->last_progress_count, "scheduler_timeout"
        );
#if SIMPLER_DFX
        // Capture the in-flight kernels' partial output before signalling the
        // cores to exit, so the dump reflects the live stuck state.
        if (is_dump_args_enabled()) {
            dump_running_task_outputs(
                cores_total_num_,
                [this](int32_t cid) {
                    return core_exec_states_[cid].running_slot_state;
                },
                [this, thread_idx](const ChipTaskSlotState &slot_state) {
                    dump_args_for_task<SUBTASK_SLOT_COUNT>(
                        thread_idx, *slot_state.task, *slot_state.payload, slot_state.active_mask,
                        ArgsDumpStage::AFTER_COMPLETION, [this](int32_t func_id) {
                            return get_function_bin_addr(func_id);
                        }
                    );
                }
            );
        }
#endif
    } else {
        log_fatal_shutdown(thread_idx, shared);
    }
    signal_emergency_shutdown(runtime);
}
