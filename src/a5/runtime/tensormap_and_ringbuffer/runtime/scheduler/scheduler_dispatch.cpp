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

#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/l2_perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"

// =============================================================================
// Dispatch helpers
// =============================================================================

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;
}

const char *SchedulerContext::shape_name(PTO2ResourceShape shape) {
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

const PTO2ResourceShape *SchedulerContext::get_dispatch_order(int32_t thread_idx) {
    static constexpr PTO2ResourceShape kEvenOrder[PTO2_NUM_RESOURCE_SHAPES] = {
        PTO2ResourceShape::MIX,
        PTO2ResourceShape::AIC,
        PTO2ResourceShape::AIV,
    };
    static constexpr PTO2ResourceShape kOddOrder[PTO2_NUM_RESOURCE_SHAPES] = {
        PTO2ResourceShape::MIX,
        PTO2ResourceShape::AIV,
        PTO2ResourceShape::AIC,
    };
    return (thread_idx % 2 == 0) ? kEvenOrder : kOddOrder;
}

int SchedulerContext::pop_ready_tasks_batch(
    PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
) {
#if PTO2_SCHED_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
    extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
    uint64_t t_pop_start = get_sys_cnt_aicpu();
    int count = sched_->get_ready_tasks_batch(
        shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx],
        l2_perf.local_dispatch_count
    );
    l2_perf.sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
    if (count > 0) {
        l2_perf.pop_hit += count;
    } else {
        l2_perf.pop_miss++;
    }
#else
    (void)thread_idx;
    int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
    return count;
}

void SchedulerContext::build_payload(
    PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    const AsyncCtx &async_ctx
) {
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
    dispatch_payload.local_context.s_block_idx = slot_state.next_block_idx;
    dispatch_payload.local_context.s_block_num = slot_state.logical_block_num;
    dispatch_payload.local_context.async_ctx = async_ctx;
    dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
    dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
}

void SchedulerContext::dispatch_subtask_to_core(
    Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    bool to_pending
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto core_id = tracker.get_core_id_by_offset(core_offset);
#if PTO2_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
    (void)runtime;
#else
    (void)runtime;
#endif
    CoreExecState &core_exec_state = core_exec_states_[core_id];
    core_exec_state.dispatch_seq++;
    uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
    static_assert(
        (TASK_ID_MASK - AICORE_EXIT_SIGNAL + 1) % 2 == 0, "Sentinel skip must be even to preserve dual-buffer parity"
    );
    if (reg_task_id >= AICORE_EXIT_SIGNAL) {
        core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
        reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
    }

    uint32_t buf_idx = reg_task_id & 1u;
    PTO2DispatchPayload &payload = payload_per_core_[core_id][buf_idx];
    PTO2DeferredCompletionIngressBuffer *deferred_ingress = &deferred_ingress_per_core_[core_id][buf_idx];
    deferred_ingress->count = 0;
    deferred_ingress->error_code = PTO2_ERROR_NONE;
    AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_ingress);
    build_payload(payload, slot_state, subslot, async_ctx);

    if (to_pending) {
        core_exec_state.pending_subslot = subslot;
        core_exec_state.pending_slot_state = &slot_state;
        core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
        if (l2_perf.l2_perf_enabled) {
            core_exec_state.pending_dispatch_timestamp = get_sys_cnt_aicpu();
        }
#endif
    } else {
        core_exec_state.running_subslot = subslot;
        core_exec_state.running_slot_state = &slot_state;
        core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
#if PTO2_PROFILING
        if (l2_perf.l2_perf_enabled) {
            core_exec_state.running_dispatch_timestamp = get_sys_cnt_aicpu();
        }
#endif
        tracker.change_core_state(core_offset);
    }

    write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));
    tracker.set_pending_occupied(core_offset);
}

void SchedulerContext::dispatch_mix_block_to_cluster(
    Runtime *runtime, int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    uint8_t cmask = slot_state.active_mask.core_mask();
    if (cmask & PTO2_SUBTASK_MASK_AIC) {
        bool aic_to_pending = to_pending && !tracker.is_aic_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC,
            aic_to_pending
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV0) {
        bool aiv0_to_pending = to_pending && !tracker.is_aiv0_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aiv0_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV0,
            aiv0_to_pending
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV1) {
        bool aiv1_to_pending = to_pending && !tracker.is_aiv1_core_idle(cluster_offset);
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aiv1_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV1,
            aiv1_to_pending
        );
    }
}

void SchedulerContext::dispatch_block(
    Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape,
    bool to_pending
) {
#if PTO2_PROFILING
    if (is_dump_tensor_enabled()) {
        dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
            thread_idx, slot_state, TensorDumpStage::BEFORE_DISPATCH,
            [](ActiveMask active_mask, int raw_subtask_id) {
                return active_mask.subtask_active(static_cast<PTO2SubtaskSlot>(raw_subtask_id));
            },
            [this](int32_t func_id) {
                return get_function_bin_addr(func_id);
            }
        );
    }
#endif
    if (shape == PTO2ResourceShape::MIX) {
        dispatch_mix_block_to_cluster(runtime, thread_idx, core_offset, slot_state, to_pending);
    } else if (shape == PTO2ResourceShape::AIC) {
        dispatch_subtask_to_core(runtime, thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending);
    } else {
        dispatch_subtask_to_core(runtime, thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending);
    }
#if PTO2_PROFILING
    sched_l2_perf_[thread_idx].phase_dispatch_count += __builtin_popcount(slot_state.active_mask.core_mask());
#endif
}

void SchedulerContext::dispatch_shape(
    Runtime *runtime, int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
    PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
) {
#if PTO2_SCHED_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
#endif
    if (entered_drain) return;

    bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
    auto cores = tracker.get_dispatchable_cores(shape, phase);
    if (!cores.has_value()) return;

    while (cores.has_value() && !entered_drain) {
        int want = cores.count();
        PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
        int got = pop_ready_tasks_batch(shape, thread_idx, local_buf, batch, want);
        if (got == 0) break;

        bool dispatched_any = false;
        for (int bi = 0; bi < got; bi++) {
            PTO2TaskSlotState *slot_state = batch[bi];

            if (slot_state->active_mask.requires_sync_start()) {
                if (is_pending) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
                int32_t available = cores.count();
                if (available < slot_state->logical_block_num) {
                    if (!enter_drain_mode(slot_state, slot_state->logical_block_num)) {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    }
                    for (int rem = bi + 1; rem < got; rem++) {
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                    }
                    entered_drain = true;
                    break;
                }
            }

            if (!cores.has_value()) {
                sched_->ready_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                break;
            }

            dispatched_any = true;
            try_pushed = true;
#if PTO2_SCHED_PROFILING
            uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
            do {
                auto core_offset = cores.pop_first();
                dispatch_block(runtime, thread_idx, core_offset, *slot_state, shape, is_pending);
                slot_state->next_block_idx++;
                LOG_DEBUG(
                    "Thread %d: Dispatched %s %s task %" PRId64 " block %d/%d to core_offset %d", thread_idx,
                    is_pending ? "pending" : "idle", shape_name(shape),
                    static_cast<int64_t>(slot_state->task->task_id.raw), slot_state->next_block_idx - 1,
                    slot_state->logical_block_num, core_offset
                );
            } while (slot_state->next_block_idx < slot_state->logical_block_num && cores.has_value());

            if (slot_state->next_block_idx < slot_state->logical_block_num) {
                sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
            }
            made_progress = true;
#if PTO2_SCHED_PROFILING
            l2_perf.sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
        }

        if (!dispatched_any) break;

        if (!cores.has_value()) {
            cores = tracker.get_dispatchable_cores(shape, phase);
        }
    }
}

// =============================================================================
// Main scheduler dispatch loop
// =============================================================================

int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    LOG_INFO_V0("Thread %d: resolve_and_dispatch entry", thread_idx);

    PTO2SharedMemoryHeader *header = sched_->sm_header;
    if (!header) {
        LOG_ERROR("PTO2 dispatch: header is null");
        return -1;
    }
    LOG_INFO_V0(
        "Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu", thread_idx, static_cast<void *>(header),
        static_cast<uint64_t>(header->rings[0].task_descriptors_offset),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    Handshake *hank = static_cast<Handshake *>(runtime->workers);
    LOG_INFO_V0(
        "Thread %d: hank=%p, window_size=%lu", thread_idx, static_cast<void *>(hank),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!init_done_.exchange(true, std::memory_order_acq_rel)) {
        LOG_INFO_V0("Thread %d: doing one-time init", thread_idx);

#if PTO2_PROFILING
        if (is_l2_swimlane_enabled()) {
            l2_perf_aicpu_init_profiling(runtime);
            l2_perf_aicpu_init_phase_profiling(sched_thread_num_);
            l2_perf_aicpu_set_orch_thread_idx(sched_thread_num_);
        }
#endif
#if PTO2_PROFILING
        if (is_dump_tensor_enabled()) {
            dump_tensor_init(orch_to_sched_ ? thread_num_ : sched_thread_num_);
        }
        if (is_pmu_enabled()) {
            pmu_aicpu_init(runtime->workers, physical_core_ids_, cores_total_num_);
            LOG_INFO_V0("PMU profiling started on %d cores", cores_total_num_);
        }
#endif

        LOG_INFO_V0("Thread %d: one-time init done", thread_idx);
        init_complete_.store(true, std::memory_order_release);
    } else {
        while (!init_complete_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    LOG_INFO_V0("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, tracker.core_num());
    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if PTO2_PROFILING
    auto &l2_perf = sched_l2_perf_[thread_idx];
    l2_perf.reset();
    l2_perf.l2_perf_enabled = is_l2_swimlane_enabled();
#endif

    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);
    }
    PTO2TaskSlotState *deferred_release_slot_states[PTO2_DEFERRED_RELEASE_CAP];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

#if PTO2_PROFILING
    l2_perf.sched_start_ts = get_sys_cnt_aicpu();
#endif

    while (true) {
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        l2_perf.sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (!tracker.has_any_running_cores()) {
            LoopAction action = handle_orchestrator_exit(thread_idx, header, runtime, task_count);
            if (action == LoopAction::BREAK_LOOP) break;
        }

        if (!cores_released && orch_to_sched_) {
            LoopAction action = handle_core_transition(cores_released);
            if (action == LoopAction::BREAK_LOOP) break;
        }

#if PTO2_PROFILING
        CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
#endif

        // Phase 1: Check running cores for completion
        int32_t completed_this_turn = 0;

        bool try_completed = tracker.has_any_running_cores();
        if (try_completed) {
            check_running_cores_for_completion(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
            );
        }
        if (completed_this_turn > 0) {
#if PTO2_SCHED_PROFILING
            sched_->tasks_completed.fetch_add(completed_this_turn, std::memory_order_relaxed);
#endif
            int32_t prev = completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            int32_t new_total = prev + completed_this_turn;
            last_progress_count = new_total;
            if (thread_idx == 0 && task_count > 0) {
                if (new_total <= PROGRESS_VERBOSE_THRESHOLD ||
                    new_total / PROGRESS_LOG_INTERVAL != prev / PROGRESS_LOG_INTERVAL || new_total >= task_count) {
                    LOG_INFO_V9(
                        "PTO2 progress: completed=%d total=%d (%.1f%%)", new_total, task_count,
                        100.0 * new_total / task_count
                    );
                }
            }
        }

        if (rt_ != nullptr && rt_->completion_ingress != nullptr &&
            (sched_->async_wait_list.count > 0 || sched_->async_wait_list.pending_completion_count > 0)) {
            PTO2AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(
                rt_->completion_ingress, sched_, local_bufs, deferred_release_slot_states, deferred_release_count,
                PTO2_DEFERRED_RELEASE_CAP
#if PTO2_SCHED_PROFILING
                ,
                thread_idx
#endif
            );
            if (poll_result.error_code != PTO2_ERROR_NONE) {
                int32_t expected = PTO2_ERROR_NONE;
                header->sched_error_code.compare_exchange_strong(
                    expected, poll_result.error_code, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                break;
            }
            if (poll_result.completed > 0) {
#if PTO2_SCHED_PROFILING
                sched_->tasks_completed.fetch_add(poll_result.completed, std::memory_order_relaxed);
#endif
                int32_t prev = completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                int32_t new_total = prev + poll_result.completed;
                last_progress_count = new_total;
                made_progress = true;
            }
        }

#if PTO2_PROFILING
        if (!try_completed) {
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_perf.sched_complete_cycle);
            if (l2_perf.l2_perf_enabled && l2_perf.phase_complete_count > 0) {
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, l2_perf.sched_loop_count,
                    l2_perf.phase_complete_count
                );
                _t0_phase = _t1;
                l2_perf.phase_complete_count = 0;
            }
        }
#endif

        bool try_pushed = false;

        // Phase 2 drain check
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(runtime, thread_idx);
            continue;
        }

        // Phase 3: Drain wiring queue (thread 0 only)
        if (thread_idx == 0) {
            int wired = sched_->drain_wiring_queue(orchestrator_done_);
            if (wired > 0) {
                made_progress = true;
#if PTO2_SCHED_PROFILING
                l2_perf.phase_wiring_count += wired;
#endif
            }
        }
#if PTO2_PROFILING
        CYCLE_COUNT_LAP(l2_perf.sched_wiring_cycle);
#endif

        // Phase 4: Two-phase dispatch (idle then pending)
        const PTO2ResourceShape *dispatch_order = get_dispatch_order(thread_idx);
        bool entered_drain = false;

        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES && !entered_drain; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            for (auto phase : {CoreTracker::DispatchPhase::IDLE, CoreTracker::DispatchPhase::PENDING}) {
                dispatch_shape(
                    runtime, thread_idx, shape, phase, local_bufs[static_cast<int32_t>(shape)], tracker, entered_drain,
                    made_progress, try_pushed
                );
            }
        }

        // Requeue local buffers to global ready queue
        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            auto &local_buf = local_bufs[static_cast<int32_t>(shape)];
            auto &ready_queue = sched_->ready_queues[static_cast<int32_t>(shape)];
#if PTO2_SCHED_PROFILING
            l2_perf.local_overflow_count += local_buf.count;
#endif
            if (local_buf.count > 0) {
                ready_queue.push_batch(local_buf.slot_states, local_buf.count);
                local_buf.count = 0;
            }
        }

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_perf.sched_dispatch_cycle);
            if (l2_perf.l2_perf_enabled && l2_perf.phase_dispatch_count > 0) {
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, l2_perf.sched_loop_count,
                    l2_perf.phase_dispatch_count
                );
                _t0_phase = _t1;
                l2_perf.phase_dispatch_count = 0;
            }
        }
#endif

#if !PTO2_PROFILING
        (void)try_completed;
        (void)try_pushed;
#endif

        if (made_progress) {
            idle_iterations = 0;
        } else {
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                int32_t fe =
                    sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
                l2_perf.fanin_edges_total += fe;
                if (fe > l2_perf.fanin_max_degree) l2_perf.fanin_max_degree = fe;
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
            idle_iterations++;

            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                LoopAction action = check_idle_fatal_error(thread_idx, header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                log_stall_diagnostics(thread_idx, task_count, idle_iterations, last_progress_count);
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                return handle_timeout_exit(
                    thread_idx, idle_iterations
#if PTO2_PROFILING
                    ,
                    l2_perf.sched_start_ts
#endif
                );
            } else {
                SPIN_WAIT_HINT();
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(l2_perf.sched_idle_cycle);
            if (l2_perf.l2_perf_enabled) {
                l2_perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_IDLE_WAIT, _t0_phase, _t1, l2_perf.sched_loop_count, 0
                );
                _t0_phase = _t1;
            }
#endif
        }
    }

#if PTO2_PROFILING
    log_l2_perf_summary(thread_idx, cur_thread_completed);
#endif

#if PTO2_PROFILING
    if (is_dump_tensor_enabled()) {
        dump_tensor_flush(thread_idx);
    }
#endif

    return cur_thread_completed;
}
