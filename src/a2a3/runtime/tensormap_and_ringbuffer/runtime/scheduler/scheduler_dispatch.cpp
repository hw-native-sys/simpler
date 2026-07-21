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

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "common.h"
#include "callable.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"

int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {
    always_assert(sched_ != nullptr);
    CoreTracker &tracker = core_trackers_[thread_idx];

    PTO2SharedMemoryHeader *header = sched_->sm_header;
    if (!header) return -1;

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel))
        pto2_init_complete_.store(true, std::memory_order_release);
    else
        while (!pto2_init_complete_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();

    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;

    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState *local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++)
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);

    const bool pmu_active = is_pmu_enabled();

    uint64_t last_progress_ts = get_sys_cnt_aicpu();

    // Dispatch-loop start timestamp for the SchedWindow phase marker (the
    // host reduces min(start)/max(end) across sched threads → the `Sched`
    // span). This one call ≈ one kernel launch.
    [[maybe_unused]] const uint64_t sched_loop_start_ts = get_sys_cnt_aicpu();

    while (true) {
        if (completed_.load(std::memory_order_acquire)) break;
        bool made_progress = false;
        if (!tracker.has_any_running_cores()) {
            LoopAction action = handle_orchestrator_exit(header, runtime);
            if (action == LoopAction::BREAK_LOOP) break;
        }

        // Phase 1: Check running cores for completion
        int32_t completed_this_turn = 0;

        if (tracker.has_any_running_cores()) {
            check_running_cores_for_completion(thread_idx, completed_this_turn, cur_thread_completed, made_progress);
        }
        if (completed_this_turn > 0) {
            completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
        }

        if (rt_ != nullptr && rt_->aicore_mailbox != nullptr &&
            (sched_->async_wait_list.count > 0 || rt_->aicore_mailbox->has_pending())) {
            AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(rt_->aicore_mailbox, sched_);
            if (poll_result.error_code != PTO2_ERROR_NONE) {
                int32_t expected = PTO2_ERROR_NONE;
                header->sched_error_code.compare_exchange_strong(
                    expected, poll_result.error_code, std::memory_order_acq_rel, std::memory_order_acquire
                );
                completed_.store(true, std::memory_order_release);
                break;
            }
            if (poll_result.completed > 0) {
                completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                made_progress = true;
            }
        }

        // Phase 2 drain check
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(thread_idx);
            continue;
        }

        // Phase 3: Drain wiring queue (thread 0 only).
        if (thread_idx == 0) {
            int wired = sched_->drain_wiring_queue(orchestrator_done_);
            if (wired > 0) made_progress = true;
        }

        if (thread_idx == 0) {
            constexpr int DUMMY_DRAIN_BATCH = 16;
            PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
            int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
            for (int di = 0; di < dummy_got; di++) {
                PTO2TaskSlotState &dummy_slot = *dummy_batch[di];
                sched_->on_mixed_task_complete(dummy_slot);
                completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                cur_thread_completed++;
            }
            if (dummy_got > 0) made_progress = true;
        }

        // Phase 4: MIX-strict-priority dispatch with phase-split and
        // cross-thread idle gating. See dispatch_ready_tasks for the policy.
        dispatch_ready_tasks(thread_idx, tracker, local_bufs, pmu_active, made_progress);

        if (made_progress) {
            idle_iterations = 0;
            last_progress_ts = get_sys_cnt_aicpu();
        } else {
            idle_iterations++;

            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                LoopAction action = check_idle_fatal_error(header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (idle_iterations % STALL_LOG_INTERVAL == 0) log_stall_diagnostics(thread_idx);
            if (get_sys_cnt_aicpu() - last_progress_ts > SCHEDULER_TIMEOUT_CYCLES) {
                bool self_owns = self_owns_running_task(thread_idx);
                bool global_stuck = !self_owns && total_tasks_ > 0 &&
                                    completed_tasks_.load(std::memory_order_relaxed) < total_tasks_ &&
                                    no_thread_owns_running_task();
                if (self_owns || global_stuck) return handle_timeout_exit(thread_idx, header, runtime);
                last_progress_ts = get_sys_cnt_aicpu();
            }
            SPIN_WAIT_HINT();
        }
    }

#if PTO2_PROFILING
    // Ride this scheduler thread's dispatch window home to the per-thread
    // phase buffer. The host reduces min(start)/max(end) across the sched
    // threads into the `Sched` marker, so Effective =
    //   max(orch_end, sched_end) - min(orch_start, sched_start)
    // ends when the LAST scheduler finishes all tasks — not when the
    // orchestrator finished submitting. Without this the sched span is
    // absent and Effective collapses to the orch-submit window. sched_end_ts
    // is the loop-exit time (completed_ observed = all tasks done).
    const uint64_t sched_end_ts = get_sys_cnt_aicpu();
    aicpu_phase_set_window(AicpuPhase::SchedWindow, sched_loop_start_ts, sched_end_ts);
#endif

    return cur_thread_completed;
}

const char *SchedulerContext::shape_name(PTO2ResourceShape shape) {
    switch (shape) {
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

int SchedulerContext::pop_ready_tasks_batch(
    PTO2ResourceShape shape, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
) {
    return sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
}

void SchedulerContext::build_payload(
    PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    const AsyncCtx &async_ctx, int32_t block_idx
) {
    int32_t slot_idx = static_cast<int32_t>(subslot);
    uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    dispatch_payload.function_bin_addr = callable->resolved_addr();
    auto &payload = *slot_state.payload;
    int n = 0;
    for (int32_t i = 0; i < payload.tensor_count; i++)
        dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
    for (int32_t i = 0; i < payload.scalar_count; i++)
        dispatch_payload.args[n++] = payload.scalars[i];
    dispatch_payload.local_context.block_idx = block_idx;
    dispatch_payload.local_context.block_num = slot_state.logical_block_num;
    dispatch_payload.local_context.async_ctx = async_ctx;
    dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
    dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
    // Speculative early-dispatch gate. Polling has no staging path, so every
    // dispatch is execute-on-pickup. Writing this per dispatch (as upstream
    // does) is what lets deinit skip the ~72 KB payload_per_core_ memset:
    // AICore reads src_payload each pickup and a stale non-zero would hang it
    // on the doorbell wait. src_payload==0 means "ready" (#1328 folded the old
    // not_ready flag into src_payload: 0=ready, non-zero=gated source pointer).
    dispatch_payload.src_payload = 0;
}

SchedulerContext::PublishHandle SchedulerContext::prepare_subtask_to_core(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot, bool to_pending,
    int32_t block_idx
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto core_id = tracker.get_core_id_by_offset(core_offset);
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
    DeferredCompletionSlab *deferred_slab = &deferred_slab_per_core_[core_id][buf_idx];
    deferred_slab->count = 0;
    deferred_slab->error_code = PTO2_ERROR_NONE;
    AsyncCtx async_ctx = AsyncCtx::make(slot_state.task->task_id, deferred_slab);
    build_payload(payload, slot_state, subslot, async_ctx, block_idx);

    if (to_pending) {
        core_exec_state.pending_subslot = subslot;
        core_exec_state.pending_slot_state = &slot_state;
        core_exec_state.pending_reg_task_id = static_cast<int32_t>(reg_task_id);
    } else {
        core_exec_state.running_subslot = subslot;
        core_exec_state.running_slot_state = &slot_state;
        core_exec_state.running_reg_task_id = static_cast<int32_t>(reg_task_id);
        tracker.change_core_state(core_offset);
    }
    tracker.set_pending_occupied(core_offset);

    uint64_t *dispatch_timestamp_slot = nullptr;

    return PublishHandle{
        core_exec_state.reg_addr, reg_task_id, core_offset, dispatch_timestamp_slot, slot_state.task->task_timing_slot
    };
}

int SchedulerContext::prepare_block_for_dispatch(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending,
    int32_t block_idx, PublishHandle *out_handles
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    if (shape == PTO2ResourceShape::MIX) {
        uint8_t cmask = slot_state.active_mask.core_mask();
        int n = 0;
        if (cmask & PTO2_SUBTASK_MASK_AIC) {
            bool p = to_pending && !tracker.is_aic_core_idle(core_offset);
            out_handles[n++] = prepare_subtask_to_core(
                thread_idx, tracker.get_aic_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIC, p, block_idx
            );
        }
        if (cmask & PTO2_SUBTASK_MASK_AIV0) {
            bool p = to_pending && !tracker.is_aiv0_core_idle(core_offset);
            out_handles[n++] = prepare_subtask_to_core(
                thread_idx, tracker.get_aiv0_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIV0, p, block_idx
            );
        }
        if (cmask & PTO2_SUBTASK_MASK_AIV1) {
            bool p = to_pending && !tracker.is_aiv1_core_idle(core_offset);
            out_handles[n++] = prepare_subtask_to_core(
                thread_idx, tracker.get_aiv1_core_offset(core_offset), slot_state, PTO2SubtaskSlot::AIV1, p, block_idx
            );
        }
        return n;
    } else if (shape == PTO2ResourceShape::AIC) {
        out_handles[0] =
            prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
        return 1;
    } else {
        out_handles[0] =
            prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
        return 1;
    }
}

void SchedulerContext::dispatch_shape(
    int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase, PTO2LocalReadyBuffer &local_buf,
    CoreTracker &tracker, bool &entered_drain, bool &made_progress
) {
    if (entered_drain) return;

    bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
    auto cores = tracker.get_dispatchable_cores(shape, phase);
    if (!cores.has_value()) return;

    while (cores.has_value() && !entered_drain) {
        int want = cores.count();
        PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
        int got = pop_ready_tasks_batch(shape, local_buf, batch, want);
        if (got == 0) break;

        bool any_sync_start = false;
        for (int bi = 0; bi < got; bi++) {
            if (batch[bi]->active_mask.requires_sync_start()) {
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
            for (int i = 0; i < handle_count; i++)
                publish_subtask_to_core(handles[i], dispatch_ts, thread_idx);
            handle_count = 0;
            made_progress = true;
        };

        for (int bi = 0; bi < got; bi++) {
            PTO2TaskSlotState *slot_state = batch[bi];

            if (slot_state->active_mask.requires_sync_start()) {
                if (is_pending) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
                int32_t available = cores.count();
                if (available < slot_state->logical_block_num) {
                    flush_publish();
                    if (!enter_drain_mode(slot_state, slot_state->logical_block_num))
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    for (int rem = bi + 1; rem < got; rem++)
                        sched_->ready_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                    entered_drain = true;
                    break;
                }
            }

            if (!cores.has_value()) {
                flush_publish();
                sched_->ready_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                break;
            }

            dispatched_any = true;
            int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
            int32_t claim = std::min(cores.count(), remaining);
            int32_t start = slot_state->next_block_idx;
            slot_state->next_block_idx += claim;

            if (slot_state->next_block_idx < slot_state->logical_block_num)
                sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);

            for (int32_t b = 0; b < claim; b++) {
                auto core_offset = cores.pop_first();
                handle_count += prepare_block_for_dispatch(
                    thread_idx, core_offset, *slot_state, shape, is_pending, start + b, &handles[handle_count]
                );
            }

            if (any_sync_start) flush_publish();
        }

        flush_publish();

        if (!dispatched_any) break;

        if (!cores.has_value()) cores = tracker.get_dispatchable_cores(shape, phase);
    }
}

void SchedulerContext::dispatch_ready_tasks(
    int32_t thread_idx, CoreTracker &tracker, PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES],
    bool pmu_active, bool &made_progress
) {
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
    for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
        auto &lb = local_bufs[s];
        int32_t excess = lb.count - thread_capacity[s];
        if (excess <= 0) continue;
        if (!has_idle_in_other_threads(thread_idx, static_cast<PTO2ResourceShape>(s))) continue;
        sched_->ready_queues[s].push_batch(&lb.slot_states[lb.count - excess], excess);
        lb.count -= excess;
    }

    auto flush_local_bufs = [&]() {
        for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
            auto &lb = local_bufs[s];
            if (lb.count > 0) {
                sched_->ready_queues[s].push_batch(lb.slot_states, lb.count);
                lb.count = 0;
            }
        }
    };
    struct FlushGuard {
        decltype(flush_local_bufs) &flush_fn;
        ~FlushGuard() { flush_fn(); }
    } flush_guard{flush_local_bufs};

    bool entered_drain = false;

    // ===== IDLE stage =====
    dispatch_shape(
        thread_idx, PTO2ResourceShape::MIX, Phase::IDLE, local_bufs[MIX_I], tracker, entered_drain, made_progress
    );
    if (entered_drain) return;

    bool skip_aic_aiv = has_residual_mix(local_bufs[MIX_I]);

    if (!skip_aic_aiv) {
        for (int i = 0; i < 2; i++) {
            PTO2ResourceShape s = aic_aiv[i];
            dispatch_shape(
                thread_idx, s, Phase::IDLE, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress
            );
            if (entered_drain) return;
        }
    }

    // Flush between IDLE and PENDING so PENDING-stage queue-size checks and any
    // peer-thread reads see the IDLE-stage release_fanin output.
    flush_local_bufs();

    if (pmu_active) return;

    if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX)) {
        dispatch_shape(
            thread_idx, PTO2ResourceShape::MIX, Phase::PENDING, local_bufs[MIX_I], tracker, entered_drain, made_progress
        );
        if (entered_drain) return;
    }

    // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
    // it set; otherwise, escalate iff PENDING-MIX left residual.
    if (!skip_aic_aiv && has_residual_mix(local_bufs[MIX_I])) skip_aic_aiv = true;

    if (skip_aic_aiv) return;

    // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
    // will pull from the global queue on its next IDLE pass.
    for (int i = 0; i < 2; i++) {
        PTO2ResourceShape s = aic_aiv[i];
        if (has_idle_in_other_threads(thread_idx, s)) continue;
        dispatch_shape(
            thread_idx, s, Phase::PENDING, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain, made_progress
        );
        if (entered_drain) return;
    }
}

bool SchedulerContext::has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const {
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        if (t == self_thread_idx) continue;
        if (core_trackers_[t].get_idle_core_offset_states(shape).has_value()) return true;
    }
    return false;
}
