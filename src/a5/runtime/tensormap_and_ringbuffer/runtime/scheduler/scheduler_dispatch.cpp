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

#include "common.h"  // debug_assert
#include "common/unified_log.h"
#include "aicpu/device_time.h"
#include "aicpu/platform_regs.h"
#include "callable.h"
#include "common/l2_swimlane_profiling.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "spin_hint.h"

// Performance profiling headers
#include "aicpu/l2_swimlane_collector_aicpu.h"
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
    case PTO2ResourceShape::DUMMY:
        return "DUMMY";
    }
    return "UNKNOWN";
}

bool SchedulerContext::has_idle_in_other_threads(int32_t self_thread_idx, PTO2ResourceShape shape) const {
    // Cross-thread read of peer trackers without explicit synchronization. The
    // backing `core_states_` is a naturally aligned uint64_t; aarch64 guarantees
    // single-copy atomicity for an 8-byte aligned load, so no torn read. The
    // value is consumed only as a scheduling *hint* — a stale read at worst
    // causes one missed/extra pending dispatch, corrected on the next iteration.
    // Drain-mode cross-thread writes are serialized by handle_drain_mode's ack
    // barrier (all peers spin out of the dispatch path before any tracker
    // mutation), so this routine is never racing the drain worker.
    for (int32_t t = 0; t < active_sched_threads_; t++) {
        if (t == self_thread_idx) continue;
        if (core_trackers_[t].get_idle_core_offset_states(shape).has_value()) {
            return true;
        }
    }
    return false;
}

int SchedulerContext::pop_ready_tasks_batch(
    PTO2ResourceShape shape, int32_t thread_idx, PTO2LocalReadyBuffer &local_buf, PTO2TaskSlotState **out, int max_count
) {
#if PTO2_PROFILING
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#if PTO2_SCHED_PROFILING
    extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
    uint64_t t_pop_start = get_sys_cnt_aicpu();
    int count = sched_->get_ready_tasks_batch(
        shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx]
    );
    l2_swimlane.sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
#else
    int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        if (count > 0) {
            l2_swimlane.pop_hit += count;
        } else {
            l2_swimlane.pop_miss++;
        }
    }
#else
    (void)thread_idx;
    int count = sched_->get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
    return count;
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
    for (int32_t i = 0; i < payload.tensor_count; i++) {
        dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
    }
    for (int32_t i = 0; i < payload.scalar_count; i++) {
        dispatch_payload.args[n++] = payload.scalars[i];
    }
    dispatch_payload.local_context.s_block_idx = block_idx;
    dispatch_payload.local_context.s_block_num = slot_state.logical_block_num;
    dispatch_payload.local_context.async_ctx = async_ctx;
    dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
    dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
}

void SchedulerContext::dispatch_subtask_to_core(
    Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    bool to_pending, int32_t block_idx
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    auto core_id = tracker.get_core_id_by_offset(core_offset);
    (void)runtime;
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

    LOG_DEBUG(
        "Thread %d: Dispatched %s %s task %" PRId64 " kernel_id=[%d,%d,%d] block_idx=%d/total_blocks=%d to"
        " core_offset=%d core_id=%d reg_task_id=%u",
        thread_idx, to_pending ? "pending" : "idle", subslot_name(subslot),
        static_cast<int64_t>(slot_state.task->task_id.raw), slot_state.task->kernel_id[0],
        slot_state.task->kernel_id[1], slot_state.task->kernel_id[2], block_idx, slot_state.logical_block_num,
        core_offset, core_id, reg_task_id
    );

    // AICore buffer rotation lives on the dispatch path: count this dispatch
    // and rotate before write_reg when we're about to cross a BUFFER_SIZE
    // boundary. The completion-before-dispatch invariant makes this race-free.
#if PTO2_PROFILING
    if (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED) {
        l2_swimlane_aicpu_on_aicore_dispatch(core_id, thread_idx);
    }
#endif

    // Publish task data (slot_state / args writes done above) before AICore
    // can observe the dispatched task_id. ARM64 needs an explicit store-store
    // fence across Normal-cacheable -> Device-nGnRnE; the old write_reg()
    // helper provided this implicitly via __sync_synchronize.
    wmb();

    // Capture dispatch timestamp at the latest possible moment — after wmb,
    // immediately before the DATA_MAIN_BASE write.
#if PTO2_PROFILING
    if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
        uint64_t dispatch_ts = get_sys_cnt_aicpu();
        if (to_pending) {
            core_exec_state.pending_dispatch_timestamp = dispatch_ts;
        } else {
            core_exec_state.running_dispatch_timestamp = dispatch_ts;
        }
    }
#endif

    write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));
    tracker.set_pending_occupied(core_offset);
}

void SchedulerContext::dispatch_mix_block_to_cluster(
    Runtime *runtime, int32_t thread_idx, int32_t cluster_offset, PTO2TaskSlotState &slot_state, bool to_pending,
    int32_t block_idx
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    uint8_t cmask = slot_state.active_mask.core_mask();
    if (cmask & PTO2_SUBTASK_MASK_AIC) {
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aic_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIC,
            to_pending, block_idx
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV0) {
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aiv0_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV0,
            to_pending, block_idx
        );
    }
    if (cmask & PTO2_SUBTASK_MASK_AIV1) {
        dispatch_subtask_to_core(
            runtime, thread_idx, tracker.get_aiv1_core_offset(cluster_offset), slot_state, PTO2SubtaskSlot::AIV1,
            to_pending, block_idx
        );
    }
}

void SchedulerContext::dispatch_block(
    Runtime *runtime, int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape,
    bool to_pending, int32_t block_idx
) {
#if PTO2_PROFILING
    if (is_dump_args_enabled()) {
        dump_args_for_task<PTO2_SUBTASK_SLOT_COUNT>(
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
        dispatch_mix_block_to_cluster(runtime, thread_idx, core_offset, slot_state, to_pending, block_idx);
    } else if (shape == PTO2ResourceShape::AIC) {
        dispatch_subtask_to_core(
            runtime, thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx
        );
    } else {
        dispatch_subtask_to_core(
            runtime, thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx
        );
    }
#if PTO2_PROFILING
    sched_l2_swimlane_[thread_idx].phase_dispatch_count += __builtin_popcount(slot_state.active_mask.core_mask());
#endif
}

void SchedulerContext::dispatch_shape(
    Runtime *runtime, int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
    PTO2LocalReadyBuffer &local_buf, CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
) {
#if PTO2_SCHED_PROFILING
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#endif
    if (entered_drain) return;

    bool is_pending = (phase == CoreTracker::DispatchPhase::PENDING);
    bool is_mix = (shape == PTO2ResourceShape::MIX);
    auto cores = is_mix ? tracker.get_cluster_offset_states() : tracker.get_dispatchable_cores(shape, phase);
    if (!cores.has_value()) return;

    while (cores.has_value() && !entered_drain) {
        int want = cores.count();
        PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
        int got = pop_ready_tasks_batch(shape, thread_idx, local_buf, batch, want);
        if (got == 0) break;

        bool dispatched_any = false;
        for (int bi = 0; bi < got; bi++) {
            PTO2TaskSlotState *slot_state = batch[bi];
            CoreTracker::BitStates selected_mix_clusters(0ULL);

            if (is_mix) {
                auto candidates = cores;
                uint8_t cmask = slot_state->active_mask.core_mask();
                auto wanted = is_pending ? CoreTracker::MixPlacement::PENDING : CoreTracker::MixPlacement::RUNNING;
                while (candidates.has_value()) {
                    int32_t cluster_offset = candidates.pop_first();
                    if (tracker.classify_mix_cluster(cluster_offset, cmask) == wanted) {
                        selected_mix_clusters |= CoreTracker::BitStates(1ULL << cluster_offset);
                    }
                }
                if (!selected_mix_clusters.has_value()) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
            }

            if (slot_state->active_mask.requires_sync_start()) {
                if (is_pending) {
                    sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
                int32_t available = is_mix ? selected_mix_clusters.count() : cores.count();
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
            // Claim a contiguous range of blocks, hand the slot back to the
            // ready queue immediately, then perform the expensive dispatches.
            // This lets other schedulers concurrently claim and dispatch the
            // remaining blocks of the same SPMD task instead of spinning while
            // this thread fills all its own cores.  Only local `start + b` is
            // read after the push -- `next_block_idx` may already be advanced
            // by another scheduler that popped the slot.
            int32_t remaining = slot_state->logical_block_num - slot_state->next_block_idx;
            int32_t available = is_mix ? selected_mix_clusters.count() : cores.count();
            int32_t claim = std::min(available, remaining);
            int32_t start = slot_state->next_block_idx;
            slot_state->next_block_idx += claim;

            if (slot_state->next_block_idx < slot_state->logical_block_num) {
                sched_->ready_queues[static_cast<int32_t>(shape)].push(slot_state);
            }

            for (int32_t b = 0; b < claim; b++) {
                auto core_offset = is_mix ? selected_mix_clusters.pop_first() : cores.pop_first();
                if (is_mix) {
                    cores.clear_bit(core_offset);
                }
                dispatch_block(runtime, thread_idx, core_offset, *slot_state, shape, is_pending, start + b);
            }
            made_progress = true;
#if PTO2_SCHED_PROFILING
            l2_swimlane.sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
        }

        if (!dispatched_any) break;

        if (!cores.has_value()) {
            cores = is_mix ? tracker.get_cluster_offset_states() : tracker.get_dispatchable_cores(shape, phase);
        }
    }
}

void SchedulerContext::dispatch_ready_tasks(
    Runtime *runtime, int32_t thread_idx, CoreTracker &tracker,
    PTO2LocalReadyBuffer (&local_bufs)[PTO2_NUM_RESOURCE_SHAPES], bool pmu_active, bool &made_progress, bool &try_pushed
) {
    using Phase = CoreTracker::DispatchPhase;
    constexpr int32_t MIX_I = static_cast<int32_t>(PTO2ResourceShape::MIX);

    // MIX is handled explicitly at the top of each stage; only AIC/AIV cycle
    // through this 2-elem array, with order toggled by thread parity for
    // shape-level load balancing across threads.
    static constexpr PTO2ResourceShape kAicAivOrder[2][2] = {
        {PTO2ResourceShape::AIC, PTO2ResourceShape::AIV},
        {PTO2ResourceShape::AIV, PTO2ResourceShape::AIC},
    };
    const PTO2ResourceShape *aic_aiv = kAicAivOrder[thread_idx & 1];

    auto flush_local_bufs = [&]() {
        for (int32_t s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
            auto &lb = local_bufs[s];
            if (lb.count > 0) {
                sched_->ready_queues[s].push_batch(lb.slot_states, lb.count);
                lb.count = 0;
            }
        }
    };
    // Every return path below must flush; wrap in RAII so we cannot forget.
    // The mid-function flush between IDLE and PENDING is still called
    // explicitly — guard only covers exit.
    struct FlushGuard {
        decltype(flush_local_bufs) &flush_fn;
        ~FlushGuard() { flush_fn(); }
    } flush_guard{flush_local_bufs};

    bool entered_drain = false;

    // ===== IDLE stage =====
    dispatch_shape(
        runtime, thread_idx, PTO2ResourceShape::MIX, Phase::IDLE, local_bufs[MIX_I], tracker, entered_drain,
        made_progress, try_pushed
    );
    if (entered_drain) return;

    // MIX-IDLE residual: AIC/AIV (both IDLE and PENDING) yield for this pass.
    // MIX-PENDING below still runs — that is the core of "mix strict priority":
    // pending slots are spent on mix before AIC/AIV get any chance.
    bool skip_aic_aiv = has_residual_mix(local_bufs[MIX_I]);

    if (!skip_aic_aiv) {
        for (int i = 0; i < 2; i++) {
            PTO2ResourceShape s = aic_aiv[i];
            dispatch_shape(
                runtime, thread_idx, s, Phase::IDLE, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain,
                made_progress, try_pushed
            );
            if (entered_drain) return;
        }
    }

    // Flush between IDLE and PENDING so PENDING-stage queue-size checks and any
    // peer-thread reads see the IDLE-stage release_fanin output.
    flush_local_bufs();

    if (pmu_active) return;

    // ===== PENDING stage =====
    // MIX-PENDING gate: skip when a peer has an idle MIX-capable cluster — that
    // peer's next IDLE-MIX iteration will pull the mix task from the global
    // queue (already flushed above) at lower latency than us pre-loading a
    // pending slot here. Forward progress for MIX is preserved: at least one
    // thread will run MIX-IDLE next pass and consume the residual.
    //
    // The gate is NOT subject to skip_aic_aiv — residual mix continues to drain
    // via pending slots on this thread when no peer is idle.
    if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX)) {
        dispatch_shape(
            runtime, thread_idx, PTO2ResourceShape::MIX, Phase::PENDING, local_bufs[MIX_I], tracker, entered_drain,
            made_progress, try_pushed
        );
        if (entered_drain) return;
    }

    // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
    // it set; otherwise, escalate iff PENDING-MIX left residual.
    if (!skip_aic_aiv && has_residual_mix(local_bufs[MIX_I])) {
        skip_aic_aiv = true;
    }

    // PENDING-MIX may have re-populated AIC/AIV local_bufs via release_fanin
    // during in-flight completions; flush_guard ensures these don't carry
    // across to the next iteration's IDLE stage.
    if (skip_aic_aiv) return;

    // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
    // will pull from the global queue on its next IDLE pass.
    for (int i = 0; i < 2; i++) {
        PTO2ResourceShape s = aic_aiv[i];
        if (has_idle_in_other_threads(thread_idx, s)) continue;
        dispatch_shape(
            runtime, thread_idx, s, Phase::PENDING, local_bufs[static_cast<int32_t>(s)], tracker, entered_drain,
            made_progress, try_pushed
        );
        if (entered_drain) return;
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

    LOG_INFO_V0("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, tracker.core_num());
    int32_t cur_thread_completed = 0;
    // Non-zero once a scheduler-hang timeout latches; returned in place of the
    // completed count so the caller still sees the negative error rc while the
    // shared end-of-loop flush below runs.
    int32_t timeout_rc = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if PTO2_PROFILING
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
    l2_swimlane.reset();
    l2_swimlane.l2_swimlane_enabled = (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED);
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

    // PMU runs require single-issue dispatch — overlapping in-flight tasks
    // pollute per-task PMU counters. Cached at function scope (parity with
    // a2a3): is_pmu_enabled() is extern "C" and the compiler cannot hoist it
    // across the dispatch loop on its own, and the value is loop-invariant
    // (PMU is latched once at kernel entry).
#if PTO2_PROFILING
    const bool pmu_active = is_pmu_enabled();
#else
    // PMU is definitionally off when profiling is compiled out; hard-set false
    // so dispatch keeps its overlapping (non-single-issue) fast path.
    constexpr bool pmu_active = false;
#endif

#if PTO2_PROFILING
    l2_swimlane.sched_start_ts = get_sys_cnt_aicpu();
#endif

#if PTO2_PROFILING
    // Queue-depth snapshot carried across the iteration boundary: each phase
    // emit consumes (phase_start_*) and refreshes them with its own end snapshot
    // so the next phase's "at_start" equals the previous phase's "at_end".
    //
    // L2SWIMLANE_NUM_QUEUE_SHAPES (3) matches PTO2_NUM_RESOURCE_SHAPES: AIC/AIV/MIX.
    //
    // **Hot-path cost discipline.** Local depth (this thread's PTO2LocalReadyBuffer)
    // is a single int read on a register-cached stack — free. Shared depth
    // (PTO2ReadyQueue::size) is two atomic relaxed loads against cache lines
    // that all peer sched threads also write to (enqueue_pos and dequeue_pos
    // bounce on every flush_local_bufs + every pop). With both phases emitting
    // per iter that's 12 cross-core loads × thousands of iters per run, a
    // measurable AICPU slowdown. Mitigation: lazy + per-iter cached shared
    // snapshot, refreshed at most once per iteration. The complete-emit and
    // dispatch-emit in the same iter both reuse the same shared sample; the
    // big transitions (local→shared flush) still show up across iter boundaries.
    static_assert(
        L2SWIMLANE_NUM_QUEUE_SHAPES == PTO2_NUM_RESOURCE_SHAPES,
        "queue snapshot width must match runtime resource shape count"
    );
    int16_t phase_start_local[L2SWIMLANE_NUM_QUEUE_SHAPES] = {0};
    int16_t phase_start_shared[L2SWIMLANE_NUM_QUEUE_SHAPES] = {0};
    int16_t iter_shared_snapshot[L2SWIMLANE_NUM_QUEUE_SHAPES] = {0};
    bool iter_shared_sampled = false;
    auto capture_local_snapshot = [&](int16_t local_out[L2SWIMLANE_NUM_QUEUE_SHAPES]) {
        for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
            local_out[s] = static_cast<int16_t>(local_bufs[s].count);
        }
    };
    auto get_or_sample_shared = [&]() -> const int16_t * {
        if (!iter_shared_sampled) {
            // Clamp to int16_t max before narrowing. PTO2_PROF_READYQUEUE_SIZE
            // is in the low thousands today but could grow with platform
            // scaling — without clamp, sizes above 32767 wrap to negatives
            // and silently corrupt the snapshot.
            constexpr size_t kMax = static_cast<size_t>(std::numeric_limits<int16_t>::max());
            for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                const size_t qsize = sched_->ready_queues[s].size();
                iter_shared_snapshot[s] = static_cast<int16_t>(std::min(qsize, kMax));
            }
            iter_shared_sampled = true;
        }
        return iter_shared_snapshot;
    };
    auto capture_phase_end = [&](int16_t local_out[L2SWIMLANE_NUM_QUEUE_SHAPES],
                                 int16_t shared_out[L2SWIMLANE_NUM_QUEUE_SHAPES]) {
        capture_local_snapshot(local_out);
        const int16_t *shared_cached = get_or_sample_shared();
        for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++)
            shared_out[s] = shared_cached[s];
    };
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        capture_phase_end(phase_start_local, phase_start_shared);
    }
#endif

    // Wall-clock timestamp of the last completed task on this thread.
    // Updated on made_progress; consulted to decide whether the wall-clock
    // budget for declaring a scheduler hang has elapsed. Initialized to
    // "now" so the first budget cycle starts when this thread does, not at
    // an undefined value.
    uint64_t last_progress_ts = get_sys_cnt_aicpu();

    while (true) {
        if (completed_.load(std::memory_order_acquire)) {
            break;
        }
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        l2_swimlane.sched_loop_count++;
        uint64_t _t0_phase = _t0;
        // Per-iter lazy shared-queue snapshot: first phase emit in this iter
        // pays the atomic-load cost, subsequent emits in the same iter reuse
        // the cached value. Reset here so we re-sample exactly once per iter
        // (or skip entirely on iters with no phase emit).
        iter_shared_sampled = false;
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
        CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
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

        if (rt_ != nullptr && rt_->aicore_mailbox != nullptr &&
            (sched_->async_wait_list.count > 0 || rt_->aicore_mailbox->has_pending())) {
            AsyncPollResult poll_result = sched_->async_wait_list.poll_and_complete<false>(
                rt_->aicore_mailbox, sched_, local_bufs, deferred_release_slot_states, deferred_release_count,
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
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_swimlane.sched_complete_cycle);
            if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && l2_swimlane.phase_complete_count > 0) {
                // Local depth is cheap (this thread's own buffer counter).
                // Shared depth is NOT sampled here: complete's release_fanin
                // pushes to local_bufs in the fast path (try_push succeeds
                // until cap=64). Shared only changes on dispatch's flush
                // path. Carrying phase_start_shared forward as end_shared
                // is the right answer 99% of the time AND skips three
                // contended atomic loads per emit.
                int16_t phase_end_local[L2SWIMLANE_NUM_QUEUE_SHAPES];
                capture_local_snapshot(phase_end_local);
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Complete, _t0_phase, _t1, l2_swimlane.sched_loop_count,
                    l2_swimlane.phase_complete_count, /*pop_hit=*/0, /*pop_miss=*/0, phase_start_local,
                    phase_start_shared, phase_end_local, phase_start_shared
                );
                for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                    phase_start_local[s] = phase_end_local[s];
                    // phase_start_shared unchanged — carried forward
                }
                _t0_phase = _t1;
                l2_swimlane.phase_complete_count = 0;
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
        int wired = 0;
        if (thread_idx == 0) {
            wired = sched_->drain_wiring_queue(orchestrator_done_);
            if (wired > 0) {
                made_progress = true;
#if PTO2_SCHED_PROFILING
                l2_swimlane.phase_wiring_count += wired;
#endif
            }
        }
#if PTO2_PROFILING
        CYCLE_COUNT_LAP(l2_swimlane.sched_wiring_cycle);
        // Wire outer phase: emit one bar covering this iter's drain_wiring_queue
        // pass when it wired any tasks. tasks_processed = wired count.
        if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && wired > 0) {
            int16_t phase_end_local[L2SWIMLANE_NUM_QUEUE_SHAPES];
            capture_local_snapshot(phase_end_local);
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::Wire, _t0_phase, _t1, l2_swimlane.sched_loop_count,
                static_cast<uint32_t>(wired), /*pop_hit=*/0, /*pop_miss=*/0, phase_start_local, phase_start_shared,
                phase_end_local, phase_start_shared
            );
            for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                phase_start_local[s] = phase_end_local[s];
            }
            _t0_phase = _t1;
        }
#endif

        // Phase 3b: Drain dummy ready queue (thread 0 only).
        //
        // Dependency-only tasks bypass AICore dispatch: they go through the
        // scheduler so fanin/fanout edges stay consistent, but completion is
        // signalled inline here. Pinned to thread 0 to avoid cross-thread
        // races and to keep cache hot near the wiring drain above.
        if (thread_idx == 0) {
            constexpr int DUMMY_DRAIN_BATCH = 16;
            PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
            int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
#if PTO2_PROFILING
            // Dummy outer phase: covers handling of all dummies popped this
            // iter. tasks_processed = dummy_got.
            uint64_t dummy_outer_t0 =
                (dummy_got > 0 && l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
            for (int di = 0; di < dummy_got; di++) {
                PTO2TaskSlotState &dummy_slot = *dummy_batch[di];
#if PTO2_SCHED_PROFILING
                sched_->on_task_complete(dummy_slot, thread_idx, local_bufs);
#else
                sched_->on_task_complete(dummy_slot, local_bufs);
#endif
                // Dummy tasks have no subtasks to retire and no fanout pre-conditions
                // beyond their own producers; release self-reference so the slot can
                // reach CONSUMED once all consumers drain.
                deferred_release_slot_states[deferred_release_count++] = &dummy_slot;
                if (deferred_release_count >= PTO2_DEFERRED_RELEASE_CAP) {
                    while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                        (void)sched_->on_task_release(
                            *deferred_release_slot_states[--deferred_release_count], thread_idx
                        );
#else
                        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                    }
                }
                int32_t prev = completed_tasks_.fetch_add(1, std::memory_order_relaxed);
                last_progress_count = prev + 1;
                cur_thread_completed++;
            }
            if (dummy_got > 0) {
                made_progress = true;
            }
#if PTO2_PROFILING
            if (dummy_outer_t0 != 0) {
                int16_t phase_end_local[L2SWIMLANE_NUM_QUEUE_SHAPES];
                capture_local_snapshot(phase_end_local);
                uint64_t dummy_outer_t1 = get_sys_cnt_aicpu();
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Dummy, dummy_outer_t0, dummy_outer_t1,
                    l2_swimlane.sched_loop_count, static_cast<uint32_t>(dummy_got), /*pop_hit=*/0,
                    /*pop_miss=*/0, phase_start_local, phase_start_shared, phase_end_local, phase_start_shared
                );
                for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                    phase_start_local[s] = phase_end_local[s];
                }
                _t0_phase = dummy_outer_t1;
            }
#endif
        }

        // Phase 4: MIX-strict-priority dispatch with phase-split and
        // cross-thread idle gating. See dispatch_ready_tasks for the policy.
        // pmu_active is cached at function scope above (loop-invariant).
        dispatch_ready_tasks(runtime, thread_idx, tracker, local_bufs, pmu_active, made_progress, try_pushed);

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_swimlane.sched_dispatch_cycle);
            if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && l2_swimlane.phase_dispatch_count > 0) {
                // Final-drain at loop end emits the trailing-idle tail so
                // sum-of-deltas == run-cumulative.
                uint64_t pop_hit_delta = l2_swimlane.pop_hit - l2_swimlane.pop_hit_at_last_emit;
                uint64_t pop_miss_delta = l2_swimlane.pop_miss - l2_swimlane.pop_miss_at_last_emit;
                // L2SwimlaneAicpuPhaseRecord's extras are uint32 — a delta that overflows means
                // an emit was missed for ~4 billion pops, which is well outside any
                // realistic dispatch cadence and silently truncates without this guard.
                debug_assert(pop_hit_delta < (1ULL << 32));
                debug_assert(pop_miss_delta < (1ULL << 32));
                int16_t phase_end_local[L2SWIMLANE_NUM_QUEUE_SHAPES];
                int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
                capture_phase_end(phase_end_local, phase_end_shared);
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Dispatch, _t0_phase, _t1, l2_swimlane.sched_loop_count,
                    l2_swimlane.phase_dispatch_count, static_cast<uint32_t>(pop_hit_delta),
                    static_cast<uint32_t>(pop_miss_delta), phase_start_local, phase_start_shared, phase_end_local,
                    phase_end_shared
                );
                for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                    phase_start_local[s] = phase_end_local[s];
                    phase_start_shared[s] = phase_end_shared[s];
                }
                _t0_phase = _t1;
                l2_swimlane.phase_dispatch_count = 0;
                l2_swimlane.pop_hit_at_last_emit = l2_swimlane.pop_hit;
                l2_swimlane.pop_miss_at_last_emit = l2_swimlane.pop_miss;
            }
        }
#endif

#if !PTO2_PROFILING
        (void)try_completed;
        (void)try_pushed;
#endif

        if (made_progress) {
            idle_iterations = 0;
            last_progress_ts = get_sys_cnt_aicpu();
        } else {
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
            idle_iterations++;

            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                LoopAction action = check_idle_fatal_error(thread_idx, header, runtime);
                if (action == LoopAction::BREAK_LOOP) break;
            }

            if (idle_iterations % STALL_LOG_INTERVAL == 0) {
                log_stall_diagnostics(thread_idx, total_tasks_, idle_iterations, last_progress_count);
            }
            // Wall-clock budget gate, with two fatal-latch branches:
            //
            // 1. Self owns a RUNNING task — first-hand evidence the
            //    dispatch is stuck. Latch.
            // 2. No thread anywhere owns a RUNNING task AND tasks remain
            //    unfinished — the system is in a pre-dispatch / WAIT-only
            //    deadlock (e.g. dependency cycle). Ownerless idle threads
            //    are the only observers; let this one latch on the global
            //    evidence (`completed_tasks_ < total_tasks_` and
            //    `no_thread_owns_running_task()`).
            //
            // Otherwise: a sibling thread owns a RUNNING task but hasn't
            // hit its own budget yet (typical distributed startup-skew
            // case) — refresh last_progress_ts and keep spinning. The
            // STALL diagnostic above still fires periodically so
            // observability is preserved.
            if (get_sys_cnt_aicpu() - last_progress_ts > SCHEDULER_TIMEOUT_CYCLES) {
                bool self_owns = self_owns_running_task(thread_idx);
                bool global_stuck = !self_owns && total_tasks_ > 0 &&
                                    completed_tasks_.load(std::memory_order_relaxed) < total_tasks_ &&
                                    no_thread_owns_running_task();
                if (self_owns || global_stuck) {
                    // Latch the error + emergency_shutdown, then break to the
                    // shared end-of-loop cleanup so the diagnostic buffers get
                    // flushed to the host. An early return here would strand the
                    // stuck task's already-dumped inputs and every completed
                    // task's in/out records in the unflushed per-thread dump
                    // buffer — exactly the state we need to triage the hang.
                    timeout_rc = handle_timeout_exit(
                        thread_idx, header, runtime, idle_iterations, last_progress_count
#if PTO2_PROFILING
                        ,
                        l2_swimlane.sched_start_ts
#endif
                    );
                    break;
                }
                last_progress_ts = get_sys_cnt_aicpu();
            }
            SPIN_WAIT_HINT();
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
            // a2a3 design has Complete + Dispatch sched phases only; idle gaps
            // are reconstructed at post-process time from sched record spacing.
            (void)_t0_phase;
#endif
        }
    }

    // Drain any entries left in the deferred-release batch. The in-loop flush
    // only fires on idle iterations and on buffer-full; a loop exit while the
    // last iteration made progress can leave entries un-released. Drop them
    // here so every consumed producer slot completes its on_task_release
    // regardless of which loop-exit path fired.
    while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
        (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
    }

#if PTO2_PROFILING
    // Final-drain: emit any pop_hit / pop_miss accrued since the last
    // dispatch emit (typically the trailing idle loops while waiting for
    // orchestrator_done_) as a zero-duration synthetic dispatch record so
    // sum(record.pop_*) reconciles with the run-cumulative counter.
    // Gate on SCHED_PHASES — at lower levels the phase buffer is never
    // flushed (see below), so writing this record would be wasted work.
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        uint64_t final_pop_hit_delta = l2_swimlane.pop_hit - l2_swimlane.pop_hit_at_last_emit;
        uint64_t final_pop_miss_delta = l2_swimlane.pop_miss - l2_swimlane.pop_miss_at_last_emit;
        debug_assert(final_pop_hit_delta < (1ULL << 32));
        debug_assert(final_pop_miss_delta < (1ULL << 32));
        if (final_pop_hit_delta != 0 || final_pop_miss_delta != 0) {
            uint64_t t_now = get_sys_cnt_aicpu();
            int16_t phase_end_local[L2SWIMLANE_NUM_QUEUE_SHAPES];
            int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
            capture_phase_end(phase_end_local, phase_end_shared);
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::Dispatch, t_now, t_now, l2_swimlane.sched_loop_count, 0,
                static_cast<uint32_t>(final_pop_hit_delta), static_cast<uint32_t>(final_pop_miss_delta),
                phase_end_local, phase_end_shared, phase_end_local, phase_end_shared
            );
            l2_swimlane.pop_hit_at_last_emit = l2_swimlane.pop_hit;
            l2_swimlane.pop_miss_at_last_emit = l2_swimlane.pop_miss;
        }
    }
    log_l2_swimlane_summary(thread_idx, cur_thread_completed);
#endif

#if PTO2_PROFILING
    if (l2_swimlane.l2_swimlane_enabled) {
        l2_swimlane_aicpu_flush(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
        if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
            l2_swimlane_aicpu_flush_sched_phase_buffer(thread_idx);
        }
    }
#endif
#if PTO2_PROFILING
    if (is_dump_args_enabled()) {
        dump_args_flush(thread_idx);
    }
#endif
#if PTO2_PROFILING
    if (is_pmu_enabled()) {
        pmu_aicpu_flush_buffers(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
    }
#endif

    return timeout_rc != 0 ? timeout_rc : cur_thread_completed;
}
