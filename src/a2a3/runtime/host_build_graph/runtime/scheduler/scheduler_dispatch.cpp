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

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// =============================================================================
// Dispatch helpers
// =============================================================================

namespace {
inline constexpr int32_t PTO2_DEFERRED_RELEASE_CAP = 256;
}

// The early-dispatch core bitmask (PTO2_EARLY_DISPATCH_CORE_MASK_WORDS * 64 bits) must cover
// every global core_id, and the per-core doorbell table is sized to match.
static_assert(
    RUNTIME_MAX_WORKER <= PTO2_EARLY_DISPATCH_CORE_MASK_WORDS * 64,
    "staged_core_mask too small for RUNTIME_MAX_WORKER cores"
);

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
    PTO2ReadyQueue *queues, PTO2ResourceShape shape, int32_t thread_idx, PTO2TaskSlotState **out, int max_count
) {
#if SIMPLER_DFX
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
#if SIMPLER_SCHED_PROFILING
    extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
    uint64_t t_pop_start = get_sys_cnt_aicpu();
    int count = sched_->get_ready_tasks_batch(
        queues, shape, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx]
    );
    l2_swimlane.sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
#else
    int count = sched_->get_ready_tasks_batch(queues, shape, out, max_count);
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
    int count = sched_->get_ready_tasks_batch(queues, shape, out, max_count);
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
    dispatch_payload.local_context.block_idx = block_idx;
    dispatch_payload.local_context.block_num = slot_state.logical_block_num;
    dispatch_payload.local_context.async_ctx = async_ctx;
    dispatch_payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.local_context);
    dispatch_payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&dispatch_payload.global_context);
    // Early-dispatch: a task being staged (Hook 1 set early_dispatch_state to
    // STAGING before this call) is gated — the AICore must wait for the
    // DATA_MAIN_BASE high-32 doorbell. All other dispatches run on pickup.
    dispatch_payload.not_ready =
        (slot_state.payload->early_dispatch_state.load(std::memory_order_relaxed) == PTO2_EARLY_DISPATCH_STAGING) ? 1 :
                                                                                                                    0;
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
    // boundary. The completion-before-dispatch invariant makes this race-free
    // (all prior tasks on this core have FIN'd, so AICore has dcci'd their
    // records out of the old buffer). Gated on the same enable bit as flush
    // so level=1 (AICORE_TIMING-only) participates without needing complete_task.
#if SIMPLER_DFX
    if (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED) {
        l2_swimlane_aicpu_on_aicore_dispatch(core_id, thread_idx, reg_task_id);
    }
#endif

    uint64_t *dispatch_timestamp_slot = nullptr;
#if SIMPLER_DFX
    if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
        dispatch_timestamp_slot =
            to_pending ? &core_exec_state.pending_dispatch_timestamp : &core_exec_state.running_dispatch_timestamp;
    }
#endif

    return PublishHandle{core_exec_state.reg_addr, reg_task_id, core_offset, dispatch_timestamp_slot};
}

int SchedulerContext::prepare_block_for_dispatch(
    int32_t thread_idx, int32_t core_offset, PTO2TaskSlotState &slot_state, PTO2ResourceShape shape, bool to_pending,
    int32_t block_idx, PublishHandle *out_handles
) {
#if SIMPLER_DFX
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
#if SIMPLER_DFX
        sched_l2_swimlane_[thread_idx].phase_dispatch_count += __builtin_popcount(cmask);
#endif
        return n;
    } else if (shape == PTO2ResourceShape::AIC) {
        out_handles[0] =
            prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIC, to_pending, block_idx);
#if SIMPLER_DFX
        sched_l2_swimlane_[thread_idx].phase_dispatch_count += 1;
#endif
        return 1;
    } else {
        out_handles[0] =
            prepare_subtask_to_core(thread_idx, core_offset, slot_state, PTO2SubtaskSlot::AIV0, to_pending, block_idx);
#if SIMPLER_DFX
        sched_l2_swimlane_[thread_idx].phase_dispatch_count += 1;
#endif
        return 1;
    }
}

void SchedulerContext::dispatch_shape(
    int32_t thread_idx, PTO2ReadyQueue *disp_queues, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase,
    CoreTracker &tracker, bool &entered_drain, bool &made_progress, bool &try_pushed
) {
#if SIMPLER_SCHED_PROFILING
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
        int got = pop_ready_tasks_batch(disp_queues, shape, thread_idx, batch, want);
        if (got == 0) break;

        // sync_start exclusion gate.
        //
        // When the popped batch contains a sync_start task we MUST publish each
        // prior task with its own wmb so AICore receives them with time
        // separation. The drain coordinator's `count_global_available()` check
        // reads the per-thread CoreTracker, and although `prepare_block_for_dispatch`
        // marks cores occupied synchronously, the head-start between successive
        // tasks is what lets the surrounding completion loop catch up on FINs in
        // the retry window when the sync_start task hits insufficient resources.
        // Bursting all prior tasks at the end of the pop (cross-task batching)
        // collapses that head-start and causes spmd_sync_start_stress to time
        // out via 507018 on ~40% of runs — see
        // docs/investigations/2026-06-cross-task-batched-publish.md.
        //
        // When the batch carries no sync_start task, no drain entry can happen
        // in this pop, so we hoist `handles[]`, `wmb()`, and the publish loop
        // out of the per-task body. One wmb amortizes across all tasks and one
        // dispatch_ts is shared, which restores ~60 ns first-to-last AICore
        // start span for single-block decode kernels (out_proj, q_proj, ...).
        // Detection is a single mask check per task — cheap relative to even
        // one register write.
        bool any_sync_start = false;
        for (int bi = 0; bi < got; bi++) {
            if (batch[bi]->active_mask.requires_sync_start()) {
                any_sync_start = true;
                break;
            }
        }

        // handles[] is sized for the MIX worst case: total claims across the
        // pop bounded by `cores.count() ≤ MAX_CLUSTERS`, and each block
        // contributes ≤ 3 subtasks for MIX.
        PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
        int handle_count = 0;
        bool dispatched_any = false;
        // Slots dispatched this pop whose dispatch_fanin must be propagated to
        // consumers. Deferred until AFTER publish (below) so a flagged producer's
        // fanout walk never sits between claiming cores and publishing its own
        // blocks — doing it inline delays this thread's blocks while peer threads
        // co-dispatching the same SPMD task publish immediately, misaligning the
        // task's block starts. Bounded by cores.count() ≤ MAX_CLUSTERS dispatches.
        PTO2TaskSlotState *prop_list[CoreTracker::MAX_CLUSTERS];
        int prop_n = 0;
#if SIMPLER_SCHED_PROFILING
        uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif

        // Flush prepared-but-unpublished handles. Required before
        // `enter_drain_mode` so the drain coordinator sees cores as occupied,
        // and at the per-task boundary when `any_sync_start` is true.
        auto flush_publish = [&]() {
            if (handle_count == 0) return;
            wmb();
            uint64_t dispatch_ts = 0;
#if SIMPLER_DFX
            if (l2_swimlane_level_ >= L2SwimlaneLevel::AICPU_TIMING) {
                dispatch_ts = get_sys_cnt_aicpu();
            }
#endif
            for (int i = 0; i < handle_count; i++) {
                publish_subtask_to_core(handles[i], dispatch_ts);
            }
            handle_count = 0;
            made_progress = true;
        };

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
                    disp_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
            }

            // (Early-dispatch pre-staged tasks never reach this ready-pop: they are
            // released by their doorbell in release_fanin_and_check_ready the
            // instant their last producer completes — see try_early_dispatch_release.)

            if (slot_state->active_mask.requires_sync_start()) {
                if (is_pending) {
                    disp_queues[static_cast<int32_t>(shape)].push(slot_state);
                    continue;
                }
                int32_t available = is_mix ? selected_mix_clusters.count() : cores.count();
                if (available < slot_state->logical_block_num) {
                    flush_publish();
                    if (!enter_drain_mode(slot_state, slot_state->logical_block_num)) {
                        disp_queues[static_cast<int32_t>(shape)].push(slot_state);
                    }
                    for (int rem = bi + 1; rem < got; rem++) {
                        disp_queues[static_cast<int32_t>(shape)].push(batch[rem]);
                    }
                    entered_drain = true;
                    break;
                }
            }

            if (!cores.has_value()) {
                flush_publish();
                disp_queues[static_cast<int32_t>(shape)].push_batch(&batch[bi], got - bi);
                break;
            }

            dispatched_any = true;
            try_pushed = true;
            // Record for deferred dispatch_fanin propagation after this pop's
            // blocks are published (see after the loop). propagate's own guard
            // filters non-flagged slots, so recording unconditionally is cheap.
            if (prop_n < static_cast<int>(sizeof(prop_list) / sizeof(prop_list[0]))) {
                prop_list[prop_n++] = slot_state;
            }
            // Claim a contiguous range of blocks, hand the slot back to the
            // ready queue immediately, then perform the expensive dispatches.
            // This lets other schedulers concurrently claim and dispatch the
            // remaining blocks of the same SPMD task instead of spinning while
            // this thread fills all its own cores. Only local `start + b` is
            // read after the push — `next_block_idx` may already be advanced
            // by another scheduler that popped the slot.
            int32_t start = slot_state->next_block_idx.load(std::memory_order_relaxed);
            int32_t remaining = slot_state->logical_block_num - start;
            int32_t available = is_mix ? selected_mix_clusters.count() : cores.count();
            int32_t claim = std::min(available, remaining);
            slot_state->next_block_idx.store(static_cast<int16_t>(start + claim), std::memory_order_relaxed);

            if (start + claim < slot_state->logical_block_num) {
                disp_queues[static_cast<int32_t>(shape)].push(slot_state);
            }

            for (int32_t b = 0; b < claim; b++) {
                auto core_offset = is_mix ? selected_mix_clusters.pop_first() : cores.pop_first();
                if (is_mix) {
                    cores.clear_bit(core_offset);
                }
                handle_count += prepare_block_for_dispatch(
                    thread_idx, core_offset, *slot_state, shape, is_pending, start + b, &handles[handle_count]
                );
            }

            // Sync_start exclusion: flush per task so prior tasks have head-
            // start time before any sync_start drain check. Normal batches
            // fall through and accumulate for one cross-task flush at the
            // end of the pop.
            if (any_sync_start) {
                flush_publish();
            }
        }

        flush_publish();
        // Blocks are published; now propagate dispatch_fanin for any flagged
        // producers dispatched above (knob A: producer is running). Off the
        // pre-publish path so it cannot delay or misalign their blocks.
        for (int i = 0; i < prop_n; i++) {
            sched_->propagate_dispatch_fanin(*prop_list[i]);
        }
#if SIMPLER_SCHED_PROFILING
        l2_swimlane.sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif

        if (!dispatched_any) break;

        if (!cores.has_value()) {
            cores = is_mix ? tracker.get_cluster_offset_states() : tracker.get_dispatchable_cores(shape, phase);
        }
    }
}

template <typename StageFn, typename ResidualMixFn>
void SchedulerContext::run_staging_order(
    int32_t thread_idx, bool pmu_active, StageFn &&stage, ResidualMixFn &&residual_mix
) {
    using Phase = CoreTracker::DispatchPhase;

    // MIX is handled explicitly at the top of each stage; only AIC/AIV cycle
    // through this 2-elem array, with order toggled by thread parity for
    // shape-level load balancing across threads.
    static constexpr PTO2ResourceShape kAicAivOrder[2][2] = {
        {PTO2ResourceShape::AIC, PTO2ResourceShape::AIV},
        {PTO2ResourceShape::AIV, PTO2ResourceShape::AIC},
    };
    const PTO2ResourceShape *aic_aiv = kAicAivOrder[thread_idx & 1];

    // ===== IDLE stage =====
    if (stage(PTO2ResourceShape::MIX, Phase::IDLE)) return;

    // MIX-IDLE residual: AIC/AIV (both IDLE and PENDING) yield for this pass.
    // MIX-PENDING below still runs — that is the core of "mix strict priority":
    // pending slots are spent on mix before AIC/AIV get any chance.
    bool skip_aic_aiv = residual_mix();

    if (!skip_aic_aiv) {
        for (int i = 0; i < 2; i++) {
            if (stage(aic_aiv[i], Phase::IDLE)) return;
        }
    }

    if (pmu_active) return;

    // ===== PENDING stage =====
    // MIX-PENDING gate: skip when a peer has an idle MIX-capable cluster — that
    // peer's next IDLE-MIX iteration will pull the mix task from the global
    // queue at lower latency than us pre-loading a pending slot here. Forward
    // progress for MIX is preserved: at least one thread will run MIX-IDLE next
    // pass and consume the residual.
    //
    // The gate is NOT subject to skip_aic_aiv — residual mix continues to drain
    // via pending slots on this thread when no peer is idle.
    if (!has_idle_in_other_threads(thread_idx, PTO2ResourceShape::MIX)) {
        if (stage(PTO2ResourceShape::MIX, Phase::PENDING)) return;
    }

    // Re-check after MIX-PENDING. If MIX-IDLE already set skip_aic_aiv, leave
    // it set; otherwise, escalate iff PENDING-MIX left residual.
    if (!skip_aic_aiv && residual_mix()) {
        skip_aic_aiv = true;
    }

    if (skip_aic_aiv) return;

    // AIC/AIV-PENDING gate: a peer-idle skip is a delay, not a loss — the peer
    // will pull from the global queue on its next IDLE pass.
    for (int i = 0; i < 2; i++) {
        PTO2ResourceShape s = aic_aiv[i];
        if (has_idle_in_other_threads(thread_idx, s)) continue;
        if (stage(s, Phase::PENDING)) return;
    }
}

void SchedulerContext::dispatch_ready_tasks(
    int32_t thread_idx, CoreTracker &tracker, bool pmu_active, bool &made_progress, bool &try_pushed
) {
    // Normal ready dispatch (is_ready): dispatch_shape places each block on pickup and
    // signals a stop by setting entered_drain when it enters a sync_start drain.
    bool entered_drain = false;

    // Tier 0: ready sync_start cohorts take cores before any regular ready task
    // (sync_start > MIX > C/V within the normal source). Same order and machinery,
    // fed from ready_sync_queues; an oversized cohort arms the stop-the-world drain
    // (entered_drain), which also short-circuits the regular tier below.
    run_staging_order(
        thread_idx, pmu_active,
        [&](PTO2ResourceShape shape, CoreTracker::DispatchPhase phase) {
            dispatch_shape(
                thread_idx, sched_->ready_sync_queues, shape, phase, tracker, entered_drain, made_progress, try_pushed
            );
            return entered_drain;
        },
        [&] {
            return has_residual_sync_mix();
        }
    );
    if (entered_drain) return;

    // Tier 1: regular ready work.
    run_staging_order(
        thread_idx, pmu_active,
        [&](PTO2ResourceShape shape, CoreTracker::DispatchPhase phase) {
            dispatch_shape(
                thread_idx, sched_->ready_queues, shape, phase, tracker, entered_drain, made_progress, try_pushed
            );
            return entered_drain;
        },
        [&] {
            return has_residual_mix();
        }
    );
}

// Stage the ALREADY-CLAIMED range [start, start+count) of consumer `c` onto
// thread_idx's idle then pending cores. The caller (the queue drain) has advanced
// next_block_idx by `count` under pop-exclusivity AND re-pushed `c` for peers
// BEFORE calling this — so this, the expensive prepare+publish, runs CONCURRENTLY
// with peers staging other ranges of the same consumer. This mirrors the normal
// SPMD dispatch path (claim range -> store next_block_idx -> re-push -> dispatch).
// `idle`/`pend` are this thread's free-core sets, sized so idle.count+pend.count >=
// count (the caller clamped the claim to them), so all `count` blocks get a core.
//
// Rule 1: idle cores -> gated task in the RUNNING slot. Rule 2: PENDING slot of
// cores running a real task -> promoted in when that task FINs (gated-pending Case
// 3.3 in decide_slot_transition completes the running FIN + promotes instead of
// waiting for an ack the gated task never sends). Each staged core stays
// pending_occupied while gated, so no second gated block stacks on it.
//
// Self-ring: release flips STAGING->DISPATCHED then rings the mask. A block staged
// after that flip isn't in the mask release read, so this thread rings it here. The
// seq_cst order between "OR mask then load early_dispatch_state" (here) and "store DISPATCHED
// then read mask" (release) guarantees every gated core's doorbell fires.
int32_t SchedulerContext::stage_consumer_blocks(
    int32_t thread_idx, PTO2TaskSlotState *c, PTO2ResourceShape shape, int32_t start, int32_t count,
    CoreTracker::BitStates &idle, CoreTracker::BitStates &pend
) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    // Stamp the real pre-stage time (NOT 0) so the swimlane shows these blocks
    // dispatched during the producer's run, not at trace start.
    uint64_t early_dispatch_ts = get_sys_cnt_aicpu();
    uint64_t my_cores[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS] = {0};  // cores this thread gated (for self-ring)
    int32_t staged = 0;
    int32_t block = start;
    // Mirror the normal flush_publish (scheduler_dispatch.cpp wmb()+publish loop):
    // prepare ALL claimed blocks' payloads (idle bucket -> running slot, pend bucket
    // -> gated pending), then ONE wmb(), then publish. The wmb guarantees the
    // not_ready gate + args are globally visible before any DATA_MAIN_BASE token —
    // without it a gated core can pick up the token and dcci a stale payload. The
    // shared `count` budget bounds total blocks <= free clusters/cores, so both
    // buckets fit one handles[] buffer.
    PublishHandle handles[CoreTracker::MAX_CLUSTERS * 3];
    int n = 0;
    auto prepare_from = [&](CoreTracker::BitStates &avail, bool to_pending) {
        while (count > 0 && avail.has_value()) {
            int32_t core_offset = avail.pop_first();
            n += prepare_block_for_dispatch(thread_idx, core_offset, *c, shape, to_pending, block, &handles[n]);
            block++;
            count--;
            staged++;
        }
    };
    if (idle.has_value()) prepare_from(idle, /*to_pending=*/false);
    if (pend.has_value()) prepare_from(pend, /*to_pending=*/true);
    if (n > 0) {
        wmb();
        for (int i = 0; i < n; i++) {
            publish_subtask_to_core(handles[i], early_dispatch_ts);
            int32_t cid = tracker.get_core_id_by_offset(handles[i].core_offset);
            sched_->early_dispatch_doorbell_table[cid].addr = handles[i].reg_addr;
            sched_->early_dispatch_doorbell_table[cid].token = handles[i].reg_task_id;
            my_cores[cid >> 6] |= (1ULL << (cid & 63));
        }
    }
    // Publish all this thread's gated cores into the shared mask in one OR per word
    // (vs one per subtask) so release sees them; seq_cst keeps the self-ring order.
    for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++)
        if (my_cores[w] != 0) c->payload->staged_core_mask[w].fetch_or(my_cores[w], std::memory_order_seq_cst);

    // If release already flipped DISPATCHED, it may have read the mask before our
    // bits landed — ring our own cores so none is left gated forever.
    if (staged > 0 &&
        c->payload->early_dispatch_state.load(std::memory_order_seq_cst) == PTO2_EARLY_DISPATCH_DISPATCHED) {
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
            uint64_t bits = my_cores[w];
            while (bits != 0) {
                int cid = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                PTO2SchedulerState::ring_one_doorbell(
                    sched_->early_dispatch_doorbell_table[cid].addr, sched_->early_dispatch_doorbell_table[cid].token
                );
            }
        }
    }
    return staged;
}

// Early-dispatch analog of dispatch_shape: drain early_dispatch_queues[shape] and
// pre-stage claimed block ranges onto this thread's `shape` cores for `phase`. IDLE
// stages onto idle cores (RUNNING slot, gated); PENDING stages onto a running core's
// gated pending slot. Candidates are pushed to the shape's queue EVENT-DRIVEN by
// propagate_dispatch_fanin, so the shape is the queue index (no per-consumer
// to_shape()). Returns the number of blocks staged.
int32_t
SchedulerContext::early_dispatch_shape(int32_t thread_idx, PTO2ResourceShape shape, CoreTracker::DispatchPhase phase) {
    CoreTracker &tracker = core_trackers_[thread_idx];
    int32_t s = static_cast<int32_t>(shape);
    bool is_mix = (shape == PTO2ResourceShape::MIX);
    bool is_idle = (phase == CoreTracker::DispatchPhase::IDLE);

    // Size the pop exactly as dispatch_shape does: MIX to the cluster count, else the
    // phase's dispatchable-core count. Skip the queue entirely when no core is free
    // for this shape+phase (avoids a pointless pop + immediate push-back).
    CoreTracker::BitStates cores =
        is_mix ? tracker.get_cluster_offset_states() : tracker.get_dispatchable_cores(shape, phase);
    if (!cores.has_value()) return 0;

    int32_t total_staged = 0;
    PTO2TaskSlotState *batch[CoreTracker::MAX_CLUSTERS * 3];
    // Batch-pop in one queue op (fewer CAS than one pop per consumer); the pop is
    // bounded by the shape's capacity so the stack buffer always holds it. Then for
    // each consumer: CLAIM a range sized to THIS thread's free cores by advancing
    // next_block_idx with a CAS (atomic — next_block_idx is shared with normal
    // dispatch, which also claims it if release routes the consumer to the ready
    // queue, so a plain store could double-dispatch), RE-PUSH it for peers, THEN do
    // the expensive prepare+publish. Re-pushing before staging lets peers claim the
    // next range and stage CONCURRENTLY — a wide consumer (online_softmax, 48 blocks)
    // is filled by all idle threads in parallel. When cores run out mid-batch the
    // unprocessed remainder is pushed back for peers (mirrors normal's push_batch of
    // the unconsumed tail).
    int got = sched_->early_dispatch_queues[s].pop_batch(batch, cores.count());
    for (int bi = 0; bi < got; bi++) {
        PTO2TaskSlotState *c = batch[bi];
        if (c->payload->early_dispatch_state.load(std::memory_order_acquire) != PTO2_EARLY_DISPATCH_STAGING)
            continue;  // released

        // The single free-core bucket for this phase. For MIX, an active-mask-aware
        // whole-cluster scan keeps only the clusters whose placement matches the phase
        // (RUNNING placement for IDLE, PENDING placement for PENDING), matching normal
        // dispatch's classify_mix_cluster — unused cores in the cluster are ignored, so
        // a MIX whose unused AIV is busy is not stranded. For AIC/AIV it is just the
        // phase's dispatchable cores.
        CoreTracker::BitStates bucket;
        if (is_mix) {
            auto wanted = is_idle ? CoreTracker::MixPlacement::RUNNING : CoreTracker::MixPlacement::PENDING;
            uint8_t cmask = c->active_mask.core_mask();
            CoreTracker::BitStates candidates = tracker.get_cluster_offset_states();
            while (candidates.has_value()) {
                int32_t cluster_offset = candidates.pop_first();
                if (tracker.classify_mix_cluster(cluster_offset, cmask) == wanted) {
                    bucket |= CoreTracker::BitStates(1ULL << cluster_offset);
                }
            }
        } else {
            bucket = tracker.get_dispatchable_cores(shape, phase);
        }
        int32_t freecores = bucket.has_value() ? bucket.count() : 0;
        if (freecores == 0) {  // no cores for this shape+phase — give this + the unprocessed rest back
            sched_->early_dispatch_queues[s].push_batch(&batch[bi], got - bi);
            break;
        }
        // CAS-claim a contiguous range [start, start+claim) sized to this thread's
        // free cores; CAS keeps it atomic against peers AND normal dispatch.
        int32_t start = 0, claim = 0;
        while (true) {
            int16_t cur = c->next_block_idx.load(std::memory_order_relaxed);
            if (cur >= c->logical_block_num) break;  // fully claimed
            int32_t cnt = c->logical_block_num - cur;
            if (cnt > freecores) cnt = freecores;
            if (c->next_block_idx.compare_exchange_weak(
                    cur, static_cast<int16_t>(cur + cnt), std::memory_order_seq_cst, std::memory_order_relaxed
                )) {
                start = cur;
                claim = cnt;
                break;
            }
        }
        if (claim == 0) continue;  // nothing left to claim -> drop (no re-push)
        // Re-push for concurrent peers BEFORE the expensive staging.
        if (start + claim < c->logical_block_num) {
            if (!sched_->early_dispatch_queues[s].push(c))
                LOG_INFO_V9(
                    "[EARLY_DISPATCH] queue full on re-push, consumer=%" PRId64,
                    static_cast<int64_t>(c->task->task_id.raw)
                );
        }
        // stage_consumer_blocks fills the idle bucket (RUNNING slot) then the pend
        // bucket (gated pending); pass this phase's bucket in the matching slot and an
        // empty other so only the phase's cores are staged.
        CoreTracker::BitStates empty(0ULL);
        total_staged += is_idle ? stage_consumer_blocks(thread_idx, c, shape, start, claim, bucket, empty) :
                                  stage_consumer_blocks(thread_idx, c, shape, start, claim, empty, bucket);
    }
    return total_staged;
}

// Early-dispatch drain (idle pass), mirroring dispatch_ready_tasks: owns its own
// gating and progress-flag updates, and orders staging the same way — MIX strict
// priority, IDLE stage before PENDING stage, cross-thread idle gating
// (MIX-IDLE ▶ c/v-IDLE ▶ MIX-PEND ▶ c/v-PEND). sync_start doesn't apply here (those
// tasks are excluded from early dispatch at push time, propagate_dispatch_fanin).
// Returns the number of blocks staged this pass (for the EarlyDispatch swimlane bar).
int32_t SchedulerContext::try_early_dispatch(
    int32_t thread_idx, CoreTracker &tracker, bool pmu_active, bool &made_progress, bool &try_pushed
) {
    // Gate, owned here rather than by the caller (mirrors dispatch_ready_tasks
    // withholding PENDING under PMU internally):
    //   - pmu_active: staging gated work perturbs the single-issue PMU windows the
    //     same way dual-issue PENDING dispatch does, so early dispatch is off.
    //   - has_any_free_slot: this thread has no spare capacity to stage onto (a
    //     purely local read; a fully-occupied thread bails before touching shared
    //     queues).
    //   - ready queues empty: normal dispatch (both the ready sync_start lane and the
    //     regular ready_queues) strictly precedes early — there is no real ready task
    //     to delay only when every normal queue is drained.
    if (pmu_active || !tracker.has_any_free_slot()) return 0;
    for (int s = 0; s < PTO2_NUM_RESOURCE_SHAPES; s++) {
        if (sched_->ready_sync_queues[s].size() > 0 || sched_->ready_queues[s].size() > 0) return 0;
    }

    // Early staging (NOT is_ready): same MIX/idle/pending order as normal dispatch, via the
    // shared skeleton. early_dispatch_shape stages a gated block range and never enters drain,
    // so the stage callback always reports "no stop".
    int32_t total_staged = 0;
    run_staging_order(
        thread_idx, pmu_active,
        [&](PTO2ResourceShape shape, CoreTracker::DispatchPhase phase) {
            total_staged += early_dispatch_shape(thread_idx, shape, phase);
            return false;
        },
        [&] {
            return has_residual_early_mix();
        }
    );

    // Staging is dispatch work: reset the idle/stall clock and route this iter's tail
    // cycles to the dispatch accumulator, exactly as normal dispatch does.
    if (total_staged > 0) {
        made_progress = true;
        try_pushed = true;
    }
    return total_staged;
}

// =============================================================================
// Main scheduler dispatch loop
// =============================================================================

int32_t SchedulerContext::resolve_and_dispatch(Runtime *runtime, int32_t thread_idx) {
    always_assert(sched_ != nullptr);
    CoreTracker &tracker = core_trackers_[thread_idx];
    LOG_INFO_V0("Thread %d: resolve_and_dispatch entry", thread_idx);

    PTO2SharedMemoryHeader *header = sched_->sm_header;
    if (!header) {
        LOG_ERROR("PTO2 dispatch: header is null");
        return -1;
    }
    LOG_INFO_V0(
        "Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu", thread_idx, static_cast<void *>(header),
        static_cast<uint64_t>(header->ring.task_descriptors_offset),
        static_cast<uint64_t>(header->ring.task_window_size)
    );

    Handshake *hank = static_cast<Handshake *>(runtime->workers);
    LOG_INFO_V0(
        "Thread %d: hank=%p, window_size=%lu", thread_idx, static_cast<void *>(hank),
        static_cast<uint64_t>(header->ring.task_window_size)
    );

    LOG_INFO_V0("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_trackers_[thread_idx].core_num());
    int32_t cur_thread_completed = 0;
    // Non-zero once a scheduler-hang timeout latches; returned in place of the
    // completed count so the caller still sees the negative error rc while the
    // shared end-of-loop flush below runs.
    int32_t timeout_rc = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if SIMPLER_DFX
    auto &l2_swimlane = sched_l2_swimlane_[thread_idx];
    l2_swimlane.reset();
    l2_swimlane.l2_swimlane_enabled = (l2_swimlane_level_ != L2SwimlaneLevel::DISABLED);
#endif

    PTO2TaskSlotState *deferred_release_slot_states[PTO2_DEFERRED_RELEASE_CAP];
    int32_t deferred_release_count = 0;

    // PMU runs require single-issue dispatch — overlapping in-flight tasks
    // pollute per-task PMU counters, so skip the PENDING pre-load phase.
    // Cached at function scope: is_pmu_enabled() is extern "C" and the
    // compiler cannot hoist it across the dispatch loop on its own.
#if SIMPLER_DFX
    const bool pmu_active = is_pmu_enabled();
#else
    // PMU is definitionally off when profiling is compiled out; hard-set false
    // so dispatch keeps its overlapping (non-single-issue) fast path.
    constexpr bool pmu_active = false;
#endif

#if SIMPLER_DFX
    l2_swimlane.sched_start_ts = get_sys_cnt_aicpu();
#endif

#if SIMPLER_DFX
    // Queue-depth snapshot carried across the iteration boundary: each phase
    // emit consumes (phase_start_shared) and refreshes it with its own end
    // snapshot so the next phase's "at_start" equals the previous phase's
    // "at_end".
    //
    // L2SWIMLANE_NUM_QUEUE_SHAPES (3) matches PTO2_NUM_RESOURCE_SHAPES: AIC/AIV/MIX.
    //
    // **Hot-path cost discipline.** Shared depth (PTO2ReadyQueue::size) is two
    // atomic relaxed loads against cache lines that all peer sched threads also
    // write to (enqueue_pos and dequeue_pos bounce on every push + every pop).
    // With both phases emitting per iter that's cross-core loads × thousands of
    // iters per run, a measurable AICPU slowdown. Mitigation: lazy + per-iter
    // cached shared snapshot, refreshed at most once per iteration. The
    // complete-emit and dispatch-emit in the same iter both reuse the same
    // shared sample.
    static_assert(
        L2SWIMLANE_NUM_QUEUE_SHAPES == PTO2_NUM_RESOURCE_SHAPES,
        "queue snapshot width must match runtime resource shape count"
    );
    int16_t phase_start_shared[L2SWIMLANE_NUM_QUEUE_SHAPES] = {0};
    int16_t iter_shared_snapshot[L2SWIMLANE_NUM_QUEUE_SHAPES] = {0};
    bool iter_shared_sampled = false;
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
    auto capture_phase_end = [&](int16_t shared_out[L2SWIMLANE_NUM_QUEUE_SHAPES]) {
        const int16_t *shared_cached = get_or_sample_shared();
        for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++)
            shared_out[s] = shared_cached[s];
    };
    // Queue-mutating phases (Complete / Dummy) push newly-ready consumers
    // straight into the shared ready_queues[] (the local-first buffer is gone),
    // so their end-of-phase shared depth differs from their start. Force a fresh
    // re-sample for those emits — this also refreshes the per-iter cache so the
    // next phase's start snapshot is not stale.
    auto capture_phase_end_fresh = [&](int16_t shared_out[L2SWIMLANE_NUM_QUEUE_SHAPES]) {
        iter_shared_sampled = false;
        capture_phase_end(shared_out);
    };
    if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
        capture_phase_end(phase_start_shared);
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
#if SIMPLER_DFX
        CYCLE_COUNT_START();
        l2_swimlane.sched_loop_count++;
        uint64_t _t0_phase = _t0;
        // Release is the only "no Complete/Dispatch bar" attribution we keep —
        // emitted with its own span in the idle branch below. Iterations that
        // only scan/poll show as blank gaps; the per-loop Poll/Scan bars (PR
        // #1079 debug overlay) were removed since "scheduler is polling when
        // there's nothing to do" carries no actionable signal.
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

#if SIMPLER_DFX
        CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
#endif

        // Phase 1: Check running cores for completion
        int32_t completed_this_turn = 0;

        bool try_completed = tracker.has_any_running_cores();
        if (try_completed) {
            check_running_cores_for_completion(
                thread_idx, hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count
            );
        }
        if (completed_this_turn > 0) {
#if SIMPLER_SCHED_PROFILING
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
                rt_->aicore_mailbox, sched_, deferred_release_slot_states, deferred_release_count,
                PTO2_DEFERRED_RELEASE_CAP
#if SIMPLER_SCHED_PROFILING
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
#if SIMPLER_SCHED_PROFILING
                sched_->tasks_completed.fetch_add(poll_result.completed, std::memory_order_relaxed);
#endif
                int32_t prev = completed_tasks_.fetch_add(poll_result.completed, std::memory_order_relaxed);
                int32_t new_total = prev + poll_result.completed;
                last_progress_count = new_total;
                made_progress = true;
            }
        }

#if SIMPLER_DFX
        if (!try_completed) {
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_swimlane.sched_complete_cycle);
            // Emit on any completion work this iteration — a finished slot OR
            // sub-block retires that did not finish a slot. The latter makes the
            // SPMD harvest tail visible (count field = blocks processed this
            // iteration; on a pure-retire iteration phase_complete_count is 0).
            if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES &&
                (l2_swimlane.phase_complete_count > 0 || l2_swimlane.phase_subretire_count > 0)) {
                // Complete's release_fanin pushes newly-ready consumers into the
                // shared ready_queues[], so the end depth differs from the start.
                int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
                capture_phase_end_fresh(phase_end_shared);
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Complete, _t0_phase, _t1, l2_swimlane.sched_loop_count,
                    l2_swimlane.phase_complete_count + l2_swimlane.phase_subretire_count, /*pop_hit=*/0,
                    /*pop_miss=*/0, phase_start_shared, phase_end_shared
                );
                for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++)
                    phase_start_shared[s] = phase_end_shared[s];
                _t0_phase = _t1;
                l2_swimlane.phase_complete_count = 0;
                l2_swimlane.phase_subretire_count = 0;
            }
        }
#endif

        bool try_pushed = false;

        // Phase 2 drain check
        if (drain_state_.sync_start_pending.load(std::memory_order_acquire) != 0) {
            handle_drain_mode(thread_idx);
            continue;
        }

        // Phase 3: Drain dummy ready queue (S0/S1/S2).
        //
        // Dependency-only tasks bypass AICore dispatch: they go through the
        // scheduler so fanin/fanout edges stay consistent, but completion is
        // signalled inline here. The ready queue is MPMC, and the fanout path
        // uses per-slot locks/atomics, so multiple scheduler threads can share
        // the dependency-only resolve work.
        if (thread_idx < 3) {
            constexpr int DUMMY_DRAIN_BATCH = 8;
            PTO2TaskSlotState *dummy_batch[DUMMY_DRAIN_BATCH];
            int dummy_got = sched_->dummy_ready_queue.pop_batch(dummy_batch, DUMMY_DRAIN_BATCH);
#if SIMPLER_DFX
            // Dummy outer phase: covers handling of all dummies popped this
            // iter. Per-dummy DummyTask markers are emitted to a SEPARATE lane
            // (Worker View AICPU_N) by the converter, so they do not nest
            // under this bar. Resolve emits below DO land on the sched lane
            // and nest under this Dummy outer by time containment.
            uint64_t dummy_outer_t0 =
                (dummy_got > 0 && l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
            for (int di = 0; di < dummy_got; di++) {
                PTO2TaskSlotState &dummy_slot = *dummy_batch[di];

                // ----- DummyTask phase: dummy "task" identity marker. --------
                // The dummy has no AICore presence — start ≈ end (1 cycle
                // wide, just "we identified it"). Converter renders this on
                // Worker View's DUMMY_T{thread} lane so the DAG node is
                // visually present. tasks_processed = task_token low 32 bits
                // (= local_id within ring) so deps.json flow arrows can land.
                // The Resolve work that follows is emitted separately below.
#if SIMPLER_DFX
                if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
                    uint64_t dummy_marker_t = get_sys_cnt_aicpu();
                    uint32_t dummy_id_low32 = static_cast<uint32_t>(dummy_slot.task->task_id.raw & 0xFFFFFFFFu);
                    l2_swimlane_aicpu_record_sched_phase(
                        thread_idx, L2SwimlaneSchedPhaseKind::DummyTask, dummy_marker_t, dummy_marker_t,
                        sched_l2_swimlane_[thread_idx].sched_loop_count, dummy_id_low32
                    );
                }
#endif

                // ----- Resolve work: walk this dummy's consumer list. ------
                // Same 1 µs filter as the main-path Resolve emit suppresses
                // dummies whose consumer release runs sub-microsecond.
#if SIMPLER_DFX
                uint64_t dummy_resolve_t0 =
                    (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
                // [[maybe_unused]] silences -Werror=unused-but-set-variable on
                // the profiling-flags-smoke build path where SIMPLER_DFX is
                // OFF and the Resolve emit below is excluded.
                [[maybe_unused]] uint32_t dummy_consumers = 0;
#if SIMPLER_SCHED_PROFILING
                dummy_consumers = sched_->on_task_complete(dummy_slot, thread_idx).fanout_edges;
#else
                dummy_consumers = sched_->on_task_complete(dummy_slot);
#endif
#if SIMPLER_DFX
                if (dummy_resolve_t0 != 0) {
                    uint64_t dummy_resolve_t1 = get_sys_cnt_aicpu();
                    constexpr uint64_t RESOLVE_EMIT_MIN_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ / 1'000'000;  // 1 µs
                    if (dummy_resolve_t1 - dummy_resolve_t0 >= RESOLVE_EMIT_MIN_CYCLES) {
                        l2_swimlane_aicpu_record_sched_phase(
                            thread_idx, L2SwimlaneSchedPhaseKind::Resolve, dummy_resolve_t0, dummy_resolve_t1,
                            sched_l2_swimlane_[thread_idx].sched_loop_count, dummy_consumers
                        );
                    }
                }
#endif
                // Dummy tasks have no subtasks to retire and no fanout pre-conditions
                // beyond their own producers; release self-reference so the slot can
                // reach CONSUMED once all consumers drain.
                deferred_release_slot_states[deferred_release_count++] = &dummy_slot;
                if (deferred_release_count >= PTO2_DEFERRED_RELEASE_CAP) {
                    while (deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
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
#if SIMPLER_DFX
            // Emit Dummy outer over the whole dummy_drain pass. Span starts at
            // dummy_outer_t0 (captured before the pop_batch) and ends at "now".
            // tasks_processed = dummy_got. Advancing _t0_phase here makes the
            // following Dispatch / EarlyDispatch / second-Complete bars start
            // at this end.
            if (dummy_outer_t0 != 0) {
                uint64_t dummy_outer_t1 = get_sys_cnt_aicpu();
                int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
                capture_phase_end_fresh(phase_end_shared);
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Dummy, dummy_outer_t0, dummy_outer_t1,
                    l2_swimlane.sched_loop_count, static_cast<uint32_t>(dummy_got), /*pop_hit=*/0,
                    /*pop_miss=*/0, phase_start_shared, phase_end_shared
                );
                for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++)
                    phase_start_shared[s] = phase_end_shared[s];
                _t0_phase = dummy_outer_t1;
                // We do NOT re-sync _t0/_t1 — the dummy span will be absorbed
                // into the next CYCLE_COUNT_LAP accumulator. The phase-model
                // anchor (_t0_phase) is the authoritative source for bar spans
                // on the swimlane; the cycle accumulators are coarse aggregates.
            }
#endif
        }

        // Phase 4: MIX-strict-priority dispatch with phase-split and
        // cross-thread idle gating. See dispatch_ready_tasks for the policy.
#if SIMPLER_DFX
        uint64_t dispatch_t0 = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) ? get_sys_cnt_aicpu() : 0;
#endif
        dispatch_ready_tasks(thread_idx, tracker, pmu_active, made_progress, try_pushed);
#if SIMPLER_DFX
        // Emit Dispatch IMMEDIATELY after dispatch_ready_tasks so its span
        // covers the actual publish work — not the trailing second-poll /
        // early-dispatch time. (Pre-redesign the Dispatch emit lived at iter
        // end with span extending past the second poll, which made finish_time
        // events from the second poll fall under the Dispatch bar rather than
        // a Complete bar of their own — confusing for trace consumers.)
        if (dispatch_t0 != 0 && l2_swimlane.phase_dispatch_count > 0) {
            uint64_t dispatch_t1 = get_sys_cnt_aicpu();
            uint64_t pop_hit_delta = l2_swimlane.pop_hit - l2_swimlane.pop_hit_at_last_emit;
            uint64_t pop_miss_delta = l2_swimlane.pop_miss - l2_swimlane.pop_miss_at_last_emit;
            debug_assert(pop_hit_delta < (1ULL << 32));
            debug_assert(pop_miss_delta < (1ULL << 32));
            int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
            capture_phase_end(phase_end_shared);
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::Dispatch, _t0_phase, dispatch_t1, l2_swimlane.sched_loop_count,
                l2_swimlane.phase_dispatch_count, static_cast<uint32_t>(pop_hit_delta),
                static_cast<uint32_t>(pop_miss_delta), phase_start_shared, phase_end_shared
            );
            for (int s = 0; s < L2SWIMLANE_NUM_QUEUE_SHAPES; s++) {
                phase_start_shared[s] = phase_end_shared[s];
            }
            _t0_phase = dispatch_t1;
            l2_swimlane.phase_dispatch_count = 0;
            l2_swimlane.pop_hit_at_last_emit = l2_swimlane.pop_hit;
            l2_swimlane.pop_miss_at_last_emit = l2_swimlane.pop_miss;
        }
#endif

        // Phase 4b: early-dispatch onto spare cores, mirroring the Phase 4 call
        // shape. try_early_dispatch owns its own gating (off-PMU, this
        // thread has a spare slot, no normal ready work queued) and updates
        // made_progress / try_pushed, so this is a single unconditional call — it
        // returns 0 without staging when gated out.
#if SIMPLER_DFX
        bool early_dispatch_record = l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES;
        uint64_t early_dispatch_t0 = early_dispatch_record ? get_sys_cnt_aicpu() : 0;
#endif
        [[maybe_unused]] int32_t staged_count =
            try_early_dispatch(thread_idx, tracker, pmu_active, made_progress, try_pushed);
#if SIMPLER_DFX
        // Emit an EarlyDispatch bar so a staging-dominated iteration is attributed
        // to early-dispatch rather than disappearing into a blank gap.
        if (early_dispatch_record && staged_count > 0) {
            uint64_t early_dispatch_t1 = get_sys_cnt_aicpu();
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::EarlyDispatch, early_dispatch_t0, early_dispatch_t1,
                sched_l2_swimlane_[thread_idx].sched_loop_count, static_cast<uint32_t>(staged_count)
            );
            // prepare_block_for_dispatch bumped phase_dispatch_count while staging;
            // those blocks belong to this EarlyDispatch bar, so clear the counter
            // before it leaks into the next Dispatch bar.
            sched_l2_swimlane_[thread_idx].phase_dispatch_count = 0;
            // Advance _t0_phase so the next phase bar starts at the EarlyDispatch
            // end, not before it (otherwise their spans overlap and the
            // outer-phase mutual-exclusion breaks).
            _t0_phase = early_dispatch_t1;
        }
#endif

#if SIMPLER_DFX
        // Cycle-counter LAP for the iter tail. Dispatch's emit moved earlier
        // (see Phase 4 above) so this branch only routes the time accumulator.
        if (!try_pushed) {
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(l2_swimlane.sched_dispatch_cycle);
        }
#endif

#if !SIMPLER_DFX
        (void)try_completed;
        (void)try_pushed;
#endif

        if (made_progress) {
            idle_iterations = 0;
            last_progress_ts = get_sys_cnt_aicpu();
        } else {
#if SIMPLER_DFX
            uint64_t rel_t0 = (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES && deferred_release_count > 0) ?
                                  get_sys_cnt_aicpu() :
                                  0;
#endif
            while (deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
                (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
            }
#if SIMPLER_DFX
            // Release is a distinct operation from the poll scan — emit it with
            // its own span (Perfetto nests it inside the surrounding poll/idle
            // run by time-containment) rather than competing with poll for one
            // per-iteration label.
            if (rel_t0 != 0) {
                l2_swimlane_aicpu_record_sched_phase(
                    thread_idx, L2SwimlaneSchedPhaseKind::Release, rel_t0, get_sys_cnt_aicpu(),
                    l2_swimlane.sched_loop_count, /*tasks_processed=*/0
                );
            }
#endif
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
#if SIMPLER_DFX
                        ,
                        l2_swimlane.sched_start_ts
#endif
                    );
                    break;
                }
                last_progress_ts = get_sys_cnt_aicpu();
            }
            SPIN_WAIT_HINT();
#if SIMPLER_DFX
            CYCLE_COUNT_LAP(l2_swimlane.sched_idle_cycle);
            // _t0_phase advances through idle laps so the next emitted
            // COMPLETE/DISPATCH bar starts at the iter it actually ran in, not
            // at the start of the preceding idle stretch. The idle/poll time
            // itself is attributed by the activity-fill below — no blanks.
            if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
                _t0_phase = _t1;
            }
#endif
        }
    }

    // Drain any entries left in the deferred-release batch. The in-loop flush
    // only fires on idle iterations and on buffer-full; a loop exit while the
    // last iteration made progress can leave entries un-released. Drop them
    // here so every consumed producer slot completes its on_task_release
    // regardless of which loop-exit path fired.
    while (deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
        (void)sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
        sched_->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
    }

#if SIMPLER_DFX
    // Final-drain: emit any pop_hit / pop_miss accrued since the last
    // dispatch emit (typically the trailing idle loops while waiting for the
    // last in-flight tasks to complete) as a zero-duration synthetic dispatch record so
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
            int16_t phase_end_shared[L2SWIMLANE_NUM_QUEUE_SHAPES];
            capture_phase_end(phase_end_shared);
            l2_swimlane_aicpu_record_sched_phase(
                thread_idx, L2SwimlaneSchedPhaseKind::Dispatch, t_now, t_now, l2_swimlane.sched_loop_count, 0,
                static_cast<uint32_t>(final_pop_hit_delta), static_cast<uint32_t>(final_pop_miss_delta),
                phase_end_shared, phase_end_shared
            );
            l2_swimlane.pop_hit_at_last_emit = l2_swimlane.pop_hit;
            l2_swimlane.pop_miss_at_last_emit = l2_swimlane.pop_miss;
        }
    }
    log_l2_swimlane_summary(thread_idx, cur_thread_completed);
#endif

#if SIMPLER_DFX
    if (l2_swimlane.l2_swimlane_enabled) {
        l2_swimlane_aicpu_flush(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
        if (l2_swimlane_level_ >= L2SwimlaneLevel::SCHED_PHASES) {
            l2_swimlane_aicpu_flush_sched_phase_buffer(thread_idx);
        }
    }
#endif
#if SIMPLER_DFX
    if (is_dump_args_enabled()) {
        dump_args_flush(thread_idx);
    }
#endif
#if SIMPLER_DFX
    if (is_pmu_enabled()) {
        pmu_aicpu_flush_buffers(
            thread_idx, core_trackers_[thread_idx].core_ids(), core_trackers_[thread_idx].core_num()
        );
    }
#endif

    return timeout_rc != 0 ? timeout_rc : cur_thread_completed;
}
