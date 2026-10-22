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

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/pmu_collector_aicore.h"
// AICore dependency scheduling uses one device-side protocol.
#include "scheduler/scheduler_mix.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_watchdog.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

__aicore__ void legacy_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 32;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

struct SchedulerWorkerStats {
    uint64_t bootstrap_task_count{0};
    SchedulerReadyStats ready{};
    uint64_t executed_task_count{0};
    uint64_t task_state_poll_count{0};
    SchedulerWakeStats wake{};
    SchedulerCompletionStats completion{};
    uint64_t idle_iteration_count{0};
    uint64_t backoff_cycles{0};
    uint64_t payload_cycles{0};
    uint64_t kernel_cycles{0};
    uint64_t completion_enqueue_cycles{0};
    uint64_t bootstrap_start_cycles{0};
    uint64_t bootstrap_scan_end_cycles{0};
    uint64_t target_bootstrap_start_cycles{0};
    uint64_t target_bootstrap_end_cycles{0};
    uint64_t bootstrap_target_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t bootstrap_slot_fill_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t drain_start_cycles{0};
    uint64_t drain_end_cycles{0};
    uint64_t exit_wait_start_cycles{0};
    uint64_t exit_observed_cycles{0};
    uint64_t final_stats_publish_start_cycles{0};
    uint64_t final_stats_publish_end_cycles{0};
    uint64_t exit_ack_publish_cycles{0};
};

/**
 * Stage the PMU counters this core just froze for one executed subtask.
 *
 * Called between `pmu_aicore_end()` and the completion publication, so the
 * window the counters describe is this subtask's kernel and nothing after it.
 * The cell array is indexed exactly as the trace cells are, so a mix task's
 * three subtasks land in three cells of their own.
 *
 * `published` carries the task id plus one and is written last. The AICPU
 * reads these cells only after every core has retired, and that ordering is
 * what lets it tell a subtask that executed and counted zero from one this run
 * never reached.
 */
__aicore__ __attribute__((always_inline)) void stage_task_pmu(
    __gm__ void *scheduler_state_base, SchedulerLocalState *local, int64_t task_id, uint8_t subtask,
    uint64_t pmu_reg_base
) {
    if (local->config.pmu_cells_offset == 0 || pmu_reg_base == 0) return;
    if (subtask >= SCHEDULER_TASK_SUBTASK_SLOTS) return;
    __gm__ SchedulerTaskPmuCell *cells =
        scheduler_state_at<SchedulerTaskPmuCell>(scheduler_state_base, local->config.pmu_cells_offset);
    __gm__ SchedulerTaskPmuCell *cell =
        &cells[scheduler_trace_index_at(scheduler_state_base, local, task_id, static_cast<SubtaskSlot>(subtask))];
    pmu_aicore_read_counters(pmu_reg_base, cell->counters, &cell->total_cycles);
    cell->worker_id = static_cast<uint32_t>(local->config.worker_ids[local->config.self_lane]);
    OUT_OF_ORDER_STORE_BARRIER();
    cell->published = static_cast<uint32_t>(task_id) + 1U;
    scheduler_publish_cache_line(cell);
}

__aicore__ __attribute__((always_inline)) void execute_task(__gm__ DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((always_inline)) void local_backoff(uint32_t iterations) {
    for (volatile uint32_t spin = 0; spin < iterations; ++spin) {}
}

__aicore__ __attribute__((always_inline)) void
publish_worker_stats(__gm__ SchedulerWorkerContext *context, const SchedulerWorkerStats &stats) {
    context->bootstrap_task_count = stats.bootstrap_task_count;
    context->ready_enqueue_count = stats.ready.enqueue_count;
    context->ready_batch_count = stats.ready.batch_count;
    context->ready_pop_count = stats.ready.pop_count;
    context->ready_steal_count = stats.ready.steal_count;
    context->ready_cas_retry_count = stats.ready.cas_retry_count;
    context->ready_contention_giveup_count = stats.ready.contention_giveup_count;
    context->executed_task_count = stats.executed_task_count;
    context->task_state_poll_count = stats.task_state_poll_count;
    context->fanin_state_load_count = stats.wake.fanin_state_load_count;
    context->wake_register_count = stats.wake.wake_register_count;
    context->idle_iteration_count = stats.idle_iteration_count;
    context->backoff_cycles = stats.backoff_cycles;
    scheduler_publish_cache_line(&context->bootstrap_task_count);

    context->wake_cas_retry_count = stats.wake.wake_cas_retry_count;
    context->wake_closed_retry_count = stats.wake.wake_closed_retry_count;
    context->wake_migrate_count = stats.wake.wake_migrate_count;
    context->wake_close_count = stats.wake.wake_close_count;
    context->completion_enqueue_count = stats.completion.enqueue_count;
    context->completion_resolve_count = stats.completion.resolve_count;
    context->payload_cycles = stats.payload_cycles;
    context->kernel_cycles = stats.kernel_cycles;
    scheduler_publish_cache_line(&context->wake_cas_retry_count);

    context->completion_enqueue_cycles = stats.completion_enqueue_cycles;
    context->bootstrap_start_cycles = stats.bootstrap_start_cycles;
    context->drain_start_cycles = stats.drain_start_cycles;
    context->drain_end_cycles = stats.drain_end_cycles;
    context->exit_wait_start_cycles = stats.exit_wait_start_cycles;
    context->exit_observed_cycles = stats.exit_observed_cycles;
    context->final_stats_publish_start_cycles = stats.final_stats_publish_start_cycles;
    context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
    context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
    context->bootstrap_slot_fill_aic_cycles = stats.bootstrap_slot_fill_cycles[0];
    context->bootstrap_slot_fill_aiv_cycles = stats.bootstrap_slot_fill_cycles[1];
    scheduler_publish_cache_line(&context->completion_enqueue_cycles);
    scheduler_publish_cache_line(&context->bootstrap_slot_fill_aic_cycles);

    scheduler_gm_store(context->bootstrap_scan_end_cycles, stats.bootstrap_scan_end_cycles);
    scheduler_gm_store(context->target_bootstrap_start_cycles, stats.target_bootstrap_start_cycles);
    scheduler_gm_store(context->target_bootstrap_end_cycles, stats.target_bootstrap_end_cycles);
    scheduler_gm_store(context->bootstrap_target_aic_cycles, stats.bootstrap_target_cycles[0]);
    scheduler_gm_store(context->bootstrap_target_aiv_cycles, stats.bootstrap_target_cycles[1]);
}

// Keep the pre-kernel timestamps out of the indirect kernel call's live set.
// Inlining this path makes the resident loop spill enough state to fault on A5.
__aicore__ __attribute__((noinline)) void stage_task_trace_before_execution(
    SchedulerExecutorTaskTrace *trace, uint64_t ready_scan_start, uint64_t ready_observe, uint64_t completion_id,
    uint64_t completion_inbox_index, bool phase_timing_enabled
) {
    trace->ready_observe_cycles = ready_observe;
    if (phase_timing_enabled) {
        trace->ready_scan_start_cycles = ready_scan_start;
        trace->completion_id = completion_id;
        trace->completion_inbox_index = completion_inbox_index;
    }
}

// The Completion Inbox token follows this call, so the Scheduler cannot
// observe a partial Executor trace.
__aicore__ __attribute__((noinline)) void
stage_task_trace_before_completion(SchedulerExecutorTaskTrace *trace, uint64_t kernel_start, uint64_t kernel_end) {
    trace->kernel_start_cycles = kernel_start;
    trace->kernel_end_cycles = kernel_end;
}

__aicore__ bool bootstrap_ready_graph(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t scheduler_count, SchedulerWorkerStats *stats,
    uint64_t profiling_level, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (scheduler_count == 0 || scheduler->config.scheduler_index >= scheduler_count || ssbuf_region == nullptr ||
        scheduler == nullptr)
        return false;
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    if (phase_timing_enabled) stats->bootstrap_start_cycles = scheduler_cycles();
    SchedulerReadyBatch batches[SCHEDULER_READY_QUEUE_COUNT]{};
    uint64_t tasks_per_scheduler = graph.task_count / scheduler_count;
    uint64_t remainder = graph.task_count % scheduler_count;
    uint64_t task_begin =
        scheduler->config.scheduler_index * tasks_per_scheduler +
        (scheduler->config.scheduler_index < remainder ? scheduler->config.scheduler_index : remainder);
    uint64_t task_end = task_begin + tasks_per_scheduler + (scheduler->config.scheduler_index < remainder ? 1 : 0);
    for (uint64_t task_ref = task_begin; task_ref < task_end; ++task_ref) {
        __gm__ SchedulerTaskMetadata *metadata =
            scheduler_task_metadata_at(scheduler_state_base, scheduler, static_cast<int64_t>(task_ref));
        scheduler_observe_cache_line(metadata);
        if (!scheduler_task_is_executable(metadata->flags)) continue;
        const bool has_fanin = scheduler_task_has_fanin(metadata->flags);
        SchedulerRouteResult route =
            has_fanin ? scheduler_bootstrap_route_task(
                            graph, scheduler_state_base, scheduler, run_control, static_cast<int64_t>(task_ref),
                            phase_timing_enabled ? &stats->wake : nullptr
                        ) :
                        SchedulerRouteResult::READY_TO_ENQUEUE;
        if (route == SchedulerRouteResult::ERROR) return false;
        if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
            const uint32_t core_type = scheduler_task_ready_queue(metadata->flags, scheduler_task_lanes(*metadata));
            if (!scheduler_bootstrap_ready_batch_append(
                    scheduler_state_base, scheduler, static_cast<int64_t>(task_ref), &batches[core_type],
                    phase_timing_enabled ? &stats->ready : nullptr, profiling_level
                ))
                return false;
        }
        if (phase_timing_enabled) ++stats->bootstrap_task_count;
    }
    scheduler_cache_barrier();
    // Bootstrap publishes through the same type-routed CAS push the steady state
    // uses: a task's type decides whose inbox it lands in, and that core is not
    // this one. The push sets the target's directory bit itself, so there is no
    // per-publisher bitmap left to aggregate -- the barrier below still orders
    // every scan ahead of the first consume.
    for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
        if (!scheduler_ready_batch_push(
                scheduler_state_base, scheduler, type, &batches[type], phase_timing_enabled ? &stats->ready : nullptr
            ))
            return false;
    }
    if (phase_timing_enabled) stats->bootstrap_scan_end_cycles = scheduler_cycles();

    uint64_t arrived = scheduler_gm_fetch_add(run_control->bootstrap_scan_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) {
        scheduler_gm_store(run_control->bootstrap_scan_complete, UINT64_C(1));
    } else {
        uint32_t barrier_backoff = kInitialBackoffIterations;
        const uint64_t barrier_start = scheduler_cycles();
        while (scheduler_gm_query(run_control->bootstrap_scan_complete) == 0) {
            if (scheduler_gm_query(run_control->scheduler_error) != 0) return false;
            if (scheduler_watchdog_expired(
                    barrier_start, scheduler_cycles(), scheduler_gm_query(run_control->scheduler_timeout_cycles)
                )) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, scheduler,
                    SchedulerErrorSite::BOOTSTRAP_SCAN_TIMEOUT
                );
                return false;
            }
            local_backoff(barrier_backoff);
            if (barrier_backoff < kMaximumBackoffIterations) barrier_backoff <<= 1;
        }
    }
    // What this core published says nothing about its own inboxes now: a push
    // routes by type and lands on a peer. The mask tracks only the directory
    // bits this core owns, and the claim path activates a type the first time it
    // looks at it, so starting empty is correct rather than merely safe.
    scheduler->owner_ready_queue_mask = 0;
    if (phase_timing_enabled) stats->target_bootstrap_start_cycles = scheduler_cycles();

    for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
        scheduler_initialize_free_slot(&scheduler->slots[slot_index]);
    }
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
        uint64_t target_start = phase_timing_enabled ? scheduler_cycles() : 0;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        uint32_t type = scheduler_core_type_index(target->core_type);
        // A slot state defaults to EMPTY, not FREE, so the Mix slots this core
        // coordinates have to be opened here too; left EMPTY, placement would
        // never find capacity and Mix work would silently never dispatch.
        if (scheduler->is_mix_coordinator() && cluster_lane != 0) {
            for (uint32_t mix_slot = 0; mix_slot < SCHEDULER_MIX_SLOT_COUNT; ++mix_slot) {
                scheduler_initialize_free_slot(&scheduler->mix_slots[cluster_lane][mix_slot]);
            }
        }
        if (phase_timing_enabled) stats->bootstrap_target_cycles[type] += scheduler_cycles() - target_start;
    }
    if (phase_timing_enabled) stats->target_bootstrap_end_cycles = scheduler_cycles();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed.
    uint64_t ready_victim_cursors[SCHEDULER_READY_QUEUE_COUNT]{
        (scheduler->config.scheduler_index + 1) % scheduler_count,
        (scheduler->config.scheduler_index + 1) % scheduler_count,
    };
    bool fill_failed = false;
    if (!scheduler_fill_cluster_mix_slots(
            graph, scheduler_state_base, scheduler, run_control, &ready_victim_cursors[SCHEDULER_MIX_QUEUE],
            phase_timing_enabled ? &stats->ready : nullptr, profiling_level, true, ssbuf_region, nullptr
        ))
        return false;
    (void)scheduler_fill_own_slots(
        graph, scheduler_state_base, scheduler, run_control, ready_victim_cursors,
        phase_timing_enabled ? &stats->ready : nullptr, profiling_level, 0, &fill_failed, ssbuf_region
    );
    if (fill_failed) return false;

    // Nothing published above needs ordering against this announcement. A wake
    // list travels by atomics on both sides: a resolver detaches it with an
    // exchange and reads its links with an atomic load. A ready batch is found
    // by a poll that repeats, so a queue head this core cannot see yet is read
    // again on the next pass.
    arrived = scheduler_gm_fetch_add(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) scheduler_gm_store(run_control->bootstrap_complete, UINT64_C(1));
    return true;
}

// Keep all arguments register-passed. The local state owns the immutable
// configuration; only the SSBUF base is derived inside the resident loop.
__aicore__ bool run_ready_dispatch_loop(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, SchedulerWorkerStats *stats, uint64_t profiling_level
) {
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = scheduler_ssbuf_region(platform_ssbuf_hardware_base_address());
    const uint32_t cluster_lane = context->config.self_lane;
    // Non-zero only when this run enabled PMU and this core has a PMU window;
    // it is the whole gate, so a run with PMU off issues no counter access and
    // stages no cell.
    const uint64_t pmu_reg_base =
        SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_PMU) ? get_aicore_pmu_reg_base() : 0;
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t scheduler_count = context->config.scheduler_count;
    if (scheduler_count == 0) return false;
    if (ssbuf_region == nullptr || cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    uint64_t ready_victim_cursors[SCHEDULER_READY_QUEUE_COUNT]{
        (context->config.scheduler_index + 1) % scheduler_count,
        (context->config.scheduler_index + 1) % scheduler_count,
    };
    // Mix generations this core has already run, and what it has told the
    // coordinator about them. Both are per Mix slot and local to this core.
    uint32_t executed_mix_generations[SCHEDULER_MIX_SLOT_COUNT]{};
    uint64_t mix_completion_publication = 0;
    uint32_t scan_start = 0;
    uint64_t next_dispatch_sequence = 1;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    if (phase_timing_enabled) context->profiling->loop_iter = UINT32_MAX;
    uint64_t idle_start_cycles = 0;
    bool idle_active = false;
    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (phase_timing_enabled) {
                stats->exit_observed_cycles = get_sys_cnt_aicore();
                if (idle_active) {
                    scheduler_append_idle_activity(
                        scheduler_state_base, context, idle_start_cycles, stats->exit_observed_cycles
                    );
                }
            }
            break;
        }
        if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
            scheduler_error_poll_count = 0;
            if (scheduler_gm_query(run_control->scheduler_error) != 0) return false;
        }

        if (phase_timing_enabled) ++context->profiling->loop_iter;
        const uint64_t idle_candidate_start = phase_timing_enabled ? scheduler_cycles() : 0;

        bool scheduler_progress = false;
        if (!scheduler_refresh_ready_inbox(scheduler_state_base, context)) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::READY_INBOX_REFRESH_FAILED
            );
            return false;
        }
        {
            uint64_t direct_refilled_slot_mask = 0;
            SchedulerRefillCandidates refill_candidates{};
            const bool completion_progress = scheduler_service_cluster_completions(
                graph, scheduler_state_base, context, run_control, phase_timing_enabled ? &stats->wake : nullptr,
                phase_timing_enabled ? &stats->ready : nullptr, phase_timing_enabled ? &stats->completion : nullptr,
                profiling_level, &direct_refilled_slot_mask, ssbuf_region, &refill_candidates
            );
            scheduler_progress = completion_progress;
            if (!scheduler_fill_mix_after_completions(
                    graph, scheduler_state_base, context, run_control, &refill_candidates.mix_ready,
                    &ready_victim_cursors[SCHEDULER_MIX_QUEUE], phase_timing_enabled ? &stats->ready : nullptr,
                    profiling_level, true, ssbuf_region, &scheduler_progress
                ))
                return false;
            bool fill_failed = false;
            const bool dispatch_progress = scheduler_fill_own_slots(
                graph, scheduler_state_base, context, run_control, ready_victim_cursors,
                phase_timing_enabled ? &stats->ready : nullptr, profiling_level, direct_refilled_slot_mask,
                &fill_failed, ssbuf_region
            );
            scheduler_progress = dispatch_progress || scheduler_progress;
            if (fill_failed) return false;
        }

        if (phase_timing_enabled && scheduler_progress && idle_active) {
            scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, idle_candidate_start);
            idle_active = false;
        }

        int32_t ready_slot = -1;
        uint32_t ready_generation = 0;
        uint64_t ready_scan_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        {
            uint32_t local_slot = UINT32_MAX;
            uint64_t local_generation = 0;
            // This core filled the slot it is about to execute, so the local
            // ready mask names it exactly. The generation no longer has to equal
            // a running sequence: that requirement came from an Executor that
            // scanned SSBUF blind and could not tell whether a publication was
            // meant for it. Only the slot's own state and a non-zero generation
            // are still contracts worth checking.
            if (scheduler_local_ready_pop(context, &local_slot, &local_generation)) {
                if (scheduler_dispatch_state(local_generation) != SchedulerDispatchSlotState::READY ||
                    scheduler_dispatch_generation(local_generation) == 0) {
                    scheduler_record_error(
                        run_control, context->own_slots()[local_slot].task_ref, SchedulerGraphResult::INVALID_ARGUMENTS,
                        &graph, context, SchedulerErrorSite::EXECUTOR_PREFERRED_SLOT_INVALID
                    );
                    return false;
                }
                ready_slot = static_cast<int32_t>(local_slot);
                ready_generation = scheduler_dispatch_generation(local_generation);
            }
        }
        if (ready_slot >= 0) {
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            SchedulerLocalSlotState *local_slot = &context->slots[slot_index];
            const int32_t timing_slot = context->timing_slot(cluster_lane, slot_index);
            __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
                scheduler_state_base, context->dispatch_payload_offset(context->config.self_lane, slot_index)
            );
            const bool commit_scheduler_trace = task_timing_enabled;
            const bool commit_task_timing = timing_slot >= 0 && timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
            const bool commit_executor_trace = commit_scheduler_trace || commit_task_timing;
            uint64_t ready_observe = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (phase_timing_enabled && idle_active) {
                scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, ready_observe);
                idle_active = false;
            }
            // This core wrote this payload itself, so it needs no invalidation;
            // only the Mix path below reads a payload another core prepared.
            const int64_t task_ref = local_slot->task_ref;
            if (!scheduler_task_ref_valid(graph, task_ref) || ready_generation == 0) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
                return false;
            }
            ++next_dispatch_sequence;
            SchedulerExecutorTaskTrace execution_trace{};
            if (commit_scheduler_trace || commit_task_timing) {
                uint64_t local_completion_index = stats->completion.enqueue_count;
                stage_task_trace_before_execution(
                    &execution_trace, ready_scan_start, ready_observe,
                    scheduler_completion_id(context, local_completion_index), context->config.scheduler_index,
                    phase_timing_enabled
                );
            }
            OUT_OF_ORDER_STORE_BARRIER();
            uint64_t kernel_start = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (pmu_reg_base != 0) pmu_aicore_begin();
            execute_task(payload);
            if (pmu_reg_base != 0) {
                pmu_aicore_end();
                // The subtask this lane ran, not the lane: a task's subtasks
                // are placed on whichever lanes of the cluster are free, so a
                // mix task's three records are told apart by this and nothing
                // else.
                const uint8_t subtask_slot = scheduler_worker ? local_slot->subtask_slot : ssbuf_control->subtask_slot;
                stage_task_pmu(scheduler_state_base, context, task_ref, subtask_slot, pmu_reg_base);
            }
            uint64_t kernel_end = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (commit_executor_trace) stage_task_trace_before_completion(&execution_trace, kernel_start, kernel_end);
            if (commit_executor_trace) context->profiling->executor_traces[slot_index] = execution_trace;
            uint64_t completion_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            // A core retires its own slots, so a completion stays in local state
            // and never crosses the cluster's SSBUF.
            context->local_completed_generations[slot_index] = ready_generation;
            if (phase_timing_enabled) ++stats->completion.enqueue_count;
            uint64_t completion_end = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (phase_timing_enabled) {
                ++stats->executed_task_count;
                stats->payload_cycles += kernel_start - ready_observe;
                stats->kernel_cycles += kernel_end - kernel_start;
                stats->completion_enqueue_cycles += completion_end - completion_start;
            }
            scan_start = (slot_index + 1) % SCHEDULER_PENDING_SLOT_COUNT;
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }

        // A Mix subtask arrives on a different channel from this core's own
        // work: its slot belongs to the cluster's AIC, so the task reference
        // comes from the Mix dispatch control rather than local slot state, the
        // payload from the Mix payload array, the executor trace goes to global
        // memory, and the generation is published to the Mix completion inbox.
        // Kept as its own block rather than folded into the path above, because
        // every one of those four endpoints differs and only the kernel call is
        // shared.
        if (context->config.self_lane != 0) {
            for (uint32_t mix_slot = 0; mix_slot < SCHEDULER_MIX_SLOT_COUNT; ++mix_slot) {
                const SCHEDULER_SSBUF volatile SchedulerSsbufDispatchControl *mix_control =
                    &ssbuf_region->mix_dispatch[cluster_lane][mix_slot];
                const uint64_t mix_publication = scheduler_ssbuf_load_relaxed(&mix_control->publication);
                const uint32_t mix_generation = static_cast<uint32_t>(mix_publication);
                if (mix_generation == 0 || mix_generation == executed_mix_generations[mix_slot]) continue;
                scheduler_cache_barrier();
                const int64_t mix_task_ref = mix_control->task_ref;
                if (!scheduler_task_ref_valid(graph, mix_task_ref)) {
                    scheduler_record_error(
                        run_control, mix_task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                        SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                    );
                    return false;
                }
                __gm__ DispatchPayload *mix_payload = scheduler_state_at<DispatchPayload>(
                    scheduler_state_base, context->mix_payload_offset(cluster_lane, mix_slot)
                );
                scheduler_observe_dispatch_payload_control(mix_payload);
                scheduler_observe_dispatch_payload_arguments(mix_payload);
                scheduler_observe_dispatch_payload_barrier();
                const int32_t mix_timing_slot = static_cast<int32_t>(mix_publication >> 32);
                const bool commit_mix_task_timing =
                    mix_timing_slot >= 0 && mix_timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
                const bool commit_mix_trace = task_timing_enabled || commit_mix_task_timing;
                const uint64_t mix_ready_observe = commit_mix_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
                SchedulerExecutorTaskTrace mix_trace{};
                if (commit_mix_trace) {
                    stage_task_trace_before_execution(
                        &mix_trace, ready_scan_start, mix_ready_observe,
                        scheduler_completion_id(context, stats->completion.enqueue_count),
                        context->config.scheduler_index, phase_timing_enabled
                    );
                }
                OUT_OF_ORDER_STORE_BARRIER();
                const uint64_t mix_kernel_start = commit_mix_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
                if (pmu_reg_base != 0) pmu_aicore_begin();
                execute_task(mix_payload);
                if (pmu_reg_base != 0) {
                    pmu_aicore_end();
                    stage_task_pmu(
                        scheduler_state_base, context, mix_task_ref, mix_control->subtask_slot, pmu_reg_base
                    );
                }
                const uint64_t mix_kernel_end = commit_mix_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
                if (commit_mix_trace) {
                    stage_task_trace_before_completion(&mix_trace, mix_kernel_start, mix_kernel_end);
                    __gm__ SchedulerExecutorTaskTrace *published_mix_trace =
                        scheduler_state_at<SchedulerExecutorTaskTrace>(
                            scheduler_state_base, context->mix_trace_offset(cluster_lane, mix_slot)
                        );
                    published_mix_trace->kernel_start_cycles = mix_trace.kernel_start_cycles;
                    published_mix_trace->kernel_end_cycles = mix_trace.kernel_end_cycles;
                    published_mix_trace->ready_scan_start_cycles = mix_trace.ready_scan_start_cycles;
                    published_mix_trace->ready_observe_cycles = mix_trace.ready_observe_cycles;
                    published_mix_trace->completion_id = mix_trace.completion_id;
                    published_mix_trace->completion_inbox_index = mix_trace.completion_inbox_index;
                    scheduler_publish_cache_line(published_mix_trace);
                }
                executed_mix_generations[mix_slot] = mix_generation;
                mix_completion_publication =
                    scheduler_ssbuf_update_completion(mix_completion_publication, mix_slot, mix_generation);
                scheduler_cache_barrier();
                scheduler_ssbuf_store_relaxed(
                    &ssbuf_region->lanes[cluster_lane].mix_completion.publication, mix_completion_publication
                );
                if (phase_timing_enabled) {
                    ++stats->completion.enqueue_count;
                    ++stats->executed_task_count;
                    stats->kernel_cycles += mix_kernel_end - mix_kernel_start;
                }
                backoff_iterations = kInitialBackoffIterations;
            }
        }

        if (scheduler_progress) {
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }
        scheduler_flush_completions(run_control, context);
        if (phase_timing_enabled) ++stats->idle_iteration_count;
        uint64_t backoff_start = get_sys_cnt_aicore();
        local_backoff(backoff_iterations);
        uint64_t backoff_end = get_sys_cnt_aicore();
        if (phase_timing_enabled) stats->backoff_cycles += backoff_end - backoff_start;
        if (phase_timing_enabled) {
            if (!idle_active) {
                idle_start_cycles = idle_candidate_start;
                idle_active = true;
            }
        }
        if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
    }
    return true;
}

struct SchedulerNoProfilingState {};

template <bool WithProfiling>
__aicore__ __attribute__((noinline)) void run_resident_scheduler(
    __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles
) {
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    SchedulerGraphView graph{
        context->graph_storage_address,
        context->graph_reserved_address,
        context->graph_task_count,
        context->task_window_last_index,
    };
    SchedulerWorkerStats stats{};
    std::conditional_t<WithProfiling, SchedulerLocalProfilingState, SchedulerNoProfilingState> profile{};
    SchedulerLocalState scheduler_local_state{};
    if constexpr (WithProfiling) scheduler_local_state.profiling = &profile;
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = scheduler_ssbuf_region(platform_ssbuf_hardware_base_address());
    const bool config_valid =
        scheduler_initialize_local_config(scheduler_state_base, context, &graph, &scheduler_local_state);
    if (!config_valid)
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::BOOTSTRAP_FAILED
        );
    const uint32_t cluster_lane = scheduler_local_state.config.self_lane;
    uint64_t descriptor_cache_observed_cycles = 0;
    if (config_valid) {
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(graph.storage_address));
        descriptor_cache_observed_cycles = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (context->is_scheduler != 0) {
            scheduler_ssbuf_initialize(ssbuf_region);
        }
        if (context->is_scheduler != 0 &&
            !bootstrap_ready_graph(
                graph, scheduler_state_base, &scheduler_local_state, run_control,
                scheduler_local_state.config.scheduler_count, &stats, profiling_level, ssbuf_region
            )) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::BOOTSTRAP_FAILED
            );
        }
    }

    // Sole AICore-wide DMB: after this point a prepared slot may execute.
    uint32_t startup_signal = 0;
    bool register_release_timed_out = false;
    const uint64_t register_wait_start = get_sys_cnt_aicore();
    while (startup_signal != AICPU_IDLE_TASK_ID && startup_signal != AICORE_EXIT_SIGNAL) {
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (scheduler_watchdog_expired(
                register_wait_start, get_sys_cnt_aicore(), scheduler_gm_query(run_control->scheduler_timeout_cycles)
            )) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, context,
                SchedulerErrorSite::REGISTER_RELEASE_TIMEOUT
            );
            register_release_timed_out = true;
            break;
        }
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL || register_release_timed_out) {
        if (phase_timing_enabled) stats.exit_observed_cycles = get_sys_cnt_aicore();
    } else {
        uint64_t register_release_cycles = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (phase_timing_enabled) {
            context->trace_aicore_entry_cycles = aicore_entry_cycles;
            context->trace_handshake_publish_cycles = handshake_publish_cycles;
            context->trace_register_release_cycles = register_release_cycles;
            context->trace_descriptor_cache_observed_cycles = descriptor_cache_observed_cycles;
            scheduler_publish_cache_line(&context->trace_aicore_entry_cycles);
            scheduler_publish_cache_line(&context->trace_register_release_cycles);
        }
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        if (config_valid) {
            if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM || !scheduler_ssbuf_is_initialized(ssbuf_region)) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
            } else {
                (void)run_ready_dispatch_loop(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &stats, profiling_level
                );
            }
        }
    }

    scheduler_flush_completions(run_control, &scheduler_local_state);

    if (phase_timing_enabled && stats.exit_wait_start_cycles == 0) stats.exit_wait_start_cycles = get_sys_cnt_aicore();
    const uint64_t exit_wait_start = get_sys_cnt_aicore();
    while (stats.exit_observed_cycles == 0 &&
           static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
        if (scheduler_watchdog_expired(
                exit_wait_start, get_sys_cnt_aicore(), scheduler_gm_query(run_control->scheduler_timeout_cycles)
            )) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, context,
                SchedulerErrorSite::EXIT_WAIT_TIMEOUT
            );
            break;
        }
        SPIN_WAIT_HINT();
    }
    if (phase_timing_enabled && stats.exit_observed_cycles == 0) stats.exit_observed_cycles = get_sys_cnt_aicore();
    if (phase_timing_enabled) {
        stats.final_stats_publish_start_cycles = get_sys_cnt_aicore();
        publish_worker_stats(context, stats);
        stats.final_stats_publish_end_cycles = get_sys_cnt_aicore();
        stats.exit_ack_publish_cycles = get_sys_cnt_aicore();
        context->final_stats_publish_end_cycles = stats.final_stats_publish_end_cycles;
        context->exit_ack_publish_cycles = stats.exit_ack_publish_cycles;
        scheduler_publish_cache_line(&context->completion_enqueue_cycles);
    }
    // A local register/exit watchdog may have ended the loop without AICPU
    // signalling EXIT. Its current-run signal is published after the return
    // gate reset, so wait for it before making the acknowledgement visible.
    // The finalizer also folds this core's `scheduler_error` once observed.
    wait_for_aicpu_exit_signal();
    // Complete preceding GM writes before ACK. Stays outside profiling: the
    // tail is optional, this ordering is not.
    OUT_OF_ORDER_STORE_BARRIER();
}

}  // namespace

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    // The host's selection and its bootstrap address are on their own line, so
    // observing the handshake line does not observe them. One observe covers the
    // whole struct because it is exactly one 64-byte line.
    __gm__ SchedulerBootstrapInputs *bootstrap = &runtime->dev.scheduler_bootstrap;
    scheduler_observe_cache_line(bootstrap);
    const uint32_t runtime_mode = bootstrap->runtime_mode;
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->dev.workers[block_idx]);
    const uint32_t profiling_flag = get_aicore_profiling_flag();
    scheduler_observe_cache_line(handshake);
    if (runtime_mode != SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING &&
        runtime_mode != SCHEDULER_RUNTIME_MODE_RESIDENT_READY) {
        legacy_aicore_execute(runtime, block_idx, core_type);
        acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
        return;
    }
    const bool chip_swimlane_enabled = SIMPLER_GET_DFX_FLAG(profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    uint64_t aicore_entry_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
    const uint64_t report_epoch = get_aicore_report_epoch();
    if (report_epoch != 0) {
        // Native program run. Everything this report stands for is payload —
        // `aicore_done` and the two reverse-hand-off words this core is about
        // to wait on, which a predecessor run left set. Clearing them before
        // the marker is what makes an accepted report also mean "READY is not
        // last run's". `report_epoch` alone comes after the barrier, so the
        // AICPU cannot see this run's epoch without all of it.
        //
        // Only a stamped run may clear them: an unstamped one cannot tell its
        // own leftovers from a predecessor's, and its AICPU does not gate the
        // reply on the epoch either.
        aicore_stage_native_report(
            handshake, get_physical_core_id(), core_type, static_cast<uint32_t>(block_idx) + 1,
            SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING
        );
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->report_epoch = report_epoch;
    } else {
        // Kernel/persistent launch: unchanged. `aicore_done` is itself the
        // marker, its per-run reset is what makes it meaningful, and no stamp
        // is written.
        handshake->physical_core_id = get_physical_core_id();
        handshake->core_type = core_type;
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->aicore_done = block_idx + 1;
    }
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    uint64_t handshake_publish_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

    // AICPU publishes the fully configured context through GM. This lets
    // AICore perform all pre-kernel work without consuming a DMB launch.
    // The host-published base plus this core's index gives the same address the
    // AICPU republishes on the handshake at hand-off, so it carries enough
    // immutable scheduler-state addressing for the READY watchdog. AICPU fills
    // the topology-dependent context fields before READY, so this pre-publish
    // view must not be reused for execution.
    __gm__ SchedulerWorkerContext *pending_context = reinterpret_cast<__gm__ SchedulerWorkerContext *>(
        scheduler_worker_context_address(bootstrap->worker_context_base, block_idx)
    );
    scheduler_observe_cache_line(pending_context);
    __gm__ void *pending_scheduler_state_base =
        reinterpret_cast<__gm__ void *>(pending_context->scheduler_state_base_address);
    __gm__ SchedulerRunControl *pending_run_control =
        scheduler_state_at<SchedulerRunControl>(pending_scheduler_state_base, pending_context->run_control_offset);
    scheduler_observe_cache_line(pending_run_control);
    uint32_t startup_signal = 0;
    const uint64_t context_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_RESIDENT_READY && startup_signal != AICORE_EXIT_SIGNAL) {
        scheduler_observe_cache_line(handshake);
        if (scheduler_gm_query(pending_run_control->scheduler_error) != 0) {
            startup_signal = AICORE_EXIT_SIGNAL;
            break;
        }
        if (scheduler_watchdog_expired(
                context_wait_start, get_sys_cnt_aicore(),
                scheduler_gm_query(pending_run_control->scheduler_timeout_cycles)
            )) {
            scheduler_record_error(
                pending_run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, nullptr,
                pending_context, SchedulerErrorSite::CONTEXT_READY_TIMEOUT
            );
            startup_signal = AICORE_EXIT_SIGNAL;
            break;
        }
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        // A local watchdog or scheduler error can request exit before AICPU
        // has reset this run's return gate. Wait for its actual DMB EXIT,
        // which AICPU publishes only after that reset. Do this before ACK:
        // after ACK the AICPU may close the window back to IDLE.
        wait_for_aicpu_exit_signal();
        // The AICPU reads this acknowledgement as the point this core's writes
        // have landed, and the timeout above publishes its error through
        // `pending_run_control`. Complete the preceding GM writes first.
        OUT_OF_ORDER_STORE_BARRIER();
        acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
        return;
    }

    // READY publishes AICPU's topology-dependent context updates. Re-observe
    // them after the acquire point instead of using the pre-publish cache view.
    scheduler_observe_cache_line(handshake);
    __gm__ SchedulerWorkerContext *context = reinterpret_cast<__gm__ SchedulerWorkerContext *>(handshake->task);
    scheduler_observe_cache_line(context);
    scheduler_observe_cache_line(&context->task_metadata_offset);
    __gm__ void *scheduler_state_base = reinterpret_cast<__gm__ void *>(context->scheduler_state_base_address);
    __gm__ SchedulerRunControl *run_control =
        scheduler_state_at<SchedulerRunControl>(scheduler_state_base, context->run_control_offset);
    scheduler_observe_cache_line(run_control);
    scheduler_observe_cache_line(&run_control->chip_swimlane_level);
    const uint64_t profiling_level = scheduler_gm_query(run_control->chip_swimlane_level);
    if (scheduler_task_timing_enabled(profiling_level) ||
        scheduler_gm_query(run_control->sampled_task_timing_enabled) != 0) {
        run_resident_scheduler<true>(
            context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, handshake_publish_cycles
        );
    } else {
        run_resident_scheduler<false>(
            context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, handshake_publish_cycles
        );
    }
    acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
}
