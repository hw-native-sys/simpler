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
// AICore dependency scheduling uses one device-side protocol.
#include "scheduler/scheduler_dispatch.h"
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

    scheduler_gm_publish(context->bootstrap_scan_end_cycles, stats.bootstrap_scan_end_cycles);
    scheduler_gm_publish(context->target_bootstrap_start_cycles, stats.target_bootstrap_start_cycles);
    scheduler_gm_publish(context->target_bootstrap_end_cycles, stats.target_bootstrap_end_cycles);
    scheduler_gm_publish(context->bootstrap_target_aic_cycles, stats.bootstrap_target_cycles[0]);
    scheduler_gm_publish(context->bootstrap_target_aiv_cycles, stats.bootstrap_target_cycles[1]);
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
    uint64_t profiling_level, SchedulerDeferredAivQueue *deferred_aiv,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (scheduler_count == 0 || scheduler->config.scheduler_index >= scheduler_count || ssbuf_region == nullptr ||
        scheduler == nullptr)
        return false;
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    if (phase_timing_enabled) stats->bootstrap_start_cycles = scheduler_cycles();
    SchedulerReadyBatch batches[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t tasks_per_scheduler = graph.task_count / scheduler_count;
    uint64_t remainder = graph.task_count % scheduler_count;
    uint64_t task_begin =
        scheduler->config.scheduler_index * tasks_per_scheduler +
        (scheduler->config.scheduler_index < remainder ? scheduler->config.scheduler_index : remainder);
    uint64_t task_end = task_begin + tasks_per_scheduler + (scheduler->config.scheduler_index < remainder ? 1 : 0);
    for (uint64_t task_id = task_begin; task_id < task_end; ++task_id) {
        __gm__ SchedulerTaskMetadata *metadata =
            scheduler_task_metadata_at(scheduler_state_base, scheduler, static_cast<int64_t>(task_id));
        scheduler_observe_cache_line(metadata);
        if (!scheduler_task_is_executable(metadata->flags)) continue;
        const bool has_fanin = scheduler_task_has_fanin(metadata->flags);
        SchedulerRouteResult route =
            has_fanin ? scheduler_bootstrap_route_task(
                            graph, scheduler_state_base, scheduler, run_control, static_cast<int64_t>(task_id),
                            phase_timing_enabled ? &stats->wake : nullptr
                        ) :
                        SchedulerRouteResult::READY_TO_ENQUEUE;
        if (route == SchedulerRouteResult::ERROR) return false;
        if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
            const uint32_t core_type =
                scheduler_metadata_core_type_index(scheduler_metadata_single_subtask_slot(metadata->active_mask));
            if (!scheduler_bootstrap_ready_batch_append(
                    scheduler_state_base, scheduler, static_cast<int64_t>(task_id), &batches[core_type],
                    phase_timing_enabled ? &stats->ready : nullptr, profiling_level
                ))
                return false;
        }
        if (phase_timing_enabled) ++stats->bootstrap_task_count;
    }
    scheduler_cache_barrier();
    uint64_t ready_types = 0;
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        if (!scheduler_bootstrap_ready_batch_publish(
                scheduler_state_base, scheduler, type, scheduler->config.scheduler_index, &batches[type],
                phase_timing_enabled ? &stats->ready : nullptr, &ready_types
            ))
            return false;
    }
    __gm__ SchedulerReadyDirectory *ready_directory = scheduler_ready_directory_at(scheduler_state_base, scheduler);
    scheduler_gm_store(ready_directory->bootstrap_ready_types[scheduler->config.scheduler_index], ready_types);
    if (phase_timing_enabled) stats->bootstrap_scan_end_cycles = scheduler_cycles();

    uint64_t arrived = scheduler_gm_fetch_add(run_control->bootstrap_scan_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) {
        scheduler_bootstrap_ready_directory_publish(scheduler_state_base, scheduler, scheduler_count);
        scheduler_gm_publish(run_control->bootstrap_scan_complete, UINT64_C(1));
    } else {
        uint32_t barrier_backoff = kInitialBackoffIterations;
        const uint64_t barrier_start = scheduler_cycles();
        while (scheduler_gm_query(run_control->bootstrap_scan_complete) == 0) {
            if (scheduler_gm_query(run_control->scheduler_error) != 0) return false;
            if (scheduler_watchdog_expired(
                    barrier_start, scheduler_cycles(), scheduler_gm_query(run_control->scheduler_timeout_cycles)
                )) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::TIMEOUT, &graph, scheduler,
                    SchedulerErrorSite::BOOTSTRAP_SCAN_TIMEOUT
                );
                return false;
            }
            local_backoff(barrier_backoff);
            if (barrier_backoff < kMaximumBackoffIterations) barrier_backoff <<= 1;
        }
    }
    scheduler->owner_ready_queue_mask = static_cast<uint32_t>(ready_types);
    if (phase_timing_enabled) stats->target_bootstrap_start_cycles = scheduler_cycles();

    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
        uint64_t target_start = phase_timing_enabled ? scheduler_cycles() : 0;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        uint32_t type = scheduler_core_type_index(target->core_type);
        for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
            scheduler_initialize_free_slot(&scheduler->slots[cluster_lane][slot_index]);
        }
        scheduler_gm_publish(target->bootstrap_done, UINT64_C(1));
        if (phase_timing_enabled) stats->bootstrap_target_cycles[type] += scheduler_cycles() - target_start;
    }
    if (phase_timing_enabled) stats->target_bootstrap_end_cycles = scheduler_cycles();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed.
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{
        (scheduler->config.scheduler_index + 1) % scheduler_count,
        (scheduler->config.scheduler_index + 1) % scheduler_count,
    };
    bool fill_failed = false;
    (void)scheduler_fill_cluster_normal_slots(
        graph, scheduler_state_base, scheduler, run_control, ready_victim_cursors,
        phase_timing_enabled ? &stats->ready : nullptr, profiling_level, 0, deferred_aiv, &fill_failed, ssbuf_region
    );
    if (fill_failed) return false;
    // No peer can make progress while the launch gate is closed. Materialize
    // every Scheduler reservation now so local deferred state and FILLING slots
    // never cross the bootstrap boundary.
    while (deferred_aiv != nullptr && deferred_aiv->count != 0) {
        if (!scheduler_publish_deferred_aiv_local(
                graph, scheduler_state_base, scheduler, run_control, deferred_aiv, profiling_level, nullptr,
                ssbuf_region
            ))
            return false;
    }

    // This completion publication is observed by AICPU before it emits the
    // one and only DMB release. Schedulers do not wait on another barrier.
    arrived = scheduler_gm_fetch_add(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) scheduler_gm_publish(run_control->bootstrap_complete, UINT64_C(1));
    return true;
}

// Keep all arguments register-passed. The local state owns the immutable
// configuration; only the SSBUF base is derived inside the resident loop.
__aicore__ bool run_ready_dispatch_loop(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, SchedulerWorkerStats *stats, uint64_t profiling_level,
    SchedulerDeferredAivQueue *deferred_aiv
) {
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = scheduler_ssbuf_region(platform_ssbuf_hardware_base_address());
    const uint32_t cluster_lane = context->config.self_lane;
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t scheduler_count = context->config.scheduler_count;
    bool scheduler_worker = context->is_scheduler();
    if (scheduler_count == 0) return false;
    if (ssbuf_region == nullptr || cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{
        scheduler_worker ? (context->config.scheduler_index + 1) % scheduler_count : 0,
        scheduler_worker ? (context->config.scheduler_index + 1) % scheduler_count : 0,
    };
    uint32_t seen_generation[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint64_t completion_publication = 0;
    uint32_t scan_start = 0;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    if (phase_timing_enabled) context->profiling->loop_iter = UINT32_MAX;
    uint64_t idle_start_cycles = 0;
    bool idle_active = false;
    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (phase_timing_enabled) {
                stats->exit_observed_cycles = get_sys_cnt_aicore();
                if (scheduler_worker && idle_active) {
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

        if (phase_timing_enabled && scheduler_worker) ++context->profiling->loop_iter;
        const uint64_t idle_candidate_start = phase_timing_enabled && scheduler_worker ? scheduler_cycles() : 0;

        bool scheduler_progress = false;
        if (scheduler_worker && !scheduler_refresh_ready_inbox(scheduler_state_base, context)) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::READY_INBOX_REFRESH_FAILED
            );
            return false;
        }
        uint32_t published_local_slot = UINT32_MAX;
        if (scheduler_worker && deferred_aiv != nullptr && deferred_aiv->count != 0) {
            const uint32_t deferred_before = deferred_aiv->count;
            if (!scheduler_drain_deferred_aiv_to_peer(
                    graph, scheduler_state_base, context, run_control, deferred_aiv,
                    phase_timing_enabled ? &stats->wake : nullptr, phase_timing_enabled ? &stats->ready : nullptr,
                    phase_timing_enabled ? &stats->completion : nullptr, profiling_level, ssbuf_region
                ))
                return false;
            scheduler_progress = deferred_aiv->count != deferred_before;
            if (deferred_aiv->count != 0 && !scheduler_publish_deferred_aiv_local(
                                                graph, scheduler_state_base, context, run_control, deferred_aiv,
                                                profiling_level, &published_local_slot, ssbuf_region
                                            ))
                return false;
        }
        if (scheduler_worker && published_local_slot == UINT32_MAX) {
            uint64_t direct_refilled_slot_mask = 0;
            const bool completion_progress = scheduler_service_cluster_completions(
                graph, scheduler_state_base, context, run_control, phase_timing_enabled ? &stats->wake : nullptr,
                phase_timing_enabled ? &stats->ready : nullptr, phase_timing_enabled ? &stats->completion : nullptr,
                ready_victim_cursors, profiling_level, &direct_refilled_slot_mask, ssbuf_region
            );
            scheduler_progress = completion_progress;
            bool fill_failed = false;
            const bool dispatch_progress = scheduler_fill_cluster_normal_slots(
                graph, scheduler_state_base, context, run_control, ready_victim_cursors,
                phase_timing_enabled ? &stats->ready : nullptr, profiling_level, direct_refilled_slot_mask,
                deferred_aiv, &fill_failed, ssbuf_region
            );
            scheduler_progress = dispatch_progress || scheduler_progress;
            if (fill_failed) return false;
            if (deferred_aiv != nullptr && deferred_aiv->count != 0) {
                const uint32_t deferred_before = deferred_aiv->count;
                if (!scheduler_drain_deferred_aiv_to_peer(
                        graph, scheduler_state_base, context, run_control, deferred_aiv,
                        phase_timing_enabled ? &stats->wake : nullptr, phase_timing_enabled ? &stats->ready : nullptr,
                        phase_timing_enabled ? &stats->completion : nullptr, profiling_level, ssbuf_region
                    ))
                    return false;
                scheduler_progress = scheduler_progress || deferred_aiv->count != deferred_before;
                if (deferred_aiv->count != 0 && !scheduler_publish_deferred_aiv_local(
                                                    graph, scheduler_state_base, context, run_control, deferred_aiv,
                                                    profiling_level, &published_local_slot, ssbuf_region
                                                ))
                    return false;
            }
        }

        if (phase_timing_enabled && scheduler_worker && scheduler_progress && idle_active) {
            scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, idle_candidate_start);
            idle_active = false;
        }

        int32_t ready_slot = -1;
        uint32_t ready_generation = 0;
        uint64_t ready_publication = 0;
        uint64_t ready_scan_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (scheduler_worker) {
            uint32_t local_slot = UINT32_MAX;
            uint64_t local_generation = 0;
            if (scheduler_local_ready_pop(context, scan_start, &local_slot, &local_generation)) {
                if (scheduler_dispatch_state(local_generation) != SchedulerDispatchSlotState::READY ||
                    scheduler_dispatch_generation(local_generation) == 0 ||
                    scheduler_dispatch_generation(local_generation) == seen_generation[local_slot]) {
                    scheduler_record_error(
                        run_control, context->slots[cluster_lane][local_slot].task_id,
                        SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                        SchedulerErrorSite::EXECUTOR_PREFERRED_SLOT_INVALID
                    );
                    return false;
                }
                ready_slot = static_cast<int32_t>(local_slot);
                ready_generation = scheduler_dispatch_generation(local_generation);
            }
        } else {
            for (uint32_t offset = 0; offset < SCHEDULER_PENDING_SLOT_COUNT; ++offset) {
                uint32_t slot_index = (scan_start + offset) % SCHEDULER_PENDING_SLOT_COUNT;
                const uint64_t publication =
                    scheduler_ssbuf_load_relaxed(&ssbuf_region->lanes[cluster_lane].dispatch[slot_index].publication);
                const uint32_t generation = static_cast<uint32_t>(publication);
                if (phase_timing_enabled) ++stats->task_state_poll_count;
                if (generation != 0 && generation != seen_generation[slot_index]) {
                    ready_slot = static_cast<int32_t>(slot_index);
                    ready_generation = generation;
                    ready_publication = publication;
                    break;
                }
            }
        }
        if (ready_slot >= 0) {
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            SchedulerLocalSlotState *local_slot = &context->slots[cluster_lane][slot_index];
            const SCHEDULER_SSBUF volatile SchedulerSsbufDispatchControl *ssbuf_control =
                &ssbuf_region->lanes[cluster_lane].dispatch[slot_index];
            const int32_t timing_slot = scheduler_worker ? context->timing_slot(cluster_lane, slot_index) :
                                                           static_cast<int32_t>(ready_publication >> 32);
            __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
                scheduler_state_base, context->dispatch_payload_offset(context->config.self_lane, slot_index)
            );
            const bool commit_scheduler_trace = task_timing_enabled;
            const bool commit_task_timing = timing_slot >= 0 && timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
            const bool commit_executor_trace = commit_scheduler_trace || commit_task_timing;
            uint64_t ready_observe = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (phase_timing_enabled && scheduler_worker && idle_active) {
                scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, ready_observe);
                idle_active = false;
            }
            scheduler_observe_dispatch_payload_control(payload);
            scheduler_observe_dispatch_payload_arguments(payload);
            scheduler_observe_dispatch_payload_barrier();
            const int64_t task_id = scheduler_worker ? local_slot->task_id : ssbuf_control->task_id;
            if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count || ready_generation == 0) {
                scheduler_record_error(
                    run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
                return false;
            }
            seen_generation[slot_index] = ready_generation;
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
            execute_task(payload);
            uint64_t kernel_end = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (commit_executor_trace) stage_task_trace_before_completion(&execution_trace, kernel_start, kernel_end);
            if (scheduler_worker) {
                if (commit_executor_trace) context->profiling->executor_traces[slot_index] = execution_trace;
            } else {
                if (commit_executor_trace) {
                    SCHEDULER_SSBUF volatile SchedulerExecutorTaskTrace *published_trace =
                        &ssbuf_region->lanes[cluster_lane].traces[slot_index].payload;
                    published_trace->kernel_start_cycles = execution_trace.kernel_start_cycles;
                    published_trace->kernel_end_cycles = execution_trace.kernel_end_cycles;
                    published_trace->ready_scan_start_cycles = execution_trace.ready_scan_start_cycles;
                    published_trace->ready_observe_cycles = execution_trace.ready_observe_cycles;
                    published_trace->completion_id = execution_trace.completion_id;
                    published_trace->completion_inbox_index = execution_trace.completion_inbox_index;
                }
            }
            uint64_t completion_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (scheduler_worker) {
                context->local_completed_generations[slot_index] = ready_generation;
            } else {
                completion_publication =
                    scheduler_ssbuf_update_completion(completion_publication, slot_index, ready_generation);
                scheduler_cache_barrier();
                scheduler_ssbuf_store_relaxed(
                    &ssbuf_region->lanes[cluster_lane].completion.publication, completion_publication
                );
            }
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

        if (scheduler_progress) {
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }
        // Deferred AIV reservations still have work to dispatch, so this pass is not idle.
        if (scheduler_worker && (deferred_aiv == nullptr || deferred_aiv->count == 0))
            scheduler_flush_completions(run_control, context);
        if (phase_timing_enabled) ++stats->idle_iteration_count;
        uint64_t backoff_start = get_sys_cnt_aicore();
        local_backoff(backoff_iterations);
        uint64_t backoff_end = get_sys_cnt_aicore();
        if (phase_timing_enabled) stats->backoff_cycles += backoff_end - backoff_start;
        if (phase_timing_enabled) {
            if (scheduler_worker && !idle_active) {
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
    SchedulerDeferredAivQueue deferred_aiv{};
    std::conditional_t<WithProfiling, SchedulerLocalProfilingState, SchedulerNoProfilingState> profile{};
    SchedulerLocalState scheduler_local_state{};
    if constexpr (WithProfiling) scheduler_local_state.profiling = &profile;
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = scheduler_ssbuf_region(platform_ssbuf_hardware_base_address());
    const bool config_valid =
        scheduler_initialize_local_config(scheduler_state_base, context, &graph, &scheduler_local_state);
    if (!config_valid)
        scheduler_record_error(
            run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::BOOTSTRAP_FAILED
        );
    const uint32_t cluster_lane = scheduler_local_state.config.self_lane;
    const uint32_t scheduler_lane = scheduler_local_state.config.scheduler_lane;
    uint64_t descriptor_cache_observed_cycles = 0;
    if (config_valid) {
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(graph.storage_address));
        descriptor_cache_observed_cycles = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (context->is_scheduler != 0) {
            scheduler_ssbuf_initialize(ssbuf_region, scheduler_lane);
        }
        if (context->is_scheduler != 0 &&
            !bootstrap_ready_graph(
                graph, scheduler_state_base, &scheduler_local_state, run_control,
                scheduler_local_state.config.scheduler_count, &stats, profiling_level, &deferred_aiv, ssbuf_region
            )) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
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
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::TIMEOUT, &graph, context,
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
            if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM || scheduler_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
                !scheduler_ssbuf_is_initialized(ssbuf_region, scheduler_lane)) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
            } else {
                (void)run_ready_dispatch_loop(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &stats, profiling_level,
                    &deferred_aiv
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
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::TIMEOUT, &graph, context,
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
    // `platform_deinit_aicore_regs` waits for this acknowledgement, and the run's
    // finalizer folds this core's `scheduler_error` once it is observed. Complete
    // the preceding GM writes first. Stays outside the profiling block above: the
    // tail is optional, this is not.
    OUT_OF_ORDER_STORE_BARRIER();
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
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
                pending_run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::TIMEOUT, nullptr, pending_context,
                SchedulerErrorSite::CONTEXT_READY_TIMEOUT
            );
            startup_signal = AICORE_EXIT_SIGNAL;
            break;
        }
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        // The AICPU reads this acknowledgement as the point this core's writes
        // have landed, and the timeout above publishes its error through
        // `pending_run_control`. Complete the preceding GM writes first.
        OUT_OF_ORDER_STORE_BARRIER();
        write_reg(RegId::COND, AICORE_EXITED_VALUE);
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
}
