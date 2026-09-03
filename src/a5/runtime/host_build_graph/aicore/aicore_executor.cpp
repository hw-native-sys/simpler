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
// Cluster-local dependency scheduling uses one device-side protocol.
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
constexpr uint32_t kMaximumBackoffIterations = 128;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

static_assert(SCHEDULER_CALLABLE_CAPACITY == RUNTIME_MAX_FUNC_ID, "AICore ready scheduler layout mismatch");

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
    uint64_t bootstrap_end_cycles{0};
    uint64_t target_bootstrap_start_cycles{0};
    uint64_t target_bootstrap_end_cycles{0};
    uint64_t bootstrap_target_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t bootstrap_ready_claim_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t bootstrap_slot_fill_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t drain_start_cycles{0};
    uint64_t drain_end_cycles{0};
    uint64_t exit_wait_start_cycles{0};
    uint64_t exit_observed_cycles{0};
    uint64_t final_stats_publish_start_cycles{0};
    uint64_t final_stats_publish_end_cycles{0};
    uint64_t exit_ack_publish_cycles{0};
};

struct SchedulerInterTaskTiming {
    uint64_t completion_service_cycles{0};
    uint64_t dispatch_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerCompletionServiceTiming completion{};
    uint64_t gang_service_cycles{0};
    SchedulerNormalDispatchTiming dispatch{};
    uint64_t ready_poll_cycles{0};
    uint64_t backoff_cycles{0};

    __aicore__ void reset() { *this = {}; }
};

__aicore__ __attribute__((always_inline)) void publish_scheduler_tail_trace(
    __gm__ SchedulerWorkerContext *context, uint64_t start_cycles, uint64_t end_cycles,
    const SchedulerInterTaskTiming &timing
) {
    __gm__ SchedulerTailTrace *trace = &context->scheduler_tail_trace;
    trace->start_cycles = start_cycles;
    trace->end_cycles = end_cycles;
    trace->completion_scan_cycles = timing.completion.scan_cycles;
    trace->completion_consume_cycles = timing.completion.consume_cycles;
    trace->completion_resolve_cycles = timing.completion.resolve_cycles;
    trace->completion_ready_publish_cycles = timing.completion.ready_publish_cycles;
    trace->completion_refill_cycles = timing.completion.refill_cycles;
    trace->completion_finalize_cycles = timing.completion.finalize_cycles;
    trace->gang_service_cycles = timing.gang_service_cycles;
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        trace->dispatch_probe_cycles[type] = timing.dispatch.probe_cycles[type];
        trace->dispatch_claim_cycles[type] = timing.dispatch.claim_cycles[type];
        trace->dispatch_prepare_cycles[type] = timing.dispatch.prepare_cycles[type];
        trace->dispatch_materialize_cycles[type] = timing.dispatch.materialize_cycles[type];
        trace->dispatch_publish_cycles[type] = timing.dispatch.publish_cycles[type];
    }
    trace->ready_poll_cycles = timing.ready_poll_cycles;
    trace->backoff_cycles = timing.backoff_cycles;
    scheduler_publish_cache_line(&trace->start_cycles);
    scheduler_publish_cache_line(&trace->dispatch_materialize_cycles[0]);
    scheduler_gm_publish(trace->valid, UINT64_C(1));
}

struct SchedulerExecutionRecord {
    int64_t task_id{SCHEDULER_TASK_ID_INVALID};
    uint64_t claim_worker_id{0};
    uint64_t claim_start_cycles{0};
    uint64_t claim_end_cycles{0};
    SchedulerReadySource ready_source{SchedulerReadySource::LOCAL};
};

__aicore__ __attribute__((always_inline)) void execute_task(__gm__ DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

__aicore__ __attribute__((always_inline)) bool
should_commit_scheduler_trace(__gm__ void *, __gm__ SchedulerWorkerContext *, __gm__ SchedulerDispatchSlot *slot) {
    return slot->gang == 0;
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
    context->ready_to_kernel_cycles = stats.completion.ready_to_kernel_cycles;
    context->ready_to_kernel_max_cycles = stats.completion.ready_to_kernel_max_cycles;
    context->payload_cycles = stats.payload_cycles;
    context->kernel_cycles = stats.kernel_cycles;
    scheduler_publish_cache_line(&context->wake_cas_retry_count);

    context->completion_enqueue_cycles = stats.completion_enqueue_cycles;
    context->bootstrap_start_cycles = stats.bootstrap_start_cycles;
    context->bootstrap_end_cycles = stats.bootstrap_end_cycles;
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
    scheduler_gm_publish(context->bootstrap_ready_claim_aic_cycles, stats.bootstrap_ready_claim_cycles[0]);
    scheduler_gm_publish(context->bootstrap_ready_claim_aiv_cycles, stats.bootstrap_ready_claim_cycles[1]);
}

// Keep the pre-kernel timestamps out of the indirect kernel call's live set.
// Inlining this path makes the resident loop spill enough state to fault on A5.
__aicore__ __attribute__((noinline)) void stage_task_trace_before_execution(
    __gm__ SchedulerDispatchSlot *slot, uint64_t ready_scan_start, uint64_t ready_observe, uint64_t completion_id,
    uint64_t completion_inbox_index, bool phase_timing_enabled
) {
    __gm__ SchedulerExecutorTaskTrace *trace = &slot->executor_trace;
    scheduler_gm_store(trace->ready_observe_cycles, ready_observe);
    if (phase_timing_enabled) {
        scheduler_gm_store(trace->ready_scan_start_cycles, ready_scan_start);
        scheduler_gm_store(trace->completion_id, completion_id);
        scheduler_gm_store(trace->completion_inbox_index, completion_inbox_index);
    }
}

// Publish the staging generation only after every field is device-visible. The
// completion token follows this call, so the Scheduler cannot observe a partial
// Executor trace.
__aicore__ __attribute__((noinline)) void stage_task_trace_before_completion(
    __gm__ SchedulerDispatchSlot *slot, uint64_t kernel_start, uint64_t kernel_end, uint64_t completion_ready,
    bool phase_timing_enabled
) {
    __gm__ SchedulerExecutorTaskTrace *trace = &slot->executor_trace;
    scheduler_gm_store(trace->kernel_start_cycles, kernel_start);
    scheduler_gm_store(trace->kernel_end_cycles, kernel_end);
    if (phase_timing_enabled) {
        scheduler_gm_store(trace->completion_end_cycles, completion_ready);
        scheduler_gm_store(trace->completion_bookkeeping_end_cycles, completion_ready);
    }
    scheduler_gm_publish(trace->generation, slot->generation);
}

__aicore__ bool bootstrap_ready_graph(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t scheduler_count, SchedulerWorkerStats *stats,
    uint64_t profiling_level, SchedulerDeferredAivQueue *deferred_aiv, __gm__ SchedulerReadyOwnerState *ready_owner
) {
    if (scheduler_count == 0 || scheduler->inbox_index >= scheduler_count || ready_owner == nullptr) return false;
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    if (phase_timing_enabled) stats->bootstrap_start_cycles = scheduler_cycles();
    SchedulerReadyBatch batches[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t tasks_per_scheduler = graph.task_count / scheduler_count;
    uint64_t remainder = graph.task_count % scheduler_count;
    uint64_t task_begin = scheduler->inbox_index * tasks_per_scheduler +
                          (scheduler->inbox_index < remainder ? scheduler->inbox_index : remainder);
    uint64_t task_end = task_begin + tasks_per_scheduler + (scheduler->inbox_index < remainder ? 1 : 0);
    for (uint64_t task_id = task_begin; task_id < task_end; ++task_id) {
        __gm__ SchedulerTaskMetadata *metadata =
            scheduler_task_metadata_at(scheduler_state_base, scheduler, static_cast<int64_t>(task_id));
        scheduler_observe_cache_line(metadata);
        if (!scheduler_task_is_executable(metadata->flags)) continue;
        const bool has_fanin = scheduler_task_has_fanin(metadata->flags);
        const uint64_t fanin_start_cycles = phase_timing_enabled && has_fanin ? scheduler_cycles() : 0;
        SchedulerRouteResult route =
            has_fanin ? scheduler_bootstrap_route_task(
                            graph, scheduler_state_base, scheduler, run_control, static_cast<int64_t>(task_id),
                            phase_timing_enabled ? &stats->wake : nullptr
                        ) :
                        SchedulerRouteResult::READY_TO_ENQUEUE;
        if (phase_timing_enabled && has_fanin) {
            __gm__ SchedulerTaskTrace *traces =
                scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->trace_cells_offset);
            __gm__ SchedulerTaskTrace *trace = &traces[task_id];
            trace->fanin_start_cycles = fanin_start_cycles;
            trace->fanin_end_cycles = scheduler_cycles();
            trace->fanin_scheduler_worker_id = scheduler->worker_index;
            trace->fanin_loop_iter = 0;
            scheduler_publish_cache_line(&trace->ready_transition_cycles);
        }
        if (route == SchedulerRouteResult::ERROR) return false;
        if (route == SchedulerRouteResult::READY_TO_ENQUEUE &&
            !scheduler_bootstrap_ready_batch_append(
                scheduler_state_base, scheduler, static_cast<int64_t>(task_id),
                &batches[scheduler_metadata_core_type_index(scheduler_metadata_single_subtask_slot(metadata
                                                                                                       ->active_mask))],
                phase_timing_enabled ? &stats->ready : nullptr, profiling_level
            )) {
            return false;
        }
        if (phase_timing_enabled) ++stats->bootstrap_task_count;
    }
    scheduler_cache_barrier();
    uint64_t ready_types = 0;
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        if (!scheduler_bootstrap_ready_batch_publish(
                scheduler_state_base, scheduler, type, scheduler->inbox_index, &batches[type],
                phase_timing_enabled ? &stats->ready : nullptr, &ready_types
            ))
            return false;
    }
    __gm__ SchedulerReadyDirectory *ready_directory = scheduler_ready_directory_at(scheduler_state_base, scheduler);
    scheduler_gm_store(ready_directory->bootstrap_ready_types[scheduler->inbox_index], ready_types);
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
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        scheduler_gm_store(
            ready_owner->queues[type].advertised, (ready_types & (UINT64_C(1) << type)) != 0 ? UINT64_C(1) : UINT64_C(0)
        );
    }
    if (phase_timing_enabled) stats->target_bootstrap_start_cycles = scheduler_cycles();

    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
        uint64_t target_start = phase_timing_enabled ? scheduler_cycles() : 0;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        scheduler_observe_cache_line(target);
        scheduler_observe_cache_line(&target->task_metadata_offset);
        if (target->active == 0) continue;
        uint32_t type = scheduler_core_type_index(target->core_type);
        for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
            __gm__ SchedulerDispatchSlot *slot =
                scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, slot_index);
            scheduler_observe_cache_line(slot);
            scheduler_initialize_free_slot(slot);
        }
        scheduler_gm_publish(target->bootstrap_done, UINT64_C(1));
        if (phase_timing_enabled) stats->bootstrap_target_cycles[type] += scheduler_cycles() - target_start;
    }
    if (phase_timing_enabled) stats->target_bootstrap_end_cycles = scheduler_cycles();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed.
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{
        (scheduler->inbox_index + 1) % scheduler_count,
        (scheduler->inbox_index + 1) % scheduler_count,
    };
    bool fill_failed = false;
    (void)scheduler_fill_cluster_normal_slots(
        graph, scheduler_state_base, scheduler, run_control, ready_victim_cursors,
        phase_timing_enabled ? &stats->ready : nullptr, profiling_level, 0, nullptr, deferred_aiv, ready_owner,
        &fill_failed
    );
    if (fill_failed) return false;
    // No peer can make progress while the launch gate is closed. Materialize
    // every Scheduler reservation now so local deferred state and FILLING slots
    // never cross the bootstrap boundary.
    while (deferred_aiv != nullptr && deferred_aiv->count != 0) {
        if (!scheduler_publish_deferred_aiv_local(
                graph, scheduler_state_base, scheduler, run_control, deferred_aiv, profiling_level, nullptr
            ))
            return false;
    }

    // This completion publication is observed by AICPU before it emits the
    // one and only DMB release. Schedulers do not wait on another barrier.
    arrived = scheduler_gm_fetch_add(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) scheduler_gm_publish(run_control->bootstrap_complete, UINT64_C(1));
    if (phase_timing_enabled) stats->bootstrap_end_cycles = scheduler_cycles();
    return true;
}

__aicore__ bool run_ready_dispatch_loop(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, SchedulerWorkerStats *stats, uint64_t profiling_level,
    SchedulerDeferredAivQueue *deferred_aiv
) {
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    uint64_t scheduler_count = scheduler_gm_query(run_control->scheduler_count);
    bool scheduler_worker = context->is_scheduler != 0;
    if (scheduler_count == 0) return false;
    __gm__ SchedulerReadyOwnerState *ready_owner = nullptr;
    if (scheduler_worker) {
        ready_owner = scheduler_ready_owner_state_at(scheduler_state_base, context);
    }
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{
        scheduler_worker ? (context->inbox_index + 1) % scheduler_count : 0,
        scheduler_worker ? (context->inbox_index + 1) % scheduler_count : 0,
    };
    uint64_t seen_publication[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint64_t inter_task_start_cycles = context->trace_register_release_cycles;
    uint32_t scan_start = 0;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    uint32_t loop_iter = 0;
    uint64_t idle_start_cycles = 0;
    bool idle_active = false;
    SchedulerInterTaskTiming inter_task_timing{};
    while (true) {
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (phase_timing_enabled) {
                stats->exit_observed_cycles = get_sys_cnt_aicore();
                if (scheduler_worker && idle_active) {
                    scheduler_append_idle_activity(
                        scheduler_state_base, context, idle_start_cycles, stats->exit_observed_cycles
                    );
                }
                publish_scheduler_tail_trace(
                    context, inter_task_start_cycles, stats->exit_observed_cycles, inter_task_timing
                );
            }
            break;
        }
        if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
            scheduler_error_poll_count = 0;
            if (scheduler_gm_query(run_control->scheduler_error) != 0) return false;
        }

        if (phase_timing_enabled && scheduler_worker) context->profiling_loop_iter = loop_iter++;
        const uint64_t idle_candidate_start = phase_timing_enabled && scheduler_worker ? scheduler_cycles() : 0;

        bool scheduler_progress = false;
        if (scheduler_worker && !scheduler_ready_owner_maintain(scheduler_state_base, context, ready_owner)) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::READY_OWNER_MAINTENANCE_FAILED
            );
            return false;
        }
        uint32_t preferred_ready_slot = UINT32_MAX;
        if (scheduler_worker && deferred_aiv != nullptr && deferred_aiv->count != 0) {
            const uint32_t deferred_before = deferred_aiv->count;
            if (!scheduler_drain_deferred_aiv_to_peer(
                    graph, scheduler_state_base, context, run_control, deferred_aiv,
                    phase_timing_enabled ? &stats->wake : nullptr, phase_timing_enabled ? &stats->ready : nullptr,
                    phase_timing_enabled ? &stats->completion : nullptr, profiling_level, nullptr, nullptr, ready_owner
                ))
                return false;
            scheduler_progress = deferred_aiv->count != deferred_before;
            if (deferred_aiv->count != 0 && !scheduler_publish_deferred_aiv_local(
                                                graph, scheduler_state_base, context, run_control, deferred_aiv,
                                                profiling_level, &preferred_ready_slot
                                            ))
                return false;
        }
        if (scheduler_worker && preferred_ready_slot == UINT32_MAX) {
            uint64_t operation_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            uint64_t direct_refilled_slot_mask = 0;
            SchedulerCompletionServiceTiming completion_timing{};
            const bool completion_progress = scheduler_service_cluster_completions(
                graph, scheduler_state_base, context, run_control, phase_timing_enabled ? &stats->wake : nullptr,
                phase_timing_enabled ? &stats->ready : nullptr, phase_timing_enabled ? &stats->completion : nullptr,
                ready_victim_cursors, profiling_level, &direct_refilled_slot_mask,
                phase_timing_enabled ? &completion_timing : nullptr, ready_owner
            );
            if (phase_timing_enabled) {
                uint64_t completion_total = get_sys_cnt_aicore() - operation_start;
                uint64_t completion_detail = completion_timing.consume_cycles + completion_timing.resolve_cycles +
                                             completion_timing.ready_publish_cycles + completion_timing.refill_cycles +
                                             completion_timing.finalize_cycles;
                completion_timing.scan_cycles +=
                    completion_total > completion_detail ? completion_total - completion_detail : 0;
                inter_task_timing.completion_service_cycles += completion_total;
                inter_task_timing.completion.scan_cycles += completion_timing.scan_cycles;
                inter_task_timing.completion.consume_cycles += completion_timing.consume_cycles;
                inter_task_timing.completion.resolve_cycles += completion_timing.resolve_cycles;
                inter_task_timing.completion.ready_publish_cycles += completion_timing.ready_publish_cycles;
                inter_task_timing.completion.refill_cycles += completion_timing.refill_cycles;
                inter_task_timing.completion.finalize_cycles += completion_timing.finalize_cycles;
            }
            scheduler_progress = completion_progress;
            operation_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            SchedulerNormalDispatchTiming dispatch_timing{};
            bool fill_failed = false;
            const bool dispatch_progress = scheduler_fill_cluster_normal_slots(
                graph, scheduler_state_base, context, run_control, ready_victim_cursors,
                phase_timing_enabled ? &stats->ready : nullptr, profiling_level, direct_refilled_slot_mask,
                phase_timing_enabled ? &dispatch_timing : nullptr, deferred_aiv, ready_owner, &fill_failed
            );
            scheduler_progress = dispatch_progress || scheduler_progress;
            if (fill_failed) return false;
            if (phase_timing_enabled) {
                inter_task_timing.dispatch_cycles[0] += get_sys_cnt_aicore() - operation_start;
                for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
                    inter_task_timing.dispatch.probe_cycles[type] += dispatch_timing.probe_cycles[type];
                    inter_task_timing.dispatch.claim_cycles[type] += dispatch_timing.claim_cycles[type];
                    inter_task_timing.dispatch.prepare_cycles[type] += dispatch_timing.prepare_cycles[type];
                    inter_task_timing.dispatch.materialize_cycles[type] += dispatch_timing.materialize_cycles[type];
                    inter_task_timing.dispatch.publish_cycles[type] += dispatch_timing.publish_cycles[type];
                }
            }
            if (deferred_aiv != nullptr && deferred_aiv->count != 0) {
                const uint32_t deferred_before = deferred_aiv->count;
                if (!scheduler_drain_deferred_aiv_to_peer(
                        graph, scheduler_state_base, context, run_control, deferred_aiv,
                        phase_timing_enabled ? &stats->wake : nullptr, phase_timing_enabled ? &stats->ready : nullptr,
                        phase_timing_enabled ? &stats->completion : nullptr, profiling_level, nullptr, nullptr,
                        ready_owner
                    ))
                    return false;
                scheduler_progress = scheduler_progress || deferred_aiv->count != deferred_before;
                if (deferred_aiv->count != 0 && !scheduler_publish_deferred_aiv_local(
                                                    graph, scheduler_state_base, context, run_control, deferred_aiv,
                                                    profiling_level, &preferred_ready_slot
                                                ))
                    return false;
            }
        }

        if (phase_timing_enabled && scheduler_worker && scheduler_progress && idle_active) {
            scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, idle_candidate_start);
            idle_active = false;
        }

        int32_t ready_slot = -1;
        uint64_t ready_publication = 0;
        uint64_t ready_scan_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (preferred_ready_slot != UINT32_MAX) {
            __gm__ SchedulerDispatchSlot *slot =
                scheduler_dispatch_slot_at(scheduler_state_base, context, context->worker_index, preferred_ready_slot);
            const uint64_t publication = scheduler_gm_query(slot->publication);
            if (phase_timing_enabled) ++stats->task_state_poll_count;
            if (publication == seen_publication[preferred_ready_slot] ||
                scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::READY) {
                scheduler_record_error(
                    run_control, slot->task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_PREFERRED_SLOT_INVALID
                );
                return false;
            }
            ready_slot = static_cast<int32_t>(preferred_ready_slot);
            ready_publication = publication;
        } else {
            for (uint32_t offset = 0; offset < SCHEDULER_PENDING_SLOT_COUNT; ++offset) {
                uint32_t slot_index = (scan_start + offset) % SCHEDULER_PENDING_SLOT_COUNT;
                __gm__ SchedulerDispatchSlot *slot =
                    scheduler_dispatch_slot_at(scheduler_state_base, context, context->worker_index, slot_index);
                uint64_t publication = scheduler_gm_query(slot->publication);
                if (phase_timing_enabled) ++stats->task_state_poll_count;
                if (publication != seen_publication[slot_index] &&
                    scheduler_dispatch_state(publication) == SchedulerDispatchSlotState::READY) {
                    ready_slot = static_cast<int32_t>(slot_index);
                    ready_publication = publication;
                    break;
                }
            }
        }
        uint64_t ready_poll_end = phase_timing_enabled ? get_sys_cnt_aicore() : 0;

        if (ready_slot >= 0) {
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            __gm__ SchedulerDispatchSlot *slot =
                scheduler_dispatch_slot_at(scheduler_state_base, context, context->worker_index, slot_index);
            __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
                scheduler_state_base,
                context->dispatch_payload_offset + static_cast<uint64_t>(slot_index) * sizeof(DispatchPayload)
            );
            uint64_t ready_observe = get_sys_cnt_aicore();
            if (phase_timing_enabled && scheduler_worker && idle_active) {
                scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, ready_observe);
                idle_active = false;
            }
            scheduler_invalidate_cache_line(slot);
            scheduler_observe_dispatch_payload_control(payload);
            scheduler_observe_dispatch_payload_arguments(payload);
            scheduler_observe_dispatch_payload_barrier();
            if (slot->pending_slot != slot_index || slot->task_id < 0 ||
                static_cast<uint64_t>(slot->task_id) >= graph.task_count ||
                slot->generation != scheduler_dispatch_generation(ready_publication)) {
                scheduler_record_error(
                    run_control, slot->task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
                return false;
            }
            seen_publication[slot_index] = ready_publication;
            SchedulerExecutionRecord record{
                slot->task_id,
                slot->claim_worker_id,
                slot->claim_start_cycles,
                slot->claim_end_cycles,
                static_cast<SchedulerReadySource>(slot->ready_source),
            };
            const bool commit_scheduler_trace =
                task_timing_enabled && should_commit_scheduler_trace(scheduler_state_base, context, slot);
            __gm__ SchedulerTaskMetadata *task_metadata =
                scheduler_task_metadata_at(scheduler_state_base, context, record.task_id);
            scheduler_observe_cache_line(task_metadata);
            const bool commit_task_timing = task_metadata->timing_slot >= 0 &&
                                            task_metadata->timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT &&
                                            should_commit_scheduler_trace(scheduler_state_base, context, slot);
            if (commit_scheduler_trace || commit_task_timing) {
                uint64_t local_completion_index = stats->completion.enqueue_count;
                stage_task_trace_before_execution(
                    slot, ready_scan_start, ready_observe, scheduler_completion_id(context, local_completion_index),
                    context->scheduler_index, phase_timing_enabled
                );
            }
            OUT_OF_ORDER_STORE_BARRIER();
            uint64_t kernel_start = get_sys_cnt_aicore();
            if (phase_timing_enabled) {
                __gm__ SchedulerTaskControl *control =
                    scheduler_task_control_at(scheduler_state_base, context, record.task_id);
                scheduler_observe_cache_line(&control->next_waiter);
                if (control->ready_publish_cycles != 0 && kernel_start >= control->ready_publish_cycles) {
                    uint64_t lag = kernel_start - control->ready_publish_cycles;
                    stats->completion.ready_to_kernel_cycles += lag;
                    if (lag > stats->completion.ready_to_kernel_max_cycles)
                        stats->completion.ready_to_kernel_max_cycles = lag;
                }
            }
            execute_task(payload);
            uint64_t kernel_end = get_sys_cnt_aicore();
            scheduler_publish_dispatch_payload(payload);
            __gm__ SchedulerCompletionInbox *completion_line =
                scheduler_completion_inbox_at(scheduler_state_base, context, context->worker_index);
            uint64_t completion_start = get_sys_cnt_aicore();
            if (commit_scheduler_trace || commit_task_timing) {
                uint64_t completion_ready = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
                stage_task_trace_before_completion(
                    slot, kernel_start, kernel_end, completion_ready, phase_timing_enabled
                );
            }
            scheduler_gm_store(completion_line->completed_generations[slot_index], slot->generation);
            if (phase_timing_enabled) ++stats->completion.enqueue_count;
            uint64_t completion_end = get_sys_cnt_aicore();
            if (phase_timing_enabled) {
                ++stats->executed_task_count;
                stats->payload_cycles += kernel_start - ready_observe;
                stats->kernel_cycles += kernel_end - kernel_start;
                stats->completion_enqueue_cycles += completion_end - completion_start;
            }
            scan_start = (slot_index + 1) % SCHEDULER_PENDING_SLOT_COUNT;
            backoff_iterations = kInitialBackoffIterations;
            if (commit_scheduler_trace) {
                inter_task_start_cycles = get_sys_cnt_aicore();
            } else if (phase_timing_enabled) {
                inter_task_start_cycles = get_sys_cnt_aicore();
            }
            inter_task_timing.reset();
            continue;
        }

        if (phase_timing_enabled) inter_task_timing.ready_poll_cycles += ready_poll_end - ready_scan_start;

        if (scheduler_progress) {
            backoff_iterations = kInitialBackoffIterations;
            continue;
        }
        if (phase_timing_enabled) ++stats->idle_iteration_count;
        uint64_t backoff_start = get_sys_cnt_aicore();
        local_backoff(backoff_iterations);
        uint64_t backoff_end = get_sys_cnt_aicore();
        if (phase_timing_enabled) stats->backoff_cycles += backoff_end - backoff_start;
        if (phase_timing_enabled) {
            inter_task_timing.backoff_cycles += backoff_end - backoff_start;
            if (scheduler_worker && !idle_active) {
                idle_start_cycles = idle_candidate_start;
                idle_active = true;
            }
        }
        if (backoff_iterations < kMaximumBackoffIterations) backoff_iterations <<= 1;
    }
    return true;
}

}  // namespace

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->workers[block_idx]);
    const uint32_t profiling_flag = get_aicore_profiling_flag();
    scheduler_observe_cache_line(handshake);
    if ((handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING &&
         handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_RESIDENT_READY)) {
        legacy_aicore_execute(runtime, block_idx, core_type);
        return;
    }
    const bool chip_swimlane_enabled = SIMPLER_GET_DFX_FLAG(profiling_flag, SIMPLER_DFX_FLAG_CHIP_SWIMLANE);
    uint64_t aicore_entry_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;
    handshake->physical_core_id = get_physical_core_id();
    handshake->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    handshake->aicore_done = block_idx + 1;
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    uint64_t handshake_publish_cycles = chip_swimlane_enabled ? get_sys_cnt_aicore() : 0;

    // AICPU publishes the fully configured context through GM. This lets
    // AICore perform all pre-kernel work without consuming a DMB launch.
    // Host publishes enough immutable scheduler-state addressing for the READY
    // watchdog. AICPU fills the topology-dependent context fields before READY,
    // so this pre-publish view must not be reused for execution.
    __gm__ SchedulerWorkerContext *pending_context = reinterpret_cast<__gm__ SchedulerWorkerContext *>(handshake->task);
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
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    SchedulerGraphView graph{
        context->graph_storage_address,
        context->graph_reserved_address,
        context->graph_task_count,
        context->task_window_last_index,
    };
    SchedulerWorkerStats stats{};
    SchedulerDeferredAivQueue deferred_aiv{};
    __gm__ SchedulerReadyOwnerState *ready_owner = nullptr;
    uint64_t descriptor_cache_observed_cycles = 0;
    if (context->active != 0) {
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(graph.storage_address));
        descriptor_cache_observed_cycles = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (context->is_scheduler != 0) {
            ready_owner = scheduler_ready_owner_state_at(scheduler_state_base, context);
            scheduler_ready_owner_init(ready_owner);
        }
        if (context->is_scheduler != 0 &&
            !bootstrap_ready_graph(
                graph, scheduler_state_base, context, run_control, scheduler_gm_query(run_control->scheduler_count),
                &stats, profiling_level, &deferred_aiv, ready_owner
            )) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::BOOTSTRAP_FAILED
            );
        }
    }

    // Sole AICore-wide DMB: after this point a prepared slot may execute.
    startup_signal = 0;
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
        if (context->active != 0) {
            (void)run_ready_dispatch_loop(
                graph, scheduler_state_base, context, run_control, &stats, profiling_level, &deferred_aiv
            );
        }
    }

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
    write_reg(RegId::COND, AICORE_EXITED_VALUE);
}
