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

#pragma once

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/pmu_collector_aicore.h"
// AICore dependency scheduling uses one device-side protocol.
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_async_finish.h"
#include "scheduler/scheduler_graph_control.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_args_dump_tensor.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_watchdog.h"

#include "resident_scheduler.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 256;
constexpr uint32_t kMaximumIdleBackoffIterations = 128;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

struct SchedulerWorkerStats {
    uint64_t bootstrap_task_count{0};
    uint64_t bootstrap_start_cycles{0};
    uint64_t target_bootstrap_end_cycles{0};
    uint64_t completion_count{0};
    uint64_t exit_observed_cycles{0};
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
    __gm__ void *scheduler_state_base, SchedulerLocalState *local, int64_t task_ref, uint8_t subtask,
    uint32_t block_idx, uint64_t pmu_reg_base
) {
    if (local->config.pmu_cells_offset == 0 || pmu_reg_base == 0) return;
    if (subtask >= SCHEDULER_TASK_SUBTASK_SLOTS) return;
    __gm__ SchedulerTaskPmuCell *cells =
        scheduler_state_at<SchedulerTaskPmuCell>(scheduler_state_base, local->config.pmu_cells_offset);
    __gm__ SchedulerTaskPmuCell *cell = &cells[scheduler_trace_index_at(
        scheduler_state_base, local, task_ref, static_cast<SubtaskSlot>(subtask), block_idx
    )];
    pmu_aicore_read_counters(pmu_reg_base, cell->counters, &cell->total_cycles);
    cell->worker_id = static_cast<uint32_t>(local->config.worker_ids[local->config.self_lane]);
    scheduler_store_barrier();
    cell->published = static_cast<uint32_t>(task_ref) + 1U;
    scheduler_publish_cache_line(cell);
}

__aicore__ __attribute__((always_inline)) void execute_task(__gm__ DispatchPayload *payload) {
    // Dispatch binds a valid per-core payload before making its slot READY.
    if (payload->function_bin_addr == 0) return;
    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    scheduler_store_barrier();
}

__aicore__ __attribute__((noinline)) bool publish_args_dump_request(
    __gm__ SchedulerArgsDumpRequest *request, uint64_t epoch, uint64_t task_id, ArgsDumpStage stage, uint32_t worker,
    uint64_t cursor, uint64_t byte_cursor
) {
    const uint64_t tick = request->tick;
    if (tick == UINT64_MAX) return false;
    request->run_epoch = epoch;
    request->task_id = task_id;
    request->stage = static_cast<uint64_t>(stage);
    request->publisher = scheduler_args_dump_publisher(worker);
    request->arg_index = cursor;
    request->byte_cursor = byte_cursor;
    OUT_OF_ORDER_STORE_BARRIER();
    request->tick = tick + 1;
    scheduler_publish_cache_line(request);
    return true;
}

struct ArgsDumpLineWriter {
    __aicore__ void operator()(uint64_t line) const {
        scheduler_writeback_cache_line(reinterpret_cast<__gm__ void *>(line));
    }
};

struct ArgsDumpOutputPublisher {
    const SchedulerGraphAccess *graph;
    int64_t task_ref;
    __gm__ SchedulerArgsDumpTask *task;

    __aicore__ bool operator()() const {
#if defined(__CCE_AICORE__)
        pipe_barrier(PIPE_ALL);
#endif
        SchedulerTaskPayloadArguments arguments{};
        if (scheduler_parse_task_payload_arguments(*graph, task_ref, &arguments) != SchedulerGraphResult::OK ||
            arguments.tensor_count != task->tensor_count ||
            (arguments.tensor_count != 0 && reinterpret_cast<uint64_t>(arguments.tensors) != task->tensors_address))
            return false;
        for (int32_t arg = 0; arg < arguments.tensor_count; ++arg) {
            if ((task->output_tensor_mask & (UINT64_C(1) << arg)) == 0) continue;
            const auto *tensor = reinterpret_cast<const __gm__ simpler::hbg::Tensor *>(arguments.tensors) + arg;
            if (!scheduler_args_dump_tensor_lines(tensor, ArgsDumpLineWriter{})) return false;
        }
        return true;
    }
};

__aicore__ __attribute__((noinline)) bool args_dump_boundary(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run_control,
    int64_t task_ref, ArgsDumpStage stage, __gm__ GlobalContext &global
) {
    const uint64_t address = local->config.shared_context->args_dump_address;
    if (address == 0) return true;
    auto *image = reinterpret_cast<__gm__ SchedulerArgsDumpHeader *>(address);
    scheduler_observe_cache_line(image);
    scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(image) + 64);
    if (task_ref < 0 || static_cast<uint64_t>(task_ref) >= image->task_count) return false;
    auto *map = reinterpret_cast<__gm__ uint32_t *>(address + image->task_indices_offset);
    scheduler_observe_cache_line(&map[task_ref]);
    auto *task = scheduler_args_dump_task(image, task_ref);
    if (task == nullptr) return true;
    scheduler_observe_cache_line(task);
    scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(task) + 64);
    scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(task) + 128);
    const uint32_t worker = static_cast<uint32_t>(local->worker_id());
    if (worker >= image->worker_count) return false;
    const bool before = stage == ArgsDumpStage::BEFORE_DISPATCH;
    if (before) scheduler_args_dump_limit_async_output(task, global);
    const ArgsDumpOutputPublisher publish_outputs{&graph, task_ref, task};
    const auto arrival = before ? scheduler_args_dump_enter(task) : scheduler_args_dump_return(task, publish_outputs);
    if (arrival == SchedulerArgsDumpArrival::Ready || (!before && arrival == SchedulerArgsDumpArrival::Waiting))
        return true;
    if (arrival == SchedulerArgsDumpArrival::Invalid) return false;
    const bool publisher = arrival == SchedulerArgsDumpArrival::Publisher;
    auto *request = reinterpret_cast<__gm__ SchedulerArgsDumpRequest *>(address + image->requests_offset) + worker;
    if (publisher) {
        scheduler_gm_store(
            before ? task->publisher_before : task->publisher_after, scheduler_args_dump_publisher(worker)
        );
        scheduler_cache_barrier();
        scheduler_observe_cache_line(request);
        if (!publish_args_dump_request(request, image->run_epoch, task_ref, stage, worker, 0, 0)) return false;
    }
    uint64_t progress_cycles = get_sys_cnt_aicore();
    uint64_t last_tick = 0;
    uint64_t last_cursor = 0;
    uint64_t last_bytes = 0;
    const uint64_t timeout_cycles = run_control->scheduler_timeout_cycles;
    while (true) {
        const uint64_t now = get_sys_cnt_aicore();
        if (scheduler_gm_query(run_control->scheduler_error) != 0 ||
            scheduler_watchdog_expired(progress_cycles, now, timeout_cycles)) {
            (void)scheduler_gm_compare_exchange(task->aborted, UINT64_C(0), UINT64_C(1));
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::TIMEOUT, &graph, local,
                SchedulerErrorSite::ARGS_DUMP_BOUNDARY_TIMEOUT
            );
        }
        if (before && scheduler_args_dump_can_execute(task)) return true;
        const bool aborted = scheduler_args_dump_load(task->aborted) != 0;
        const uint64_t publication = scheduler_args_dump_load(before ? task->publisher_before : task->publisher_after);
        if (publication != 0 && publication <= image->worker_count) {
            const uint32_t source_worker = static_cast<uint32_t>(publication) - 1;
            auto *ack = reinterpret_cast<__gm__ SchedulerArgsDumpAck *>(address + image->acknowledgements_offset) +
                        source_worker;
            scheduler_observe_cache_line(ack);
            SchedulerArgsDumpAck observed{};
            const bool committed = scheduler_args_dump_observe_ack(ack, &observed);
            const auto status = scheduler_args_dump_ack_status(observed);
            const bool same_phase =
                committed &&
                scheduler_args_dump_ack_matches_phase(observed, image->run_epoch, task->task_ref, stage, publication);
            if (same_phase && status == SchedulerArgsDumpStatus::Aborted) return false;
            if (!aborted && same_phase) {
                if (publisher && status == SchedulerArgsDumpStatus::Sealed &&
                    scheduler_args_dump_seal(task, stage, observed, image->run_epoch, request->tick))
                    return true;
                if (observed.tick != last_tick && status == SchedulerArgsDumpStatus::InProgress &&
                    (observed.arg_index > last_cursor ||
                     (observed.arg_index == last_cursor && observed.byte_cursor > last_bytes))) {
                    last_tick = observed.tick;
                    last_cursor = observed.arg_index;
                    last_bytes = observed.byte_cursor;
                    progress_cycles = now;
                }
                if (publisher && observed.tick == request->tick &&
                    (status == SchedulerArgsDumpStatus::InProgress || status == SchedulerArgsDumpStatus::WouldBlock) &&
                    !publish_args_dump_request(
                        request, image->run_epoch, task_ref, stage, worker, observed.arg_index, observed.byte_cursor
                    ))
                    return false;
            }
        }
        if (aborted && scheduler_watchdog_expired(progress_cycles, now, timeout_cycles)) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::TIMEOUT, &graph, local,
                SchedulerErrorSite::ARGS_DUMP_ABORT_TIMEOUT
            );
            return false;
        }
    }
}

__aicore__ __attribute__((always_inline)) void local_backoff(uint32_t iterations) {
    for (uint32_t spin = 0; spin < iterations; ++spin) {
        __asm__ volatile("NOP");
    }
}

__aicore__ __attribute__((always_inline)) void
publish_worker_stats(__gm__ SchedulerWorkerContext *context, const SchedulerWorkerStats &stats) {
    context->bootstrap_task_count = stats.bootstrap_task_count;
    context->bootstrap_start_cycles = stats.bootstrap_start_cycles;
    context->target_bootstrap_end_cycles = stats.target_bootstrap_end_cycles;
    scheduler_publish_cache_line(&context->bootstrap_task_count);
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

template <bool WithNativeGraph, bool WithProfiling>
__aicore__ bool bootstrap_ready_graph(
    const SchedulerGraphView &published_graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t scheduler_count, SchedulerWorkerStats *stats,
    uint64_t observed_profiling_level
) {
    const uint64_t profiling_level = WithProfiling ? observed_profiling_level : 0;
    // The caller has validated the published format and node count.
    const SchedulerGraphAccess graph =
        WithNativeGraph ? published_graph :
                          scheduler_ordinary_graph_view(
                              published_graph.storage_address, static_cast<uint32_t>(published_graph.task_count),
                              published_graph.task_window_last_index
                          );
    if (scheduler == nullptr || scheduler_count == 0 || scheduler->config.scheduler_index >= scheduler_count)
        return false;
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    SchedulerScalarScope scalar_bootstrap(scheduler, SchedPhaseKind::ScalarBootstrap);
    const uint64_t timeout_cycles = scheduler_gm_query(run_control->scheduler_timeout_cycles);
    if (phase_timing_enabled) stats->bootstrap_start_cycles = scheduler_cycles();
    // A prebuilt CSR image publishes the graph work before any worker starts.
    if (WithNativeGraph ||
        scheduler_gm_query(run_control->bootstrap_scan_complete) != SCHEDULER_BOOTSTRAP_HOST_SEEDED) {
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
            bool control_node = false;
            if (graph.native_header_offset != 0) {
                SchedulerNodeKind kind{};
                if (scheduler_node_kind(graph, static_cast<int64_t>(task_ref), &kind) != SchedulerGraphResult::OK)
                    return false;
                if (kind == SchedulerNodeKind::HOST_DONE) continue;
                control_node = scheduler_node_is_control(kind);
                if (kind == SchedulerNodeKind::CHILD_KERNEL &&
                    !scheduler_bootstrap_bind_arguments(graph, static_cast<int64_t>(task_ref), scheduler, run_control))
                    return false;
            }
            if (!control_node && !scheduler_task_is_executable(metadata->flags)) continue;
            if (!control_node && scheduler_task_requires_admission(metadata->flags) &&
                !scheduler_bootstrap_prepare_admission(
                    graph, scheduler_state_base, scheduler, run_control, scheduler->admission,
                    static_cast<int64_t>(task_ref)
                ))
                return false;
            const bool has_fanin = scheduler_task_has_fanin(metadata->flags);
            SchedulerRouteResult route =
                has_fanin ? scheduler_bootstrap_route_task(
                                graph, scheduler_state_base, scheduler, run_control, static_cast<int64_t>(task_ref)
                            ) :
                            SchedulerRouteResult::READY_TO_ENQUEUE;
            if (route == SchedulerRouteResult::ERROR) return false;
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                if (control_node) {
                    if (!scheduler_control_enqueue(
                            graph, scheduler_state_base, scheduler, run_control, static_cast<int64_t>(task_ref)
                        ))
                        return false;
                    continue;
                }
                const uint32_t core_type = scheduler_task_ready_queue(metadata->flags, scheduler_task_lanes(*metadata));
                if (!scheduler_ready_batch_append(
                        scheduler_state_base, scheduler, static_cast<int64_t>(task_ref), &batches[core_type]
                    ))
                    return false;
            }
            if (phase_timing_enabled) ++stats->bootstrap_task_count;
        }
        scheduler_cache_barrier();
        // The scan barrier orders metadata and child arguments before consumption.
        // A contended publication remains owned by this scheduler's persistent
        // publisher and is serviced after the scan and in the dispatch loop.
        for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
            if (!scheduler_ready_batch_push(scheduler_state_base, scheduler, type, &batches[type])) return false;
        }

        const uint64_t arrived = scheduler_gm_fetch_add(run_control->bootstrap_scan_arrived_count, UINT64_C(1)) + 1;
        if (arrived == scheduler_count) {
            scheduler_gm_store(run_control->bootstrap_scan_complete, SCHEDULER_BOOTSTRAP_DEVICE_SCANNED);
        } else {
            uint32_t barrier_backoff = kInitialBackoffIterations;
            const uint64_t barrier_start = scheduler_cycles();
            while (scheduler_gm_query(run_control->bootstrap_scan_complete) == 0) {
                if (scheduler_read_worker_error(run_control, scheduler->worker_id()) != 0) return false;
                if (scheduler_watchdog_expired(barrier_start, scheduler_cycles(), timeout_cycles)) {
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
    }
    for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
        scheduler_initialize_free_slot(&scheduler->slots[slot_index]);
    }
    if (phase_timing_enabled) stats->target_bootstrap_end_cycles = scheduler_cycles();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed.
    bool fill_failed = false;
    if (graph.native_header_offset != 0) {
        bool progress = false;
        if (!scheduler_service_controls(
                graph, scheduler_state_base, scheduler, run_control, profiling_level, &progress
            ))
            return false;
    }
    (void)scheduler_fill_own_slots<WithProfiling>(
        graph, scheduler_state_base, scheduler, run_control, profiling_level, &fill_failed
    );
    if (fill_failed) return false;

    // Nothing published above needs ordering against this announcement. A wake
    // list travels by atomics on both sides: a resolver detaches it with an
    // exchange and reads its links with an atomic load. A ready batch is found
    // by a poll that repeats, so a queue head this core cannot see yet is read
    // again on the next pass.
    const uint64_t arrived = scheduler_gm_fetch_add(run_control->bootstrap_arrived_count, UINT64_C(1)) + 1;
    if (arrived == scheduler_count) scheduler_gm_store(run_control->bootstrap_complete, UINT64_C(1));
    return true;
}

// The local state owns the immutable configuration; all arguments are register-passed.
template <bool WithNativeGraph, bool WithProfiling>
__aicore__ bool run_ready_dispatch_loop(
    const SchedulerGraphView &published_graph, __gm__ void *scheduler_state_base,
    SchedulerLocalState *__restrict__ context, __gm__ SchedulerRunControl *run_control,
    SchedulerWorkerStats *__restrict__ stats, uint64_t observed_profiling_level
) {
    const uint64_t profiling_level = WithProfiling ? observed_profiling_level : 0;
    // The caller has validated the published format and node count.
    const SchedulerGraphAccess graph =
        WithNativeGraph ? published_graph :
                          scheduler_ordinary_graph_view(
                              published_graph.storage_address, static_cast<uint32_t>(published_graph.task_count),
                              published_graph.task_window_last_index
                          );
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
    if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    uint32_t backoff_iterations = kInitialBackoffIterations;
    uint32_t scheduler_error_poll_count = 0;
    if (phase_timing_enabled) context->profiling->loop_iter = UINT32_MAX;
    uint64_t idle_start_cycles = 0;
    bool idle_active = false;
    bool async_serviced = false;
    auto service_async = [&](bool &progress) __aicore__ {
        if (context->async == nullptr || async_serviced) return true;
        async_serviced = true;
        if (!scheduler_async_service_due(*context->async)) return true;
        SchedulerScalarScope scalar_async(context, SchedPhaseKind::ScalarAsyncService);
        bool advanced = false;
        const bool ok =
            scheduler_service_async_slice(graph, scheduler_state_base, context, run_control, profiling_level, advanced);
        progress = progress || advanced;
        return ok;
    };
    enum class Boundary { Proceed, Exit, Error };
    auto check_boundary = [&]() __aicore__ {
        SchedulerScalarScope scalar_loop(context, SchedPhaseKind::ScalarLoopControl);
        if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL) {
            if (phase_timing_enabled) {
                stats->exit_observed_cycles = get_sys_cnt_aicore();
                if (idle_active) {
                    scheduler_append_idle_activity(
                        scheduler_state_base, context, idle_start_cycles, stats->exit_observed_cycles
                    );
                }
            }
            return Boundary::Exit;
        }
        if (++scheduler_error_poll_count == kSchedulerErrorPollInterval) {
            scheduler_error_poll_count = 0;
            if (scheduler_read_worker_error(run_control, context->worker_id()) != 0) return Boundary::Error;
        }

        return Boundary::Proceed;
    };
    Boundary boundary = check_boundary();
    while (boundary == Boundary::Proceed) {
        async_serviced = false;
        if (phase_timing_enabled) ++context->profiling->loop_iter;
        const uint64_t idle_candidate_start = phase_timing_enabled ? scheduler_cycles() : 0;
        SchedulerScalarIteration scalar_iteration(phase_timing_enabled ? &context->profiling->scalar : nullptr);

        bool scheduler_progress = false;
        {
            if (graph.native_header_offset != 0) {
                SchedulerScalarScope scalar_control(context, SchedPhaseKind::ScalarGraphControl);
                bool control_progress = false;
                if (!scheduler_service_controls(
                        graph, scheduler_state_base, context, run_control, profiling_level, &control_progress
                    ))
                    return false;
                scheduler_progress = scheduler_progress || control_progress;
                if (context->pending_completed != 0) {
                    const uint64_t now = scheduler_cycles();
                    if (context->pending_completed >= 32 ||
                        now - context->last_completion_flush_cycles >= PLATFORM_PROF_SYS_CNT_FREQ / 1000) {
                        scheduler_flush_completions(run_control, context);
                        context->last_completion_flush_cycles = now;
                    }
                }
            }
            bool fill_failed = false;
            const bool dispatch_progress = scheduler_fill_own_slots<WithProfiling>(
                graph, scheduler_state_base, context, run_control, profiling_level, &fill_failed
            );
            scheduler_progress = dispatch_progress || scheduler_progress;
            if (fill_failed) return false;
        }

        if (phase_timing_enabled && scheduler_progress && idle_active) {
            scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, idle_candidate_start);
            idle_active = false;
        }

        int32_t ready_slot = -1;
        uint64_t ready_scan_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        SchedulerScalarScope scalar_scan(context, SchedPhaseKind::ScalarExecutorScan, -1, ready_scan_start);
        {
            uint32_t local_slot = UINT32_MAX;
            if (context->local_ready_mask != 0) {
                if (!scheduler_local_ready_peek(context, &local_slot) ||
                    context->slots[local_slot].state != SchedulerDispatchSlotState::READY) {
                    scheduler_record_error(
                        run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph,
                        context, SchedulerErrorSite::EXECUTOR_PREFERRED_SLOT_INVALID
                    );
                    return false;
                }
                ready_slot = static_cast<int32_t>(local_slot);
                if (context->async != nullptr && context->async->business_returns >= 4) {
                    if (!service_async(scheduler_progress)) return false;
                }
                if (context->admission != nullptr) {
                    bool permit = false;
                    if (!scheduler_ordered_can_enter_slot<WithProfiling>(
                            graph, scheduler_state_base, context, run_control, local_slot, &permit
                        ))
                        return false;
                    if (!permit) ready_slot = -1;
                }
            }
        }
        scalar_scan.close();
        if (ready_slot >= 0) {
            scalar_iteration.progress = true;
            uint32_t slot_index = static_cast<uint32_t>(ready_slot);
            if (!scheduler_local_ready_execute(context, slot_index)) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
                return false;
            }
            SchedulerLocalSlotState *local_slot = &context->slots[slot_index];
            const int32_t timing_slot = context->timing_slot(slot_index);
            __gm__ DispatchPayload *payload = scheduler_state_at<DispatchPayload>(
                scheduler_state_base, context->dispatch_payload_offset(context->config.self_lane, slot_index)
            );
            const bool commit_scheduler_trace = task_timing_enabled;
            const bool commit_task_timing =
                WithProfiling && timing_slot >= 0 && timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
            const bool commit_executor_trace = commit_scheduler_trace || commit_task_timing;
            uint64_t ready_observe = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            // This core prepares every payload it executes, including Mix lanes.
            const int64_t task_ref = local_slot->task_ref;
            SchedulerScalarScope scalar_prepare(context, SchedPhaseKind::ScalarPrepare, task_ref, ready_observe);
            if (phase_timing_enabled && idle_active) {
                scheduler_append_idle_activity(scheduler_state_base, context, idle_start_cycles, ready_observe);
                idle_active = false;
            }
            SchedulerExecutorTaskTrace execution_trace{};
            if (commit_scheduler_trace || commit_task_timing) {
                uint64_t local_completion_index = stats->completion_count;
                stage_task_trace_before_execution(
                    &execution_trace, ready_scan_start, ready_observe,
                    scheduler_completion_id(context, local_completion_index), context->config.scheduler_index,
                    phase_timing_enabled
                );
            }
            if (commit_scheduler_trace)
                scheduler_capture_execution_identity(scheduler_state_base, context, slot_index, payload);
            // Dependencies and final admission are settled. Acquire payload on
            // the actual consumer core before writing its fresh async context;
            // another core's completion observation cannot invalidate this cache.
            if (payload->function_bin_addr != 0 && scheduler_async_work_activated(context->async)) {
                SchedulerScalarScope scalar_cache(context, SchedPhaseKind::ScalarCacheObserve, task_ref);
                scheduler_observe_data_cache(payload);
            }
            uint32_t execution_group_index = pto2::resident::ASYNC_INVALID_INDEX;
            {
                pto2::resident::AsyncWriter writer{};
                if (!scheduler_async_begin_execution(
                        context->async, task_ref, local_slot->block_idx, local_slot->subtask_slot,
                        payload->global_context, writer
                    )) {
                    scheduler_record_error(
                        run_control, task_ref, scheduler_async_error_result(context->async), &graph, context,
                        SchedulerErrorSite::ASYNC_CONTEXT_INVALID, local_slot
                    );
                    return false;
                }
                execution_group_index = writer.group_index;
            }
            if (context->config.shared_context->args_dump_address != 0 &&
                !args_dump_boundary(
                    graph, context, run_control, task_ref, ArgsDumpStage::BEFORE_DISPATCH, payload->global_context
                )) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::ARGS_DUMP_REFUSED
                );
                return false;
            }
            scheduler_store_barrier();
            scalar_prepare.close();
            uint64_t kernel_start = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            if (phase_timing_enabled) ++context->profiling->scalar.depth;
            if (pmu_reg_base != 0) pmu_aicore_begin();
            if (WithProfiling && scheduler_protocol_enabled(context))
                scheduler_execute_protocol_task(scheduler_state_base, context, slot_index, payload);
            else execute_task(payload);
            if (pmu_reg_base != 0) {
                pmu_aicore_end();
                // The subtask this lane ran, not the lane: a task's subtasks
                // are placed on whichever lanes of the cluster are free, so a
                // mix task's three records are told apart by this and nothing
                // else.
                stage_task_pmu(
                    scheduler_state_base, context, task_ref, local_slot->subtask_slot, local_slot->block_idx,
                    pmu_reg_base
                );
            }
            if (phase_timing_enabled) --context->profiling->scalar.depth;
            uint64_t kernel_end = commit_executor_trace || phase_timing_enabled ? get_sys_cnt_aicore() : 0;
            SchedulerScalarScope scalar_seal(context, SchedPhaseKind::ScalarSeal, task_ref, kernel_end);
            if (commit_executor_trace) stage_task_trace_before_completion(&execution_trace, kernel_start, kernel_end);
            if (commit_executor_trace) context->profiling->executor_traces[slot_index] = execution_trace;
            // The kernel receives only payload->args and its GM contexts. No
            // private slot/admission address escapes, and this core cannot run
            // its scheduler while inside the kernel. Dispatch's validated task
            // and group binding therefore remain owned through normal return.
            if (execution_group_index != pto2::resident::ASYNC_INVALID_INDEX) {
                const auto writer = scheduler_async_writer_for_group(*context->async, execution_group_index);
                if (!scheduler_async_seal_execution<true>(
                        context->async, task_ref, local_slot->block_idx, local_slot->subtask_slot, writer
                    )) {
                    scheduler_record_error(
                        run_control, task_ref, scheduler_async_error_result(context->async), &graph, context,
                        SchedulerErrorSite::ASYNC_SEAL_FAILED, local_slot
                    );
                    return false;
                }
            }
            if (context->config.shared_context->args_dump_address != 0 &&
                !args_dump_boundary(
                    graph, context, run_control, task_ref, ArgsDumpStage::AFTER_COMPLETION, payload->global_context
                )) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::ARGS_DUMP_REFUSED
                );
                return false;
            }
            scalar_seal.close();
            if (context->async != nullptr) ++context->async->business_returns;
            if (phase_timing_enabled) ++stats->completion_count;
            backoff_iterations = kInitialBackoffIterations;
            boundary = check_boundary();
            if (boundary != Boundary::Proceed) break;
            SchedulerScalarScope scalar_completions(context, SchedPhaseKind::ScalarCompletions, task_ref);
            const bool completion_ok =
                WithProfiling && scheduler_protocol_enabled(context) ?
                    scheduler_complete_own_slot<true, true>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    ) :
                    scheduler_complete_own_slot<false, true>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    );
            if (!completion_ok) return false;
            continue;
        }

        if (!service_async(scheduler_progress)) return false;
        if (scheduler_progress) {
            scalar_iteration.progress = true;
            backoff_iterations = kInitialBackoffIterations;
            boundary = check_boundary();
            continue;
        }
        {
            SchedulerScalarScope scalar_flush(context, SchedPhaseKind::ScalarFlushCompletions);
            scheduler_flush_completions(run_control, context);
        }
        if (scheduler_prepare_advance_hint(graph, scheduler_state_base, context)) {
            scheduler_prepare_advance_membership(scheduler_state_base, context, run_control);
            boundary = check_boundary();
            continue;
        }
        const uint64_t backoff_start = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        local_backoff(backoff_iterations);
        if (phase_timing_enabled)
            context->profiling->scalar.append(SchedPhaseKind::ScalarBackoff, backoff_start, get_sys_cnt_aicore());
        if (phase_timing_enabled) {
            if (!idle_active) {
                idle_start_cycles = idle_candidate_start;
                idle_active = true;
            }
        }
        if (backoff_iterations < kMaximumIdleBackoffIterations) backoff_iterations <<= 1;
        boundary = check_boundary();
    }
    return boundary != Boundary::Error;
}

struct SchedulerNoProfilingState {};

template <bool WithProfiling, bool WithNativeGraph>
__aicore__ __attribute__((noinline)) void run_resident_scheduler(
    __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t observed_profiling_level, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles
) {
    // Each resident translation unit owns its block-local observer slot. A prior
    // aborted launch may not have unwound the binding before this invocation.
    static_assert(!WithProfiling || kSchedulerScalarInstrumentation);
    scheduler_scalar_bind(nullptr);
    const uint64_t profiling_level = WithProfiling ? observed_profiling_level : 0;
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    // READY publishes this run's timeout; no writer changes it during execution.
    const uint64_t timeout_cycles = scheduler_gm_query(run_control->scheduler_timeout_cycles);
    scheduler_observe_cache_line(&context->graph_task_count);
    const SchedulerGraphView graph{
        context->graph_storage_address,
        context->graph_task_count,
        context->task_window_last_index,
        context->scheduler_state_base_address,
        WithNativeGraph ? context->native_header_offset : 0,
        context->scheduler_state_bytes,
        static_cast<uint32_t>(context->original_task_count),
        static_cast<uint16_t>(context->graph_view_version),
        static_cast<uint16_t>(context->graph_view_bytes),
    };
    SchedulerWorkerStats stats{};
    std::conditional_t<WithProfiling, SchedulerLocalProfilingState, SchedulerNoProfilingState> profile{};
    SchedulerLocalState scheduler_local_state{};
    SchedulerScalarBinding scalar_binding;
    SchedulerAdmissionLocalState admission{};
    SchedulerAsyncLocalState async{};
    if constexpr (WithProfiling) scheduler_local_state.profiling = &profile;
    // READY publishes a new image even when its GM address was used by an older
    // run. Observe it before local-config validation reads the native header.
    const bool native_extent_valid =
        graph.native_header_offset == 0 ||
        scheduler_native_header_extent_valid(
            graph.scheduler_state_address, graph.scheduler_state_bytes, graph.native_header_offset
        );
    if (graph.native_header_offset != 0 && native_extent_valid)
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(graph.scheduler_state_address));
    bool config_valid =
        native_extent_valid &&
        scheduler_initialize_local_config(scheduler_state_base, context, &graph, &scheduler_local_state);
    if constexpr (WithProfiling) {
        if (config_valid && phase_timing_enabled) {
            profile.scalar.buffer = scheduler_activity_buffer_at(scheduler_state_base, &scheduler_local_state);
            if (profile.scalar.buffer != nullptr) {
                profile.scalar.records = scheduler_state_at<SchedulerScalarStorageSlot>(
                    scheduler_state_base, scheduler_gm_query(profile.scalar.buffer->scalar_records_offset)
                );
                profile.scalar.window_start = aicore_entry_cycles;
                profile.scalar.function_min_cycles = profiling_level >= 4 ? 0 : PLATFORM_PROF_SYS_CNT_FREQ / 1000000;
                scalar_binding.attach(&profile.scalar);
            }
        }
    }
    if (config_valid) {
        config_valid = scheduler_initialize_admission(scheduler_state_base, &scheduler_local_state, &admission);
        if (config_valid && admission.capacity != 0) scheduler_local_state.admission = &admission;
    }
    if (config_valid) {
        auto *summary = scheduler_state_at<SchedulerGraphSummary>(
            scheduler_state_base, scheduler_local_state.config.graph_summary_offset
        );
        config_valid = scheduler_initialize_async(
            scheduler_state_base, context->scheduler_state_bytes, summary->async_layout_offset, summary->run_epoch,
            graph.task_count, async
        );
        if (config_valid && async.layout != nullptr) {
            async.check_cursor = static_cast<uint32_t>(context->worker_index % async.layout->group_count);
            async.finish_cursor = static_cast<uint32_t>(context->worker_index % async.layout->async_task_count);
            scheduler_local_state.async = &async;
        }
    }
    if (!config_valid)
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::BOOTSTRAP_FAILED
        );
    const uint32_t cluster_lane = scheduler_local_state.config.self_lane;
    uint64_t descriptor_cache_observed_cycles = 0;
    if (config_valid) {
        if (graph.native_header_offset == 0)
            scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(graph.storage_address));
        descriptor_cache_observed_cycles = phase_timing_enabled ? get_sys_cnt_aicore() : 0;
        if (context->is_scheduler != 0 && !bootstrap_ready_graph<WithNativeGraph, WithProfiling>(
                                              graph, scheduler_state_base, &scheduler_local_state, run_control,
                                              scheduler_local_state.config.scheduler_count, &stats, profiling_level
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
    SchedulerScalarScope scalar_launch_wait(&scheduler_local_state, SchedPhaseKind::ScalarLaunchWait);
    const uint64_t register_wait_start = get_sys_cnt_aicore();
    while (startup_signal != AICPU_IDLE_TASK_ID && startup_signal != AICORE_EXIT_SIGNAL) {
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (scheduler_watchdog_expired(register_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, context,
                SchedulerErrorSite::REGISTER_RELEASE_TIMEOUT
            );
            register_release_timed_out = true;
            break;
        }
        SPIN_WAIT_HINT();
    }
    scalar_launch_wait.close();
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
        }
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        if (config_valid) {
            if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
            } else {
                (void)run_ready_dispatch_loop<WithNativeGraph, WithProfiling>(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &stats, profiling_level
                );
            }
        }
    }

    SchedulerScalarScope scalar_exit(&scheduler_local_state, SchedPhaseKind::ScalarExitDrain);
    scheduler_flush_completions(run_control, &scheduler_local_state);

    const uint64_t exit_wait_start = get_sys_cnt_aicore();
    while (stats.exit_observed_cycles == 0 &&
           static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
        if (scheduler_watchdog_expired(exit_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            const int64_t task = scheduler_local_state.admission != nullptr ?
                                     scheduler_local_state.admission->current_task_ref :
                                     SCHEDULER_TASK_REF_INVALID;
            scheduler_record_error(
                run_control, task, SchedulerGraphResult::TIMEOUT, &graph, &scheduler_local_state,
                SchedulerErrorSite::EXIT_WAIT_TIMEOUT
            );
            break;
        }
        SPIN_WAIT_HINT();
    }
    if (phase_timing_enabled && stats.exit_observed_cycles == 0) stats.exit_observed_cycles = get_sys_cnt_aicore();
    // A late worker can observe EXIT without ever observing the short IDLE
    // interval. Both startup and dispatch-loop exits close their obligations
    // here, before the acknowledgement allows the AICPU to retire that core.
    wait_for_aicpu_exit_signal();
    if (config_valid && scheduler_read_error(run_control) == 0) {
        const uint64_t drain_start = get_sys_cnt_aicore();
        while (true) {
            bool drained = false;
            if (!scheduler_drain_ordered_exit<WithProfiling>(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &drained
                ) ||
                drained)
                break;
            if (scheduler_watchdog_expired(drain_start, get_sys_cnt_aicore(), timeout_cycles)) {
                const int64_t task = scheduler_local_state.admission != nullptr ?
                                         scheduler_local_state.admission->current_task_ref :
                                         SCHEDULER_TASK_REF_INVALID;
                scheduler_record_error(
                    run_control, task, SchedulerGraphResult::TIMEOUT, &graph, &scheduler_local_state,
                    SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
                );
                break;
            }
            SPIN_WAIT_HINT();
        }
    }
    if (phase_timing_enabled) publish_worker_stats(context, stats);
    if constexpr (WithProfiling) {
        if (profile.protocol_layout_offset != 0)
            scheduler_protocol_finish(
                scheduler_state_base, &scheduler_local_state, config_valid && scheduler_read_error(run_control) == 0
            );
    }
    scalar_exit.close();
    if constexpr (WithProfiling) {
        if (phase_timing_enabled) profile.scalar.finish(get_sys_cnt_aicore());
    }
    // Complete preceding GM writes before ACK. Stays outside profiling: the
    // tail is optional, this ordering is not.
    OUT_OF_ORDER_STORE_BARRIER();
}

}  // namespace

namespace simpler::hbg::resident {

template <bool WithProfiling, bool WithNativeGraph, CoreType Architecture>
__aicore__ void
run(__gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles, uint64_t handshake_publish_cycles) {
    run_resident_scheduler<WithProfiling, WithNativeGraph>(
        context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, handshake_publish_cycles
    );
}

}  // namespace simpler::hbg::resident
