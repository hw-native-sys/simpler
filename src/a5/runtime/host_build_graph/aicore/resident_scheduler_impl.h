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

#include "common/device_phase.h"

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
#include "kernel_args.h"
#include "runtime.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_watchdog.h"

#include "resident_scheduler.h"

typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

namespace {

constexpr uint32_t kInitialBackoffIterations = 8;
constexpr uint32_t kMaximumBackoffIterations = 256;
constexpr uint32_t kSchedulerErrorPollInterval = 64;

struct SchedulerWorkerStats {
    uint64_t bootstrap_task_count{0};
    uint64_t bootstrap_start_cycles{0};
    uint64_t target_bootstrap_end_cycles{0};
    uint64_t exit_observed_cycles{0};
    uint64_t deferred_idle_start{0};
    uint64_t deferred_idle_end{0};
    // Reuse the existing dispatch-loop argument for the kernel endpoints.
    AicoreWorkWindow work_window{};
};

// This object's address never reaches a kernel or scheduler helper. Keep the
// first-call flag and rolling end local instead of storing them on every call.
struct KernelWorkWindowCapture {
    AicoreWorkWindow *output;
    bool started;
    uint64_t last_end;

    inline __aicore__ __attribute__((always_inline)) explicit KernelWorkWindowCapture(AicoreWorkWindow *window) :
        output(window),
        started(window->start_cycle != 0),
        last_end(window->end_cycle) {}
    inline __aicore__ __attribute__((always_inline)) ~KernelWorkWindowCapture() { output->end_cycle = last_end; }
    KernelWorkWindowCapture(const KernelWorkWindowCapture &) = delete;
    KernelWorkWindowCapture &operator=(const KernelWorkWindowCapture &) = delete;
};

// The original idle timestamps survive the kernel in the existing worker state.
__aicore__ __attribute__((noinline)) void
publish_deferred_idle_activity(__gm__ void *base, SchedulerLocalState *local, SchedulerWorkerStats *stats) {
    SchedulerScalarScope scalar_idle(local, SchedPhaseKind::ScalarPublishIdle);
    if (stats->deferred_idle_end == 0) return;
    scheduler_append_idle_activity(base, local, stats->deferred_idle_start, stats->deferred_idle_end);
    stats->deferred_idle_end = 0;
}

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

__aicore__ __attribute__((always_inline)) void execute_task(uint64_t function_address, __gm__ uint64_t *args) {
    if (function_address == 0) return;
    UnifiedKernelFunc kernel = reinterpret_cast<UnifiedKernelFunc>(function_address);
    kernel(reinterpret_cast<__gm__ int64_t *>(args));
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
        SchedulerGraphResult argument_status;
        {
            SchedulerScalarScope scalar_arguments(SchedPhaseKind::ScalarGraphPayloadArguments, task_ref);
            argument_status = scheduler_parse_task_payload_arguments(*graph, task_ref, &arguments);
        }
        if (argument_status != SchedulerGraphResult::OK || arguments.tensor_count != task->tensor_count ||
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
    int64_t task_ref, ArgsDumpStage stage, __gm__ GlobalContext &global, uint64_t address
) {
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
    scheduler_store_signal(context->bootstrap_task_count, stats.bootstrap_task_count);
    scheduler_store_signal(context->bootstrap_start_cycles, stats.bootstrap_start_cycles);
    scheduler_store_signal(context->target_bootstrap_end_cycles, stats.target_bootstrap_end_cycles);
}

// Keep the pre-kernel timestamps out of the indirect kernel call's live set.
// Inlining this path makes the resident loop spill enough state to fault on A5.
__aicore__ __attribute__((noinline)) void
stage_task_trace_before_execution(SchedulerExecutorTaskTrace *trace, uint64_t ready_observe) {
    SchedulerScalarScope scalar_trace(SchedPhaseKind::ScalarTraceBeforeExecution);
    trace->ready_observe_cycles = ready_observe;
}

__aicore__ __attribute__((noinline)) void
stage_task_trace_before_completion(SchedulerExecutorTaskTrace *trace, uint64_t kernel_start, uint64_t kernel_end) {
    SchedulerScalarScope scalar_trace(SchedPhaseKind::ScalarTraceBeforeCompletion);
    trace->kernel_start_cycles = kernel_start;
    trace->kernel_end_cycles = kernel_end;
}

template <bool WithNativeGraph, bool WithProfiling, bool ScheduleTiming = false>
__aicore__ bool bootstrap_ready_graph(
    const SchedulerGraphView &published_graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t worker_count, SchedulerWorkerStats *stats,
    uint64_t observed_profiling_level
) {
    const uint64_t profiling_level = ScheduleTiming ? SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL :
                                     WithProfiling  ? observed_profiling_level :
                                                      0;
    constexpr bool record_protocol = WithProfiling && kSchedulerScalarInstrumentation;
    // The caller has validated the published format and node count.
    const SchedulerGraphAccess graph =
        WithNativeGraph ? published_graph :
                          scheduler_ordinary_graph_view(
                              published_graph.storage_address, static_cast<uint32_t>(published_graph.task_count),
                              published_graph.task_window_last_index
                          );
    if (scheduler == nullptr || worker_count == 0 || worker_count != scheduler->config.runtime_worker_count ||
        worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0 || scheduler->worker_id() >= worker_count)
        return false;
    const bool phase_timing_enabled =
        kSchedulerScalarInstrumentation && scheduler_phase_timing_enabled(profiling_level);
    SchedulerScalarScope scalar_bootstrap(scheduler, SchedPhaseKind::ScalarBootstrap);
    const uint64_t timeout_cycles = scheduler_gm_query(run_control->scheduler_timeout_cycles);
    if (phase_timing_enabled) stats->bootstrap_start_cycles = scheduler_cycles();
    const auto *summary =
        scheduler_state_at<SchedulerGraphSummary>(scheduler_state_base, scheduler->config.graph_summary_offset);
    const uint64_t cluster = scheduler->config.shared_context->cluster_index;
    auto &cluster_state =
        scheduler_state_at<SchedulerBootstrapCluster>(scheduler_state_base, summary->bootstrap_clusters_offset)[cluster]
            .state;
    if (scheduler->cluster_signal != nullptr) {
        if (scheduler->is_mix_coordinator()) {
            scheduler_cluster_signal_clear(scheduler->cluster_signal, scheduler->config.prepared->run_epoch);
            scheduler_cluster_barrier();
            if (scheduler_gm_fetch_add(cluster_state, SCHEDULER_BOOTSTRAP_CLUSTER_INIT) != 0) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::BOOTSTRAP_CLUSTER_INVALID_STATE
                );
                return false;
            }
        } else {
            SchedulerScalarScope scalar_wait(scheduler, SchedPhaseKind::ScalarBootstrapInitWait);
            const uint64_t wait_start = scheduler_cycles();
            while (true) {
                if (scheduler_read_worker_error(run_control, scheduler->worker_id()) != 0 ||
                    static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL)
                    return false;
                const uint64_t hint = scheduler_gm_query(cluster_state);
                if (hint != 0) {
                    const uint64_t confirmed = scheduler_gm_fetch_add(cluster_state, UINT64_C(0));
                    if (confirmed > SCHEDULER_BOOTSTRAP_CLUSTER_DONE ||
                        (confirmed != 0 && (confirmed & SCHEDULER_BOOTSTRAP_CLUSTER_INIT) == 0)) {
                        scheduler_record_error(
                            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph,
                            scheduler, SchedulerErrorSite::BOOTSTRAP_CLUSTER_INVALID_STATE
                        );
                        return false;
                    }
                    if ((confirmed & SCHEDULER_BOOTSTRAP_CLUSTER_INIT) != 0) break;
                }
                if (scheduler_watchdog_expired(wait_start, scheduler_cycles(), timeout_cycles)) {
                    scheduler_record_error(
                        run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, scheduler,
                        SchedulerErrorSite::BOOTSTRAP_CLUSTER_INIT_TIMEOUT
                    );
                    return false;
                }
                SPIN_WAIT_HINT();
            }
            scheduler_cache_barrier();
        }
    }
    // INIT proves this cluster's old SSBUF state can no longer be observed.
    if (phase_timing_enabled) stats->target_bootstrap_end_cycles = scheduler_cycles();

    // Prepare the first executable wave while the sole DMB launch gate is
    // still closed.
    bool fill_failed = false;
    if (graph.native_header_offset != 0) {
        bool progress = false;
        if (!scheduler_progress_graph_controls(
                graph, scheduler_state_base, scheduler, run_control, profiling_level, &progress
            ))
            return false;
    }
    (void)scheduler_find_local_execution<record_protocol>(
        graph, scheduler_state_base, scheduler, run_control, profiling_level, &fill_failed
    );
    if (fill_failed) return false;

    // AIC and AIV1 set their own DONE bits after the first fill. AIV0 waits for both before counting the cluster.
    scheduler_cache_barrier();
    if (scheduler_read_worker_error(run_control, scheduler->worker_id()) != 0) return false;
    if (!scheduler->is_mix_coordinator()) {
        const uint64_t done =
            scheduler->config.self_lane == 0 ? SCHEDULER_BOOTSTRAP_AIC_DONE : SCHEDULER_BOOTSTRAP_AIV1_DONE;
        const uint64_t previous = scheduler_gm_fetch_add(cluster_state, done);
        if ((previous & SCHEDULER_BOOTSTRAP_CLUSTER_INIT) == 0 || (previous & done) != 0 ||
            previous > SCHEDULER_BOOTSTRAP_CLUSTER_DONE) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                SchedulerErrorSite::BOOTSTRAP_CLUSTER_INVALID_STATE
            );
            return false;
        }
        return true;
    }
    {
        SchedulerScalarScope scalar_wait(scheduler, SchedPhaseKind::ScalarBootstrapPeerWait);
        const uint64_t wait_start = scheduler_cycles();
        while (true) {
            if (scheduler_read_worker_error(run_control, scheduler->worker_id()) != 0 ||
                static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL)
                return false;
            const uint64_t hint = scheduler_gm_query(cluster_state);
            if (hint == SCHEDULER_BOOTSTRAP_CLUSTER_DONE || hint > SCHEDULER_BOOTSTRAP_CLUSTER_DONE ||
                (hint != 0 && (hint & SCHEDULER_BOOTSTRAP_CLUSTER_INIT) == 0)) {
                const uint64_t confirmed = scheduler_gm_fetch_add(cluster_state, UINT64_C(0));
                if (confirmed > SCHEDULER_BOOTSTRAP_CLUSTER_DONE ||
                    (confirmed != 0 && (confirmed & SCHEDULER_BOOTSTRAP_CLUSTER_INIT) == 0)) {
                    scheduler_record_error(
                        run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph,
                        scheduler, SchedulerErrorSite::BOOTSTRAP_CLUSTER_INVALID_STATE
                    );
                    return false;
                }
                if (confirmed == SCHEDULER_BOOTSTRAP_CLUSTER_DONE) break;
            }
            if (scheduler_watchdog_expired(wait_start, scheduler_cycles(), timeout_cycles)) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, &graph, scheduler,
                    SchedulerErrorSite::BOOTSTRAP_CLUSTER_PEER_TIMEOUT
                );
                return false;
            }
            SPIN_WAIT_HINT();
        }
    }
    scheduler_cache_barrier();
    if (scheduler_read_worker_error(run_control, scheduler->worker_id()) != 0) return false;
    const uint64_t previous = scheduler_gm_fetch_add(
        run_control->bootstrap_arrived_count, static_cast<uint64_t>(PLATFORM_CORES_PER_BLOCKDIM)
    );
    if (previous % PLATFORM_CORES_PER_BLOCKDIM != 0 || previous > worker_count - PLATFORM_CORES_PER_BLOCKDIM) {
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::BOOTSTRAP_CLUSTER_INVALID_STATE
        );
        return false;
    }
    if (previous + PLATFORM_CORES_PER_BLOCKDIM == worker_count)
        scheduler_gm_store(run_control->bootstrap_complete, UINT64_C(1));
    return true;
}

// The local state owns the immutable configuration; all arguments are register-passed.
template <bool WithNativeGraph, bool WithProfiling, bool ScheduleTiming = false>
__aicore__ bool run_ready_dispatch_loop(
    const SchedulerGraphView &published_graph, __gm__ void *scheduler_state_base,
    SchedulerLocalState *__restrict__ context, __gm__ SchedulerRunControl *run_control,
    SchedulerWorkerStats *__restrict__ stats, uint64_t observed_profiling_level
) {
    KernelWorkWindowCapture work_capture(&stats->work_window);
    const uint64_t profiling_level = ScheduleTiming ? SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL :
                                     WithProfiling  ? observed_profiling_level :
                                                      0;
    constexpr bool record_protocol = WithProfiling && kSchedulerScalarInstrumentation;
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
    const bool pmu_enabled = SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_PMU);
    const uint64_t pmu_reg_base = __builtin_expect(pmu_enabled, false) ? get_aicore_pmu_reg_base() : 0;
    const uint64_t dump_address = context->config.shared_context->args_dump_address;
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool phase_timing_enabled =
        kSchedulerScalarInstrumentation && scheduler_phase_timing_enabled(profiling_level);
    const uint64_t worker_count = context->config.runtime_worker_count;
    if (worker_count == 0) return false;
    if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    uint32_t scheduler_error_poll_count = 0;
    uint32_t execution_service_rounds = 0;
    if (phase_timing_enabled) context->profiling->loop_iter = UINT32_MAX;
    uint64_t idle_start_cycles = 0;
    bool idle_active = false;
    bool controls_serviced = false;
    auto service_controls = [&](bool &progress) __aicore__ {
        if (controls_serviced) return true;
        controls_serviced = true;
        SchedulerScalarScope scalar_control(context, SchedPhaseKind::ScalarGraphControl);
        bool advanced = false;
        if (!scheduler_progress_graph_controls(
                graph, scheduler_state_base, context, run_control, profiling_level, &advanced
            ))
            return false;
        scalar_control.close();
        progress = progress || advanced;
        return true;
    };
    enum class Boundary { Proceed, Exit, Error };
    auto check_exit_or_error = [&]() __aicore__ {
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
    Boundary boundary = check_exit_or_error();
    while (boundary == Boundary::Proceed) {
        controls_serviced = false;
        if (phase_timing_enabled) ++context->profiling->loop_iter;
        const uint64_t idle_candidate_start = phase_timing_enabled ? scheduler_cycles() : 0;
        SchedulerScalarIteration scalar_iteration(phase_timing_enabled ? &context->profiling->scalar : nullptr);

        bool scheduler_progress = false;
        if (context->async != nullptr && scheduler_async_service_due(*context->async)) {
            SchedulerScalarScope scalar_async(context, SchedPhaseKind::ScalarAsyncService);
            if (!scheduler_service_async_slice(
                    graph, scheduler_state_base, context, run_control, profiling_level, scheduler_progress
                ))
                return false;
        }
        uint32_t selected = UINT32_MAX;
        const bool service_due = ++execution_service_rounds >= 4;
        if (service_due) {
            execution_service_rounds = 0;
            if (!service_controls(scheduler_progress)) return false;
        }
        bool failed = false;
        const bool advanced = scheduler_find_local_execution<record_protocol>(
            graph, scheduler_state_base, context, run_control, profiling_level, &failed, &selected
        );
        if (failed) return false;
        scheduler_progress = scheduler_progress || advanced;
        if (selected == UINT32_MAX && !service_controls(scheduler_progress)) return false;
        if (phase_timing_enabled && scheduler_progress && idle_active) {
            stats->deferred_idle_start = idle_start_cycles;
            stats->deferred_idle_end = idle_candidate_start;
            idle_active = false;
        }

        if (selected != UINT32_MAX) {
            auto &candidate = context->execution_at(selected);
            if (candidate.state != SchedulerDispatchSlotState::READY ||
                scheduler_prepared_phase(candidate.prepared.claimed_state) != SchedulerPreparedPhase::CLAIMED ||
                scheduler_prepared_execute_owner(candidate.prepared.claimed_state) != context->worker_id()) {
                scheduler_record_error(
                    run_control, candidate.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_PREFERRED_SLOT_INVALID
                );
                return false;
            }
            bool permit = false;
            if (!scheduler_check_entry_permit<record_protocol>(
                    graph, scheduler_state_base, context, run_control, selected, &permit, true
                ))
                return false;
            if (!permit) {
                // Keep the owner in place and serve the obligation blocking entry.
                selected = UINT32_MAX;
                if (!service_controls(scheduler_progress)) return false;
                if (!scheduler_progress_mix_groups(
                        graph, scheduler_state_base, context, run_control, profiling_level, &scheduler_progress
                    ))
                    return false;
            }
        }
        if (selected != UINT32_MAX) {
            scalar_iteration.progress = true;
            const uint32_t slot_index = selected;
            auto *local_slot = &context->execution_at(slot_index);
            local_slot->state = SchedulerDispatchSlotState::EXECUTING;
            const int32_t timing_slot = context->timing_slot(slot_index);
            auto *call = scheduler_prepared_call_at(
                scheduler_state_base, context->config.prepared, local_slot->prepared.execution_index
            );
            auto *binding = scheduler_execution_binding_at(
                scheduler_state_base, context->config.prepared, local_slot->prepared.execution_index
            );
            const uint64_t function_address =
                local_slot->predicate == SchedulerPredicateState::FAIL ? 0 : call->function_bin_addr;
            const bool commit_scheduler_trace = task_timing_enabled;
            const bool commit_task_timing =
                WithProfiling && timing_slot >= 0 && timing_slot < SCHEDULER_TASK_TIMING_SLOT_COUNT;
            const bool commit_executor_trace = commit_scheduler_trace || commit_task_timing;
            uint64_t ready_observe =
                __builtin_expect(commit_executor_trace || phase_timing_enabled, false) ? get_sys_cnt_aicore() : 0;
            const int64_t task_ref = local_slot->task_ref;
            SchedulerScalarScope scalar_prepare(context, SchedPhaseKind::ScalarPrepare, task_ref, ready_observe);
            if (phase_timing_enabled && idle_active) {
                stats->deferred_idle_start = idle_start_cycles;
                stats->deferred_idle_end = ready_observe;
                idle_active = false;
            }
            SchedulerExecutorTaskTrace execution_trace{};
            if constexpr (ScheduleTiming) {
                context->profiling->executor_trace.ready_observe_cycles = ready_observe;
            } else if (__builtin_expect(commit_executor_trace, false)) {
                stage_task_trace_before_execution(&execution_trace, ready_observe);
            }
            if (__builtin_expect(commit_scheduler_trace, false))
                scheduler_capture_execution_identity(
                    scheduler_state_base, context, slot_index, function_address, binding->local_context
                );
            // Async-protected business inputs require observation on the actual
            // consumer after dependencies settle, independently of the call package.
            if (function_address != 0 && local_slot->may_observe_async_inputs && context->async != nullptr) {
                const uint8_t count = call->input_cache_spans;
                if (count != SCHEDULER_INPUT_CACHE_SKIP && scheduler_async_work_activated(context->async)) {
                    SchedulerScalarScope scalar_cache(context, SchedPhaseKind::ScalarCacheObserve, task_ref);
                    scheduler_observe_input_cache(call, count);
                }
            }
            uint32_t execution_group_index = pto2::resident::ASYNC_INVALID_INDEX;
            if (context->async != nullptr &&
                binding->global_context.async_group_index != pto2::resident::ASYNC_INVALID_INDEX) {
                pto2::resident::AsyncWriter writer{};
                if (!scheduler_async_begin_prebound_execution(
                        context->async, task_ref, binding->global_context, writer
                    )) {
                    scheduler_record_error(
                        run_control, task_ref, scheduler_async_error_result(context->async), &graph, context,
                        SchedulerErrorSite::ASYNC_CONTEXT_INVALID, local_slot
                    );
                    (void)scheduler_cancel_unstarted_slot<true>(scheduler_state_base, context, slot_index);
                    return false;
                }
                execution_group_index = writer.group_index;
            }
            if (__builtin_expect(dump_address != 0, false) &&
                !args_dump_boundary(
                    graph, context, run_control, task_ref, ArgsDumpStage::BEFORE_DISPATCH, binding->global_context,
                    dump_address
                )) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::ARGS_DUMP_REFUSED
                );
                (void)scheduler_cancel_unstarted_slot<true>(scheduler_state_base, context, slot_index);
                return false;
            }
            if (local_slot->cluster_mix && function_address != 0) {
                auto *cell =
                    WithProfiling && scheduler_protocol_enabled(context) ?
                        scheduler_prepared_control_at(
                            scheduler_state_base, context->config.prepared, local_slot->prepared.execution_index
                        ) :
                        nullptr;
                scheduler_mark_mix_kernel_entered(local_slot->prepared, cell);
            }
            // MIX entry acquired the SSBUF generation after fan-in. A synchronous
            // L0 member has only changed private state since that acquire.
            if (WithProfiling || !local_slot->cluster_mix || function_address == 0 || dump_address != 0 ||
                pmu_reg_base != 0 || execution_group_index != pto2::resident::ASYNC_INVALID_INDEX ||
                (context->async != nullptr && local_slot->may_observe_async_inputs))
                scheduler_store_barrier();
            scalar_prepare.close();
            const bool first_kernel = function_address != 0 && !work_capture.started;
            uint64_t kernel_start =
                commit_executor_trace || phase_timing_enabled || first_kernel ? get_sys_cnt_aicore() : 0;
            if (first_kernel) {
                stats->work_window.start_cycle = kernel_start;
                work_capture.started = true;
            }
            if (phase_timing_enabled) ++context->profiling->scalar.depth;
            if (__builtin_expect(pmu_reg_base != 0, false)) pmu_aicore_begin();
            if (record_protocol && scheduler_protocol_enabled(context))
                scheduler_execute_protocol_task(
                    scheduler_state_base, context, slot_index, function_address, call->args, binding->local_context
                );
            else execute_task(function_address, call->args);
            if (__builtin_expect(pmu_reg_base != 0, false)) {
                pmu_aicore_end();
                // PMU records use the logical subtask slot. MIX fixes each subtask to its
                // named cluster lane, while a pure vector block can run on either AIV.
                stage_task_pmu(
                    scheduler_state_base, context, task_ref, local_slot->subtask_slot, local_slot->block_idx,
                    pmu_reg_base
                );
            }
            if (phase_timing_enabled) --context->profiling->scalar.depth;
            uint64_t kernel_end =
                commit_executor_trace || phase_timing_enabled || function_address != 0 ? get_sys_cnt_aicore() : 0;
            if (function_address != 0) work_capture.last_end = kernel_end;
            SchedulerScalarScope scalar_seal(context, SchedPhaseKind::ScalarSeal, task_ref, kernel_end);
            if constexpr (ScheduleTiming) {
                context->profiling->executor_trace.kernel_start_cycles = kernel_start;
                context->profiling->executor_trace.kernel_end_cycles = kernel_end;
            } else if (__builtin_expect(commit_executor_trace, false)) {
                stage_task_trace_before_completion(&execution_trace, kernel_start, kernel_end);
                context->profiling->executor_trace = execution_trace;
            }
            // The kernel receives only call->args and its GM contexts. No
            // private slot/admission address escapes, and this core cannot run
            // its scheduler while inside the kernel. Dispatch's validated task
            // and group binding therefore remain owned through normal return.
            bool has_deferred = false;
            if (execution_group_index != pto2::resident::ASYNC_INVALID_INDEX) {
                const auto writer = scheduler_async_writer_for_group(*context->async, execution_group_index);
                if (!scheduler_async_seal_execution<true>(
                        context->async, task_ref, local_slot->block_idx, local_slot->subtask_slot, writer, &has_deferred
                    )) {
                    scheduler_record_error(
                        run_control, task_ref, scheduler_async_error_result(context->async), &graph, context,
                        SchedulerErrorSite::ASYNC_SEAL_FAILED, local_slot
                    );
                    (void)scheduler_complete_lane_execution<record_protocol, false>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    );
                    return false;
                }
            }
            if (__builtin_expect(dump_address != 0, false) &&
                !args_dump_boundary(
                    graph, context, run_control, task_ref, ArgsDumpStage::AFTER_COMPLETION, binding->global_context,
                    dump_address
                )) {
                scheduler_record_error(
                    run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::ARGS_DUMP_REFUSED
                );
                (void)scheduler_complete_lane_execution<record_protocol, false>(
                    graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                );
                return false;
            }
            const bool possible_async = execution_group_index != pto2::resident::ASYNC_INVALID_INDEX;
            local_slot->completion_credits = 1 + static_cast<uint8_t>(possible_async && !has_deferred);
            if (possible_async && !has_deferred && local_slot->cluster_mix) {
                const auto *counter = scheduler_execution_counter_at(
                    scheduler_state_base, context->config.completion_counters_offset, task_ref
                );
                if (scheduler_execution_uses_counter(counter) &&
                    !scheduler_counter_publish_condition(
                        scheduler_state_base, *counter, task_ref, local_slot->block_idx, local_slot->subtask_slot,
                        context->config.task_metadata_offset,
                        __builtin_expect(context->profiling == nullptr, true) ?
                            0 :
                            context->profiling->protocol_layout_offset,
                        static_cast<uint32_t>(context->worker_id()), false
                    )) {
                    scheduler_record_error(
                        run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                        SchedulerErrorSite::ASYNC_SEAL_FAILED, local_slot
                    );
                    (void)scheduler_complete_lane_execution<record_protocol, false>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    );
                    return false;
                }
            }
            scalar_seal.close();
            SchedulerScalarScope scalar_completions(context, SchedPhaseKind::ScalarCompletions, task_ref);
            const bool completion_ok =
                record_protocol && scheduler_protocol_enabled(context) ?
                    scheduler_complete_lane_execution<true>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    ) :
                    scheduler_complete_lane_execution<false>(
                        graph, scheduler_state_base, context, run_control, slot_index, profiling_level
                    );
            scalar_completions.close();
            if (!completion_ok) return false;
            context->previous_round_idle = false;
            if (phase_timing_enabled) publish_deferred_idle_activity(scheduler_state_base, context, stats);
            boundary = check_exit_or_error();
            if (boundary == Boundary::Proceed && context->is_mix_coordinator() &&
                (slot_index >= SCHEDULER_CLUSTER_MIX_EXECUTION || scheduler_next_mix_after_ordinary(*context)) &&
                !scheduler_continue_coordinator_head(
                    graph, scheduler_state_base, context, run_control, profiling_level
                ))
                return false;
            continue;
        }

        if (phase_timing_enabled) publish_deferred_idle_activity(scheduler_state_base, context, stats);
        context->previous_round_idle = !scheduler_progress;
        if (scheduler_progress) {
            scalar_iteration.progress = true;
            boundary = check_exit_or_error();
            continue;
        }
        if (phase_timing_enabled) {
            if (!idle_active) {
                idle_start_cycles = idle_candidate_start;
                idle_active = true;
            }
        }
        boundary = check_exit_or_error();
    }
    return boundary != Boundary::Error;
}

struct SchedulerNoProfilingState {};

template <bool WithProfiling, bool WithNativeGraph, bool ScheduleTiming = false>
__aicore__ __attribute__((noinline)) void run_resident_scheduler(
    __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t observed_profiling_level, uint64_t aicore_entry_cycles, AicoreWorkWindow *work_window
) {
    scheduler_invalidate_startup_cache();
#if defined(__CCE_AICORE__)
    // Each resident launch starts with ordinary stores, independent of work
    // previously dispatched on this physical core. Kernels select their own
    // atomic mode when their stores require it.
    set_atomic_none();
#endif
    // Each resident translation unit owns its block-local observer slot. A prior
    // aborted launch may not have unwound the binding before this invocation.
    static_assert(WithProfiling || !kSchedulerScalarInstrumentation);
    scheduler_scalar_bind(nullptr);
    const uint64_t profiling_level = ScheduleTiming ? SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL :
                                     WithProfiling  ? observed_profiling_level :
                                                      0;
    constexpr bool record_protocol = WithProfiling && kSchedulerScalarInstrumentation;
    const bool phase_timing_enabled =
        kSchedulerScalarInstrumentation && scheduler_phase_timing_enabled(profiling_level);
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
    };
    SchedulerWorkerStats stats{};
    std::conditional_t<WithProfiling, SchedulerLocalProfilingState, SchedulerNoProfilingState> profile{};
    SchedulerLocalState scheduler_local_state{};
    SchedulerScalarBinding scalar_binding;
    SchedulerAsyncLocalState async{};
    // The A5 toolchain configures a 32 KiB stack shared by this state, callees and spills.
    constexpr size_t resident_state_bytes = sizeof(graph) + sizeof(stats) + sizeof(profile) +
                                            sizeof(scheduler_local_state) + sizeof(scalar_binding) + sizeof(async);
    static_assert(resident_state_bytes < 0x8000, "resident scheduler state exhausts the A5 AICore stack");
    if constexpr (WithProfiling) {
        profile.capture_l2 = profiling_level == SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL;
        scheduler_local_state.profiling = &profile;
    }
    // READY publishes a new image even when its GM address was used by an older
    // run. Observe it before local-config validation reads the native header.
    const bool native_extent_valid =
        graph.native_header_offset == 0 ||
        scheduler_native_header_extent_valid(
            graph.scheduler_state_address, graph.scheduler_state_bytes, graph.native_header_offset
        );
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
                profile.scalar.capture_trace_control = profiling_level >= 4;
                profile.scalar.function_min_cycles =
                    profiling_level >= 4 ? 0 : PLATFORM_PROF_SYS_CNT_FREQ / UINT64_C(1000000);
                scalar_binding.attach(&profile.scalar);
            }
        }
    }
    if (config_valid) {
        config_valid = scheduler_initialize_order(scheduler_state_base, &scheduler_local_state);
    }
    if (config_valid) {
        auto *summary = scheduler_state_at<SchedulerGraphSummary>(
            scheduler_state_base, scheduler_local_state.config.graph_summary_offset
        );
        config_valid = scheduler_initialize_async(
            scheduler_state_base, context->scheduler_state_bytes, summary->async_layout_offset, summary->run_epoch,
            graph.task_count, scheduler_local_state.config.task_controls_offset, async
        );
        if (config_valid && async.layout != nullptr) {
            async.completion_counters_offset = context->completion_counters_offset;
            async.task_metadata_offset = context->task_metadata_offset;
            async.protocol_layout_offset = __builtin_expect(scheduler_local_state.profiling == nullptr, true) ?
                                               0 :
                                               scheduler_local_state.profiling->protocol_layout_offset;
            async.worker_index = static_cast<uint32_t>(context->worker_index);
            scheduler_local_state.async = &async;
        }
        if (config_valid && summary->mix_task_count != 0)
            scheduler_local_state.cluster_signal = scheduler_cluster_signal_at(platform_ssbuf_hardware_base_address());
    }
    if (!config_valid)
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::BOOTSTRAP_FAILED
        );
    const uint32_t cluster_lane = scheduler_local_state.config.self_lane;
    if (config_valid) {
        if (!bootstrap_ready_graph<WithNativeGraph, WithProfiling, ScheduleTiming>(
                graph, scheduler_state_base, &scheduler_local_state, run_control,
                scheduler_local_state.config.runtime_worker_count, &stats, profiling_level
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
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        if (config_valid) {
            if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::EXECUTOR_INVALID_DISPATCH_SLOT
                );
            } else {
                (void)run_ready_dispatch_loop<WithNativeGraph, WithProfiling, ScheduleTiming>(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &stats, profiling_level
                );
            }
        }
    }

    SchedulerScalarScope scalar_exit(&scheduler_local_state, SchedPhaseKind::ScalarExitDrain);
    if (phase_timing_enabled) publish_deferred_idle_activity(scheduler_state_base, &scheduler_local_state, &stats);

    const uint64_t exit_wait_start = get_sys_cnt_aicore();
    while (stats.exit_observed_cycles == 0 &&
           static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) != AICORE_EXIT_SIGNAL) {
        if (scheduler_watchdog_expired(exit_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            const int64_t task = scheduler_unretired_mix_task(scheduler_local_state);
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
    if (config_valid && scheduler_read_error(run_control) != 0) {
        if (!scheduler_cancel_unstarted_work(scheduler_state_base, &scheduler_local_state))
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph,
                &scheduler_local_state, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
            );
    }
    if (config_valid && scheduler_read_error(run_control) == 0 &&
        scheduler_has_execution_tokens(&scheduler_local_state)) {
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph,
            &scheduler_local_state, SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
        );
    }
    if (config_valid && scheduler_read_error(run_control) == 0) {
        const uint64_t drain_start = get_sys_cnt_aicore();
        while (true) {
            bool drained = false;
            if (!scheduler_drain_ordered_exit(
                    graph, scheduler_state_base, &scheduler_local_state, run_control, &drained
                ) ||
                drained)
                break;
            if (scheduler_watchdog_expired(drain_start, get_sys_cnt_aicore(), timeout_cycles)) {
                const int64_t task = scheduler_ordered_drain_task(&scheduler_local_state);
                scheduler_record_error(
                    run_control, task, SchedulerGraphResult::TIMEOUT, &graph, &scheduler_local_state,
                    SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
                );
                break;
            }
            SPIN_WAIT_HINT();
        }
    }
    if constexpr (WithProfiling) {
        if (config_valid && (phase_timing_enabled || scheduler_protocol_enabled(&scheduler_local_state)))
            scheduler_snapshot_counter_completions(graph, scheduler_state_base, &scheduler_local_state, run_control);
    }
    if (phase_timing_enabled) publish_worker_stats(context, stats);
    if constexpr (record_protocol) {
        if (profile.protocol_layout_offset != 0)
            scheduler_protocol_finish(
                scheduler_state_base, &scheduler_local_state, config_valid && scheduler_read_error(run_control) == 0
            );
    }
    scalar_exit.close();
    if constexpr (WithProfiling) {
        if (phase_timing_enabled) profile.scalar.finish(get_sys_cnt_aicore());
    }
    *work_window = stats.work_window;
    // Complete preceding GM writes before ACK. Stays outside profiling: the
    // tail is optional, this ordering is not.
    OUT_OF_ORDER_STORE_BARRIER();
}

}  // namespace

namespace simpler::hbg::resident {

template <bool WithProfiling, bool WithNativeGraph, CoreType Architecture>
__aicore__ void
run(__gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control, __gm__ void *scheduler_state_base,
    uint64_t profiling_level, uint64_t aicore_entry_cycles, AicoreWorkWindow *work_window) {
    run_resident_scheduler<WithProfiling, WithNativeGraph>(
        context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, work_window
    );
}

}  // namespace simpler::hbg::resident
