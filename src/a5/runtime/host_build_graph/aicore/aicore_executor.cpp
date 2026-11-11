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

#include "scheduler/scheduler_scalar_policy.h"

#if defined(__CCE_AICORE__)
namespace {
// Scalar observer binding belongs to the resident translation units.
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = false;
};
}  // namespace
#endif

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/pmu_collector_aicore.h"
// AICore dependency scheduling uses one device-side protocol.
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_async_finish.h"
#include "scheduler/scheduler_graph_control.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_watchdog.h"

#include "resident_scheduler.h"

__aicore__ void legacy_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type);

namespace {

template <bool WithProfiling, bool WithNativeGraph>
__aicore__ void dispatch_resident_scheduler(
    CoreType core_type, __gm__ SchedulerWorkerContext *context, __gm__ SchedulerRunControl *run_control,
    __gm__ void *scheduler_state_base, uint64_t profiling_level, uint64_t aicore_entry_cycles,
    uint64_t handshake_publish_cycles
) {
    if (core_type == CoreType::AIC) {
        simpler::hbg::resident::run<WithProfiling, WithNativeGraph, CoreType::AIC>(
            context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, handshake_publish_cycles
        );
    } else {
        simpler::hbg::resident::run<WithProfiling, WithNativeGraph, CoreType::AIV>(
            context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles, handshake_publish_cycles
        );
    }
}

}  // namespace

__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    // Block-local storage can outlive a launch that never unwound its stack.
    scheduler_scalar_bind(nullptr);
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
        scheduler_store_barrier();
        handshake->report_epoch = report_epoch;
    } else {
        // Kernel/persistent launch: unchanged. `aicore_done` is itself the
        // marker, its per-run reset is what makes it meaningful, and no stamp
        // is written.
        handshake->physical_core_id = get_physical_core_id();
        handshake->core_type = core_type;
        scheduler_store_barrier();
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
    bool context_wait_timed_out = false;
    const uint64_t context_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_RESIDENT_READY && startup_signal != AICORE_EXIT_SIGNAL) {
        scheduler_observe_cache_line(handshake);
        // Configuration remains valid while AICPU initializes execution state.
        // The error record does not: it may still belong to the preceding run.
        context_wait_timed_out =
            context_wait_timed_out || scheduler_watchdog_expired(
                                          context_wait_start, get_sys_cnt_aicore(),
                                          scheduler_gm_query(pending_run_control->scheduler_timeout_cycles)
                                      );
        startup_signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (startup_signal == AICORE_EXIT_SIGNAL) {
        // Failed initialization owns its error reporting. EXIT permits retirement,
        // but does not publish a writable scheduler context to this core.
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
    if (context_wait_timed_out) {
        scheduler_record_error(
            run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::TIMEOUT, nullptr, context,
            SchedulerErrorSite::CONTEXT_READY_TIMEOUT
        );
        wait_for_aicpu_exit_signal();
        scheduler_store_barrier();
        acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
        return;
    }
    scheduler_observe_cache_line(&run_control->chip_swimlane_level);
    const uint64_t profiling_level = scheduler_gm_query(run_control->chip_swimlane_level);
    // READY fixes the graph format for the whole run.
    scheduler_observe_cache_line(&context->native_header_offset);
    const bool native_graph = context->native_header_offset != 0;
    if (scheduler_task_timing_enabled(profiling_level) ||
        scheduler_gm_query(run_control->sampled_task_timing_enabled) != 0) {
        if (native_graph) {
            dispatch_resident_scheduler<true, true>(
                core_type, context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles,
                handshake_publish_cycles
            );
        } else {
            dispatch_resident_scheduler<true, false>(
                core_type, context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles,
                handshake_publish_cycles
            );
        }
    } else {
        if (native_graph) {
            dispatch_resident_scheduler<false, true>(
                core_type, context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles,
                handshake_publish_cycles
            );
        } else {
            dispatch_resident_scheduler<false, false>(
                core_type, context, run_control, scheduler_state_base, profiling_level, aicore_entry_cycles,
                handshake_publish_cycles
            );
        }
    }
    acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
}
