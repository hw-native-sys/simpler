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

#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>

#include "aicpu/cache_maintenance.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host_build_graph/runtime_status.h"
#include "runtime_status/error_names.h"
#include "runtime_status/scheduler_error.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

// This bridge is intentionally AICPU/host-only. AICore code publishes its
// detailed first error through SchedulerRunControl; AICPU maps that internal
// diagnostic to the existing host runtime status ABI without making this
// std::atomic helper visible to either AICore compiler.
inline int32_t aicore_scheduler_runtime_error_code(uint64_t scheduler_error) {
    if (scheduler_error == 0) return SIMPLER_ERROR_NONE;
    if (scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)) {
        return SIMPLER_ERROR_SCHEDULER_TIMEOUT;
    }
    if (scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::ASYNC_COMPLETION_INVALID))
        return SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    if (scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW))
        return SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
    if (scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::ASYNC_REGISTRATION_FAILED))
        return SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED;
    return SIMPLER_ERROR_INVALID_ARGS;
}

inline const char *aicore_scheduler_error_name(uint64_t scheduler_error) {
    if (scheduler_error == 0) return "NONE";
    return error_name(aicore_scheduler_runtime_error_code(scheduler_error));
}

inline bool latch_aicore_scheduler_runtime_error(std::atomic<int32_t> *sched_error_code, uint64_t scheduler_error) {
    return sched_error_code != nullptr &&
           try_latch_runtime_error(*sched_error_code, aicore_scheduler_runtime_error_code(scheduler_error));
}

inline bool record_aicore_scheduler_runtime_error(
    SchedulerRunControl *run_control, SchedulerGraphResult scheduler_error, SchedulerErrorSite error_site
) {
    if (run_control == nullptr || scheduler_error == SchedulerGraphResult::OK) return false;
    uint64_t expected = 0;
    if (!__atomic_compare_exchange_n(
            &run_control->error_claimed, &expected, UINT64_C(1), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        ))
        return false;
    __atomic_store_n(&run_control->error_task_ref, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_core_id, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_core_type, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_run_epoch, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_order_key, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_block_idx, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_group_seq, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_worker_index, UINT64_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_cluster_lane, UINT32_MAX, __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_identity_fields, UINT32_C(0), __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->error_site, static_cast<uint64_t>(error_site), __ATOMIC_RELAXED);
    // Identity write-back completes before the independent status line can
    // become visible to an AICore device load.
    cache_flush_range(
        reinterpret_cast<const uint8_t *>(run_control) + offsetof(SchedulerRunControl, error_claimed),
        offsetof(SchedulerRunControl, scheduler_error) - offsetof(SchedulerRunControl, error_claimed)
    );
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker)
        __atomic_store_n(
            &run_control->worker_errors[worker].status, static_cast<uint64_t>(scheduler_error), __ATOMIC_RELEASE
        );
    cache_flush_range(run_control->worker_errors, sizeof(run_control->worker_errors));
    __atomic_store_n(&run_control->scheduler_error, static_cast<uint64_t>(scheduler_error), __ATOMIC_RELEASE);
    cache_flush_range(
        reinterpret_cast<const uint8_t *>(run_control) + offsetof(SchedulerRunControl, scheduler_error),
        sizeof(SchedulerRunControl) - offsetof(SchedulerRunControl, scheduler_error)
    );
    return true;
}

inline void log_aicore_scheduler_error_identity(SchedulerRunControl *run, const char *phase) {
    if (run == nullptr) return;
    const auto *error_line = reinterpret_cast<const uint8_t *>(run) + offsetof(SchedulerRunControl, error_claimed);
    cache_invalidate_range(error_line, sizeof(SchedulerRunControl) - offsetof(SchedulerRunControl, error_claimed));
    const uint64_t status = __atomic_load_n(&run->scheduler_error, __ATOMIC_ACQUIRE);
    if (status == 0) return;
    // The identity spans two 64B lines, separate from the publication word.
    // Reload both after acquiring the status to exclude prefetched identity.
    cache_invalidate_range(error_line, 128);
    LOG_ERROR(
        "A5 HBG scheduler identity: phase=%s task=%" PRIu64 " status=%" PRIu64 " core=%" PRIu64 " type=%" PRIu64
        " error=%s site=%" PRIu64 " epoch=%" PRIu64 " order=%" PRIu64 " block=%" PRIu64 " group_seq=%" PRIu64
        " worker=%" PRIu64 " lane=%" PRIu32 " identity_fields=0x%" PRIx32,
        phase, run->error_task_ref, status, run->error_core_id, run->error_core_type,
        aicore_scheduler_error_name(status), run->error_site, run->error_run_epoch, run->error_order_key,
        run->error_block_idx, run->error_group_seq, run->error_worker_index, run->error_cluster_lane,
        run->error_identity_fields
    );
}

// The enclosing error snapshot validates the task-control and plan regions.
inline uint64_t aicore_prepared_completion_state(uint64_t base, const SchedulerPreparedLayout &layout, int64_t ref) {
    if (ref < 0 || static_cast<uint64_t>(ref) >= layout.task_count) return 0;
    const auto *task = reinterpret_cast<const SchedulerTaskControl *>(base + layout.task_controls_offset) + ref;
    cache_invalidate_range(
        reinterpret_cast<const uint8_t *>(task) + offsetof(SchedulerTaskControl, completion_state),
        sizeof(task->completion_state)
    );
    return __atomic_load_n(&task->completion_state, __ATOMIC_ACQUIRE);
}

inline bool aicore_prepared_retirement_inferred(
    uint64_t base, const SchedulerPreparedLayout &layout, uint64_t index, uint64_t state,
    const SchedulerPreparedCall &call, uint64_t completion
) {
    if (scheduler_prepared_phase(state) != SchedulerPreparedPhase::CLAIMED ||
        (state & SCHEDULER_PREPARED_MIX_OWNED) != 0 || call.task_ref < 0 ||
        static_cast<uint64_t>(call.task_ref) >= layout.task_count || index >= layout.execution_count)
        return false;
    const auto *plan = reinterpret_cast<const SchedulerExecutionPlan *>(base + layout.plans_offset) + call.task_ref;
    cache_invalidate_range(plan, sizeof(*plan));
    rmb();
    return scheduler_prepared_retirement_inferred(state, layout, index, call, *plan, completion);
}

inline void log_aicore_prepared_cell(uint64_t base, const SchedulerPreparedLayout &layout, uint64_t index) {
    const auto *control = reinterpret_cast<const SchedulerPreparedControl *>(base + layout.controls_offset) + index;
    const auto *shared_call = reinterpret_cast<const SchedulerPreparedCall *>(base + layout.calls_offset) + index;
    cache_invalidate_range(control, sizeof(*control));
    const uint64_t state = __atomic_load_n(&control->state, __ATOMIC_ACQUIRE);
    cache_invalidate_range(shared_call, sizeof(*shared_call));
    rmb();
    const auto call = *shared_call;
    const uint64_t completion = aicore_prepared_completion_state(base, layout, call.task_ref);
    const bool inferred = aicore_prepared_retirement_inferred(base, layout, index, state, call, completion);
    const auto phase = inferred ? SchedulerPreparedPhase::RETURNED : scheduler_prepared_phase(state);
    const uint32_t owner = scheduler_prepared_owner(state);
    const uint32_t executor = scheduler_prepared_execute_owner(state);
    const uint64_t binding = base + layout.bindings_offset + index * sizeof(SchedulerExecutionBinding);
    LOG_ERROR(
        "A5 HBG prepared cell: index=%" PRIu64 " state=0x%" PRIx64 " phase=%" PRIu64 " raw_phase=%" PRIu64
        " return_inferred=%u completion=0x%" PRIx64 " source=host owner=%" PRId64 " execute_owner=%" PRId64
        " mix_owner_recorded=%u mix_kernel_entered=%s call_epoch=%" PRIu64 " call_index=%" PRIu64 " call_task=%" PRId64
        " call_block=%" PRIu32 " call_lane=%u num_args=%u function=0x%" PRIx64,
        index, state, static_cast<uint64_t>(phase), state & UINT64_C(255), static_cast<unsigned>(inferred), completion,
        owner == UINT32_MAX ? INT64_C(-1) : static_cast<int64_t>(owner),
        executor == UINT32_MAX ? INT64_C(-1) : static_cast<int64_t>(executor),
        static_cast<unsigned>((state & SCHEDULER_PREPARED_MIX_OWNED) != 0),
        scheduler_prepared_mix_entry(state, "unknown"), call.run_epoch, call.execution_index, call.task_ref,
        call.block_idx, static_cast<unsigned>(call.subtask_slot), static_cast<unsigned>(call.num_args),
        call.function_bin_addr
    );
    LOG_ERROR(
        "A5 HBG prepared contexts: index=%" PRIu64 " local=0x%" PRIx64 " expected_local=0x%" PRIx64 " global=0x%" PRIx64
        " expected_global=0x%" PRIx64,
        index, call.args[SPMD_LOCAL_CONTEXT_INDEX], binding + offsetof(SchedulerExecutionBinding, local_context),
        call.args[SPMD_GLOBAL_CONTEXT_INDEX], binding + offsetof(SchedulerExecutionBinding, global_context)
    );
    for (uint16_t arg = 0; arg < call.num_args && arg < 8; ++arg)
        LOG_ERROR(
            "A5 HBG prepared argument: index=%" PRIu64 " arg=%u value=0x%" PRIx64, index, static_cast<unsigned>(arg),
            call.args[arg]
        );
}

inline void log_aicore_mix_proxy(
    uint64_t base, const SchedulerPreparedLayout &layout, uint64_t worker, const SchedulerWorkerContext &context,
    int32_t bank, uint64_t snapshot
) {
    const uint32_t ref = scheduler_mix_proxy_task_ref(snapshot);
    const uint32_t block = scheduler_mix_proxy_block_idx(snapshot);
    LOG_ERROR(
        "A5 HBG MIX proxy: epoch=%" PRIu64 " worker=%" PRIu64 " physical_core_id=%" PRId32 " cluster=%" PRIu64
        " bank=%" PRId32 " task=%" PRIu32 " block=%" PRIu32 " phase=published",
        layout.run_epoch, worker, context.physical_core_id, context.cluster_index, bank, ref, block
    );
    if (ref >= layout.task_count) return;
    const auto *shared_plan = reinterpret_cast<const SchedulerExecutionPlan *>(base + layout.plans_offset) + ref;
    cache_invalidate_range(shared_plan, sizeof(*shared_plan));
    rmb();
    const auto plan = *shared_plan;
    if (block >= plan.block_num || plan.active_mask_bits > 7 || plan.active_lane_count < 2 ||
        plan.active_lane_count != __builtin_popcount(plan.active_mask_bits) ||
        plan.execution_count != static_cast<uint32_t>(plan.block_num) * plan.active_lane_count ||
        plan.execution_begin > layout.execution_count ||
        plan.execution_count > layout.execution_count - plan.execution_begin)
        return;
    const auto *controls = reinterpret_cast<const SchedulerPreparedControl *>(base + layout.controls_offset);
    uint64_t index = plan.execution_begin + static_cast<uint64_t>(block) * plan.active_lane_count;
    // Each raw state carries the fixed owner's worker ID and its observed phase.
    uint64_t states[SUBTASK_SLOT_COUNT]{};
    for (uint32_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
        if ((plan.active_mask_bits & (1U << lane)) == 0) continue;
        cache_invalidate_range(&controls[index], sizeof(controls[index]));
        states[lane] = __atomic_load_n(&controls[index++].state, __ATOMIC_ACQUIRE);
    }
    LOG_ERROR(
        "A5 HBG MIX members: task=%" PRIu32 " block=%" PRIu32 " active_mask=0x%x aic=0x%" PRIx64 " aiv0=0x%" PRIx64
        " aiv1=0x%" PRIx64 " completion=0x%" PRIx64,
        ref, block, static_cast<unsigned>(plan.active_mask_bits), states[0], states[1], states[2],
        aicore_prepared_completion_state(base, layout, ref)
    );
}

inline void log_aicore_prepared_error(const SchedulerWorkerContext *context, SchedulerRunControl *run) {
    if (context == nullptr || run == nullptr) return;
    const auto *error_begin = reinterpret_cast<const uint8_t *>(run) + offsetof(SchedulerRunControl, error_claimed);
    cache_invalidate_range(
        error_begin, offsetof(SchedulerRunControl, worker_errors) - offsetof(SchedulerRunControl, error_claimed)
    );
    if (__atomic_load_n(&run->scheduler_error, __ATOMIC_ACQUIRE) == 0) return;
    const uint64_t base = context->scheduler_state_base_address;
    const uint64_t bytes = context->scheduler_state_bytes;
    const auto region_valid = [bytes](uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment) {
        return offset != 0 && offset % alignment == 0 && offset <= bytes && count <= (bytes - offset) / stride;
    };
    if (base == 0 || bytes > UINT64_MAX - base ||
        !region_valid(
            context->prepared_layout_offset, 1, sizeof(SchedulerPreparedLayout), alignof(SchedulerPreparedLayout)
        )) {
        LOG_ERROR("A5 HBG prepared snapshot: invalid state or prepared layout range");
        return;
    }
    const auto *shared_layout =
        reinterpret_cast<const SchedulerPreparedLayout *>(base + context->prepared_layout_offset);
    cache_invalidate_range(shared_layout, sizeof(*shared_layout));
    rmb();
    const auto layout = *shared_layout;
    LOG_ERROR(
        "A5 HBG prepared failure: layout_epoch=%" PRIu64 " error_epoch=%" PRIu64 " task=%" PRIu64 " block=%" PRIu64
        " worker=%" PRIu64 " physical_lane=%" PRIu32,
        layout.run_epoch, run->error_run_epoch, run->error_task_ref, run->error_block_idx, run->error_worker_index,
        run->error_cluster_lane
    );
    if (layout.task_controls_offset != context->task_controls_offset ||
        !region_valid(
            layout.task_controls_offset, layout.task_count, sizeof(SchedulerTaskControl), alignof(SchedulerTaskControl)
        ) ||
        !region_valid(
            layout.plans_offset, layout.task_count, sizeof(SchedulerExecutionPlan), alignof(SchedulerExecutionPlan)
        ) ||
        !region_valid(
            layout.controls_offset, layout.execution_count, sizeof(SchedulerPreparedControl),
            alignof(SchedulerPreparedControl)
        ) ||
        !region_valid(
            layout.calls_offset, layout.execution_count, sizeof(SchedulerPreparedCall), alignof(SchedulerPreparedCall)
        ) ||
        !region_valid(
            layout.bindings_offset, layout.execution_count, sizeof(SchedulerExecutionBinding),
            alignof(SchedulerExecutionBinding)
        )) {
        LOG_ERROR("A5 HBG prepared snapshot: invalid execution storage range");
        return;
    }
    const uint64_t workers = run->active_worker_count;
    if (workers <= SCHEDULER_WORKER_CAPACITY &&
        region_valid(
            context->worker_contexts_offset, workers, sizeof(SchedulerWorkerContext), alignof(SchedulerWorkerContext)
        )) {
        const auto *contexts = reinterpret_cast<const SchedulerWorkerContext *>(base + context->worker_contexts_offset);
        for (uint64_t worker = 0; worker < workers; ++worker) {
            const auto &entry = contexts[worker];
            cache_invalidate_range(
                reinterpret_cast<const uint8_t *>(&entry) + offsetof(SchedulerWorkerContext, mix_proxy_published),
                sizeof(entry.mix_proxy_published)
            );
            scheduler_visit_mix_proxy_records(entry, [&](int32_t bank, uint64_t state) {
                log_aicore_mix_proxy(base, layout, worker, entry, bank, state);
            });
        }
    }
    if (run->error_task_ref >= layout.task_count || (run->error_identity_fields & SCHEDULER_ERROR_HAS_BLOCK) == 0) {
        // Keep a separate diagnostic budget for each shared phase.
        uint32_t reported[static_cast<uint32_t>(SchedulerPreparedPhase::ENTERED) + 1]{};
        const auto *controls = reinterpret_cast<const SchedulerPreparedControl *>(base + layout.controls_offset);
        for (uint64_t index = 0; index < layout.execution_count; ++index) {
            cache_invalidate_range(&controls[index], sizeof(controls[index]));
            const uint64_t state = __atomic_load_n(&controls[index].state, __ATOMIC_ACQUIRE);
            const auto phase = scheduler_prepared_phase(state);
            if (phase < SchedulerPreparedPhase::CLAIMED || phase > SchedulerPreparedPhase::ENTERED ||
                (phase == SchedulerPreparedPhase::RETURNED && (state & SCHEDULER_PREPARED_MIX_OWNED) == 0) ||
                reported[static_cast<uint32_t>(phase)] >= 8)
                continue;
            const auto *call = reinterpret_cast<const SchedulerPreparedCall *>(base + layout.calls_offset) + index;
            cache_invalidate_range(call, sizeof(*call));
            rmb();
            const uint64_t completion = aicore_prepared_completion_state(base, layout, call->task_ref);
            if (aicore_prepared_retirement_inferred(base, layout, index, state, *call, completion) ||
                (phase == SchedulerPreparedPhase::RETURNED && scheduler_completion_done(completion)))
                continue;
            log_aicore_prepared_cell(base, layout, index);
            ++reported[static_cast<uint32_t>(phase)];
        }
        return;
    }
    const auto *shared_plan =
        reinterpret_cast<const SchedulerExecutionPlan *>(base + layout.plans_offset) + run->error_task_ref;
    cache_invalidate_range(shared_plan, sizeof(*shared_plan));
    rmb();
    const auto plan = *shared_plan;
    LOG_ERROR(
        "A5 HBG prepared task plan: begin=%" PRIu64 " count=%" PRIu32 " blocks=%u active_mask=0x%x active_lanes=%u",
        plan.execution_begin, plan.execution_count, static_cast<unsigned>(plan.block_num),
        static_cast<unsigned>(plan.active_mask_bits), static_cast<unsigned>(plan.active_lane_count)
    );
    if (run->error_block_idx >= plan.block_num) return;
    // Physical AIV lanes need not match the task's logical subtask slot.
    uint32_t lane_rank = 0;
    for (uint32_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
        if ((plan.active_mask_bits & (1U << lane)) == 0) continue;
        const uint64_t relative = run->error_block_idx * plan.active_lane_count + lane_rank++;
        if (relative >= plan.execution_count || plan.execution_begin >= layout.execution_count ||
            relative >= layout.execution_count - plan.execution_begin) {
            LOG_ERROR("A5 HBG prepared snapshot: invalid block execution range");
            return;
        }
        log_aicore_prepared_cell(base, layout, plan.execution_begin + relative);
    }
}

/**
 * The run's host runtime status with the AICore's recorded error folded into
 * the shared header first.
 *
 * Folding is first-wins (`latch_aicore_scheduler_runtime_error` compare-
 * exchanges from `SIMPLER_ERROR_NONE`), so an error already latched keeps its
 * own code and a control holding none writes nothing. A null `sched_error_code`
 * has nothing to report; a null `run_control` reports the header as it stands,
 * because a caller that could not look has not established that the run was
 * clean.
 *
 * This folds whatever the control holds **at the moment of the call** and makes
 * no demand on when that is: the supervisor calls it mid-run and the
 * failed-init exit calls it before any shutdown. Whether a fold is *complete*
 * is a property of the call site rather than of this function — only the
 * finalizer's call runs after every participant has finished its shutdown
 * partition, which is what lets that one cover every core instead of the
 * caller's own.
 */
inline int32_t
settle_aicore_runtime_status(const SchedulerRunControl *run_control, std::atomic<int32_t> *sched_error_code) {
    if (sched_error_code == nullptr) return SIMPLER_ERROR_NONE;
    if (run_control != nullptr) {
        (void)latch_aicore_scheduler_runtime_error(sched_error_code, run_control->scheduler_error);
    }
    return runtime_status_from_error_code(sched_error_code->load(std::memory_order_acquire));
}
