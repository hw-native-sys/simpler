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
/**
 * Materializing Graph bodies for percounter's lanes.
 *
 * The Host supplies the calendar rows and the offsets of their bindings.
 * AICPU threads that do not supervise localize each invocation and materialize
 * its descriptors and argument spans in slices, independently of fanin.
 * Calendar rows wait for the invocation's materialized prefix before loading
 * their kernel and argument bindings.
 */

#pragma once

#include <cinttypes>
#include <cstdint>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_time.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "device/graph_image_source.h"
#include "host_build_graph/graph_execution.h"
#include "runtime.h"
#include "scheduler/calendar_wire.h"
#include "scheduler/wire_types.h"
#include "scheduler/scheduler_graph.h"

namespace simpler::percounter {

template <typename T>
inline T *percounter_region(PercounterControl *control, uint64_t offset) {
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(control) + offset);
}

/**
 * Drop whatever this AICPU's caches hold of the lines the materializing threads
 * read or partly rewrite. Leader only, after the host's upload and before any
 * thread materializes.
 *
 * The templates and invocation records remain live through every materializer.
 */
inline void percounter_graph_prepare_init(PercounterControl *control) {
    const uint64_t invocation_count = control->graph_invocation_count;
    if (invocation_count == 0) return;
    cache_invalidate_range(
        reinterpret_cast<const void *>(control->callable_addresses_address),
        static_cast<size_t>(control->callable_addresses_count) * sizeof(uint64_t)
    );
    cache_invalidate_range(
        percounter_region<uint8_t>(control, control->graph_invocations_offset),
        static_cast<size_t>(invocation_count) * sizeof(PercounterGraphInvocation)
    );
    cache_invalidate_range(
        percounter_region<uint8_t>(control, control->graph_prepare_offset), sizeof(PercounterGraphPrepare)
    );
    const auto *invocations = percounter_region<PercounterGraphInvocation>(control, control->graph_invocations_offset);
    for (uint64_t k = 0; k < invocation_count; ++k) {
        const auto *tmpl = percounter_region<PercounterGraphTemplate>(control, invocations[k].template_offset);
        cache_invalidate_range(tmpl, sizeof(*tmpl));
    }
}

inline void percounter_graph_prepare_fail(PercounterGraphPrepare *prepare, uint64_t rank, SchedulerGraphResult why) {
    // The invocation is claimed first, and carries `rank + 1` so that zero
    // still means none: readers gate on `error`, so writing it first would let
    // one report a failure against an invocation number not yet stored.
    uint64_t unclaimed = 0;
    if (!__atomic_compare_exchange_n(
            &prepare->error_invocation, &unclaimed, rank + 1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        ))
        return;
    uint64_t expected = 0;
    __atomic_compare_exchange_n(
        &prepare->error, &expected, static_cast<uint64_t>(why), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    );
}

/** Materialize and publish the invocation at `rank`. Returns false on any failure, already recorded. */
inline bool percounter_prepare_invocation(
    PercounterControl *control, PercounterGraphPrepare *prepare, uint64_t rank, const GraphImageView &view,
    int32_t thread_idx
) {
    auto *invocation = percounter_region<PercounterGraphInvocation>(control, control->graph_invocations_offset) + rank;
    const uint64_t timing_offset = invocation->ready_timing_offset;
    auto *ready_timing =
        timing_offset == 0 ? nullptr : percounter_region<PercounterGraphReadyTiming>(control, timing_offset);
    if (ready_timing != nullptr) {
        invocation->materializer_thread_id = thread_idx;
        invocation->materialize_start_cycles = get_sys_cnt_aicpu();
    }
    const auto *tmpl = percounter_region<const PercounterGraphTemplate>(control, invocation->template_offset);
    const uint64_t body_base = invocation->body_base;

    auto *storage = reinterpret_cast<ChipTaskStorage *>(control->graph_storage_address);
    if (storage == nullptr || invocation->shell_task_id >= control->graph_storage_count) {
        percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    cache_invalidate_range(&storage[invocation->shell_task_id], sizeof(ChipTaskStorage));
    ChipTaskSlotState &outer = storage[invocation->shell_task_id].slot;
    GraphExecution *execution = graph_execution_localize(outer, view, GraphArgumentStorage::KERNEL_ARGUMENTS);
    if (execution == nullptr || static_cast<uint64_t>(execution->task_count) != tmpl->task_count) {
        LOG_ERROR(
            "A5 percounter: Graph localization rank=%" PRIu64 " shell=%" PRIu64 " expected=%" PRIu64 " actual=%d", rank,
            invocation->shell_task_id, tmpl->task_count, execution == nullptr ? -1 : execution->task_count
        );
        percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }
    auto *calendar = scheduler_state_at<CalendarRangeHeader>(control, control->calendar_offset);
    cache_invalidate_range(calendar, sizeof(*calendar));
    const auto *patches = reinterpret_cast<const CalendarDispatchPatch *>(
                              reinterpret_cast<const uint8_t *>(calendar) + calendar->patch_offset
                          ) +
                          body_base;
    cache_invalidate_range(patches, static_cast<size_t>(tmpl->task_count) * sizeof(*patches));
    int32_t materialized = 0;
    uint32_t ready_count = 0;
    while (true) {
        // At the slice boundary, so a failure anywhere stops this invocation
        // too rather than only the taking of the next one. A graph's
        // invocations are many slices each, which is most of what a cancelled
        // teardown would otherwise still wait for.
        if (__atomic_load_n(&prepare->error, __ATOMIC_ACQUIRE) != 0) return false;
        int32_t made = 0;
        const GraphMaterializeResult result =
            graph_execution_materialize_slice(outer, *execution, view, PERCOUNTER_GRAPH_SLICE_TASKS, &made);
        if (result != GraphMaterializeResult::PENDING && result != GraphMaterializeResult::MATERIALIZED) {
            LOG_ERROR(
                "A5 percounter: Graph materialization rank=%" PRIu64 " result=%d materialized=%d", rank,
                static_cast<int>(result), execution->materialized_tasks
            );
            percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        const int32_t end = execution->materialized_tasks;
        for (int32_t i = materialized; i < end; ++i) {
            const auto &task = execution->task_at(i);
            const auto &payload = task.payload;
            if (payload.tensor_count > 0)
                cache_flush_range(
                    payload.tensor_data(), static_cast<size_t>(payload.tensor_count) * sizeof(*payload.tensor_data())
                );
            auto *arguments = execution->kernel_arguments(i);
            cache_flush_range(arguments, (payload.tensor_count + payload.scalar_count) * sizeof(uint64_t));

            const uint32_t predicate_offset = patches[i].predicate_extra_offset;
            if (predicate_offset != 0) {
                auto *extra = reinterpret_cast<CalendarExecutionExtra *>(
                    reinterpret_cast<uint8_t *>(calendar) + predicate_offset
                );
                cache_invalidate_range(extra, sizeof(*extra));
                cache_flush_range(&payload.predicate, sizeof(payload.predicate));
                extra->predicate_address = reinterpret_cast<uint64_t>(&payload.predicate);
                cache_flush_range(extra, sizeof(*extra));
            }
            const auto *entries = reinterpret_cast<const uint64_t *>(control->callable_addresses_address);
            for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
                const uint32_t offset = patches[i].dispatch_offsets[lane];
                if (offset == 0) continue;
                const int32_t kernel = task.task.kernel_id[lane];
                if (kernel < 0 || static_cast<uint64_t>(kernel) >= control->callable_addresses_count ||
                    entries == nullptr || entries[kernel] == 0) {
                    percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_CALLABLE);
                    return false;
                }
                auto *binding = reinterpret_cast<CalendarDispatch *>(reinterpret_cast<uint8_t *>(calendar) + offset);
                cache_invalidate_range(binding, sizeof(*binding));
                binding->kernel_entry = entries[kernel];
                binding->args_address = reinterpret_cast<uint64_t>(arguments);
                cache_flush_range(binding, sizeof(*binding));
            }
        }
        wmb();
        __atomic_store_n(&invocation->calendar_ready, static_cast<uint64_t>(end), __ATOMIC_RELEASE);
        cache_flush_range(&invocation->calendar_ready, sizeof(invocation->calendar_ready));
        materialized = end;
        if (made > 0) {
            uint64_t expected = 0;
            const uint64_t now = get_sys_cnt_aicpu();
            if (ready_timing != nullptr) ready_timing[ready_count++] = {now, static_cast<uint64_t>(end)};
            __atomic_compare_exchange_n(
                &prepare->first_publish_cycles, &expected, now, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
            );
        }
        if (result == GraphMaterializeResult::MATERIALIZED) break;
    }
    if (ready_timing != nullptr) {
        invocation->materialize_end_cycles = get_sys_cnt_aicpu();
        if (ready_count != 0) cache_flush_range(ready_timing, ready_count * sizeof(*ready_timing));
        cache_flush_range(invocation, sizeof(*invocation));
    }
    return true;
}

/**
 * Take invocations until none is left or one has failed. Any AICPU thread may
 * call this; the threads share the work through the prepare record.
 */
inline void percounter_prepare_graphs(Runtime *runtime, PercounterControl *control, int32_t thread_idx) {
    if (control == nullptr || control->graph_invocation_count == 0) return;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    const GraphImageView view = graph_image_view_for_reader(*runtime);
    const uint64_t invocation_count = control->graph_invocation_count;
    while (__atomic_load_n(&prepare->error, __ATOMIC_ACQUIRE) == 0) {
        const uint64_t rank = __atomic_fetch_add(&prepare->next, uint64_t{1}, __ATOMIC_ACQ_REL);
        if (rank >= invocation_count) return;
        if (!view.valid()) {
            LOG_ERROR("A5 percounter: invalid Graph image for invocation=%" PRIu64 "", rank);
            percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
            return;
        }
        if (!percounter_prepare_invocation(control, prepare, rank, view, thread_idx)) return;
        if (__atomic_add_fetch(&prepare->prepared, uint64_t{1}, __ATOMIC_ACQ_REL) == invocation_count)
            __atomic_store_n(&prepare->prepared_cycles, get_sys_cnt_aicpu(), __ATOMIC_RELEASE);
    }
}

/**
 * Stop the assistants from taking further work, for a run that has failed
 * elsewhere.
 *
 * They take the next invocation only while `error` reads zero, so a failure
 * this record does not carry leaves them preparing every invocation that is
 * left. The completion gate waits for each of them, so that work is added to
 * a teardown the scheduler watchdog no longer bounds -- long enough, on a
 * graph of any size, for the op-execute timeout to fire and bury the error
 * that started it.
 *
 * First writer wins, as in percounter_graph_prepare_fail: a materialization
 * failure already here names the invocation it happened on, which this does
 * not, so it is the better record of the two.
 */
inline void percounter_graph_prepare_stop(PercounterControl *control, uint64_t error) {
    if (control->graph_invocation_count == 0 || error == 0) return;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    uint64_t expected = 0;
    __atomic_compare_exchange_n(&prepare->error, &expected, error, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

/**
 * The first materialization failure, as a SchedulerGraphResult, or 0.
 *
 * `report` names the invocation it happened on, which only the first reader
 * of a given failure should do: a second line for one failure reads as a
 * second failure.
 */
inline uint64_t percounter_graph_prepare_error(PercounterControl *control, bool report = true) {
    if (control->graph_invocation_count == 0) return 0;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    const uint64_t error = __atomic_load_n(&prepare->error, __ATOMIC_ACQUIRE);
    if (error != 0 && report) {
        // Zero where a teardown published the failure rather than an
        // invocation's own materialization, which names none.
        const uint64_t claimed = __atomic_load_n(&prepare->error_invocation, __ATOMIC_ACQUIRE);
        if (claimed == 0) {
            LOG_ERROR(
                "A5 percounter: Graph materialization stopped over %" PRIu64 " invocation(s): %" PRIu64,
                control->graph_invocation_count, error
            );
        } else {
            LOG_ERROR(
                "A5 percounter: Graph invocation %" PRIu64 " (of %" PRIu64 ") failed to materialize: %" PRIu64,
                claimed - 1, control->graph_invocation_count, error
            );
        }
    }
    return error;
}

/** One line per run: how soon the first body was runnable and when the last was. */
inline void percounter_graph_prepare_report(PercounterControl *control, uint64_t start_cycles) {
    if (control->graph_invocation_count == 0) return;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    const uint64_t first = __atomic_load_n(&prepare->first_publish_cycles, __ATOMIC_ACQUIRE);
    const uint64_t done = __atomic_load_n(&prepare->prepared_cycles, __ATOMIC_ACQUIRE);
    const auto us = [start_cycles](uint64_t cycles) -> double {
        if (cycles < start_cycles) return 0.0;
        return static_cast<double>(cycles - start_cycles) * 1e6 / static_cast<double>(PLATFORM_PROF_SYS_CNT_FREQ);
    };
    LOG_INFO(
        "A5 percounter: %" PRIu64 " Graph invocations prepared %" PRIu64 "; first body published at +%.1f us, "
        "last at +%.1f us",
        control->graph_invocation_count, __atomic_load_n(&prepare->prepared, __ATOMIC_ACQUIRE),
        first != 0 ? us(first) : -1.0, done != 0 ? us(done) : -1.0
    );
}

}  // namespace simpler::percounter
