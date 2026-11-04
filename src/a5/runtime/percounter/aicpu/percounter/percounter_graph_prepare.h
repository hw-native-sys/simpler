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
 * The host planned every invocation -- its body's ids, its range in each list
 * -- but left the ranges zero and the body counters unwritten. Here the AICPU
 * threads that do not supervise take the invocations in priority order. For
 * each one they call upstream's localize, then materialize it a slice at a time,
 * and after every slice publish what it made runnable:
 *
 *   1. the slice's body tasks' counters, preset and tagged with the invocation;
 *   2. the longest prefix of each of the invocation's ranges whose tasks are
 *      all materialized, as order entries carrying the tasks' percounter ids.
 *
 * Counters before entries: a lane holding an entry reads its own counter line
 * and its producers', and a producer always has the lower body index, so it is
 * in this slice or an earlier one.
 *
 * Header-only because aicpu_executor.cpp is included by name into a unit test.
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
#include "scheduler/percounter/percounter_types.h"
#include "scheduler/scheduler_graph.h"

namespace simpler::hbg::percounter {

template <typename T>
inline T *percounter_region(PercounterControl *control, uint64_t offset) {
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(control) + offset);
}

/**
 * Drop whatever this AICPU's caches hold of the lines the materializing threads
 * read or partly rewrite. Leader only, after the host's upload and before any
 * thread materializes.
 *
 * An order list line holds the host's entries beside a range the AICPU fills,
 * and the flush after a fill writes the whole line back: a copy cached from an
 * earlier run would put that run's entries over this one's.
 */
inline void percounter_graph_prepare_init(PercounterControl *control) {
    const uint64_t invocation_count = control->graph_invocation_count;
    if (invocation_count == 0) return;
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
        cache_invalidate_range(
            percounter_region<uint8_t>(control, tmpl->tasks_offset),
            static_cast<size_t>(tmpl->task_count) * sizeof(PercounterBodyTask)
        );
        cache_invalidate_range(
            percounter_region<uint8_t>(control, tmpl->fanin_offsets_offset),
            static_cast<size_t>(tmpl->task_count + 1) * sizeof(int32_t)
        );
        for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
            cache_invalidate_range(
                percounter_region<uint8_t>(control, tmpl->orders_offset[l]),
                static_cast<size_t>(tmpl->order_count[l]) * sizeof(uint64_t)
            );
        }
    }
    cache_invalidate_range(
        percounter_region<uint8_t>(control, control->orders_offset[0]),
        static_cast<size_t>(control->order_count[0]) * sizeof(uint64_t)
    );
    cache_invalidate_range(
        percounter_region<uint8_t>(control, control->orders_offset[1]),
        static_cast<size_t>(control->order_count[1]) * sizeof(uint64_t)
    );
    if (control->mix_order_count != 0) {
        cache_invalidate_range(
            percounter_region<uint8_t>(control, control->mix_order_offset),
            static_cast<size_t>(control->mix_order_count) * sizeof(uint64_t)
        );
    }
}

inline void percounter_graph_prepare_fail(PercounterGraphPrepare *prepare, uint64_t rank, SchedulerGraphResult why) {
    uint64_t expected = 0;
    if (__atomic_compare_exchange_n(
            &prepare->error, &expected, static_cast<uint64_t>(why), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        )) {
        __atomic_store_n(&prepare->error_invocation, rank, __ATOMIC_RELEASE);
    }
}

/** Write back one materialized body task: its storage entry and its argument spans. */
inline void percounter_flush_body_task(ChipTaskStorage &storage) {
    cache_flush_range(&storage, sizeof(storage));
    TaskPayload &payload = storage.payload;
    if (payload.tensor_count > 0)
        cache_flush_range(
            payload.tensor_data(), static_cast<size_t>(payload.tensor_count) * sizeof(*payload.tensor_data())
        );
    if (payload.scalar_count > 0)
        cache_flush_range(payload.scalar_data(), static_cast<size_t>(payload.scalar_count) * sizeof(uint64_t));
}

/** Materialize and publish the invocation at `rank`. Returns false on any failure, already recorded. */
inline bool percounter_prepare_invocation(
    PercounterControl *control, PercounterGraphPrepare *prepare, uint64_t rank, const GraphImageView &view
) {
    auto *invocation = percounter_region<PercounterGraphInvocation>(control, control->graph_invocations_offset) + rank;
    const auto *tmpl = percounter_region<const PercounterGraphTemplate>(control, invocation->template_offset);
    const auto *body = percounter_region<const PercounterBodyTask>(control, tmpl->tasks_offset);
    const auto *fanin_rows = percounter_region<const int32_t>(control, tmpl->fanin_offsets_offset);
    auto *counters = percounter_region<PercounterCounter>(control, control->counters_offset);
    uint64_t *lists[PERCOUNTER_LIST_COUNT] = {
        percounter_region<uint64_t>(control, control->orders_offset[0]),
        percounter_region<uint64_t>(control, control->orders_offset[1]),
        control->mix_order_count != 0 ? percounter_region<uint64_t>(control, control->mix_order_offset) : nullptr,
    };
    const uint64_t body_base = invocation->body_base;

    auto *storage = reinterpret_cast<ChipTaskStorage *>(control->graph_storage_address);
    if (storage == nullptr || invocation->shell_task_id >= control->graph_storage_count) {
        percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    cache_invalidate_range(&storage[invocation->shell_task_id], sizeof(ChipTaskStorage));
    ChipTaskSlotState &outer = storage[invocation->shell_task_id].slot;
    GraphExecution *execution = graph_execution_localize(outer, view);
    if (execution == nullptr || static_cast<uint64_t>(execution->task_count) != tmpl->task_count) {
        percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }
    invocation->storage_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(execution->task_storage));
    cache_flush_range(&invocation->storage_address, sizeof(invocation->storage_address));

    uint64_t published[PERCOUNTER_LIST_COUNT] = {0, 0, 0};
    int32_t materialized = 0;
    while (true) {
        int32_t made = 0;
        const GraphMaterializeResult result =
            graph_execution_materialize_slice(outer, *execution, view, PERCOUNTER_GRAPH_SLICE_TASKS, &made);
        if (result != GraphMaterializeResult::PENDING && result != GraphMaterializeResult::MATERIALIZED) {
            percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        const int32_t end = execution->materialized_tasks;
        for (int32_t i = materialized; i < end; ++i) {
            percounter_flush_body_task(execution->task_at(i));
            PercounterCounter &counter = counters[body_base + static_cast<uint64_t>(i)];
            counter.value = percounter_counter_preset(static_cast<int64_t>(body[i].subtask_total));
            counter.storage_address = invocation->storage_address;
            counter.body_index = static_cast<uint32_t>(i);
            counter.flags = body[i].flags;
            counter.shell_task_id = static_cast<uint32_t>(invocation->shell_task_id);
            counter.fanin_count = static_cast<uint32_t>(fanin_rows[i + 1] - fanin_rows[i]);
            counter.fanin_offset = tmpl->fanin_indices_offset + static_cast<uint64_t>(fanin_rows[i]) * sizeof(uint64_t);
            counter.invocation = static_cast<uint32_t>(rank);
            cache_flush_range(&counter, sizeof(counter));
        }
        materialized = end;
        bool any = false;
        for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
            const uint64_t count = tmpl->order_count[l];
            if (count == 0) continue;
            const auto *source = percounter_region<const uint64_t>(control, tmpl->orders_offset[l]);
            uint64_t *target = lists[l] + invocation->order_base[l];
            const uint64_t first = published[l];
            uint64_t next = first;
            while (next < count && percounter_order_task_id(source[next]) < materialized) {
                const uint64_t entry = source[next];
                target[next] = percounter_order_entry(
                    static_cast<int64_t>(body_base) + percounter_order_task_id(entry),
                    percounter_order_block_idx(entry), percounter_order_block_num(entry)
                );
                ++next;
            }
            if (next != first) {
                cache_flush_range(target + first, static_cast<size_t>(next - first) * sizeof(uint64_t));
                published[l] = next;
                any = true;
            }
        }
        if (any) {
            uint64_t expected = 0;
            const uint64_t now = get_sys_cnt_aicpu();
            __atomic_compare_exchange_n(
                &prepare->first_publish_cycles, &expected, now, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
            );
        }
        if (result == GraphMaterializeResult::MATERIALIZED) break;
    }
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
        if (published[l] != tmpl->order_count[l]) {
            percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
    }
    return true;
}

/**
 * Take invocations until none is left or one has failed. Any AICPU thread may
 * call this; the threads share the work through the prepare record.
 */
inline void percounter_prepare_graphs(Runtime *runtime, PercounterControl *control) {
    if (control == nullptr || control->graph_invocation_count == 0) return;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    const GraphImageView view = graph_image_view_for_reader(*runtime);
    const uint64_t invocation_count = control->graph_invocation_count;
    while (__atomic_load_n(&prepare->error, __ATOMIC_ACQUIRE) == 0) {
        const uint64_t rank = __atomic_fetch_add(&prepare->next, uint64_t{1}, __ATOMIC_ACQ_REL);
        if (rank >= invocation_count) return;
        if (!view.valid()) {
            percounter_graph_prepare_fail(prepare, rank, SchedulerGraphResult::INVALID_ARGUMENTS);
            return;
        }
        if (!percounter_prepare_invocation(control, prepare, rank, view)) return;
        if (__atomic_add_fetch(&prepare->prepared, uint64_t{1}, __ATOMIC_ACQ_REL) == invocation_count)
            __atomic_store_n(&prepare->prepared_cycles, get_sys_cnt_aicpu(), __ATOMIC_RELEASE);
    }
}

/** The first materialization failure, as a SchedulerGraphResult, or 0. */
inline uint64_t percounter_graph_prepare_error(PercounterControl *control) {
    if (control->graph_invocation_count == 0) return 0;
    auto *prepare = percounter_region<PercounterGraphPrepare>(control, control->graph_prepare_offset);
    const uint64_t error = __atomic_load_n(&prepare->error, __ATOMIC_ACQUIRE);
    if (error != 0) {
        LOG_ERROR(
            "A5 HBG percounter: Graph invocation %" PRIu64 " (of %" PRIu64 ") failed to materialize: %" PRIu64,
            __atomic_load_n(&prepare->error_invocation, __ATOMIC_ACQUIRE), control->graph_invocation_count, error
        );
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
        "A5 HBG percounter: %" PRIu64 " Graph invocations prepared %" PRIu64 "; first body published at +%.1f us, "
        "last at +%.1f us",
        control->graph_invocation_count, __atomic_load_n(&prepare->prepared, __ATOMIC_ACQUIRE),
        first != 0 ? us(first) : -1.0, done != 0 ? us(done) : -1.0
    );
}

}  // namespace simpler::hbg::percounter
