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

#include <cstdint>

#include "aicpu/cache_maintenance.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "runtime.h"
#include "scheduler/calendar_wire.h"
#include "scheduler/wire_types.h"

namespace simpler::percounter {

template <typename T>
inline const T *calendar_pmu_table(const uint8_t *image, uint64_t bytes, uint64_t offset, uint64_t count) {
    if (offset < sizeof(CalendarRangeHeader) || offset % alignof(T) != 0 || offset > bytes ||
        count > (bytes - offset) / sizeof(T))
        return nullptr;
    return reinterpret_cast<const T *>(image + offset);
}

// A retired core is the only writer of its cells. Missing cells prevent the
// collector from settling this run's accounting against a partial capture.
inline bool collect_calendar_pmu_records(Runtime *runtime, int32_t thread_idx, uint64_t *missing_out) {
    *missing_out = 0;
    auto *control = reinterpret_cast<const PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    if (control == nullptr) return false;
    const uint64_t calendar_offset = control->calendar_offset;
    if (calendar_offset < sizeof(PercounterControl) || calendar_offset > control->image_bytes ||
        control->image_bytes - calendar_offset < sizeof(CalendarRangeHeader))
        return false;
    const auto *image = reinterpret_cast<const uint8_t *>(control) + calendar_offset;
    const uint64_t bytes = control->image_bytes - calendar_offset;

    const auto *calendar = reinterpret_cast<const CalendarRangeHeader *>(image);
    const uint32_t workers = calendar->worker_count;
    if (workers != static_cast<uint32_t>(runtime->get_worker_count()) || workers % PLATFORM_CORES_PER_BLOCKDIM != 0)
        return false;
    const auto *indices =
        calendar_pmu_table<uint32_t>(image, bytes, calendar->worker_index_offset, uint64_t{workers} + 1);
    const auto *ranges = calendar_pmu_table<CalendarRange>(image, bytes, calendar->range_offset, calendar->range_count);
    if (indices == nullptr || ranges == nullptr || indices[0] != 0 || indices[workers] != calendar->range_count)
        return false;

    const uint32_t cubes = workers / PLATFORM_CORES_PER_BLOCKDIM;
    for (uint32_t worker = 0; worker < workers; ++worker) {
        const uint32_t begin = indices[worker];
        const uint32_t end = indices[worker + 1];
        if (begin > end || end > calendar->range_count) return false;
        const auto core_type = worker < cubes ? CoreType::AIC : CoreType::AIV;
        for (uint32_t r = begin; r < end; ++r) {
            const auto &range = ranges[r];
            const auto *rows =
                calendar_pmu_table<CalendarExecutionRow>(image, bytes, range.rows_offset, range.row_count);
            const auto *cells = calendar_pmu_table<CalendarPmuCell>(image, bytes, range.pmu_offset, range.row_count);
            if (rows == nullptr || cells == nullptr) return false;
            cache_invalidate_range(cells, range.row_count * sizeof(CalendarPmuCell));
            rmb();
            for (uint32_t i = 0; i < range.row_count; ++i) {
                const auto &row = rows[i];
                if ((row.flags & CALENDAR_EXEC_CONTROL) != 0) continue;
                const auto &cell = cells[i];
                if (cell.publication == CalendarPmuPublication::Skipped && (row.flags & CALENDAR_EXEC_PREDICATE) != 0)
                    continue;
                if (cell.publication != CalendarPmuPublication::Executed || cell.kernel_id == UINT16_MAX) {
                    ++*missing_out;
                    continue;
                }
                const uint64_t binding = range.binding_offset;
                const auto *dispatch =
                    calendar_pmu_table<CalendarDispatch>(image, bytes, binding + row.dispatch_offset, 1);
                if (dispatch == nullptr) return false;
                const auto *extra =
                    calendar_pmu_table<CalendarExecutionExtra>(image, bytes, binding + dispatch->extra_offset, 1);
                if (extra == nullptr) return false;
                const TaskId task = extra->shell_task_id < 0 ?
                                        TaskId::make_global(static_cast<int32_t>(extra->task_id)) :
                                        TaskId::make_sub_task(extra->shell_task_id, extra->local_task_id);
                if (!task.is_valid()) return false;
                pmu_aicpu_record_staged_task(
                    static_cast<int>(worker), thread_idx, task, cell.kernel_id, core_type, cell.total_cycles,
                    cell.counters
                );
            }
        }
    }
    return *missing_out == 0;
}

}  // namespace simpler::percounter
