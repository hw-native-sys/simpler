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

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_phase_aicpu.h"
#include "scheduler/wire_types.h"

namespace simpler::percounter {

// All lanes have acknowledged retirement before their private rows are read.
inline void publish_calendar_task_timing(const PercounterControl *control, int worker_count) {
    if (control == nullptr || worker_count <= 0 || worker_count > PLATFORM_MAX_CORES) return;
    auto *records = aicpu_task_timing_records(get_platform_phase_base(), 0);
    if (records == nullptr) return;

    const uint64_t offset = control->task_timing_offset;
    if (offset == 0) return;
    const size_t bytes = worker_count * NUM_TASK_TIMING_SLOTS * sizeof(TaskTimingRecord);
    if (offset % alignof(TaskTimingRecord) != 0 || offset > control->image_bytes ||
        bytes > control->image_bytes - offset)
        return;
    const auto *slots = reinterpret_cast<const TaskTimingRecord *>(reinterpret_cast<const uint8_t *>(control) + offset);
    cache_invalidate_range(slots, bytes);

    uint64_t starts[NUM_TASK_TIMING_SLOTS];
    uint64_t finishes[NUM_TASK_TIMING_SLOTS];
    reduce_task_timing_slots(slots, worker_count, starts, finishes);
    for (int slot = 0; slot < NUM_TASK_TIMING_SLOTS; ++slot)
        records[slot] = {starts[slot], finishes[slot]};
    aicpu_publish_task_timing_tail_usage(1);
}

}  // namespace simpler::percounter
