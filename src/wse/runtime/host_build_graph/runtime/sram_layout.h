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
#include "sram_access.h"
#include "scheduler/scheduler_graph.h"

namespace wse {
// The last Block publishes completion here. For a Group, Block 0's explicitly
// assigned Core owns the join and the optional sync-start arrival count.
struct TaskSignal {
    uint32_t remaining_blocks, remaining_arrivals, expected_arrivals, completed;
};
struct TaskDependency {
    const TaskSignal *signal;
};
struct TaskRecord {
    uint64_t ref, entry;
    TaskSignal *signal;
    TaskDependency *predecessors;
    TaskRecord *next_control;
    uint32_t blocks, has_predicate, predecessor_count;
    SchedulerDispatchPredicate predicate;
};
struct Work {
    TaskRecord *task;
    uint32_t block, row, thread_num, remaining_pes;
    uint8_t abi_slots[PE_PER_CORE];
    bool started, execute;
    uint32_t tensor_count, scalar_count;
    uint64_t tensors_offset, scalars_offset;
};
struct PeCalendar {
    uint32_t *rows;
    uint32_t head, size;
    uint64_t active_row;
    bool arrived;
};
struct CoreControl {
    uint32_t initialized, ready, quiet, retired;
    uint64_t epoch, base;
    Work *blocks;
    PeCalendar calendars[PE_PER_CORE];
    SramAccess access;
    TaskRecord *first_control;
};
static_assert(std::is_standard_layout_v<TaskSignal> && std::is_trivially_copyable_v<TaskSignal>);
static_assert(std::is_standard_layout_v<TaskRecord> && std::is_trivially_copyable_v<TaskRecord>);
static_assert(std::is_standard_layout_v<CoreControl> && std::is_trivially_copyable_v<CoreControl>);
}  // namespace wse
