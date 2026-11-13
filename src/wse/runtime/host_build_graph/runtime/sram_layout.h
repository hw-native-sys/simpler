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
struct Work {
    uint64_t task;
    uint32_t block, thread_num;
    uint32_t thread_pes[PE_PER_CORE];
    uint32_t tensor_count, scalar_count;
    uint64_t tensors_offset, scalars_offset;
};
struct Queue {
    uint64_t work_offset, capacity, head, tail;
};
struct CoreControl {
    uint32_t initialized, ready, quiet, retired;
    uint64_t expected_tasks;
    Queue queue;
    uint64_t block_epoch, block_serial;
    uint32_t active, started, finished, execute;
};
struct TaskRecord {
    uint64_t entry, predecessors_offset;
    uint32_t blocks, host_done, has_predicate, predecessor_count;
    uint32_t remaining_blocks, completed;
    SchedulerDispatchPredicate predicate;
};
struct RunControl {
    uint64_t epoch, expected_tasks, completed;
    uint64_t tasks_offset, core_offsets_offset;
    SramAccess access;
    uint32_t core_count, task_count, configured, go, stop;
    int32_t init_error;
};
static_assert(std::is_standard_layout_v<RunControl> && std::is_trivially_copyable_v<RunControl>);
static_assert(std::is_standard_layout_v<TaskRecord> && std::is_trivially_copyable_v<TaskRecord>);
static_assert(std::is_standard_layout_v<CoreControl> && std::is_trivially_copyable_v<CoreControl>);
}  // namespace wse
