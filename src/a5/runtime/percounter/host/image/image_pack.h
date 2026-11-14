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

#include "host_build_graph/shared_memory.h"
#include "host_build_graph/graph_execution.h"
#include "task_image_layout.h"

static_assert(sizeof(ChipTaskStorage) == SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
static_assert(offsetof(ChipTaskStorage, payload) == SCHEDULER_GRAPH_PAYLOAD_OFFSET);
static_assert(offsetof(TaskPayload, predicate) == SCHEDULER_GRAPH_PREDICATE_OFFSET);
static_assert(offsetof(SharedMemoryHeader, sched_error_code) == TASK_IMAGE_ERROR_CODE_OFFSET);

namespace task_image {
inline constexpr bool native_graph_plan = false;
inline sm_layout::SegmentOffsets segments(const sm_layout::BindUsage &used) {
    return sm_layout::segment_offsets(sm_layout::image_extents(used));
}
inline uint64_t pack(
    const char *source, uint64_t capacity, const sm_layout::BindUsage &used, const sm_layout::HeapRebase &heap,
    char *out
) {
    return sm_layout::compact_live_image(source, capacity, used, heap, out);
}
inline const TaskPayload &payload(const void *image, const sm_layout::SegmentOffsets &layout, uint64_t task) {
    return (reinterpret_cast<const ChipTaskStorage *>(static_cast<const char *>(image) + layout.storage) + task)
        ->payload;
}
inline bool graph_storage_bytes(const GraphDefinition &definition, uint64_t *bytes) {
    GraphExecutionStorageLayout layout{};
    if (!graph_execution_storage_layout(
            definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
            &layout
        ) ||
        layout.total_bytes != definition.execution_storage_bytes)
        return false;
    *bytes = layout.total_bytes;
    return true;
}
}  // namespace task_image
