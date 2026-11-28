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

#include <cstring>

#include "host_build_graph/shared_memory.h"
#include "host_build_graph/graph_execution.h"
#include "task_image_layout.h"
#include "dispatch_payload.h"

static_assert(sizeof(ChipTaskStorage) == SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
static_assert(offsetof(ChipTaskStorage, payload) == SCHEDULER_GRAPH_PAYLOAD_OFFSET);
static_assert(offsetof(TaskPayload, predicate) == SCHEDULER_GRAPH_PREDICATE_OFFSET);
static_assert(offsetof(SharedMemoryHeader, sched_error_code) == TASK_IMAGE_ERROR_CODE_OFFSET);

namespace task_image {
using Storage = ChipTaskStorage;
inline sm_layout::SegmentOffsets layout(const char *source, uint64_t capacity, const sm_layout::BindUsage &used) {
    always_assert(used.submitted_tasks <= capacity);
    auto result = sm_layout::segment_offsets(sm_layout::image_extents(used));
    const auto from = sm_layout::segment_offsets(capacity);
    const auto *storage = reinterpret_cast<const ChipTaskStorage *>(source + from.storage);
    uint64_t boundary_scalars = 0;
    for (uint64_t i = 0; i < used.submitted_tasks; ++i) {
        if (storage[i].slot.task_kind != TaskKind::GRAPH) continue;
        const int32_t count = storage[i].payload.scalar_count;
        always_assert(count >= 0 && static_cast<uint64_t>(count) <= GRAPH_MAX_SCALAR_ARGS);
        boundary_scalars += static_cast<uint64_t>(count);
    }
    result.end =
        result.scalar_pool +
        CHIP_ALIGN_UP(
            used.submitted_tasks * sizeof(DispatchPayload) + boundary_scalars * sizeof(uint64_t), CHIP_ALIGN_SIZE
        );
    always_assert(used.submitted_tasks <= capacity && result.end <= static_cast<uint64_t>(INT32_MAX));
    return result;
}
inline uint64_t pack(
    const char *source, uint64_t capacity, const sm_layout::BindUsage &used, const sm_layout::HeapRebase &heap,
    char *out, uint64_t device_base
) {
    const auto limits = sm_layout::mirror_extents(capacity);
    always_assert(
        used.submitted_tasks <= capacity && used.tensor_elems <= limits.tensor_elems &&
        used.scalar_elems <= limits.scalar_elems && used.fanin_elems <= limits.fanin_elems
    );
    const auto from = sm_layout::segment_offsets(capacity);
    const auto to = layout(source, capacity, used);
    const auto copy = [&](uint64_t destination, uint64_t origin, uint64_t bytes) {
        std::memcpy(out + destination, source + origin, static_cast<size_t>(bytes));
    };
    copy(0, 0, to.storage);
    auto &tasks = reinterpret_cast<SharedMemoryHeader *>(out)->tasks;
    tasks.task_storage = nullptr;
    tasks.task_states = nullptr;
    copy(to.storage, from.storage, used.submitted_tasks * sizeof(ChipTaskStorage));
    copy(to.task_states, from.task_states, used.submitted_tasks * sizeof(std::atomic<ChipTaskState>));
    copy(to.fanin_pool, from.fanin_pool, used.fanin_elems * sizeof(int32_t));
    copy(to.tensor_pool, from.tensor_pool, used.tensor_elems * sizeof(simpler::hbg::Tensor));
    std::memset(out + to.scalar_pool, 0, static_cast<size_t>(to.end - to.scalar_pool));
    const auto *input = reinterpret_cast<const ChipTaskStorage *>(source + from.storage);
    const auto *source_tensors = reinterpret_cast<const simpler::hbg::Tensor *>(source + from.tensor_pool);
    const auto *source_scalars = reinterpret_cast<const uint64_t *>(source + from.scalar_pool);
    const auto *source_fanins = reinterpret_cast<const int32_t *>(source + from.fanin_pool);
    auto *storage = reinterpret_cast<ChipTaskStorage *>(out + to.storage);
    auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(out + to.tensor_pool);
    auto *fanins = reinterpret_cast<int32_t *>(out + to.fanin_pool);
    auto *arguments = reinterpret_cast<DispatchPayload *>(out + to.scalar_pool);
    auto *boundary = reinterpret_cast<uint64_t *>(arguments + used.submitted_tasks);
    for (uint64_t i = 0; i < used.submitted_tasks; ++i) {
        const auto &src = input[i].payload;
        auto &dst = storage[i].payload;
        const bool graph = input[i].slot.task_kind == TaskKind::GRAPH;
        always_assert(src.tensor_count >= 0 && src.scalar_count >= 0 && src.fanin_count >= 0);
        always_assert(graph || (src.tensor_count <= MAX_TENSOR_ARGS && src.scalar_count <= MAX_SCALAR_ARGS));
        const auto *input_tensors = src.tensor_data();
        const auto *input_scalars = src.scalar_data();
        const auto *input_fanins = src.fanin_data();
        always_assert(
            src.tensor_count == 0 ||
            (input_tensors != nullptr && input_tensors >= source_tensors &&
             static_cast<uint64_t>(input_tensors - source_tensors) + src.tensor_count <= used.tensor_elems)
        );
        always_assert(
            src.scalar_count == 0 ||
            (input_scalars != nullptr && input_scalars >= source_scalars &&
             static_cast<uint64_t>(input_scalars - source_scalars) + src.scalar_count <= used.scalar_elems)
        );
        always_assert(
            src.fanin_count == 0 ||
            (input_fanins != nullptr && input_fanins >= source_fanins &&
             static_cast<uint64_t>(input_fanins - source_fanins) + src.fanin_count <= used.fanin_elems)
        );
        auto *task_tensors = src.tensor_count == 0 ? nullptr : tensors + (input_tensors - source_tensors);
        auto *task_scalars = graph ? boundary : arguments[i].args + src.tensor_count;
        dst.bind_regions(
            task_tensors, task_scalars, src.fanin_count == 0 ? nullptr : fanins + (input_fanins - source_fanins)
        );
        if (src.scalar_count != 0)
            std::memcpy(task_scalars, input_scalars, static_cast<size_t>(src.scalar_count) * sizeof(uint64_t));
        if (graph) boundary += src.scalar_count;
        else
            for (int32_t j = 0; j < src.tensor_count; ++j)
                arguments[i].args[j] =
                    device_base + static_cast<uint64_t>(reinterpret_cast<char *>(&task_tensors[j]) - out);
        storage[i].task.packed_buffer_base = reinterpret_cast<void *>(
            sm_layout::rebased_heap_addr(reinterpret_cast<uint64_t>(storage[i].task.packed_buffer_base), heap)
        );
        storage[i].task.packed_buffer_end = reinterpret_cast<void *>(
            sm_layout::rebased_heap_addr(reinterpret_cast<uint64_t>(storage[i].task.packed_buffer_end), heap)
        );
        dst.predicate.addr = sm_layout::rebased_heap_addr(dst.predicate.addr, heap);
    }
    for (uint64_t i = 0; i < used.tensor_elems; ++i) {
        const TaskId owner = tensors[i].owner_task_id;
        always_assert(!owner.is_valid() || owner.is_global());
        tensors[i].buffer.addr = sm_layout::rebased_heap_addr(tensors[i].buffer.addr, heap);
    }
    return to.end;
}
inline const ChipTaskStorage &task_at(const void *image, const sm_layout::SegmentOffsets &layout, uint64_t task) {
    return reinterpret_cast<const ChipTaskStorage *>(static_cast<const char *>(image) + layout.storage)[task];
}
inline const TaskPayload &payload(const void *image, const sm_layout::SegmentOffsets &layout, uint64_t task) {
    return task_at(image, layout, task).payload;
}
inline bool graph_storage_bytes(const Graph &definition, uint64_t *bytes) {
    GraphExecutionStorageLayout layout{};
    if (!graph_execution_storage_layout(
            definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, definition.edge_count,
            &layout, GraphArgumentStorage::KERNEL_ARGUMENTS
        ) ||
        layout.total_bytes > static_cast<size_t>(INT32_MAX))
        return false;
    *bytes = layout.total_bytes;
    return true;
}
}  // namespace task_image
