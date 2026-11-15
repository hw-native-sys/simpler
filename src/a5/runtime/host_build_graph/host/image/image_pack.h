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
#include <new>

#include "host_build_graph/shared_memory.h"
#include "host_build_graph/host/graph_argument_layout.h"
#include "task_image_types.h"
#include "../../aicpu/task_image_status.h"

namespace task_image {
inline constexpr bool native_graph_plan = true;

inline sm_layout::SegmentOffsets segments(const sm_layout::BindUsage &used) {
    sm_layout::SegmentOffsets out{};
    uint64_t cursor = sizeof(TaskImageStatus);
    out.storage = cursor;
    cursor +=
        CHIP_ALIGN_UP(sm_layout::live_slot_pitch(used.submitted_tasks) * sizeof(TaskImageStorage), CHIP_ALIGN_SIZE);
    out.fanin_pool = cursor;
    cursor += CHIP_ALIGN_UP(used.fanin_elems * sizeof(int32_t), CHIP_ALIGN_SIZE);
    out.tensor_pool = cursor;
    cursor += CHIP_ALIGN_UP(used.tensor_elems * sizeof(simpler::hbg::Tensor), CHIP_ALIGN_SIZE);
    out.scalar_pool = cursor;
    cursor += CHIP_ALIGN_UP(used.scalar_elems * sizeof(uint64_t), CHIP_ALIGN_SIZE);
    out.end = cursor;
    return out;
}

inline uint64_t pack(
    const char *source, uint64_t capacity, const sm_layout::BindUsage &used, const sm_layout::HeapRebase &heap,
    char *out
) {
    const auto limits = sm_layout::mirror_extents(capacity);
    always_assert(
        used.submitted_tasks <= capacity && used.fanin_elems <= limits.fanin_elems &&
        used.tensor_elems <= limits.tensor_elems && used.scalar_elems <= limits.scalar_elems
    );
    const auto from = sm_layout::segment_offsets(capacity);
    const auto to = segments(used);
    always_assert(to.end <= static_cast<uint64_t>(INT32_MAX));
    std::memset(out, 0, static_cast<size_t>(to.fanin_pool));
    new (out) TaskImageStatus{};
    const auto copy_pool = [&](uint64_t destination, uint64_t origin, uint64_t bytes) {
        std::memcpy(out + destination, source + origin, static_cast<size_t>(bytes));
        const uint64_t padding = CHIP_ALIGN_UP(bytes, CHIP_ALIGN_SIZE) - bytes;
        if (padding != 0) std::memset(out + destination + bytes, 0, static_cast<size_t>(padding));
    };
    copy_pool(to.fanin_pool, from.fanin_pool, used.fanin_elems * sizeof(int32_t));
    copy_pool(to.tensor_pool, from.tensor_pool, used.tensor_elems * sizeof(simpler::hbg::Tensor));
    copy_pool(to.scalar_pool, from.scalar_pool, used.scalar_elems * sizeof(uint64_t));
    const auto *source_storage = reinterpret_cast<const ChipTaskStorage *>(source + from.storage);
    const auto *source_tensors = reinterpret_cast<const simpler::hbg::Tensor *>(source + from.tensor_pool);
    const auto *source_scalars = reinterpret_cast<const uint64_t *>(source + from.scalar_pool);
    const auto *source_fanins = reinterpret_cast<const int32_t *>(source + from.fanin_pool);
    auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(out + to.tensor_pool);
    auto *scalars = reinterpret_cast<uint64_t *>(out + to.scalar_pool);
    auto *fanins = reinterpret_cast<int32_t *>(out + to.fanin_pool);
    auto *records = reinterpret_cast<TaskImageStorage *>(out + to.storage);
    for (uint64_t i = 0; i < used.submitted_tasks; ++i) {
        auto &record = *new (&records[i]) TaskImageStorage{};
        const auto &input = source_storage[i];
        const auto &payload = input.payload;
        record.task.task_id = input.task.task_id;
        for (uint32_t lane = 0; lane < 3; ++lane)
            record.task.kernel_id[lane] = input.task.kernel_id[lane];
        record.payload.tensor_count = payload.tensor_count;
        record.payload.scalar_count = payload.scalar_count;
        record.payload.fanin_count = payload.fanin_count;
        const auto *input_tensors = payload.tensor_data();
        const auto *input_scalars = payload.scalar_data();
        const auto *input_fanins = payload.fanin_data();
        always_assert(
            input_tensors == nullptr ||
            (input_tensors >= source_tensors && input_tensors <= source_tensors + used.tensor_elems &&
             static_cast<uint64_t>(payload.tensor_count) <= used.tensor_elems - (input_tensors - source_tensors))
        );
        always_assert(
            input_scalars == nullptr ||
            (input_scalars >= source_scalars && input_scalars <= source_scalars + used.scalar_elems &&
             static_cast<uint64_t>(payload.scalar_count) <= used.scalar_elems - (input_scalars - source_scalars))
        );
        always_assert(
            input_fanins == nullptr ||
            (input_fanins >= source_fanins && input_fanins <= source_fanins + used.fanin_elems &&
             static_cast<uint64_t>(payload.fanin_count) <= used.fanin_elems - (input_fanins - source_fanins))
        );
        record.payload.bind_regions(
            input_tensors == nullptr ? nullptr : tensors + (input_tensors - source_tensors),
            input_scalars == nullptr ? nullptr : scalars + (input_scalars - source_scalars),
            input_fanins == nullptr ? nullptr : fanins + (input_fanins - source_fanins)
        );
        record.payload.predicate.addr = sm_layout::rebased_heap_addr(payload.predicate.addr, heap);
        record.payload.predicate.target = payload.predicate.target;
        record.payload.predicate.elem_size = payload.predicate.elem_size;
        record.payload.predicate.op = static_cast<uint8_t>(payload.predicate.op);
    }
    for (uint64_t i = 0; i < used.tensor_elems; ++i) {
        const auto owner = tensors[i].owner_task_id;
        always_assert(!owner.is_valid() || owner.is_global());
        tensors[i].buffer.addr = sm_layout::rebased_heap_addr(tensors[i].buffer.addr, heap);
    }
    return to.end;
}

inline const TaskImagePayload &payload(const void *image, const sm_layout::SegmentOffsets &layout, uint64_t task) {
    return (reinterpret_cast<const TaskImageStorage *>(static_cast<const char *>(image) + layout.storage) + task)
        ->payload;
}

inline bool graph_storage_bytes(const Graph &definition, uint64_t *bytes) {
    return graph_native_argument_storage_bytes(definition, bytes);
}
}  // namespace task_image
