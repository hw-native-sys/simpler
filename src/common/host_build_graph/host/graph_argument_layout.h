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
#include <cstring>

#include "host_build_graph/graph_definition.h"
#include "host_build_graph/tensor.h"

inline bool graph_argument_layout_child(
    int32_t tensors, int32_t scalars, bool predicate, uint64_t *cursor, uint64_t *tensor_offset,
    uint64_t *scalar_offset, uint64_t *predicate_offset
) {
    if (cursor == nullptr || tensors < 0 || tensors > MAX_TENSOR_ARGS || scalars < 0 || scalars > MAX_SCALAR_ARGS)
        return false;
    const auto reserve = [&](uint64_t bytes, uint64_t *offset) {
        if (*cursor > UINT64_MAX - 63 || bytes > UINT64_MAX - 63) return false;
        const uint64_t begin = (*cursor + 63) & ~UINT64_C(63);
        const uint64_t size = (bytes + 63) & ~UINT64_C(63);
        if (size > UINT64_MAX - begin) return false;
        *offset = begin;
        *cursor = begin + size;
        return true;
    };
    return reserve(static_cast<uint64_t>(tensors) * sizeof(simpler::hbg::Tensor), tensor_offset) &&
           reserve(static_cast<uint64_t>(scalars) * sizeof(uint64_t), scalar_offset) &&
           (!predicate || reserve(64, predicate_offset));
}

inline bool graph_native_argument_storage_bytes(const GraphDefinition &definition, uint64_t *bytes) {
    if (bytes == nullptr || definition.task_count <= 0 || definition.task_count > SUB_TASK_MAX_NUM ||
        definition.off_sub_tasks > definition.total_bytes ||
        static_cast<uint64_t>(definition.task_count) >
            (definition.total_bytes - definition.off_sub_tasks) / sizeof(SubTaskDefinition))
        return false;
    uint64_t cursor = 0;
    for (int32_t i = 0; i < definition.task_count; ++i) {
        SubTaskDefinition task{};
        std::memcpy(
            &task,
            reinterpret_cast<const uint8_t *>(&definition) + definition.off_sub_tasks +
                static_cast<uint64_t>(i) * sizeof(task),
            sizeof(task)
        );
        uint64_t tensors = 0, scalars = 0, predicate = 0;
        if (!graph_argument_layout_child(
                task.tensor_count, task.scalar_count, TaskAttrs(task.task_attrs).has_predicate(), &cursor, &tensors,
                &scalars, &predicate
            ))
            return false;
    }
    *bytes = cursor;
    return true;
}
