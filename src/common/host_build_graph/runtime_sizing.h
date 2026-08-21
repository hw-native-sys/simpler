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
#include <limits>
#include <optional>

namespace hbg_runtime_sizing {

struct HbgRuntimeSizing {
    uint64_t task_capacity;
    uint64_t heap_size;
};

// RuntimeEnv is packed to byte alignment. Its first uint64_t happens to start
// at offset 24 in CallConfig, but the containing object itself is not guaranteed
// to be 8-byte aligned, so direct uint64_t loads would still be undefined.
inline uint64_t read_packed_override(const void *base) {
    if (base == nullptr) return 0;
    uint64_t value;
    std::memcpy(&value, base, sizeof(value));
    return value;
}

inline bool resolve(
    HbgRuntimeSizing defaults, std::optional<uint64_t> env_task_capacity, std::optional<uint64_t> env_heap,
    const void *task_capacity_override, const void *heap_override, HbgRuntimeSizing *out
) {
    if (out == nullptr) return false;

    HbgRuntimeSizing sizing = defaults;
    if (env_task_capacity.has_value()) sizing.task_capacity = *env_task_capacity;
    if (env_heap.has_value()) sizing.heap_size = *env_heap;

    const uint64_t packed_task_capacity = read_packed_override(task_capacity_override);
    const uint64_t packed_heap_size = read_packed_override(heap_override);
    if (packed_task_capacity != 0) sizing.task_capacity = packed_task_capacity;
    if (packed_heap_size != 0) sizing.heap_size = packed_heap_size;

    *out = sizing;
    const bool task_capacity_valid =
        sizing.task_capacity >= 4 &&
        sizing.task_capacity <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) &&
        (sizing.task_capacity & (sizing.task_capacity - 1)) == 0;
    return task_capacity_valid && sizing.heap_size >= 1024;
}

}  // namespace hbg_runtime_sizing
