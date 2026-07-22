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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pto_task_id.h"
#include "pto_types.h"

// Versioned independently from the runtime ABI so cached definitions are
// invalidated when Graph Execution materialization semantics change.
inline constexpr uint64_t PTO2_GRAPH_CACHE_SCHEMA_VERSION = 6;
inline constexpr uint32_t PTO2_GRAPH_MAX_TENSOR_ARGS = 32;
inline constexpr uint32_t PTO2_GRAPH_MAX_SCALAR_ARGS = 32;

struct PTO2GraphScopeResult {
    bool execute_block{true};
    bool recording{false};
    PTO2TaskId task_id{PTO2TaskId::invalid()};
};

using PTO2GraphSubmitResult = PTO2GraphScopeResult;

constexpr uint64_t pto2_graph_hash_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * 1099511628211ULL;
}

inline uint64_t pto2_graph_hash_bytes(uint64_t h, const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h = pto2_graph_hash_byte(h, p[i]);
    }
    return h;
}

constexpr uint64_t pto2_graph_const_hash_impl(const char *s, uint64_t h) {
    return (*s == '\0') ? h : pto2_graph_const_hash_impl(s + 1, pto2_graph_hash_byte(h, static_cast<uint8_t>(*s)));
}

constexpr uint64_t PTO2_GRAPH_KEY(const char *s) { return pto2_graph_const_hash_impl(s, 1469598103934665603ULL); }

inline bool rt_graph_args_cacheable(const L2TaskArgs &args) {
    if (args.has_error || args.tensor_count() > static_cast<int32_t>(PTO2_GRAPH_MAX_TENSOR_ARGS) ||
        args.scalar_count() > static_cast<int32_t>(PTO2_GRAPH_MAX_SCALAR_ARGS)) {
        return false;
    }
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        // A Graph boundary is caller-owned storage. Runtime-allocated
        // TensorCreateInfo outputs remain on the ordinary submit path.
        if (args.tag(i) == TensorArgType::OUTPUT) return false;
    }
    return true;
}

inline uint64_t rt_graph_make_key(uint64_t graph_id, const L2TaskArgs &args) {
    uint64_t h = 1469598103934665603ULL;
    h = pto2_graph_hash_bytes(h, &PTO2_GRAPH_CACHE_SCHEMA_VERSION, sizeof(PTO2_GRAPH_CACHE_SCHEMA_VERSION));
    h = pto2_graph_hash_bytes(h, &graph_id, sizeof(graph_id));
    int32_t tensor_count = args.tensor_count();
    int32_t scalar_count = args.scalar_count();
    h = pto2_graph_hash_bytes(h, &tensor_count, sizeof(tensor_count));
    h = pto2_graph_hash_bytes(h, &scalar_count, sizeof(scalar_count));
    for (int32_t i = 0; i < tensor_count; ++i) {
        const Tensor &tensor = args.tensor(i).ref();
        TensorArgType type = args.tag(i);
        h = pto2_graph_hash_bytes(h, &tensor.buffer.size, sizeof(tensor.buffer.size));
        h = pto2_graph_hash_bytes(h, &tensor.ndims, sizeof(tensor.ndims));
        h = pto2_graph_hash_bytes(h, &tensor.dtype, sizeof(tensor.dtype));
        h = pto2_graph_hash_bytes(h, &tensor.manual_dep, sizeof(tensor.manual_dep));
        h = pto2_graph_hash_bytes(h, &tensor.is_contiguous, sizeof(tensor.is_contiguous));
        h = pto2_graph_hash_bytes(h, &type, sizeof(type));
        h = pto2_graph_hash_bytes(h, tensor.shapes, sizeof(uint32_t) * tensor.ndims);
        h = pto2_graph_hash_bytes(h, tensor.strides, sizeof(uint32_t) * tensor.ndims);
    }
    return h;
}
