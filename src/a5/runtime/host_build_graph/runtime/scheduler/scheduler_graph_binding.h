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

#include "scheduler_graph.h"
#include "scheduler_memory.h"

enum class SchedulerGraphBindingError : uint32_t {
    NONE = 0,
    NODE,
    DEFINITION,
    INSTANCE,
    TENSOR,
    SCALAR,
    PREDICATE,
};

inline __aicore__ SchedulerErrorSite scheduler_graph_binding_error_site(SchedulerGraphBindingError error) {
    return static_cast<SchedulerErrorSite>(
        static_cast<uint64_t>(SchedulerErrorSite::GRAPH_BIND_NODE) + static_cast<uint32_t>(error) - 1
    );
}

static inline __aicore__ void scheduler_graph_argument_cache_lines(uint64_t address, uint64_t bytes) {
    for (uint64_t offset = 0; offset < bytes; offset += 64) {
        auto *line = reinterpret_cast<__gm__ void *>(address + offset);
        scheduler_invalidate_cache_line(line);
    }
}

static inline __aicore__ void scheduler_graph_arguments_cache(
    const __gm__ SchedulerGraphInstance &instance, const __gm__ SchedulerGraphChildPlan &child
) {
    scheduler_graph_argument_cache_lines(
        instance.args_base + child.tensors_offset, static_cast<uint64_t>(child.tensor_count) * TASK_IMAGE_TENSOR_STRIDE
    );
    scheduler_graph_argument_cache_lines(
        instance.args_base + child.scalars_offset, static_cast<uint64_t>(child.scalar_count) * sizeof(uint64_t)
    );
    if (child.predicate_offset != SCHEDULER_NATIVE_NO_PREDICATE)
        scheduler_graph_argument_cache_lines(instance.args_base + child.predicate_offset, 64);
    scheduler_cache_barrier();
}

static inline __aicore__ SchedulerGraphBindingError
scheduler_observe_graph_arguments(const SchedulerGraphAccess &graph, int64_t ref) {
    if (graph.native_header_offset == 0) return SchedulerGraphBindingError::NONE;
    SchedulerNativeNode node{};
    if (!scheduler_native_node_checked(graph, ref, &node)) return SchedulerGraphBindingError::NODE;
    if (node.binding->kind == SchedulerNodeKind::TOP_KERNEL) return SchedulerGraphBindingError::NONE;
    auto *child = scheduler_native_child_plan_checked(graph, node);
    if (node.binding->kind != SchedulerNodeKind::CHILD_KERNEL || child == nullptr ||
        (node.definition->args_bytes != 0 && (node.instance->args_base == 0 || node.instance->args_base % 64 != 0)) ||
        node.definition->args_bytes > UINT64_MAX - node.instance->args_base)
        return SchedulerGraphBindingError::INSTANCE;
    scheduler_graph_arguments_cache(*node.instance, *child);
    return SchedulerGraphBindingError::NONE;
}
