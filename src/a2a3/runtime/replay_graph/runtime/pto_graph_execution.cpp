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

#include "pto_graph_execution.h"

#include <cstdlib>
#include <limits>
#include <new>

namespace {

PTO2GraphExecution *g_graph_executions = nullptr;
PTO2GraphExecution *g_graph_execution_pool = nullptr;
size_t g_graph_execution_pool_bytes = 0;
int32_t g_graph_execution_pool_blocks = 0;

constexpr size_t PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES = 16ULL * 1024 * 1024;
constexpr int32_t PTO2_GRAPH_EXECUTION_POOL_MAX_BLOCKS = 64;

bool checked_align_up(size_t value, size_t alignment, size_t *result) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
    *result = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool graph_execution_allocation_size(int32_t node_capacity, size_t *nodes_offset, size_t *allocation_bytes) {
    if (node_capacity <= 0) return false;
    const size_t node_count_size = static_cast<size_t>(node_capacity);
    if (node_count_size > std::numeric_limits<size_t>::max() / sizeof(PTO2GraphNodeStorage)) return false;
    const size_t nodes_bytes = node_count_size * sizeof(PTO2GraphNodeStorage);
    if (!checked_align_up(sizeof(PTO2GraphExecution), alignof(PTO2GraphNodeStorage), nodes_offset) ||
        *nodes_offset > std::numeric_limits<size_t>::max() - nodes_bytes) {
        return false;
    }
    return checked_align_up(*nodes_offset + nodes_bytes, alignof(PTO2GraphNodeStorage), allocation_bytes);
}

size_t graph_execution_allocation_size(const PTO2GraphExecution &execution) {
    size_t nodes_offset = 0;
    size_t allocation_bytes = 0;
    bool valid = graph_execution_allocation_size(execution.node_capacity, &nodes_offset, &allocation_bytes);
    return valid ? allocation_bytes : 0;
}

void release_execution_definition(PTO2GraphExecution *execution) {
    execution->materialized_nodes = 0;
    if (execution->definition_refcount != nullptr) {
        execution->definition_refcount->fetch_sub(1, std::memory_order_acq_rel);
        execution->definition_refcount = nullptr;
    }
}

void destroy_execution_nodes(PTO2GraphExecution *execution) {
    for (int32_t i = 0; i < execution->constructed_nodes; ++i) {
        execution->node_storage[i].~PTO2GraphNodeStorage();
    }
    execution->constructed_nodes = 0;
}

void reset_execution(PTO2GraphExecution *execution, int32_t node_count, uint64_t graph_key) {
    execution->state.store(PTO2GraphExecutionState::SUBMITTED, std::memory_order_relaxed);
    execution->activation_requested.store(0, std::memory_order_relaxed);
    execution->materialize_busy.store(0, std::memory_order_relaxed);
    execution->remaining_nodes.store(node_count, std::memory_order_relaxed);
    execution->retired_nodes.store(0, std::memory_order_relaxed);
    execution->node_count = node_count;
    execution->materialized_nodes = 0;
    execution->graph_key = graph_key;
    execution->definition_affine_reuse = graph_key != 0 && execution->materialized_graph_key == graph_key &&
                                         execution->materialized_node_count == node_count &&
                                         execution->constructed_nodes >= node_count;
    execution->outer_slot = nullptr;
    execution->nodes = nullptr;
    execution->graph_definition = nullptr;
    execution->definition_refcount = nullptr;
    execution->topology = PTO2GraphTopologyView{};
    execution->args.clear();
    execution->next = nullptr;
}

void destroy_execution(PTO2GraphExecution *execution) {
    if (execution == nullptr) return;
    release_execution_definition(execution);
    destroy_execution_nodes(execution);
    execution->~PTO2GraphExecution();
    std::free(execution);
}

void recycle_execution(PTO2GraphExecution *execution) {
    release_execution_definition(execution);
    size_t allocation_bytes = graph_execution_allocation_size(*execution);
    if (allocation_bytes == 0 || allocation_bytes > PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES ||
        g_graph_execution_pool_blocks >= PTO2_GRAPH_EXECUTION_POOL_MAX_BLOCKS ||
        g_graph_execution_pool_bytes > PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES - allocation_bytes) {
        destroy_execution_nodes(execution);
        execution->~PTO2GraphExecution();
        std::free(execution);
        return;
    }
    execution->next = g_graph_execution_pool;
    g_graph_execution_pool = execution;
    g_graph_execution_pool_bytes += allocation_bytes;
    g_graph_execution_pool_blocks++;
}

PTO2GraphExecution *take_pooled_execution(int32_t node_count, uint64_t graph_key) {
    PTO2GraphExecution **best_link = nullptr;
    PTO2GraphExecution *best = nullptr;
    bool best_is_affine = false;
    PTO2GraphExecution **link = &g_graph_execution_pool;
    while (*link != nullptr) {
        PTO2GraphExecution *candidate = *link;
        if (candidate->node_capacity >= node_count) {
            bool candidate_is_affine = graph_key != 0 && candidate->materialized_graph_key == graph_key &&
                                       candidate->materialized_node_count == node_count &&
                                       candidate->constructed_nodes >= node_count;
            if (best == nullptr || (candidate_is_affine && !best_is_affine) ||
                (candidate_is_affine == best_is_affine && candidate->node_capacity < best->node_capacity)) {
                best = candidate;
                best_link = link;
                best_is_affine = candidate_is_affine;
            }
        }
        link = &candidate->next;
    }
    if (best == nullptr) return nullptr;

    *best_link = best->next;
    g_graph_execution_pool_bytes -= graph_execution_allocation_size(*best);
    g_graph_execution_pool_blocks--;
    reset_execution(best, node_count, graph_key);
    return best;
}

}  // namespace

PTO2GraphExecution *pto2_graph_execution_create(int32_t node_count, uint64_t graph_key) {
    if (node_count <= 0) return nullptr;

    if (PTO2GraphExecution *pooled = take_pooled_execution(node_count, graph_key); pooled != nullptr) return pooled;

    size_t nodes_offset = 0;
    size_t allocation_bytes = 0;
    if (!graph_execution_allocation_size(node_count, &nodes_offset, &allocation_bytes)) return nullptr;

    void *storage = nullptr;
    if (::posix_memalign(&storage, alignof(PTO2GraphNodeStorage), allocation_bytes) != 0) return nullptr;

    auto *execution = new (storage) PTO2GraphExecution{};
    execution->node_capacity = node_count;
    execution->node_storage = reinterpret_cast<PTO2GraphNodeStorage *>(static_cast<char *>(storage) + nodes_offset);
    reset_execution(execution, node_count, graph_key);
    return execution;
}

void pto2_graph_execution_publish(PTO2GraphExecution *execution) {
    if (execution == nullptr) return;
    execution->next = g_graph_executions;
    g_graph_executions = execution;
}

void pto2_graph_execution_discard(PTO2GraphExecution *execution) {
    if (execution != nullptr) recycle_execution(execution);
}

void pto2_graph_execution_collect_retired() {
    PTO2GraphExecution **link = &g_graph_executions;
    while (*link != nullptr) {
        PTO2GraphExecution *execution = *link;
        bool completed = execution->state.load(std::memory_order_acquire) == PTO2GraphExecutionState::COMPLETED;
        bool retired = execution->retired_nodes.load(std::memory_order_acquire) >= execution->node_count;
        if (!completed || !retired) {
            link = &execution->next;
            continue;
        }
        *link = execution->next;
        recycle_execution(execution);
    }
}

void pto2_graph_execution_destroy_all() {
    while (g_graph_executions != nullptr) {
        PTO2GraphExecution *execution = g_graph_executions;
        g_graph_executions = execution->next;
        destroy_execution(execution);
    }
    while (g_graph_execution_pool != nullptr) {
        PTO2GraphExecution *execution = g_graph_execution_pool;
        g_graph_execution_pool = execution->next;
        destroy_execution(execution);
    }
    g_graph_execution_pool_bytes = 0;
    g_graph_execution_pool_blocks = 0;
}
