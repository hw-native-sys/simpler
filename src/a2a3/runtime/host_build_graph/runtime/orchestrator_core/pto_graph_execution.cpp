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
std::atomic_flag g_graph_execution_lock = ATOMIC_FLAG_INIT;

PTO2GraphSubmission *g_graph_submission_pool = nullptr;
size_t g_graph_submission_pool_bytes = 0;
int32_t g_graph_submission_pool_blocks = 0;

constexpr size_t PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES = 16ULL * 1024 * 1024;
constexpr int32_t PTO2_GRAPH_EXECUTION_POOL_MAX_BLOCKS = 64;
constexpr size_t PTO2_GRAPH_SUBMISSION_POOL_MAX_BYTES = 16ULL * 1024 * 1024;
constexpr int32_t PTO2_GRAPH_SUBMISSION_POOL_MAX_BLOCKS = 64;

struct GraphExecutionLockGuard {
    GraphExecutionLockGuard() {
        while (g_graph_execution_lock.test_and_set(std::memory_order_acquire)) {}
    }
    ~GraphExecutionLockGuard() { g_graph_execution_lock.clear(std::memory_order_release); }
};

bool checked_align_up(size_t value, size_t alignment, size_t *result) {
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
    *result = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool graph_execution_layout(
    int32_t node_capacity, size_t definition_capacity, size_t *nodes_offset, size_t *definition_offset,
    size_t *allocation_bytes
) {
    if (node_capacity <= 0) return false;
    if (static_cast<size_t>(node_capacity) > std::numeric_limits<size_t>::max() / sizeof(PTO2GraphNodeStorage)) {
        return false;
    }
    const size_t nodes_bytes = static_cast<size_t>(node_capacity) * sizeof(PTO2GraphNodeStorage);
    if (!checked_align_up(sizeof(PTO2GraphExecution), alignof(PTO2GraphNodeStorage), nodes_offset) ||
        *nodes_offset > std::numeric_limits<size_t>::max() - nodes_bytes ||
        !checked_align_up(*nodes_offset + nodes_bytes, alignof(Tensor), definition_offset) ||
        *definition_offset > std::numeric_limits<size_t>::max() - definition_capacity) {
        return false;
    }
    return checked_align_up(*definition_offset + definition_capacity, alignof(PTO2GraphNodeStorage), allocation_bytes);
}

bool graph_submission_layout(size_t definition_capacity, size_t *definition_offset, size_t *allocation_bytes) {
    if (definition_capacity == 0 ||
        !checked_align_up(sizeof(PTO2GraphSubmission), alignof(Tensor), definition_offset) ||
        *definition_offset > std::numeric_limits<size_t>::max() - definition_capacity) {
        return false;
    }
    return checked_align_up(*definition_offset + definition_capacity, alignof(Tensor), allocation_bytes);
}

void destroy_execution_nodes(PTO2GraphExecution *execution) {
    for (int32_t i = 0; i < execution->constructed_nodes; ++i) {
        execution->node_storage[i].~PTO2GraphNodeStorage();
    }
    execution->constructed_nodes = 0;
}

void reset_execution(PTO2GraphExecution *execution, int32_t node_count, uint64_t graph_key) {
    size_t nodes_offset = 0;
    size_t definition_offset = 0;
    size_t bytes = 0;
    if (!graph_execution_layout(
            execution->node_capacity, execution->definition_capacity, &nodes_offset, &definition_offset, &bytes
        )) {
        return;
    }
    execution->definition_affine_reuse = graph_key != 0 && execution->materialized_graph_key == graph_key &&
                                         execution->materialized_node_count == node_count &&
                                         execution->constructed_nodes >= node_count;
    execution->state.store(PTO2GraphExecutionState::SUBMITTED, std::memory_order_relaxed);
    execution->activation_requested.store(0, std::memory_order_relaxed);
    execution->materialize_busy.store(0, std::memory_order_relaxed);
    execution->remaining_nodes.store(node_count, std::memory_order_relaxed);
    execution->retired_nodes.store(0, std::memory_order_relaxed);
    execution->node_count = node_count;
    execution->materialized_nodes = 0;
    execution->allocation_bytes = bytes;
    execution->graph_key = graph_key;
    execution->outer_slot = nullptr;
    execution->nodes = nullptr;
    execution->node_storage =
        reinterpret_cast<PTO2GraphNodeStorage *>(reinterpret_cast<char *>(execution) + nodes_offset);
    execution->definition_storage = reinterpret_cast<char *>(execution) + definition_offset;
    if (!execution->definition_affine_reuse) {
        execution->graph_definition = nullptr;
        execution->topology = PTO2GraphTopologyView{};
    }
    execution->args.clear();
    execution->next = nullptr;
}

void destroy_execution(PTO2GraphExecution *execution) {
    if (execution == nullptr) return;
    destroy_execution_nodes(execution);
    execution->~PTO2GraphExecution();
    std::free(execution);
}

void recycle_execution_locked(PTO2GraphExecution *execution) {
    if (execution == nullptr) return;
    const size_t bytes = execution->allocation_bytes;
    if (bytes == 0 || bytes > PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES ||
        g_graph_execution_pool_blocks >= PTO2_GRAPH_EXECUTION_POOL_MAX_BLOCKS ||
        g_graph_execution_pool_bytes > PTO2_GRAPH_EXECUTION_POOL_MAX_BYTES - bytes) {
        destroy_execution(execution);
        return;
    }
    execution->outer_slot = nullptr;
    execution->nodes = nullptr;
    execution->next = g_graph_execution_pool;
    g_graph_execution_pool = execution;
    g_graph_execution_pool_bytes += bytes;
    g_graph_execution_pool_blocks++;
}

PTO2GraphExecution *take_pooled_execution_locked(int32_t node_count, uint64_t graph_key, size_t definition_bytes) {
    PTO2GraphExecution **best_link = nullptr;
    PTO2GraphExecution *best = nullptr;
    bool best_is_affine = false;
    PTO2GraphExecution **link = &g_graph_execution_pool;
    while (*link != nullptr) {
        PTO2GraphExecution *candidate = *link;
        if (candidate->node_capacity >= node_count && candidate->definition_capacity >= definition_bytes) {
            const bool candidate_is_affine = graph_key != 0 && candidate->materialized_graph_key == graph_key &&
                                             candidate->materialized_node_count == node_count &&
                                             candidate->constructed_nodes >= node_count;
            if (best == nullptr || (candidate_is_affine && !best_is_affine) ||
                (candidate_is_affine == best_is_affine && candidate->allocation_bytes < best->allocation_bytes)) {
                best = candidate;
                best_link = link;
                best_is_affine = candidate_is_affine;
            }
        }
        link = &candidate->next;
    }
    if (best == nullptr) return nullptr;
    *best_link = best->next;
    g_graph_execution_pool_bytes -= best->allocation_bytes;
    g_graph_execution_pool_blocks--;
    reset_execution(best, node_count, graph_key);
    return best;
}

void reset_submission(PTO2GraphSubmission *submission, int32_t node_count, uint64_t graph_key) {
    size_t definition_offset = 0;
    size_t bytes = 0;
    if (!graph_submission_layout(submission->definition_capacity, &definition_offset, &bytes)) return;
    submission->local_execution.store(nullptr, std::memory_order_relaxed);
    submission->activation_requested.store(0, std::memory_order_relaxed);
    submission->node_count = node_count;
    submission->allocation_bytes = bytes;
    submission->graph_key = graph_key;
    submission->outer_slot = nullptr;
    submission->definition_storage = reinterpret_cast<char *>(submission) + definition_offset;
    submission->graph_definition = nullptr;
    submission->args.clear();
    submission->next = nullptr;
    submission->upload_next = nullptr;
}

void recycle_submission(PTO2GraphSubmission *submission) {
    if (submission == nullptr) return;
    const size_t bytes = submission->allocation_bytes;
    if (bytes == 0 || bytes > PTO2_GRAPH_SUBMISSION_POOL_MAX_BYTES ||
        g_graph_submission_pool_blocks >= PTO2_GRAPH_SUBMISSION_POOL_MAX_BLOCKS ||
        g_graph_submission_pool_bytes > PTO2_GRAPH_SUBMISSION_POOL_MAX_BYTES - bytes) {
        submission->~PTO2GraphSubmission();
        std::free(submission);
        return;
    }
    submission->next = g_graph_submission_pool;
    submission->upload_next = nullptr;
    g_graph_submission_pool = submission;
    g_graph_submission_pool_bytes += bytes;
    g_graph_submission_pool_blocks++;
}

PTO2GraphSubmission *take_pooled_submission(int32_t node_count, uint64_t graph_key, size_t definition_bytes) {
    PTO2GraphSubmission **best_link = nullptr;
    PTO2GraphSubmission *best = nullptr;
    PTO2GraphSubmission **link = &g_graph_submission_pool;
    while (*link != nullptr) {
        PTO2GraphSubmission *candidate = *link;
        if (candidate->definition_capacity >= definition_bytes &&
            (best == nullptr || candidate->allocation_bytes < best->allocation_bytes)) {
            best = candidate;
            best_link = link;
        }
        link = &candidate->next;
    }
    if (best == nullptr) return nullptr;
    *best_link = best->next;
    g_graph_submission_pool_bytes -= best->allocation_bytes;
    g_graph_submission_pool_blocks--;
    reset_submission(best, node_count, graph_key);
    return best;
}

template <typename T>
bool relocate_graph_pointer(T *&ptr, uintptr_t host_begin, size_t bytes, intptr_t delta) {
    if (ptr == nullptr) return true;
    uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    if (value < host_begin || value >= host_begin + bytes) return false;
    ptr = reinterpret_cast<T *>(static_cast<intptr_t>(value) + delta);
    return true;
}

}  // namespace

PTO2GraphExecution *pto2_graph_execution_create(int32_t node_count, uint64_t graph_key, size_t definition_bytes) {
    if (node_count <= 0 || definition_bytes == 0) return nullptr;
    {
        GraphExecutionLockGuard guard;
        if (PTO2GraphExecution *pooled = take_pooled_execution_locked(node_count, graph_key, definition_bytes)) {
            return pooled;
        }
    }

    size_t nodes_offset = 0;
    size_t definition_offset = 0;
    size_t allocation_bytes = 0;
    if (!graph_execution_layout(node_count, definition_bytes, &nodes_offset, &definition_offset, &allocation_bytes)) {
        return nullptr;
    }
    void *storage = nullptr;
    if (::posix_memalign(&storage, alignof(PTO2GraphNodeStorage), allocation_bytes) != 0) return nullptr;
    auto *execution = new (storage) PTO2GraphExecution{};
    execution->node_capacity = node_count;
    execution->definition_capacity = definition_bytes;
    execution->allocation_bytes = allocation_bytes;
    reset_execution(execution, node_count, graph_key);
    return execution;
}

void pto2_graph_execution_discard(PTO2GraphExecution *execution) {
    GraphExecutionLockGuard guard;
    recycle_execution_locked(execution);
}

void pto2_graph_execution_publish(PTO2GraphExecution *execution) {
    if (execution == nullptr) return;
    GraphExecutionLockGuard guard;
    execution->next = g_graph_executions;
    g_graph_executions = execution;
}

void pto2_graph_execution_collect_retired() {
    GraphExecutionLockGuard guard;
    PTO2GraphExecution **link = &g_graph_executions;
    while (*link != nullptr) {
        PTO2GraphExecution *execution = *link;
        const bool completed = execution->state.load(std::memory_order_acquire) == PTO2GraphExecutionState::COMPLETED;
        const bool retired = execution->retired_nodes.load(std::memory_order_acquire) >= execution->node_count;
        if (!completed || !retired) {
            link = &execution->next;
            continue;
        }
        *link = execution->next;
        recycle_execution_locked(execution);
    }
}

PTO2GraphSubmission *pto2_graph_submission_create(int32_t node_count, uint64_t graph_key, size_t definition_bytes) {
    if (node_count <= 0 || definition_bytes == 0) return nullptr;
    if (PTO2GraphSubmission *pooled = take_pooled_submission(node_count, graph_key, definition_bytes)) return pooled;

    size_t definition_offset = 0;
    size_t allocation_bytes = 0;
    if (!graph_submission_layout(definition_bytes, &definition_offset, &allocation_bytes)) return nullptr;
    void *storage = nullptr;
    if (::posix_memalign(&storage, alignof(Tensor), allocation_bytes) != 0) return nullptr;
    auto *submission = new (storage) PTO2GraphSubmission{};
    submission->definition_capacity = definition_bytes;
    submission->allocation_bytes = allocation_bytes;
    reset_submission(submission, node_count, graph_key);
    return submission;
}

void pto2_graph_submission_discard(PTO2GraphSubmission *submission) { recycle_submission(submission); }

void pto2_graph_submission_release_uploaded(PTO2GraphSubmission *submission) { recycle_submission(submission); }

size_t pto2_graph_submission_allocation_bytes(const PTO2GraphSubmission *submission) {
    return submission == nullptr ? 0 : submission->allocation_bytes;
}

bool pto2_graph_submission_relocate_for_upload(
    PTO2GraphSubmission *submission, void *device_base, uintptr_t host_sm_begin, size_t host_sm_size, intptr_t sm_delta
) {
    if (submission == nullptr || device_base == nullptr || submission->allocation_bytes == 0) return false;
    const uintptr_t host_begin = reinterpret_cast<uintptr_t>(submission);
    const intptr_t graph_delta =
        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(device_base)) - static_cast<intptr_t>(host_begin);

    if (submission->outer_slot != nullptr) {
        const uintptr_t outer = reinterpret_cast<uintptr_t>(submission->outer_slot);
        if (outer < host_sm_begin || outer >= host_sm_begin + host_sm_size) return false;
        submission->outer_slot = reinterpret_cast<PTO2TaskSlotState *>(static_cast<intptr_t>(outer) + sm_delta);
    }
    if (!relocate_graph_pointer(
            submission->definition_storage, host_begin, submission->allocation_bytes, graph_delta
        )) {
        return false;
    }
    if (submission->graph_definition != nullptr) {
        auto *definition = const_cast<void *>(submission->graph_definition);
        if (!relocate_graph_pointer(definition, host_begin, submission->allocation_bytes, graph_delta)) return false;
        submission->graph_definition = definition;
    }
    submission->local_execution.store(nullptr, std::memory_order_relaxed);
    submission->next = nullptr;
    submission->upload_next = nullptr;
    return true;
}
