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

#include <atomic>
#include <cstdint>

#include "pto_graph_cache.h"
#include "pto_runtime2_types.h"
#include "task_args.h"

enum class PTO2GraphExecutionState : uint8_t {
    SUBMITTED = 0,
    MATERIALIZING = 1,
    PREPARED = 2,
    ACTIVE = 3,
    COMPLETED = 4,
};

enum class PTO2GraphMaterializeResult : uint8_t {
    INVALID = 0,
    BUSY = 1,
    PENDING = 2,
    PREPARED = 3,
};

constexpr int32_t PTO2_GRAPH_MATERIALIZE_SLICE_NODES = 4;

struct PTO2GraphTopologyView {
    int32_t edge_count{0};
    int32_t root_count{0};
    const uint32_t *fanout_offsets{nullptr};
    const uint16_t *fanout_indices{nullptr};
    const uint16_t *fanin_counts{nullptr};
    const uint16_t *root_indices{nullptr};
};

struct alignas(64) PTO2GraphNodeStorage {
    PTO2TaskDescriptor task;
    PTO2TaskPayload payload;
    PTO2TaskSlotState slot;
};

struct PTO2GraphExecution {
    std::atomic<PTO2GraphExecutionState> state{PTO2GraphExecutionState::SUBMITTED};
    std::atomic<uint8_t> activation_requested{0};
    // A prepare token owns at most one bounded slice; this claim protects the
    // cursor if another scheduler observes a requeued token concurrently.
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_nodes{0};
    std::atomic<int32_t> retired_nodes{0};
    int32_t node_count{0};
    int32_t node_capacity{0};
    int32_t materialized_nodes{0};
    // Static node contents survive in the bounded AICPU-local pool.  A block
    // with the same graph key can therefore patch only per-invocation fields.
    int32_t materialized_node_count{0};
    int32_t constructed_nodes{0};
    size_t allocation_bytes{0};
    size_t definition_capacity{0};
    uint64_t graph_key{0};
    uint64_t materialized_graph_key{0};
    bool definition_affine_reuse{false};
    PTO2TaskSlotState *outer_slot{nullptr};
    PTO2GraphNodeStorage *nodes{nullptr};
    PTO2GraphNodeStorage *node_storage{nullptr};
    void *definition_storage{nullptr};
    const void *graph_definition{nullptr};
    PTO2GraphTopologyView topology;
    ChipStorageTaskArgs args;
    PTO2GraphExecution *next{nullptr};
};

// Compact host-built object uploaded for one Graph task.  It contains the
// immutable Definition plus this invocation's TaskArgs, but deliberately no
// expanded node storage.  Scheduler workers localize it into an AICPU pool.
struct PTO2GraphSubmission {
    std::atomic<PTO2GraphExecution *> local_execution{nullptr};
    std::atomic<uint8_t> activation_requested{0};
    int32_t node_count{0};
    size_t allocation_bytes{0};
    size_t definition_capacity{0};
    uint64_t graph_key{0};
    PTO2TaskSlotState *outer_slot{nullptr};
    void *definition_storage{nullptr};
    const void *graph_definition{nullptr};
    ChipStorageTaskArgs args;
    PTO2GraphSubmission *next{nullptr};
    PTO2GraphSubmission *upload_next{nullptr};
};

PTO2GraphExecution *
pto2_graph_execution_create(int32_t node_count, uint64_t graph_key = 0, size_t definition_bytes = 0);
void pto2_graph_execution_discard(PTO2GraphExecution *execution);
void pto2_graph_execution_publish(PTO2GraphExecution *execution);
void pto2_graph_execution_collect_retired();

PTO2GraphSubmission *
pto2_graph_submission_create(int32_t node_count, uint64_t graph_key = 0, size_t definition_bytes = 0);
void pto2_graph_submission_discard(PTO2GraphSubmission *submission);
void pto2_graph_submission_release_uploaded(PTO2GraphSubmission *submission);
size_t pto2_graph_submission_allocation_bytes(const PTO2GraphSubmission *submission);
bool pto2_graph_submission_relocate_for_upload(
    PTO2GraphSubmission *submission, void *device_base, uintptr_t host_sm_begin, size_t host_sm_size, intptr_t sm_delta
);
bool pto2_graph_definition_relocate_for_upload(PTO2GraphSubmission *submission, void *device_base);
PTO2GraphExecution *pto2_graph_execution_localize(PTO2TaskSlotState &outer_slot);
PTO2GraphMaterializeResult pto2_graph_execution_materialize_slice(
    PTO2TaskSlotState &outer_slot, PTO2GraphExecution &execution, int32_t max_nodes,
    int32_t *nodes_materialized = nullptr
);

inline PTO2GraphExecution *pto2_graph_execution_from_task(const PTO2TaskDescriptor &task) {
    return static_cast<PTO2GraphExecution *>(task.graph_execution);
}

inline PTO2GraphSubmission *pto2_graph_submission_from_task(const PTO2TaskDescriptor &task) {
    return static_cast<PTO2GraphSubmission *>(task.graph_execution);
}

inline bool pto2_graph_execution_begin_materialize(PTO2GraphExecution &execution) {
    PTO2GraphExecutionState expected = PTO2GraphExecutionState::SUBMITTED;
    return execution.state.compare_exchange_strong(
        expected, PTO2GraphExecutionState::MATERIALIZING, std::memory_order_acq_rel, std::memory_order_acquire
    );
}

inline void pto2_graph_execution_publish_materialized(PTO2GraphExecution &execution) {
    execution.state.store(PTO2GraphExecutionState::PREPARED, std::memory_order_release);
}

inline bool pto2_graph_execution_activate(PTO2GraphExecution &execution) {
    PTO2GraphExecutionState expected = PTO2GraphExecutionState::PREPARED;
    return execution.state.compare_exchange_strong(
        expected, PTO2GraphExecutionState::ACTIVE, std::memory_order_acq_rel, std::memory_order_acquire
    );
}

inline bool pto2_graph_execution_complete_node(PTO2GraphExecution &execution) {
    return execution.remaining_nodes.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void pto2_graph_execution_mark_completed(PTO2GraphExecution &execution) {
    execution.state.store(PTO2GraphExecutionState::COMPLETED, std::memory_order_release);
}

inline void pto2_graph_execution_retire_node(PTO2GraphExecution &execution) {
    execution.retired_nodes.fetch_add(1, std::memory_order_release);
}
