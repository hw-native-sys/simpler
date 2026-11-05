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

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "host_build_graph/task_id.h"
#include "host_build_graph/runtime_types.h"
#include "tensor.h"

#include "graph_definition.h"
// A body's producers precede their consumers, so a fanin CSR row holds at most
// task_count - 1 entries, and both of the slot's row cursors index one of them.
// A sub-task's row has no cap of its own — unlike a GLOBAL task's inline row,
// which append_fanin_or_fail holds to CHIP_MAX_FANIN — so this is what bounds
// them, and a cursor too narrow for it would report a row scanned that was not.
static_assert(SUB_TASK_MAX_NUM - 1 < 0xFFFF, "a fanin CSR row index must fit ChipTaskSlotState::wake_scan_cursor");
static_assert(
    SUB_TASK_MAX_NUM - 1 < 0xFFFF && CHIP_MAX_FANIN - 1 < 0xFFFF,
    "a fanin row index from either cohort must fit ChipTaskSlotState::ed_publish_scan_cursor"
);
inline constexpr int32_t GRAPH_MATERIALIZE_SLICE_TASKS = 4;

enum class GraphExecutionPhase : uint8_t {
    SUBMITTED = 0,
    MATERIALIZED = 1,
    COMPLETED = 2,
};

enum class GraphMaterializeResult : uint8_t {
    INVALID = 0,
    BUSY = 1,
    PENDING = 2,
    MATERIALIZED = 3,
};

inline constexpr uint8_t GRAPH_EXECUTION_PHASE_MASK = 0x7;
inline constexpr uint8_t GRAPH_EXECUTION_EXTERNAL_READY = 0x8;

struct GraphExecution {
    // SUBMITTED includes both queued and partially materialized executions.
    // The phase and monotonic external-readiness flag share an atomic control byte.
    std::atomic<uint8_t> control{static_cast<uint8_t>(GraphExecutionPhase::SUBMITTED)};
    std::atomic<uint8_t> materialize_busy{0};
    std::atomic<int32_t> remaining_tasks{0};
    std::atomic<int32_t> retired_tasks{0};
    // Incremental activation: tasks in [0, published_tasks) are fully
    // materialized and registered, so a route pass may consider them. route_cursor
    // is the next such task index a route pass will claim; roots below it have
    // been pushed to the ready queue exactly once. Both advance monotonically and
    // reset per (re)submission.
    std::atomic<int32_t> published_tasks{0};
    std::atomic<int32_t> route_cursor{0};
    int32_t task_count{0};
    int32_t materialized_tasks{0};
    int32_t constructed_tasks{0};
    int32_t consumed_tensor_args{0};
    ChipTaskSlotState *outer_slot{nullptr};
    // Immutable after localization; published_tasks bounds access to constructed entries.
    ChipTaskStorage *task_storage{nullptr};
    // Polling-progress state, one ChipTaskState byte per sub-task, in the
    // storage tail. Carries the same PENDING -> PUBLISHED -> COMPLETED meaning
    // as the shared-memory task_states array a GLOBAL task uses, against
    // sub-task local ids instead of task-table slots, so both cohorts answer
    // readiness the same way.
    //
    // A byte array of its own rather than a field of ChipTaskStorage, for the
    // reason the top-level array gives: a fanin scan reads many producers'
    // states at once, which a couple of cache lines answer here and would take
    // one line per producer inside the ChipTaskStorage stride. It is also
    // read-mostly — each byte is written once per execution at completion —
    // so concurrent scanners share those lines rather than contend for them.
    std::atomic<ChipTaskState> *task_states{nullptr};
    // This execution's task argument pools, in the storage tail past task_storage.
    // Every task payload's tensor and scalar deltas point here; its pool position is
    // the Definition's tensor_offset / scalar_offset.
    simpler::hbg::Tensor *task_tensor_pool{nullptr};
    uint64_t *task_scalar_pool{nullptr};
    // Where this execution's Definition image starts inside the launch section,
    // not a pointer into it. The section belongs to whichever thread's launch
    // arguments carried it, and that thread may return while a peer is still
    // reading, so every reader resolves this offset against its own view.
    uint32_t definition_offset{0};
    // The fanin CSR, copied by value into this execution's own storage tail.
    // The dependency rows are read on every fanin scan, producer lookup, wait
    // registration and completion, so they are the one part of the image the
    // execution keeps: a borrowed pointer here would name a package this
    // execution outlives, and decoding per edge would put a bounded copy in
    // those loops.
    int32_t *fanin_offsets{nullptr};
    uint16_t *fanin_indices{nullptr};
    int32_t edge_count{0};
    // Base of the graph heap this execution was given. A body tensor's recorded offset is
    // relative to the recording's own output region, which is an affine image of this heap,
    // so the two added together are the real address.
    uintptr_t heap_base{0};
    // This invocation's actual arguments, held in the outer Graph task's own argument
    // pool. They are simpler::hbg::Tensors like every other task's, so a rebind that
    // resolves to one copies it across rather than converting: the boundary is the one
    // input that does not come from the image.
    const simpler::hbg::Tensor *boundary_tensors{nullptr};
    int32_t boundary_tensor_count{0};
    const uint64_t *boundary_scalars{nullptr};
    int32_t boundary_scalar_count{0};

    ChipTaskStorage &task_at(int32_t index) const { return task_storage[index]; }

    // Readiness accessors, named and ordered as SharedMemoryTaskHeader's so a
    // reader of one cohort reads the other the same way. The byte only ever
    // advances, so an index a scan has already cleared stays cleared.
    bool is_completed(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_COMPLETED;
    }

    void store_completed(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_COMPLETED, order);
    }

    bool is_published(int32_t index, std::memory_order order = std::memory_order_acquire) const {
        return task_states[index].load(order) >= CHIP_TASK_PUBLISHED;
    }

    void store_published(int32_t index, std::memory_order order = std::memory_order_release) const {
        task_states[index].store(CHIP_TASK_PUBLISHED, order);
    }

    void reset_task_state(int32_t index) const {
        task_states[index].store(CHIP_TASK_PENDING, std::memory_order_relaxed);
    }
};

static_assert(std::is_trivially_destructible_v<ChipTaskStorage>);
// The tensor pool starts right after the sub-task array, and the scalar pool starts
// after a whole number of ChipTensors.
static_assert(
    alignof(ChipTaskStorage) % alignof(simpler::hbg::Tensor) == 0,
    "a sub-task entry must be at least simpler::hbg::Tensor-aligned: the tensor pool follows the "
    "sub-task array"
);
static_assert(
    sizeof(simpler::hbg::Tensor) % alignof(uint64_t) == 0, "the tensor stride must keep the scalar pool aligned"
);
static_assert(std::is_trivially_destructible_v<GraphExecution>);
// The whole storage is aligned for its widest member, so one base check covers the
// header as well as the sub-task array that follows it.
static_assert(
    alignof(ChipTaskStorage) % alignof(GraphExecution) == 0,
    "the sub-task array's alignment must subsume the execution header's"
);

// The outer GRAPH task's heap tail occupies
// [GraphExecution][ChipTaskStorage x task_count][simpler::hbg::Tensor x tensor_arg_count]
// [uint64_t x scalar_arg_count][int32_t x (task_count + 1)][uint16_t x edge_count]
// [std::atomic<ChipTaskState> x task_count].
//
// The tensor and scalar regions are the sub-task payloads' argument pools, indexed by
// the Definition's own tensor_offset / scalar_offset — which is why the Definition's
// arg-table counts size them rather than a per-task sum: sub-task i's arguments
// occupy [offset, offset + count) in both the table and the pool.
//
// The two regions after them are this execution's own copy of the Definition's fanin
// CSR, rows then indices, filled by value at localize because the image they come from
// is the localizing thread's launch arguments and no peer may address it. A sub-task
// payload still carries no fanin region of its own: its dependencies are those rows, so
// its fanin_count stays 0 and its fanin delta unbound.
struct GraphExecutionStorageLayout {
    size_t tasks_offset;
    size_t tensors_offset;
    size_t scalars_offset;
    size_t fanin_offsets_offset;
    size_t fanin_indices_offset;
    size_t states_offset;
    size_t total_bytes;
};

inline bool graph_execution_storage_layout(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, int32_t edge_count,
    GraphExecutionStorageLayout *out
) {
    if (out == nullptr || task_count <= 0 || task_count > SUB_TASK_MAX_NUM || tensor_arg_count < 0 ||
        scalar_arg_count < 0 || edge_count < 0) {
        return false;
    }
    constexpr size_t ALIGNMENT = alignof(ChipTaskStorage);
    out->tasks_offset = (sizeof(GraphExecution) + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    out->tensors_offset = out->tasks_offset + static_cast<size_t>(task_count) * sizeof(ChipTaskStorage);
    out->scalars_offset = out->tensors_offset + static_cast<size_t>(tensor_arg_count) * sizeof(simpler::hbg::Tensor);
    // The fanin CSR follows the scalar pool, whose element is 8-byte aligned, so
    // the row array's 4 and the index array's 2 are already satisfied and no
    // padding is spent. Row count is task_count + 1, the same length the image
    // carries.
    out->fanin_offsets_offset = out->scalars_offset + static_cast<size_t>(scalar_arg_count) * sizeof(uint64_t);
    out->fanin_indices_offset = out->fanin_offsets_offset + (static_cast<size_t>(task_count) + 1) * sizeof(int32_t);
    // The state array is last because it is the one region with no alignment of
    // its own: a byte needs none, so appending it disturbs no other section's
    // offset. Every other region is entered through a typed pointer whose
    // alignment the base already guarantees.
    out->states_offset = out->fanin_indices_offset + static_cast<size_t>(edge_count) * sizeof(uint16_t);
    out->total_bytes = out->states_offset + static_cast<size_t>(task_count) * sizeof(std::atomic<ChipTaskState>);
    return true;
}

inline bool graph_execution_storage_bytes(
    int32_t task_count, int32_t tensor_arg_count, int32_t scalar_arg_count, int32_t edge_count, size_t *storage_bytes
) {
    GraphExecutionStorageLayout layout{};
    if (storage_bytes == nullptr ||
        !graph_execution_storage_layout(task_count, tensor_arg_count, scalar_arg_count, edge_count, &layout)) {
        return false;
    }
    *storage_bytes = layout.total_bytes;
    return true;
}

// Both take the calling thread's own view of the Definition section: the bytes
// live in that thread's launch arguments (or, in simulation, in the run-owned
// host snapshot), and no view is ever stored where another thread could read it
// after its owner returned. `graph_image_view.h` holds the view and its decoders.
struct GraphImageView;

GraphExecution *graph_execution_localize(ChipTaskSlotState &outer_slot, const GraphImageView &image);
GraphMaterializeResult graph_execution_materialize_slice(
    ChipTaskSlotState &outer_slot, GraphExecution &execution, const GraphImageView &image, int32_t max_tasks,
    int32_t *tasks_materialized = nullptr
);

// An outer GRAPH slot's graph_context is null until graph_execution_localize puts the
// execution there — the task names its Definition by descriptor offset, not by a
// context pointer — so this cast is only valid after that call. What makes it safe is
// the boot sequence, not this slot: every
// AICPU thread localizes a disjoint slice of the task window in classify_partition and
// all of them barrier before runtime_init_ready_ is published, so no dispatch — and
// therefore no caller of this function — observes an unlocalized GRAPH slot. A slot
// whose localization failed carries nullptr.
inline GraphExecution *graph_execution_from_outer_slot(ChipTaskSlotState &slot) {
    return slot.task_kind == TaskKind::GRAPH ? static_cast<GraphExecution *>(slot.graph_context) : nullptr;
}

inline GraphExecutionPhase
graph_execution_phase(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return static_cast<GraphExecutionPhase>(execution.control.load(order) & GRAPH_EXECUTION_PHASE_MASK);
}

inline bool
graph_execution_external_ready(const GraphExecution &execution, std::memory_order order = std::memory_order_acquire) {
    return (execution.control.load(order) & GRAPH_EXECUTION_EXTERNAL_READY) != 0;
}

inline void graph_execution_set_phase(
    GraphExecution &execution, GraphExecutionPhase next, std::memory_order order = std::memory_order_release
) {
    uint8_t observed = execution.control.load(std::memory_order_relaxed);
    // Masked, so a phase added past GRAPH_EXECUTION_PHASE_MASK cannot reach the
    // readiness bit sharing this byte.
    const uint8_t next_phase = static_cast<uint8_t>(static_cast<uint8_t>(next) & GRAPH_EXECUTION_PHASE_MASK);
    while (!execution.control.compare_exchange_weak(
        observed, static_cast<uint8_t>((observed & ~GRAPH_EXECUTION_PHASE_MASK) | next_phase), order,
        std::memory_order_relaxed
    )) {}
}

inline bool graph_execution_signal_external_ready(GraphExecution &execution) {
    return (execution.control.fetch_or(GRAPH_EXECUTION_EXTERNAL_READY, std::memory_order_acq_rel) &
            GRAPH_EXECUTION_EXTERNAL_READY) == 0;
}

inline bool graph_execution_complete_sub_task(GraphExecution &execution) {
    return execution.remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1;
}

inline void graph_execution_mark_completed(GraphExecution &execution) {
    graph_execution_set_phase(execution, GraphExecutionPhase::COMPLETED);
}

inline void graph_execution_retire_sub_task(GraphExecution &execution) {
    execution.retired_tasks.fetch_add(1, std::memory_order_release);
}
