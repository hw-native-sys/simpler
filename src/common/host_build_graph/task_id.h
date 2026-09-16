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
/**
 * The `hbg` runtime's task handle and the layout it encodes into one.
 *
 * The layout is private to this runtime. `tmr` has its own TaskId, encoding a ring
 * slot in the same 64 bits (src/common/tensormap_and_ringbuffer/task_id.h). The two
 * are distinct types in distinct namespaces, and a translation unit sees exactly one
 * of them. Nothing in the include path enforces that — src/common is on every
 * target's — so the trailing using-declaration does: reaching both headers from one
 * scope is a compile error, not a silent bind to whichever came first.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace simpler::hbg {

/**
 * TaskId: a 64-bit task handle, `(id space << 32) | low field`.
 *
 * What every holder may rely on regardless of layout: the handle is 8 bytes,
 * copyable, comparable for identity, and has one reserved sentinel.
 *
 * Invalid sentinel: raw == UINT64_MAX. No valid task encodes it — the id space
 * occupies the high word and has no 0xFFFFFFFF member.
 */
struct TaskId {
    uint64_t raw;

    /**
     * Which id space a task id belongs to, held in the high 32 bits.
     *
     * Everything this runtime schedules is a task. The high bits say whether a task
     * belongs to a Graph task or stands on its own, which is also what decides where
     * its id resolves:
     *
     *   GLOBAL   — a task of the run itself, holding a slot in the shared-memory task
     *              table. Its low bits are the task allocator's local id, resolvable
     *              via get_slot_by_task_id().
     *   IN_GRAPH — a task belonging to one Graph task's body. It lives in that Graph's
     *              own storage, not in the task table, so its low bits are the packed
     *              pair below and must never be resolved against a table slot.
     *   PARAM    — a formal parameter of a Graph boundary, or a view derived from one.
     *              It names no task at all: it is the provenance mark a recording
     *              stamps on its boundary tensors so classification can tell a
     *              parameter from a body-local output and from an object that entered
     *              the body without passing through the boundary. Its low bits are the
     *              parameter index, and nothing resolves it against a table slot.
     *
     * An IN_GRAPH id is minted twice for the same task, in two disjoint scopes. The
     * recorder mints one per task it records, with `graph_task_id` fixed at 0: a
     * Definition is shared by every shell that replays it, so record time has no
     * single Graph task to name, and the low field is then the in-graph local id alone —
     * which is what keeps it inside the task chains the recording's hazard map is
     * dimensioned for. Materialize mints the other, with the replaying shell's real
     * local id. The two never meet: a recorded id lives only in the recorder thread's
     * private map and the body's own locals, and a materialized task is addressed by
     * index rather than looked up by id.
     */
    enum class Space : uint32_t { GLOBAL = 0, IN_GRAPH = 1, PARAM = 2 };

    // An in-graph task's low bits pack its Graph task's local id above the task's own
    // in-graph local id. graph_execution.h asserts MAX_IN_GRAPH_TASKS fits the low half.
    static constexpr uint32_t IN_GRAPH_LOCAL_ID_BITS = 10;

    static constexpr TaskId invalid() { return TaskId{UINT64_MAX}; }

    // A local id is signed throughout this runtime — it is a task-table index, and the
    // table, the task states and the fanin payload all address slots with int32_t.
    // The low field is therefore narrowed to uint32_t before it is widened into the raw
    // word, so a negative value cannot sign-extend over the id-space bits above it.
    static constexpr TaskId make_global(int32_t local_id) {
        return TaskId{
            (static_cast<uint64_t>(Space::GLOBAL) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(local_id))
        };
    }

    static constexpr TaskId make_in_graph(int32_t graph_task_id, int32_t in_graph_local_id) {
        const uint32_t packed =
            (static_cast<uint32_t>(graph_task_id) << IN_GRAPH_LOCAL_ID_BITS) | static_cast<uint32_t>(in_graph_local_id);
        return TaskId{(static_cast<uint64_t>(Space::IN_GRAPH) << 32) | static_cast<uint64_t>(packed)};
    }

    // A boundary parameter's low field is the parameter index alone, not the packed
    // pair above: a parameter belongs to a boundary rather than to any task, so there
    // is no owning Graph task to name beside it. local_id() therefore round-trips the
    // index this was minted with.
    static constexpr TaskId make_param(int32_t param_index) {
        return TaskId{
            (static_cast<uint64_t>(Space::PARAM) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(param_index))
        };
    }

    constexpr bool is_valid() const { return raw != UINT64_MAX; }

    constexpr Space space() const { return static_cast<Space>(static_cast<uint32_t>(raw >> 32)); }

    // True exactly when this id names a slot in the shared-memory task table, which is
    // what every caller that is about to resolve one is really asking. An IN_GRAPH or
    // PARAM id answers false and must not reach get_slot_by_task_id().
    constexpr bool is_global() const { return space() == Space::GLOBAL; }

    // The low 32 bits. A task-table local id for a GLOBAL task; the packed pair above
    // for an IN_GRAPH one, which is why callers that resolve a table slot must gate on
    // is_global() first. Both fit int32_t: a table index is bounded by the run's task
    // capacity, and a packed pair by MAX_IN_GRAPH_TASKS above that same capacity.
    //
    // GLOBAL and PARAM round-trip their minting argument. For an IN_GRAPH id the
    // in_graph_local_id given to make_in_graph() is the low IN_GRAPH_LOCAL_ID_BITS of
    // what this returns, not the whole of it.
    constexpr int32_t local_id() const { return static_cast<int32_t>(raw & 0xFFFFFFFFu); }

    constexpr bool operator==(const TaskId &other) const { return raw == other.raw; }
    constexpr bool operator!=(const TaskId &other) const { return raw != other.raw; }
};

static_assert(
    std::is_trivially_copyable_v<TaskId> && std::is_standard_layout_v<TaskId>,
    "TaskId crosses the host-device boundary and must stay a POD wire type"
);
static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");

}  // namespace simpler::hbg

// A translation unit includes only its own runtime's task_id.h, so the unqualified
// name names this type. Two of these declarations in one scope are ill-formed, which
// is what makes a build that reaches both runtimes fail here rather than silently
// pick one.
//
// The Tensor next door carries no such declaration and is spelled simpler::hbg::Tensor
// at its call sites. TaskId differs because generated kernels and orchestration sources
// emit the bare name, and codegen has no runtime to qualify it with.
using simpler::hbg::TaskId;
