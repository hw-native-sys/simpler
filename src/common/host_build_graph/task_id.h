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

// PTO_DEVICE_FUNC, which marks the accessors AICore is allowed to call. ccec
// qualifies every function with an execution location and defaults to [host], so
// an unmarked member is rejected in an [aicore] function even when it only reads
// a field.
#include "data_type.h"

namespace simpler::hbg {

/**
 * TaskId: a 64-bit task handle, `(id space << 62) | (parent << 32) | local id`.
 *
 * What every holder may rely on regardless of layout: the handle is 8 bytes,
 * copyable, comparable for identity, and has one reserved sentinel.
 *
 * Invalid sentinel: raw == UINT64_MAX. Its top two bits read as id space 3, which
 * no real space takes — the static_assert below is what keeps that true.
 */
class TaskId {
public:
    // A default-constructed handle is the invalid sentinel. The task table and the
    // Definition image are allocated as arrays, so a default-constructible slot is
    // required, and a slot no factory has filled yet must not read as a live task --
    // zero would, being a legitimate GLOBAL/local-0 mint.
    //
    // What the initializer costs is trivial default construction: every
    // default-constructed handle is written, so an array of them is no longer free to
    // declare. Trivially copyable and standard layout are untouched, which is what the
    // memcpy boundary needs. A hot-path array therefore belongs in storage that is
    // constructed once -- the scheduler's per-thread staging scratch, not a stack frame
    // the dispatch path re-enters.
    TaskId() = default;

    // What the templates below check their operands against. A handle keeps its
    // address space through template deduction, so `__gm__ TaskId` and `TaskId`
    // arrive as distinct types that is_same_v cannot relate and no cv-stripping
    // reaches -- this marker is what tells them apart from any other type that
    // happens to carry a `raw` member.
    static constexpr bool kIsTaskIdHandle = true;

    /**
     * Which id space a task id belongs to, held in the top two bits.
     *
     * Everything this runtime schedules is a task. The space says whether a task
     * belongs to another task or stands on its own, which is also what decides
     * where its id resolves:
     *
     *   GLOBAL   — a task of the run itself, holding a slot in the shared-memory task
     *              table. Its low field is the task allocator's local id, resolvable
     *              via get_slot_by_task_id().
     *   SUB_TASK — a task belonging to one modular task's body. It lives in that
     *              task's own storage, not in the task table, so its low field must
     *              never be resolved against a table slot. `parent_id()` names
     *              the modular task it belongs to.
     *   PARAM    — a formal parameter of a modular task's boundary, or a view derived
     *              from one. It names no task at all: it is the provenance mark a
     *              recording stamps on its boundary tensors so classification can tell
     *              a parameter from a body-local output and from an object that entered
     *              the body without passing through the boundary. Its low field is the
     *              parameter index and its parent field is zero, and nothing resolves
     *              it against a table slot.
     *
     * A SUB_TASK id is minted twice for the same task, in two disjoint scopes: once
     * per task the recorder records, and once per task a replay materializes, each
     * naming the modular task it is under. The two never meet — a recorded id lives
     * only in the recorder thread's private map and the body's own locals, and a
     * materialized task is addressed by index rather than looked up by id.
     */
    enum class Space : uint32_t { GLOBAL = 0, SUB_TASK = 1, PARAM = 2 };

    static_assert(
        static_cast<uint32_t>(Space::GLOBAL) < 3 && static_cast<uint32_t>(Space::SUB_TASK) < 3 &&
            static_cast<uint32_t>(Space::PARAM) < 3,
        "invalid() is UINT64_MAX, whose top two bits read as space 3: no real space may take that value, or a "
        "live id would compare equal to the sentinel"
    );

    static constexpr uint32_t SPACE_SHIFT = 62;
    static constexpr uint32_t PARENT_SHIFT = 32;
    // A parent is a GLOBAL task's table index, so the width of that field is what
    // bounds how many GLOBAL tasks a run may hold: resolve_graph_task_capacity caps
    // the task count at GLOBAL_TASK_MAX_NUM, and the mint below masks rather than
    // fails, so a larger count would truncate a parent silently. The bits between
    // this field and the space are reserved and read back as zero.
    static constexpr uint32_t PARENT_BITS = 20;
    static constexpr uint32_t PARENT_MASK = (1u << PARENT_BITS) - 1;
    static constexpr int32_t GLOBAL_TASK_MAX_NUM = static_cast<int32_t>(1u << PARENT_BITS);

    static constexpr TaskId invalid() { return TaskId(UINT64_MAX); }

    // A local id is signed throughout this runtime — it is a task-table index, and the
    // table, the task states and the fanin payload all address slots with int32_t.
    // The low field is therefore narrowed to uint32_t before it is widened into the raw
    // word, so a negative value cannot sign-extend over the fields above it.
    static constexpr TaskId make_global(int32_t local_id) {
        return TaskId(
            (static_cast<uint64_t>(Space::GLOBAL) << SPACE_SHIFT) |
            static_cast<uint64_t>(static_cast<uint32_t>(local_id))
        );
    }

    static constexpr TaskId make_sub_task(int32_t parent_id, int32_t local_id) {
        return TaskId(
            (static_cast<uint64_t>(Space::SUB_TASK) << SPACE_SHIFT) |
            (static_cast<uint64_t>(static_cast<uint32_t>(parent_id) & PARENT_MASK) << PARENT_SHIFT) |
            static_cast<uint64_t>(static_cast<uint32_t>(local_id))
        );
    }

    // A boundary parameter's parent field is zero: a parameter belongs to a boundary
    // rather than to any task, so there is no owning task to name beside it.
    static constexpr TaskId make_param(int32_t param_index) {
        return TaskId(
            (static_cast<uint64_t>(Space::PARAM) << SPACE_SHIFT) |
            static_cast<uint64_t>(static_cast<uint32_t>(param_index))
        );
    }

    constexpr bool is_valid() const { return raw_ != UINT64_MAX; }

    constexpr Space space() const { return static_cast<Space>(static_cast<uint32_t>(raw_ >> SPACE_SHIFT)); }

    // True exactly when this id names a slot in the shared-memory task table, which is
    // what every caller that is about to resolve one is really asking. A SUB_TASK or
    // PARAM id answers false and must not reach get_slot_by_task_id().
    constexpr bool is_global() const { return space() == Space::GLOBAL; }

    // This id's space as a name, for diagnostics. A bare enumerator value in an error
    // message makes the reader open this header to learn what "1" was; the name does
    // not. Returns a string literal, so it is safe on the AICPU.
    constexpr const char *space_name() const {
        switch (space()) {
        case Space::GLOBAL:
            return "GLOBAL";
        case Space::SUB_TASK:
            return "SUB_TASK";
        case Space::PARAM:
            return "PARAM";
        default:
            // Only the invalid sentinel reaches here: no mint produces space 3.
            return "INVALID";
        }
    }

    // The low 32 bits, which every space round-trips from its minting argument: a
    // task-table local id for a GLOBAL task, the in-body index for a SUB_TASK, the
    // parameter index for a PARAM. Callers that resolve a table slot must gate on
    // is_global() first.
    //
    // A SUB_TASK's low field is its index within one modular task's body, so it is NOT
    // unique across the modular tasks that replay one Definition — two of them hold the
    // same low field for their respective first sub-task. Identity comparisons must use
    // the whole encoded word, which the parent field keeps distinct.
    constexpr int32_t local_id() const { return static_cast<int32_t>(raw_ & 0xFFFFFFFFu); }

    // The modular task this id belongs to, meaningful for SUB_TASK only. Zero for the
    // other spaces, which name no owning task.
    constexpr int32_t parent_id() const { return static_cast<int32_t>((raw_ >> PARENT_SHIFT) & PARENT_MASK); }

    constexpr bool operator==(const TaskId &other) const { return raw_ == other.raw_; }
    constexpr bool operator!=(const TaskId &other) const { return raw_ != other.raw_; }

    // A total order, so a handle can key an ordered container or sort a list
    // without being turned into an integer first. It orders by the encoded word,
    // which puts id spaces in enumerator order and is otherwise arbitrary: the
    // order is consistent, not meaningful. Nothing may read domain significance
    // into "less than".
    constexpr bool operator<(const TaskId &other) const { return raw_ < other.raw_; }

    // The handle's bits as one integer, for the uses that cannot take the handle
    // itself: formatting it into a log line, spreading it over a hash table's
    // slots, and the records this runtime serializes as numbers. It reads no
    // field out of it -- everything that wants a field has an accessor above,
    // and everything that wants identity has operator==; this is not a way
    // around either.
    //
    // Static and templated rather than a const member because AICore calls these
    // too, and ccec pins a member's implicit `this` to Local Memory -- a handle
    // in device global memory could not be reached through one. A template
    // parameter keeps the address space it was deduced from, so one definition
    // serves a handle wherever it lives.
    template <typename T>
    PTO_DEVICE_FUNC static constexpr uint64_t to_uint64(const T &id) {
        static_assert(T::kIsTaskIdHandle, "to_uint64() takes a TaskId handle");
        return id.raw_;
    }

    // Copy one handle onto another, in whatever pair of address spaces they
    // live: ccec cannot do it through operator=, which is a member and so has
    // the same pinned-`this` problem. Deduction covers all four combinations of
    // local and __gm__ with one definition.
    template <typename Dst, typename Src>
    PTO_DEVICE_FUNC static void assign(Dst &dst, const Src &src) {
        static_assert(Dst::kIsTaskIdHandle, "assign() takes a TaskId handle as its destination");
        static_assert(Src::kIsTaskIdHandle, "assign() takes a TaskId handle as its source");
        dst.raw_ = src.raw_;
    }

private:
    // The only way to turn a word into a handle, reachable from the factories
    // above and nowhere else. An arbitrary integer is not a task id, so nothing
    // outside this class may mint one from bits -- a caller that has the bits
    // and wants the handle back is reading a record whose field is already a
    // TaskId.
    constexpr explicit TaskId(uint64_t bits) :
        raw_(bits) {}

    // The sentinel is the default, so a slot that was never minted reads as invalid
    // rather than as a GLOBAL task 0. Only the factories and the private constructor
    // above ever set this to anything else.
    uint64_t raw_ = UINT64_MAX;
};

static_assert(
    std::is_trivially_copyable_v<TaskId> && std::is_standard_layout_v<TaskId>,
    "TaskId crosses the host-device boundary and must stay a POD wire type"
);
static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");

}  // namespace simpler::hbg

// std::hash, so a handle can key an unordered container directly. Absent on
// AICore, which has no <functional> and no such container -- and where the
// handle is only ever read out of a dispatch payload.
#if !defined(__DAV_VEC__) && !defined(__DAV_CUBE__)
#include <functional>

template <>
struct std::hash<simpler::hbg::TaskId> {
    size_t operator()(const simpler::hbg::TaskId &id) const noexcept {
        return std::hash<uint64_t>{}(simpler::hbg::TaskId::to_uint64(id));
    }
};
#endif

// A translation unit includes only its own runtime's task_id.h, so the unqualified
// name names this type. Two of these declarations in one scope are ill-formed, which
// is what makes a build that reaches both runtimes fail here rather than silently
// pick one.
//
// The Tensor next door carries no such declaration and is spelled simpler::hbg::Tensor
// at its call sites. TaskId differs because generated kernels and orchestration sources
// emit the bare name, and codegen has no runtime to qualify it with.
using simpler::hbg::TaskId;
