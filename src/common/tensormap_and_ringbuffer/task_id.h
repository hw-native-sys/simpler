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
 * The `tmr` runtime's task handle and the layout it encodes into one.
 *
 * The layout is private to this runtime. `hbg` has its own TaskId, encoding an id
 * space in the same 64 bits (src/common/host_build_graph/task_id.h). The two are
 * distinct types in distinct namespaces, and a translation unit sees exactly one of
 * them. Nothing in the include path enforces that — src/common is on every target's
 * — so the trailing using-declaration does: reaching both headers from one scope is
 * a compile error, not a silent bind to whichever came first.
 */

#pragma once

#include <cstdint>
#include <type_traits>

// PTO_DEVICE_FUNC, which marks the accessors AICore is allowed to call. ccec
// qualifies every function with an execution location and defaults to [host], so
// an unmarked member is rejected in an [aicore] function even when it only reads
// a field.
#include "data_type.h"

namespace simpler::tmr {

/**
 * TaskId: a 64-bit task handle, `(ring_id << 32) | local_id`.
 *
 * ring_id:  which ring layer the task was placed on (0..CHIP_MAX_RING_DEPTH-1)
 * local_id: that ring's monotonic task counter
 *
 * Every task this runtime mints lives on a ring, so the pair fully identifies a
 * ring slot: `rings[id.ring()].get_slot_by_task_id(id.local_id())`.
 *
 * What every holder may rely on regardless of layout: the handle is 8 bytes,
 * copyable, comparable for identity, and has one reserved sentinel.
 *
 * Invalid sentinel: raw == UINT64_MAX. No valid task encodes it — a ring id is a
 * uint8_t, so bits 63-40 of a minted handle are always zero and raw stays below
 * 2^40 whatever the local counter reaches.
 */
class TaskId {
public:
    // A default-constructed handle is the invalid sentinel. The image's task tables
    // are allocated as arrays, so a default-constructible slot is required, and a slot
    // no factory has filled yet must not read as a live task -- zero would, being a
    // legitimate (ring 0, local 0) mint.
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
    // happens to carry a `raw_` member.
    static constexpr bool kIsTaskIdHandle = true;

    static constexpr TaskId invalid() { return TaskId(UINT64_MAX); }

    // A local id is signed throughout this runtime -- it is a ring's task counter, and
    // the allocator, the slot lookup and the reclaim head all carry it as int32_t. The
    // low field is narrowed to uint32_t before it is widened into the raw word, so a
    // negative counter cannot sign-extend over the ring id above it.
    static constexpr TaskId make(uint8_t ring_id, int32_t local_id) {
        return TaskId((static_cast<uint64_t>(ring_id) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(local_id)));
    }

    constexpr bool is_valid() const { return raw_ != UINT64_MAX; }

    constexpr uint8_t ring() const { return static_cast<uint8_t>(raw_ >> 32); }

    constexpr int32_t local_id() const { return static_cast<int32_t>(raw_ & 0xFFFFFFFFu); }

    constexpr bool operator==(const TaskId &other) const { return raw_ == other.raw_; }
    constexpr bool operator!=(const TaskId &other) const { return raw_ != other.raw_; }

    // A total order, so a handle can key an ordered container or sort a list
    // without being turned into an integer first. It orders by the encoded word,
    // which groups by ring and is otherwise arbitrary: the order is consistent,
    // not meaningful. Nothing may read domain significance into "less than".
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

    uint64_t raw_;
};

static_assert(
    std::is_trivially_copyable_v<TaskId> && std::is_standard_layout_v<TaskId>,
    "TaskId crosses the host-device boundary and must stay a POD wire type"
);
static_assert(sizeof(TaskId) == 8, "TaskId must stay 8 bytes (shared memory ABI)");

}  // namespace simpler::tmr

// std::hash, so a handle can key an unordered container directly. Absent on
// AICore, which has no <functional> and no such container -- and where the
// handle is only ever read out of a dispatch payload.
#if !defined(__DAV_VEC__) && !defined(__DAV_CUBE__)
#include <functional>

template <>
struct std::hash<simpler::tmr::TaskId> {
    size_t operator()(const simpler::tmr::TaskId &id) const noexcept {
        return std::hash<uint64_t>{}(simpler::tmr::TaskId::to_uint64(id));
    }
};
#endif

// A translation unit includes only its own runtime's task_id.h, so the unqualified
// name names this type. Two of these declarations in one scope are ill-formed, which
// is what makes a build that reaches both runtimes fail here rather than silently
// pick one.
//
// The Tensor next door carries no such declaration and is spelled simpler::tmr::Tensor
// at its call sites. TaskId differs because generated kernels and orchestration sources
// emit the bare name, and codegen has no runtime to qualify it with.
using simpler::tmr::TaskId;
