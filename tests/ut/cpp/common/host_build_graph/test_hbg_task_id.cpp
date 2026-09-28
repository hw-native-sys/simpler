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
 * The TaskId handle's field layout, and the two properties the rest of the runtime
 * rests on: every space round-trips what it was minted with, and a parent wider than
 * the field is masked rather than refused.
 *
 * The second is why resolve_graph_task_capacity holds ring_task_window to
 * GLOBAL_TASK_MAX_NUM. That guard cannot be reached through a bind -- a slot costs
 * kilobytes, so the shared-memory limit refuses such a count first -- so the
 * truncation it prevents is only observable here, at the mint.
 *
 * Then the handle's own surface, which holders depend on without reading a field:
 * erasure to an integer is the encoded word, the order and the hash are consistent
 * with equality, and assign() copies the whole handle.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <unordered_set>

#include "host_build_graph/task_id.h"

namespace {

using simpler::hbg::TaskId;

// Every space round-trips the local id it was minted with. This is the whole point of
// holding the space in the top two bits: the low word belongs to the local id alone,
// so one accessor answers for all three spaces.
TEST(HbgTaskId, EverySpaceRoundTripsItsLocalId) {
    EXPECT_EQ(TaskId::make_global(0).local_id(), 0);
    EXPECT_EQ(TaskId::make_global(12345).local_id(), 12345);
    EXPECT_EQ(TaskId::make_sub_task(7, 0).local_id(), 0);
    EXPECT_EQ(TaskId::make_sub_task(7, 1023).local_id(), 1023);
    EXPECT_EQ(TaskId::make_param(0).local_id(), 0);
    EXPECT_EQ(TaskId::make_param(63).local_id(), 63);

    EXPECT_EQ(TaskId::make_global(1).space(), TaskId::Space::GLOBAL);
    EXPECT_EQ(TaskId::make_sub_task(1, 1).space(), TaskId::Space::SUB_TASK);
    EXPECT_EQ(TaskId::make_param(1).space(), TaskId::Space::PARAM);

    EXPECT_TRUE(TaskId::make_global(1).is_global());
    EXPECT_FALSE(TaskId::make_sub_task(1, 1).is_global());
    EXPECT_FALSE(TaskId::make_param(1).is_global());
}

// Only a sub-task names an owning task; the other two spaces read back zero rather
// than whatever happened to sit in those bits.
TEST(HbgTaskId, OnlyASubTaskCarriesAParent) {
    EXPECT_EQ(TaskId::make_sub_task(0, 5).parent_id(), 0);
    EXPECT_EQ(TaskId::make_sub_task(9, 5).parent_id(), 9);
    EXPECT_EQ(TaskId::make_global(12345).parent_id(), 0);
    EXPECT_EQ(TaskId::make_param(63).parent_id(), 0);
}

// A sub-task's low field is its index within one body, so it is NOT unique across the
// modular tasks that replay one Definition. Identity therefore has to compare the whole
// handle, which the parent keeps distinct — the scheduler and the recording maps rely
// on this.
TEST(HbgTaskId, TwoBodiesFirstSubTasksShareALocalIdAndDifferAsHandles) {
    const TaskId first_of_a = TaskId::make_sub_task(3, 0);
    const TaskId first_of_b = TaskId::make_sub_task(4, 0);

    EXPECT_EQ(first_of_a.local_id(), first_of_b.local_id()) << "the low field alone cannot tell the two apart";
    EXPECT_NE(first_of_a, first_of_b) << "the whole handle must, or one body's task would resolve as another's";
    EXPECT_EQ(first_of_a, TaskId::make_sub_task(3, 0)) << "equal parents and indices are the same task";
}

// The widest parent the field holds survives; one past it is masked back to 0 instead
// of failing. A run whose task count was not capped would therefore mint a sub-task
// naming the wrong modular task, with no diagnostic anywhere — which is the reason
// resolve_graph_task_capacity rejects a ring_task_window above this bound.
TEST(HbgTaskId, AParentPastTheFieldIsMaskedNotRefused) {
    const int32_t widest = TaskId::GLOBAL_TASK_MAX_NUM - 1;
    EXPECT_EQ(TaskId::make_sub_task(widest, 0).parent_id(), widest) << "the cap itself must round-trip losslessly";

    EXPECT_EQ(TaskId::make_sub_task(TaskId::GLOBAL_TASK_MAX_NUM, 0).parent_id(), 0)
        << "one past the cap wraps silently; resolve_graph_task_capacity is what keeps a run from getting here";
    EXPECT_EQ(TaskId::make_sub_task(TaskId::GLOBAL_TASK_MAX_NUM + 3, 0).parent_id(), 3);

    // And the truncated handle collides with a real one, which is what makes the
    // silence dangerous rather than merely lossy.
    EXPECT_EQ(TaskId::make_sub_task(TaskId::GLOBAL_TASK_MAX_NUM, 0), TaskId::make_sub_task(0, 0));
}

// A local id is signed throughout this runtime, so the mint narrows it to uint32_t
// before widening. Without that, a negative value would sign-extend over the parent
// and the space, turning a GLOBAL task into some other space entirely.
TEST(HbgTaskId, ANegativeLocalIdDoesNotSignExtendOverTheFieldsAboveIt) {
    const TaskId negative_global = TaskId::make_global(-1);
    EXPECT_EQ(negative_global.space(), TaskId::Space::GLOBAL);
    EXPECT_EQ(negative_global.parent_id(), 0);
    EXPECT_EQ(negative_global.local_id(), -1);

    const TaskId negative_sub_task = TaskId::make_sub_task(5, -1);
    EXPECT_EQ(negative_sub_task.space(), TaskId::Space::SUB_TASK);
    EXPECT_EQ(negative_sub_task.parent_id(), 5) << "the parent must survive a negative local id beside it";
}

// The sentinel is UINT64_MAX, whose top two bits read as space 3. No real space takes
// that value — a static_assert in the header pins it — so no live handle can compare
// equal to it.
TEST(HbgTaskId, TheInvalidSentinelIsDisjointFromEverySpace) {
    EXPECT_FALSE(TaskId::invalid().is_valid());
    EXPECT_EQ(TaskId::to_uint64(TaskId::invalid()), UINT64_MAX)
        << "the sentinel's bit value is published into shared memory, where host reads it as a number";

    EXPECT_TRUE(TaskId::make_global(0).is_valid());
    EXPECT_TRUE(TaskId::make_sub_task(0, 0).is_valid());
    EXPECT_TRUE(TaskId::make_param(0).is_valid());

    // The extreme of each space, where a live value comes closest to the sentinel.
    EXPECT_NE(TaskId::make_global(-1), TaskId::invalid());
    EXPECT_NE(TaskId::make_sub_task(TaskId::GLOBAL_TASK_MAX_NUM - 1, -1), TaskId::invalid());
    EXPECT_NE(TaskId::make_param(-1), TaskId::invalid());
    EXPECT_TRUE(TaskId::make_sub_task(TaskId::GLOBAL_TASK_MAX_NUM - 1, -1).is_valid());
}

// A space's name is what diagnostics print, so it must not drift from the enumerator
// it describes: an error message saying "id space 1" makes the reader open the header.
TEST(HbgTaskId, EverySpaceNamesItself) {
    EXPECT_STREQ(TaskId::make_global(0).space_name(), "GLOBAL");
    EXPECT_STREQ(TaskId::make_sub_task(3, 0).space_name(), "SUB_TASK");
    EXPECT_STREQ(TaskId::make_param(0).space_name(), "PARAM");
    EXPECT_STREQ(TaskId::invalid().space_name(), "INVALID")
        << "the sentinel's space 3 belongs to no mint, and must not read as a real one";
}

// The handle is 8 bytes and trivially copyable: it travels in device-copied structs,
// so this is a wire property, not an implementation detail.
TEST(HbgTaskId, TheHandleStaysAnEightBytePod) {
    static_assert(sizeof(TaskId) == 8);
    static_assert(std::is_trivially_copyable_v<TaskId>);
    static_assert(std::is_standard_layout_v<TaskId>);
    EXPECT_EQ(sizeof(TaskId), 8u);
}

// to_uint64() is the encoded word itself, not a digest of it: the DFX records this
// runtime serializes as numbers are read back by tools that decode the fields out of
// what was written, so anything other than the identity would make them disagree.
TEST(HbgTaskId, ErasureToAnIntegerIsTheEncodedWord) {
    const TaskId sub_task = TaskId::make_sub_task(5, 9);
    const uint64_t word = TaskId::to_uint64(sub_task);

    EXPECT_EQ(word >> TaskId::SPACE_SHIFT, static_cast<uint64_t>(TaskId::Space::SUB_TASK));
    EXPECT_EQ((word >> TaskId::PARENT_SHIFT) & TaskId::PARENT_MASK, 5u);
    EXPECT_EQ(static_cast<int32_t>(word & 0xFFFFFFFFu), 9);

    // Distinct handles erase to distinct words, which is what lets a record hold the
    // number and a consumer recover the identity.
    EXPECT_NE(TaskId::to_uint64(TaskId::make_global(1)), TaskId::to_uint64(TaskId::make_param(1)));
    EXPECT_NE(TaskId::to_uint64(TaskId::make_sub_task(0, 1)), TaskId::to_uint64(TaskId::make_sub_task(1, 1)))
        << "two modular tasks share a low field, so only the parent keeps their first sub-tasks apart";
}

// operator< orders by the encoded word. The order is consistent, not meaningful: a
// caller may sort or key an ordered container with it, and may read nothing about the
// tasks from which of two handles compares less.
TEST(HbgTaskId, LessThanOrdersByTheEncodedWord) {
    const TaskId lo = TaskId::make_global(1);
    const TaskId hi = TaskId::make_global(2);

    EXPECT_TRUE(lo < hi);
    EXPECT_FALSE(hi < lo);
    EXPECT_FALSE(lo < lo) << "a strict weak order is irreflexive, which std::sort and std::map both require";
    EXPECT_EQ(lo < hi, TaskId::to_uint64(lo) < TaskId::to_uint64(hi));

    // The spaces land in enumerator order because the space field is the top two bits.
    EXPECT_TRUE(TaskId::make_global(0) < TaskId::make_sub_task(0, 0));
    EXPECT_TRUE(TaskId::make_sub_task(0, 0) < TaskId::make_param(0));
    EXPECT_TRUE(TaskId::make_param(-1) < TaskId::invalid()) << "the sentinel is the maximum of the order";
}

// std::hash agrees with operator==, the invariant an unordered container rests on: two
// handles that compare equal must land in the same bucket. dep_gen's edge maps key on
// the handle directly because of it.
TEST(HbgTaskId, HashAgreesWithEquality) {
    const std::hash<TaskId> hash;

    EXPECT_EQ(hash(TaskId::make_sub_task(5, 9)), hash(TaskId::make_sub_task(5, 9)));
    EXPECT_EQ(hash(TaskId::invalid()), hash(TaskId::invalid()));

    std::unordered_set<TaskId> seen;
    EXPECT_TRUE(seen.insert(TaskId::make_global(7)).second);
    EXPECT_FALSE(seen.insert(TaskId::make_global(7)).second) << "an equal handle must be found, not added again";
    EXPECT_TRUE(seen.insert(TaskId::make_param(7)).second) << "same low field, different space -- a different task";
    EXPECT_EQ(seen.count(TaskId::make_global(7)), 1u);
    EXPECT_EQ(seen.count(TaskId::make_global(8)), 0u);
}

// assign() copies the whole handle. It exists because ccec pins a member's implicit
// `this` to Local Memory, so a __gm__ handle cannot be reached through operator=; the
// host build has no such constraint, which is what lets this case check the semantics
// the AICore instantiation relies on.
TEST(HbgTaskId, AssignCopiesTheWholeHandle) {
    const TaskId src = TaskId::make_sub_task(5, 9);
    TaskId dst = TaskId::invalid();

    TaskId::assign(dst, src);

    EXPECT_EQ(dst, src);
    EXPECT_EQ(TaskId::to_uint64(dst), TaskId::to_uint64(src));
    EXPECT_EQ(dst.space(), TaskId::Space::SUB_TASK);
    EXPECT_EQ(dst.parent_id(), 5);
    EXPECT_EQ(dst.local_id(), 9);

    // The sentinel travels like any other value; no field is treated specially.
    TaskId::assign(dst, TaskId::invalid());
    EXPECT_FALSE(dst.is_valid());
}

}  // namespace
