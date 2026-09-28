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
 * The TaskId handle's field layout, and the handle surface its holders depend on
 * without reading a field: erasure to an integer is the encoded word, the order and
 * the hash are consistent with equality, and assign() copies the whole handle.
 *
 * The layout matters beyond this runtime. `tests/ut/py/tmr/test_task_id_layout.py`
 * greps the header for the ring shift so the Python decoder in
 * `simpler_setup/tools/tmr/task_id.py` cannot drift from it; that regex sees the
 * shift's spelling, not what a mint actually produces, which is what the round-trip
 * cases here cover.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <type_traits>
#include <unordered_set>

#include "tensormap_and_ringbuffer/task_id.h"

namespace {

using simpler::tmr::TaskId;

// A mint round-trips both of its fields. The ring sits above the low word, so one
// accessor answers for each without the other's value reaching it.
TEST(TmrTaskId, AMintRoundTripsItsRingAndLocalId) {
    EXPECT_EQ(TaskId::make(0, 0).ring(), 0u);
    EXPECT_EQ(TaskId::make(0, 0).local_id(), 0);

    EXPECT_EQ(TaskId::make(3, 12345).ring(), 3u);
    EXPECT_EQ(TaskId::make(3, 12345).local_id(), 12345);

    // The widest value each field holds: ring() is uint8_t, the width of its field,
    // and the local id occupies the whole low word.
    EXPECT_EQ(TaskId::make(255, 0).ring(), 255u);
    EXPECT_EQ(TaskId::make(0, INT32_MAX).local_id(), INT32_MAX);
    EXPECT_EQ(TaskId::make(255, INT32_MAX).ring(), 255u);
    EXPECT_EQ(TaskId::make(255, INT32_MAX).local_id(), INT32_MAX);
}

// The ring and the local id are independent: two mints sharing one field stay distinct
// through the other. A ring's task counter restarts per ring, so identity comparisons
// across rings rest on this.
TEST(TmrTaskId, TheTwoFieldsAreIndependent) {
    EXPECT_NE(TaskId::make(0, 7), TaskId::make(1, 7)) << "same counter on two rings names two tasks";
    EXPECT_NE(TaskId::make(1, 7), TaskId::make(1, 8));
    EXPECT_EQ(TaskId::make(1, 7), TaskId::make(1, 7));
}

// A local id is signed throughout this runtime — the allocator, the slot lookup and
// the reclaim head all carry the ring's task counter as int32_t. The factory narrows
// the low field to uint32_t before widening it, so a negative counter cannot
// sign-extend over the ring id above it.
TEST(TmrTaskId, ANegativeLocalIdDoesNotSignExtendOverTheRing) {
    const TaskId negative = TaskId::make(3, -1);

    EXPECT_EQ(negative.ring(), 3u) << "the ring must survive a negative counter beside it";
    EXPECT_EQ(negative.local_id(), -1);
    EXPECT_EQ(TaskId::to_uint64(negative), (static_cast<uint64_t>(3) << 32) | 0xFFFFFFFFu);
}

// The sentinel is UINT64_MAX, and no mint reaches it: ring() is uint8_t, so a minted
// word leaves bits 63-40 clear where the sentinel has them set. Its fields still read
// as (255, -1) — the same pair make(255, -1) reads back — so is_valid() is the test
// and a field's value never is.
TEST(TmrTaskId, TheInvalidSentinelIsDisjointFromEveryLiveMint) {
    EXPECT_FALSE(TaskId::invalid().is_valid());
    EXPECT_EQ(TaskId::to_uint64(TaskId::invalid()), UINT64_MAX)
        << "the sentinel's bit value is published into shared memory, where host reads it as a number";

    EXPECT_TRUE(TaskId::make(0, 0).is_valid());
    EXPECT_TRUE(TaskId::make(255, INT32_MAX).is_valid());

    // The widest pair, where a live value comes closest to the sentinel: its fields
    // read identically and the handles are still distinct.
    const TaskId widest = TaskId::make(255, -1);
    EXPECT_EQ(widest.ring(), TaskId::invalid().ring());
    EXPECT_EQ(widest.local_id(), TaskId::invalid().local_id());
    EXPECT_NE(widest, TaskId::invalid()) << "the bits above the ring field are what separate them";
    EXPECT_TRUE(widest.is_valid());
}

// A minted handle's top 24 bits are zero: ring() is uint8_t, so bits 63-40 are never
// written whatever the local counter reaches. The header's doc comment rests on this.
TEST(TmrTaskId, AMintLeavesTheTopBitsClear) {
    EXPECT_LT(TaskId::to_uint64(TaskId::make(255, INT32_MAX)), 1ull << 40);
    EXPECT_LT(TaskId::to_uint64(TaskId::make(255, -1)), 1ull << 40);
}

// The handle is 8 bytes and trivially copyable: it travels in device-copied structs,
// so this is a wire property, not an implementation detail.
TEST(TmrTaskId, TheHandleStaysAnEightBytePod) {
    static_assert(sizeof(TaskId) == 8);
    static_assert(std::is_trivially_copyable_v<TaskId>);
    static_assert(std::is_standard_layout_v<TaskId>);
    EXPECT_EQ(sizeof(TaskId), 8u);
}

// to_uint64() is the encoded word itself, not a digest of it: the DFX records this
// runtime serializes as numbers are read back by tools that decode the fields out of
// what was written, so anything other than the identity would make them disagree.
TEST(TmrTaskId, ErasureToAnIntegerIsTheEncodedWord) {
    const uint64_t word = TaskId::to_uint64(TaskId::make(3, 9));

    EXPECT_EQ(word >> 32, 3u);
    EXPECT_EQ(static_cast<int32_t>(word & 0xFFFFFFFFu), 9);
    EXPECT_NE(TaskId::to_uint64(TaskId::make(0, 9)), TaskId::to_uint64(TaskId::make(1, 9)));
}

// operator< orders by the encoded word. The order is consistent, not meaningful: a
// caller may sort or key an ordered container with it, and may read nothing about the
// tasks from which of two handles compares less.
TEST(TmrTaskId, LessThanOrdersByTheEncodedWord) {
    const TaskId lo = TaskId::make(0, 1);
    const TaskId hi = TaskId::make(0, 2);

    EXPECT_TRUE(lo < hi);
    EXPECT_FALSE(hi < lo);
    EXPECT_FALSE(lo < lo) << "a strict weak order is irreflexive, which std::sort and std::map both require";
    EXPECT_EQ(lo < hi, TaskId::to_uint64(lo) < TaskId::to_uint64(hi));

    // Handles group by ring because the ring field sits above the counter.
    EXPECT_TRUE(TaskId::make(0, INT32_MAX) < TaskId::make(1, 0));
    EXPECT_TRUE(TaskId::make(1, 0) < TaskId::invalid()) << "the sentinel is the maximum of the order";
}

// std::hash agrees with operator==, the invariant an unordered container rests on: two
// handles that compare equal must land in the same bucket. dep_gen replay's edge maps
// key on the handle directly because of it.
TEST(TmrTaskId, HashAgreesWithEquality) {
    const std::hash<TaskId> hash;

    EXPECT_EQ(hash(TaskId::make(3, 9)), hash(TaskId::make(3, 9)));
    EXPECT_EQ(hash(TaskId::invalid()), hash(TaskId::invalid()));

    std::unordered_set<TaskId> seen;
    EXPECT_TRUE(seen.insert(TaskId::make(1, 7)).second);
    EXPECT_FALSE(seen.insert(TaskId::make(1, 7)).second) << "an equal handle must be found, not added again";
    EXPECT_TRUE(seen.insert(TaskId::make(2, 7)).second) << "same counter, different ring -- a different task";
    EXPECT_EQ(seen.count(TaskId::make(1, 7)), 1u);
    EXPECT_EQ(seen.count(TaskId::make(1, 8)), 0u);
}

// assign() copies the whole handle. It exists because ccec pins a member's implicit
// `this` to Local Memory, so a __gm__ handle cannot be reached through operator=; the
// host build has no such constraint, which is what lets this case check the semantics
// the AICore instantiation relies on.
TEST(TmrTaskId, AssignCopiesTheWholeHandle) {
    const TaskId src = TaskId::make(3, 9);
    TaskId dst = TaskId::invalid();

    TaskId::assign(dst, src);

    EXPECT_EQ(dst, src);
    EXPECT_EQ(TaskId::to_uint64(dst), TaskId::to_uint64(src));
    EXPECT_EQ(dst.ring(), 3u);
    EXPECT_EQ(dst.local_id(), 9);

    // The sentinel travels like any other value; no field is treated specially.
    TaskId::assign(dst, TaskId::invalid());
    EXPECT_FALSE(dst.is_valid());
}

}  // namespace
