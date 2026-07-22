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
#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "common/device_phase.h"
#include "pto_runtime2_types.h"
#include "pto_types.h"

namespace {

// --- Fixed-buffer layout ---------------------------------------------------

TEST(TaskTimingSlots, RecordLayout) {
    EXPECT_EQ(NUM_TASK_TIMING_SLOTS, 16);
    EXPECT_EQ(sizeof(TaskTimingRecord), 16u);
    EXPECT_EQ(TASK_TIMING_SLOT_NONE, -1);
}

TEST(TaskTimingSlots, TailFollowsPhaseRegion) {
    for (int threads : {1, 4, 24}) {
        // Tail begins exactly where the phase region ends.
        EXPECT_EQ(
            task_timing_tail_offset(threads),
            static_cast<size_t>(aicpu_phase_buffer_slots(threads)) * sizeof(AicpuPhaseRecord)
        );
        // Total = phase region + tail, both sized for `threads`.
        EXPECT_EQ(
            device_phase_buffer_bytes(threads),
            task_timing_tail_offset(threads) +
                static_cast<size_t>(task_timing_buffer_slots(threads)) * sizeof(TaskTimingRecord)
        );
    }
}

// --- Cross-thread min/max reduction ---------------------------------------

TEST(TaskTimingSlots, ReduceMinDispatchMaxFinish) {
    const int threads = 3;
    std::vector<TaskTimingRecord> buf(
        static_cast<size_t>(threads) * NUM_TASK_TIMING_SLOTS, TaskTimingRecord{kPhaseUnset, 0}
    );
    // Slot 0 tagged on all three threads: earliest dispatch on t1, latest finish on t2.
    buf[0 * NUM_TASK_TIMING_SLOTS + 0] = {100, 200};
    buf[1 * NUM_TASK_TIMING_SLOTS + 0] = {80, 210};
    buf[2 * NUM_TASK_TIMING_SLOTS + 0] = {120, 250};

    uint64_t dispatch[NUM_TASK_TIMING_SLOTS];
    uint64_t finish[NUM_TASK_TIMING_SLOTS];
    reduce_task_timing_slots(buf.data(), threads, dispatch, finish);

    EXPECT_EQ(dispatch[0], 80u);
    EXPECT_EQ(finish[0], 250u);
}

TEST(TaskTimingSlots, ReduceSkipsIncompleteAndUnset) {
    const int threads = 2;
    std::vector<TaskTimingRecord> buf(
        static_cast<size_t>(threads) * NUM_TASK_TIMING_SLOTS, TaskTimingRecord{kPhaseUnset, 0}
    );
    // Slot 1: dispatched but never finished (finish stays 0) -> incomplete.
    buf[0 * NUM_TASK_TIMING_SLOTS + 1] = {300, 0};
    // Slot 2: finish <= dispatch -> incomplete.
    buf[0 * NUM_TASK_TIMING_SLOTS + 2] = {500, 400};

    uint64_t dispatch[NUM_TASK_TIMING_SLOTS];
    uint64_t finish[NUM_TASK_TIMING_SLOTS];
    reduce_task_timing_slots(buf.data(), threads, dispatch, finish);

    for (int s : {1, 2, 7}) {  // 7 is fully untouched
        EXPECT_EQ(dispatch[s], kPhaseUnset) << "slot " << s;
        EXPECT_EQ(finish[s], 0u) << "slot " << s;
    }
}

// --- Host resolve: origin anchoring + fallback ----------------------------

namespace {
// Identity cycle->ns so assertions read in raw offsets.
uint64_t identity_ns(uint64_t c) { return c; }
}  // namespace

TEST(TaskTimingSlots, ResolveAnchorsToPhaseOrigin) {
    const int threads = 1;
    std::vector<TaskTimingRecord> buf(NUM_TASK_TIMING_SLOTS, TaskTimingRecord{kPhaseUnset, 0});
    buf[0] = {1000, 1500};  // dispatch, finish (absolute cycles)

    uint64_t disp[NUM_TASK_TIMING_SLOTS], fin[NUM_TASK_TIMING_SLOTS];
    resolve_task_timing_slots_ns(buf.data(), threads, /*phase_origin=*/900, identity_ns, disp, fin);

    EXPECT_EQ(disp[0], 100u);  // 1000 - 900
    EXPECT_EQ(fin[0], 600u);   // 1500 - 900
}

TEST(TaskTimingSlots, ResolveFallsBackToEarliestDispatchWhenNoPhaseOrigin) {
    // host_build_graph stamps no phases -> phase_origin == kPhaseUnset. Slots
    // must still resolve, anchored to the earliest tagged dispatch.
    const int threads = 1;
    std::vector<TaskTimingRecord> buf(NUM_TASK_TIMING_SLOTS, TaskTimingRecord{kPhaseUnset, 0});
    buf[0] = {2000, 2200};  // earliest dispatch (2000) becomes the origin
    buf[1] = {2100, 2500};

    uint64_t disp[NUM_TASK_TIMING_SLOTS], fin[NUM_TASK_TIMING_SLOTS];
    resolve_task_timing_slots_ns(buf.data(), threads, kPhaseUnset, identity_ns, disp, fin);

    EXPECT_EQ(disp[0], 0u);    // 2000 - 2000 (origin)
    EXPECT_EQ(fin[0], 200u);   // 2200 - 2000
    EXPECT_EQ(disp[1], 100u);  // 2100 - 2000
    EXPECT_EQ(fin[1], 500u);   // 2500 - 2000
}

TEST(TaskTimingSlots, ResolveZeroesIncompleteAndUntagged) {
    const int threads = 1;
    std::vector<TaskTimingRecord> buf(NUM_TASK_TIMING_SLOTS, TaskTimingRecord{kPhaseUnset, 0});
    buf[3] = {500, 0};  // dispatched, never finished -> incomplete

    uint64_t disp[NUM_TASK_TIMING_SLOTS], fin[NUM_TASK_TIMING_SLOTS];
    resolve_task_timing_slots_ns(buf.data(), threads, /*phase_origin=*/100, identity_ns, disp, fin);

    for (int s : {3, 5}) {  // 3 incomplete, 5 untouched
        EXPECT_EQ(disp[s], 0u) << "slot " << s;
        EXPECT_EQ(fin[s], 0u) << "slot " << s;
    }
}

// --- Descriptor stays 40B; timing tag now rides TaskAttrs ----------------

TEST(TaskTimingSlots, DescriptorDoesNotGrow) {
    // The timing tag moved off the descriptor onto TaskAttrs; the descriptor
    // (shared-memory ABI) keeps its size and packed_buffer_base offset.
    EXPECT_EQ(sizeof(PTO2TaskDescriptor), 40u);
    EXPECT_EQ(offsetof(PTO2TaskDescriptor, packed_buffer_base), 24u);
}

TEST(TaskTimingSlots, MarkersTimingRoundTrip) {
    EXPECT_EQ(sizeof(TaskAttrs), 1u);

    TaskAttrs m;
    // Untagged by default.
    EXPECT_FALSE(m.is_timed());
    EXPECT_EQ(m.timing_slot(), TASK_TIMING_SLOT_NONE);

    // Tagging endpoints of the valid range roundtrips through the 4-bit field.
    m.set_timing_slot(0);
    EXPECT_TRUE(m.is_timed());
    EXPECT_EQ(m.timing_slot(), 0);
    m.set_timing_slot(15);
    EXPECT_TRUE(m.is_timed());
    EXPECT_EQ(m.timing_slot(), 15);

    // A negative slot clears the tag back to untagged.
    m.set_timing_slot(TASK_TIMING_SLOT_NONE);
    EXPECT_FALSE(m.is_timed());
    EXPECT_EQ(m.timing_slot(), TASK_TIMING_SLOT_NONE);
}

TEST(TaskTimingSlots, MarkersFlagsIndependent) {
    // Each flag is independent and does not disturb the timing tag.
    TaskAttrs m;
    m.set_timing_slot(9);
    m.set_early_resolve(true);
    m.set_sync_start();
    m.set_predicate();
    EXPECT_TRUE(m.allow_early_resolve());
    EXPECT_TRUE(m.requires_sync_start());
    EXPECT_TRUE(m.has_predicate());
    EXPECT_EQ(m.timing_slot(), 9);

    m.set_early_resolve(false);
    EXPECT_FALSE(m.allow_early_resolve());
    EXPECT_TRUE(m.requires_sync_start());
    EXPECT_TRUE(m.has_predicate());
    EXPECT_EQ(m.timing_slot(), 9);
}

// --- L0TaskArgs setter: bounds + sentinel ---------------------------------

TEST(TaskTimingSlots, ArgDefaultsUntagged) {
    L0TaskArgs args;
    EXPECT_EQ(args.task_timing_slot(), TASK_TIMING_SLOT_NONE);
    EXPECT_FALSE(args.has_error);
}

TEST(TaskTimingSlots, ArgSetValidSlot) {
    L0TaskArgs args;
    args.set_task_timing_slot(0);
    EXPECT_EQ(args.task_timing_slot(), 0);
    args.set_task_timing_slot(15);
    EXPECT_EQ(args.task_timing_slot(), 15);
    EXPECT_FALSE(args.has_error);
}

TEST(TaskTimingSlots, ArgRejectsOutOfRange) {
    for (int bad : {-1, 16, 100}) {
        L0TaskArgs args;
        args.set_task_timing_slot(bad);
        EXPECT_TRUE(args.has_error) << "slot " << bad;
        // Rejected value must not be recorded.
        EXPECT_EQ(args.task_timing_slot(), TASK_TIMING_SLOT_NONE) << "slot " << bad;
    }
}

TEST(TaskTimingSlots, ArgClearResetsSlot) {
    L0TaskArgs args;
    args.set_task_timing_slot(5);
    args.clear();
    EXPECT_EQ(args.task_timing_slot(), TASK_TIMING_SLOT_NONE);
}

}  // namespace
