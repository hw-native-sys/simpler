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
 * A slot state reaches its payload and descriptor through a delta from its own
 * address, which is what lets the shared-memory image be copied to the device
 * verbatim. The property the copy depends on is that the delta still resolves
 * after the whole block moves — that is what these tests pin.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "pto_runtime2_types.h"

namespace {

// A block laid out the way the shared-memory image is: the referring field and
// its targets in one contiguous span, with a target on each side of the field so
// both delta signs are covered.
struct Block {
    PTO2TaskPayload before_payload;
    PTO2RelativePtr<PTO2TaskPayload> to_before;
    PTO2RelativePtr<PTO2TaskPayload> to_after;
    PTO2TaskPayload after_payload;
};

}  // namespace

TEST(HbgSelfRelativePtr, ZeroedMemoryReadsAsUnbound) {
    PTO2RelativePtr<PTO2TaskPayload> pointer{};

    EXPECT_EQ(pointer.get(), nullptr);
    EXPECT_TRUE(pointer == nullptr);
    EXPECT_FALSE(static_cast<bool>(pointer));
}

TEST(HbgSelfRelativePtr, ResolvesTargetsOnBothSides) {
    Block block{};
    block.to_before.reset(&block.before_payload);
    block.to_after.reset(&block.after_payload);

    EXPECT_EQ(block.to_before.get(), &block.before_payload);
    EXPECT_EQ(block.to_after.get(), &block.after_payload);
    EXPECT_TRUE(block.to_before != nullptr);
    EXPECT_TRUE(static_cast<bool>(block.to_after));
}

TEST(HbgSelfRelativePtr, SetNullRebindsToUnbound) {
    Block block{};
    block.to_after.reset(&block.after_payload);
    ASSERT_NE(block.to_after.get(), nullptr);

    block.to_after.reset(nullptr);

    EXPECT_EQ(block.to_after.get(), nullptr);
    EXPECT_TRUE(block.to_after == nullptr);
}

// The reason the relocation pass is gone: a raw pointer written on the host names
// a host address the device would dereference verbatim, while a delta names the
// same *relative* position in whichever copy is being read.
TEST(HbgSelfRelativePtr, SurvivesABlockCopyToAnotherAddress) {
    // Two real objects rather than raw byte buffers: PTO2TaskSlotState is
    // alignas(64), which std::vector<std::byte>::data() does not promise, and a
    // memcpy into untyped storage would not begin the object's lifetime.
    Block source{};
    Block destination{};
    Block *origin = &source;
    origin->to_before.reset(&origin->before_payload);
    origin->to_after.reset(&origin->after_payload);

    // A separate object, so the copy lands at an unrelated address the way the
    // device image does.
    std::memcpy(&destination, &source, sizeof(Block));
    Block *moved = &destination;
    ASSERT_NE(reinterpret_cast<void *>(moved), reinterpret_cast<void *>(origin));

    EXPECT_EQ(moved->to_before.get(), &moved->before_payload);
    EXPECT_EQ(moved->to_after.get(), &moved->after_payload);
}

// bind_buffers is the only writer, and the slot state it writes into is copied as
// part of the same image as its payload and descriptor.
TEST(HbgSelfRelativePtr, SlotStateBindingSurvivesTheImageCopy) {
    struct Image {
        PTO2TaskDescriptor descriptor;
        PTO2TaskPayload payload;
        PTO2TaskSlotState slot;
    };

    // Real objects, not byte buffers: the slot state is alignas(64), which
    // std::vector<std::byte>::data() does not promise.
    Image source{};
    Image destination{};
    Image *origin = &source;
    origin->slot.bind_buffers(&origin->payload, &origin->descriptor);
    ASSERT_EQ(origin->slot.payload.get(), &origin->payload);
    ASSERT_EQ(origin->slot.task.get(), &origin->descriptor);

    std::memcpy(&destination, &source, sizeof(Image));
    Image *moved = &destination;
    ASSERT_NE(reinterpret_cast<void *>(moved), reinterpret_cast<void *>(origin));

    EXPECT_EQ(moved->slot.payload.get(), &moved->payload);
    EXPECT_EQ(moved->slot.task.get(), &moved->descriptor);
}

// The descriptor ABI AICore consumes and the one-cache-line slot state are both
// unchanged by the representation: the eight bytes the two deltas save become
// tail padding inside the same 64.
TEST(HbgSelfRelativePtr, KeepsTheSharedMemoryAbiSizes) {
    EXPECT_EQ(sizeof(PTO2TaskSlotState), 64u);
    EXPECT_EQ(sizeof(PTO2TaskDescriptor), 40u);
    EXPECT_EQ(sizeof(PTO2RelativePtr<PTO2TaskPayload>), 8u);
    EXPECT_EQ(offsetof(PTO2TaskDescriptor, packed_buffer_base), 24u);
}
