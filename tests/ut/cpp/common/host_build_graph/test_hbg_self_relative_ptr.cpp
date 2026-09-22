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
#include <vector>

#include "host_build_graph/runtime_types.h"

namespace {

using simpler::hbg::SelfRelativePtr;

// A block laid out the way the shared-memory image is: the referring field and
// its targets in one contiguous span, with a target on each side of the field so
// both delta signs are covered.
struct Block {
    TaskPayload before_payload;
    SelfRelativePtr<TaskPayload> to_before;
    SelfRelativePtr<TaskPayload> to_after;
    TaskPayload after_payload;
};

}  // namespace

TEST(HbgSelfRelativePtr, ZeroedMemoryReadsAsUnbound) {
    std::vector<std::byte> storage(sizeof(SelfRelativePtr<TaskPayload>), std::byte{0});
    auto *pointer = reinterpret_cast<SelfRelativePtr<TaskPayload> *>(storage.data());

    EXPECT_EQ(pointer->get(), nullptr);
    EXPECT_TRUE(*pointer == nullptr);
    EXPECT_FALSE(static_cast<bool>(*pointer));
}

TEST(HbgSelfRelativePtr, ResolvesTargetsOnBothSides) {
    Block block{};
    block.to_before.set(&block.before_payload);
    block.to_after.set(&block.after_payload);

    EXPECT_EQ(block.to_before.get(), &block.before_payload);
    EXPECT_EQ(block.to_after.get(), &block.after_payload);
    EXPECT_TRUE(block.to_before != nullptr);
    EXPECT_TRUE(static_cast<bool>(block.to_after));
}

TEST(HbgSelfRelativePtr, SetNullRebindsToUnbound) {
    Block block{};
    block.to_after.set(&block.after_payload);
    ASSERT_NE(block.to_after.get(), nullptr);

    block.to_after.set(nullptr);

    EXPECT_EQ(block.to_after.get(), nullptr);
    EXPECT_TRUE(block.to_after == nullptr);
}

// The reason the relocation pass is gone: a raw pointer written on the host names
// a host address the device would dereference verbatim, while a delta names the
// same *relative* position in whichever copy is being read.
TEST(HbgSelfRelativePtr, SurvivesABlockCopyToAnotherAddress) {
    // Two real objects rather than raw byte buffers: ChipTaskSlotState is
    // alignas(64), which std::vector<std::byte>::data() does not promise, and a
    // memcpy into untyped storage would not begin the object's lifetime.
    Block source{};
    Block destination{};
    Block *origin = &source;
    origin->to_before.set(&origin->before_payload);
    origin->to_after.set(&origin->after_payload);

    // A separate object, so the copy lands at an unrelated address the way the
    // device image does.
    std::memcpy(&destination, &source, sizeof(Block));
    Block *moved = &destination;
    ASSERT_NE(reinterpret_cast<void *>(moved), reinterpret_cast<void *>(origin));

    EXPECT_EQ(moved->to_before.get(), &moved->before_payload);
    EXPECT_EQ(moved->to_after.get(), &moved->after_payload);
}

// A task's three records reach each other by ChipTaskStorage's layout alone, so the
// resolution is against whichever copy of the storage the caller holds — nothing is
// stored, and an image copy needs no fix-up.
TEST(HbgSelfRelativePtr, SiblingAccessorsResolveWithinTheirOwnStorage) {
    // Real objects, not byte buffers: the storage is alignas(64), which
    // std::vector<std::byte>::data() does not promise.
    ChipTaskStorage source{};
    ChipTaskStorage destination{};

    EXPECT_EQ(&source.slot.to_payload(), &source.payload);
    EXPECT_EQ(&source.slot.to_descriptor(), &source.task);
    EXPECT_EQ(&source.task.to_slot(), &source.slot);
    EXPECT_EQ(&source.task.to_payload(), &source.payload);
    EXPECT_EQ(&source.payload.to_slot(), &source.slot);
    EXPECT_EQ(&source.payload.to_descriptor(), &source.task);

    std::memcpy(&destination, &source, sizeof(ChipTaskStorage));
    ASSERT_NE(reinterpret_cast<void *>(&destination), reinterpret_cast<void *>(&source));

    // The copy's records name the copy, not the original.
    EXPECT_EQ(&destination.slot.to_payload(), &destination.payload);
    EXPECT_EQ(&destination.slot.to_descriptor(), &destination.task);
    EXPECT_EQ(&destination.payload.to_slot(), &destination.slot);
}

// The descriptor and the one-cache-line slot state both occupy exactly the cache
// line ChipTaskStorage places them on, so the container's slot offset is the
// descriptor's size and its payload offset is twice that.
TEST(HbgSelfRelativePtr, KeepsTheSharedMemoryAbiSizes) {
    EXPECT_EQ(sizeof(ChipTaskSlotState), 64u);
    EXPECT_EQ(sizeof(TaskDescriptor), 64u);
    EXPECT_EQ(sizeof(SelfRelativePtr<simpler::hbg::Tensor>), 4u);
    EXPECT_EQ(offsetof(TaskDescriptor, packed_buffer_base), 24u);
    EXPECT_EQ(sizeof(ChipTaskStorage), 320u);
    EXPECT_EQ(offsetof(ChipTaskStorage, slot), sizeof(TaskDescriptor));
    EXPECT_EQ(offsetof(ChipTaskStorage, payload), offsetof(ChipTaskStorage, slot) + sizeof(ChipTaskSlotState));
}
