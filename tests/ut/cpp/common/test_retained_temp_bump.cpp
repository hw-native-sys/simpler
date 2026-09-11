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
// The retained temporary buffer's grow/slice logic, exercised against a fake
// HostApi that stands in for the platform's {addr, size} slot. The runtimes
// that stage through it are covered end to end elsewhere
// (test_trb_runtime_temp_buffer.cpp and the pipeline-slot scene tests); this
// file is about the mechanism alone.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "common/host_api.h"
#include "utils/retained_temp_bump.h"

namespace {

// Stands in for one pipeline slot on a DeviceRunner: a remembered {addr, size}
// plus counts of the allocator traffic the bump generates.
struct FakeSlot {
    void *addr = nullptr;
    size_t size = 0;
    int malloc_count = 0;
    int free_count = 0;
    bool malloc_fails = false;
    std::vector<void *> live;

    void release_all() {
        for (void *p : live) {
            std::free(p);
        }
        live.clear();
        addr = nullptr;
        size = 0;
    }
};

FakeSlot *g_slot = nullptr;

void *fake_device_malloc(void * /*runner_ctx*/, size_t size) {
    ++g_slot->malloc_count;
    if (g_slot->malloc_fails) {
        return nullptr;
    }
    // Deliberately NOT over-aligned. The sim backend's device_malloc is
    // std::malloc (src/common/platform/sim/host/memory_allocator.cpp), which
    // guarantees only max_align_t, so a fake that handed back 1024-aligned
    // memory would hide whether the bump aligns its own base.
    void *p = std::malloc(size);
    g_slot->live.push_back(p);
    return p;
}

void fake_device_free(void * /*runner_ctx*/, void *dev_ptr) {
    ++g_slot->free_count;
    for (size_t i = 0; i < g_slot->live.size(); ++i) {
        if (g_slot->live[i] == dev_ptr) {
            g_slot->live.erase(g_slot->live.begin() + static_cast<long>(i));
            break;
        }
    }
    std::free(dev_ptr);
}

void fake_get_retained_temp_buffer(void * /*runner_ctx*/, uint32_t /*pipeline_slot*/, void **addr, size_t *size) {
    if (addr != nullptr) *addr = g_slot->addr;
    if (size != nullptr) *size = g_slot->size;
}

void fake_set_retained_temp_buffer(void * /*runner_ctx*/, uint32_t /*pipeline_slot*/, void *addr, size_t size) {
    g_slot->addr = addr;
    g_slot->size = size;
}

const HostApiOps &fake_ops() {
    static const HostApiOps ops = []() {
        HostApiOps result{};
        result.device_malloc = fake_device_malloc;
        result.device_free = fake_device_free;
        result.get_retained_temp_buffer = fake_get_retained_temp_buffer;
        result.set_retained_temp_buffer = fake_set_retained_temp_buffer;
        return result;
    }();
    return ops;
}

class RetainedTempBumpTest : public ::testing::Test {
protected:
    void SetUp() override { g_slot = &slot_; }
    void TearDown() override {
        slot_.release_all();
        g_slot = nullptr;
    }

    FakeSlot slot_;
    HostApi api_{nullptr, 0, 0, &fake_ops()};
};

constexpr size_t kAlign = RetainedTempBump::kAlignment;

}  // namespace

TEST_F(RetainedTempBumpTest, AlignUpRoundsToTheSliceAlignment) {
    EXPECT_EQ(RetainedTempBump::align_up(0), 0u);
    EXPECT_EQ(RetainedTempBump::align_up(1), kAlign);
    EXPECT_EQ(RetainedTempBump::align_up(kAlign), kAlign);
    EXPECT_EQ(RetainedTempBump::align_up(kAlign + 1), 2 * kAlign);
}

TEST_F(RetainedTempBumpTest, FirstRunAllocatesAndPublishesTheSlot) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 3 * kAlign));

    EXPECT_EQ(slot_.malloc_count, 1);
    EXPECT_EQ(slot_.free_count, 0);
    EXPECT_NE(slot_.addr, nullptr);
    // The slot carries the raw allocation, which is over-sized by the headroom
    // begin() may spend aligning the base.
    EXPECT_EQ(slot_.size, 3 * kAlign + kAlign - 1);
    // What the caller is promised is usable bytes, not the raw size.
    EXPECT_GE(bump.capacity(), 3 * kAlign);
}

TEST_F(RetainedTempBumpTest, SlicesAreAlignedContiguousAndDisjoint) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 3 * kAlign));

    void *a = bump.acquire(10);
    void *b = bump.acquire(kAlign);
    void *c = bump.acquire(1);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);

    // Not slot_.addr: the base is aligned inside the raw allocation.
    EXPECT_GE(reinterpret_cast<uintptr_t>(a), reinterpret_cast<uintptr_t>(slot_.addr));
    EXPECT_LT(reinterpret_cast<uintptr_t>(a) - reinterpret_cast<uintptr_t>(slot_.addr), kAlign);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b) - reinterpret_cast<uintptr_t>(a), kAlign);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(c) - reinterpret_cast<uintptr_t>(a), 2 * kAlign);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a) % kAlign, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b) % kAlign, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(c) % kAlign, 0u);
}

TEST_F(RetainedTempBumpTest, ARunThatFitsReusesTheRetainedBufferWithNoAllocation) {
    RetainedTempBump first;
    ASSERT_TRUE(first.begin(&api_, 4 * kAlign));
    void *raw = slot_.addr;
    void *base = first.acquire(kAlign);
    ASSERT_NE(base, nullptr);

    // A later, smaller run keeps the larger buffer: the slot only grows.
    RetainedTempBump second;
    ASSERT_TRUE(second.begin(&api_, kAlign));
    EXPECT_EQ(slot_.malloc_count, 1);
    EXPECT_EQ(slot_.free_count, 0);
    EXPECT_EQ(slot_.addr, raw);
    EXPECT_EQ(slot_.size, 4 * kAlign + kAlign - 1);
    // ...and the cursor restarts, so the second run's first slice is the same
    // aligned base the first run got.
    EXPECT_EQ(second.acquire(kAlign), base);
}

TEST_F(RetainedTempBumpTest, ALargerRunGrowsTheSlotAndFreesTheOldBuffer) {
    RetainedTempBump first;
    ASSERT_TRUE(first.begin(&api_, kAlign));

    RetainedTempBump second;
    ASSERT_TRUE(second.begin(&api_, 8 * kAlign));
    EXPECT_EQ(slot_.malloc_count, 2);
    EXPECT_EQ(slot_.free_count, 1);
    EXPECT_EQ(slot_.size, 8 * kAlign + kAlign - 1);
    EXPECT_GE(second.capacity(), 8 * kAlign);
}

TEST_F(RetainedTempBumpTest, AcquireBeyondTheReservedSizeMisses) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 2 * kAlign));
    ASSERT_NE(bump.acquire(2 * kAlign), nullptr);

    // The run asked for 2*kAlign, so that much must slice; the headroom left
    // over from base alignment is not promised and is not relied on here.
    EXPECT_EQ(bump.next_offset(), 2 * kAlign);
    EXPECT_GE(bump.capacity(), 2 * kAlign);
    EXPECT_EQ(bump.acquire(bump.capacity()), nullptr);
}

TEST_F(RetainedTempBumpTest, SlicesAreAlignedEvenWhenTheBackendReturnsUnalignedMemory) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 3 * kAlign));

    // The fake allocator is plain malloc, so the retained allocation is almost
    // never 1024-aligned; every slice handed out must be regardless.
    ASSERT_NE(slot_.addr, nullptr);
    for (size_t bytes : {size_t{1}, kAlign, size_t{7}}) {
        void *p = bump.acquire(bytes);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % kAlign, 0u) << "slice " << p << " is not " << kAlign << "-aligned";
    }
    // The slot still names the raw allocation, not the aligned base: that is the
    // pointer device_free must receive at finalize.
    EXPECT_TRUE(std::find(slot_.live.begin(), slot_.live.end(), slot_.addr) != slot_.live.end());
}

TEST_F(RetainedTempBumpTest, AnOversizedRequestCannotWrapPastTheBuffer) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 2 * kAlign));
    ASSERT_NE(bump.acquire(kAlign), nullptr);

    // A byte count that overflowed upstream would make `offset + bytes` wrap to
    // a small value; the slice must be refused rather than handed out.
    EXPECT_EQ(bump.acquire(SIZE_MAX), nullptr);
    EXPECT_EQ(bump.acquire(SIZE_MAX - kAlign + 1), nullptr);
    // ...and the cursor is untouched by a refusal, so the next real slice fits.
    EXPECT_NE(bump.acquire(kAlign), nullptr);
}

TEST_F(RetainedTempBumpTest, ARunNeedingNothingLeavesTheSlotUntouched) {
    RetainedTempBump bump;
    ASSERT_TRUE(bump.begin(&api_, 0));

    EXPECT_EQ(slot_.malloc_count, 0);
    EXPECT_EQ(slot_.free_count, 0);
    EXPECT_EQ(slot_.addr, nullptr);
    // No buffer means nothing to slice at all. Callers skip empty tensors before
    // they get here, so this pins the class rather than a reachable path.
    EXPECT_EQ(bump.acquire(0), nullptr);
}

TEST_F(RetainedTempBumpTest, AFailedGrowClearsTheSlotRatherThanLeavingTheFreedBuffer) {
    RetainedTempBump first;
    ASSERT_TRUE(first.begin(&api_, kAlign));
    ASSERT_NE(slot_.addr, nullptr);

    slot_.malloc_fails = true;
    RetainedTempBump second;
    EXPECT_FALSE(second.begin(&api_, 8 * kAlign));

    // The old buffer was released to make room, so the slot must not keep
    // naming it — a later run would otherwise free it a second time.
    EXPECT_EQ(slot_.free_count, 1);
    EXPECT_EQ(slot_.addr, nullptr);
    EXPECT_EQ(slot_.size, 0u);
    EXPECT_EQ(second.acquire(kAlign), nullptr);
}
