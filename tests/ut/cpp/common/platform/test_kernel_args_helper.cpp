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
// Exercise the production onboard helper and allocator with host-backed RTS
// storage. The stub observes API boundaries; descriptor preparation stays real.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <utility>

#include "device_runner_helpers.h"

namespace {
// Byte every fresh device block is seeded with, standing in for what the device
// left there. A range still holding it was not copied into.
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int alloc_rc = 0;
    int copy_rc = 0;
    int copies = 0;
    int allocations = 0;
    int frees = 0;
    // What a publication is expected to carry. Which of the two prefixes that
    // is depends on the destination block, not on the case: the first
    // publication onto a block adds the handshake region, every later one stops
    // before it. The stub reads the slot rather than a fixed number so each
    // copy is checked against the state it was actually issued in.
    uint64_t initializing_copy_bytes = 0;
    uint64_t steady_copy_bytes = 0;
    const SlotPersistentArgs *destination = nullptr;
} rts;

class KernelArgsPublication : public ::testing::Test {
protected:
    void SetUp() override {
        rts = {};
        rts.initializing_copy_bytes = runtime_device_initialized_prefix_size(runtime);
        rts.steady_copy_bytes = runtime_device_copy_size(runtime);
        rts.destination = &slot;
    }
    void TearDown() override {
        EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
        EXPECT_EQ(allocator.get_allocation_count(), 0U);
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;

    int published_worker_count() const {
        DeviceRuntimeLaunchDesc descriptor;
        std::memcpy(&descriptor, slot.runtime_args, sizeof(descriptor));
        return descriptor.worker_count;
    }
};
}  // namespace

extern "C" rtError_t rtMalloc(void **ptr, uint64_t bytes, uint32_t, uint16_t) {
    if (rts.alloc_rc != 0) return rts.alloc_rc;
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) return -1;
    std::memset(*ptr, kDeviceMark, bytes);
    ++rts.allocations;
    return 0;
}
extern "C" rtError_t rtFree(void *ptr) {
    ++rts.frees;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    ++rts.copies;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    // A published prefix, not the allocated extent: a publication that carried
    // the whole descriptor would reach storage no host value belongs in. The
    // commit happens after this call returns, so the slot still shows the state
    // this copy was issued in.
    const bool initializing = rts.destination != nullptr && !rts.destination->workers_initialized;
    EXPECT_EQ(bytes, initializing ? rts.initializing_copy_bytes : rts.steady_copy_bytes);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0) return rts.copy_rc;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }

TEST_F(KernelArgsPublication, PrepareDoesNotPublishAndOwnsAnIndependentSnapshot) {
    EXPECT_FALSE(helper.runtime_args_published());
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_NE(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(slot.runtime_bytes, sizeof(DeviceRuntimeLaunchDesc));
    EXPECT_EQ(allocator.committed_bytes(), sizeof(DeviceRuntimeLaunchDesc));
    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    for (size_t i = 0; i < slot.runtime_bytes; ++i)
        ASSERT_EQ(bytes[i], kDeviceMark);

    runtime.dev.worker_count = 19;
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(helper.args.runtime_args, slot.runtime_args);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_NE(helper.publish_runtime_args(), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_TRUE(helper.runtime_args_published());
}

TEST_F(KernelArgsPublication, FailedPublicationWithdrawsRunViewButRetainsSlotForFreshPrepare) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    rts.copy_rc = -91;
    EXPECT_EQ(helper.publish_runtime_args(), -91);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
    EXPECT_EQ(rts.frees, 0);

    rts.copy_rc = 0;
    EXPECT_NE(helper.publish_runtime_args(), 0);
    EXPECT_EQ(rts.copies, 1);
    runtime.dev.worker_count = 3;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    EXPECT_EQ(published_worker_count(), 3);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_EQ(rts.allocations, 1);
}

TEST_F(KernelArgsPublication, AllocationFailureLeavesNoPublishableRun) {
    rts.alloc_rc = -92;
    EXPECT_NE(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_EQ(allocator.get_allocation_count(), 0U);
    EXPECT_NE(helper.publish_runtime_args(), 0);
    EXPECT_EQ(rts.copies, 0);
}

TEST_F(KernelArgsPublication, RepreparePreservesThePendingSnapshotAndDestination) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    runtime.dev.worker_count = 19;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(helper.args.runtime_args, destination);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.copies, 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, RejectedReprepareDoesNotTouchAnotherSlot) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    SlotPersistentArgs other;
    runtime.dev.worker_count = 19;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, other), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(other.runtime_args, nullptr);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(helper.args.runtime_args, slot.runtime_args);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(release_slot_persistent_args(other, allocator), 0);
}

TEST_F(KernelArgsPublication, MismatchedSlotLeavesNoPublishableRun) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    helper.release_run_view();
    Runtime *destination = slot.runtime_args;
    ++slot.runtime_bytes;
    EXPECT_NE(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_NE(helper.publish_runtime_args(), 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, ReleaseDiscardsPendingSourceWithoutFreeingDestination) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    helper.release_run_view();
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_NE(helper.publish_runtime_args(), 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(rts.frees, 0);
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_EQ(rts.frees, 1);
}

TEST_F(KernelArgsPublication, MovedHelperRetainsSnapshotAfterSourceRunViewIsReleased) {
    runtime.dev.worker_count = 11;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    KernelArgsHelper moved(std::move(helper));
    helper.release_run_view();
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_NE(helper.publish_runtime_args(), 0);
    runtime.dev.worker_count = 23;
    ASSERT_EQ(moved.publish_runtime_args(), 0);
    EXPECT_EQ(published_worker_count(), 11);
    EXPECT_EQ(rts.copies, 1);
    moved.release_run_view();
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, FreshPrepareCannotReuseAnEarlierPublicationVerdict) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    ASSERT_TRUE(helper.runtime_args_published());

    runtime.dev.worker_count = 19;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(rts.allocations, 1);
    rts.copy_rc = -93;
    EXPECT_EQ(helper.publish_runtime_args(), -93);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, PublishedVerdictMovesAndIsRevokedByAbandonment) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    KernelArgsHelper moved(std::move(helper));
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_TRUE(moved.runtime_args_published());
    helper.release_run_view();
    EXPECT_TRUE(moved.runtime_args_published());
    moved.abandon_after_device_failure();
    EXPECT_FALSE(moved.runtime_args_published());
    EXPECT_EQ(moved.args.runtime_args, nullptr);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, ReleaseAllowsFreshPrepareWithoutRetainingPublication) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    helper.release_run_view();
    EXPECT_FALSE(helper.runtime_args_published());
    runtime.dev.worker_count = 23;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(helper.runtime_args_published());
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 23);
    EXPECT_EQ(rts.allocations, 1);
}

// A publication carries the uploaded prefix into a block allocated to the full
// device extent. Where a descriptor ends in storage the device initializes, that
// range must still hold what was there before the copy: the seed the allocator
// laid down stands in for the device's own write, so it is already in place when
// the real publication runs.
TEST_F(KernelArgsPublication, PublicationLeavesTheDeviceInitializedTailUntouched) {
    // The longest publication there is: the first one onto a block, which adds
    // the handshake region. Anything past it is the tail no host copy reaches.
    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    // The descriptor this fixture is built against declares the gate array, so the
    // shortfall is a property of the type, asserted rather than skipped: a change
    // that widened the copy back to the extent must fail here, not opt out.
    ASSERT_EQ(extent_bytes, sizeof(DeviceRuntimeLaunchDesc));
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates))
        << "no publication may reach the device-initialized gate tail";
    ASSERT_LT(image_bytes, extent_bytes);

    runtime.dev.worker_count = 5;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    // Bounds before any tail read: an allocation that shrank to the prefix is
    // reported here rather than read past below.
    ASSERT_EQ(slot.runtime_bytes, extent_bytes) << "the allocation does not cover the device-read tail";
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    ASSERT_EQ(rts.copies, 1);

    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    EXPECT_EQ(published_worker_count(), 5) << "the prefix did not arrive";
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "the publication reached the device-initialized tail at byte " << i;
    }
}

// The slot's block outlives the run that published into it, so a second run on
// the same slot must leave that range alone too — it reuses the block rather
// than reallocating, and the device's contents there are not the host's to
// rewrite.
TEST_F(KernelArgsPublication, RepeatedPublicationPreservesTheDeviceInitializedTail) {
    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates));
    ASSERT_LT(image_bytes, extent_bytes);

    runtime.dev.worker_count = 5;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);
    Runtime *const destination = slot.runtime_args;

    runtime.dev.worker_count = 9;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(), 0);

    EXPECT_EQ(slot.runtime_args, destination) << "the slot reallocated instead of reusing its block";
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.copies, 2);
    ASSERT_EQ(slot.runtime_bytes, extent_bytes);

    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    EXPECT_EQ(published_worker_count(), 9) << "the second run's prefix did not arrive";
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "a repeated publication reached the tail at byte " << i;
    }
}
