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
struct RtsState {
    int alloc_rc = 0;
    int copy_rc = 0;
    int copies = 0;
    int allocations = 0;
    int frees = 0;
} rts;

class KernelArgsPublication : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
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
    std::memset(*ptr, 0xa5, bytes);
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
    EXPECT_EQ(bytes, sizeof(DeviceRuntimeLaunchDesc));
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
        ASSERT_EQ(bytes[i], 0xa5);

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
