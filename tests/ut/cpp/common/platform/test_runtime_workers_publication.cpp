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
// Which prefix each publication carries, driven through the real onboard
// helper and allocator with host-backed RTS storage. The question these cases
// answer is per device allocation, not per slot: a block that has never been
// published onto gets the handshake region, and only a block that has starts
// receiving the shorter prefix.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

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
    uint64_t last_copy_bytes = 0;
} rts;

class WorkersPublication : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
    void TearDown() override {
        EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
        EXPECT_EQ(allocator.get_allocation_count(), expected_leaked_allocations);
    }

    // Blocks the case dropped without freeing. Only the quarantine path does
    // that, and it does it deliberately: a poisoned card cannot retire
    // per-resource frees, so the allocation outlives the slot and the force
    // reset in finalize() is what reclaims it.
    size_t expected_leaked_allocations = 0;

    uint64_t initializing_bytes() const { return runtime_device_initialized_prefix_size(runtime); }
    uint64_t steady_bytes() const { return runtime_device_copy_size(runtime); }

    // One prepare+publish against the slot, returning the publish rc.
    int publish_once() {
        const int prepare_rc = helper.prepare_runtime_args(runtime, allocator, slot);
        if (prepare_rc != 0) return prepare_rc;
        return helper.publish_runtime_args(/*launch_route_permitted=*/false);
    }

    // The block's handshake region, as the device would see it.
    const unsigned char *device_workers() const {
        return reinterpret_cast<const unsigned char *>(slot.runtime_args) + offsetof(DeviceRuntimeLaunchDesc, workers);
    }
    bool device_workers_all(unsigned char value) const {
        const unsigned char *bytes = device_workers();
        for (size_t i = 0; i < sizeof(DeviceRuntimeLaunchDesc::workers); ++i) {
            if (bytes[i] != value) return false;
        }
        return true;
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;
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
    rts.last_copy_bytes = bytes;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0) return rts.copy_rc;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }

TEST_F(WorkersPublication, TheFirstPublicationOntoABlockCarriesTheHandshakeRegion) {
    ASSERT_EQ(publish_once(), 0);

    EXPECT_EQ(rts.copies, 1) << "one H2D, not two";
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(slot.workers_initialized);
    EXPECT_TRUE(device_workers_all(0u)) << "the block's handshake region still holds allocator bytes";
}

TEST_F(WorkersPublication, AReusedBlockPublishesOnlyTheShorterPrefix) {
    ASSERT_EQ(publish_once(), 0);
    helper.release_run_view();

    // Stand in for what the device wrote during the first run. A second
    // publication that carried this region would erase it.
    std::memset(const_cast<unsigned char *>(device_workers()), 0x5a, sizeof(DeviceRuntimeLaunchDesc::workers));

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.copies, 2) << "still one H2D per publication";
    EXPECT_EQ(rts.last_copy_bytes, steady_bytes());
    EXPECT_LT(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(device_workers_all(0x5a)) << "a steady-state run overwrote the device's own handshake words";
}

TEST_F(WorkersPublication, AFailedFirstPublicationLeavesTheBlockUninitialized) {
    rts.copy_rc = -91;
    EXPECT_EQ(publish_once(), -91);
    EXPECT_FALSE(slot.workers_initialized) << "a copy that failed cannot have defined the region";
    EXPECT_NE(slot.runtime_args, nullptr) << "the block is retained for the retry";
    EXPECT_EQ(rts.frees, 0);

    rts.copy_rc = 0;
    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes()) << "the retry must initialize, not assume";
    EXPECT_TRUE(slot.workers_initialized);
    EXPECT_TRUE(device_workers_all(0u));
}

TEST_F(WorkersPublication, ANewAllocationDoesNotInheritInitialization) {
    ASSERT_EQ(publish_once(), 0);
    ASSERT_TRUE(slot.workers_initialized);

    ASSERT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_FALSE(slot.workers_initialized) << "the verdict belonged to the freed block";

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.allocations, 2);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes()) << "a fresh block must be initialized again";
}

TEST_F(WorkersPublication, AnAbandonedSlotDoesNotInheritInitialization) {
    ASSERT_EQ(publish_once(), 0);
    ASSERT_TRUE(slot.workers_initialized);

    // The quarantine path drops ownership without freeing: the card is poisoned
    // and per-resource frees are unsafe.
    abandon_slot_persistent_args(slot);
    expected_leaked_allocations = 1;
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.frees, 0);

    helper.release_run_view();
    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
}

// The commit belongs to the publication, not to a caller that has to remember
// to make it: preparing alone must not mark the block initialized.
TEST_F(WorkersPublication, PreparingWithoutPublishingCommitsNothing) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.copies, 0);

    helper.release_run_view();
    EXPECT_FALSE(slot.workers_initialized);

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(slot.workers_initialized);
}
